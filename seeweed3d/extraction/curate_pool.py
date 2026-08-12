#!/usr/bin/env python3
"""
SeeWeed3D - pool curation: drop redundant / bad frames WITHOUT touching files
============================================================================
Two things go wrong with a real capture run:

  1. You moved slowly (typically at the start), so consecutive pooled frames
     show almost the same ground. Annotating 20 near-identical frames costs 20x
     the effort for ~1x the information, and over-weights whatever plants
     happened to be in the slow segment.
  2. Some frames are simply bad (blur, glare, an obstruction) and you want them
     gone after looking at the previews.

WHY THIS DOES NOT DELETE OR RENAME ANYTHING
-------------------------------------------
The filename `<session_id>_<video_frame_idx:06d>.png` IS the join key. The same
name appears in rgb/, depth/, right/ and conf/, and the number is the frame's
index in the source video. Deleting the rgb file but not the depth file
silently desynchronises the pair; renaming files to close the gaps destroys the
link back to the video and to meta/frames_index.csv, and breaks the project
invariant that every stream is aligned BY INDEX (video time is not real time -
capture can drop frames while the encoder assumes constant fps).

Gaps in the numbering are correct and expected: they mean "this frame exists in
the video but is not in the pool".

So curation is recorded in meta/pool.csv - the manifest every later stage
already reads - as two columns:

    dropped      0 = use it, 1 = skip it
    drop_reason  why, so a decision is auditable months later

Nothing on disk is removed, so any drop is reversible (RESTORE_ALL below) and
you can always re-examine a dropped frame.

HOW REDUNDANT FRAMES ARE FOUND
------------------------------
Overlap is a question about how far the CAMERA TRAVELLED, so that is what gets
measured, preferring real physical evidence over a proxy:

  1. POSE (best): v2 captures record tx_mm/ty_mm/tz_mm from ZED positional
     tracking. Euclidean distance between two frames is then literal camera
     travel in millimetres.
  2. IMAGE SHIFT (fallback): phase correlation between consecutive frames gives
     the dominant translation in pixels, expressed as a fraction of frame width
     so the threshold is resolution independent.

Travel accumulates from the LAST KEPT FRAME, not the previous frame. That
distinction is the whole point: crawling along at 2 mm/frame, every consecutive
pair looks "different enough" pairwise while the batch as a whole barely moves.
Accumulating from the last kept frame keeps one frame per MIN_TRAVEL_MM of
actual ground covered, which is the property you want.

    python seeweed3d/extraction/curate_pool.py      # DRY RUN by default
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.progress import Progress  # noqa: E402

# #############################################################################
# ##  DATASET_ROOT  -  the OUTPUT_ROOT you gave extract_sessions.py          ##
# ##  DRY_RUN       -  True = report only, write nothing. Run this first.    ##
# #############################################################################

DATASET_ROOT = r"E:\Dataset_Vidalia\Vidalia_visit_1_2025_all_sessions"
DRY_RUN = False

CONFIG = {
    "DATASET_ROOT": DATASET_ROOT,
    "DRY_RUN": DRY_RUN,

    # Empty = every session. List session ids to curate only those.
    "ONLY_SESSIONS": [],

    # -- 1. Drop near-duplicate frames (the "I was moving slowly" case) -------
    "DROP_REDUNDANT": True,

    # Minimum camera travel between kept frames, in millimetres, when pose is
    # available. CALIBRATE THIS to your mount height: it should be a meaningful
    # fraction of the ground footprint of one frame. Too small keeps duplicates;
    # too large throws away genuinely new ground.
    "MIN_TRAVEL_MM": 100.0,

    # Fallback when pose is missing/unreliable: minimum image shift between kept
    # frames as a fraction of frame width (resolution independent).
    "MIN_SHIFT_FRAC": 0.25,

    # Phase correlation runs on a downscaled grayscale copy - the dominant
    # translation survives downscaling and it is far cheaper.
    "SHIFT_WORK_WIDTH": 320,

    # Pose is only trusted when the tracker said so. Anything else falls back to
    # image shift rather than silently trusting a bad pose.
    "POSE_OK_STATES": ("OK", "SEARCHING_FLOOR_PLANE"),

    # -- 2. Drop specific frames by hand (the "this frame is bad" case) -------
    # Accepts, per session id:
    #   "sess_000123"  "sess_000123.png"  "sess_000123.jpg"  (preview name is
    #   fine - the extension and path are ignored), a bare index "123", or an
    #   inclusive index range "0-250".
    # Example:
    #   "MANUAL_DROPS": {"weed1_20260108_143022": ["0-250", "1187", "sess_001900.jpg"]}
    "MANUAL_DROPS": {},
    "MANUAL_DROP_REASON": "manual",

    # Print where the drops fall across each session. This is what tells you
    # whether a threshold is right - see drop_histogram().
    "SHOW_DROP_HISTOGRAM": True,

    # Candidate thresholds to report alongside the chosen one. The measurement
    # is already done by then, so each extra candidate is nearly free, and the
    # table is the honest way to pick a value: it shows the actual trade-off on
    # YOUR footage instead of asking you to guess. Set to [] to hide.
    "SWEEP_SHIFT_FRAC": [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00],
    "SWEEP_TRAVEL_MM": [25, 50, 100, 200, 400, 800],

    # -- Undo -----------------------------------------------------------------
    # True clears every drop in the selected sessions and writes nothing else.
    # Use it to start over; no image files were ever touched.
    "RESTORE_ALL": False,
}

DROPPED_COL = "dropped"
REASON_COL = "drop_reason"

# =============================================================================


def read_pool(session_dir):
    """(rows, fieldnames) from meta/pool.csv, or (None, None) if absent."""
    pool_csv = Path(session_dir) / "meta" / "pool.csv"
    if not pool_csv.exists():
        return None, None
    with open(pool_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fields = list(rdr.fieldnames or [])
    for col in (DROPPED_COL, REASON_COL):
        if col not in fields:
            fields.append(col)
    for r in rows:
        r.setdefault(DROPPED_COL, "0")
        r.setdefault(REASON_COL, "")
    return rows, fields


def write_pool(session_dir, rows, fields):
    """Rewrite meta/pool.csv in place, preserving every original column."""
    pool_csv = Path(session_dir) / "meta" / "pool.csv"
    with open(pool_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def is_dropped(row):
    """True if this pool row has been curated out. Absent column = kept, so
    a pool.csv written before curation existed still reads correctly."""
    return str(row.get(DROPPED_COL, "0")).strip() in ("1", "true", "True")


def frame_index(row):
    """Video frame index of a pool row, from the column or parsed from the
    filename, which encodes it. -1 when neither is usable."""
    v = row.get("video_frame_idx")
    if v not in (None, ""):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            pass
    stem = Path(row.get("filename", "")).stem
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def pose_xyz(row, ok_states):
    """(x, y, z) millimetres when the tracker reported a usable pose, else None.

    A pose recorded while tracking was lost is worse than no pose at all - it
    would report a huge bogus jump and keep a frame that should be dropped (or
    the reverse) - so anything outside ok_states is refused here and the caller
    falls back to measuring the images themselves."""
    state = str(row.get("pose_state", "")).strip()
    if state and ok_states and not any(s in state for s in ok_states):
        return None
    try:
        vals = [float(row[k]) for k in ("tx_mm", "ty_mm", "tz_mm")]
    except (KeyError, TypeError, ValueError):
        return None
    return None if any(np.isnan(v) for v in vals) else tuple(vals)


def pose_travel_mm(a, b):
    """Euclidean camera travel in mm between two poses."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _work_gray(path, work_width):
    """Downscaled float32 grayscale, or None if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if work_width and w > work_width:
        img = cv2.resize(img, (work_width, max(1, round(h * work_width / w))),
                         interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def image_shift_vec(prev_gray, cur_gray):
    """Translation between two frames as a (dx, dy) fraction of frame width.

    Phase correlation, not feature matching: it is a global estimate that does
    not need texture to be locally distinctive, which suits repetitive ground
    cover where feature matching is unreliable. A Hann window suppresses the
    edge discontinuity that would otherwise dominate the spectrum.

    Returns the VECTOR, not its magnitude, because the caller sums consecutive
    steps: summing magnitudes would treat camera jitter - and the strictly
    positive noise floor of phase correlation on near-identical frames - as
    forward progress, so a stationary camera would eventually "travel" far
    enough to keep a frame that is an exact duplicate."""
    if prev_gray is None or cur_gray is None or prev_gray.shape != cur_gray.shape:
        return None
    win = cv2.createHanningWindow((prev_gray.shape[1], prev_gray.shape[0]),
                                  cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(prev_gray, cur_gray, win)
    w = max(1, prev_gray.shape[1])
    return (float(dx) / w, float(dy) / w)


def frame_positions(live, rgb_dir, cfg, progress=None):
    """Cumulative camera position for each live frame, measured once.

    Returns (positions, forced_keep, signal, unit). Distance between two
    positions is the travel between those frames, so a selection at ANY
    threshold is then a cheap arithmetic pass - which is what makes
    sweep_thresholds() practically free after this has run.

    Mode is decided per session, not per pair: pose when every frame has a
    trusted one, image shift otherwise. Mixing millimetres and frame-widths
    inside one session would mean the threshold silently changes meaning
    partway through, which is not something you could reason about."""
    ok_states = cfg["POSE_OK_STATES"]
    poses = [pose_xyz(r, ok_states) for r in live]
    if poses and all(p is not None for p in poses):
        if progress:
            for _ in live:
                progress.update()
        return ([np.asarray(p, float) for p in poses],
                [False] * len(live), "pose", "mm")

    # Image mode: cumulative sum of consecutive phase-correlation steps.
    # Consecutive, not straight-to-anchor: once the view has moved
    # substantially the two frames no longer overlap, and phase correlation of
    # non-overlapping images is meaningless. Summing small reliable steps stays
    # valid. Summed as a VECTOR - summing magnitudes would count jitter, and
    # phase correlation's strictly positive noise floor on near-identical
    # frames, as forward progress.
    work_width = cfg["SHIFT_WORK_WIDTH"]
    positions, forced = [np.zeros(2)], [False]
    cum = np.zeros(2)
    prev = _work_gray(rgb_dir / live[0].get("filename", ""), work_width)
    if progress:
        progress.update()
    for row in live[1:]:
        cur = _work_gray(rgb_dir / row.get("filename", ""), work_width)
        step = image_shift_vec(prev, cur)
        if step is None:
            forced.append(True)             # unmeasurable -> never dropped
        else:
            cum = cum + np.asarray(step, float)
            forced.append(False)
        positions.append(cum.copy())
        prev = cur
        if progress:
            progress.update()
    return positions, forced, "image", "frac"


def select_keeps(positions, forced, threshold):
    """Indices to KEEP: greedy, measuring travel from the last KEPT frame.

    Measuring from the last kept frame rather than the previous frame is the
    whole point. Crawling at 5 mm/frame, every consecutive pair looks
    "different enough" pairwise, so a pairwise rule with a 20 mm threshold
    drops nothing at all."""
    if not positions:
        return []
    keep, anchor = [0], 0
    for i in range(1, len(positions)):
        if forced[i] or float(np.linalg.norm(positions[i] - positions[anchor])) >= threshold:
            keep.append(i)
            anchor = i
    return keep


def sweep_thresholds(positions, forced, thresholds, unit):
    """How many frames survive at each candidate threshold.

    The single number a run reports ("53% dropped") cannot tell you whether a
    threshold is right, and re-running the whole measurement per candidate is
    slow. The measurement is already done, so every threshold is one cheap
    pass over it - turning an unanswerable question into a table."""
    n = len(positions)
    out = []
    for t in thresholds:
        kept = len(select_keeps(positions, forced, t))
        row = {"threshold": t, "kept": kept, "dropped": n - kept,
               "kept_pct": 100.0 * kept / max(1, n)}
        if unit == "frac":
            # Fraction of frame width travelled between kept frames, so the
            # part of the view they share is what is left over. This is the
            # number worth choosing on: it is how much of each frame you would
            # be annotating twice.
            row["overlap_pct"] = max(0.0, 1.0 - t) * 100.0
        out.append(row)
    return out


def mark_redundant(rows, session_dir, cfg, progress=None):
    """Flag near-duplicate frames, keeping one per threshold of real travel.

    Already-dropped rows are left alone and are not eligible anchors, so this
    composes with manual drops in either order.

    Returns (n_dropped, signal, sweep) - sweep is the threshold table, or []."""
    rgb_dir = Path(session_dir) / "rgb"
    live = [r for r in rows if not is_dropped(r)]
    if len(live) < 2:
        return 0, "none", []

    positions, forced, signal, unit = frame_positions(live, rgb_dir, cfg, progress)
    threshold = cfg["MIN_TRAVEL_MM"] if unit == "mm" else cfg["MIN_SHIFT_FRAC"]
    keep = set(select_keeps(positions, forced, threshold))

    n_dropped = 0
    for i, row in enumerate(live):
        if i in keep:
            continue
        row[DROPPED_COL] = "1"
        row[REASON_COL] = "redundant"
        n_dropped += 1

    candidates = (cfg.get("SWEEP_TRAVEL_MM") if unit == "mm"
                  else cfg.get("SWEEP_SHIFT_FRAC")) or []
    sweep = sweep_thresholds(positions, forced, candidates, unit) if candidates else []
    return n_dropped, signal, sweep


def diagnose_pose(rows, ok_states):
    """Why pose was or wasn't usable, as a one-line explanation.

    A 'fell back to image shift' run is not self-explanatory - the cause could
    be a v1 capture with no pose at all, or a v2 capture whose tracker was lost.
    Those call for different responses, so the difference is reported."""
    have_cols = sum(1 for r in rows
                    if any(str(r.get(k, "")).strip() for k in ("tx_mm", "ty_mm", "tz_mm")))
    if not have_cols:
        return ("no pose recorded (v1 capture, or positional tracking was off) "
                "- MIN_SHIFT_FRAC is what matters here, MIN_TRAVEL_MM is unused")
    usable = sum(1 for r in rows if pose_xyz(r, ok_states) is not None)
    if usable == 0:
        states = sorted({str(r.get("pose_state", "")).strip() for r in rows
                         if str(r.get("pose_state", "")).strip()})
        return (f"pose columns present but no frame had a trusted pose_state "
                f"(saw: {', '.join(states) or 'blank'}) - add to POSE_OK_STATES "
                f"if one of those is in fact reliable")
    if usable < len(rows):
        return f"pose usable on {usable}/{len(rows)} frames - the rest fell back to image shift"
    return f"pose usable on all {usable} frames"


def drop_histogram(rows, buckets=10, width=44):
    """Where the drops fall across the session, as a text bar chart.

    This is the number that decides whether a threshold is right. Drops
    concentrated at the start match the 'I was slow setting off' case and are
    exactly what curation is for. Drops spread evenly across the whole session
    mean the threshold is simply too aggressive and is thinning good, genuinely
    new ground - which no amount of staring at the total percentage would tell
    you."""
    if not rows:
        return []
    n = len(rows)
    size = max(1, -(-n // buckets))          # ceil, so the last bucket is short
    out = []
    for b in range(0, n, size):
        chunk = rows[b:b + size]
        d = sum(1 for r in chunk if is_dropped(r))
        frac = d / len(chunk)
        bar = "#" * int(round(frac * width))
        lo, hi = b, b + len(chunk) - 1
        out.append(f"      frames {lo:>5}-{hi:<5} {frac * 100:5.1f}% "
                   f"|{bar:<{width}}| {d}/{len(chunk)}")
    return out


def parse_drop_tokens(tokens):
    """Drop spec -> (explicit frame indices, [(lo, hi) inclusive ranges]).

    Accepts a bare index, an inclusive 'lo-hi' range, or any filename whose stem
    ends in the index - so a preview name (.jpg) works as well as the source
    .png, which matters because previews are what you actually look at when
    deciding a frame is bad."""
    indices, ranges = set(), []
    for tok in tokens:
        t = str(tok).strip()
        if not t:
            continue
        if t.isdigit():
            indices.add(int(t))
            continue
        lo_hi = t.split("-")
        if len(lo_hi) == 2 and all(p.strip().isdigit() for p in lo_hi):
            lo, hi = (int(p) for p in lo_hi)
            ranges.append((min(lo, hi), max(lo, hi)))
            continue
        tail = Path(t).stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            indices.add(int(tail))
        else:
            print(f"      [WARN] cannot interpret drop token {tok!r} - ignored")
    return indices, ranges


def apply_manual_drops(rows, tokens, reason):
    """Flag rows named by the drop spec. Returns (n_newly_dropped, n_unmatched)."""
    indices, ranges = parse_drop_tokens(tokens)
    if not indices and not ranges:
        return 0, 0
    matched, n = set(), 0
    for row in rows:
        idx = frame_index(row)
        if idx < 0:
            continue
        hit = idx in indices or any(lo <= idx <= hi for lo, hi in ranges)
        if not hit:
            continue
        matched.add(idx)
        if not is_dropped(row):
            row[DROPPED_COL] = "1"
            row[REASON_COL] = reason
            n += 1
    return n, len(indices - matched)


def restore_all(rows):
    """Clear every drop. Nothing was deleted, so this fully undoes curation."""
    n = 0
    for row in rows:
        if is_dropped(row):
            n += 1
        row[DROPPED_COL], row[REASON_COL] = "0", ""
    return n


#: Keeping less than this fraction of a pool is almost always the threshold
#: being wrong for the footage rather than the footage being that redundant.
#: A slowly-driven session genuinely is highly redundant, so this is set low -
#: it flags "you deleted nearly everything", not "you were slightly generous".
OVER_CURATION_KEEP_FRAC = 0.15

#: Below this many surviving frames a session has stopped being an annotation
#: batch whatever the percentage says. Ten frames is not a training set and is
#: not a test set.
OVER_CURATION_MIN_FRAMES = 12


def over_curation_warning(before, after, cfg, sweep=None):
    """Warn when curation kept so little that the threshold is likely wrong.

    Curation is the one step here that decides how much data exists downstream,
    and it reports success either way: '754 -> 30 usable frames' is a normal
    line, not an error, and the run exits 0. The default MIN_SHIFT_FRAC ships
    tuned for whatever campaign it was last used on, so a slower drive silently
    loses 96% of a pool and the number only looks wrong once somebody adds it
    up across sessions.

    The warning names a looser threshold from the sweep that was already
    computed, so the fix is a value to type rather than another dry run.
    """
    if before <= 0 or not cfg.get("DROP_REDUNDANT", True):
        return []
    keep = after / float(before)
    if keep >= OVER_CURATION_KEEP_FRAC and after >= OVER_CURATION_MIN_FRAMES:
        return []

    out = [f"        [!] kept only {after} of {before} frames "
           f"({keep:.0%}) - this is probably the threshold, not the footage."]
    # The sweep is (value, kept, dropped, overlap-ish) rows already gathered
    # for the table below; find the loosest value that would keep a usable
    # batch, so the suggestion is grounded in this session's own numbers.
    suggestion = None
    for row in (sweep or []):
        try:
            val, kept = float(row["threshold"]), int(row["kept"])
        except (TypeError, ValueError, KeyError):
            continue
        if kept >= max(OVER_CURATION_MIN_FRAMES, 0.25 * before):
            if suggestion is None or val > suggestion[0]:
                suggestion = (val, kept)
    if suggestion:
        out.append(f"            MIN_SHIFT_FRAC {suggestion[0]:g} would keep "
                   f"{suggestion[1]} - see the sweep below.")
    else:
        out.append("            even the loosest value in the sweep keeps "
                   "little, so this session really is that redundant.")
    out.append("            RESTORE_ALL = True undoes this; no image was "
               "touched.")
    return out


def curate_session(sid, session_dir, cfg):
    rows, fields = read_pool(session_dir)
    if rows is None:
        print(f"  [{sid}] no meta/pool.csv - run extract_sessions.py first")
        return None
    if not rows:
        print(f"  [{sid}] pool.csv is empty")
        return None

    before = sum(1 for r in rows if not is_dropped(r))

    if cfg["RESTORE_ALL"]:
        n = restore_all(rows)
        print(f"  [{sid}] restored {n} previously dropped frames -> {len(rows)} usable")
        if not cfg["DRY_RUN"]:
            write_pool(session_dir, rows, fields)
        return {"session": sid, "restored": n}

    n_manual, n_unmatched = apply_manual_drops(
        rows, cfg["MANUAL_DROPS"].get(sid, []), cfg["MANUAL_DROP_REASON"])
    if n_unmatched:
        print(f"  [{sid}] [WARN] {n_unmatched} manually listed frame(s) are not "
              f"in this session's pool - check the ids")

    n_redundant, signal, sweep = 0, "off", []
    if cfg["DROP_REDUNDANT"]:
        prog = Progress(sum(1 for r in rows if not is_dropped(r)),
                        f"  [{sid}]", unit="frames")
        n_redundant, signal, sweep = mark_redundant(rows, session_dir, cfg, prog)
        prog.close(note=f"{n_redundant} redundant")

    after = sum(1 for r in rows if not is_dropped(r))
    pct = 100.0 * (before - after) / max(1, before)
    print(f"  [{sid}] {before} -> {after} usable frames "
          f"({before - after} dropped, {pct:.0f}%)"
          f" | redundant={n_redundant} (by {signal}) manual={n_manual}")
    for line in over_curation_warning(before, after, cfg, sweep):
        print(line)

    if cfg["DROP_REDUNDANT"]:
        print(f"        pose: {diagnose_pose(rows, cfg['POSE_OK_STATES'])}")
    if cfg.get("SHOW_DROP_HISTOGRAM", True) and (before - after):
        print("        where the drops fall:")
        for line in drop_histogram(rows):
            print(line)
        print("        (bunched = a slow patch, thinned as intended. Flat = you "
              "moved at a steady speed, so the threshold alone sets the rate.)")

    if sweep:
        cur = cfg["MIN_TRAVEL_MM"] if signal == "pose" else cfg["MIN_SHIFT_FRAC"]
        unit = "mm" if signal == "pose" else "frac"
        print(f"        threshold sweep (current = {cur}):")
        head = "          value    kept   dropped"
        print(head + ("   overlap between kept frames" if unit == "frac" else ""))
        for s in sweep:
            mark = " <-- current" if abs(s["threshold"] - cur) < 1e-9 else ""
            ov = (f"   {s['overlap_pct']:5.0f}% of each frame re-annotated"
                  if "overlap_pct" in s else "")
            print(f"          {s['threshold']:<7} {s['kept']:>5} {s['dropped']:>9}"
                  f"{ov}{mark}")
        if unit == "frac":
            print("        Lower value = more frames, more overlap, more "
                  "duplicated annotation work.")

    if cfg["DRY_RUN"]:
        print("        DRY RUN - pool.csv not modified. Set DRY_RUN = False to apply.")
    else:
        write_pool(session_dir, rows, fields)
        print(f"        wrote {Path(session_dir) / 'meta' / 'pool.csv'}")

    return {"session": sid, "before": before, "after": after,
            "redundant": n_redundant, "manual": n_manual, "signal": signal,
            "sweep": sweep}


def main():
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"])
    from common.dataset_paths import require_sessions_root
    sessions_root = require_sessions_root(root)
    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    if cfg["ONLY_SESSIONS"]:
        sids = [s for s in sids if s in cfg["ONLY_SESSIONS"]]
    if not sids:
        sys.exit("No sessions selected.")

    mode = "RESTORE" if cfg["RESTORE_ALL"] else ("DRY RUN" if cfg["DRY_RUN"] else "APPLY")
    print(f"Pool curation on {len(sids)} session(s)  [{mode}]")
    print("  No image files are deleted or renamed - drops are recorded in "
          "meta/pool.csv only.\n")

    stats = [s for s in (curate_session(sid, sessions_root / sid, cfg)
                         for sid in sids) if s]
    if stats and not cfg["RESTORE_ALL"]:
        b = sum(s.get("before", 0) for s in stats)
        a = sum(s.get("after", 0) for s in stats)
        print(f"\nTotal: {b} -> {a} usable frames ({b - a} dropped)")
        if cfg["DRY_RUN"]:
            print("Nothing was written. Review the numbers, then set DRY_RUN = False.")


if __name__ == "__main__":
    main()
