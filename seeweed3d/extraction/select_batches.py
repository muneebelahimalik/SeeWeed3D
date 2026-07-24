#!/usr/bin/env python3
"""
SeeWeed3D - Stage 2: Frame selection and CVAT batch export
===========================================================
Turns the extracted pool into annotation batches: QC gating, near-duplicate
removal, diversity selection, deliberate hard-case quotas, and a CVAT label
schema matching the project ontology.

SPLIT DISCIPLINE (enforced here, not left to good intentions)
-------------------------------------------------------------
Adjacent video frames are near-identical. Splitting them randomly leaks the
test set into training and inflates every metric. Sessions listed in
HOLDOUT_SESSIONS are removed from all training batches and can only be drawn
into a batch explicitly marked pool="holdout".

Run extraction/extract_sessions.py first.
"""

import csv
import json
import random
import shutil
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# #############################################################################
# ##   DATASET_ROOT  -  MUST match OUTPUT_ROOT from extract_sessions.py       ##
# ##   This is the one path you must set. Everything else has defaults.       ##
# #############################################################################

DATASET_ROOT = r"M:\Research\Data\SeeWeed3D\dataset"

# =============================================================================
# CONFIG - advanced tuning below; defaults are sensible
# =============================================================================

CONFIG = {
    "DATASET_ROOT": DATASET_ROOT,

    # -- Held-out sessions: whole sessions reserved for testing. ---------------
    # Put entire SESSIONS here, never individual frames. Fill this in after you
    # look at registry.csv; ideally a different date/field from the training data.
    "HOLDOUT_SESSIONS": [
        # "vid2_20250305_090000",
    ],

    # -- Quality gates. Frames failing these never reach an annotator. ---------
    # Tune after reading reports/pool_stats.csv from your first run - these are
    # starting points, not universal truths.
    "GATES": {
        "min_sharpness":      60.0,   # Laplacian variance; below this = unusable blur
        "min_veg_frac":       0.02,   # skip bare-soil frames with nothing to label
        "max_clip_frac":      0.15,   # blown-out glare
        "max_dark_frac":      0.35,
        "min_depth_valid_frac_veg": 0.30,  # need depth ON the plants for 3D work
    },

    # -- Near-duplicate removal ------------------------------------------------
    # Minimum 64-bit pHash Hamming distance between any two selected frames.
    # Higher = more visually distinct frames, fewer of them.
    "MIN_PHASH_DISTANCE": 8,

    # -- Batches to build ------------------------------------------------------
    # stress_fraction: share deliberately drawn from HARD frames (glare, wet,
    # blur, depth holes). The gold set needs these; a clean-frames-only gold set
    # will overstate zero-shot prelabel quality.
    "BATCHES": [
        {"name": "b01_gold",  "n": 80,  "stress_fraction": 0.35, "pool": "train"},
        {"name": "b02_seed",  "n": 500, "stress_fraction": 0.20, "pool": "train"},
        # {"name": "b03_test", "n": 200, "stress_fraction": 0.25, "pool": "holdout"},
    ],

    # Per-session cap as a fraction of batch size - stops one long session
    # dominating a batch.
    "MAX_SESSION_SHARE": 0.30,

    "COPY_DEPTH_INTO_BATCH": False,  # CVAT cannot use 16-bit PNGs; keep them out
    "RANDOM_SEED": 1337,
}

# =============================================================================
# CVAT label schema - project ontology (paste into CVAT "Raw" label editor)
# =============================================================================

CVAT_LABELS = [
    {"name": "onion plant", "type": "mask", "color": "#33ddff", "attributes": [
        {"name": "difficulty", "input_type": "select", "mutable": True,
         "values": ["normal", "overlapping", "blurred", "shadowed", "wet", "truncated"],
         "default_value": "normal"}]},
    {"name": "weed instance", "type": "mask", "color": "#ff6037", "attributes": [
        {"name": "morphology", "input_type": "select", "mutable": False,
         "values": ["broadleaf", "grass", "sedge", "unknown"], "default_value": "unknown"},
        {"name": "growth_stage", "input_type": "select", "mutable": True,
         "values": ["cotyledon", "2-leaf", "3-5-leaf", "later", "unknown"],
         "default_value": "unknown"},
        {"name": "lep_visibility", "input_type": "select", "mutable": True,
         "values": ["visible", "partially_occluded_inferable", "not_visible"],
         "default_value": "visible"},
        {"name": "targetable", "input_type": "select", "mutable": True,
         "values": ["yes", "no", "uncertain"], "default_value": "yes"},
        {"name": "difficulty", "input_type": "select", "mutable": True,
         "values": ["normal", "overlapping", "blurred", "shadowed", "wet", "truncated"],
         "default_value": "normal"},
        {"name": "species", "input_type": "text", "mutable": True, "default_value": ""}]},
    {"name": "weed LEP", "type": "points", "color": "#fffc00", "attributes": [
        {"name": "lep_visibility", "input_type": "select", "mutable": True,
         "values": ["visible", "partially_occluded_inferable"], "default_value": "visible"},
        {"name": "confidence", "input_type": "select", "mutable": True,
         "values": ["certain", "uncertain"], "default_value": "certain"}]},
    {"name": "AMT region", "type": "polygon", "color": "#ff00cc", "attributes": []},
    {"name": "unknown vegetation", "type": "mask", "color": "#aaaaaa", "attributes": []},
    {"name": "ambiguous cluster", "type": "mask", "color": "#8a2be2", "attributes": []},
    {"name": "ignore region", "type": "mask", "color": "#000000", "attributes": [
        {"name": "reason", "input_type": "select", "mutable": False,
         "values": ["severe_blur", "ambiguity", "labeling_uncertainty", "out_of_range"],
         "default_value": "ambiguity"}]},
]

# =============================================================================


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def fnum(row, key, default=None):
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_pool(root):
    rows = []
    for pool_csv in sorted((root / "sessions").glob("*/meta/pool.csv")):
        sid = pool_csv.parts[-3]
        for r in csv.DictReader(open(pool_csv, encoding="utf-8")):
            r["session_id"] = r.get("session_id") or sid
            r["_rgb"] = root / "sessions" / sid / "rgb" / r["filename"]
            r["_depth"] = root / "sessions" / sid / "depth" / r["filename"]
            rows.append(r)
    return rows


def gate_failures(row, g):
    """Return the names of every gate this frame fails - empty list = passes."""
    f = []
    if fnum(row, "sharpness", 0) < g["min_sharpness"]:
        f.append("min_sharpness")
    if fnum(row, "veg_frac", 0) < g["min_veg_frac"]:
        f.append("min_veg_frac")
    if fnum(row, "clip_frac", 1) > g["max_clip_frac"]:
        f.append("max_clip_frac")
    if fnum(row, "dark_frac", 1) > g["max_dark_frac"]:
        f.append("max_dark_frac")
    dv = fnum(row, "depth_valid_frac_veg")
    if dv is not None and dv < g["min_depth_valid_frac_veg"]:
        f.append("min_depth_valid_frac_veg")
    return f


def passes(row, g):
    return not gate_failures(row, g)


def report_gates(rows, g):
    """Tell the user WHICH gate is doing the rejecting, and what value would help."""
    counts = defaultdict(int)
    for r in rows:
        for name in gate_failures(r, g):
            counts[name] += 1
    if not counts:
        return
    print("  Gate rejections (a frame can fail several):")
    obs = {
        "min_sharpness": ("sharpness", "p90", False),
        "min_veg_frac": ("veg_frac", "p90", False),
        "max_clip_frac": ("clip_frac", "p10", True),
        "max_dark_frac": ("dark_frac", "p10", True),
        "min_depth_valid_frac_veg": ("depth_valid_frac_veg", "p90", False),
    }
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        col, which, upper = obs[name]
        vals = sorted(v for v in (fnum(r, col) for r in rows) if v is not None)
        hint = ""
        if vals:
            v = vals[int(0.9 * (len(vals) - 1))] if which == "p90" \
                else vals[int(0.1 * (len(vals) - 1))]
            hint = f" | observed {'p10' if upper else 'p90'}={v:.4g}, current={g[name]:.4g}"
        print(f"    {name:28s} rejects {n:5d}/{len(rows)}{hint}")


def difficulty(row):
    """Higher = harder. Used to fill the deliberate hard-case quota."""
    s = 0.0
    sh = fnum(row, "sharpness", 500)
    s += max(0.0, 1.0 - sh / 300.0)                       # motion blur
    s += min(1.0, fnum(row, "clip_frac", 0) / 0.10) * 0.8  # glare / wet specular
    s += min(1.0, fnum(row, "dark_frac", 0) / 0.30) * 0.6  # deep shadow
    dv = fnum(row, "depth_valid_frac_veg")
    if dv is not None:
        s += (1.0 - dv) * 1.2                              # depth holes on foliage
    return s


def greedy_diverse(cands, n, min_dist, per_session_cap, taken_hashes, taken_ids):
    """Accept a frame only if it is at least min_dist bits from everything already
    chosen. Falls back to a relaxed threshold rather than returning short."""
    chosen = []
    counts = defaultdict(int)
    for threshold in (min_dist, max(2, min_dist // 2), 0):
        for row in cands:
            if len(chosen) >= n:
                break
            key = (row["session_id"], row["filename"])
            if key in taken_ids:
                continue
            if per_session_cap and counts[row["session_id"]] >= per_session_cap:
                continue
            h = row.get("phash", "")
            if threshold > 0 and h:
                if any(hamming(h, o) < threshold for o in taken_hashes):
                    continue
            chosen.append(row)
            counts[row["session_id"]] += 1
            taken_ids.add(key)
            if h:
                taken_hashes.append(h)
        if len(chosen) >= n:
            break
    return chosen


def build_batch(spec, rows, root, cfg, taken_hashes, taken_ids):
    holdout = set(cfg["HOLDOUT_SESSIONS"])
    if spec["pool"] == "holdout":
        pool = [r for r in rows if r["session_id"] in holdout]
    else:
        pool = [r for r in rows if r["session_id"] not in holdout]

    kept = [r for r in pool if passes(r, cfg["GATES"])]
    if not kept:
        print(f"  [{spec['name']}] no frames passed the gates")
        report_gates(pool, cfg["GATES"])
        return None

    n = spec["n"]
    n_stress = int(round(n * spec["stress_fraction"]))
    cap = max(1, int(n * cfg["MAX_SESSION_SHARE"]))

    ranked = sorted(kept, key=difficulty, reverse=True)
    stress_pool = ranked[:max(n_stress * 6, 40)]
    normal_pool = ranked[max(n_stress * 6, 40):] or ranked

    # zlib.crc32 gives a STABLE per-batch offset. Python's built-in hash() is
    # salted per process (PYTHONHASHSEED), so it would make RANDOM_SEED useless
    # for reproducibility across runs - which is the whole point of a seed.
    name_offset = zlib.crc32(spec["name"].encode("utf-8"))
    rng = random.Random(cfg["RANDOM_SEED"] + name_offset)
    rng.shuffle(stress_pool)
    rng.shuffle(normal_pool)

    sel = greedy_diverse(stress_pool, n_stress, cfg["MIN_PHASH_DISTANCE"],
                         cap, taken_hashes, taken_ids)
    sel += greedy_diverse(normal_pool, n - len(sel), cfg["MIN_PHASH_DISTANCE"],
                          cap, taken_hashes, taken_ids)

    if not sel:
        print(f"  [{spec['name']}] selected 0 frames - nothing left after "
              f"dedup/quotas (earlier batches may have taken everything). Skipped.")
        return None

    bdir = root / "batches" / spec["name"]
    (bdir / "images").mkdir(parents=True, exist_ok=True)
    if cfg["COPY_DEPTH_INTO_BATCH"]:
        (bdir / "depth").mkdir(parents=True, exist_ok=True)

    man = []
    for r in sel:
        src = Path(r["_rgb"])
        if not src.exists():
            continue
        shutil.copy2(src, bdir / "images" / src.name)
        if cfg["COPY_DEPTH_INTO_BATCH"] and Path(r["_depth"]).exists():
            shutil.copy2(r["_depth"], bdir / "depth" / src.name)
        man.append({k: r.get(k, "") for k in (
            "filename", "session_id", "trip", "site", "field", "scene_hint",
            "video_frame_idx", "capture_frame_idx", "timestamp_ns",
            "sharpness", "veg_frac", "clip_frac", "dark_frac",
            "depth_valid_frac", "depth_valid_frac_veg", "depth_median_mm", "phash")}
            | {"difficulty_score": round(difficulty(r), 3),
               "batch": spec["name"], "split_pool": spec["pool"]})

    if not man:
        print(f"  [{spec['name']}] {len(sel)} frames selected but none copied - "
              f"their RGB files are missing under sessions/. Re-run stage 1?")
        return None

    with open(bdir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        w.writeheader()
        w.writerows(man)

    (bdir / "cvat_labels.json").write_text(json.dumps(CVAT_LABELS, indent=2))

    per_sess = defaultdict(int)
    for m in man:
        per_sess[m["session_id"]] += 1
    (bdir / "batch.json").write_text(json.dumps({
        "batch": spec["name"], "pool": spec["pool"],
        "requested": n, "selected": len(man),
        "stress_requested": n_stress,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gates": cfg["GATES"], "min_phash_distance": cfg["MIN_PHASH_DISTANCE"],
        "frames_per_session": dict(per_sess),
        "holdout_sessions": cfg["HOLDOUT_SESSIONS"],
    }, indent=2))

    print(f"  [{spec['name']}] {len(man)}/{n} frames "
          f"from {len(per_sess)} session(s) -> {bdir}")
    if len(man) < n:
        print(f"      ! short by {n - len(man)}: raise MAX_POOL_PER_SESSION in "
              f"stage 1, lower MIN_PHASH_DISTANCE, or loosen GATES")
    return bdir


def main():
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"])
    rows = load_pool(root)
    if not rows:
        raise SystemExit(f"No pool.csv found under {root}/sessions. Run stage 1 first.")

    print(f"Pool: {len(rows)} frames across "
          f"{len({r['session_id'] for r in rows})} session(s)")
    kept = [r for r in rows if passes(r, cfg["GATES"])]
    print(f"Passing QC gates: {len(kept)} ({len(kept)/len(rows):.0%})")
    report_gates(rows, cfg["GATES"])
    print()

    (root / "reports").mkdir(exist_ok=True)
    stats = []
    for sid in sorted({r["session_id"] for r in rows}):
        sr = [r for r in rows if r["session_id"] == sid]
        sk = [r for r in sr if passes(r, cfg["GATES"])]
        stats.append({"session_id": sid, "pool": len(sr), "passed_gates": len(sk),
                      "pass_rate": round(len(sk) / len(sr), 3),
                      "is_holdout": sid in cfg["HOLDOUT_SESSIONS"]})
    with open(root / "reports" / "pool_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)

    taken_hashes, taken_ids = [], set()
    for spec in cfg["BATCHES"]:
        build_batch(spec, rows, root, cfg, taken_hashes, taken_ids)

    print(f"\nReports: {root / 'reports' / 'pool_stats.csv'}")
    print("Batches never share frames - taken IDs carry across the whole run.")


if __name__ == "__main__":
    main()
