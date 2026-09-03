#!/usr/bin/env python3
"""
SeeWeed3D - which annotated frames still have plants nobody labelled?

    python -m seeweed3d.annotation.missed_plants

THE QUESTION, AND WHY IT DECIDES SOMETHING
-------------------------------------------
A frame is used as training data whole: every pixel not inside a mask is
supervised as BACKGROUND. So one seedling the prelabeler missed and the
annotator did not notice does not merely fail to help - it teaches that a plant
of that size is soil, which on a weeder is a weed that never gets treated.

That makes "were any plants missed here" a property worth measuring before a
frame trains, not after. It is also the question that decides whether a drive
can be used as WHOLE FRAMES or only as a source of INSTANCE CUT-OUTS:

    few missed plants   use the frames. Real scenes, real weed-beside-weed
                        context, real lighting - all things a cut-out loses.
    many missed plants  the frame is not safe as a background. Cut the
                        annotated instances out and composite them into a
                        screened background instead (compose_mixed.py), which
                        carries the labelled pixels and leaves the mistake
                        behind.

BLOBS, NOT A FRACTION
---------------------
A frame whose masks all sit two pixels inside their leaves has a large unclaimed
FRACTION and nothing missing. A frame with one unlabelled seedling among forty
labelled plants has a tiny fraction and a real hole in it. So the claimed mask
is dilated to absorb annotation slop, small flecks are discarded, and what
survives is counted as plant-shaped things nobody labelled.

WHAT IT CANNOT TELL YOU
-----------------------
The vegetation prior calls moss, algae and green debris vegetation, and misses
dark or very small seedlings entirely. So a blob is a PLACE TO LOOK, not a
proven missed weed - and a frame with zero blobs is not proven clean either.
The overlays exist because the number alone cannot settle it; open them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from common.run_dirs import stamped  # noqa: E402
from common.vegetation import (CLAIM_DILATE_PX,  # noqa: E402
                               MIN_UNCLAIMED_BLOB_PX, unclaimed_blobs)
from training import pseudo_label as pl  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: WHAT TO AUDIT. Session folders (annotations/ + rgb/) or folders whose
#: children are sessions - the same shapes the dataset builds accept.
SESSIONS = [
    r"E:\Dataset_Vidalia\Weeds_20260108_1\sessions\vid3_20260108_110444",
    r"E:\Dataset_Vidalia\Weeds_20260108_3_good\sessions\vid2_20260108_122731",
    r"E:\Dataset_Vidalia\Mix_raj_Batch 01",
]

#: Restrict to the frames a build actually uses, in make_dataset's
#: `<session>:<range>` form. Auditing frames nobody trains on wastes the
#: report - and the 251 unreviewed frames of a 326-frame export would drown the
#: 75 corrected ones this is asking about.
INCLUDE_FRAMES = "vid3_20260108_110444:1-75"

#: A frame with more than this many plant-shaped unclaimed blobs is reported.
#: 0 reports every frame that has any.
MIN_BLOBS_TO_REPORT = 1

#: A frame at or above this many blobs is called UNSAFE AS A BACKGROUND: enough
#: missing that training on it whole teaches more background than plant.
UNSAFE_BLOBS = 3

#: Overlays for the worst frames. The number cannot settle whether a blob is a
#: missed weed or a patch of moss; only the picture can.
WORST_FRAMES = 15

#: Tuning for what counts as unclaimed. See common/vegetation.py.
DILATE_PX = CLAIM_DILATE_PX
MIN_BLOB_PX = MIN_UNCLAIMED_BLOB_PX

OUT_ROOT = r"E:\Dataset_Vidalia\audits"

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

C_CLAIMED = (0, 165, 255)     # orange: what somebody annotated
C_MISSED = (0, 0, 255)        # red: vegetation nobody claimed


def audit_frame(bgr, claimed, dilate_px=DILATE_PX, min_blob_px=MIN_BLOB_PX,
                veg=None):
    """One frame: how many plant-shaped things nobody labelled."""
    veg = pl.veg_of(bgr) if veg is None else veg
    n, px, mask = unclaimed_blobs(veg, claimed, dilate_px, min_blob_px)
    veg_px = int(np.asarray(veg, bool).sum())
    return {"n_missed": n, "missed_px": px, "veg_px": veg_px,
            "missed_frac": (px / veg_px) if veg_px else 0.0}, mask


def verdict(per_frame, unsafe=UNSAFE_BLOBS):
    """What the counts say about how this drive may be used.

    The two usable outcomes are 'train on the frames' and 'use it only as a
    cut-out source', and they are decided by the same number - so the number
    reports the decision rather than leaving it to be inferred."""
    n = len(per_frame)
    if not n:
        return "No frames were audited."
    bad = [k for k, v in per_frame.items() if v["n_missed"] >= unsafe]
    any_missed = [k for k, v in per_frame.items() if v["n_missed"]]
    share = len(bad) / n
    if not any_missed:
        return ("CLEAN. No frame has a plant-sized patch of vegetation outside "
                "an annotation, so these frames are safe to train on WHOLE - "
                "which keeps the real scene context a cut-out would lose. "
                "Note the prior misses dark and very small seedlings, so open "
                "a few overlays before treating this as proof.")
    if share < 0.10:
        return (f"MOSTLY CLEAN. {len(bad)} of {n} frame(s) have {unsafe}+ "
                f"unlabelled plant-shaped patches. Train on the frames and fix "
                f"or exclude those few - the listed ones are where to look.")
    return (f"NOT SAFE AS WHOLE FRAMES. {len(bad)} of {n} frame(s) "
            f"({share:.0%}) carry {unsafe}+ unlabelled plant-shaped patches, so "
            f"training on them teaches those plants are soil. Either finish the "
            f"annotation, or use this drive as a CUT-OUT SOURCE only: "
            f"compose_mixed.py carries the labelled instances into a screened "
            f"background and leaves the missed ones behind.")


def summarise(per_frame):
    n = len(per_frame)
    tot = sum(v["n_missed"] for v in per_frame.values())
    with_any = sum(1 for v in per_frame.values() if v["n_missed"])
    return {"frames": n, "missed_blobs": tot, "frames_with_missed": with_any,
            "mean_blobs_per_frame": (tot / n) if n else 0.0}


def worst(per_frame, n=WORST_FRAMES):
    return sorted((k for k, v in per_frame.items() if v["n_missed"]),
                  key=lambda k: (-per_frame[k]["n_missed"],
                                 -per_frame[k]["missed_px"], k))[:n]


def format_report(by_session, out_dir=None, unsafe=UNSAFE_BLOBS):
    L = ["", "  Plants nobody labelled", "  " + "-" * 40]
    for sess, per_frame in sorted(by_session.items()):
        s = summarise(per_frame)
        L += ["", f"  {sess}",
              f"    {s['frames']} frame(s), {s['missed_blobs']} unlabelled "
              f"plant-shaped patch(es) in {s['frames_with_missed']} frame(s)",
              f"    mean {s['mean_blobs_per_frame']:.2f} per frame"]
        listed = worst(per_frame, 8)
        if listed:
            L.append("    worst:")
            for k in listed:
                v = per_frame[k]
                L.append(f"      {k:<44}{v['n_missed']:>4} patch(es)"
                         f"{v['missed_frac']:>8.0%} of its vegetation")
        L += [f"    VERDICT: {verdict(per_frame, unsafe)}"]
    L += ["",
          "  [i] a patch is a PLACE TO LOOK, not a proven missed weed. The "
          "vegetation prior",
          "      calls moss, algae and green debris vegetation, and misses "
          "dark or very small",
          "      seedlings entirely - so zero patches is not proof of a clean "
          "frame either.",
          "      Open the overlays."]
    if out_dir:
        L += ["", f"  -> {out_dir}"]
    return "\n".join(L + [""])


def draw(bgr, claimed, missed):
    """Annotated plants outlined orange, unclaimed vegetation filled red."""
    import cv2
    out = bgr.copy()
    fill = out.copy()
    fill[np.asarray(missed, bool)] = C_MISSED
    out = cv2.addWeighted(fill, 0.45, out, 0.55, 0.0)
    m = np.asarray(claimed, bool).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, C_CLAIMED, 2)
    return out


def main():
    import cv2
    from evaluation.crop_risk import load_polygons, rasterise
    from training import prepare_dataset as pdz

    out_dir = Path(stamped(OUT_ROOT, "missed_plants"))
    (out_dir / "worst").mkdir(parents=True, exist_ok=True)

    sessions = []
    for root in SESSIONS:
        p = Path(root)
        if not p.is_dir():
            print(f"  [!] not found, skipped: {p}")
            continue
        sessions += ([p] if (p / "rgb").is_dir() or (p / "annotations").is_dir()
                     else [d for d in sorted(p.iterdir()) if d.is_dir()])
    if not sessions:
        raise SystemExit("ERROR: no session folders under SESSIONS.")

    by_session, records = {}, {}
    for sess in sessions:
        gt = load_polygons(sess)
        if not gt:
            print(f"  [!] no annotations in {sess.name}, skipped")
            continue
        if INCLUDE_FRAMES:
            keep = _selected_stems(sess, INCLUDE_FRAMES, pdz)
            if keep is not None:
                gt = {k: v for k, v in gt.items() if k in keep}
        per_frame = {}
        print(f"\n  {sess.name}: auditing {len(gt)} frame(s)")
        for stem, insts in sorted(gt.items()):
            img = _find(stem, [sess / "rgb", sess])
            if img is None:
                continue
            bgr = cv2.imread(str(img))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            claimed = np.zeros((h, w), bool)
            for _, polys, _ in insts:
                claimed |= rasterise(polys, h, w)
            rec, mask = audit_frame(bgr, claimed)
            if rec["n_missed"] >= MIN_BLOBS_TO_REPORT:
                per_frame[stem] = rec
                records[f"{sess.name}/{stem}"] = (str(img), claimed.shape)
        by_session[sess.name] = per_frame

    if not by_session:
        raise SystemExit("ERROR: nothing was audited.")

    written = []
    for sess, per_frame in by_session.items():
        for stem in worst(per_frame, WORST_FRAMES):
            key = f"{sess}/{stem}"
            if key not in records:
                continue
            img_path, _ = records[key]
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            claimed = np.zeros((h, w), bool)
            for _, polys, _ in load_polygons(Path(img_path).parent.parent
                                             ).get(stem, []):
                claimed |= rasterise(polys, h, w)
            _, mask = audit_frame(bgr, claimed)
            name = key.replace("/", "__") + ".jpg"
            cv2.imwrite(str(out_dir / "worst" / name),
                        cv2.resize(draw(bgr, claimed, mask), None,
                                   fx=0.5, fy=0.5))
            written.append(name)

    report = format_report(by_session, out_dir)
    print(report)
    (out_dir / "missed_plants.json").write_text(json.dumps({
        "sessions": [str(s) for s in sessions],
        "include_frames": INCLUDE_FRAMES,
        "dilate_px": DILATE_PX, "min_blob_px": MIN_BLOB_PX,
        "unsafe_blobs": UNSAFE_BLOBS,
        "per_session": {k: {"summary": summarise(v), "frames": v}
                        for k, v in by_session.items()},
        "overlays": written,
    }, indent=2), encoding="utf-8")
    (out_dir / "missed_plants.txt").write_text(report, encoding="utf-8")
    return 0


def _selected_stems(sess, spec, pdz):
    """The frame stems a build's INCLUDE_FRAMES keeps for this session."""
    try:
        files = pdz.find_annotation_files(sess)
    except SystemExit:
        return None
    from training import datumaro_multitask as dmm
    frames = []
    for f in files:
        got, _ = dmm.load_datumaro(f, fallback_session=dmm.batch_session_id(f))
        frames += got
    if not frames:
        return None
    try:
        kept, _ = pdz.select_frames(frames, spec, None)
    except SystemExit:
        return None
    return {Path(r.image_path or r.item_id).stem for r in kept}


def _find(stem, roots):
    for root in roots:
        d = Path(root)
        if not d.is_dir():
            continue
        for suf in (".png", ".jpg", ".jpeg"):
            p = d / f"{stem}{suf}"
            if p.is_file():
                return p
    return None


if __name__ == "__main__":
    raise SystemExit(main())
