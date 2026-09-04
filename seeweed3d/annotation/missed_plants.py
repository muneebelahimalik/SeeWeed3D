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
#:
#: THE COMPOSITE RUN IS AUDITED TOO, and it is the one entry here whose answer
#: is not already known. compose_mixed.py screens every background so that all
#: its vegetation is already labelled, and then pastes labelled instances - so
#: composites SHOULD come back clean. If they do not, either a paste and its
#: mask disagree, or MIN_BLOB_PX is calling shadow and soil texture a plant at
#: this resolution. Both are worth finding before 200 frames train.
#:
#: It also writes the overlays, which compose_mixed does not: the run's own
#: report says which contact band each instance achieved, but only a picture
#: shows a paste seam or a shadow pointing the wrong way.
SESSIONS = [
    r"E:\Dataset_Vidalia\Weeds_20260108_1\sessions\vid3_20260108_110444",
    r"E:\Dataset_Vidalia\Weeds_20260108_3_good\sessions\vid2_20260108_122731",
    r"E:\Dataset_Vidalia\Mix_raj_Batch 01",
    r"E:\Dataset_Vidalia\synthetic\synth_mixed_20260904_0156",
]

#: Restrict to the frames a build actually uses, in make_dataset's
#: `<session>:<range>` form. Auditing frames nobody trains on wastes the
#: report - and the 251 unreviewed frames of a 326-frame export would drown the
#: 75 corrected ones this is asking about.
INCLUDE_FRAMES = "vid3_20260108_110444:1-75"

#: THE TWO WAYS A DRIVE CAN BE USED. This audit recommends one of them; the
#: dataset builds are where the choice actually gets made.
WHOLE, CUTOUT = "whole", "cutout"
USE_PHRASE = {WHOLE: "trains as WHOLE FRAMES",
              CUTOUT: "used as a CUT-OUT SOURCE only"}

#: DRIVES WHOSE USE IS ALREADY SETTLED, and why. `<session>: (use, reason)`.
#:
#: This audit is a heuristic over a colour prior. It cannot tell a seedling
#: from moss, and it re-runs its opinion from scratch every time - so without
#: this it goes on recommending against decisions that were made deliberately,
#: with reasons, months ago. A tool that argues with a settled question every
#: run teaches the reader to skip the line it prints, and then the one run
#: where the numbers really did change gets skipped too.
#:
#: So a settled drive prints what was decided FIRST, and whether this audit
#: agrees. The COUNTS ARE STILL SHOWN in full: a drive can get worse after it
#: was decided, and noticing that is the whole point of running this again.
#:
#: Recording a decision here does not make it. Change the use in
#: training/datasets/mixed.py; this only stops the audit from re-arguing it.
DECIDED = {
    "Mix_raj_Batch_01": (
        WHOLE,
        "training/datasets/mixed.py, deliberately against this audit - seven "
        "frames corrected by hand in CVAT several times, and the only observed "
        "onion-and-weed contact in the project, which a cut-out would destroy"),
    "vid2_20260108_122731": (
        CUTOUT, "training/datasets/mixed.py, on this audit's recommendation"),
    "vid3_20260108_110444": (
        CUTOUT, "training/datasets/mixed.py, on this audit's recommendation"),
}

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

#: A mask less than this fraction vegetation is reported as claiming soil.
#:
#: DELIBERATELY FORGIVING, because a low number here is often correct. A polygon
#: around a thin curved onion leaf or a grass blade encloses a lot of soil no
#: matter how carefully it is drawn, and this project's own policy is to "err
#: large on onion masks" so the crop is over- rather than under-protected.
#:
#: What it is looking for is the other end: a mask so blobby it has swallowed
#: the soil AND whatever was growing in it. An onion mask that has absorbed an
#: adjacent weed trains that weed as crop, which protects it from ever being
#: shot - the mirror of a missed weed and just as expensive.
MIN_INSTANCE_VEG = 0.35

OUT_ROOT = r"E:\Dataset_Vidalia\audits"

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

C_CLAIMED = (0, 165, 255)     # orange: what somebody annotated
C_MISSED = (0, 0, 255)        # red: vegetation nobody claimed
C_SOIL_MASK = (0, 220, 255)   # yellow: a mask that is mostly soil


def audit_frame(bgr, claimed, dilate_px=DILATE_PX, min_blob_px=MIN_BLOB_PX,
                veg=None, instances=None, min_instance_veg=None):
    """One frame, BOTH ways a mask and its plant can disagree.

    A mask can fall short of its plant or reach past it, and the two are
    different mistakes with different costs:

        vegetation nobody claimed   a plant trained as background - on a weeder,
                                    a weed that never gets treated.
        a mask claiming soil        boundary slop at best; at worst an onion
                                    mask grown so blobby it has swallowed an
                                    adjacent weed, which trains the weed as
                                    crop and protects it from ever being shot.

    Only the first is visible from the union of masks, because a mask covering
    soil takes nothing away from the union. So the second is measured per
    INSTANCE - how much of each mask is actually vegetation - which is the same
    quantity pseudo_label uses to judge a prediction, applied here to a human's
    polygon."""
    veg = pl.veg_of(bgr) if veg is None else veg
    floor = (MIN_INSTANCE_VEG if min_instance_veg is None
             else float(min_instance_veg))
    n, px, mask = unclaimed_blobs(veg, claimed, dilate_px, min_blob_px)
    veg_px = int(np.asarray(veg, bool).sum())

    precisions, soil_idx = [], []
    for i, m in enumerate(instances or []):
        q = pl.instance_quality(m, veg)
        precisions.append(q["veg_precision"])
        if q["area_px"] and q["veg_precision"] < floor:
            soil_idx.append(i)
    return {"n_missed": n, "missed_px": px, "veg_px": veg_px,
            "missed_frac": (px / veg_px) if veg_px else 0.0,
            "n_instances": len(precisions), "n_soil_masks": len(soil_idx),
            "median_instance_veg": (float(np.median(precisions))
                                    if precisions else 1.0),
            "soil_idx": soil_idx}, mask


def _norm(name):
    """A session key that survives the ways one drive gets written down.

    The folder is `Mix_raj_Batch 01`, the session id is `Mix_raj_Batch_01`, and
    a decision recorded against one of them has to reach the other - a note
    that silently matches nothing is worse than no note."""
    return "".join(c for c in str(name).lower() if c.isalnum())


def decision_for(session, decided=DECIDED):
    for k, v in (decided or {}).items():
        if _norm(k) == _norm(session):
            return v
    return None


def stale_decisions(audited, decided=DECIDED):
    """Entries in DECIDED naming drives this run did not audit.

    A decision is about a drive; if the drive is gone from SESSIONS the note is
    describing something nobody is looking at any more, and it will go on being
    trusted until somebody checks."""
    seen = {_norm(s) for s in audited}
    return [k for k in (decided or {}) if _norm(k) not in seen]


def classify(per_frame, unsafe=UNSAFE_BLOBS):
    """What the counts say about how this drive may be used.

    The two usable outcomes are 'train on the frames' and 'use it only as a
    cut-out source', and they are decided by the same number - so the number
    reports the decision rather than leaving it to be inferred.

    Returns (recommended_use, text). The use is separate from the prose because
    a drive whose use is already settled needs the two COMPARED, not just
    printed one after the other."""
    n = len(per_frame)
    if not n:
        return None, "No frames were audited."
    bad = [k for k, v in per_frame.items() if v["n_missed"] >= unsafe]
    any_missed = [k for k, v in per_frame.items() if v["n_missed"]]
    share = len(bad) / n
    if not any_missed:
        return WHOLE, (
            "CLEAN. No frame has a plant-sized patch of vegetation outside "
            "an annotation, so these frames are safe to train on WHOLE - "
            "which keeps the real scene context a cut-out would lose. "
            "Note the prior misses dark and very small seedlings, so open "
            "a few overlays before treating this as proof.")
    if share < 0.10:
        return WHOLE, (
            f"MOSTLY CLEAN. {len(bad)} of {n} frame(s) have {unsafe}+ "
            f"unlabelled plant-shaped patches. Train on the frames and fix "
            f"or exclude those few - the listed ones are where to look.")
    return CUTOUT, (
        f"NOT SAFE AS WHOLE FRAMES. {len(bad)} of {n} frame(s) "
        f"({share:.0%}) carry {unsafe}+ unlabelled plant-shaped patches, so "
        f"training on them teaches those plants are soil. Either finish the "
        f"annotation, or use this drive as a CUT-OUT SOURCE only: "
        f"compose_mixed.py carries the labelled instances into a screened "
        f"background and leaves the missed ones behind.")


def verdict(per_frame, unsafe=UNSAFE_BLOBS, decided=None):
    """The recommendation, and - where the question is already settled - what
    was actually decided instead.

    A heuristic that goes on recommending against a decision somebody already
    made, with reasons, is not being careful; it is training the reader to skip
    the line. So a decided drive says so FIRST, and the audit's own opinion
    becomes supporting detail. The counts above it never change: a drive can
    get worse after it was decided, and that is exactly what this should still
    be able to show."""
    use, text = classify(per_frame, unsafe)
    if not decided or use is None:
        return text
    chose, why = decided
    if chose == use:
        return f"SETTLED: {USE_PHRASE[chose]} - {why}. This audit agrees: {text}"
    return (f"SETTLED: {USE_PHRASE[chose]} - {why}.\n"
            f"    This audit disagrees and is OVERRULED. It would say: {text}")


def summarise(per_frame):
    n = len(per_frame)
    tot = sum(v["n_missed"] for v in per_frame.values())
    with_any = sum(1 for v in per_frame.values() if v["n_missed"])
    soil = sum(v.get("n_soil_masks", 0) for v in per_frame.values())
    insts = sum(v.get("n_instances", 0) for v in per_frame.values())
    med = [v["median_instance_veg"] for v in per_frame.values()
           if v.get("n_instances")]
    return {"frames": n, "missed_blobs": tot, "frames_with_missed": with_any,
            "mean_blobs_per_frame": (tot / n) if n else 0.0,
            "instances": insts, "soil_masks": soil,
            "soil_mask_rate": (soil / insts) if insts else 0.0,
            "median_instance_veg": float(np.median(med)) if med else 1.0}


def worst(per_frame, n=WORST_FRAMES):
    return sorted((k for k, v in per_frame.items() if v["n_missed"]),
                  key=lambda k: (-per_frame[k]["n_missed"],
                                 -per_frame[k]["missed_px"], k))[:n]


def format_report(by_session, out_dir=None, unsafe=UNSAFE_BLOBS):
    L = ["", "  Plants nobody labelled", "  " + "-" * 40]
    for sess, per_frame in sorted(by_session.items()):
        s = summarise(per_frame)
        L += ["", f"  {sess}",
              f"    {s['frames']} frame(s), {s['instances']} annotated "
              f"instance(s)",
              f"    MASK TOO SMALL  {s['missed_blobs']} unlabelled plant-shaped "
              f"patch(es) in {s['frames_with_missed']} frame(s)"
              f"  (mean {s['mean_blobs_per_frame']:.2f}/frame)",
              f"    MASK TOO BIG    {s['soil_masks']} mask(s) under "
              f"{int(MIN_INSTANCE_VEG * 100)}% vegetation "
              f"({s['soil_mask_rate']:.1%}); median mask is "
              f"{s['median_instance_veg']:.0%} vegetation"]
        listed = worst(per_frame, 8)
        if listed:
            L.append("    worst:")
            for k in listed:
                v = per_frame[k]
                L.append(f"      {k:<44}{v['n_missed']:>4} patch(es)"
                         f"{v['missed_frac']:>8.0%} of its vegetation")
        L += [f"    VERDICT: {verdict(per_frame, unsafe, decision_for(sess))}"]
    for k in stale_decisions(by_session):
        L += ["", f"  [!] DECIDED names {k!r}, which this run did not audit. "
                  f"That decision is",
              "      about a drive nobody is looking at any more - check it is "
              "still in SESSIONS."]
    L += ["",
          "  [i] MASK TOO BIG is the forgiving half. A polygon around a thin "
          "onion leaf or a",
          "      grass blade encloses soil however carefully it is drawn, and "
          "erring large on",
          "      crop masks is this project's own policy. It is looking for the "
          "other end: a",
          "      mask so blobby it swallowed an adjacent weed, which trains "
          "that weed as CROP",
          "      and protects it from ever being shot.",
          "",
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


def draw(bgr, claimed, missed, instances=None, soil_idx=None):
    """Both failure modes in one picture, because they are fixed differently.

    orange   an annotated plant
    yellow   a mask that is mostly soil - too big, or around something thin
    red      vegetation nobody claimed - too small, or missed entirely
    """
    import cv2
    out = bgr.copy()
    fill = out.copy()
    fill[np.asarray(missed, bool)] = C_MISSED
    out = cv2.addWeighted(fill, 0.45, out, 0.55, 0.0)
    soil = set(soil_idx or ())
    if instances:
        for i, m in enumerate(instances):
            cnts, _ = cv2.findContours(np.asarray(m, bool).astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1,
                             C_SOIL_MASK if i in soil else C_CLAIMED, 2)
    else:
        cnts, _ = cv2.findContours(np.asarray(claimed, bool).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            per_inst = [rasterise(polys, h, w) for _, polys, _ in insts]
            claimed = np.zeros((h, w), bool)
            for m in per_inst:
                claimed |= m
            rec, mask = audit_frame(bgr, claimed, instances=per_inst)
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
            got = load_polygons(Path(img_path).parent.parent).get(stem, [])
            per_inst = [rasterise(polys, h, w) for _, polys, _ in got]
            claimed = np.zeros((h, w), bool)
            for m in per_inst:
                claimed |= m
            rec, mask = audit_frame(bgr, claimed, instances=per_inst)
            name = key.replace("/", "__") + ".jpg"
            cv2.imwrite(str(out_dir / "worst" / name),
                        cv2.resize(draw(bgr, claimed, mask, per_inst,
                                        rec.get("soil_idx")),
                                   None, fx=0.5, fy=0.5))
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
    # NARROW THE SPEC TO THIS EXPORT first. select_frames refuses a spec naming
    # a session it cannot see, and this walks one session at a time - so the
    # shared INCLUDE_FRAMES raised on every session it did not mention. It was
    # caught and turned into "no filtering", which happened to be right for
    # vid2 and would have silently audited all 326 frames of a drive the spec
    # restricted to 75.
    sub = pdz.spec_for_sessions(spec, {r.session_id for r in frames})
    if not sub:
        return None
    kept, _ = pdz.select_frames(frames, sub, None)
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
