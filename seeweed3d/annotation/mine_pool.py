#!/usr/bin/env python3
"""
SeeWeed3D - choose the NEXT frames to annotate (EDIT THE CONFIG BELOW)
=====================================================================
    python seeweed3d/annotation/mine_pool.py

Runs your trained model over the unlabelled pool, ranks every frame by how much
annotating it would teach, picks a spread-out batch, and writes it as a
CVAT-ready folder WITH THE MODEL'S OWN PREDICTIONS ALREADY IN IT.

WHY THIS IS THE HIGHEST-VALUE STEP LEFT
---------------------------------------
The dataset is the limit, not the architecture. Two runs in a row said so: at
conf 0.15 - the most permissive setting the crop-safety numbers allow - 18% of
small weeds are still never found, and `other_weed` sits at AP 0.25 with 86
ground-truth instances. Neither is fixed by a hyperparameter.

Annotating the next 60 frames AT RANDOM mostly re-teaches what the model
already knows. Ranking them first is the difference between 60 frames of new
information and 60 more pictures of a primrose.

MODEL-IN-THE-LOOP, NOT SAM 3
----------------------------
The SAM 3 prelabelers propose masks with GENERIC classes, so every frame needs
its labels rewritten by hand - which is why the earlier CVAT tasks arrived full
of confidently wrong classes. Your own checkpoint knows this ontology, so the
export here is CORRECTION rather than annotation: the classes are usually
right, and the work is fixing boundaries and adding what was missed.

That last part is the one thing to watch. A pre-labelled frame biases an
annotator toward accepting what is there and not noticing what is absent, and
MISSED weeds are exactly this project's failure mode. The batch report prints
the model's own recall at the export threshold as a reminder of roughly how
many instances per frame it is expected to have left out.

WHAT IT DOES NOT DO
-------------------
It does not pseudo-label for TRAINING. Predictions here go to a human, never
straight into a manifest. See docs/dataset_growth.md for why that line is
drawn where it is - briefly, a confidently wrong onion mask taught back to the
model is a crop-safety failure that improves every metric on the page.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CROP_CLASS, coco_categories  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

CONFIG = {
    # -- The model that does the ranking ---------------------------------------
    "CHECKPOINT": r"E:\Dataset_Vidalia\training1\run4\best.pt",
    "BACKEND": "maskrcnn",        # "maskrcnn" | "rfdetr"
    "DEVICE": "cuda",

    # The confidence the PRE-LABELS are written at. Lower than you would deploy:
    # a spurious mask costs one delete, a missing one costs the annotator
    # noticing an absence, which is far harder. See the note above.
    "CONF": 0.20,

    # -- What is already labelled ----------------------------------------------
    # OUT_DIR from make_dataset.py. Used for two things: to SKIP frames already
    # annotated, and to count class frequencies so scarce classes score higher.
    "DATASET_DIR": r"E:\Dataset_Vidalia\training1",

    # -- The pool to mine ------------------------------------------------------
    "SESSIONS_ROOT": r"E:\Dataset_Vidalia\sessions",

    # [] = every session found. Naming sessions here restricts the scan.
    "ONLY_SESSIONS": [],

    # HOLD THESE OUT. A session you intend to use as a test set must never be
    # annotated into training, or the test set stops being one. This is the
    # only place that discipline is enforceable before the annotation happens.
    "HOLDOUT_SESSIONS": [],

    # Scan every Nth frame. Consecutive ZED frames are near-identical, so a
    # stride of 1 wastes inference on duplicates the diversity pass then throws
    # away anyway.
    "STRIDE": 5,

    # 0 = no cap. Inference over the pool is the slow part; cap it while you
    # are trying settings out.
    "MAX_SCAN": 0,

    # -- The batch -------------------------------------------------------------
    # How many frames to send to CVAT. Size it to what you will actually
    # annotate this round; an over-large batch just goes stale.
    "BATCH_SIZE": 60,

    "OUT_DIR": r"E:\Dataset_Vidalia\next_batch",

    # Hardlink images instead of copying. Same volume only; falls back to copy.
    "LINK_IMAGES": True,

    # Drop disconnected mask fragments smaller than this fraction of the
    # instance's largest component before writing polygons. At CONF 0.20 a
    # predicted mask is often one plant plus a scattering of specks where the
    # sigmoid crossed threshold on soil texture, and every speck is a polygon
    # someone has to delete by hand. Relative, not absolute, so a genuine
    # cotyledon survives whole. 0 disables.
    "MIN_FRAGMENT_FRAC": 0.15,
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def labelled_item_ids(dataset_dir):
    """item_ids already annotated, so the pool excludes them.

    Re-annotating a frame is not merely wasted time: the same frame in two
    CVAT tasks produces two versions of the truth, and prepare_dataset has no
    way to know which one to believe."""
    man = Path(dataset_dir) / "seg_manifest.json"
    if not man.exists():
        return set()
    doc = json.loads(man.read_text(encoding="utf-8"))
    return {f.get("item_id") for f in doc.get("frames", []) if f.get("item_id")}


def pool_frames(sessions_root, only=(), holdout=(), stride=1, limit=0,
                exclude_ids=()):
    """Unlabelled candidate frames, newest discipline first.

    depth/ is excluded here for the same reason as everywhere else: a depth
    PNG carries its RGB frame's exact filename, and a naive walk hands 16-bit
    depth to the model as a picture."""
    root = Path(sessions_root)
    if not root.is_dir():
        raise SystemExit(f"ERROR: SESSIONS_ROOT does not exist: {root}")
    only, holdout = set(only or ()), set(holdout or ())
    exclude_ids = set(exclude_ids or ())

    out = []
    for sess in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and sess.name not in only:
            continue
        if sess.name in holdout:
            continue
        rgb = sess / "rgb" if (sess / "rgb").is_dir() else sess
        files = [f for f in sorted(rgb.rglob("*"))
                 if f.suffix.lower() in IMAGE_SUFFIXES
                 and not any(q.name.lower().startswith("depth")
                             for q in f.relative_to(rgb).parents)]
        files = files[::max(1, int(stride))]
        out.extend((sess.name, f) for f in files if f.stem not in exclude_ids)
    if limit:
        out = out[:int(limit)]
    return out


def _onion_distance_map(det):
    """Distance in pixels from every pixel to the nearest predicted onion, or
    None when this model cannot predict the crop at all.

    None is not zero. A weeds-only checkpoint has no opinion about where the
    crop is, and scoring its frames as low-risk would be exactly backwards."""
    import cv2
    onion = det.onion_safety_mask()
    if onion is None or not onion.any():
        return None
    return cv2.distanceTransform((~onion).astype(np.uint8), cv2.DIST_L2, 3)


def frame_result(det, session_id, frame_id):
    """Detections -> the FrameResult shape active_learning.score_frame reads.

    Only the fields the scorer uses are filled. LEP terms stay empty because
    Stage B is not trained yet; that term then contributes nothing rather than
    contributing noise."""
    dist = _onion_distance_map(det)
    targets = []
    for i in range(len(det)):
        name = det.class_name(i)
        m = np.asarray(det.masks[i]).astype(bool)
        t = {"instance_index": i, "class_name": name,
             "class_confidence": float(det.scores[i]),
             "mask_area_px": int(m.sum()), "abstained": False,
             "safety_status": "candidate", "rejection_reasons": [],
             "safety_notes": {}}
        if dist is not None and name != CROP_CLASS and m.any():
            t["safety_notes"]["onion_distance_px"] = float(dist[m].min())
        targets.append(t)
    return {"session_id": session_id, "frame_id": frame_id,
            "width": det.width, "height": det.height,
            "n_instances": len(det), "targets": targets}


def _clean(mask, min_frac):
    if not min_frac:
        return mask
    from perception.segmenter import drop_fragments
    return drop_fragments(mask, min_frac)


def mask_to_polygons(mask, min_area_px=24, epsilon_frac=0.004):
    """Boolean mask -> COCO polygons, simplified enough to be editable.

    A per-pixel contour is technically exact and useless in CVAT: dragging a
    3000-vertex polygon is slower than redrawing it. approxPolyDP at a fraction
    of the perimeter keeps the shape while leaving handles a human can move."""
    import cv2
    m = np.asarray(mask).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append([float(v) for xy in a for v in xy])
    return out


def _place(src, dst, link=True):
    if dst.exists():
        return
    if link:
        try:
            import os
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def export_batch(selected, predictions, out_dir, classes, link=True):
    """CVAT-ready folder: images plus the model's predictions as prelabels."""
    out = Path(out_dir)
    ready = out / "cvat_ready"
    ready.mkdir(parents=True, exist_ok=True)

    cats = coco_categories(classes)
    cat_id = {c["name"]: c["id"] for c in cats}
    images, anns, ann_id = [], [], 1
    n_inst = 0
    for img_id, s in enumerate(selected, start=1):
        rec = predictions[s["frame_id"]]
        src = Path(rec["path"])
        _place(src, ready / src.name, link)
        images.append({"id": img_id, "file_name": src.name,
                       "height": rec["height"], "width": rec["width"]})
        for inst in rec["instances"]:
            if inst["class_name"] not in cat_id:
                continue
            for poly in inst["polygons"]:
                xs, ys = poly[0::2], poly[1::2]
                anns.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cat_id[inst["class_name"]],
                    "segmentation": [poly], "iscrowd": 0,
                    "bbox": [min(xs), min(ys), max(xs) - min(xs),
                             max(ys) - min(ys)],
                    "area": float(inst["area_px"]),
                })
                ann_id += 1
            n_inst += 1

    (out / "instances_default.json").write_text(json.dumps({
        "info": {"description": "SeeWeed3D model-in-the-loop prelabels",
                 "date_created": datetime.now(timezone.utc).isoformat()},
        "licenses": [], "images": images, "annotations": anns,
        "categories": cats}, indent=2), encoding="utf-8")
    return {"n_images": len(images), "n_instances": n_inst,
            "cvat_ready": str(ready)}


def mine(cfg=None):
    import cv2
    from common.torch_utils import require_device
    from perception.segmenter import build_segmenter
    from training.active_learning import (appearance_descriptor,
                                          labelled_class_frequencies,
                                          select_next_batch)

    c = dict(CONFIG if cfg is None else cfg)
    device = require_device(c["DEVICE"])
    ckpt = Path(c["CHECKPOINT"])
    if not ckpt.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt}\n"
                         f"Train one first: seeweed3d/training/train_model.py")

    done = labelled_item_ids(c["DATASET_DIR"])
    frames = pool_frames(c["SESSIONS_ROOT"], c.get("ONLY_SESSIONS"),
                         c.get("HOLDOUT_SESSIONS"), c.get("STRIDE", 1),
                         c.get("MAX_SCAN", 0), done)
    if not frames:
        raise SystemExit(
            f"ERROR: no unlabelled frames under {c['SESSIONS_ROOT']}.\n"
            f"{len(done)} item_ids are already annotated and were skipped; "
            f"check ONLY_SESSIONS / HOLDOUT_SESSIONS.")
    print(f"  pool: {len(frames)} frames | {len(done)} already annotated "
          f"and skipped")

    seg = build_segmenter(c["BACKEND"], str(ckpt), conf=c["CONF"],
                          device=device)
    seg.load()

    results, descriptors, predictions = [], {}, {}
    for k, (sess, path) in enumerate(frames):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        det = seg(bgr)
        fid = path.stem
        results.append(frame_result(det, sess, fid))
        descriptors[fid] = appearance_descriptor(bgr)
        predictions[fid] = {
            "path": str(path), "session_id": sess,
            "height": int(bgr.shape[0]), "width": int(bgr.shape[1]),
            "instances": [
                {"class_name": det.class_name(i),
                 "score": float(det.scores[i]),
                 "area_px": int(np.asarray(det.masks[i]).astype(bool).sum()),
                 "polygons": mask_to_polygons(
                     _clean(det.masks[i], c.get("MIN_FRAGMENT_FRAC", 0.0)))}
                for i in range(len(det))],
        }
        if (k + 1) % 50 == 0:
            print(f"    scanned {k + 1}/{len(frames)}")

    freq = labelled_class_frequencies(
        seg_manifest=Path(c["DATASET_DIR"]) / "seg_manifest.json")
    report = select_next_batch(results, freq, k=c["BATCH_SIZE"],
                               descriptors=descriptors)

    classes = list(getattr(seg, "classes", None) or [])
    ex = export_batch(report["selected"], predictions, c["OUT_DIR"], classes,
                      c.get("LINK_IMAGES", True))
    report["export"] = ex
    report["checkpoint"] = str(ckpt)
    report["conf"] = c["CONF"]
    Path(c["OUT_DIR"], "batch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    _summarise(report, freq, predictions)
    return report


def _summarise(report, freq, predictions):
    print(f"\n  selected {report['n_selected']} of {report['n_pool']} frames")
    if report.get("note"):
        print(f"  [note] {report['note']}")
    print("\n  already labelled, per class:")
    for k in sorted(freq, key=lambda x: freq[x]):
        flag = "  <- NONE" if freq[k] == 0 else ("  <- scarce"
                                                 if freq[k] < 10 else "")
        print(f"    {k:<28}{freq[k]:>6}{flag}")

    why = Counter()
    for s in report["selected"]:
        for r in s.get("reasons", []):
            why[r.split(":")[0]] += 1
    if why:
        print("\n  why these frames were chosen:")
        for r, n in why.most_common():
            print(f"    {n:>4}x  {r}")

    ex = report["export"]
    print(f"\n  -> {ex['cvat_ready']}  ({ex['n_images']} images, "
          f"{ex['n_instances']} prelabelled instances)")
    print(f"  -> {Path(report['export']['cvat_ready']).parent / 'instances_default.json'}")
    print("\n  In CVAT: create the task from cvat_ready/, paste the label")
    print("  schema (annotation/regen_cvat_labels.py), then import")
    print("  instances_default.json as COCO 1.0.")
    print("\n  These are PREDICTIONS, not truth. The model's own recall is well")
    print("  under 1.0, so assume every frame is missing instances - the ones")
    print("  it missed are the ones worth the most.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--dataset")
    p.add_argument("--sessions")
    p.add_argument("--out")
    p.add_argument("--backend", choices=["maskrcnn", "rfdetr"])
    p.add_argument("--device")
    p.add_argument("--conf", type=float)
    p.add_argument("--stride", type=int)
    p.add_argument("--max-scan", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--only-sessions", nargs="*")
    p.add_argument("--holdout-sessions", nargs="*")
    a = p.parse_args(argv)
    c = dict(CONFIG)
    for flag, key in (("checkpoint", "CHECKPOINT"), ("dataset", "DATASET_DIR"),
                      ("sessions", "SESSIONS_ROOT"), ("out", "OUT_DIR"),
                      ("backend", "BACKEND"), ("device", "DEVICE"),
                      ("conf", "CONF"), ("stride", "STRIDE"),
                      ("max_scan", "MAX_SCAN"), ("batch_size", "BATCH_SIZE"),
                      ("only_sessions", "ONLY_SESSIONS"),
                      ("holdout_sessions", "HOLDOUT_SESSIONS")):
        v = getattr(a, flag)
        if v is not None:
            c[key] = v
    return mine(c)


if __name__ == "__main__":
    main()
