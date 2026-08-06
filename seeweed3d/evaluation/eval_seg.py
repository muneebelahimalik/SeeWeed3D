#!/usr/bin/env python3
"""
SeeWeed3D - Stage A evaluation on a held-out split.

Training reports loss. Loss is not a model quality number: it is not comparable
between runs with different class counts, it says nothing about whether small
weeds are found, and it says nothing about whether onions are protected. This
script produces the numbers you would actually report.

    python -m seeweed3d.evaluation.eval_seg \
        --checkpoint  D:/runs/seg_v1/best.pt \
        --dataset     D:/training/subset35 \
        --images-root D:/Dataset_Vidalia/sessions \
        --split val --device cuda

Three separate tables, because they answer different questions:

  detection   mAP@50 and mAP@50:95, per class. Score-ranked, 101-point
              interpolated - the standard definition, so these are comparable
              with published numbers.
  operating   precision/recall at the confidence you would actually deploy at.
              mAP is threshold-free; a robot is not.
  crop safety missed onion pixels. This is the number that decides whether the
              laser is safe to arm, and it is deliberately NOT averaged into
              any headline score - a model can have excellent mAP and still
              miss the one onion that matters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402
from evaluation.metrics import (mask_iou, boundary_f_score,  # noqa: E402
                                match_instances)

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def average_precision(scores, is_tp, n_gt):
    """101-point interpolated AP, the COCO definition.

    `is_tp` must already be assigned by score-descending greedy matching, which
    is what makes this AP rather than a threshold-free F-score: a duplicate
    detection of an already-matched object is a false positive, and ranking
    decides which of the duplicates gets to be the true positive."""
    if n_gt == 0:
        return None                       # undefined, not zero
    if not len(scores):
        return 0.0
    order = np.argsort(-np.asarray(scores, float))
    tp = np.asarray(is_tp, bool)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # Make precision monotonically decreasing, then sample at 101 recalls.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    q = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, q, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(
        idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


def match_for_ap(pred_masks, pred_scores, gt_masks, iou_threshold):
    """Score-descending greedy matching within ONE class and ONE frame.

    Returns a bool per prediction. Highest-scoring prediction claims the best
    remaining ground truth, which is the COCO rule."""
    order = np.argsort(-np.asarray(pred_scores, float))
    taken, is_tp = set(), np.zeros(len(pred_scores), bool)
    for i in order:
        best_j, best_iou = None, iou_threshold
        for j in range(len(gt_masks)):
            if j in taken:
                continue
            v = mask_iou(pred_masks[i], gt_masks[j])
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j is not None:
            taken.add(best_j)
            is_tp[i] = True
    return is_tp


def evaluate(checkpoint, dataset_dir, images_root, split="val", device="cpu",
             conf=0.5, ap_conf=0.05, small_area_px=1500, min_area_px=16,
             mask_threshold=0.5):
    import cv2
    from perception.segmenter import MaskRCNNSegmenter
    from training.seg_dataset import polygons_to_mask, resolve_image

    man_path = Path(dataset_dir) / "seg_manifest.json"
    if not man_path.exists():
        raise SystemExit(f"ERROR: {man_path} not found. Build the dataset "
                         f"with prepare_dataset first.")
    doc = json.loads(man_path.read_text(encoding="utf-8"))
    frames = [f for f in doc["frames"] if f.get("split") == split]
    if not frames:
        raise SystemExit(
            f"ERROR: split {split!r} has no frames. See "
            f"{Path(dataset_dir) / 'splits' / 'splits_summary.json'}.")

    # ap_conf, not conf: AP integrates a precision/recall curve, so it needs the
    # low-scoring tail. Thresholding at the operating point first would truncate
    # the curve and inflate AP.
    seg = MaskRCNNSegmenter(checkpoint, conf=min(conf, ap_conf), device=device,
                            mask_threshold=mask_threshold).load()
    classes = list(seg.classes or doc.get("classes") or CLASSES)
    manifest_classes = list(doc.get("classes") or CLASSES)
    if classes != manifest_classes:
        raise SystemExit(
            f"ERROR: checkpoint classes {classes} do not match dataset classes "
            f"{manifest_classes}. Evaluating across a class-list mismatch "
            f"silently compares different labels - rebuild or retrain so the "
            f"two agree.")

    # A single path or a list - NOT collapsed to one Path here, since a
    # merged multi-source build's sessions can live under more than one
    # sessions folder. resolve_image tries every root.
    root = images_root or doc.get("images_root") or "."
    per_class_ap = {c: {t: {"scores": [], "is_tp": [], "n_gt": 0}
                        for t in IOU_THRESHOLDS} for c in classes}
    op = {c: {"tp": 0, "n_pred": 0, "n_gt": 0, "ious": []} for c in classes}
    small_total = small_hit = 0
    crop_gt_px = crop_missed_px = 0
    crop_boundary_f, crop_frames = [], 0
    n_frames = 0

    for rec in frames:
        try:
            path = resolve_image(rec["image_path"], root, rec.get("session_id"))
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"ERROR: cannot read {path}")
        h, w = bgr.shape[:2]

        gt_masks, gt_names = [], []
        for inst in rec["instances"]:
            m = polygons_to_mask(inst["polygons"], h, w).astype(bool)
            if int(m.sum()) < min_area_px:
                continue
            gt_masks.append(m)
            gt_names.append(inst["class_name"])

        det = seg(bgr)
        pred_masks = [det.masks[i].astype(bool) for i in range(len(det))]
        pred_names = [det.class_name(i) for i in range(len(det))]
        pred_scores = [float(det.scores[i]) for i in range(len(det))]
        n_frames += 1

        for c in classes:
            gi = [k for k, n in enumerate(gt_names) if n == c]
            pi = [k for k, n in enumerate(pred_names) if n == c]
            gm = [gt_masks[k] for k in gi]
            pm = [pred_masks[k] for k in pi]
            ps = [pred_scores[k] for k in pi]
            for t in IOU_THRESHOLDS:
                b = per_class_ap[c][t]
                b["n_gt"] += len(gm)
                if pm:
                    tp = match_for_ap(pm, ps, gm, t)
                    b["scores"].extend(ps)
                    b["is_tp"].extend(tp.tolist())

        # Operating point: the same greedy IoU matching used elsewhere in the
        # codebase, applied to detections above the DEPLOYED confidence.
        keep = [i for i in range(len(det)) if pred_scores[i] >= conf]
        op_masks = [pred_masks[i] for i in keep]
        op_names = [pred_names[i] for i in keep]
        m50, _, _ = match_instances(op_masks, op_names, gt_masks, gt_names, 0.5)
        matched_gt = {x["gt"] for x in m50}
        for c in classes:
            op[c]["n_gt"] += sum(1 for n in gt_names if n == c)
            op[c]["n_pred"] += sum(1 for n in op_names if n == c)
        for x in m50:
            c = gt_names[x["gt"]]
            op[c]["tp"] += 1
            op[c]["ious"].append(x["iou"])

        for j, m in enumerate(gt_masks):
            if gt_names[j] != CROP_CLASS and int(m.sum()) <= small_area_px:
                small_total += 1
                small_hit += int(j in matched_gt)

        # Crop safety at the operating point, in PIXELS. Instance recall would
        # score a barely-overlapping onion detection as a save; the laser aims
        # at pixels, so pixels are what is counted.
        gt_crop = np.zeros((h, w), bool)
        for j, m in enumerate(gt_masks):
            if gt_names[j] == CROP_CLASS:
                gt_crop |= m
        if gt_crop.any():
            pred_crop = np.zeros((h, w), bool)
            for i in keep:
                if pred_names[i] == CROP_CLASS:
                    pred_crop |= pred_masks[i]
            crop_gt_px += int(gt_crop.sum())
            crop_missed_px += int((gt_crop & ~pred_crop).sum())
            crop_boundary_f.append(boundary_f_score(pred_crop, gt_crop))
            crop_frames += 1

    detection = {}
    for c in classes:
        aps = {}
        for t in IOU_THRESHOLDS:
            b = per_class_ap[c][t]
            aps[t] = average_precision(b["scores"], b["is_tp"], b["n_gt"])
        vals = [v for v in aps.values() if v is not None]
        detection[c] = {
            "n_gt": per_class_ap[c][0.5]["n_gt"],
            "ap50": aps[0.5],
            "ap50_95": float(np.mean(vals)) if vals else None,
        }
    present = [c for c in classes if detection[c]["ap50"] is not None]
    summary = {
        "split": split, "n_frames": n_frames,
        "classes": classes,
        # Averaged over classes PRESENT in this split only. A class with no
        # ground truth here has undefined AP, and scoring it as 0 would punish
        # the model for a gap in the annotation set rather than in the model.
        "map50": (float(np.mean([detection[c]["ap50"] for c in present]))
                  if present else None),
        "map50_95": (float(np.mean([detection[c]["ap50_95"] for c in present]))
                     if present else None),
        "classes_without_ground_truth": [c for c in classes if c not in present],
    }

    operating = {"conf": conf}
    for c in classes:
        d = op[c]
        operating[c] = {
            "n_gt": d["n_gt"], "n_pred": d["n_pred"], "tp": d["tp"],
            "precision": float(d["tp"] / d["n_pred"]) if d["n_pred"] else None,
            "recall": float(d["tp"] / d["n_gt"]) if d["n_gt"] else None,
            "mean_iou": float(np.mean(d["ious"])) if d["ious"] else None,
        }
    operating["small_weed_recall"] = (float(small_hit / small_total)
                                      if small_total else None)
    operating["small_weed_n"] = small_total

    crop = {
        "frames_with_onion": crop_frames,
        "onion_gt_px": crop_gt_px,
        "missed_onion_px": crop_missed_px,
        "missed_onion_fraction": (float(crop_missed_px / crop_gt_px)
                                  if crop_gt_px else None),
        "onion_boundary_f": (float(np.mean(crop_boundary_f))
                             if crop_boundary_f else None),
    }
    if not crop_frames:
        crop["note"] = ("no onion_plant ground truth in this split - crop "
                        "safety is UNMEASURED, not passing")

    return {"summary": summary, "detection": detection,
            "operating_point": operating, "crop_safety": crop}


def format_report(res):
    s, d, o, c = (res["summary"], res["detection"], res["operating_point"],
                  res["crop_safety"])
    L = []
    def f(v, n=3):
        return "  -  " if v is None else f"{v:.{n}f}"
    L.append(f"split={s['split']}  frames={s['n_frames']}")
    L.append(f"mAP@50={f(s['map50'])}   mAP@50:95={f(s['map50_95'])}")
    L.append("")
    L.append(f"{'class':<28}{'n_gt':>6}{'AP50':>9}{'AP50-95':>10}"
             f"{'P':>8}{'R':>8}{'IoU':>8}")
    for cls in s["classes"]:
        L.append(f"{cls:<28}{d[cls]['n_gt']:>6}{f(d[cls]['ap50']):>9}"
                 f"{f(d[cls]['ap50_95']):>10}{f(o[cls]['precision']):>8}"
                 f"{f(o[cls]['recall']):>8}{f(o[cls]['mean_iou']):>8}")
    L.append("")
    L.append(f"P/R/IoU measured at conf={o['conf']}")
    if s["classes_without_ground_truth"]:
        L.append(f"NO ground truth in this split (AP undefined, excluded from "
                 f"the mean): {', '.join(s['classes_without_ground_truth'])}")
    L.append(f"small-weed recall (<=1500 px): {f(o['small_weed_recall'])} "
             f"over {o['small_weed_n']} instances")
    L.append("")
    L.append("CROP SAFETY")
    if "note" in c:
        L.append(f"  {c['note']}")
    else:
        L.append(f"  missed onion pixels: {c['missed_onion_px']} / "
                 f"{c['onion_gt_px']}  ({f(c['missed_onion_fraction'], 4)})")
        L.append(f"  onion boundary F:    {f(c['onion_boundary_f'])}")
        L.append("  Missed onion pixels are pixels the system believes are "
                 "safe to fire at.")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="prepare_dataset output dir")
    p.add_argument("--images-root", required=True, nargs="+",
                   help="sessions root(s); more than one if the datasets were "
                        "not captured under a common parent")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.5,
                   help="deployment confidence, for the P/R table")
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--out", default=None, help="write metrics JSON here")
    a = p.parse_args(argv)

    res = evaluate(a.checkpoint, a.dataset, a.images_root, a.split, a.device,
                   conf=a.conf, mask_threshold=a.mask_threshold)
    print(format_report(res))
    out = Path(a.out) if a.out else Path(a.checkpoint).parent / \
        f"metrics_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
