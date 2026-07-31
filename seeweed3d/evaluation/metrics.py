#!/usr/bin/env python3
"""
SeeWeed3D - evaluation metrics for segmentation, LEP, safety and 3D.

Pure numpy so metrics can be recomputed from stored results without a GPU or a
model. Every function returns a dict, and no function invents a number it
cannot compute: when the inputs for a metric are absent it is reported as None
rather than defaulted to 0, because a 0 in an accuracy table is indistinguishable
from a real measurement.

CROP SAFETY IS ASYMMETRIC and the metrics say so. Missing an onion (recall) can
destroy the crop; a false onion merely skips a weed. Onion RECALL and missed
onion AREA are therefore first-class, separate from IoU, which averages the two
error directions together and can look healthy while crop tissue is missed.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


def mask_dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum() + b.sum())
    return float(2.0 * (a & b).sum() / denom) if denom else 0.0


def boundary_f_score(pred, gt, tolerance_px=2):
    """F-score of boundary pixels within a tolerance.

    Reported because IoU is dominated by a plant's interior and barely moves
    when the leaf margin is wrong by several pixels - and the margin is exactly
    what a laser aiming at a crown near a crop edge depends on."""
    import cv2
    pred, gt = pred.astype(np.uint8), gt.astype(np.uint8)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    k = np.ones((3, 3), np.uint8)
    pb = (cv2.dilate(pred, k) - cv2.erode(pred, k)).astype(bool)
    gb = (cv2.dilate(gt, k) - cv2.erode(gt, k)).astype(bool)
    if not pb.any() or not gb.any():
        return 0.0
    dt_g = cv2.distanceTransform(1 - gb.astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform(1 - pb.astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((dt_g[pb] <= tolerance_px).mean())
    rec = float((dt_p[gb] <= tolerance_px).mean())
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def match_instances(pred_masks, pred_classes, gt_masks, gt_classes,
                    iou_threshold=0.5):
    """Greedy, highest-IoU-first, class-aware matching. Returns
    (matches, unmatched_pred, unmatched_gt)."""
    pairs = []
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            if pred_classes[i] != gt_classes[j]:
                continue
            iou = mask_iou(pm, gm)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matches.append({"pred": i, "gt": j, "iou": float(iou)})
    return (matches,
            [i for i in range(len(pred_masks)) if i not in used_p],
            [j for j in range(len(gt_masks)) if j not in used_g])


def segmentation_metrics(pred_masks, pred_classes, gt_masks, gt_classes,
                         iou_thresholds=None, small_area_px=1500):
    """Per-class precision/recall/AP-style summary plus crop-safety pixels."""
    thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
    out = {"per_class": {}, "mask_ap50_95": None, "small_weed_recall": None}

    per_thr = []
    for thr in thresholds:
        m, up, ug = match_instances(pred_masks, pred_classes, gt_masks,
                                    gt_classes, thr)
        tp, fp, fn = len(m), len(up), len(ug)
        per_thr.append(tp / max(1, tp + fp + fn))
    out["mask_ap50_95"] = float(np.mean(per_thr)) if per_thr else None

    m50, up50, ug50 = match_instances(pred_masks, pred_classes, gt_masks,
                                      gt_classes, 0.5)
    matched_gt = {x["gt"] for x in m50}
    for c in CLASSES:
        gt_idx = [j for j, g in enumerate(gt_classes) if g == c]
        pr_idx = [i for i, p in enumerate(pred_classes) if p == c]
        if not gt_idx and not pr_idx:
            continue
        tp = sum(1 for j in gt_idx if j in matched_gt)
        out["per_class"][c] = {
            "n_gt": len(gt_idx), "n_pred": len(pr_idx), "tp": tp,
            "precision": float(tp / len(pr_idx)) if pr_idx else None,
            "recall": float(tp / len(gt_idx)) if gt_idx else None,
            "mean_iou": float(np.mean([x["iou"] for x in m50
                                       if gt_classes[x["gt"]] == c]))
            if any(gt_classes[x["gt"]] == c for x in m50) else None}

    small = [j for j in range(len(gt_masks))
             if gt_classes[j] != CROP_CLASS
             and int(np.asarray(gt_masks[j]).sum()) <= small_area_px]
    if small:
        out["small_weed_recall"] = float(
            sum(1 for j in small if j in matched_gt) / len(small))
    return out


def onion_safety_metrics(pred_onion, gt_onion):
    """Crop-protection metrics, deliberately separate from the class table.

    missed_onion_px is the headline: those are crop pixels the system believes
    are safe to fire at."""
    if gt_onion is None:
        return {"onion_recall": None, "note": "no ground-truth onion mask"}
    gt = np.asarray(gt_onion).astype(bool)
    pred = (np.zeros_like(gt) if pred_onion is None
            else np.asarray(pred_onion).astype(bool))
    inter = int((pred & gt).sum())
    missed = int((gt & ~pred).sum())
    return {"onion_recall": float(inter / gt.sum()) if gt.sum() else None,
            "onion_precision": float(inter / pred.sum()) if pred.sum() else None,
            "onion_iou": mask_iou(pred, gt),
            "onion_dice": mask_dice(pred, gt),
            "onion_boundary_f": boundary_f_score(pred, gt),
            "missed_onion_px": missed,
            "missed_onion_fraction": float(missed / gt.sum()) if gt.sum() else None}


# --------------------------------------------------------------------------- #
# LEP localisation
# --------------------------------------------------------------------------- #
def lep_errors(pred_uv, gt_uv, plant_radius_px=None):
    """Per-instance pixel error, and error normalised by plant size.

    The normalised error is the one that compares fairly across growth stages:
    5 px on a cotyledon is a miss, 5 px on a large rosette is a hit."""
    p = np.asarray(pred_uv, np.float64).reshape(-1, 2)
    g = np.asarray(gt_uv, np.float64).reshape(-1, 2)
    err = np.linalg.norm(p - g, axis=1)
    out = {"n": int(err.size), "errors_px": err.tolist()}
    if err.size:
        out.update({"mean_px": float(err.mean()),
                    "median_px": float(np.median(err)),
                    "p95_px": float(np.percentile(err, 95))})
        for t in (2, 5, 10, 15):
            out[f"pct_within_{t}px"] = float((err <= t).mean() * 100.0)
    if plant_radius_px is not None:
        r = np.asarray(plant_radius_px, np.float64).reshape(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(r > 0, err / r, np.nan)
        finite = norm[np.isfinite(norm)]
        if finite.size:
            out["median_normalised"] = float(np.median(finite))
            for f in (0.1, 0.25, 0.5):
                out[f"pct_within_{f}_radius"] = float((finite <= f).mean() * 100.0)
    return out


def lep_inside_mask_rate(pred_uv, masks):
    """Fraction of predictions landing on their own plant. A prediction outside
    its owning mask is a wrong-instance error, not merely an inaccurate one."""
    ok = 0
    n = 0
    for uv, m in zip(pred_uv, masks):
        if m is None:
            continue
        n += 1
        u, v = int(round(uv[0])), int(round(uv[1]))
        if 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and m[v, u]:
            ok += 1
    return float(ok / n) if n else None


def compare_lep_methods(gt_uv, methods, plant_radius_px=None):
    """Head-to-head against every baseline the plan requires.

    `methods` maps a name -> predicted uv array. Pass bbox_center, centroid,
    dt_peak and the perception/lep.py estimate alongside the learned model, so
    the comparison is computed from one stored result set."""
    return {name: lep_errors(uv, gt_uv, plant_radius_px)
            for name, uv in methods.items()}


def uncertainty_calibration(sigmas, errors, n_bins=5):
    """Does predicted sigma track actual error?

    An uncertainty that does not correlate with error is worse than none: the
    abstention threshold would then reject good targets and pass bad ones."""
    s = np.asarray(sigmas, np.float64)
    e = np.asarray(errors, np.float64)
    keep = np.isfinite(s) & np.isfinite(e)
    s, e = s[keep], e[keep]
    if s.size < n_bins:
        return {"n": int(s.size), "spearman": None,
                "note": "too few samples to calibrate"}
    order = np.argsort(s)
    bins = np.array_split(order, n_bins)
    rows = [{"mean_sigma_px": float(s[b].mean()),
             "mean_error_px": float(e[b].mean()), "n": int(b.size)}
            for b in bins if b.size]
    rs, re = np.argsort(np.argsort(s)), np.argsort(np.argsort(e))
    rho = float(np.corrcoef(rs, re)[0, 1]) if s.size > 2 else None
    return {"n": int(s.size), "bins": rows, "spearman": rho}


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
def safety_metrics(targets, gt_lookup=None):
    """Aggregate safety behaviour over a set of WeedTarget-like dicts.

    unsafe_target_rate is the number that matters most: candidates that should
    never have been approved."""
    n = len(targets)
    if not n:
        return {"n": 0}
    cand = [t for t in targets if t.get("safety_status") == "candidate"]
    reasons = defaultdict(int)
    for t in targets:
        for r in t.get("rejection_reasons", []):
            reasons[r] += 1

    out = {"n": n, "n_candidates": len(cand),
           "abstention_rate": float(sum(1 for t in targets
                                        if t.get("abstained")) / n),
           "rejection_reasons": dict(reasons),
           "onion_conflict_rate": float(reasons.get("onion_safety_conflict", 0) / n)}

    if gt_lookup:
        unsafe = wrong_inst = 0
        targetable_total = targetable_hit = 0
        for t in targets:
            key = (t.get("frame_id"), t.get("instance_index"))
            gt = gt_lookup.get(key)
            if gt is None:
                continue
            if gt.get("class_name") == CROP_CLASS and \
                    t.get("safety_status") == "candidate":
                unsafe += 1
            if gt.get("targetable") == "yes" and gt.get("class_name") != CROP_CLASS:
                targetable_total += 1
                if t.get("safety_status") == "candidate":
                    targetable_hit += 1
            if t.get("safety_status") == "candidate" and \
                    gt.get("wrong_instance"):
                wrong_inst += 1
        out.update({
            "unsafe_target_rate": float(unsafe / n),
            "wrong_instance_lep_rate": float(wrong_inst / max(1, len(cand))),
            "recall_among_targetable": (float(targetable_hit / targetable_total)
                                        if targetable_total else None),
            "false_candidate_rate": (float((len(cand) - targetable_hit) /
                                           max(1, len(cand)))
                                     if targetable_total else None)})
    return out


# --------------------------------------------------------------------------- #
# 3D  (only meaningful with reference 3D labels)
# --------------------------------------------------------------------------- #
def metrics_3d(pred_xyz, gt_xyz):
    """3D error. Returns None-filled results when no reference exists.

    NEVER call the internal covariance an accuracy: it measures the spread of
    the depth samples, not agreement with a surveyed point."""
    if gt_xyz is None or not len(gt_xyz):
        return {"n": 0, "note": "no reference 3D labels; 3D accuracy is NOT "
                                "measurable from depth self-consistency"}
    p = np.asarray(pred_xyz, np.float64).reshape(-1, 3)
    g = np.asarray(gt_xyz, np.float64).reshape(-1, 3)
    d = np.linalg.norm(p - g, axis=1)
    xy = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
    z = np.abs(p[:, 2] - g[:, 2])
    return {"n": int(d.size), "mean_mm": float(d.mean()),
            "median_mm": float(np.median(d)),
            "p95_mm": float(np.percentile(d, 95)),
            "xy_median_mm": float(np.median(xy)),
            "z_median_mm": float(np.median(z))}


def latency_summary(timings):
    """p50/p95 per stage from a list of per-frame timing dicts."""
    if not timings:
        return {}
    keys = set().union(*[set(t) for t in timings])
    out = {}
    for k in sorted(keys):
        vals = [float(t[k]) for t in timings if k in t]
        if vals:
            out[k] = {"p50_ms": float(np.percentile(vals, 50)),
                      "p95_ms": float(np.percentile(vals, 95)),
                      "mean_ms": float(np.mean(vals)), "n": len(vals)}
    return out
