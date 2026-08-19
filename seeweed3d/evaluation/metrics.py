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


def mask_extent(m):
    """(y0, y1, x0, x1, area) for a mask, or None if it is empty.

    Two full-array reductions and then a sum over the bounding box only, so the
    cost is linear in the frame rather than in the frame times the number of
    pairs it will be compared against."""
    m = np.asarray(m, bool)
    rows = m.any(1)
    if not rows.any():
        return None
    y = np.flatnonzero(rows)
    x = np.flatnonzero(m.any(0))
    y0, y1, x0, x1 = int(y[0]), int(y[-1]), int(x[0]), int(x[-1])
    return y0, y1, x0, x1, int(m[y0:y1 + 1, x0:x1 + 1].sum())


def mask_iou_matrix(pred_masks, gt_masks):
    """(P, G) IoU matrix. EXACT - the same numbers mask_iou would give.

    The pairwise loop is what makes evaluation tractable at ZED resolution.
    A naive P x G x 2208 x 1242 boolean AND/OR costs about 4 ms per pair, so
    300 RF-DETR detections against 20 ground-truth instances across 10 IoU
    thresholds and 16 frames is an HOUR of pure numpy - which is where this
    evaluation was actually stalling, not in the model.

    Two exact savings, no approximation:
      * bounding boxes that do not intersect cannot have overlapping masks, so
        the pair is skipped without touching a pixel. Most pairs are this.
      * for the rest, the AND is computed over the intersection BOX rather than
        the frame - a few hundred pixels instead of 2.7 million.
    """
    P, G = len(pred_masks), len(gt_masks)
    out = np.zeros((P, G), float)
    if not P or not G:
        return out
    pe = [mask_extent(m) for m in pred_masks]
    ge = [mask_extent(m) for m in gt_masks]
    for i, pi_ in enumerate(pe):
        if pi_ is None:
            continue
        py0, py1, px0, px1, pa = pi_
        a = np.asarray(pred_masks[i], bool)
        for j, gj in enumerate(ge):
            if gj is None:
                continue
            gy0, gy1, gx0, gx1, ga = gj
            y0, y1 = max(py0, gy0), min(py1, gy1)
            x0, x1 = max(px0, gx0), min(px1, gx1)
            if y0 > y1 or x0 > x1:
                continue                      # boxes disjoint -> IoU 0
            b = np.asarray(gt_masks[j], bool)
            inter = int(np.count_nonzero(a[y0:y1 + 1, x0:x1 + 1]
                                         & b[y0:y1 + 1, x0:x1 + 1]))
            if inter:
                out[i, j] = inter / (pa + ga - inter)
    return out


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
                    iou_threshold=0.5, iou=None):
    """Greedy, highest-IoU-first, class-aware matching. Returns
    (matches, unmatched_pred, unmatched_gt).

    `iou` accepts a precomputed (P, G) matrix from mask_iou_matrix. Callers
    that match the same masks at several thresholds or confidences should pass
    one: the matrix is the entire cost, and recomputing it per threshold is the
    difference between seconds and an hour."""
    if iou is None:
        iou = mask_iou_matrix(pred_masks, gt_masks)
    pairs = []
    for i in range(len(pred_masks)):
        for j in range(len(gt_masks)):
            if pred_classes[i] != gt_classes[j]:
                continue
            v = float(iou[i, j])
            if v >= iou_threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matches.append({"pred": i, "gt": j, "iou": float(v)})
    return (matches,
            [i for i in range(len(pred_masks)) if i not in used_p],
            [j for j in range(len(gt_masks)) if j not in used_g])


def segmentation_metrics(pred_masks, pred_classes, gt_masks, gt_classes,
                         iou_thresholds=None, small_area_px=1500,
                         classes=None):
    """Per-class precision/recall/AP-style summary plus crop-safety pixels.

    `classes` is the class list to report over; pass the model's ACTIVE list
    when it was trained on a reduced set, otherwise the table gains rows for
    classes the model was never taught and cannot predict."""
    class_list = list(classes or CLASSES)
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
    for c in class_list:
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
# Mixed scenes: the asymmetry, and instance identity
# --------------------------------------------------------------------------- #
#: A predicted instance must claim this fraction of a true instance before it
#: counts as having swallowed it, and a prediction must lie this far inside one
#: true instance before it counts as a fragment of it. Below a half the same
#: pair could be called both at once.
CLAIM_FRACTION = 0.5

#: Half-width, in pixels, of the band around an onion/weed contact. Errors here
#: are the ones that put a laser next to crop, so they are reported apart from
#: the frame-wide numbers that would average them away.
CONTACT_BAND_PX = 12


def _union(masks, classes, want_crop):
    """Union of every mask whose class is (or is not) the crop."""
    out = None
    for m, c in zip(masks, classes):
        if (c == CROP_CLASS) != want_crop:
            continue
        a = np.asarray(m).astype(bool)
        out = a if out is None else (out | a)
    return out


def _dilate(mask, px):
    import cv2
    if mask is None or px <= 0:
        return mask
    k = 2 * int(px) + 1
    return cv2.dilate(np.asarray(mask, np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                      ).astype(bool)


def crop_confusion(pred_masks, pred_classes, gt_masks, gt_classes,
                   contact_band_px=CONTACT_BAND_PX):
    """The two directions of crop/weed error, never averaged together.

    onion_as_weed is the one that fires a laser at the crop. weed_as_onion only
    skips a weed. A mean IoU over the frame reports these as one number, and a
    model can look excellent while making the first kind - which is why this is
    reported apart from the class table, exactly as onion_safety_metrics is.

    The contact band restricts the same measurement to pixels near an
    onion/weed boundary, because that is where the decision is actually
    exercised: a frame of well-separated plants can score perfectly while every
    contact in it is wrong."""
    gt_on = _union(gt_masks, gt_classes, True)
    gt_wd = _union(gt_masks, gt_classes, False)
    pr_on = _union(pred_masks, pred_classes, True)
    pr_wd = _union(pred_masks, pred_classes, False)

    shape = None
    for m in list(gt_masks) + list(pred_masks):
        shape = np.asarray(m).shape
        break
    if shape is None:
        return {"note": "no masks"}
    z = np.zeros(shape, bool)
    gt_on = z if gt_on is None else gt_on
    gt_wd = z if gt_wd is None else gt_wd
    pr_on = z if pr_on is None else pr_on
    pr_wd = z if pr_wd is None else pr_wd

    on_as_weed = int((gt_on & pr_wd).sum())
    wd_as_onion = int((gt_wd & pr_on).sum())
    # Crop the model saw as nothing at all. Not the same failure as calling it
    # weed - it will not be fired at - but it is still crop the system cannot
    # protect, so it is counted rather than folded into either direction.
    on_unclaimed = int((gt_on & ~pr_on & ~pr_wd).sum())

    out = {
        "onion_as_weed_px": on_as_weed,
        "onion_as_weed_fraction": (float(on_as_weed / gt_on.sum())
                                   if gt_on.any() else None),
        "weed_as_onion_px": wd_as_onion,
        "weed_as_onion_fraction": (float(wd_as_onion / gt_wd.sum())
                                   if gt_wd.any() else None),
        "onion_unclaimed_px": on_unclaimed,
        "gt_onion_px": int(gt_on.sum()), "gt_weed_px": int(gt_wd.sum()),
    }

    band = None
    if gt_on.any() and gt_wd.any():
        band = _dilate(gt_on, contact_band_px) & _dilate(gt_wd, contact_band_px)
    if band is not None and band.any():
        b_on, b_wd = gt_on & band, gt_wd & band
        out["contact_band_px_count"] = int(band.sum())
        out["contact_onion_as_weed_px"] = int((b_on & pr_wd).sum())
        out["contact_onion_as_weed_fraction"] = (
            float((b_on & pr_wd).sum() / b_on.sum()) if b_on.any() else None)
        out["contact_weed_as_onion_fraction"] = (
            float((b_wd & pr_on).sum() / b_wd.sum()) if b_wd.any() else None)
    else:
        # No contact in this frame. Reported as None, not 0: "nothing to get
        # wrong here" and "got the hard part right" are different claims.
        out["contact_band_px_count"] = 0
        out["contact_onion_as_weed_fraction"] = None
        out["contact_weed_as_onion_fraction"] = None
    return out


def identity_errors(pred_masks, gt_masks, claim_fraction=CLAIM_FRACTION):
    """Merges and fragments - the failure the mixed prelabeler actually has.

    Class-agnostic on purpose: identity is a separate question from labelling,
    and a merge of two plants is the same defect whatever they are called.

      merged    one prediction claims most of two or more true instances
      fragment  two or more predictions lie mostly inside one true instance

    Both are reported because they are opposite failures with opposite fixes,
    and a single "instance count error" cancels them against each other - a
    frame that merges two plants and shatters a third scores perfectly."""
    P, G = len(pred_masks), len(gt_masks)
    pm = [np.asarray(m).astype(bool) for m in pred_masks]
    gm = [np.asarray(m).astype(bool) for m in gt_masks]
    p_area = [max(1, int(m.sum())) for m in pm]
    g_area = [max(1, int(m.sum())) for m in gm]

    merged, fragmented = [], []
    inter = np.zeros((P, G), np.int64)
    for i in range(P):
        for j in range(G):
            inter[i, j] = int((pm[i] & gm[j]).sum())

    for i in range(P):
        claimed = [j for j in range(G)
                   if inter[i, j] / g_area[j] >= claim_fraction]
        if len(claimed) >= 2:
            merged.append({"pred": i, "gt": claimed})
    for j in range(G):
        inside = [i for i in range(P)
                  if inter[i, j] / p_area[i] >= claim_fraction]
        if len(inside) >= 2:
            fragmented.append({"gt": j, "pred": inside})

    return {"n_pred": P, "n_gt": G, "count_error": P - G,
            "n_merged_predictions": len(merged),
            "n_merged_gt_instances": sum(len(m["gt"]) for m in merged),
            "n_fragmented_gt": len(fragmented),
            "merge_rate": float(len(merged) / P) if P else None,
            "fragment_rate": float(len(fragmented) / G) if G else None,
            "merged": merged, "fragmented": fragmented}


def cluster_over_prediction(pred_masks, pred_classes, gt_masks, gt_classes,
                            cluster_class="weed_cluster",
                            claim_fraction=CLAIM_FRACTION):
    """Predicted clusters covering ground truth that IS separable.

    Tracked because annotation policy becomes deployed policy: a cluster class
    used when separation is merely tedious teaches the model to do the same at
    runtime, where it means weeds that never receive an individual LEP. A rising
    rate is the early warning, visible long before a missed weed in the field.

    Only the direction that costs targets is counted. Predicting separate
    instances where the truth is a cluster is not the same defect - it produces
    targets that can be checked, not targets that silently never exist."""
    gm = [np.asarray(m).astype(bool) for m in gt_masks]
    g_area = [max(1, int(m.sum())) for m in gm]
    n_clusters = bad = 0
    for m, c in zip(pred_masks, pred_classes):
        if c != cluster_class:
            continue
        n_clusters += 1
        a = np.asarray(m).astype(bool)
        covered = [j for j in range(len(gm))
                   if gt_classes[j] != cluster_class
                   and int((a & gm[j]).sum()) / g_area[j] >= claim_fraction]
        if len(covered) >= 2:
            bad += 1
    return {"n_predicted_clusters": n_clusters,
            "clusters_over_separable_gt": bad,
            "cluster_over_prediction_rate": (float(bad / n_clusters)
                                             if n_clusters else None)}


def mixed_scene_metrics(pred_masks, pred_classes, gt_masks, gt_classes,
                        contact_band_px=CONTACT_BAND_PX,
                        claim_fraction=CLAIM_FRACTION):
    """Every mixed-scene number for ONE frame, in one call.

    Deliberately NOT a single score. The three groups answer different
    questions - is the crop safe, are the instances right, is the cluster class
    being over-used - and collapsing them would reproduce the averaging this
    module exists to avoid."""
    return {
        "crop": crop_confusion(pred_masks, pred_classes, gt_masks, gt_classes,
                               contact_band_px),
        "identity": identity_errors(pred_masks, gt_masks, claim_fraction),
        "cluster": cluster_over_prediction(pred_masks, pred_classes, gt_masks,
                                           gt_classes,
                                           claim_fraction=claim_fraction),
    }


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
