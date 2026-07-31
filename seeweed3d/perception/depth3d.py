#!/usr/bin/env python3
"""
SeeWeed3D - RGB-D conversion of an accepted LEP into a 3D camera-frame point.

Builds on `common/depth_utils.py` (the canonical depth reader and the robust
neighbourhood sampler) and adds what a laser target needs beyond a single
distance: probability-weighted sampling, ownership intersection, discontinuity
detection, a 3D covariance, and an explicit refusal.

THE FAILURE MODE THIS EXISTS TO STOP
------------------------------------
A stereo pixel at a plant crown is frequently wrong: it may be the soil seen
through the whorl, an occluding neighbour leaf, or an unmatched pixel. Reading
one pixel and back-projecting it produces a confident 3D point that is metres
from the plant. Every function here therefore returns a REASON alongside its
result and abstains rather than interpolating over a discontinuity - the depth
gap between a leaf and the ground behind it is exactly where a naive average
lands on nothing at all.

No metric accuracy is claimed anywhere. The covariance describes the SPREAD of
the depth samples that produced the point, which is an internal consistency
measure, not agreement with a surveyed ground truth. Establishing metric
accuracy needs 3D reference labels this repository does not yet have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.depth_utils import backproject  # noqa: E402


def _disc(shape, u, v, radius_px):
    h, w = shape
    ui, vi = int(round(u)), int(round(v))
    y0, y1 = max(0, vi - radius_px), min(h, vi + radius_px + 1)
    x0, x1 = max(0, ui - radius_px), min(w, ui + radius_px + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    within = ((yy - vi) ** 2 + (xx - ui) ** 2) <= radius_px ** 2
    return (slice(y0, y1), slice(x0, x1)), within


def sample_depth_weighted(depth_mm, valid, uv, weight_map=None, mask=None,
                          radius_px=7, min_valid_px=8, max_spread_mm=40.0,
                          discontinuity_mm=25.0):
    """Robust depth at a predicted LEP, weighted by the model's own confidence.

    weight_map (the LEP heatmap resampled to full-frame) lets the sample follow
    where the network actually believes the crown is, instead of weighting a
    pixel 6 px away as heavily as the peak. `mask` restricts sampling to the
    OWNING instance, which is what stops a neighbouring plant's leaf - often at
    a very different depth - from contributing.

    Returns (z_mm | None, stats). None always comes with stats['reason'],
    machine-readable so the caller can record why it abstained."""
    if depth_mm is None or valid is None:
        return None, {"reason": "insufficient_valid_depth", "n_valid": 0,
                      "valid_fraction": 0.0}
    sel = _disc(depth_mm.shape, uv[0], uv[1], radius_px)
    if sel is None:
        return None, {"reason": "outside_frame", "n_valid": 0,
                      "valid_fraction": 0.0}
    win, within = sel
    ok = valid[win] & within & np.isfinite(depth_mm[win])
    if mask is not None:
        ok &= mask[win].astype(bool)

    n_window = int(within.sum())
    vals = depth_mm[win][ok]
    stats = {"n_valid": int(vals.size),
             "valid_fraction": float(vals.size / max(1, n_window))}
    if vals.size < min_valid_px:
        stats["reason"] = "insufficient_valid_depth"
        return None, stats

    w = None
    if weight_map is not None:
        w = np.clip(weight_map[win][ok].astype(np.float64), 0.0, None)
        if w.sum() <= 1e-9:
            w = None

    # --- Bimodality FIRST, on the raw samples ------------------------------
    # This has to precede outlier rejection. MAD centres on the majority
    # surface and discards the other one as "outliers", so a window straddling
    # a leaf and the soil 500 mm behind it would otherwise be reduced to a
    # confident reading of whichever surface happened to win - the exact
    # confidently-wrong failure this function exists to prevent. When both
    # surfaces have real support we do not know which one the crown is on, so
    # the honest answer is to refuse.
    s = np.sort(vals)
    if s.size >= 4:
        gaps = np.diff(s)
        gi = int(np.argmax(gaps))
        gap = float(gaps[gi])
        stats["max_gap_mm"] = gap
        if gap > discontinuity_mm:
            below = gi + 1
            minority = min(below, s.size - below) / float(s.size)
            stats["minority_fraction"] = float(minority)
            # A handful of stray pixels is an outlier population (MAD handles
            # it); a substantial second cluster is a genuine surface boundary.
            if minority >= 0.15:
                stats["reason"] = "depth_discontinuity"
                return None, stats

    # --- Robust centre ------------------------------------------------------
    # With a confidence map, centre the robust estimate on the WEIGHTED median:
    # outlier rejection should be anchored where the model believes the crown
    # is, not on whatever surface occupies the most pixels in the window.
    if w is not None:
        order = np.argsort(vals)
        cw = np.cumsum(w[order])
        if cw[-1] > 1e-9:
            med = float(vals[order][int(np.searchsorted(cw, 0.5 * cw[-1]))])
        else:
            med = float(np.median(vals))
    else:
        med = float(np.median(vals))

    mad = float(np.median(np.abs(vals - med))) * 1.4826
    keep = np.abs(vals - med) <= max(3.0 * mad, 5.0)
    inl = vals[keep]
    stats.update({"median_mm": med, "mad_mm": float(mad),
                  "n_inliers": int(inl.size)})
    if inl.size < min_valid_px:
        stats["reason"] = "outlier_dominated"
        return None, stats

    spread = float(inl.max() - inl.min())
    stats["spread_mm"] = spread
    if spread > max_spread_mm:
        stats["reason"] = "depth_discontinuity"
        return None, stats

    if w is not None and float(w[keep].sum()) > 1e-9:
        wi = w[keep]
        z = float(np.sum(inl * wi) / np.sum(wi))
        stats["weighted"] = True
    else:
        z = float(np.median(inl))
        stats["weighted"] = False

    stats["std_mm"] = float(np.std(inl))
    stats["reason"] = "ok"
    return z, stats


def localize_lep_3d(depth_mm, valid, uv, K, weight_map=None, mask=None,
                    radius_px=7, min_valid_px=8, max_spread_mm=40.0,
                    min_valid_fraction=0.35):
    """Accepted 2D LEP -> 3D camera-frame point (mm) with uncertainty.

    Returns a dict that always includes `ok` and `reason`. The 3D covariance is
    propagated from the pixel and depth spreads through the pinhole model, so
    an uncertain depth widens the 3D uncertainty rather than being hidden by a
    confident-looking single number."""
    out = {"ok": False, "reason": "not_evaluated", "xyz_mm": None,
           "sigma_mm": None, "covariance": None, "depth_stats": {},
           "used_depth": True}

    z, stats = sample_depth_weighted(
        depth_mm, valid, uv, weight_map=weight_map, mask=mask,
        radius_px=radius_px, min_valid_px=min_valid_px,
        max_spread_mm=max_spread_mm)
    out["depth_stats"] = stats
    if z is None:
        out["reason"] = stats.get("reason", "insufficient_valid_depth")
        return out
    if stats.get("valid_fraction", 0.0) < min_valid_fraction:
        out["reason"] = "insufficient_valid_depth"
        return out

    xyz = backproject(uv[0], uv[1], z, K)
    fx, fy, cx, cy = K
    # Pinhole propagation. sigma_z from the inlier spread; sigma_u/v from the
    # sampling radius, which bounds how far the reported pixel may sit from the
    # true crown given the neighbourhood that produced the depth.
    sz = float(stats.get("std_mm", 0.0)) or 1.0
    su = sv = float(radius_px) / 2.0
    dx_du = z / fx
    dx_dz = (uv[0] - cx) / fx
    dy_dv = z / fy
    dy_dz = (uv[1] - cy) / fy
    var_x = (dx_du * su) ** 2 + (dx_dz * sz) ** 2
    var_y = (dy_dv * sv) ** 2 + (dy_dz * sz) ** 2
    var_z = sz ** 2
    cov = np.diag([var_x, var_y, var_z])

    out.update({"ok": True, "reason": "ok",
                "xyz_mm": [float(v) for v in xyz],
                "covariance": cov.tolist(),
                "sigma_mm": float(np.sqrt(max(var_x, var_y, var_z))),
                "sigma_xyz_mm": [float(np.sqrt(var_x)), float(np.sqrt(var_y)),
                                 float(np.sqrt(var_z))]})
    return out


def bed_plane_fallback(uv, K, plane, sigma_mm=40.0):
    """Intersect the pixel ray with an assumed bed plane when depth failed.

    EXPLICITLY A FALLBACK. It assumes the growth point lies on the bed surface,
    which is wrong by the plant's own height - so it returns a deliberately
    large uncertainty and `is_fallback=True`, and the safety layer treats it as
    lower-grade evidence. It exists so a depth dropout degrades gracefully
    instead of losing the plant entirely, not as a substitute for measurement.

    plane: (n, d) with n . X + d = 0 in camera coordinates, millimetres."""
    fx, fy, cx, cy = K
    ray = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0], np.float64)
    n = np.asarray(plane[0], np.float64)
    d = float(plane[1])
    denom = float(n @ ray)
    if abs(denom) < 1e-9:
        return {"ok": False, "reason": "bed_plane_parallel", "xyz_mm": None,
                "is_fallback": True, "used_depth": False}
    t = -d / denom
    if t <= 0:
        return {"ok": False, "reason": "bed_plane_behind_camera", "xyz_mm": None,
                "is_fallback": True, "used_depth": False}
    xyz = ray * t
    return {"ok": True, "reason": "bed_plane_fallback",
            "xyz_mm": [float(v) for v in xyz],
            "sigma_mm": float(sigma_mm),
            "covariance": np.diag([sigma_mm ** 2] * 3).tolist(),
            "is_fallback": True, "used_depth": False}
