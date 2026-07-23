#!/usr/bin/env python3
"""
SeeWeed3D - canonical depth reader.

Import this everywhere instead of calling cv2.imread on a depth PNG. The two
mistakes it exists to prevent:

  1. Rescaling by `depth_vis_max_mm` (3000) from session_meta.txt. That value
     drives the capture GUI preview ONLY. Stored depth is raw millimetres.
  2. Treating 0 as a distance. 0 is the invalid sentinel - the capture code
     collapses NaN, +inf (too far) and -inf (too near) to 0 before encoding.
"""

import json
from pathlib import Path

import cv2
import numpy as np

INVALID = 0
MM_PER_UNIT = 1.0


def load_depth_mm(path):
    """Return (depth_mm float32 with NaN for invalid, valid bool mask)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16, got {raw.dtype}. "
                         "The file was probably re-encoded and depth is destroyed.")
    valid = raw != INVALID
    d = raw.astype(np.float32) * MM_PER_UNIT
    d[~valid] = np.nan
    return d, valid


def load_intrinsics(session_dir):
    """fx, fy, cx, cy from the rectified left camera (zero distortion)."""
    c = json.loads((Path(session_dir) / "meta" / "calibration.json").read_text())
    L = c["left"]
    return L["fx"], L["fy"], L["cx"], L["cy"]


def backproject(u, v, z_mm, K):
    """Pixel + depth -> 3D point in the camera frame, millimetres."""
    fx, fy, cx, cy = K
    return np.array([z_mm * (u - cx) / fx, z_mm * (v - cy) / fy, z_mm], np.float64)


def robust_point_depth(depth_mm, valid, u, v, mask=None, radius_px=7,
                       min_valid=8, max_spread_mm=40.0):
    """Depth at a predicted LEP, read as a neighbourhood rather than one pixel.

    A single stereo pixel at the growth point may be soil, an occluding leaf, or
    an invalid match. This samples a disc around (u, v), optionally intersected
    with the weed mask, rejects outliers, and abstains rather than guessing.

    Returns (z_mm, stats). z_mm is None when the estimate is not trustworthy -
    the caller should then fall back to bed-plane intersection or mark the weed
    not targetable.
    """
    h, w = depth_mm.shape
    ui, vi = int(round(u)), int(round(v))
    y0, y1 = max(0, vi - radius_px), min(h, vi + radius_px + 1)
    x0, x1 = max(0, ui - radius_px), min(w, ui + radius_px + 1)

    patch = depth_mm[y0:y1, x0:x1]
    ok = valid[y0:y1, x0:x1].copy()
    if mask is not None:
        ok &= mask[y0:y1, x0:x1].astype(bool)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    ok &= ((yy - vi) ** 2 + (xx - ui) ** 2) <= radius_px ** 2

    vals = patch[ok]
    stats = {"n_valid": int(vals.size),
             "valid_fraction": float(ok.sum() / max(1, ok.size))}
    if vals.size < min_valid:
        stats["reason"] = "insufficient_valid_depth"
        return None, stats

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    inl = vals[np.abs(vals - med) <= max(3.0 * mad, 5.0)]
    stats.update({"median_mm": med, "mad_mm": mad,
                  "n_inliers": int(inl.size),
                  "spread_mm": float(inl.max() - inl.min()) if inl.size else None})
    if inl.size < min_valid:
        stats["reason"] = "outlier_dominated"
        return None, stats
    if stats["spread_mm"] is not None and stats["spread_mm"] > max_spread_mm:
        # Likely straddling a leaf edge and the soil behind it.
        stats["reason"] = "depth_discontinuity"
        return None, stats
    return float(np.median(inl)), stats
