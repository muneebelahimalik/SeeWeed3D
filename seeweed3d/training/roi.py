#!/usr/bin/env python3
"""
SeeWeed3D - ROI extraction with an exactly invertible coordinate transform.

Stage B sees a weed as a fixed-size square crop. Every pixel-space quantity -
the RGB crop, the owning instance mask, depth, and the LEP point - must undergo
the SAME transform, and the predicted point must map back to full-frame
coordinates exactly. A silent half-pixel error here becomes a systematic aiming
bias that no amount of training corrects, so the transform is a small explicit
object with a tested round-trip rather than arithmetic inlined at each call.

Aspect ratio is preserved by letterboxing. Stretching a crop to a square changes
the apparent angles between leaves, and phyllotactic symmetry about the meristem
is precisely the structure the LEP head is meant to read.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiTransform:
    """full_frame -> ROI: scale about a crop origin, then a letterbox offset.

        u_roi = (u_full - x0) * scale + pad_x
        v_roi = (v_full - y0) * scale + pad_y

    Frozen because a transform that changed after the crop was taken would
    silently invalidate every coordinate derived from it."""

    x0: float
    y0: float
    scale: float
    pad_x: float
    pad_y: float
    out_size: int
    src_w: float
    src_h: float

    def to_roi(self, u, v):
        return ((u - self.x0) * self.scale + self.pad_x,
                (v - self.y0) * self.scale + self.pad_y)

    def to_full(self, u, v):
        return ((u - self.pad_x) / self.scale + self.x0,
                (v - self.pad_y) / self.scale + self.y0)

    def to_roi_array(self, pts):
        a = np.asarray(pts, np.float64).reshape(-1, 2)
        out = np.empty_like(a)
        out[:, 0] = (a[:, 0] - self.x0) * self.scale + self.pad_x
        out[:, 1] = (a[:, 1] - self.y0) * self.scale + self.pad_y
        return out

    def to_dict(self):
        return {"x0": self.x0, "y0": self.y0, "scale": self.scale,
                "pad_x": self.pad_x, "pad_y": self.pad_y,
                "out_size": self.out_size}


def expand_box(bbox, expand_ratio, frame_w, frame_h, min_box_px=16):
    """Grow a bbox about its centre and clamp it to the frame.

    Returns (x0, y0, w, h) as floats. The expansion brings in the soil ring that
    makes local height meaningful and the context that says which plant a leaf
    belongs to."""
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h, float(min_box_px)) * float(expand_ratio)
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    # Clamp, keeping the square inside the frame where possible.
    x0 = max(0.0, min(x0, frame_w - 1.0))
    y0 = max(0.0, min(y0, frame_h - 1.0))
    w1 = min(side, frame_w - x0)
    h1 = min(side, frame_h - y0)
    return x0, y0, max(1.0, w1), max(1.0, h1)


def make_transform(bbox, out_size, expand_ratio, frame_w, frame_h,
                   min_box_px=16):
    """RoiTransform for one instance."""
    x0, y0, w, h = expand_box(bbox, expand_ratio, frame_w, frame_h, min_box_px)
    scale = float(out_size) / max(w, h)
    pad_x = (out_size - w * scale) / 2.0
    pad_y = (out_size - h * scale) / 2.0
    return RoiTransform(x0=x0, y0=y0, scale=scale, pad_x=pad_x, pad_y=pad_y,
                        out_size=int(out_size), src_w=w, src_h=h)


def _warp(src, tf, interp, border_value=0):
    """Apply a RoiTransform with a single affine warp.

    One warpAffine rather than crop-then-resize: it is the same arithmetic the
    transform object describes, so the image and the coordinates cannot drift
    apart, and it handles a crop that runs off the frame edge without a
    separate padding step."""
    M = np.array([[tf.scale, 0.0, -tf.x0 * tf.scale + tf.pad_x],
                  [0.0, tf.scale, -tf.y0 * tf.scale + tf.pad_y]], np.float64)
    return cv2.warpAffine(src, M, (tf.out_size, tf.out_size), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)


def extract_roi(bgr, mask, tf, depth_mm=None, pad_value=0):
    """Crop RGB, the owning mask and (optionally) depth under one transform.

    Returns a dict. The mask uses NEAREST so it stays binary; depth uses NEAREST
    too, because interpolating across a depth discontinuity invents a distance
    that lies between a leaf and the soil behind it - a value no surface
    occupies."""
    roi = {"rgb": _warp(bgr, tf, cv2.INTER_LINEAR, pad_value)}
    roi["mask"] = _warp(mask.astype(np.uint8), tf, cv2.INTER_NEAREST, 0).astype(bool)
    if depth_mm is not None:
        d = np.where(np.isfinite(depth_mm), depth_mm, 0.0).astype(np.float32)
        warped = _warp(d, tf, cv2.INTER_NEAREST, 0)
        roi["depth_mm"] = np.where(warped > 0, warped, np.nan).astype(np.float32)
        roi["depth_valid"] = warped > 0
    else:
        roi["depth_mm"] = None
        roi["depth_valid"] = np.zeros((tf.out_size, tf.out_size), bool)
    return roi


def local_height_map(depth_mm, inst_mask, cfg):
    """Height above the LOCAL soil reference, normalised, plus its validity.

    WHY NOT RAW DEPTH: absolute depth encodes camera mount height, so a network
    fed raw depth learns "a plant is 900 mm away" and fails the moment the rig
    is raised. Height above the soil immediately around THIS plant is a physical
    property of the plant, transferable across cameras and mount heights.

    The reference is the median depth of an annulus outside the instance, which
    is soil in a top-down view. When too little of that ring is valid, no
    reference exists and the channel is reported invalid rather than guessed -
    an invented reference silently biases every height on the plant.

    Returns (height_norm float32 in [0,1], valid bool). height_norm is 0 where
    invalid, and the validity channel is what tells the network so."""
    h, w = inst_mask.shape
    out = np.zeros((h, w), np.float32)
    if depth_mm is None:
        return out, np.zeros((h, w), bool)

    finite = np.isfinite(depth_mm) & (np.nan_to_num(depth_mm) > 0)
    if not finite.any() or not inst_mask.any():
        return out, np.zeros((h, w), bool)

    d8 = inst_mask.astype(np.uint8)
    # Annulus radii scale with the instance, so a seedling and a large rosette
    # both sample soil just beyond their own canopy.
    area = float(inst_mask.sum())
    r = max(2.0, np.sqrt(area / np.pi))
    k_in = max(1, int(round(r * (cfg.soil_ring_inner_ratio - 1.0))))
    k_out = max(k_in + 1, int(round(r * (cfg.soil_ring_outer_ratio - 1.0))))
    inner = cv2.dilate(d8, np.ones((2 * k_in + 1,) * 2, np.uint8)).astype(bool)
    outer = cv2.dilate(d8, np.ones((2 * k_out + 1,) * 2, np.uint8)).astype(bool)
    ring = outer & ~inner & finite

    if int(ring.sum()) < cfg.soil_ring_min_valid_px:
        return out, np.zeros((h, w), bool)

    reference = float(np.median(depth_mm[ring]))
    # Camera looks down: nearer (smaller depth) is higher above the soil.
    height = reference - np.where(finite, depth_mm, reference)
    height = np.clip(height, cfg.height_min_mm, cfg.height_max_mm)
    span = max(1e-6, cfg.height_max_mm - cfg.height_min_mm)
    out = ((height - cfg.height_min_mm) / span).astype(np.float32)
    out[~finite] = 0.0
    return out, finite


def build_geometry_channels(roi, cfg):
    """Stack the geometry branch input: [mask, height_norm, depth_valid].

    Three channels, each independently meaningful:
      mask         - which pixels belong to THIS plant (ownership)
      height_norm  - camera-transferable elevation above local soil
      depth_valid  - where the height channel may be believed at all

    Keeping validity separate is what lets the network distinguish "flat ground"
    from "no measurement", which a single channel with zeros cannot express."""
    mask = roi["mask"].astype(np.float32)
    if cfg.use_depth and roi.get("depth_mm") is not None:
        height, valid = local_height_map(roi["depth_mm"], roi["mask"], cfg)
    else:
        height = np.zeros_like(mask)
        valid = np.zeros(mask.shape, bool)
    return np.stack([mask, height, valid.astype(np.float32)], 0)
