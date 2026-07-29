#!/usr/bin/env python3
"""
SeeWeed3D - shared vegetation / image preprocessing.

Used by both the onion and weed prelabelers so the two cannot drift apart.
Functions take explicit parameters (not a config dict) so they are reusable and
directly testable; each caller passes its own CONFIG values in.
"""

import cv2
import numpy as np


def excess_green(bgr):
    """Excess-Green index (ExG), the standard vegetation index for RGB field
    imagery. Positive on plant tissue, ~0 or negative on soil."""
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    s = b + g + r + 1e-6
    return 2 * (g / s) - (r / s) - (b / s)


def white_balance(bgr, cast_ratio=1.15):
    """Gray-world white balance, applied only when the frame is actually
    colour-cast (channel means diverge past cast_ratio).

    Some ZED frames have a severe green white-balance error - the whole frame
    reads green - which would otherwise be unusable. Neutral frames are returned
    unchanged, so this is safe to leave on."""
    means = bgr.reshape(-1, 3).mean(axis=0) + 1e-6
    if float(means.max() / means.min()) < cast_ratio:
        return bgr
    gray = float(means.mean())
    return np.clip(bgr.astype(np.float32) * (gray / means), 0, 255).astype(np.uint8)


def remove_small(mask, min_px):
    """Drop connected components below min_px."""
    if min_px <= 0 or not mask.any():
        return mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[lbl == i] = True
    return keep


def vegetation_mask(bgr, exg_threshold=0.05, min_saturation=40, morph_kernel=3,
                    min_component_px=80):
    """Vegetation mask: ExG gated by green dominance and saturation.

    ExG alone masks bare soil whenever a frame has a green colour-cast (soil then
    reads as weak green). Requiring green to be the dominant channel AND the
    pixel to be saturated rejects that, because colour-cast soil is desaturated
    while real foliage is saturated green."""
    veg = excess_green(bgr) > exg_threshold
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    veg &= (g >= r) & (g >= b)
    veg &= cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1] >= min_saturation
    veg = veg.astype(np.uint8)
    if morph_kernel > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel,) * 2)
        veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, ker)
        veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, ker)
    return remove_small(veg.astype(bool), min_component_px)


def vegetation_score(bgr, exg_threshold=0.05, min_saturation=40, softness=0.04):
    """Soft plant likelihood in [0, 1], the continuous counterpart of
    vegetation_mask().

    The binary mask is the right tool for deciding what is a plant; for deciding
    exactly WHERE a plant boundary falls, a hard threshold throws away the very
    gradient that locates the edge. This keeps that gradient: a smooth ramp
    across the ExG threshold, gated the same way by green dominance and
    saturation so a colour-cast soil cannot score as plant."""
    exg = excess_green(bgr)
    # Logistic ramp centred on the threshold: 0.5 at the threshold, saturating
    # within a few hundredths of ExG either side.
    score = 1.0 / (1.0 + np.exp(-(exg - exg_threshold) / max(1e-6, softness)))
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    score *= ((g >= r) & (g >= b)).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    score *= np.clip(sat / max(1.0, float(min_saturation)), 0.0, 1.0)
    return score.astype(np.float32)


def component_boxes(mask, min_area_px, pad_px=0, max_boxes=None):
    """Bounding boxes [x1,y1,x2,y2] of mask components >= min_area_px, largest
    first. Used to prompt SAM 3 with real plant exemplars."""
    if not mask.any():
        return []
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append((area, [max(0, x - pad_px), max(0, y - pad_px),
                           min(w, x + bw + pad_px), min(h, y + bh + pad_px)]))
    out.sort(key=lambda t: -t[0])
    boxes = [b for _, b in out]
    return boxes[:max_boxes] if max_boxes else boxes
