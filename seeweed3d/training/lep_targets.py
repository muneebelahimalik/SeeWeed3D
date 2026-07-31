#!/usr/bin/env python3
"""
SeeWeed3D - LEP heatmap targets, soft-argmax decoding, and joint augmentation.

Pure numpy: targets and augmentation are geometry, not learning, so they stay
testable without torch and are shared unchanged between training and evaluation.

THREE THINGS THIS MODULE IS CAREFUL ABOUT
-----------------------------------------
1. SUB-PIXEL TRUTH IS PRESERVED. The Gaussian is rendered about the exact float
   coordinate and the float coordinate is returned alongside it. Rounding the
   target to the heatmap grid would cap achievable accuracy at the stride
   (4 px here) before training even starts.

2. MISSING IS NOT ZERO. A `not_visible` weed has no LEP, and its heatmap loss is
   MASKED OUT rather than supervised toward an all-zero map. Supervising zeros
   would teach "this plant has no growth point", which is false - the point
   exists, the annotator simply could not see it.

3. AUGMENTATION MOVES EVERYTHING TOGETHER. RGB, the owning mask, depth and the
   LEP share one transform. Mosaic and MixUp are deliberately unsupported for
   ROI LEP training: both composite pixels from different plants into one image,
   which destroys the biological ownership the whole task depends on.
"""

from __future__ import annotations

import cv2
import numpy as np

# Augmentations that are unsafe for this task, named so the refusal is explicit
# rather than an omission someone later "fixes".
FORBIDDEN_AUGMENTATIONS = ("mosaic", "mixup", "cutmix", "copy_paste")


class AugmentationError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Heatmap targets
# --------------------------------------------------------------------------- #
def heatmap_size(out_size, stride):
    return int(out_size) // int(stride)


def resolve_sigma(cfg, plant_radius_px=None):
    """Gaussian sigma in HEATMAP pixels.

    Optionally scaled with plant size so a cotyledon and a large rosette are
    supervised at comparable RELATIVE precision - a fixed sigma is a loose
    target on a seedling and an over-tight one on a big rosette."""
    sigma = float(cfg.sigma_px)
    if cfg.sigma_scale_with_plant and plant_radius_px:
        sigma = float(cfg.sigma_scale_with_plant) * float(plant_radius_px) / cfg.stride
    return float(np.clip(sigma, cfg.sigma_min_px, cfg.sigma_max_px))


def make_heatmap(uv_roi, out_size, cfg, plant_radius_px=None):
    """Gaussian target for one LEP.

    uv_roi is in ROI pixels; the peak is placed at the corresponding sub-pixel
    heatmap location. Returns (heatmap float32 HxW peaking at 1.0, uv_hm).
    Returns an all-zero map when the point falls outside the ROI - the caller
    masks the loss in that case rather than supervising a peak that is not
    there."""
    hm_size = heatmap_size(out_size, cfg.stride)
    hm = np.zeros((hm_size, hm_size), np.float32)
    # Pixel-centre convention: ROI pixel centre (i+0.5) maps to heatmap centre.
    cx = (float(uv_roi[0]) + 0.5) / cfg.stride - 0.5
    cy = (float(uv_roi[1]) + 0.5) / cfg.stride - 0.5
    if not (-1.0 <= cx <= hm_size and -1.0 <= cy <= hm_size):
        return hm, (cx, cy)

    sigma = resolve_sigma(cfg, plant_radius_px)
    rad = max(1, int(np.ceil(3.0 * sigma)))
    x0, x1 = int(np.floor(cx)) - rad, int(np.ceil(cx)) + rad + 1
    y0, y1 = int(np.floor(cy)) - rad, int(np.ceil(cy)) + rad + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(hm_size, x1), min(hm_size, y1)
    if x1 <= x0 or y1 <= y0:
        return hm, (cx, cy)

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    hm[y0:y1, x0:x1] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) /
                              (2.0 * sigma * sigma))
    return hm, (cx, cy)


def heatmap_to_roi(uv_hm, stride):
    """Inverse of the ROI->heatmap mapping used by make_heatmap()."""
    return ((float(uv_hm[0]) + 0.5) * stride - 0.5,
            (float(uv_hm[1]) + 0.5) * stride - 0.5)


def target_weight(visibility, cfg):
    """Supervision weight for one sample's heatmap/coordinate loss.

    0 for `not_visible`: no LEP exists to supervise, and inventing one is worse
    than skipping the sample. Reduced (not zero) for `partially_occluded_
    inferable`, whose ground truth is real but less certain."""
    v = str(visibility)
    if v == "not_visible":
        return 0.0
    if v == "partially_occluded_inferable":
        return float(cfg.partial_visibility_weight)
    return 1.0


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def soft_argmax(heatmap, beta=1.0, eps=1e-9):
    """Expected coordinate under the (temperature-scaled) heatmap.

    Sub-pixel by construction, and differentiable, unlike an integer argmax
    whose error is floor-limited by the stride. Returns (x, y) in heatmap
    pixels."""
    hm = np.asarray(heatmap, np.float64)
    if beta != 1.0:
        hm = np.exp(beta * (hm - hm.max()))
    hm = np.clip(hm, 0.0, None)
    total = hm.sum()
    if total <= eps:
        h, w = hm.shape
        return ((w - 1) / 2.0, (h - 1) / 2.0)
    ys, xs = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    return (float((hm * xs).sum() / total), float((hm * ys).sum() / total))


def heatmap_uncertainty(heatmap, eps=1e-9):
    """Spatial covariance of the heatmap, as the estimate's own uncertainty.

    A confident prediction is a tight unimodal blob; an ambiguous one is broad
    or bimodal, and both inflate the covariance. sigma_px (sqrt of the larger
    eigenvalue) is therefore a usable abstention signal that costs nothing extra
    to compute - and it is reported in HEATMAP pixels, so callers must scale by
    the stride to get ROI pixels."""
    hm = np.clip(np.asarray(heatmap, np.float64), 0.0, None)
    total = hm.sum()
    if total <= eps:
        return {"cov": [[0.0, 0.0], [0.0, 0.0]], "sigma_px": 0.0, "peak": 0.0}
    ys, xs = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    mx = float((hm * xs).sum() / total)
    my = float((hm * ys).sum() / total)
    vxx = float((hm * (xs - mx) ** 2).sum() / total)
    vyy = float((hm * (ys - my) ** 2).sum() / total)
    vxy = float((hm * (xs - mx) * (ys - my)).sum() / total)
    cov = np.array([[vxx, vxy], [vxy, vyy]], np.float64)
    evals = np.linalg.eigvalsh(cov)
    return {"cov": cov.tolist(),
            "sigma_px": float(np.sqrt(max(0.0, float(evals.max())))),
            "peak": float(hm.max())}


def decode_lep(heatmap, tf, cfg, beta=1.0):
    """Heatmap -> full-frame LEP with uncertainty, in one place.

    Keeping decode in one function means training-time and deployment-time
    decoding cannot diverge - a classic source of an offset that only appears
    in the field."""
    uv_hm = soft_argmax(heatmap, beta=beta)
    unc = heatmap_uncertainty(heatmap)
    uv_roi = heatmap_to_roi(uv_hm, cfg.stride)
    uv_full = tf.to_full(*uv_roi) if tf is not None else uv_roi
    sigma_roi = unc["sigma_px"] * cfg.stride
    return {"uv_full": (float(uv_full[0]), float(uv_full[1])),
            "uv_roi": (float(uv_roi[0]), float(uv_roi[1])),
            "uv_heatmap": (float(uv_hm[0]), float(uv_hm[1])),
            "peak": unc["peak"],
            "sigma_roi_px": float(sigma_roi),
            # Back to full-frame pixels: the ROI was scaled by tf.scale.
            "sigma_px": float(sigma_roi / tf.scale) if tf is not None
            else float(sigma_roi),
            "covariance": unc["cov"]}


# --------------------------------------------------------------------------- #
# Joint augmentation
# --------------------------------------------------------------------------- #
class JointAugment:
    """Geometric + photometric augmentation applied identically to every layer.

    Only augmentations that preserve a single plant's identity are offered.
    Requesting a compositing augmentation raises rather than silently ignoring
    it, because a silently-dropped request looks like it was applied."""

    def __init__(self, hflip=0.5, vflip=0.5, rot90=True, max_rotate_deg=15.0,
                 scale_jitter=0.1, brightness=0.2, seed=0, forbid=()):
        for name in forbid:
            if str(name).lower() in FORBIDDEN_AUGMENTATIONS:
                raise AugmentationError(
                    f"'{name}' composites pixels from different plants into one "
                    f"ROI, which destroys the mask/LEP ownership this task is "
                    f"built on. It is not supported for LEP training.")
        self.hflip, self.vflip, self.rot90 = hflip, vflip, rot90
        self.max_rotate_deg = max_rotate_deg
        self.scale_jitter = scale_jitter
        self.brightness = brightness
        self.rng = np.random.default_rng(seed)

    def __call__(self, rgb, mask, uv, depth_mm=None, geom=None):
        """Returns (rgb, mask, uv, depth_mm, geom) with uv moved consistently."""
        h, w = mask.shape[:2]
        x, y = float(uv[0]), float(uv[1])

        if self.rng.random() < self.hflip:
            rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()
            if depth_mm is not None:
                depth_mm = depth_mm[:, ::-1].copy()
            if geom is not None:
                geom = geom[:, :, ::-1].copy()
            x = (w - 1) - x
        if self.rng.random() < self.vflip:
            rgb, mask = rgb[::-1].copy(), mask[::-1].copy()
            if depth_mm is not None:
                depth_mm = depth_mm[::-1].copy()
            if geom is not None:
                geom = geom[:, ::-1].copy()
            y = (h - 1) - y
        if self.rot90 and self.rng.random() < 0.5:
            k = int(self.rng.integers(1, 4))
            for _ in range(k):
                rgb = np.rot90(rgb, 1).copy()
                mask = np.rot90(mask, 1).copy()
                if depth_mm is not None:
                    depth_mm = np.rot90(depth_mm, 1).copy()
                if geom is not None:
                    geom = np.stack([np.rot90(g, 1) for g in geom]).copy()
                # np.rot90 is counter-clockwise: (x, y) -> (y, W-1-x).
                x, y = y, (w - 1) - x
                w, h = h, w

        if self.max_rotate_deg > 0 or self.scale_jitter > 0:
            ang = float(self.rng.uniform(-self.max_rotate_deg, self.max_rotate_deg))
            sc = 1.0 + float(self.rng.uniform(-self.scale_jitter, self.scale_jitter))
            M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), ang, sc)
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                                  flags=cv2.INTER_NEAREST).astype(bool)
            if depth_mm is not None:
                d = np.where(np.isfinite(depth_mm), depth_mm, 0.0).astype(np.float32)
                d = cv2.warpAffine(d, M, (w, h), flags=cv2.INTER_NEAREST)
                depth_mm = np.where(d > 0, d, np.nan).astype(np.float32)
            if geom is not None:
                geom = np.stack([
                    cv2.warpAffine(g, M, (w, h),
                                   flags=cv2.INTER_NEAREST if i != 1
                                   else cv2.INTER_LINEAR)
                    for i, g in enumerate(geom)])
            p = M @ np.array([x, y, 1.0])
            x, y = float(p[0]), float(p[1])

        if self.brightness > 0:
            f = 1.0 + float(self.rng.uniform(-self.brightness, self.brightness))
            rgb = np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)

        return rgb, mask, (x, y), depth_mm, geom


# --------------------------------------------------------------------------- #
# Depth degradation (training only)
# --------------------------------------------------------------------------- #
def simulate_depth_degradation(depth_mm, cfg, rng):
    """Make training depth look like field depth.

    Stereo depth in a crop canopy is not the clean map a synthetic pipeline
    produces: it has holes on specular leaves, range-dependent noise,
    quantisation, and it sometimes fails outright. A model trained on clean
    depth silently degrades in the field, so the degradation is applied at
    training time and full dropout is included so the RGB path must remain
    self-sufficient.

    Returns (depth_or_None, dropped_bool). None means depth is unavailable for
    this sample, which is the case the RGB-only ablation must survive."""
    if depth_mm is None:
        return None, True
    if rng.random() < cfg.depth_dropout_p:
        return None, True

    d = depth_mm.astype(np.float32).copy()
    finite = np.isfinite(d) & (np.nan_to_num(d) > 0)

    if cfg.hole_dropout_p > 0:
        holes = rng.random(d.shape) < cfg.hole_dropout_p
        d[holes] = np.nan
        finite &= ~holes
    if cfg.noise_mm_per_m > 0:
        metres = np.where(finite, d, 0.0) / 1000.0
        d = np.where(finite,
                     d + rng.normal(0.0, 1.0, d.shape).astype(np.float32)
                     * cfg.noise_mm_per_m * np.maximum(metres, 0.1), d)
    if cfg.quantisation_mm > 0:
        d = np.where(finite,
                     np.round(d / cfg.quantisation_mm) * cfg.quantisation_mm, d)
    d[~finite] = np.nan
    return d.astype(np.float32), False
