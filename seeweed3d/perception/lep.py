#!/usr/bin/env python3
"""
SeeWeed3D - Leaf Emergence Point (LEP) estimation
=================================================
Biologically-grounded estimation of the LEP / apical meristem of a weed plant:
the point the laser must hit to cause irreversible growth failure.

WHY A MULTI-EVIDENCE MODEL
--------------------------
A single geometric trick (e.g. "deepest interior point of the mask") often lands
near the right place, but it cannot be defended as *the leaf emergence point* -
it is a property of the silhouette, not of the plant. This module instead treats
LEP estimation as evidence fusion over channels that are each independently
motivated by rosette/tiller architecture, so the estimate can be justified, and
so disagreement between channels becomes a measurable uncertainty rather than a
silent error.

THE EVIDENCE CHANNELS AND THEIR BIOLOGY
---------------------------------------
1. PetioleConvergence  - In a rosette, every leaf radiates from the shoot apical
   meristem, so the petiole axes converge on it. Skeletonising the plant mask and
   locating the junction where branches meet therefore points at the meristem.
   This is the strongest purely structural evidence, and it is what a botanist
   would use by eye.

2. RadialIsotropy      - Rosette phyllotaxy arranges leaves at regular angular
   intervals about the meristem. Seen FROM the meristem, plant tissue is
   distributed almost uniformly in angle; seen from any off-centre point it is
   not. Scoring angular uniformity therefore peaks at the growth point. This is
   independent of scale and of how many leaves there are.

3. YoungTissue         - The defining feature of the LEP is the youngest emerging
   leaf tissue. Young expanding leaves have not yet accumulated full chlorophyll,
   so they reflect more light and sit closer to yellow-green than mature leaves.
   A chromatic "youth" score therefore marks the active growth point directly,
   and is the only channel that keys on the actual biology rather than geometry.

4. CanopyHeight        - Leaves stack where they emerge, and the growing tip is
   elevated relative to the surrounding expanded leaves and soil. Where stereo
   depth is available, height above the local soil plane is an independent,
   purely physical vote for the crown. Optional: contributes only when enough
   valid depth exists on the plant.

5. MedialAxis          - The meristem is maximally interior to the leaf whorl, so
   the distance transform peaks near it. Weakest evidence alone (it is a
   silhouette property) but cheap, stable, and a good regulariser.

WHAT THE ESTIMATOR RETURNS
--------------------------
A fused sub-pixel point, a per-channel breakdown (each channel's own argmax), an
inter-channel agreement distance, a 2x2 spatial covariance, a calibrated
confidence, and a visibility verdict that can ABSTAIN. Abstention matters: a
confidently wrong LEP aims a 60 W laser at the wrong tissue, whereas an abstained
one is simply not treated.

The per-channel breakdown is deliberately preserved so an ablation ("which
evidence actually carries the estimate?") can be reported from stored results
with no re-processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from common.ontology import ROSETTE_CLASSES  # noqa: E402

LEP_METHOD_VERSION = "seeweed3d/lep/1.0"


# --------------------------------------------------------------------------- #
# Inputs and outputs
# --------------------------------------------------------------------------- #
@dataclass
class PlantContext:
    """One plant instance, cropped to its bounding box.

    mask     : bool HxW, the instance silhouette
    bgr      : uint8 HxWx3, the same crop of the (white-balanced) image
    depth_mm : optional float HxW, metric depth; NaN or <=0 where invalid
    origin   : (x0, y0) of the crop in the full frame, so results map back
    class_name : weed class, used to pick channel weights (grass vs rosette)
    """
    mask: np.ndarray
    bgr: np.ndarray
    depth_mm: Optional[np.ndarray] = None
    origin: tuple = (0, 0)
    class_name: str = "other_weed"

    @property
    def has_depth(self) -> bool:
        if self.depth_mm is None:
            return False
        d = self.depth_mm
        return bool(np.isfinite(d).any() and (np.nan_to_num(d) > 0).any())


@dataclass
class LEPResult:
    """Estimate plus everything needed to defend or audit it."""
    uv: tuple                      # (x, y) in FULL-frame pixels, sub-pixel
    uv_local: tuple                # (x, y) within the crop
    confidence: float              # [0, 1]
    visibility: str                # visible | partially_occluded_inferable | not_visible
    agreement_px: float            # spread of the per-channel argmaxes
    covariance: list               # 2x2 spatial covariance of the fused evidence
    sigma_px: float                # sqrt of the larger covariance eigenvalue
    channels: dict = field(default_factory=dict)   # name -> {uv, weight, peak}
    method_version: str = LEP_METHOD_VERSION

    def as_row(self, prefix="lep"):
        """Flat dict for CSV export."""
        row = {f"{prefix}_x": round(self.uv[0], 2), f"{prefix}_y": round(self.uv[1], 2),
               f"{prefix}_confidence": round(self.confidence, 4),
               f"{prefix}_visibility": self.visibility,
               f"{prefix}_agreement_px": round(self.agreement_px, 2),
               f"{prefix}_sigma_px": round(self.sigma_px, 2),
               f"{prefix}_method": self.method_version}
        for name, c in self.channels.items():
            row[f"{prefix}_{name}_x"] = round(c["uv"][0], 1)
            row[f"{prefix}_{name}_y"] = round(c["uv"][1], 1)
        return row


# --------------------------------------------------------------------------- #
# Evidence channels
# --------------------------------------------------------------------------- #
def _norm(score, mask):
    """Normalise a score map to [0, 1] inside the mask, 0 outside."""
    s = np.zeros_like(score, dtype=np.float32)
    m = mask & np.isfinite(score)
    if not m.any():
        return s
    v = score[m]
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        s[m] = 1.0
        return s
    s[m] = (score[m] - lo) / (hi - lo)
    return s


class LEPEvidence(ABC):
    """One source of evidence for where the leaf emergence point is."""
    name: str = "evidence"
    rationale: str = ""

    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    @abstractmethod
    def score(self, ctx: PlantContext) -> np.ndarray:
        """Return a float32 map, high where the LEP is likely. Masked+normalised."""

    def available(self, ctx: PlantContext) -> bool:
        return True

    def describe(self):
        return {"name": self.name, "weight": self.weight, "rationale": self.rationale}


class MedialAxisEvidence(LEPEvidence):
    name = "medial_axis"
    rationale = ("The meristem is maximally interior to the leaf whorl, so the "
                 "distance transform of the plant silhouette peaks near it. A "
                 "silhouette property rather than a biological one, so it is "
                 "used as a stable regulariser rather than primary evidence.")

    def score(self, ctx):
        dt = cv2.distanceTransform(ctx.mask.astype(np.uint8), cv2.DIST_L2, 5)
        return _norm(dt, ctx.mask)


def zhang_suen_thin(mask, max_iter=60):
    """Zhang-Suen thinning to a 1-px skeleton. Implemented here so the module has
    no opencv-contrib dependency, and stays deterministic across machines."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P2 = np.roll(img, 1, 0)
            P6 = np.roll(img, -1, 0)
            P4 = np.roll(img, -1, 1)
            P8 = np.roll(img, 1, 1)
            P3 = np.roll(P2, -1, 1)
            P9 = np.roll(P2, 1, 1)
            P5 = np.roll(P6, -1, 1)
            P7 = np.roll(P6, 1, 1)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(seq)
            A = sum(((a == 0) & (b == 1)).astype(np.uint8)
                    for a, b in zip(seq, seq[1:] + seq[:1]))
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


class PetioleConvergenceEvidence(LEPEvidence):
    name = "petiole_convergence"
    rationale = ("Every leaf of a rosette radiates from the shoot apical "
                 "meristem, so the petiole axes converge on it. Skeletonising "
                 "the plant and locating where skeleton branches meet therefore "
                 "points directly at the meristem - the same cue a botanist uses "
                 "by eye, and the strongest structural evidence available.")

    def __init__(self, weight=1.0, blur_frac=0.12):
        super().__init__(weight)
        self.blur_frac = blur_frac

    def score(self, ctx):
        skel = zhang_suen_thin(ctx.mask)
        if not skel.any():
            return np.zeros(ctx.mask.shape, np.float32)
        s = skel.astype(np.uint8)
        # Branch degree: skeleton pixels whose neighbourhood contains >=3 other
        # skeleton pixels are junctions where leaf axes meet.
        k = np.ones((3, 3), np.uint8)
        deg = cv2.filter2D(s, -1, k, borderType=cv2.BORDER_CONSTANT) - s
        junction = np.zeros(ctx.mask.shape, np.float32)
        junction[skel] = np.clip(deg[skel].astype(np.float32) - 2.0, 0, None)
        # Skeleton pixels also carry their inscribed radius: the convergence
        # point of a rosette is thick (many petioles overlapping), leaf tips are
        # thin. Weighting by radius suppresses spurious tip junctions.
        dt = cv2.distanceTransform(ctx.mask.astype(np.uint8), cv2.DIST_L2, 5)
        junction *= (dt / (dt.max() + 1e-6))
        sigma = max(2.0, self.blur_frac * np.sqrt(ctx.mask.sum()))
        blurred = cv2.GaussianBlur(junction, (0, 0), sigma)
        return _norm(blurred, ctx.mask)


class RadialIsotropyEvidence(LEPEvidence):
    name = "radial_isotropy"
    rationale = ("Rosette phyllotaxy places leaves at regular angular intervals "
                 "about the meristem, so seen FROM the growth point the plant's "
                 "tissue is almost uniform in angle, while from any off-centre "
                 "point it is not. Angular uniformity therefore peaks at the "
                 "meristem, independently of plant scale or leaf count.")

    def __init__(self, weight=1.0, n_bins=16, grid_stride=4, max_samples=600):
        super().__init__(weight)
        self.n_bins, self.grid_stride, self.max_samples = n_bins, grid_stride, max_samples

    def score(self, ctx):
        mask = ctx.mask
        ys, xs = np.nonzero(mask)
        if xs.size < 8:
            return np.zeros(mask.shape, np.float32)
        rng = np.random.default_rng(0)                       # deterministic
        if xs.size > self.max_samples:
            idx = rng.choice(xs.size, self.max_samples, replace=False)
            sx, sy = xs[idx], ys[idx]
        else:
            sx, sy = xs, ys

        h, w = mask.shape
        st = max(1, self.grid_stride)
        out = np.zeros((h, w), np.float32)
        # Only evaluate candidates well inside the plant: the meristem cannot be
        # on the silhouette boundary.
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        cand = mask & (dt > 0.15 * dt.max())
        cy_idx, cx_idx = np.nonzero(cand[::st, ::st])
        if cy_idx.size == 0:
            return np.zeros(mask.shape, np.float32)
        best = np.zeros(cy_idx.size, np.float32)
        for i, (gy, gx) in enumerate(zip(cy_idx, cx_idx)):
            py, px = gy * st, gx * st
            ang = np.arctan2(sy - py, sx - px)
            hist, _ = np.histogram(ang, bins=self.n_bins, range=(-np.pi, np.pi))
            p = hist.astype(np.float32)
            tot = p.sum()
            if tot <= 0:
                continue
            p /= tot
            nz = p > 0
            # Normalised angular entropy: 1.0 = perfectly isotropic surroundings.
            best[i] = float(-(p[nz] * np.log(p[nz])).sum() / np.log(self.n_bins))
        out[cy_idx * st, cx_idx * st] = best
        if st > 1:
            out = cv2.dilate(out, np.ones((st * 2 + 1,) * 2, np.uint8))
        out = cv2.GaussianBlur(out, (0, 0), max(1.5, st))
        return _norm(out, mask)


class YoungTissueEvidence(LEPEvidence):
    name = "young_tissue"
    rationale = ("The LEP is by definition the centre of the youngest emerging "
                 "leaf tissue. Young expanding leaves have not yet accumulated "
                 "full chlorophyll, so they reflect more light and sit closer to "
                 "yellow-green than mature leaves. This is the only channel that "
                 "keys on plant physiology rather than geometry, so it is what "
                 "makes the estimate a LEAF EMERGENCE point rather than a "
                 "shape centre.")

    def __init__(self, weight=1.0, clip_pct=99.0):
        super().__init__(weight)
        self.clip_pct = clip_pct

    def score(self, ctx):
        mask = ctx.mask
        if not mask.any():
            return np.zeros(mask.shape, np.float32)
        lab = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, b = lab[:, :, 0], lab[:, :, 2]          # lightness, blue<->yellow axis
        # Higher L (more reflectance) and higher b* (toward yellow) both indicate
        # less chlorophyll, i.e. younger tissue.
        youth = 0.5 * _norm(L, mask) + 0.5 * _norm(b, mask)
        # Specular highlights on wet leaves also raise L; clip the extreme tail so
        # a glint cannot dominate the vote.
        v = youth[mask]
        if v.size:
            hi = np.percentile(v, self.clip_pct)
            youth = np.minimum(youth, hi)
        sigma = max(2.0, 0.08 * np.sqrt(mask.sum()))
        return _norm(cv2.GaussianBlur(youth, (0, 0), sigma), mask)


class CanopyHeightEvidence(LEPEvidence):
    name = "canopy_height"
    rationale = ("Leaves stack where they emerge and the growing tip is elevated "
                 "relative to the expanded leaves and the soil around it, so "
                 "height above the local soil plane is an independent physical "
                 "vote for the crown. Requires valid stereo depth on the plant, "
                 "and abstains when there is not enough of it.")

    def __init__(self, weight=1.0, min_valid_frac=0.25):
        super().__init__(weight)
        self.min_valid_frac = min_valid_frac

    def available(self, ctx):
        if not ctx.has_depth:
            return False
        d = np.nan_to_num(ctx.depth_mm, nan=0.0)
        valid = (d > 0) & ctx.mask
        denom = int(ctx.mask.sum())
        return denom > 0 and (valid.sum() / denom) >= self.min_valid_frac

    def score(self, ctx):
        d = np.nan_to_num(ctx.depth_mm, nan=0.0).astype(np.float32)
        valid = (d > 0) & ctx.mask
        if not valid.any():
            return np.zeros(ctx.mask.shape, np.float32)
        # Soil reference: depth just outside the plant. Nearer camera = smaller
        # depth, so height above soil is (soil_depth - plant_depth).
        ring = cv2.dilate(ctx.mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ring &= ~ctx.mask
        soil = d[(d > 0) & ring]
        ref = float(np.median(soil)) if soil.size else float(np.percentile(d[valid], 90))
        height = np.zeros_like(d)
        height[valid] = ref - d[valid]
        height = cv2.GaussianBlur(height, (0, 0), max(1.5, 0.05 * np.sqrt(ctx.mask.sum())))
        return _norm(height, valid)


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {
    # Rosette dicots (brassica, primrose, most broadleaf weeds): structure and
    # phyllotaxy are highly informative, so convergence and isotropy lead.
    "rosette": {"petiole_convergence": 1.0, "radial_isotropy": 0.9,
                "young_tissue": 0.8, "canopy_height": 0.6, "medial_axis": 0.4},
    # Grasses/tillers: leaves emerge from a basal point, not a radial rosette, so
    # angular isotropy is much weaker evidence while convergence still holds.
    "grass": {"petiole_convergence": 1.0, "radial_isotropy": 0.3,
              "young_tissue": 0.8, "canopy_height": 0.5, "medial_axis": 0.5},
}



class LEPEstimator:
    """Fuses evidence channels into one defensible LEP estimate.

    Fusion is a weighted sum of per-channel score maps (each normalised inside
    the plant mask), followed by a sub-pixel soft-argmax over the dominant peak.
    Because the channels are independent, their mutual DISAGREEMENT is a direct,
    interpretable uncertainty measure, and drives abstention.
    """

    def __init__(self, channels=None, top_frac=0.25,
                 agreement_good_frac=0.10, agreement_bad_frac=0.30,
                 min_confidence=0.35):
        self.channels = channels if channels is not None else [
            PetioleConvergenceEvidence(), RadialIsotropyEvidence(),
            YoungTissueEvidence(), CanopyHeightEvidence(), MedialAxisEvidence()]
        self.top_frac = top_frac
        # Agreement thresholds are expressed as a fraction of plant radius, so
        # they are scale free: a 5 px spread means something very different on a
        # cotyledon than on a mature rosette.
        self.agreement_good_frac = agreement_good_frac
        self.agreement_bad_frac = agreement_bad_frac
        self.min_confidence = min_confidence

    def weights_for(self, class_name):
        key = "rosette" if class_name in ROSETTE_CLASSES else "grass"
        return DEFAULT_WEIGHTS[key]

    def describe(self):
        """Full provenance of the method, for the paper and for session metadata."""
        return {"method_version": LEP_METHOD_VERSION,
                "fusion": "weighted sum of mask-normalised evidence maps, "
                          "sub-pixel soft-argmax over the dominant peak",
                "channels": [c.describe() for c in self.channels],
                "class_weights": DEFAULT_WEIGHTS}

    def estimate(self, ctx: PlantContext) -> Optional[LEPResult]:
        mask = ctx.mask
        if mask is None or not mask.any():
            return None
        weights = self.weights_for(ctx.class_name)
        radius = float(np.sqrt(mask.sum() / np.pi))       # equivalent-disc radius

        fused = np.zeros(mask.shape, np.float32)
        used, total_w = {}, 0.0
        for ch in self.channels:
            w = float(weights.get(ch.name, ch.weight))
            if w <= 0 or not ch.available(ctx):
                continue
            s = ch.score(ctx)
            if not np.isfinite(s).any() or s.max() <= 0:
                continue
            peak = np.unravel_index(int(np.argmax(s)), s.shape)
            used[ch.name] = {"uv": (float(peak[1]), float(peak[0])), "weight": w,
                             "peak": float(s.max())}
            fused += w * s
            total_w += w
        if total_w <= 0 or fused.max() <= 0:
            return None
        fused /= total_w

        # Sub-pixel location: intensity-weighted centroid of the dominant peak
        # region. Using a region rather than a single argmax pixel makes the
        # estimate robust to one-pixel noise and yields a covariance for free.
        thr = fused.max() * (1.0 - self.top_frac)
        top = fused >= thr
        top &= mask
        if not top.any():
            top = fused >= fused.max()
        wts = fused[top].astype(np.float64)
        ys, xs = np.nonzero(top)
        cx = float((xs * wts).sum() / wts.sum())
        cy = float((ys * wts).sum() / wts.sum())

        dx, dy = xs - cx, ys - cy
        wsum = wts.sum()
        cov = [[float((wts * dx * dx).sum() / wsum), float((wts * dx * dy).sum() / wsum)],
               [float((wts * dx * dy).sum() / wsum), float((wts * dy * dy).sum() / wsum)]]
        eig = np.linalg.eigvalsh(np.array(cov)) if wsum > 0 else np.array([0.0, 0.0])
        sigma = float(np.sqrt(max(0.0, float(eig.max()))))

        # Inter-channel agreement: median distance of each channel's own argmax
        # from the fused estimate. Small = the independent lines of evidence
        # coincide, which is exactly the argument that the point is real.
        if used:
            d = [float(np.hypot(c["uv"][0] - cx, c["uv"][1] - cy)) for c in used.values()]
            agreement = float(np.median(d))
        else:
            agreement = float("inf")

        good, bad = self.agreement_good_frac * radius, self.agreement_bad_frac * radius
        agree_score = float(np.clip((bad - agreement) / max(1e-6, bad - good), 0.0, 1.0))
        # Peak sharpness: a tight peak relative to the plant is a decisive
        # estimate; a diffuse one means the evidence is spread over the plant.
        sharp = float(np.clip(1.0 - (sigma / max(1e-6, 0.6 * radius)), 0.0, 1.0))
        n_factor = min(1.0, len(used) / 3.0)      # more agreeing channels = better
        confidence = float(np.clip(0.5 * agree_score + 0.35 * sharp + 0.15 * n_factor,
                                   0.0, 1.0))

        if confidence >= 0.6 and agreement <= good * 1.5:
            visibility = "visible"
        elif confidence >= self.min_confidence:
            visibility = "partially_occluded_inferable"
        else:
            visibility = "not_visible"

        ox, oy = ctx.origin
        return LEPResult(uv=(cx + ox, cy + oy), uv_local=(cx, cy),
                         confidence=confidence, visibility=visibility,
                         agreement_px=agreement, covariance=cov, sigma_px=sigma,
                         channels=used)


def crop_context(mask_full, bgr_full, bbox, depth_full=None, pad=8,
                 class_name="other_weed"):
    """Build a PlantContext for one instance from full-frame inputs."""
    x, y, w, h = bbox
    H, W = mask_full.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    d = depth_full[y0:y1, x0:x1] if depth_full is not None else None
    return PlantContext(mask=mask_full[y0:y1, x0:x1],
                        bgr=bgr_full[y0:y1, x0:x1],
                        depth_mm=d, origin=(x0, y0), class_name=class_name)
