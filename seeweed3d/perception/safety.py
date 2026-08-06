#!/usr/bin/env python3
"""
SeeWeed3D - safety-aware treatment-candidate decision.

THIS MODULE PRODUCES CANDIDATES ONLY. It has no I/O, no actuator handle, and no
dependency that could reach one. Turning a candidate into a laser command is a
separate, deliberate act by the control layer - keeping that boundary in the
type system is what stops a perception bug from becoming a fired laser.

DESIGN: THE FUNCTION CAN ONLY REJECT
------------------------------------
`decide()` starts from "reject" and every check can veto. There is no branch
that promotes a candidate; acceptance is simply the absence of any veto, and
every veto records a machine-readable reason. That structure means a new failure
mode is added by writing one more check, and forgetting to handle a case fails
CLOSED (no target) rather than open (an unverified target).

ALL reasons are recorded, not just the first, so a rejected weed can be
diagnosed in one pass instead of iteratively.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CROP_CLASS  # noqa: E402

# Machine-readable rejection reasons. Stable strings: they are logged, counted
# in the safety metrics, and read by the control layer.
R_ONION = "onion_plant"
R_CLUSTER = "weed_cluster"
R_NOT_VISIBLE = "lep_not_visible"
R_LOW_CONF = "low_lep_confidence"
R_HIGH_UNC = "high_heatmap_uncertainty"
R_OUTSIDE_MASK = "outside_owning_mask"
R_ONION_CONFLICT = "onion_safety_conflict"
R_NO_DEPTH = "insufficient_valid_depth"
R_DEPTH_DISC = "depth_discontinuity"
R_NOT_TARGETABLE = "not_targetable"
R_CLASS_UNCERTAIN = "class_uncertain"
R_3D_UNCERTAIN = "high_3d_uncertainty"
R_NO_LEP = "no_lep_predicted"
R_CROP_UNVERIFIABLE = "crop_protection_unavailable"

ALL_REJECTION_REASONS = (
    R_ONION, R_CLUSTER, R_NOT_VISIBLE, R_LOW_CONF, R_HIGH_UNC, R_OUTSIDE_MASK,
    R_ONION_CONFLICT, R_NO_DEPTH, R_DEPTH_DISC, R_NOT_TARGETABLE,
    R_CLASS_UNCERTAIN, R_3D_UNCERTAIN, R_NO_LEP, R_CROP_UNVERIFIABLE)


@dataclass
class SafetyDecision:
    """Outcome for one weed. `is_candidate` is the only field the control layer
    may act on, and it is False unless every check passed."""
    is_candidate: bool = False
    abstained: bool = True
    reasons: list = field(default_factory=list)
    snapped_px: float = 0.0          # tiny in-tolerance correction actually applied
    notes: dict = field(default_factory=dict)

    def reject(self, reason):
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.is_candidate = False
        self.abstained = True
        return self

    def to_dict(self):
        return asdict(self)


def _distance_inside_mask(mask, u, v):
    """Signed distance to the mask boundary: >=0 inside, <0 outside (pixels).

    Uses the distance transform of the mask and of its complement, which is
    exact on the pixel grid and needs no contour extraction."""
    import cv2
    h, w = mask.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return -float("inf")
    m = mask.astype(np.uint8)
    if m[vi, ui]:
        return float(cv2.distanceTransform(m, cv2.DIST_L2, 3)[vi, ui])
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    return -float(outside[vi, ui])


def check_onion_conflict(onion_mask, u, v, spot_radius_px, margin_px):
    """Does the laser spot (plus margin) touch any onion tissue?

    Tests a DISC, not the centre pixel: the beam has physical extent, and a spot
    whose centre clears the crop while its edge does not would still damage it.
    Returns (conflict, min_distance_px)."""
    if onion_mask is None or not np.any(onion_mask):
        return False, float("inf")
    import cv2
    h, w = onion_mask.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return False, float("inf")
    if onion_mask[vi, ui]:
        return True, 0.0
    dist = cv2.distanceTransform((~onion_mask.astype(bool)).astype(np.uint8),
                                 cv2.DIST_L2, 3)
    d = float(dist[vi, ui])
    return d <= (float(spot_radius_px) + float(margin_px)), d


def decide(*, class_name, class_confidence, lep, instance_mask, onion_mask,
           cfg, visibility=None, visibility_conf=None, targetable=None,
           targetable_conf=None, depth_result=None, is_cluster=None):
    """Full safety decision for one detected weed.

    lep: dict from lep_targets.decode_lep(), or None when no LEP was produced.
    depth_result: dict from depth3d.localize_lep_3d(), or None if depth was not
    used (which is allowed - the 2D candidate can still be rejected/accepted on
    2D grounds and simply carries no 3D point).

    Returns SafetyDecision. Reasons accumulate; nothing short-circuits, so one
    call explains every problem with the candidate."""
    d = SafetyDecision()

    # --- Class-level vetoes -------------------------------------------------
    if class_name == CROP_CLASS:
        d.reject(R_ONION)
    if is_cluster if is_cluster is not None else (class_name == "weed_cluster"):
        d.reject(R_CLUSTER)
    if class_confidence is None or float(class_confidence) < cfg.min_class_confidence:
        d.reject(R_CLASS_UNCERTAIN)

    # --- Annotator/model verdicts ------------------------------------------
    if visibility == "not_visible":
        d.reject(R_NOT_VISIBLE)
    if visibility_conf is not None and float(visibility_conf) < cfg.min_visibility_confidence:
        d.reject(R_NOT_VISIBLE)
    if targetable == "no":
        d.reject(R_NOT_TARGETABLE)
    if targetable == "uncertain":
        d.reject(R_NOT_TARGETABLE)
    if targetable_conf is not None and float(targetable_conf) < cfg.min_targetable_confidence:
        d.reject(R_NOT_TARGETABLE)

    # --- The LEP itself -----------------------------------------------------
    if lep is None:
        d.reject(R_NO_LEP)
        return d

    u, v = float(lep["uv_full"][0]), float(lep["uv_full"][1])
    if float(lep.get("peak", 0.0)) < cfg.min_lep_confidence:
        d.reject(R_LOW_CONF)
    if float(lep.get("sigma_px", 0.0)) > cfg.max_lep_sigma_px:
        d.reject(R_HIGH_UNC)

    # --- Ownership ----------------------------------------------------------
    if instance_mask is not None:
        signed = _distance_inside_mask(instance_mask, u, v)
        d.notes["lep_signed_distance_px"] = (
            None if signed == -float("inf") else round(float(signed), 2))
        if signed == -float("inf"):
            d.reject(R_OUTSIDE_MASK)
        elif signed < 0:
            outside_by = -signed
            if outside_by > cfg.lep_outside_mask_tolerance_px:
                d.reject(R_OUTSIDE_MASK)
            elif outside_by <= cfg.allow_snap_to_mask_px:
                # A sub-pixel-scale correction is allowed, and is RECORDED. It
                # is never a silent reprojection of a badly wrong point: beyond
                # allow_snap_to_mask_px the candidate is rejected above.
                d.snapped_px = float(outside_by)
                d.notes["snapped"] = True
            else:
                d.reject(R_OUTSIDE_MASK)

    # --- Crop safety --------------------------------------------------------
    # onion_mask is None means the segmenter CANNOT predict the crop class at
    # all - a weed-only model, trained from an export with no onion instances.
    # That is not the same statement as "looked and found no onions", and
    # treating it as such is the fail-open case this whole module exists to
    # prevent: the laser would be cleared to fire in a crop row by a model that
    # is structurally incapable of seeing the crop.
    #
    # Firing anyway requires cfg.allow_missing_crop_mask, which is a claim ABOUT
    # THE FIELD - that no crop is present - and can only be made by whoever is
    # standing in it. It is recorded either way.
    d.notes["crop_mask_available"] = onion_mask is not None
    if onion_mask is None:
        if not getattr(cfg, "allow_missing_crop_mask", False):
            d.reject(R_CROP_UNVERIFIABLE)
        d.notes["onion_distance_px"] = None
    else:
        conflict, dist = check_onion_conflict(onion_mask, u, v,
                                              cfg.laser_spot_radius_px,
                                              cfg.onion_safety_margin_px)
        d.notes["onion_distance_px"] = (None if dist == float("inf")
                                        else round(float(dist), 2))
        if conflict:
            d.reject(R_ONION_CONFLICT)

    # --- Depth / 3D ---------------------------------------------------------
    if depth_result is not None:
        stats = depth_result.get("depth_stats", {}) or {}
        if not depth_result.get("ok"):
            reason = depth_result.get("reason", R_NO_DEPTH)
            d.reject(R_DEPTH_DISC if reason == R_DEPTH_DISC else R_NO_DEPTH)
        else:
            if stats.get("valid_fraction") is not None and \
                    float(stats["valid_fraction"]) < cfg.min_depth_valid_fraction:
                d.reject(R_NO_DEPTH)
            if stats.get("spread_mm") is not None and \
                    float(stats["spread_mm"]) > cfg.max_depth_spread_mm:
                d.reject(R_DEPTH_DISC)
            if depth_result.get("sigma_mm") is not None and \
                    float(depth_result["sigma_mm"]) > cfg.max_3d_sigma_mm:
                d.reject(R_3D_UNCERTAIN)

    if not d.reasons:
        d.is_candidate = True
        d.abstained = False
    return d
