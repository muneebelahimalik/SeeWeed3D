#!/usr/bin/env python3
"""
SeeWeed3D - structured inference output.

One dataclass per weed, carrying everything needed to (a) act, (b) audit the
decision months later, and (c) compute every evaluation metric without
re-running inference. Provenance fields (session, frame, instance) are first
because a result that cannot be traced back to its frame cannot be checked
against the annotation that disagrees with it.

`safety_status` and `rejection_reasons` are the fields the control layer reads.
Everything else is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

SCHEMA_VERSION = "seeweed3d/perception_result/1.0"

STATUS_CANDIDATE = "candidate"     # safe to consider for treatment
STATUS_ABSTAIN = "abstain"         # explicitly refused; see rejection_reasons


@dataclass
class WeedTarget:
    """One detected plant and its treatment verdict."""

    # -- provenance --------------------------------------------------------
    session_id: str = ""
    frame_id: str = ""
    instance_index: int = -1

    # -- segmentation ------------------------------------------------------
    class_name: str = ""
    class_index: int = -1
    class_confidence: float = 0.0
    bbox_xywh: Optional[list] = None
    mask_ref: Optional[str] = None       # key/handle into the frame's mask store
    mask_area_px: int = 0

    # -- LEP (2D) ----------------------------------------------------------
    lep_uv: Optional[list] = None        # full-frame (u, v), sub-pixel
    lep_peak: float = 0.0                # heatmap peak, the model's confidence
    lep_sigma_px: float = 0.0            # spatial spread -> abstention signal
    lep_covariance: Optional[list] = None
    visibility: str = "unknown"
    visibility_probs: Optional[list] = None
    targetable: str = "unknown"
    targetability_probs: Optional[list] = None
    lep_snapped_px: float = 0.0          # recorded in-tolerance correction

    # -- depth / 3D --------------------------------------------------------
    used_depth: bool = False
    depth_valid_fraction: float = 0.0
    depth_spread_mm: Optional[float] = None
    depth_stats: dict = field(default_factory=dict)
    xyz_mm: Optional[list] = None        # camera frame
    xyz_sigma_mm: Optional[float] = None
    xyz_covariance: Optional[list] = None
    is_3d_fallback: bool = False         # bed-plane intersection, not measured

    # -- safety ------------------------------------------------------------
    safety_status: str = STATUS_ABSTAIN
    abstained: bool = True
    rejection_reasons: list = field(default_factory=list)
    safety_notes: dict = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)


@dataclass
class FrameResult:
    """All targets for one frame, plus the crop-safety mask that gated them."""

    session_id: str = ""
    frame_id: str = ""
    width: int = 0
    height: int = 0
    targets: list = field(default_factory=list)      # WeedTarget
    onion_mask_ref: Optional[str] = None
    onion_area_px: int = 0
    n_instances: int = 0
    timings_ms: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def candidates(self):
        return [t for t in self.targets if t.safety_status == STATUS_CANDIDATE]

    @property
    def abstentions(self):
        return [t for t in self.targets if t.abstained]

    def reason_counts(self):
        counts = {}
        for t in self.targets:
            for r in t.rejection_reasons:
                counts[r] = counts.get(r, 0) + 1
        return counts

    def to_dict(self):
        d = asdict(self)
        d["n_candidates"] = len(self.candidates)
        d["reason_counts"] = self.reason_counts()
        return d
