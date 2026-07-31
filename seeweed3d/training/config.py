#!/usr/bin/env python3
"""
SeeWeed3D - configuration dataclasses for the supervised perception baseline.

Every tunable lives here rather than as a scattered constant, so a run is
reproducible from one object and an experiment is one diff. Nothing in this
module imports torch, ultralytics or datumaro, so it can be read and tested in
the lightweight environment.

PATHS ARE NEVER HARD-CODED. Defaults are None or relative; the caller supplies
absolute paths. That keeps machine-specific roots (E:\\..., /home/...) out of
the repository entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence


# --------------------------------------------------------------------------- #
# Annotation contract
# --------------------------------------------------------------------------- #
@dataclass
class AnnotationContract:
    """Rules a verified CVAT/Datumaro export must satisfy to be trainable.

    These are validation thresholds, not preferences: a violation means the
    annotation cannot be turned into a correct training target, so it is
    reported and (where it would corrupt a target) excluded."""

    # An LEP must lie inside its owning weed mask. Annotators click at finite
    # precision and polygon edges are quantised, so a small outside distance is
    # tolerated and recorded rather than rejected outright.
    lep_inside_mask_tolerance_px: float = 4.0

    # Visibility values that require a grouped LEP to exist. A not_visible weed
    # legitimately has no LEP and must never be forced to invent one.
    visibility_requiring_lep: Sequence[str] = ("visible",
                                               "partially_occluded_inferable")

    # Classes that are never a single treatable plant, so they must not carry a
    # single LEP target even if one was drawn.
    non_targetable_classes: Sequence[str] = ("weed_cluster",)

    # Minimum instance mask area. Below this an annotation is more likely a
    # stray click than a plant, but it is REPORTED rather than dropped silently.
    min_instance_area_px: int = 24

    # Fail the whole import on an unknown label instead of skipping it. A label
    # the ontology does not define means the CVAT schema and the code have
    # diverged, and silently dropping it would train on an incomplete target.
    strict_unknown_labels: bool = True


# --------------------------------------------------------------------------- #
# ROI extraction (Stage B input)
# --------------------------------------------------------------------------- #
@dataclass
class RoiConfig:
    """How a weed instance becomes a fixed-size network input.

    Aspect ratio is PRESERVED with padding rather than stretched: stretching
    changes the apparent phyllotaxy of a rosette, which is exactly the structure
    the LEP head has to read."""

    out_size: int = 128                 # square network input, HxW
    # Expand the instance bbox before cropping. The LEP sits at the crown, but
    # the surrounding soil ring is what makes local height meaningful, and
    # context helps disambiguate which plant a leaf belongs to.
    expand_ratio: float = 1.35
    min_box_px: int = 16                # never crop below this, avoids 0-size ROIs
    pad_value: int = 0                  # letterbox fill for RGB


# --------------------------------------------------------------------------- #
# LEP heatmap targets and losses
# --------------------------------------------------------------------------- #
@dataclass
class HeatmapConfig:
    """Gaussian target generation for LEP supervision."""

    stride: int = 4                     # heatmap is out_size/stride per side
    sigma_px: float = 2.0               # base sigma, in HEATMAP pixels
    # Scale sigma with plant size so a cotyledon and a large rosette are
    # supervised at a comparable RELATIVE precision. 0 disables scaling.
    sigma_scale_with_plant: float = 0.0
    sigma_min_px: float = 1.0
    sigma_max_px: float = 6.0
    # Supervision weight for LEPs the annotator could only infer. Lower than 1
    # because the ground truth itself is less certain; 0 would discard a real
    # and useful signal.
    partial_visibility_weight: float = 0.5


@dataclass
class LossWeights:
    """Relative weights of the multitask LEP losses.

    Kept few and each individually justified - a long list of unvalidated
    auxiliary losses is harder to debug than the task itself."""

    heatmap: float = 1.0                # per-pixel localisation (MSE on Gaussian)
    soft_argmax: float = 0.1            # direct coordinate error, sub-pixel
    visibility: float = 0.5             # 3-way classification
    targetability: float = 0.5          # 3-way classification
    # Penalises probability mass placed outside the owning instance mask. This
    # is the loss that teaches "the LEP belongs to THIS plant", which is the
    # difference between a safe target and a neighbouring plant's crown.
    outside_mask: float = 0.2


# --------------------------------------------------------------------------- #
# Geometry / depth representation
# --------------------------------------------------------------------------- #
@dataclass
class DepthRepresentationConfig:
    """How depth enters the network.

    Raw absolute depth is deliberately NOT used as a fourth channel: it encodes
    camera mount height, so a model trained at one height silently fails at
    another. Height above the LOCAL soil reference is camera-transferable, which
    is the property that matters for deployment on a different rig."""

    use_depth: bool = True
    # Ring outside the instance used to estimate the local soil/bed reference.
    soil_ring_inner_ratio: float = 1.10   # x instance radius
    soil_ring_outer_ratio: float = 1.60
    soil_ring_min_valid_px: int = 40      # below this the reference is unreliable
    # Physical clipping. Plants below/above these heights are implausible and
    # would otherwise let a depth artefact dominate the normalised channel.
    height_min_mm: float = -50.0
    height_max_mm: float = 400.0
    # Probability that depth is dropped entirely during training, so the network
    # cannot become dependent on a stream that is often invalid in the field.
    depth_dropout_p: float = 0.25
    # Simulated degradation, applied only in training.
    hole_dropout_p: float = 0.15
    noise_mm_per_m: float = 5.0           # range-dependent stereo noise
    quantisation_mm: float = 1.0


@dataclass
class ModelConfig:
    """LEPRoiNet architecture and which inputs the ablation enables."""

    # "rgb" | "rgb_mask" | "rgb_mask_geom"
    # rgb_mask_geom is the full model; the other two are the required ablations
    # and are also the fallbacks when depth is unavailable at runtime.
    input_mode: str = "rgb_mask_geom"
    width: int = 32                       # base channel count
    n_visibility: int = 3
    n_targetability: int = 3
    heatmap_stride: int = 4
    pretrained_backbone: bool = False     # torchvision weights need network access


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 1234
    num_workers: int = 0
    device: str = "cpu"                   # caller sets "cuda" when available
    amp: bool = False


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
@dataclass
class SafetyConfig:
    """Thresholds for the treatment-candidate decision.

    Every one of these is a REJECTION threshold. The decision function has no
    way to approve something that fails one - it can only abstain."""

    # Classification
    min_class_confidence: float = 0.50
    # LEP quality
    min_lep_confidence: float = 0.30
    max_lep_sigma_px: float = 12.0
    min_visibility_confidence: float = 0.50
    min_targetable_confidence: float = 0.50
    # Ownership. Beyond this the LEP is not on the plant it claims to be on.
    lep_outside_mask_tolerance_px: float = 3.0
    # Crop safety. The laser spot has physical extent, so keep the whole spot
    # plus a margin clear of onion tissue, not merely the centre pixel.
    laser_spot_radius_px: float = 6.0
    onion_safety_margin_px: float = 12.0
    # Depth / 3D
    min_depth_valid_fraction: float = 0.35
    max_depth_spread_mm: float = 40.0
    max_3d_sigma_mm: float = 15.0
    # A tiny numerical nudge of a LEP back inside its mask is permitted, but
    # only this far, and it is always recorded in the result.
    allow_snap_to_mask_px: float = 2.0


# --------------------------------------------------------------------------- #
# Top-level bundles
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Where data comes from and how splits are formed."""

    datumaro_root: Optional[str] = None   # dir holding annotations/*.json
    images_root: Optional[str] = None     # dataset sessions root, for image lookup
    out_root: Optional[str] = None        # where manifests/reports are written
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    holdout_sessions: Sequence[str] = ()
    seed: int = 1234


@dataclass
class PipelineConfig:
    """Everything the runtime inference pipeline needs."""

    roi: RoiConfig = field(default_factory=RoiConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    depth: DepthRepresentationConfig = field(default_factory=DepthRepresentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    seg_conf: float = 0.25
    seg_iou: float = 0.60
    max_instances: int = 300

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentConfig:
    """One reproducible experiment."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    contract: AnnotationContract = field(default_factory=AnnotationContract)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)
