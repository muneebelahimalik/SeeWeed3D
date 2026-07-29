#!/usr/bin/env python3
"""
SeeWeed3D - Weed instance prelabeling with SAM 3  (WEED-ONLY scenes)
====================================================================
Builds a multi-class weed INSTANCE dataset from weed-only recordings, with a
proposed LEP/AMT growth point per instance, exported for correction in CVAT.

    Run extraction/extract_sessions.py first (this reads its output pool).

WHY THIS DIFFERS FROM THE ONION PRELABELER
------------------------------------------
Onions need one high-recall SEMANTIC safety mask (coverage over separation).
Weeds need the opposite: every individual plant separated, classified by
morphology, and given a treatment point. So this module adds instance
separation, per-instance shape descriptors, a morphology proposal, and an LEP
proposal.

In a weed-only field there is no crop to confuse, so vegetation == weed is a
legitimate high-recall prior - exactly as vegetation == onion is in an
onion-only field. Do NOT point this script at mixed scenes.

PIPELINE
--------
1. White balance -> vegetation prior (ExG + green dominance + saturation).
2. Vegetation blobs become SAM 3 exemplar boxes; SAM 3 concept segmentation
   returns every matching instance as its own mask (the text concept does not
   ground on top-down field imagery, so exemplars are the default).
3. Instance masks are validated against the vegetation prior and de-duplicated
   by mask NMS.
4. Per instance: shape descriptors -> provisional morphology class, growth-stage
   estimate, and THREE candidate treatment points -
       lep_dt   : distance-transform peak (deepest interior point). For a
                  rosette this is the centre of the plant, i.e. the growth
                  point - the best geometric prior for the LEP/AMT.
       centroid : mask centroid            (project plan baseline)
       bbox_ctr : bounding-box centre      (project plan baseline)
   Storing all three gives the plan's LEP-method comparison for free once human
   LEP labels exist.
5. Export COCO (categories = morphology classes) + per-instance CSV + instance
   crops (ready for DINO embedding / cluster-then-label) + previews.

THE PROVISIONAL CLASS IS A TIME-SAVER, NOT A CLAIM
--------------------------------------------------
The morphology heuristic exists to pre-fill CVAT so annotators correct rather
than classify from scratch. Thresholds are priors to be calibrated against the
first verified round - they are not a trained classifier, and every instance is
expected to be reviewed.
"""

import contextlib
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.vegetation import (component_boxes, remove_small,  # noqa: E402
                               vegetation_mask, vegetation_score,
                               white_balance)
from common.progress import Progress  # noqa: E402
from common.ontology import (CLASS_COLORS_BGR, LEP_LABEL,  # noqa: E402
                             WEED_CLASSES, coco_categories, cvat_labels)
from perception.lep import LEPEstimator, LEPResult, crop_context  # noqa: E402

# #############################################################################
# ##   DATASET_ROOT   -  the OUTPUT_ROOT you gave extract_sessions.py        ##
# ##   SAM_VERSION    -  "sam3" or "sam3.1"                                  ##
# ##   SAM_CHECKPOINT -  local .pt path, or None to auto-download            ##
# ##   ONLY_SESSIONS  -  MUST list your WEED-ONLY sessions                   ##
# #############################################################################

DATASET_ROOT   = r"E:\Dataset_Vidalia"
SAM_VERSION    = "sam3"
SAM_CHECKPOINT = r"E:\Models\sam3.pt"

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "DATASET_ROOT":   DATASET_ROOT,
    "SAM_VERSION":    SAM_VERSION,
    "SAM_CHECKPOINT": SAM_CHECKPOINT,
    "OUTPUT_SUBDIR":  "auto_labels_weeds",

    # Weed-only sessions. Empty = every session under sessions/ (only correct if
    # ALL of them are weed-only recordings).
    "ONLY_SESSIONS": [],

    "CVAT_READY_SUBDIR":  "cvat_ready",
    "FLAGGED_RGB_SUBDIR": "flagged_rgb",
    "SAVE_INSTANCE_CROPS": True,   # per-instance crops for DINO cluster-then-label

    # -- Preprocessing ---------------------------------------------------------
    "WHITE_BALANCE": True,
    "WB_CAST_RATIO": 1.15,

    # -- Vegetation prior ------------------------------------------------------
    "EXG_THRESHOLD": 0.05,
    "VEG_MIN_SATURATION": 40,
    "VEG_MORPH_KERNEL": 3,
    "VEG_MIN_COMPONENT_PX": 150,

    # -- SAM 3 prompting -------------------------------------------------------
    "SAM_PROMPT_MODE": "auto_exemplar",     # auto_exemplar | text | manual
    "SAM_TEXT_PROMPTS": ["plant", "weed", "green plant"],
    "EXEMPLARS": {},                         # {session_id: [[x1,y1,x2,y2], ...]}
    # Exemplars are all submitted in ONE forward pass, so prompting with more of
    # them costs almost nothing while directly raising recall: a plant that never
    # becomes an exemplar and is not similar to one that did is a plant SAM has
    # no reason to return. Dense field frames hold far more than 30 plants, so
    # the old cap silently prompted on only the largest few.
    "EXEMPLAR_MIN_AREA_PX": 200,
    "EXEMPLAR_MAX_BOXES": 60,
    "EXEMPLAR_PAD_PX": 8,
    "SAM_CONF": 0.25,
    "DEVICE": "cuda",

    # -- Instance filtering ----------------------------------------------------
    # MIN_INSTANCE_AREA_PX controls how small a detection may be. It was briefly
    # raised to 700 on the assumption that the many small detections in dense
    # field frames were noise; checking the imagery showed they are real
    # cotyledon-stage weeds, so 700 silently deleted genuine plants. Kept at 250:
    # for a laser weeder a missed small weed is a worse failure than an extra
    # instance the annotator deletes in one click. Raise only if previews show
    # detections on bare soil rather than on real seedlings.
    "MIN_INSTANCE_AREA_PX": 250,     # ~16x16 px at 2208x1242
    "MAX_INSTANCE_FRAC": 0.25,       # one weed covering >25% of frame = failure
    "INSTANCE_VEG_OVERLAP_MIN": 0.35,  # instance must sit on vegetation
    "NMS_IOU": 0.65,                 # de-duplicate overlapping SAM instances

    # -- Recall backstop -------------------------------------------------------
    # A weed SAM misses is dropped silently and never reaches the annotator, so
    # it never enters the training target - the worst failure mode in this
    # pipeline. In a weed-only scene every vegetation blob IS a plant, so any
    # substantial unclaimed vegetation component is recovered as an instance.
    # See recover_missed_plants().
    "RECOVER_MISSED_PLANTS": True,
    "RECOVER_COVERED_DILATE_PX": 3,   # tolerance between SAM edge and veg prior
    "RECOVER_MAX_CLAIMED_FRAC": 0.30,  # blob already this claimed -> not a new plant

    # -- Morphology heuristic --------------------------------------------------
    # WHAT SHAPE CAN AND CANNOT TELL YOU:
    #   CAN  - grass vs rosette. Elongation separates them cleanly: a blade has
    #          aspect ~20, a rosette ~1. (Circularity cannot: a rosette's
    #          radiating leaves give it a spiky perimeter, ~0.15, as low as a
    #          blade's ~0.13. Solidity cannot flag grass: a straight blade is
    #          nearly its own convex hull, ~0.9.)
    #   CAN  - an intermingled cluster, via multiple growth-point peaks.
    #   CANNOT - SPECIES. cutleaf_evening_primrose vs wild_radish is an
    #          appearance question, not a shape one (both are rosettes), so
    #          those are NEVER auto-assigned. Non-grass, non-cluster
    #          instances go to DEFAULT_SPECIES_CLASS for you (or the DINO
    #          cluster-then-label stage) to resolve.
    "GRASS_MIN_ASPECT": 3.0,          # >= this -> grass
    "DEFAULT_SPECIES_CLASS": "other_weed",

    # -- Intermingled cluster detection (deliberately HIGH threshold) ----------
    # A cluster is only declared when several distinct growth points sit inside
    # one connected mask, i.e. individual LEPs genuinely cannot be assigned.
    # Raise CLUSTER_MIN_PEAKS / CLUSTER_MIN_AREA_PX to make it rarer still.
    "CLUSTER_MIN_PEAKS": 3,           # distinct growth-point peaks in one mask
    "CLUSTER_MIN_AREA_PX": 20000,     # and it must be large
    "PEAK_REL_THRESHOLD": 0.5,        # peak counts if >= this fraction of max radius
    "PEAK_MIN_SEPARATION_PX": 15,

    # -- LEP estimation --------------------------------------------------------
    # Multi-evidence estimator (perception/lep.py): petiole convergence, radial
    # isotropy, young-tissue chromatics, canopy height and medial-axis
    # interiority. Set USE_FUSED_LEP False to fall back to the plain
    # distance-transform peak.
    "USE_FUSED_LEP": True,
    "USE_DEPTH_FOR_LEP": True,    # enables the canopy-height channel when depth exists
    "LEP_CROP_PAD_PX": 10,

    # -- Growth stage from instance area (px). Calibrate to your mount height. -
    "STAGE_COTYLEDON_MAX_PX": 1200,
    "STAGE_EARLY_MAX_PX": 6000,

    # -- Frame-level safety ----------------------------------------------------
    "MAX_MASK_FRACTION": 0.5,        # total veg > this = colour-cast/glare failure

    # -- Boundary quality ------------------------------------------------------
    # Snap each instance edge onto the image's own plant/soil evidence. SAM
    # gives the structure; the vegetation score decides exactly where the leaf
    # margin falls, within a narrow band. Set BAND to 0 to disable.
    "BOUNDARY_REFINE_BAND_PX": 3,
    "BOUNDARY_REFINE_VEG_MIN": 0.5,   # plant likelihood needed to keep a band pixel
    "VEG_SCORE_SOFTNESS": 0.04,       # ExG ramp width for the soft score

    # -- Splitting touching plants ---------------------------------------------
    # Two rosettes growing into each other form one connected blob, so SAM
    # returns them as a single instance. Marker-controlled watershed seeded on
    # the detected GROWTH POINTS cuts along the neck between the canopies.
    # Falls back to the unsplit mask whenever the split is not clean.
    "SPLIT_TOUCHING_INSTANCES": True,
    "SPLIT_MIN_PEAKS": 2,             # need at least this many growth points
    "SPLIT_SEED_RADIUS_FRAC": 0.55,   # seed disc as a fraction of inscribed radius
    "SPLIT_MIN_PART_AREA_PX": 250,    # a part below this is not a plant
    "SPLIT_MIN_COVERAGE": 0.80,       # split must retain this much of the blob

    # -- Polygon export --------------------------------------------------------
    # Tolerance scales with instance size for roughly constant relative fidelity
    # (a fixed value erases shape on seedlings and bloats vertices on rosettes).
    "POLY_APPROX_EPS_FRAC": 0.010,
    "POLY_APPROX_EPS_MIN": 0.5,
    "POLY_APPROX_EPS_MAX": 1.5,
    "POLY_MIN_PART_AREA_PX": 60,      # keep detached tissue above this
    "POLY_APPROX_EPS": 1.5,           # legacy, used only by mask_polygon()

    # -- Run control -----------------------------------------------------------
    "LIMIT_PER_SESSION": 20,         # start small; set None for the full pool
    "SAVE_PREVIEWS": True,
    "PREVIEW_SCALE": 0.5,
}

# Class names come from common/ontology.py so they cannot drift between stages.
# Species (cutleaf_evening_primrose, wild_radish) are NEVER auto-assigned: both
# are rosette-forming, so shape cannot tell them apart. The prelabeler only
# proposes grass_weed, weed_cluster, or the DEFAULT_SPECIES_CLASS fallback.
AUTO_CLASSES = {"grass_weed", "weed_cluster", "other_weed"}

# =============================================================================


def weed_cvat_labels():
    """CVAT label schema for weed verification (paste into the Raw editor).

    Includes onion_plant even though a weed-only scene should not contain the
    crop: if one does appear, the annotator can label it correctly instead of
    being forced to call it a weed. That is a crop-safety protection."""
    return cvat_labels()


# --------------------------------------------------------------------------- #
# SAM 3 (GPU-only part, isolated; lazy imports keep the rest testable)
# --------------------------------------------------------------------------- #
def load_sam3(cfg):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    want = cfg["DEVICE"]
    device = "cuda" if (want.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if want.startswith("cuda") and device == "cpu":
        print("[WARN] CUDA requested but unavailable - running SAM 3 on CPU (slow).")
    ckpt = cfg["SAM_CHECKPOINT"]
    if ckpt is None and cfg["SAM_VERSION"] != "sam3":
        from sam3.model_builder import download_ckpt_from_hf
        ckpt = download_ckpt_from_hf(version=cfg["SAM_VERSION"])
    print(f"[INFO] loading {cfg['SAM_VERSION']} on {device} "
          f"({'auto-download' if ckpt is None else ckpt})")
    model = build_sam3_image_model(device=device, checkpoint_path=ckpt,
                                   load_from_HF=(ckpt is None), eval_mode=True)
    processor = Sam3Processor(model)
    try:
        processor.set_confidence_threshold(cfg["SAM_CONF"])
    except Exception as e:
        print(f"[WARN] could not set confidence threshold: {e}")
    return {"model": model, "processor": processor, "device": device, "torch": torch}


def _state_masks(state):
    """Binary instance masks (original resolution) from a processor state."""
    masks = state.get("masks") if isinstance(state, dict) else None
    if masks is None:
        return []
    arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        return []
    return [arr[i].astype(bool) for i in range(arr.shape[0]) if arr[i].ndim == 2]


def sam3_instances(predictor, image, cfg, exemplars=None):
    """Instance masks for one image. SAM 3 concept segmentation returns every
    matching instance separately, which is exactly the instance segmentation we
    want. exemplars: list of xyxy px boxes, [] for none, None for text mode."""
    from PIL import Image
    if exemplars is not None and len(exemplars) == 0:
        return []
    torch = predictor["torch"]
    processor, device = predictor["processor"], predictor["device"]
    img = (Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
           if isinstance(image, np.ndarray) else Image.open(str(image)).convert("RGB"))
    w, h = img.size
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else contextlib.nullcontext())
    out = []
    with amp:
        state = processor.set_image(img)
        if exemplars is not None:
            processor.reset_all_prompts(state)
            for x1, y1, x2, y2 in (list(map(float, b)) for b in exemplars):
                box = [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                       (x2 - x1) / w, (y2 - y1) / h]
                state = processor.add_geometric_prompt(box=box, label=True, state=state)
            out.extend(_state_masks(state))
        else:
            for text in cfg["SAM_TEXT_PROMPTS"]:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt=text, state=state)
                out.extend(_state_masks(state))
    return out


# --------------------------------------------------------------------------- #
# Instance filtering  (pure OpenCV/NumPy - testable without a GPU)
# --------------------------------------------------------------------------- #
def mask_iou(a, b):
    union = int((a | b).sum())
    return (int((a & b).sum()) / union) if union else 0.0


def filter_instances(masks, veg, cfg):
    """Keep plausible weed instances: on vegetation, sensible size, de-duplicated.
    Returns instance masks sorted largest-first."""
    if veg.shape[0] == 0:
        return []
    h, w = veg.shape
    frame_px = h * w
    kept = []
    cand = []
    for m in masks:
        m = np.asarray(m)
        if m.ndim != 2 or 0 in m.shape:
            continue
        if m.shape != veg.shape:
            m = cv2.resize(m.astype(np.uint8), (int(w), int(h)),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(m.sum())
        if area < cfg["MIN_INSTANCE_AREA_PX"] or area > cfg["MAX_INSTANCE_FRAC"] * frame_px:
            continue
        if float((m & veg).sum()) / area < cfg["INSTANCE_VEG_OVERLAP_MIN"]:
            continue
        cand.append((area, m))

    cand.sort(key=lambda t: -t[0])              # NMS: prefer larger instances
    for _, m in cand:
        if all(mask_iou(m, k) <= cfg["NMS_IOU"] for k in kept):
            kept.append(m)
    return kept


# --------------------------------------------------------------------------- #
# Per-instance descriptors, morphology, and treatment points
# --------------------------------------------------------------------------- #
def shape_features(mask):
    """Scale-aware shape descriptors used for the morphology proposal."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    if area <= 0 or perim <= 0:
        return None
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) or area
    x, y, bw, bh = cv2.boundingRect(c)
    aspect = max(bw, bh) / max(1.0, min(bw, bh))
    if len(c) >= 5:                              # fitted ellipse is more robust
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        if min(MA, ma) > 1:
            aspect = max(MA, ma) / min(MA, ma)
    return {
        "area_px": int(mask.sum()),
        "perimeter_px": round(perim, 1),
        "solidity": round(area / hull_area, 4),          # lobed/dissected -> low
        "circularity": round(4 * np.pi * area / (perim ** 2), 4),
        "aspect_ratio": round(float(aspect), 3),
        "extent": round(area / max(1.0, bw * bh), 4),
        "bbox": [int(x), int(y), int(bw), int(bh)],
    }


def growth_peaks(mask, cfg):
    """Distinct growth-point peaks inside one mask, as [((x, y), radius), ...].

    Iteratively takes the distance-transform maximum and suppresses a disc around
    it. A single rosette yields one peak; several intermingled plants sharing one
    connected mask yield several - which is exactly the signal that individual
    LEPs cannot be assigned to that blob."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    gmax = float(dt.max())
    if gmax <= 0:
        return []
    work, peaks = dt.copy(), []
    while len(peaks) < 32:
        _, v, _, loc = cv2.minMaxLoc(work)
        if v <= 0 or v < cfg["PEAK_REL_THRESHOLD"] * gmax:
            break
        peaks.append((loc, float(v)))
        cv2.circle(work, loc, int(max(cfg["PEAK_MIN_SEPARATION_PX"], v)), 0, -1)
    return peaks


def classify_morphology(f, cfg, peaks=None):
    """Provisional class from shape. Only ever proposes what shape can support.

    grass        - by ELONGATION. Measured on synthetic and field shapes: a blade
                   has aspect ~20 while a rosette has ~1. (Circularity cannot
                   separate them - a rosette's radiating leaves make its
                   perimeter spiky, so circularity is as low as a blade's - and
                   solidity cannot flag grass, since a straight blade is nearly
                   its own convex hull.)
    weed_cluster - several distinct growth-point peaks inside one large mask,
                   i.e. intermingled plants whose LEPs cannot be separated.
                   Deliberately high thresholds so this is rare.
    SPECIES      - never auto-assigned. Both named species are rosette-forming,
                   so shape cannot tell them apart; everything else becomes
                   DEFAULT_SPECIES_CLASS with zero confidence, for the annotator
                   or the DINO cluster-then-label stage to resolve.
    """
    if f is None:
        return cfg["DEFAULT_SPECIES_CLASS"], 0.0
    ar = f["aspect_ratio"]
    if peaks is not None and len(peaks) >= cfg["CLUSTER_MIN_PEAKS"] \
            and f["area_px"] >= cfg["CLUSTER_MIN_AREA_PX"]:
        over = len(peaks) / max(1, cfg["CLUSTER_MIN_PEAKS"]) - 1.0
        return "weed_cluster", round(min(1.0, 0.5 + 0.5 * over), 3)
    if ar >= cfg["GRASS_MIN_ASPECT"]:
        margin = min(1.0, (ar / max(1e-6, cfg["GRASS_MIN_ASPECT"]) - 1.0))
        return "grass_weed", round(0.5 + 0.5 * margin, 3)
    return cfg["DEFAULT_SPECIES_CLASS"], 0.0


def growth_stage(area_px, cfg):
    if area_px <= cfg["STAGE_COTYLEDON_MAX_PX"]:
        return "cotyledon"
    if area_px <= cfg["STAGE_EARLY_MAX_PX"]:
        return "2-leaf"
    return "3-5-leaf"


def treatment_points(mask):
    """Three candidate treatment points for one instance, in pixels.

    lep_dt   - distance-transform peak: the deepest interior point of the mask.
               For a rosette weed this lands at the centre where the youngest
               leaves emerge, i.e. the growth point (LEP/AMT). This is the
               strongest purely geometric prior available before a trained
               heatmap model exists.
    centroid - mask centroid          (project plan baseline)
    bbox_ctr - bounding-box centre    (project plan baseline)

    Keeping all three lets the plan's LEP-method comparison be computed directly
    once human LEP labels exist. Also returns dt_radius_px: the inscribed-circle
    radius at the peak, a useful proxy for how well-defined the centre is.
    """
    m = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, max_r, _, max_loc = cv2.minMaxLoc(dt)
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "lep_dt": [float(max_loc[0]), float(max_loc[1])],
        "dt_radius_px": round(float(max_r), 2),
        "centroid": [round(centroid[0], 1), round(centroid[1], 1)],
        "bbox_ctr": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
        # How far the geometric centre sits from the centroid: large values flag
        # asymmetric / occluded plants where the LEP proposal is less reliable.
        "lep_centroid_dist_px": round(
            float(np.hypot(max_loc[0] - centroid[0], max_loc[1] - centroid[1])), 2),
    }


def recover_missed_plants(veg, taken, cfg):
    """Vegetation that no SAM instance claimed, returned as extra instances.

    THIS IS THE RECALL BACKSTOP. SAM only reports what it detects, so a plant it
    misses is otherwise dropped silently and never reaches the annotator - the
    single worst failure mode here, because a weed absent from the training
    target teaches the model that such plants do not exist. In a weed-only
    scene every vegetation blob IS a plant, so any substantial connected
    component of the vegetation prior that no instance covers is recovered.

    A residual is recovered only when the WHOLE vegetation blob it belongs to is
    mostly unclaimed. That distinction is what stops the backstop from
    fabricating duplicates: a leaf tip poking out past the edge of an instance
    that already covers its plant is a residual too, but its parent blob is
    almost entirely claimed, so it is rejected."""
    if not cfg.get("RECOVER_MISSED_PLANTS", True) or not veg.any():
        return []
    covered = np.zeros(veg.shape, bool)
    for m in taken:
        covered |= m

    dilated = covered
    if covered.any() and cfg["RECOVER_COVERED_DILATE_PX"] > 0:
        # Tolerate a small boundary mismatch between the SAM mask and the
        # vegetation prior, so a thin rim around a detected plant is not
        # mistaken for an undetected one.
        r = int(cfg["RECOVER_COVERED_DILATE_PX"])
        dilated = cv2.dilate(covered.astype(np.uint8),
                             np.ones((r * 2 + 1,) * 2, np.uint8)).astype(bool)

    residual = remove_small(veg & ~dilated, cfg["MIN_INSTANCE_AREA_PX"])
    if not residual.any():
        return []

    # How much of each vegetation blob the existing instances already claim.
    # Labelling the vegetation once and counting with bincount is exact and
    # costs one pass, where flooding each residual outwards would not be.
    n_veg, veg_lbl = cv2.connectedComponents(veg.astype(np.uint8), 8)
    blob_px = np.bincount(veg_lbl.ravel(), minlength=n_veg).astype(np.float64)
    claimed_px = np.bincount(veg_lbl[covered & veg].ravel(),
                             minlength=n_veg).astype(np.float64)
    claimed_frac = claimed_px / np.maximum(blob_px, 1.0)

    out = []
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg["MIN_INSTANCE_AREA_PX"]:
            continue
        comp = lbl == i
        parent = int(np.bincount(veg_lbl[comp].ravel(), minlength=n_veg)[1:].argmax()) + 1
        if claimed_frac[parent] <= cfg["RECOVER_MAX_CLAIMED_FRAC"]:
            out.append(comp)
    return out


def _bbox_window(mask, pad):
    """Slice covering the mask's content plus pad, or None if empty.

    Per-instance morphology on a full 2208x1242 frame is almost entirely spent
    on empty pixels: a plant occupies a tiny fraction of it. Working inside this
    window gives identical results for far less work."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    return (slice(y0, y1), slice(x0, x1))


def split_touching_instances(mask, peaks, cfg):
    """Split one connected mask covering several touching plants into one mask
    per plant, using marker-controlled watershed on the distance transform.

    Two rosettes growing into each other are a single connected vegetation blob,
    so SAM often returns them as ONE instance with no boundary between them.
    Watershed on the distance transform is the standard split for touching
    convex-ish objects, and here the markers are not arbitrary local maxima but
    the GROWTH POINTS already detected by growth_peaks() - i.e. the split is
    seeded on plant biology, and the cut falls along the narrow neck where the
    two canopies meet.

    Returns a list of masks. Falls back to [mask] whenever the split is not
    clean, so a doubtful split can never fabricate a boundary."""
    if not cfg.get("SPLIT_TOUCHING_INSTANCES", True):
        return [mask]
    if len(peaks) < cfg["SPLIT_MIN_PEAKS"]:
        return [mask]

    win = _bbox_window(mask, 2)
    if win is None:
        return [mask]
    full = mask
    y0, x0 = win[0].start, win[1].start
    mask = mask[win]
    peaks = [((px - x0, py - y0), r) for (px, py), r in peaks]
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # Seed each growth point with a disc scaled to its own inscribed radius, so
    # a big plant gets a big seed and a seedling a small one.
    # uint8 because cv2.dilate has no int32 path; growth_peaks caps at 32 peaks
    # so the label space is never exhausted.
    labels = np.zeros(mask.shape, np.uint8)
    for i, ((px, py), radius) in enumerate(peaks[:250], start=1):
        rr = max(1, int(radius * cfg["SPLIT_SEED_RADIUS_FRAC"]))
        cv2.circle(labels, (int(px), int(py)), rr, i, -1)
    labels[~mask] = 0
    if not (labels > 0).any():
        return [full]

    # GEODESIC assignment, not a distance-transform watershed: each pixel goes
    # to the growth point it is connected to by the shortest path THROUGH the
    # plant. That is the correct semantics - a leaf belongs to the plant whose
    # crown it physically joins - and it is far more robust on spindly plants,
    # where the distance transform is flat along a thin leaf so a watershed
    # basin boundary lands arbitrarily and can swallow a neighbour's leaves.
    # Implemented as simultaneous multi-label propagation inside the mask.
    k = np.ones((3, 3), np.uint8)
    guard = int(2 * np.hypot(*mask.shape)) + 8       # cannot loop forever
    for _ in range(guard):
        grown = cv2.dilate(labels, k)
        new = (labels == 0) & mask & (grown > 0)
        if not new.any():
            break
        labels[new] = grown[new]

    parts = []
    for i in range(1, len(peaks) + 1):
        part = (labels == i) & mask
        if part.sum() >= cfg["SPLIT_MIN_PART_AREA_PX"]:
            parts.append(part)
    # Require the split to account for most of the plant and to actually produce
    # more than one part; otherwise keep the original rather than lose tissue.
    if len(parts) < 2:
        return [full]
    covered = int(np.logical_or.reduce(parts).sum())
    if covered < cfg["SPLIT_MIN_COVERAGE"] * int(mask.sum()):
        return [full]
    # Paste each part back into full-frame coordinates.
    out = []
    for part in parts:
        f = np.zeros_like(full)
        f[win] = part
        out.append(f)
    return out


def refine_boundary(mask, veg_score, cfg):
    """Snap an instance boundary onto the image's own plant/soil evidence.

    SAM gives excellent structure but its edge can sit a few pixels off the true
    leaf margin - bleeding onto soil, or clipping a thin leaf tip. Only the
    narrow band around the boundary is re-decided, using the continuous
    vegetation score: SAM decides WHAT the object is, the image decides exactly
    where it ends. The interior and the overall shape are never touched.

    Added pixels must stay connected to the original core, so refinement cannot
    absorb a neighbouring plant."""
    band = cfg.get("BOUNDARY_REFINE_BAND_PX", 0)
    if band <= 0 or not mask.any():
        return mask
    win = _bbox_window(mask, band + 2)
    if win is None:
        return mask
    full, mask, veg_score = mask, mask[win], veg_score[win]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    m8 = mask.astype(np.uint8)
    outer = cv2.dilate(m8, k).astype(bool)
    core = cv2.erode(m8, k).astype(bool)
    if not core.any():
        return mask                     # too thin to refine safely
    ring = outer & ~core

    refined = mask.copy()
    refined[ring] = veg_score[ring] >= cfg["BOUNDARY_REFINE_VEG_MIN"]
    refined |= core                     # never eat into the interior

    # Keep only tissue connected to the original core: a refinement must not
    # bridge to a neighbouring plant that happens to be within the band.
    n, lbl = cv2.connectedComponents(refined.astype(np.uint8), 8)
    keep = np.zeros_like(refined)
    for i in range(1, n):
        comp = lbl == i
        if (comp & core).any():
            keep |= comp
    if not keep.any():
        return full
    out = np.zeros_like(full)
    out[win] = keep
    return out


def polygon_epsilon(area_px, cfg):
    """Douglas-Peucker tolerance scaled to instance size.

    A fixed tolerance is wrong at both ends: on a cotyledon seedling it erases
    real shape, and on a large rosette it leaves thousands of near-duplicate
    vertices. Scaling with the square root of area keeps roughly constant
    RELATIVE fidelity, which is what a training target needs."""
    eps = cfg["POLY_APPROX_EPS_FRAC"] * float(np.sqrt(max(1.0, area_px)))
    return float(np.clip(eps, cfg["POLY_APPROX_EPS_MIN"], cfg["POLY_APPROX_EPS_MAX"]))


def mask_polygons(mask, cfg):
    """ALL external contours of an instance as COCO polygons.

    Exporting only the largest contour silently discarded any part of a plant
    separated by an occluding leaf or a gap in the mask - real tissue, dropped
    from the training target. COCO's segmentation field is a list precisely so a
    multi-part instance can be represented, so every part above a small area
    floor is kept."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    eps = polygon_epsilon(int(mask.sum()), cfg)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < cfg["POLY_MIN_PART_AREA_PX"]:
            continue
        a = cv2.approxPolyDP(c, eps, True)
        if len(a) >= 3:
            polys.append(a.reshape(-1).astype(float).tolist())
    return polys


def mask_polygon(mask, eps=None, cfg=None):
    """Backwards-compatible single-polygon helper (largest part only)."""
    cfg = cfg or CONFIG
    polys = mask_polygons(mask, cfg)
    return max(polys, key=len) if polys else None


# --------------------------------------------------------------------------- #
# COCO export
# --------------------------------------------------------------------------- #
class WeedCoco:
    """COCO instance segmentation using the project ontology's stable category
    IDs, so weed, onion and future mixed datasets can be merged without
    remapping any annotation."""

    def __init__(self, classes=None):
        self.images, self.anns = [], []
        names = classes if classes is not None else WEED_CLASSES
        self.categories = coco_categories(names)
        self._cat = {c["name"]: c["id"] for c in self.categories}
        self._img = self._ann = 0

    def add_image(self, file_name, h, w):
        self._img += 1
        self.images.append({"id": self._img, "file_name": file_name,
                            "height": h, "width": w})
        return self._img

    def add_instance(self, image_id, cls, polygon, bbox, area_px=None):
        """polygon may be one flat [x,y,...] list or a list of them, so an
        instance split across an occlusion keeps all of its parts. area is the
        true mask area when given, not the bbox area, since a bbox badly
        overstates a thin or lobed plant."""
        segm = polygon if (polygon and isinstance(polygon[0], list)) else [polygon]
        self._ann += 1
        self.anns.append({"id": self._ann, "image_id": image_id,
                          "category_id": self._cat[cls], "segmentation": segm,
                          "area": float(area_px if area_px is not None
                                        else bbox[2] * bbox[3]),
                          "bbox": [float(v) for v in bbox], "iscrowd": 0})
        return self._ann

    def dump(self, path):
        Path(path).write_text(json.dumps({
            "info": {"description": "SeeWeed3D SAM 3 weed instance prelabels",
                     "date_created": datetime.now(timezone.utc).isoformat()},
            "licenses": [], "images": self.images, "annotations": self.anns,
            "categories": self.categories}, indent=2))


def overlay(bgr, instances, scale):
    """Preview: instance outline, a small class tag, and the proposed LEP dot.

    Text is kept small and only drawn on instances big enough to read, because a
    dense frame otherwise disappears under overlapping labels."""
    colors = CLASS_COLORS_BGR
    vis = bgr.copy()
    fs = max(0.35, min(0.6, bgr.shape[1] / 3000.0))     # scale text to the frame
    for inst in instances:
        col = colors.get(inst["cls"], (170, 170, 170))
        cnts, _ = cv2.findContours(inst["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if inst.get("source") == "vegetation":
            # White halo = recovered by the recall backstop, i.e. a plant SAM
            # did not return. Visible at any instance size, so you can judge
            # from the previews alone whether the backstop is earning its keep.
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 4)
        cv2.drawContours(vis, cnts, -1, col, 2)
        if inst["cls"] == "weed_cluster":
            # No single LEP for a cluster - mark every growth point instead.
            for (px, py), _ in inst.get("peaks", []):
                cv2.drawMarker(vis, (int(px), int(py)), (200, 60, 200),
                               cv2.MARKER_TILTED_CROSS, 14, 2)
        else:
            r = inst.get("lep")
            if r is not None:
                x, y = [int(round(v)) for v in r.uv]
                # Colour encodes visibility, so abstentions are obvious at a
                # glance: green = confident, amber = inferable, red = abstained.
                dot = {"visible": (0, 255, 0),
                       "partially_occluded_inferable": (0, 200, 255)}.get(
                           r.visibility, (0, 0, 255))
                cv2.circle(vis, (x, y), 5, dot, -1)
                cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
                # 1-sigma uncertainty of the fused evidence.
                if r.sigma_px > 1:
                    cv2.circle(vis, (x, y), int(round(r.sigma_px)), dot, 1)
            else:
                x, y = [int(round(v)) for v in inst["points"]["lep_dt"]]
                cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
                cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
        if inst["features"]["area_px"] >= 2500:          # label only if readable
            bx, by = inst["features"]["bbox"][:2]
            cv2.putText(vis, inst["cls"], (bx, max(12, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    return cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else vis


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def link_or_copy(src, dst):
    dst = Path(dst)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def pool_frames(session_dir):
    pool_csv = session_dir / "meta" / "pool.csv"
    if not pool_csv.exists():
        return []
    return [r["filename"] for r in csv.DictReader(open(pool_csv, encoding="utf-8"))
            if r.get("filename")]


def analyze_frame(bgr, sam_masks, cfg, depth_mm=None, estimator=None):
    """Vegetation prior + instance filtering + per-instance analysis.
    Returns (instances, veg_mask). Pure CPU - the SAM call happens outside."""
    veg = vegetation_mask(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                          cfg["VEG_MORPH_KERNEL"], cfg["VEG_MIN_COMPONENT_PX"])
    masks = filter_instances(sam_masks, veg, cfg)
    if estimator is None and cfg.get("USE_FUSED_LEP", True):
        estimator = LEPEstimator()

    # Boundary quality, before anything is measured or exported:
    #  1. snap each edge onto the image's plant/soil evidence,
    #  2. split blobs that contain several growth points into one plant each.
    # Both run here so shape descriptors, the LEP and the polygons are all
    # computed from the corrected instance rather than the raw SAM output.
    score = vegetation_score(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                             cfg.get("VEG_SCORE_SOFTNESS", 0.04))

    # Recall backstop BEFORE refinement, so a plant SAM missed goes through the
    # same boundary treatment as a detected one and is indistinguishable in the
    # output except for its recorded source.
    recovered = recover_missed_plants(veg, masks, cfg)
    sources = ["sam"] * len(masks) + ["vegetation"] * len(recovered)

    refined = []
    for m, src in zip(list(masks) + recovered, sources):
        m = refine_boundary(m, score, cfg)
        if m.sum() < cfg["MIN_INSTANCE_AREA_PX"]:
            continue
        for part in split_touching_instances(m, growth_peaks(m, cfg), cfg):
            refined.append((part, src))

    instances = []
    for m, src in refined:
        f = shape_features(m)
        if f is None:
            continue
        peaks = growth_peaks(m, cfg)
        cls, conf = classify_morphology(f, cfg, peaks)
        is_cluster = cls == "weed_cluster"
        inst = {
            "mask": m, "cls": cls, "cls_confidence": conf, "source": src,
            "features": f, "points": treatment_points(m), "peaks": peaks,
            # A cluster has several growth points, so no single LEP applies.
            "lep_valid": not is_cluster,
            "growth_stage": growth_stage(f["area_px"], cfg)}
        # Multi-evidence LEP: the defensible estimate. The three geometric
        # baselines in inst["points"] are kept alongside it for the plan's
        # LEP-method comparison.
        if estimator is not None and not is_cluster:
            ctx = crop_context(m, bgr, f["bbox"], depth_full=depth_mm,
                               pad=cfg.get("LEP_CROP_PAD_PX", 10), class_name=cls)
            inst["lep"] = estimator.estimate(ctx)
        instances.append(inst)
    return instances, veg


def prelabel_session(sid, session_dir, out_root, cfg, predictor, sam_fn):
    frames = pool_frames(session_dir)
    if cfg["LIMIT_PER_SESSION"]:
        frames = frames[:cfg["LIMIT_PER_SESSION"]]
    if not frames:
        print(f"  [{sid}] no pool frames - run extract_sessions.py first")
        return None

    out = out_root / sid
    cvat_dir, flag_dir = out / cfg["CVAT_READY_SUBDIR"], out / cfg["FLAGGED_RGB_SUBDIR"]
    for d in (cvat_dir, flag_dir, out / "masks"):
        d.mkdir(parents=True, exist_ok=True)
    if cfg["SAVE_PREVIEWS"]:
        (out / "preview").mkdir(parents=True, exist_ok=True)
    if cfg["SAVE_INSTANCE_CROPS"]:
        (out / "crops").mkdir(parents=True, exist_ok=True)

    coco, rows, flagged = WeedCoco(), [], []
    estimator = LEPEstimator() if cfg.get("USE_FUSED_LEP", True) else None
    manual_boxes = cfg["EXEMPLARS"].get(sid)
    stats = {"frames": 0, "instances": 0, "flagged": 0, "empty": 0,
             # Recall bookkeeping: how much of the vegetation prior actually
             # ended up inside an exported instance, and how many instances
             # only exist because the backstop recovered them.
             "recovered": 0, "veg_px": 0, "veg_covered_px": 0}
    per_class = {c: 0 for c in WEED_CLASSES}

    prog = Progress(len(frames), f"[{sid}]", unit="frames")
    for fn in frames:
        rgb_path = session_dir / "rgb" / fn
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            prog.update(note="missing frame")
            continue
        proc = white_balance(bgr, cfg["WB_CAST_RATIO"]) if cfg["WHITE_BALANCE"] else bgr

        veg_pre = vegetation_mask(proc, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                                  cfg["VEG_MORPH_KERNEL"], cfg["VEG_MIN_COMPONENT_PX"])
        if float(veg_pre.mean()) > cfg["MAX_MASK_FRACTION"]:
            # Colour-cast / glare failure: no trustworthy instances here.
            flagged.append(fn)
            link_or_copy(rgb_path, flag_dir / fn)
            stats["frames"] += 1
            stats["flagged"] += 1
            prog.update(note=f"{stats['instances']} instances, "
                             f"{stats['flagged']} flagged")
            continue

        if manual_boxes:
            exemplars = manual_boxes
        elif cfg["SAM_PROMPT_MODE"] == "auto_exemplar":
            exemplars = component_boxes(veg_pre, cfg["EXEMPLAR_MIN_AREA_PX"],
                                        cfg["EXEMPLAR_PAD_PX"], cfg["EXEMPLAR_MAX_BOXES"])
        else:
            exemplars = None

        sam_masks = sam_fn(predictor, proc, cfg, exemplars) if predictor is not None else []
        # Depth, when the session has it, activates the canopy-height evidence
        # channel: the crown sits above the surrounding soil and leaves.
        depth_mm = None
        if cfg.get("USE_DEPTH_FOR_LEP", True):
            dpath = session_dir / "depth" / fn
            if dpath.exists():
                raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
                if raw is not None and raw.dtype == np.uint16:
                    depth_mm = raw.astype(np.float32)
                    depth_mm[raw == 0] = np.nan          # 0 is the invalid sentinel
        instances, veg = analyze_frame(proc, sam_masks, cfg, depth_mm, estimator)

        link_or_copy(rgb_path, cvat_dir / fn)
        img_id = coco.add_image(fn, bgr.shape[0], bgr.shape[1])
        union = np.zeros(bgr.shape[:2], bool)
        for k, inst in enumerate(instances):
            polys = mask_polygons(inst["mask"], cfg)
            if not polys:
                continue
            coco.add_instance(img_id, inst["cls"], polys, inst["features"]["bbox"],
                              area_px=inst["features"]["area_px"])
            union |= inst["mask"]
            per_class[inst["cls"]] += 1
            p, f = inst["points"], inst["features"]
            lep_row = inst["lep"].as_row("lep") if inst.get("lep") else {}
            rows.append({
                **lep_row,
                "session_id": sid, "filename": fn, "instance_idx": k,
                "class": inst["cls"], "class_confidence": inst["cls_confidence"],
                # "sam" or "vegetation" (recall backstop) - keeps the two
                # populations separable when auditing or weighting the data.
                "source": inst.get("source", "sam"),
                "growth_stage": inst["growth_stage"],
                "n_growth_peaks": len(inst.get("peaks", [])),
                "lep_valid": int(inst.get("lep_valid", True)),
                "lep_dt_x": p["lep_dt"][0], "lep_dt_y": p["lep_dt"][1],
                "dt_radius_px": p["dt_radius_px"],
                "centroid_x": p["centroid"][0], "centroid_y": p["centroid"][1],
                "bbox_ctr_x": p["bbox_ctr"][0], "bbox_ctr_y": p["bbox_ctr"][1],
                "lep_centroid_dist_px": p["lep_centroid_dist_px"],
                **{k2: f[k2] for k2 in ("area_px", "perimeter_px", "solidity",
                                        "circularity", "aspect_ratio", "extent")}})
            if cfg["SAVE_INSTANCE_CROPS"]:
                x, y, w, h = f["bbox"]
                pad = 8
                crop = proc[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                if crop.size:
                    cv2.imwrite(str(out / "crops" /
                                    f"{Path(fn).stem}_i{k:03d}_{inst['cls'].split()[-1]}.jpg"),
                                crop)

        cv2.imwrite(str(out / "masks" / fn), (union.astype(np.uint8) * 255))
        if cfg["SAVE_PREVIEWS"]:
            cv2.imwrite(str(out / "preview" / fn.replace(".png", ".jpg")),
                        overlay(proc, instances, cfg["PREVIEW_SCALE"]))
        stats["frames"] += 1
        stats["instances"] += len(instances)
        stats["empty"] += int(not instances)
        stats["recovered"] += sum(1 for i in instances
                                  if i.get("source") == "vegetation")
        # Measured against the EXPORTED union, so this reports what an annotator
        # will actually see, not what was computed and then dropped.
        stats["veg_px"] += int(veg.sum())
        stats["veg_covered_px"] += int((veg & union).sum())
        prog.update(note=f"{stats['instances']} instances, "
                         f"{stats['flagged']} flagged")

    prog.close(note=f"{stats['instances']} instances, {stats['flagged']} flagged")
    coco.dump(out / "instances_default.json")
    (out / "weed_cvat_labels.json").write_text(json.dumps(weed_cvat_labels(), indent=2))
    if flagged:
        (out / "flagged_for_manual.txt").write_text("\n".join(flagged))
    if rows:
        # Header must be the UNION of every row's keys, not the first row's:
        # a weed_cluster instance carries no lep_* columns at all, so taking the
        # header from row 0 crashes as soon as the first instance is a cluster.
        # restval fills the blanks for rows that legitimately lack a column.
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
        for k in LEPResult.row_fields("lep"):        # keep LEP columns present
            if k not in seen:
                seen.add(k)
                fields.append(k)
        with open(out / "instances.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)

    # Method provenance travels with the data, so any result can be traced back
    # to exactly which LEP evidence model produced it.
    if estimator is not None:
        (out / "lep_method.json").write_text(json.dumps(estimator.describe(), indent=2))

    dist = " ".join(f"{c.split()[-1]}={n}" for c, n in per_class.items() if n)
    print(f"  [{sid}] {stats['frames']} frames | {stats['instances']} weed instances "
          f"| {stats['flagged']} flagged | {stats['empty']} with no instances")
    print(f"      provisional classes: {dist or 'none'}")
    # Recall readout. In a weed-only scene vegetation IS plants, so vegetation
    # left outside every exported instance is, to a first approximation, weeds
    # the annotator will never be shown. Watch this number, not the instance
    # count: a pipeline can look productive while quietly missing plants.
    if stats["veg_px"]:
        cov = 100.0 * stats["veg_covered_px"] / stats["veg_px"]
        sam_n = stats["instances"] - stats["recovered"]
        print(f"      recall: {cov:.1f}% of vegetation inside an exported "
              f"instance | {sam_n} from SAM + {stats['recovered']} recovered")
    vis_rows = [r for r in rows if r.get("lep_visibility")]
    if vis_rows:
        conf = [float(r["lep_confidence"]) for r in vis_rows]
        agree = [float(r["lep_agreement_px"]) for r in vis_rows]
        counts = {}
        for r in vis_rows:
            counts[r["lep_visibility"]] = counts.get(r["lep_visibility"], 0) + 1
        print(f"      LEP: median confidence {np.median(conf):.2f} | median "
              f"channel agreement {np.median(agree):.1f}px | "
              + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"      -> {out}")
    return stats


def main(predictor_factory=load_sam3, sam_fn=sam3_instances):
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"])
    sessions_root = root / "sessions"
    if not sessions_root.exists():
        sys.exit(f"ERROR: {sessions_root} not found. Run extract_sessions.py first.")
    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    if cfg["ONLY_SESSIONS"]:
        sids = [s for s in sids if s in cfg["ONLY_SESSIONS"]]
    if not sids:
        sys.exit("No sessions selected. Set ONLY_SESSIONS to your weed-only sessions.")

    print(f"Weed instance prelabeling on {len(sids)} session(s) with SAM 3.")
    print("  Reminder: point this at WEED-ONLY recordings only.\n")
    predictor = predictor_factory(cfg)
    out_root = root / cfg["OUTPUT_SUBDIR"]
    for sid in sids:
        prelabel_session(sid, sessions_root / sid, out_root, cfg, predictor, sam_fn)

    print(f"\nDone -> {out_root}")
    print("Next: CVAT task from <sid>/cvat_ready/, paste <sid>/weed_cvat_labels.json "
          "into the Raw label editor, import <sid>/instances_default.json (COCO 1.0), "
          "then CORRECT the class of each instance and drag the LEP points. "
          "instances.csv carries the proposed LEP/centroid/bbox-centre per instance.")


if __name__ == "__main__":
    main()
