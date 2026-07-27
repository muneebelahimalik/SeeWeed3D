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
                               vegetation_mask, white_balance)

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
    "EXEMPLAR_MIN_AREA_PX": 300,
    "EXEMPLAR_MAX_BOXES": 30,
    "EXEMPLAR_PAD_PX": 8,
    "SAM_CONF": 0.25,
    "DEVICE": "cuda",

    # -- Instance filtering ----------------------------------------------------
    "MIN_INSTANCE_AREA_PX": 250,     # below this is noise / unlabelable speck
    "MAX_INSTANCE_FRAC": 0.25,       # one weed covering >25% of frame = failure
    "INSTANCE_VEG_OVERLAP_MIN": 0.35,  # instance must sit on vegetation
    "NMS_IOU": 0.65,                 # de-duplicate overlapping SAM instances

    # -- Morphology heuristic (priors, calibrate after round 1) ----------------
    # ELONGATION is the discriminator, measured on synthetic + field shapes:
    # a grass blade has aspect ~20 and a rosette ~1. Circularity is NOT usable -
    # a rosette's radiating leaves give it a long, spiky perimeter and therefore
    # low circularity, much like a blade. Solidity is not usable for grass
    # either: a straight blade is nearly its own convex hull (solidity ~0.9).
    # Anything between the two aspect bands stays "unknown" on purpose.
    "GRASS_MIN_ASPECT": 3.0,          # >= this -> grass-like
    "BROADLEAF_MAX_ASPECT": 2.2,      # <= this (and solid enough) -> rosette
    "BROADLEAF_MIN_SOLIDITY": 0.30,

    # -- Growth stage from instance area (px). Calibrate to your mount height. -
    "STAGE_COTYLEDON_MAX_PX": 1200,
    "STAGE_EARLY_MAX_PX": 6000,

    # -- Frame-level safety ----------------------------------------------------
    "MAX_MASK_FRACTION": 0.5,        # total veg > this = colour-cast/glare failure

    # -- Polygon export --------------------------------------------------------
    "POLY_APPROX_EPS": 1.5,

    # -- Run control -----------------------------------------------------------
    "LIMIT_PER_SESSION": 20,         # start small; set None for the full pool
    "SAVE_PREVIEWS": True,
    "PREVIEW_SCALE": 0.5,
}

# Morphology classes, per the project ontology (plan section 10). "sedge" is
# never auto-assigned - it is not reliably separable from shape alone at this
# growth stage, so it is left for the annotator.
WEED_CLASSES = ["weed broadleaf", "weed grass", "weed sedge", "weed unknown"]

# =============================================================================


def weed_cvat_labels():
    """CVAT label schema for weed verification (paste into the Raw editor)."""
    attrs = [
        {"name": "growth_stage", "input_type": "select", "mutable": True,
         "values": ["cotyledon", "2-leaf", "3-5-leaf", "later", "unknown"],
         "default_value": "unknown"},
        {"name": "lep_visibility", "input_type": "select", "mutable": True,
         "values": ["visible", "partially_occluded_inferable", "not_visible"],
         "default_value": "visible"},
        {"name": "targetable", "input_type": "select", "mutable": True,
         "values": ["yes", "no", "uncertain"], "default_value": "yes"},
        {"name": "difficulty", "input_type": "select", "mutable": True,
         "values": ["normal", "overlapping", "blurred", "shadowed", "wet", "truncated"],
         "default_value": "normal"},
        {"name": "species", "input_type": "text", "mutable": True, "default_value": ""},
    ]
    colors = {"weed broadleaf": "#ff6037", "weed grass": "#ffcc00",
              "weed sedge": "#8a2be2", "weed unknown": "#aaaaaa"}
    labels = [{"name": c, "type": "polygon", "color": colors[c],
               "attributes": list(attrs)} for c in WEED_CLASSES]
    labels += [
        {"name": "weed LEP", "type": "points", "color": "#fffc00", "attributes": [
            {"name": "lep_visibility", "input_type": "select", "mutable": True,
             "values": ["visible", "partially_occluded_inferable"],
             "default_value": "visible"}]},
        {"name": "ambiguous cluster", "type": "polygon", "color": "#8a2be2",
         "attributes": []},
        {"name": "ignore region", "type": "polygon", "color": "#000000",
         "attributes": []},
    ]
    return labels


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


def classify_morphology(f, cfg):
    """Provisional morphology class from shape, driven by ELONGATION.

    Measured on synthetic and field shapes: a grass blade has aspect ~20 while a
    rosette has ~1. Circularity cannot separate them (a rosette's radiating
    leaves make its perimeter long and spiky, so its circularity is as low as a
    blade's), and solidity cannot flag grass (a straight blade is nearly its own
    convex hull). So aspect ratio decides, with solidity only used to confirm a
    compact plant is a real rosette rather than a scattered fragment.

    Deliberately conservative: anything in the ambiguous middle band stays
    'weed unknown' so the annotator is never nudged toward a confident wrong
    label. Confidence scales with distance past the threshold."""
    if f is None:
        return "weed unknown", 0.0
    ar, sol = f["aspect_ratio"], f["solidity"]
    if ar >= cfg["GRASS_MIN_ASPECT"]:
        margin = min(1.0, (ar / max(1e-6, cfg["GRASS_MIN_ASPECT"]) - 1.0))
        return "weed grass", round(0.5 + 0.5 * margin, 3)
    if ar <= cfg["BROADLEAF_MAX_ASPECT"] and sol >= cfg["BROADLEAF_MIN_SOLIDITY"]:
        margin = min(1.0, (cfg["BROADLEAF_MAX_ASPECT"] - ar) /
                     max(1e-6, cfg["BROADLEAF_MAX_ASPECT"]))
        return "weed broadleaf", round(0.5 + 0.5 * margin, 3)
    return "weed unknown", 0.0


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


def mask_polygon(mask, eps):
    """Largest external contour as a flat COCO polygon [x1,y1,x2,y2,...]."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), eps, True)
    return c.reshape(-1).astype(float).tolist() if len(c) >= 3 else None


# --------------------------------------------------------------------------- #
# COCO export
# --------------------------------------------------------------------------- #
class WeedCoco:
    """COCO instance segmentation with one category per morphology class."""

    def __init__(self):
        self.images, self.anns = [], []
        self.categories = [{"id": i + 1, "name": n, "supercategory": "weed"}
                           for i, n in enumerate(WEED_CLASSES)]
        self._cat = {n: i + 1 for i, n in enumerate(WEED_CLASSES)}
        self._img = self._ann = 0

    def add_image(self, file_name, h, w):
        self._img += 1
        self.images.append({"id": self._img, "file_name": file_name,
                            "height": h, "width": w})
        return self._img

    def add_instance(self, image_id, cls, polygon, bbox):
        self._ann += 1
        self.anns.append({"id": self._ann, "image_id": image_id,
                          "category_id": self._cat[cls], "segmentation": [polygon],
                          "area": float(bbox[2] * bbox[3]), "bbox": [float(v) for v in bbox],
                          "iscrowd": 0})
        return self._ann

    def dump(self, path):
        Path(path).write_text(json.dumps({
            "info": {"description": "SeeWeed3D SAM 3 weed instance prelabels",
                     "date_created": datetime.now(timezone.utc).isoformat()},
            "licenses": [], "images": self.images, "annotations": self.anns,
            "categories": self.categories}, indent=2))


def overlay(bgr, instances, scale):
    """Preview: instance outline + class + proposed LEP dot."""
    colors = {"weed broadleaf": (60, 90, 255), "weed grass": (0, 210, 255),
              "weed sedge": (200, 60, 200), "weed unknown": (170, 170, 170)}
    vis = bgr.copy()
    for inst in instances:
        col = colors.get(inst["cls"], (170, 170, 170))
        cnts, _ = cv2.findContours(inst["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 2)
        x, y = [int(round(v)) for v in inst["points"]["lep_dt"]]
        cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
        cv2.circle(vis, (x, y), 7, (0, 0, 0), 1)
        cv2.putText(vis, inst["cls"].replace("weed ", ""), (x + 9, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
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


def analyze_frame(bgr, sam_masks, cfg):
    """Vegetation prior + instance filtering + per-instance analysis.
    Returns (instances, veg_mask). Pure CPU - the SAM call happens outside."""
    veg = vegetation_mask(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                          cfg["VEG_MORPH_KERNEL"], cfg["VEG_MIN_COMPONENT_PX"])
    masks = filter_instances(sam_masks, veg, cfg)
    instances = []
    for m in masks:
        f = shape_features(m)
        if f is None:
            continue
        cls, conf = classify_morphology(f, cfg)
        instances.append({
            "mask": m, "cls": cls, "cls_confidence": conf,
            "features": f, "points": treatment_points(m),
            "growth_stage": growth_stage(f["area_px"], cfg)})
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
    manual_boxes = cfg["EXEMPLARS"].get(sid)
    stats = {"frames": 0, "instances": 0, "flagged": 0, "empty": 0}
    per_class = {c: 0 for c in WEED_CLASSES}

    for fn in frames:
        rgb_path = session_dir / "rgb" / fn
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            print(f"      ! missing {rgb_path}")
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
            continue

        if manual_boxes:
            exemplars = manual_boxes
        elif cfg["SAM_PROMPT_MODE"] == "auto_exemplar":
            exemplars = component_boxes(veg_pre, cfg["EXEMPLAR_MIN_AREA_PX"],
                                        cfg["EXEMPLAR_PAD_PX"], cfg["EXEMPLAR_MAX_BOXES"])
        else:
            exemplars = None

        sam_masks = sam_fn(predictor, proc, cfg, exemplars) if predictor is not None else []
        instances, _ = analyze_frame(proc, sam_masks, cfg)

        link_or_copy(rgb_path, cvat_dir / fn)
        img_id = coco.add_image(fn, bgr.shape[0], bgr.shape[1])
        union = np.zeros(bgr.shape[:2], bool)
        for k, inst in enumerate(instances):
            poly = mask_polygon(inst["mask"], cfg["POLY_APPROX_EPS"])
            if poly is None:
                continue
            coco.add_instance(img_id, inst["cls"], poly, inst["features"]["bbox"])
            union |= inst["mask"]
            per_class[inst["cls"]] += 1
            p, f = inst["points"], inst["features"]
            rows.append({
                "session_id": sid, "filename": fn, "instance_idx": k,
                "class": inst["cls"], "class_confidence": inst["cls_confidence"],
                "growth_stage": inst["growth_stage"],
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

    coco.dump(out / "instances_default.json")
    (out / "weed_cvat_labels.json").write_text(json.dumps(weed_cvat_labels(), indent=2))
    if flagged:
        (out / "flagged_for_manual.txt").write_text("\n".join(flagged))
    if rows:
        with open(out / "instances.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dist = " ".join(f"{c.split()[-1]}={n}" for c, n in per_class.items() if n)
    print(f"  [{sid}] {stats['frames']} frames | {stats['instances']} weed instances "
          f"| {stats['flagged']} flagged | {stats['empty']} with no instances")
    print(f"      provisional classes: {dist or 'none'}")
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
