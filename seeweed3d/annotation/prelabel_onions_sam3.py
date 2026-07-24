#!/usr/bin/env python3
"""
SeeWeed3D - Stage 3: Onion prelabeling with SAM 3  (ONION-ONLY scenes)
=====================================================================
Auto-generates a high-recall onion "safety mask" for every pooled frame and
writes it as CVAT-importable COCO, so annotators VERIFY masks instead of
drawing them. Meant for onion-only recordings, where the hard onion-vs-weed
decision does not exist and green vegetation can be treated as onion.

    Run extraction/extract_sessions.py first (this reads its output pool).

WHY THIS IS SAFE HERE, AND ONLY HERE
------------------------------------
In a mixed scene you must NEVER assume vegetation == onion: one missed onion
becomes a dangerous weed label. In an onion-only field there are no weeds to
confuse, so the vegetation prior is a legitimate, high-recall onion signal. Do
NOT point this script at mixed or weed scenes.

THE TECHNIQUE (see also docs)
-----------------------------
1. SAM 3 concept segmentation ("onion" text, or image-exemplar boxes) gives
   clean masks on thin, crossing leaves and returns every instance at once.
2. An Excess-Green (ExG) vegetation prior validates SAM masks (drops anything
   not sitting on green tissue) and recovers onion tissue SAM missed.
3. The fused per-frame mask is a SEMANTIC onion safety mask (coverage over
   instance separation, exactly what the laser must avoid), exported as
   polygons under the "onion plant" label for correction in CVAT.

Everything except the SAM 3 call is plain OpenCV/NumPy and is unit-testable
without a GPU. The SAM 3 call is isolated in load_sam3()/sam3_masks().

SAM 3 BACKEND
-------------
Uses Meta's official `sam3` package (github.com/facebookresearch/sam3), which
loads a `.pt` checkpoint and supports BOTH SAM 3 and the faster SAM 3.1. This
avoids depending on a transformers release that bundles SAM 3. Install once:

    pip install "git+https://github.com/facebookresearch/sam3.git"
"""

import contextlib
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# #############################################################################
# ##   DATASET_ROOT  -  the OUTPUT_ROOT you gave extract_sessions.py          ##
# ##   SAM_VERSION   -  "sam3" or "sam3.1" (the faster variant)               ##
# ##   SAM_CHECKPOINT-  path to a local .pt (sam3.pt / sam3.1_multiplex.pt),  ##
# ##                    or None to auto-download SAM_VERSION from Hugging Face ##
# ##                    (needs `huggingface-cli login` for the gated repo).    ##
# #############################################################################

DATASET_ROOT   = r"E:\Dataset_Vidalia"
SAM_VERSION    = "sam3.1"        # "sam3" | "sam3.1"
# e.g. r"C:\Users\mm17889\models\sam3\sam3.1_multiplex.pt". None => auto-download.
SAM_CHECKPOINT = None

# =============================================================================
# CONFIG - advanced tuning below; defaults are sensible for onion-only scenes
# =============================================================================

CONFIG = {
    "DATASET_ROOT":   DATASET_ROOT,
    "SAM_VERSION":    SAM_VERSION,
    "SAM_CHECKPOINT": SAM_CHECKPOINT,
    "OUTPUT_SUBDIR": "auto_labels_onion",   # written under DATASET_ROOT/

    # Which sessions to prelabel. Empty = every session found under sessions/.
    # These MUST be onion-only recordings.
    "ONLY_SESSIONS": [],

    # -- SAM 3 prompting -------------------------------------------------------
    # Text concepts are the zero-config default. Image exemplars (bboxes on a
    # reference frame) are more reliable for a specific crop; add them per
    # session in EXEMPLARS if text under-segments. Boxes are [x1,y1,x2,y2] px.
    "SAM_TEXT_PROMPTS": ["onion", "onion plant", "green onion leaves"],
    "EXEMPLARS": {
        # "vid3_20260108_132749": [[900, 500, 1050, 780], [1200, 300, 1350, 560]],
    },
    "SAM_CONF": 0.25,        # SAM 3 confidence threshold (detections below dropped)
    "DEVICE": "cuda",        # "cuda" | "cpu"

    # -- Vegetation prior (Excess-Green) --------------------------------------
    "EXG_THRESHOLD": 0.05,   # exg > this = vegetation. Lower = more permissive.
    "VEG_MORPH_KERNEL": 3,   # close/open kernel px to tidy the veg mask
    "VEG_MIN_COMPONENT_PX": 80,   # drop veg specks smaller than this

    # -- Fusion ----------------------------------------------------------------
    # A SAM mask is accepted only if this fraction of it overlaps vegetation
    # (kills SAM masks that grabbed soil / background).
    "SAM_VEG_OVERLAP_MIN": 0.30,
    # Recover onion tissue SAM missed: veg-only components >= this area are added
    # to the final mask. Keeps recall high without importing ExG noise.
    "RECOVER_VEG_MIN_PX": 400,

    # -- Polygon export --------------------------------------------------------
    "POLY_MIN_AREA_PX": 120,     # skip tiny polygons (noise, single leaf tips)
    "POLY_APPROX_EPS": 1.5,      # Douglas-Peucker simplification (px)
    "MERGE_INTO_ONE_MASK": False,  # False = one polygon per leaf clump (editable)

    # -- Run control -----------------------------------------------------------
    "LIMIT_PER_SESSION": None,   # e.g. 20 for a quick quality trial, then None
    "SAVE_PREVIEWS": True,       # overlay JPGs for fast eyeballing / FiftyOne
    "PREVIEW_SCALE": 0.5,
}

ONION_LABEL = "onion plant"     # must match cvat_labels.json / project ontology

# =============================================================================


# --------------------------------------------------------------------------- #
# SAM 3 - the only GPU-dependent part, isolated so the rest is testable.
# Uses Meta's official `sam3` package (build_sam3_image_model + Sam3Processor),
# which loads a .pt checkpoint and supports SAM 3 and SAM 3.1. Imports are lazy
# so this module loads (and the OpenCV logic is testable) without sam3/torch.
# --------------------------------------------------------------------------- #
def load_sam3(cfg):
    """Load SAM 3 / 3.1 via the official sam3 package. Returns a handle dict."""
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    want = cfg["DEVICE"]
    device = "cuda" if (want.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if want.startswith("cuda") and device == "cpu":
        print("[WARN] CUDA requested but not available - running SAM 3 on CPU (slow).")

    ckpt = cfg["SAM_CHECKPOINT"]
    if ckpt is None and cfg["SAM_VERSION"] != "sam3":
        # Auto-fetch a non-default version (e.g. sam3.1) by its HF checkpoint.
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


_SHAPE_LOGGED = False


def _state_masks(state):
    """Pull binary masks out of a processor state dict as a list of 2-D bool
    arrays, whatever nesting SAM used ([N,H,W], [N,1,H,W], [H,W], ...)."""
    global _SHAPE_LOGGED
    masks = state.get("masks") if isinstance(state, dict) else None
    if masks is None:
        return []
    arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if not _SHAPE_LOGGED:
        print(f"[DEBUG] SAM raw masks: shape={getattr(arr, 'shape', None)} "
              f"dtype={getattr(arr, 'dtype', None)}")
        _SHAPE_LOGGED = True
    arr = np.squeeze(arr)                 # drop singleton dims, e.g. [N,1,H,W]->[N,H,W]
    if arr.ndim == 2:                     # a single mask -> add batch dim
        arr = arr[None]
    if arr.ndim != 3:
        return []
    return [arr[i].astype(bool) for i in range(arr.shape[0])
            if arr[i].ndim == 2 and arr[i].size > 0]


def sam3_masks(predictor, image_path, cfg, exemplars=None):
    """Return boolean HxW onion masks for one image.

    With exemplars (positive bboxes on this session's frames) SAM 3 is prompted
    visually; otherwise each text concept is run and their masks are unioned for
    recall. Returns [] if SAM produced nothing (caller falls back to ExG)."""
    from PIL import Image
    torch = predictor["torch"]
    processor, device = predictor["processor"], predictor["device"]
    image = Image.open(str(image_path)).convert("RGB")
    w, h = image.size

    # SAM 3's backbone mixes bf16 activations with fp32 weights, which raises
    # "mat1 and mat2 must have the same dtype" outside autocast. Autocast casts
    # per-op so the two always agree; on CPU there is nothing to reconcile.
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else contextlib.nullcontext())

    out = []
    with amp:
        state = processor.set_image(image)      # backbone features computed once
        if exemplars:
            # add_geometric_prompt wants normalized [cx, cy, w, h]; config is xyxy px.
            processor.reset_all_prompts(state)
            for b in exemplars:
                x1, y1, x2, y2 = map(float, b)
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
# Vegetation prior + fusion  (pure OpenCV/NumPy - testable without a GPU)
# --------------------------------------------------------------------------- #
def vegetation_mask(bgr, cfg):
    """Excess-Green vegetation mask. In an onion-only scene this is the onion
    tissue: soil is brown (low ExG), so it is not a green-vs-anything guess."""
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    s = b + g + r + 1e-6
    exg = 2 * (g / s) - (r / s) - (b / s)
    veg = (exg > cfg["EXG_THRESHOLD"]).astype(np.uint8)
    k = cfg["VEG_MORPH_KERNEL"]
    if k > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, ker)
        veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, ker)
    return remove_small(veg.astype(bool), cfg["VEG_MIN_COMPONENT_PX"])


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


def fuse(sam_list, veg, cfg):
    """Combine SAM masks (clean boundaries) with the vegetation prior (recall).

    - Keep only SAM masks that sit mostly on vegetation.
    - Union them, then add substantial veg regions SAM missed.
    - If SAM produced nothing usable, fall back to the vegetation mask so the
      frame still gets a label rather than being silently dropped.
    Returns (final_bool_mask, stats_dict).
    """
    h, w = int(veg.shape[0]), int(veg.shape[1])
    sam_union = np.zeros((h, w), bool)
    kept = 0
    for m in sam_list:
        m = np.asarray(m)
        if m.ndim != 2 or 0 in m.shape:        # skip degenerate masks
            continue
        if m.shape != veg.shape:
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(m.sum())
        if area == 0:
            continue
        overlap = float((m & veg).sum()) / area
        if overlap >= cfg["SAM_VEG_OVERLAP_MIN"]:
            sam_union |= m
            kept += 1

    if kept == 0:
        # No trustworthy SAM mask: vegetation prior is the safest fallback.
        return veg.copy(), {"sam_masks_in": len(sam_list), "sam_kept": 0,
                            "fallback_veg_only": True}

    missed = remove_small(veg & ~sam_union, cfg["RECOVER_VEG_MIN_PX"])
    final = sam_union | missed
    return final, {"sam_masks_in": len(sam_list), "sam_kept": kept,
                   "fallback_veg_only": False,
                   "recovered_veg_px": int(missed.sum())}


# --------------------------------------------------------------------------- #
# COCO export
# --------------------------------------------------------------------------- #
def mask_to_polygons(mask, cfg):
    """Boolean mask -> list of COCO polygons ([x1,y1,x2,y2,...]). One polygon
    per external contour of each clump, simplified and area-filtered."""
    polys = []
    m = mask.astype(np.uint8)
    if cfg["MERGE_INTO_ONE_MASK"]:
        comps = [m]
    else:
        n, lbl = cv2.connectedComponents(m, 8)
        comps = [(lbl == i).astype(np.uint8) for i in range(1, n)]
    for comp in comps:
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < cfg["POLY_MIN_AREA_PX"]:
                continue
            approx = cv2.approxPolyDP(c, cfg["POLY_APPROX_EPS"], True)
            if len(approx) < 3:
                continue
            polys.append(approx.reshape(-1).astype(float).tolist())
    return polys


class Coco:
    """Minimal COCO instance-segmentation writer for CVAT import."""

    def __init__(self):
        self.images, self.anns = [], []
        self.categories = [{"id": 1, "name": ONION_LABEL, "supercategory": ""}]
        self._img, self._ann = 0, 0

    def add(self, file_name, h, w, polygons):
        self._img += 1
        self.images.append({"id": self._img, "file_name": file_name,
                            "height": h, "width": w})
        for poly in polygons:
            xs, ys = poly[0::2], poly[1::2]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            self._ann += 1
            self.anns.append({"id": self._ann, "image_id": self._img,
                              "category_id": 1, "segmentation": [poly],
                              "area": float(bbox[2] * bbox[3]), "bbox": bbox,
                              "iscrowd": 0})
        return self._img

    def dump(self, path):
        path.write_text(json.dumps(
            {"info": {"description": "SeeWeed3D SAM3 onion prelabels",
                      "date_created": datetime.now(timezone.utc).isoformat()},
             "licenses": [], "images": self.images,
             "annotations": self.anns, "categories": self.categories}, indent=2))


def overlay(bgr, mask, scale):
    vis = bgr.copy()
    tint = np.zeros_like(vis); tint[mask] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.4, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 200, 255), 2)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return vis


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def pool_frames(session_dir):
    pool_csv = session_dir / "meta" / "pool.csv"
    if not pool_csv.exists():
        return []
    return [r["filename"] for r in csv.DictReader(open(pool_csv, encoding="utf-8"))
            if r.get("filename")]


def prelabel_session(sid, session_dir, out_root, cfg, predictor, sam_fn):
    frames = pool_frames(session_dir)
    if cfg["LIMIT_PER_SESSION"]:
        frames = frames[:cfg["LIMIT_PER_SESSION"]]
    if not frames:
        print(f"  [{sid}] no pool frames - run stage 1 first"); return None

    out = out_root / sid
    (out / "masks").mkdir(parents=True, exist_ok=True)
    if cfg["SAVE_PREVIEWS"]:
        (out / "preview").mkdir(parents=True, exist_ok=True)
    coco = Coco()
    exemplars = cfg["EXEMPLARS"].get(sid)
    stats = {"frames": 0, "fallback": 0, "empty": 0, "polys": 0}

    for fn in frames:
        rgb_path = session_dir / "rgb" / fn
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            print(f"      ! missing {rgb_path}"); continue
        veg = vegetation_mask(bgr, cfg)
        sam = sam_fn(predictor, rgb_path, cfg, exemplars) if predictor is not None else []
        final, fstat = fuse(sam, veg, cfg)
        polys = mask_to_polygons(final, cfg)

        cv2.imwrite(str(out / "masks" / fn), (final.astype(np.uint8) * 255))
        if cfg["SAVE_PREVIEWS"]:
            cv2.imwrite(str(out / "preview" / fn.replace(".png", ".jpg")),
                        overlay(bgr, final, cfg["PREVIEW_SCALE"]))
        coco.add(fn, bgr.shape[0], bgr.shape[1], polys)
        stats["frames"] += 1
        stats["fallback"] += int(fstat.get("fallback_veg_only", False))
        stats["empty"] += int(not final.any())
        stats["polys"] += len(polys)

    coco.dump(out / "instances_default.json")
    print(f"  [{sid}] {stats['frames']} frames | {stats['polys']} onion polygons "
          f"| {stats['fallback']} veg-fallback | {stats['empty']} empty "
          f"-> {out}")
    return stats


def main(predictor_factory=load_sam3, sam_fn=sam3_masks):
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"])
    sessions_root = root / "sessions"
    if not sessions_root.exists():
        sys.exit(f"ERROR: {sessions_root} not found. Run extract_sessions.py first.")

    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    if cfg["ONLY_SESSIONS"]:
        sids = [s for s in sids if s in cfg["ONLY_SESSIONS"]]
    if not sids:
        sys.exit("No sessions to prelabel. Check DATASET_ROOT / ONLY_SESSIONS.")

    print(f"Prelabeling {len(sids)} onion-only session(s) with SAM 3.")
    print("  Reminder: point this at ONION-ONLY recordings only.\n")
    predictor = predictor_factory(cfg)
    out_root = root / cfg["OUTPUT_SUBDIR"]
    for sid in sids:
        prelabel_session(sid, sessions_root / sid, out_root, cfg, predictor, sam_fn)

    print(f"\nDone. COCO + masks + previews under {out_root}")
    print("Next: create a CVAT task from sessions/<sid>/rgb/, import that "
          "session's instances_default.json (COCO 1.0), VERIFY, then export "
          "the corrected masks as your training labels.")


if __name__ == "__main__":
    main()
