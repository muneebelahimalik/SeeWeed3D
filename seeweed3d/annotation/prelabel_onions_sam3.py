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
1. An Excess-Green (ExG) vegetation prior finds the onion tissue and its blobs
   become SAM 3 exemplar boxes (the text concept "onion" does not ground on
   top-down field imagery, so exemplars are the default - see SAM_PROMPT_MODE).
2. SAM 3 segments from those exemplars for clean boundaries on thin, crossing
   leaves; the ExG prior then validates SAM masks and recovers tissue SAM missed.
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
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CROP_CLASS  # noqa: E402
from common.progress import Progress  # noqa: E402
from common.vegetation import component_boxes, remove_small  # noqa: E402
from common.vegetation import vegetation_mask as _vegetation_mask  # noqa: E402
from common.vegetation import white_balance as _white_balance  # noqa: E402

# #############################################################################
# ##   DATASET_ROOT  -  the OUTPUT_ROOT you gave extract_sessions.py          ##
# ##   SAM_VERSION   -  "sam3" or "sam3.1" (the faster variant)               ##
# ##   SAM_CHECKPOINT-  path to a local .pt (sam3.pt / sam3.1_multiplex.pt),  ##
# ##                    or None to auto-download SAM_VERSION from Hugging Face ##
# ##                    (needs `huggingface-cli login` for the gated repo).    ##
# #############################################################################

DATASET_ROOT   = r"E:\Dataset_Vidalia"
SAM_VERSION    = "sam3"        # "sam3" | "sam3.1"
# e.g. r"C:\Users\mm17889\models\sam3\sam3.1_multiplex.pt". None => auto-download.
SAM_CHECKPOINT = r"E:\Models\sam3.pt"

# =============================================================================
# CONFIG - advanced tuning below; defaults are sensible for onion-only scenes
# =============================================================================

CONFIG = {
    "DATASET_ROOT":   DATASET_ROOT,
    "SAM_VERSION":    SAM_VERSION,
    "SAM_CHECKPOINT": SAM_CHECKPOINT,
    "OUTPUT_SUBDIR": "auto_labels_onion",   # written under DATASET_ROOT/
    # Per-session subfolders (under OUTPUT_SUBDIR/<sid>/) so the frame set you
    # upload to CVAT never needs manual filtering:
    #   CVAT_READY_SUBDIR  - frames WITH a usable prelabel. instances_default.json
    #                        covers exactly these frames, so uploading this folder
    #                        + importing that COCO always matches with no errors.
    #   FLAGGED_RGB_SUBDIR - frames blanked by the MAX_MASK_FRACTION safety cap
    #                        (see flagged_for_manual.txt). No auto-label exists for
    #                        these; upload this folder as its own task for a
    #                        purely manual pass, kept separate from the main set.
    "CVAT_READY_SUBDIR":  "cvat_ready",
    "FLAGGED_RGB_SUBDIR": "flagged_rgb",

    # Which sessions to prelabel. Empty = every session found under sessions/.
    # These MUST be onion-only recordings.
    "ONLY_SESSIONS": [],

    # -- Preprocessing ---------------------------------------------------------
    # Gray-world white balance neutralises a colour-cast before segmentation.
    # Some ZED frames have a strong green white-balance error - the whole frame
    # reads green - which would otherwise be flagged and lost. Soil-dominant
    # frames are barely changed; green-cast frames are recovered so ExG and SAM
    # work on them. Only kicks in when a frame is actually cast (see threshold).
    "WHITE_BALANCE": True,
    "WB_CAST_RATIO": 1.15,   # apply WB only if max/min channel-mean exceeds this

    # -- SAM 3 prompting -------------------------------------------------------
    # How SAM 3 is prompted:
    #   "auto_exemplar" - derive onion boxes from the vegetation blobs and feed
    #                     them to SAM 3 as positive exemplars. Best for a crop
    #                     the text concept "onion" does not ground (the default).
    #   "text"          - use SAM_TEXT_PROMPTS (unioned). Often returns nothing
    #                     on top-down field imagery.
    #   "manual"        - only the hand-drawn boxes in EXEMPLARS per session.
    # A per-session entry in EXEMPLARS always overrides the mode for that session.
    "SAM_PROMPT_MODE": "auto_exemplar",
    "SAM_TEXT_PROMPTS": ["onion", "onion plant", "green onion leaves"],
    "EXEMPLARS": {
        # "vid3_20260108_132749": [[900, 500, 1050, 780], [1200, 300, 1350, 560]],
    },
    # Auto-exemplar box derivation from the vegetation mask:
    "EXEMPLAR_MIN_AREA_PX": 500,   # min veg blob to use as a SAM exemplar box
    "EXEMPLAR_MAX_BOXES": 20,      # cap exemplar boxes per frame (largest first)
    "EXEMPLAR_PAD_PX": 6,          # pad each box so thin leaf tips are included
    "SAM_CONF": 0.25,        # SAM 3 confidence threshold (detections below dropped)
    "DEVICE": "cuda",        # "cuda" | "cpu"

    # -- Vegetation prior (Excess-Green) --------------------------------------
    # ExG alone masks bare soil whenever a frame has a green colour-cast (soil
    # then reads as weak green). Two extra gates fix that: a pixel is vegetation
    # only if green is the dominant channel AND it is saturated enough - real
    # onion leaves are saturated green, colour-cast soil is not.
    "EXG_THRESHOLD": 0.05,   # exg > this = vegetation. Lower = more permissive.
    "VEG_MIN_SATURATION": 40,  # HSV S (0-255); rejects desaturated colour-cast soil
    "VEG_MORPH_KERNEL": 3,   # close/open kernel px to tidy the veg mask
    "VEG_MIN_COMPONENT_PX": 300,   # drop veg specks smaller than this

    # -- Fusion ----------------------------------------------------------------
    # A SAM mask is accepted only if this fraction of it overlaps vegetation
    # (kills SAM masks that grabbed soil / background).
    "SAM_VEG_OVERLAP_MIN": 0.30,
    # Reject any single SAM mask covering more than this fraction of the frame -
    # a whole-frame "onion" detection is a false positive, not a plant.
    "SAM_MAX_MASK_FRAC": 0.5,
    # Recover onion tissue SAM missed: veg-only components >= this area are added
    # to the final mask. Keeps recall high without importing ExG noise.
    "RECOVER_VEG_MIN_PX": 400,
    # Safety cap: an onion-only field frame is mostly soil, so a final mask
    # covering more than this fraction is wrong. Such frames are blanked (no
    # auto-labels) and counted as 'flagged' for manual annotation.
    "MAX_MASK_FRACTION": 0.5,

    # -- Polygon export --------------------------------------------------------
    "POLY_MIN_AREA_PX": 300,     # skip tiny polygons (noise, single leaf tips)
    "POLY_APPROX_EPS": 1.5,      # Douglas-Peucker simplification (px)
    "MERGE_INTO_ONE_MASK": False,  # False = one polygon per leaf clump (editable)

    # -- Run control -----------------------------------------------------------
    "LIMIT_PER_SESSION": None,   # e.g. 20 for a quick quality trial, then None
    "SAVE_PREVIEWS": True,       # overlay JPGs for fast eyeballing / FiftyOne
    "PREVIEW_SCALE": 0.5,
}

ONION_LABEL = CROP_CLASS     # from common/ontology.py (single source of truth)

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

    exemplars is a list of positive [x1,y1,x2,y2] px boxes (SAM is prompted
    visually), an empty list (nothing to prompt -> no SAM), or None (fall back to
    text concepts). Returns [] if SAM produced nothing (caller then uses ExG)."""
    from PIL import Image
    if exemplars is not None and len(exemplars) == 0:
        return []                               # e.g. bare-soil frame, no boxes
    torch = predictor["torch"]
    processor, device = predictor["processor"], predictor["device"]
    # Accept a pre-loaded BGR array (so SAM sees the same white-balanced image as
    # the vegetation prior) or a path.
    if isinstance(image_path, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    else:
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
        if exemplars is not None:
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
# Preprocessing
# --------------------------------------------------------------------------- #
def white_balance(bgr, cfg):
    """Gray-world white balance to neutralise a colour-cast (shared implementation
    in common/vegetation.py), applied only when a frame is actually cast. This
    recovers the green white-balance-error frames instead of losing them, while
    leaving already-neutral frames essentially untouched."""
    if not cfg["WHITE_BALANCE"]:
        return bgr
    return _white_balance(bgr, cfg["WB_CAST_RATIO"])


# --------------------------------------------------------------------------- #
# Vegetation prior + fusion  (pure OpenCV/NumPy - testable without a GPU)
# --------------------------------------------------------------------------- #
def vegetation_mask(bgr, cfg):
    """Vegetation mask for onion-only scenes: Excess-Green gated by green
    dominance and saturation (shared implementation in common/vegetation.py), so
    a green colour-cast on bare soil is not masked as plant tissue."""
    return _vegetation_mask(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                            cfg["VEG_MORPH_KERNEL"], cfg["VEG_MIN_COMPONENT_PX"])


def auto_exemplars(veg, cfg):
    """Bounding boxes of the largest vegetation blobs, as [x1,y1,x2,y2] px, to
    prompt SAM 3 with real onion exemplars instead of the ungrounded word
    'onion'. In an onion-only scene these blobs are onion tissue."""
    return component_boxes(veg, cfg["EXEMPLAR_MIN_AREA_PX"],
                           cfg["EXEMPLAR_PAD_PX"], cfg["EXEMPLAR_MAX_BOXES"])


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
        if area == 0 or area > cfg["SAM_MAX_MASK_FRAC"] * h * w:
            continue                           # skip empty and whole-frame masks
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


def link_or_copy(src, dst):
    """Hardlink src at dst (zero extra disk space, same volume); fall back to a
    copy if hardlinking isn't possible (different volume/filesystem)."""
    dst = Path(dst)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def pool_frames(session_dir):
    """Pooled frame filenames, minus any curated out by extraction/curate_pool.py.

    A missing `dropped` column means the pool predates curation, which reads as
    "keep everything" - so an old session still behaves exactly as before."""
    pool_csv = session_dir / "meta" / "pool.csv"
    if not pool_csv.exists():
        return []
    return [r["filename"] for r in csv.DictReader(open(pool_csv, encoding="utf-8"))
            if r.get("filename")
            and str(r.get("dropped", "0")).strip() not in ("1", "true", "True")]


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
    cvat_dir = out / cfg["CVAT_READY_SUBDIR"]
    flagged_dir = out / cfg["FLAGGED_RGB_SUBDIR"]
    cvat_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir.mkdir(parents=True, exist_ok=True)
    coco = Coco()
    manual_boxes = cfg["EXEMPLARS"].get(sid)       # hand-drawn, override the mode
    stats = {"frames": 0, "fallback": 0, "empty": 0, "polys": 0, "flagged": 0}
    flagged = []

    prog = Progress(len(frames), f"[{sid}]", unit="frames")
    for fn in frames:
        rgb_path = session_dir / "rgb" / fn
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            prog.update(note="missing frame"); continue
        proc = white_balance(bgr, cfg)             # neutralise colour-cast frames
        veg = vegetation_mask(proc, cfg)

        # Choose SAM 3 prompts for this frame.
        if manual_boxes:
            exemplars = manual_boxes
        elif cfg["SAM_PROMPT_MODE"] == "auto_exemplar":
            exemplars = auto_exemplars(veg, cfg)   # onion boxes from the veg blobs
        else:
            exemplars = None                       # text mode (SAM_TEXT_PROMPTS)

        # SAM sees the same white-balanced image as the vegetation prior.
        sam = sam_fn(predictor, proc, cfg, exemplars) if predictor is not None else []
        final, fstat = fuse(sam, veg, cfg)

        # Safety cap: an onion-only frame is mostly soil. A mask covering more
        # than MAX_MASK_FRACTION is a colour-cast/glare failure - blank it and
        # route the frame to a separate manual-only set rather than exporting
        # garbage or leaving it mixed into the main prelabeled dataset.
        is_flagged = float(final.mean()) > cfg["MAX_MASK_FRACTION"]
        if is_flagged:
            final = np.zeros_like(final)
            flagged.append(fn)
            link_or_copy(rgb_path, flagged_dir / fn)
        else:
            link_or_copy(rgb_path, cvat_dir / fn)
        polys = mask_to_polygons(final, cfg)
        if not is_flagged:
            coco.add(fn, bgr.shape[0], bgr.shape[1], polys)

        cv2.imwrite(str(out / "masks" / fn), (final.astype(np.uint8) * 255))
        if cfg["SAVE_PREVIEWS"]:
            cv2.imwrite(str(out / "preview" / fn.replace(".png", ".jpg")),
                        overlay(proc, final, cfg["PREVIEW_SCALE"]))
        stats["frames"] += 1
        stats["fallback"] += int(fstat.get("fallback_veg_only", False))
        stats["empty"] += int(not final.any())
        stats["flagged"] += int(is_flagged)
        stats["polys"] += len(polys)
        prog.update(note=f"{stats['polys']} polygons, {stats['flagged']} flagged")

    prog.close(note=f"{stats['polys']} polygons, {stats['flagged']} flagged")
    coco.dump(out / "instances_default.json")
    if flagged:
        (out / "flagged_for_manual.txt").write_text("\n".join(flagged))
    print(f"  [{sid}] {stats['frames']} frames | {stats['polys']} onion polygons "
          f"| {stats['fallback']} veg-fallback | {stats['flagged']} flagged "
          f"| {stats['empty']} empty -> {out}\n"
          f"      cvat_ready/ has {stats['frames'] - stats['flagged']} frames "
          f"matching instances_default.json exactly; flagged_rgb/ has "
          f"{stats['flagged']} frames for a separate manual pass")
    return stats


def main(predictor_factory=load_sam3, sam_fn=sam3_masks):
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"])
    from common.dataset_paths import require_sessions_root
    sessions_root = require_sessions_root(root)

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
    print("Next: create a CVAT task from <sid>/cvat_ready/ (NOT sessions/<sid>/rgb/ "
          "- that folder matches instances_default.json exactly, so the COCO 1.0 "
          "import always succeeds), VERIFY, then export the corrected masks as "
          "your training labels. Frames in <sid>/flagged_rgb/ have no auto-label "
          "(see flagged_for_manual.txt) - handle them as a separate manual task.")


if __name__ == "__main__":
    main()
