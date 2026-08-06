#!/usr/bin/env python3
"""
SeeWeed3D - CVAT round-trip for onion verification
==================================================
Closes the loop around manual verification of the SAM 3 onion prelabels:

  1. write_labels : emit the CVAT label schema to paste into the task.
  2. ingest       : read your CORRECTED CVAT export (COCO 1.0) and
                    - rasterize it into training-ready binary onion masks,
                    - build a training manifest (rgb <-> mask),
                    - compare the corrections against the auto-prelabels and
                      report IoU / precision / recall per frame.

That last step is the evidence for "model-assisted vs manual annotation": if the
auto-labels already agree with the verified masks at high IoU, prelabeling is
saving real time and the pseudo-labels are trustworthy.

WORKFLOW
--------
Into CVAT (per session):
  - Create a task from  DATASET_ROOT/sessions/<sid>/rgb/
  - Paste  onion_cvat_labels.json  (written by this script) into the Raw label editor
  - Import DATASET_ROOT/auto_labels_onion/<sid>/instances_default.json  as COCO 1.0
  - Verify / correct, then EXPORT the task as COCO 1.0

Back here:
  - Put each session's export at  VERIFIED_ROOT/<sid>/instances_default.json
  - Run this script.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# #############################################################################
# ##   DATASET_ROOT   -  same as the prelabeler's DATASET_ROOT               ##
# ##   VERIFIED_ROOT  -  folder with one <session_id>/instances_default.json ##
# ##                     per session, exported from CVAT as COCO 1.0         ##
# #############################################################################

DATASET_ROOT  = r"E:\Dataset_Vidalia"
VERIFIED_ROOT = r"E:\Dataset_Vidalia\verified"

CONFIG = {
    "DATASET_ROOT":  DATASET_ROOT,
    "VERIFIED_ROOT": VERIFIED_ROOT,
    "AUTO_LABELS_SUBDIR": "auto_labels_onion",   # prelabeler output
    "OUTPUT_SUBDIR":      "training_onion",       # written under DATASET_ROOT/
    # Category name in your CVAT export. The project ontology now uses
    # "onion_plant"; set this to "onion plant" if you are ingesting an older
    # export made before the rename.
    "ONION_CATEGORY":     "onion_plant",
    "ONLY_SESSIONS":      [],       # empty = every session found in VERIFIED_ROOT
    "IOU_ACCEPT":         0.90,     # frames at/above this counted as "clean"
}

# CVAT label schema for onion verification (paste into the Raw label editor).
#
# type: "polygon", matching the weeds schema (common/ontology.py::cvat_labels),
# not "mask". A CVAT "mask" label uses the brush tool and typically exports as
# RLE; COMPRESSED RLE needs pycocotools to decode (see
# datumaro_multitask._decode_rle) and uncompressed RLE is not what CVAT emits
# by default. "polygon" is what the working weeds pipeline round-trips import
# -> correct -> Datumaro export -> prepare_dataset without any extra
# dependency, so onion tasks use the same shape type rather than risking that
# failure mode for no benefit.
ONION_CVAT_LABELS = [
    {"name": "onion_plant", "type": "polygon", "color": "#33ddff", "attributes": [
        {"name": "difficulty", "input_type": "select", "mutable": True,
         "values": ["normal", "overlapping", "blurred", "shadowed", "wet", "truncated"],
         "default_value": "normal"}]},
    {"name": "ignore_region", "type": "polygon", "color": "#000000", "attributes": [
        {"name": "reason", "input_type": "select", "mutable": False,
         "values": ["severe_blur", "ambiguity", "labeling_uncertainty", "out_of_range"],
         "default_value": "ambiguity"}]},
]

# =============================================================================


def write_labels(path):
    Path(path).write_text(json.dumps(ONION_CVAT_LABELS, indent=2))
    return path


def coco_masks(coco, category_name):
    """Rasterize a COCO instance-segmentation dict into one binary mask per
    image. Returns {file_name: bool HxW mask}. Handles polygon segmentations
    (what CVAT's COCO 1.0 export produces)."""
    cat_ids = {c["id"] for c in coco.get("categories", [])
               if c["name"] == category_name}
    imgs = {im["id"]: im for im in coco.get("images", [])}
    masks = {i: np.zeros((im["height"], im["width"]), np.uint8)
             for i, im in imgs.items()}
    for a in coco.get("annotations", []):
        if cat_ids and a.get("category_id") not in cat_ids:
            continue
        im = imgs.get(a["image_id"])
        if im is None:
            continue
        seg = a.get("segmentation")
        if not isinstance(seg, list):          # RLE not expected from CVAT COCO
            continue
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = np.array(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(masks[a["image_id"]], [pts], 1)
    return {imgs[i]["file_name"]: masks[i].astype(bool) for i in imgs}


def agreement(auto, verified):
    """IoU / precision / recall of an auto mask against a verified mask
    (verified = ground truth). Empty-vs-empty counts as perfect."""
    inter = int((auto & verified).sum())
    union = int((auto | verified).sum())
    tp, fp = inter, int((auto & ~verified).sum())
    fn = int((~auto & verified).sum())
    return {"iou": inter / union if union else 1.0,
            "precision": tp / (tp + fp) if (tp + fp) else 1.0,
            "recall": tp / (tp + fn) if (tp + fn) else 1.0,
            "verified_px": int(verified.sum()), "auto_px": int(auto.sum())}


def discover_sessions(cfg):
    vroot = Path(cfg["VERIFIED_ROOT"])
    if not vroot.exists():
        return []
    only = set(cfg["ONLY_SESSIONS"])
    out = []
    for d in sorted(p for p in vroot.iterdir() if p.is_dir()):
        j = d / "instances_default.json"
        if j.exists() and (not only or d.name in only):
            out.append((d.name, j))
    return out


def ingest(cfg):
    root = Path(cfg["DATASET_ROOT"])
    out_root = root / cfg["OUTPUT_SUBDIR"]
    out_root.mkdir(parents=True, exist_ok=True)
    write_labels(out_root / "onion_cvat_labels.json")

    sessions = discover_sessions(cfg)
    if not sessions:
        sys.exit(f"No verified exports under {cfg['VERIFIED_ROOT']} "
                 f"(expected <session_id>/instances_default.json). See the "
                 f"workflow in the module docstring.")

    manifest, agree_rows, summary = [], [], []
    for sid, vjson in sessions:
        verified = coco_masks(json.loads(vjson.read_text()), cfg["ONION_CATEGORY"])
        mask_dir = out_root / "masks" / sid
        mask_dir.mkdir(parents=True, exist_ok=True)

        for fn, vmask in verified.items():
            cv2.imwrite(str(mask_dir / fn), (vmask.astype(np.uint8) * 255))
            manifest.append({
                "session_id": sid, "filename": fn,
                "image": str(root / "sessions" / sid / "rgb" / fn),
                "mask": str(mask_dir / fn),
                "onion_px": int(vmask.sum())})

        auto_json = root / cfg["AUTO_LABELS_SUBDIR"] / sid / "instances_default.json"
        ious = []
        if auto_json.exists():
            auto = coco_masks(json.loads(auto_json.read_text()), cfg["ONION_CATEGORY"])
            for fn, vmask in verified.items():
                amask = auto.get(fn)
                if amask is None or amask.shape != vmask.shape:
                    continue
                a = agreement(amask, vmask)
                a.update({"session_id": sid, "filename": fn})
                agree_rows.append(a)
                ious.append(a["iou"])

        clean = sum(1 for v in ious if v >= cfg["IOU_ACCEPT"])
        clean_col = f"pct_iou_ge_{cfg['IOU_ACCEPT']}"
        mean_iou = round(float(np.mean(ious)), 4) if ious else None
        pct_clean = round(clean / len(ious), 3) if ious else None
        summary.append({
            "session_id": sid, "frames_verified": len(verified),
            "frames_compared": len(ious),
            "mean_iou": mean_iou,
            "median_iou": round(float(np.median(ious)), 4) if ious else None,
            clean_col: pct_clean})
        print(f"  [{sid}] {len(verified)} verified frames | "
              f"mean IoU vs auto = {mean_iou} | "
              f"clean(>={cfg['IOU_ACCEPT']}) = {pct_clean}")

    _write_csv(out_root / "manifest.csv", manifest)
    _write_csv(out_root / "agreement.csv", agree_rows)
    _write_csv(out_root / "agreement_summary.csv", summary)
    print(f"\nTraining masks + manifest + agreement report under {out_root}")
    return {"manifest": manifest, "agreement": agree_rows, "summary": summary}


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ingest(CONFIG)


if __name__ == "__main__":
    main()
