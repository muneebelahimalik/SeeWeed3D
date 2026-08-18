#!/usr/bin/env python3
"""
SeeWeed3D - seg_manifest.json -> Roboflow-style COCO, for the RF-DETR backend.

RF-DETR trains from a directory layout it discovers itself:

    <out>/train/_annotations.coco.json   + the images beside it
    <out>/valid/_annotations.coco.json   + the images beside it
    <out>/test/_annotations.coco.json    (optional)

Note `valid`, not `val` - that is Roboflow's spelling and rfdetr checks for it
literally.

WHY THIS CONVERTS RATHER THAN RE-EXPORTS FROM CVAT
--------------------------------------------------
Everything the pipeline enforces lives between the CVAT export and
seg_manifest.json: frame selection, class dropping, split assignment, the
crop-starvation check. Converting from the manifest inherits all of it. Going
back to the CVAT export for a COCO file instead would silently reintroduce the
SAM-prelabelled frames and the un-dropped classes, and the two backends would
then be trained on different data while appearing to share a dataset.

CATEGORY IDS ARE THE MANIFEST'S ACTIVE CLASS LIST, +1
-----------------------------------------------------
COCO category ids are 1-based, and RF-DETR reserves 0 for background exactly as
torchvision does. Using the ontology's global ids would leave gaps whenever
--drop-classes removed a class, and a gap in category ids is the kind of thing
a training loop turns into an off-by-one rather than an error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402
from training.seg_dataset import polygons_to_mask, resolve_image  # noqa: E402

#: Roboflow's split directory names. rfdetr looks for these literally.
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def polygon_area(poly):
    """Shoelace area of one flat [x0,y0,x1,y1,...] polygon."""
    a = np.asarray(poly, np.float64).reshape(-1, 2)
    if len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def instance_bbox_and_area(polygons, h, w):
    """COCO xywh bbox and pixel area for one instance.

    Area is the RASTERISED pixel count, not the sum of polygon areas: an
    instance exported as several polygons may have them overlap, and summing
    would double-count. COCO area feeds the small/medium/large split of the
    evaluation, so an inflated value quietly moves a small weed into the
    medium bucket - hiding the exact failure this project measures."""
    m = np.zeros((h, w), np.uint8)
    for p in polygons:
        a = np.asarray(p, np.float64).reshape(-1, 2)
        if len(a) >= 3:
            import cv2
            cv2.fillPoly(m, [np.round(a).astype(np.int32)], 1)
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None, 0.0
    x0, y0 = float(xs.min()), float(ys.min())
    bw, bh = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
    return [x0, y0, bw, bh], float(m.sum())


def build_coco(frames, classes, image_names):
    """One COCO dict for one split."""
    cats = [{"id": i + 1, "name": c, "supercategory": "plant"}
            for i, c in enumerate(classes)]
    images, annotations = [], []
    ann_id = 1
    for img_id, rec in enumerate(frames, start=1):
        h, w = int(rec["height"]), int(rec["width"])
        images.append({"id": img_id, "file_name": image_names[rec["item_id"]],
                       "height": h, "width": w})
        for inst in rec["instances"]:
            polys = [[float(v) for v in p] for p in inst["polygons"]
                     if len(p) >= 6]
            if not polys:
                continue
            bbox, area = instance_bbox_and_area(polys, h, w)
            if bbox is None or area <= 0:
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": classes.index(inst["class_name"]) + 1,
                "segmentation": polys, "bbox": bbox, "area": area,
                "iscrowd": 0,
            })
            ann_id += 1
    return {"info": {"description": "SeeWeed3D", "version": "1.0"},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": cats}


def export(dataset_dir, out_dir, images_root=None, link=False, overwrite=False):
    """seg_manifest.json -> Roboflow COCO tree. Returns a summary dict."""
    dataset_dir, out = Path(dataset_dir), Path(out_dir)
    man = dataset_dir / "seg_manifest.json"
    if not man.exists():
        raise SystemExit(f"ERROR: {man} not found. Build the dataset first "
                         f"(seeweed3d/training/make_dataset.py).")
    doc = json.loads(man.read_text(encoding="utf-8"))
    classes = list(doc.get("classes") or CLASSES)
    roots = images_root or doc.get("images_root") or "."

    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"ERROR: {out} exists and is not empty. RF-DETR discovers a "
                f"dataset by scanning it, so leftover images from a previous "
                f"build would silently join this one. Pass --overwrite, or "
                f"choose an empty directory.")
        shutil.rmtree(out)

    # category_id -> class name, recorded so INFERENCE can map predictions back
    # by id instead of by position. rfdetr remaps sparse ids to contiguous
    # labels for training and then maps them BACK to the original category_id
    # at predict time, so a model trained on ids 1..N predicts 1..N - not
    # 0..N-1. Guessing an offset there mislabels plants, and mislabelling the
    # crop is a laser pointed at an onion.
    # label_provenance is carried through from the manifest so preflight can
    # restate it at train time. It decides what the run's metrics MEAN, and the
    # COCO tree is what the trainer actually sees.
    summary = {"classes": classes, "splits": {}, "out_dir": str(out),
               "label_provenance": doc.get("label_provenance"),
               "category_ids": {str(i + 1): c for i, c in enumerate(classes)}}
    for split, dirname in SPLIT_DIRS.items():
        frames = [f for f in doc["frames"] if f.get("split") == split]
        if not frames:
            continue
        sdir = out / dirname
        sdir.mkdir(parents=True, exist_ok=True)

        names = {}
        for rec in frames:
            src = resolve_image(rec["image_path"], roots, rec.get("session_id"))
            # Flatten into the split directory, keeping the session prefix that
            # the extractor already put in every filename - so two sessions can
            # never collide here even though the tree is flat.
            name = Path(src).name
            names[rec["item_id"]] = name
            dst = sdir / name
            if dst.exists():
                continue
            if link:
                try:
                    dst.hardlink_to(src)
                    continue
                except (OSError, NotImplementedError):
                    pass          # crossing volumes, or Windows without perms
            shutil.copy2(src, dst)

        coco = build_coco(frames, classes, names)
        (sdir / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8")
        # Per-class counts, not just the total. A split's instance count says
        # nothing about whether a class is learnable, and "1200 instances"
        # made almost entirely of one class is the normal shape of this
        # dataset - so the total is the number least worth looking at.
        per_class = Counter()
        for a in coco["annotations"]:
            per_class[classes[a["category_id"] - 1]] += 1
        summary["splits"][dirname] = {
            "frames": len(coco["images"]),
            "instances": len(coco["annotations"]),
            "per_class": {c: per_class.get(c, 0) for c in classes},
        }

    if "train" not in summary["splits"]:
        raise SystemExit("ERROR: no train frames in the manifest.")
    if "valid" not in summary["splits"]:
        # rfdetr evaluates against valid/ every epoch; without it there is no
        # early stopping, no best-checkpoint selection and no metric.
        raise SystemExit(
            "ERROR: no val frames in the manifest, so RF-DETR would train "
            "with nothing to evaluate against - no early stopping and no "
            "best checkpoint. Rebuild with VAL_FRACTION > 0.")
    (out / "seeweed3d_export.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="make_dataset.py OUT_DIR")
    p.add_argument("--out", required=True, help="COCO tree to write")
    p.add_argument("--images-root", nargs="*", default=None)
    p.add_argument("--link", action="store_true",
                   help="hardlink images instead of copying (same volume only)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    roots = a.images_root if a.images_root else None
    if roots and len(roots) == 1:
        roots = roots[0]
    s = export(a.dataset, a.out, roots, link=a.link, overwrite=a.overwrite)
    for k, v in s["splits"].items():
        print(f"  {k:<6} {v['frames']:>4} frames  {v['instances']:>5} instances")
    print(f"  classes: {s['classes']}")
    print(f"  -> {s['out_dir']}")


if __name__ == "__main__":
    main()
