#!/usr/bin/env python3
"""
SeeWeed3D - torchvision-format instance segmentation dataset.

Feeds the BSD-3 Mask R-CNN backend directly from the Datumaro import, so the
permissive path needs no YOLO-format label files and no second copy of the
dataset. Ultralytics labels are still written by prepare_dataset.py for anyone
who chooses that backend, but they are not on this path.

torchvision detection models expect, per image:
    boxes  (N,4) xyxy float
    labels (N,)  int64, where 0 is BACKGROUND
    masks  (N,H,W) uint8

`labels` is therefore ontology index + 1. `MaskRCNNSegmenter` subtracts the one
again on the way out, so every index outside this file is plain ontology order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402

BACKGROUND_OFFSET = 1


#: Subfolders of a session that hold the RGB frames, in preference order. The
#: extractor writes <images_root>/<session_id>/{rgb,depth,right,conf}/, and only
#: rgb is a training image - a depth PNG has the same filename, so searching
#: blindly could return the wrong stream.
RGB_SUBDIRS = ("rgb", "")

_INDEX_CACHE = {}


def resolve_image(rel, images_root, session_id=None):
    """Find a frame on disk from the relative path a manifest stored.

    CVAT tasks are flat uploads, so an export's media path is usually a bare
    filename with no session folder in it. The canonical layout is
    <images_root>/<session_id>/rgb/<name>, and trying that directly matters:
    the fallback is a recursive scan, and running one per image per epoch over
    a dataset root holding tens of thousands of PNGs dominates training time.

    The scan is therefore built once per root and cached. The first name wins
    on collision, which is safe because extraction prefixes every frame with
    its session id, so two sessions cannot produce the same filename."""
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    root = Path(images_root)

    cand = root / p
    if cand.exists():
        return cand

    sess = session_id or (p.parts[0] if len(p.parts) > 1 else None)
    if not sess:
        # `<session>_<index>.png` - the extractor's naming, so the session is
        # recoverable even from a bare filename.
        stem = p.stem
        if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
            sess = stem.rsplit("_", 1)[0]
    if sess:
        for sub in RGB_SUBDIRS:
            cand = root / sess / sub / p.name if sub else root / sess / p.name
            if cand.exists():
                return cand

    key = str(root.resolve())
    index = _INDEX_CACHE.get(key)
    if index is None:
        index = {}
        for q in root.rglob("*"):
            if q.is_file():
                index.setdefault(q.name, q)
        _INDEX_CACHE[key] = index
    hit = index.get(p.name)
    if hit is not None:
        return hit

    raise FileNotFoundError(
        f"image {rel!r} not found under {root}.\n"
        f"Expected <images-root>/<session_id>/rgb/<name>. Point --images-root "
        f"at the SESSIONS folder - the one whose children are session id "
        f"folders - not at a session itself and not at its rgb/ subfolder.")


def polygons_to_mask(polygons, h, w):
    m = np.zeros((h, w), np.uint8)
    for p in polygons:
        a = np.asarray(p, np.float64).reshape(-1, 2)
        if len(a) >= 3:
            cv2.fillPoly(m, [np.round(a).astype(np.int32)], 1)
    return m


class SegManifestDataset(Dataset):
    """Instance segmentation samples from a segmentation manifest.

    The manifest is written by prepare_dataset.py and holds, per frame, the
    image path plus every instance's polygons and class - the same records the
    LEP manifest is derived from, so the two stages can never disagree about
    what was annotated."""

    def __init__(self, manifest, images_root=None, split="train",
                 min_area_px=16, augment=True, seed=0):
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.frames = [f for f in manifest["frames"]
                       if not split or f.get("split") == split]
        self.images_root = Path(images_root or manifest.get("images_root", "."))
        # The manifest's own class list, which may be a REDUCED active set.
        # Indexing into the full ontology here would shift every class above a
        # dropped one.
        self.classes = list(manifest.get("classes") or CLASSES)
        self.min_area_px = min_area_px
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.frames)

    def _resolve(self, rel, session_id=None):
        return resolve_image(rel, self.images_root, session_id)

    def __getitem__(self, i):
        rec = self.frames[i]
        path = self._resolve(rec["image_path"], rec.get("session_id"))
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {path}")
        h, w = bgr.shape[:2]

        masks, labels, boxes = [], [], []
        for inst in rec["instances"]:
            m = polygons_to_mask(inst["polygons"], h, w)
            if int(m.sum()) < self.min_area_px:
                continue
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            if x1 <= x0 or y1 <= y0:
                continue                      # torchvision rejects empty boxes
            masks.append(m)
            labels.append(self.classes.index(inst["class_name"])
                          + BACKGROUND_OFFSET)
            boxes.append([x0, y0, x1, y1])

        # Horizontal/vertical flips only. Mosaic and MixUp are NOT used here
        # either: for a crop-safety model, compositing an onion from one frame
        # into another fabricates crop geometry that never existed.
        if self.augment and masks:
            if self.rng.random() < 0.5:
                bgr = bgr[:, ::-1].copy()
                masks = [m[:, ::-1].copy() for m in masks]
                boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
            if self.rng.random() < 0.5:
                bgr = bgr[::-1].copy()
                masks = [m[::-1].copy() for m in masks]
                boxes = [[b[0], h - b[3], b[2], h - b[1]] for b in boxes]

        img = torch.from_numpy(
            bgr[:, :, ::-1].copy().astype(np.float32) / 255.0).permute(2, 0, 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": (torch.from_numpy(np.stack(masks)).to(torch.uint8)
                      if masks else torch.zeros((0, h, w), dtype=torch.uint8)),
            "image_id": torch.tensor(i),
        }
        return img, target


def collate(batch):
    """torchvision detection models take lists, not stacked tensors, because
    images may differ in size."""
    return tuple(zip(*batch))
