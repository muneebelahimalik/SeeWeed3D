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
import re
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


def as_roots(images_root):
    """Normalize a single path or a list/tuple of paths into list[Path].

    Multiple datasets (e.g. a weed capture set and a separately-recorded onion
    set) are not always under one sessions folder, so every images_root
    parameter accepts either form - a bare list of one is the common case and
    costs nothing."""
    if isinstance(images_root, (list, tuple)):
        return [Path(r) for r in images_root]
    return [Path(images_root)]


def _session_from_name(p):
    """`<session>_<index>.png` -> `<session>`, the extractor's naming, so the
    session is recoverable even from a bare filename with no folder in it."""
    stem = p.stem
    if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
        return stem.rsplit("_", 1)[0]
    return None


def _frame_index(name):
    """Trailing zero-padded index of an extractor filename, or None.

    `<session>_000123.png` -> "000123". The index is assigned by extraction and
    is stable across any later renaming of the session or its files, which is
    what makes it usable to re-pair an export with images whose prefix has
    since changed."""
    m = re.search(r"_(\d+)$", Path(name).stem)
    return m.group(1) if m else None


def resolve_image(rel, images_root, session_id=None, export_dir=None):
    """Find a frame on disk from the relative path a manifest stored.

    `images_root` is a single path or a list of candidate roots, tried in
    order - a merged build from several CVAT exports may have its sessions
    spread across more than one sessions folder, and each frame belongs to
    whichever root actually contains its session, not to all of them.

    CVAT tasks are flat uploads, so an export's media path is usually a bare
    filename with no session folder in it. Two layouts resolve directly:

        <root>/<session_id>/rgb/<name>    root is the SESSIONS folder
        <root>/rgb/<name>                 root IS one session's folder

    Both are accepted because both are natural ways to describe a merged build
    - one root covering many sessions, or one root per session. Trying them
    directly matters: the fallback is a recursive scan, and running one per
    image per epoch over a root holding tens of thousands of PNGs dominates
    training time.

    Every root's cheap, direct checks run before any root's expensive scan, so
    a frame that resolves canonically under the SECOND root never pays for a
    full walk of the first one first.

    The scan is built once per root and cached. The first name wins on
    collision, which is safe because extraction prefixes every frame with its
    session id, so two sessions cannot produce the same filename."""
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    roots = as_roots(images_root)
    sess = session_id or (p.parts[0] if len(p.parts) > 1 else None) \
        or _session_from_name(p)

    # The export's own folder is tried FIRST and, uniquely, is allowed to match
    # on frame index rather than on name. It is the one root known to hold this
    # frame, so an index match here re-pairs an export with images whose prefix
    # changed when the session folder was renamed - without the ambiguity that
    # makes the same match unsafe across a list of roots that all contain a
    # frame of that index.
    if export_dir:
        home = Path(export_dir)
        for sub in RGB_SUBDIRS:
            cand = (home / sub / p.name) if sub else (home / p.name)
            if cand.exists():
                return cand
        idx = _frame_index(p.name)
        if idx:
            for sub in RGB_SUBDIRS:
                d = (home / sub) if sub else home
                if not d.is_dir():
                    continue
                hits = sorted(q for q in d.glob(f"*_{idx}{p.suffix}")
                              if q.is_file())
                # Exactly one, or it is not an answer - a tie means guessing
                # which frame an annotation belongs to, and a mislabelled frame
                # is worse than a missing one.
                if len(hits) == 1:
                    return hits[0]

    for root in roots:
        cand = root / p
        if cand.exists():
            return cand
        # The root may BE a session folder rather than the folder of sessions:
        # <session>/rgb/<name>. Giving one root per session is a natural way to
        # name a merged build, and the recursive scan below would find these
        # anyway - at the cost of a full walk per root, which the docstring
        # above explains is what dominates training time. This makes it a
        # direct hit instead.
        #
        # Safe without checking the folder's name against `sess`: extraction
        # prefixes every frame with its own session id, so a file of this exact
        # name cannot be sitting in a different session's rgb/.
        for sub in RGB_SUBDIRS:
            cand = root / sub / p.name if sub else root / p.name
            if cand.exists():
                return cand
        if sess:
            for sub in RGB_SUBDIRS:
                cand = root / sess / sub / p.name if sub else root / sess / p.name
                if cand.exists():
                    return cand

    for root in roots:
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

    shown = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(
        f"image {rel!r} not found under: {shown}\n"
        f"Expected <images-root>/<session_id>/rgb/<name>, or "
        f"<images-root>/rgb/<name> when the root IS one session's folder. "
        f"Point --images-root at the SESSIONS folder - the one whose children "
        f"are session id folders - or at the session folders themselves, but "
        f"not at an rgb/ subfolder. If your sessions are split across more "
        f"than one parent folder, pass all of them.")


#: Augmentation preset names. See build_augmentation() for what each does and,
#: more importantly, for what is deliberately NOT in any of them.
AUG_PRESETS = ("none", "flip", "standard", "strong")


def build_augmentation(preset="standard", min_box_px=2):
    """Joint image/mask/box augmentation, or None.

    Built on `torchvision.transforms.v2` rather than Albumentations for two
    reasons. It is BSD-3 and already a dependency, whereas the maintained
    Albumentations is now AlbumentationsX under AGPL-3.0/commercial - the same
    licence trap as Ultralytics, and the MIT `albumentations` is frozen at
    2.0.8. And v2 transforms operate on tv_tensors, so masks, boxes and labels
    are transformed and FILTERED TOGETHER; hand-rolled augmentation that flips
    an image and its masks but forgets a box is a class of silent corruption
    this avoids structurally.

    WHAT IS DELIBERATELY ABSENT, in every preset:

      Mosaic, MixUp, CopyPaste. For a crop-safety model these fabricate crop
      geometry that never existed - an onion pasted from another frame teaches
      the model about a plant arrangement no field produced. The same argument
      that keeps them out of Stage B keeps them out here.

      Vertical flip is in `flip` for backward compatibility but NOT in
      `standard`. Top-down field imagery has a real light direction; mirroring
      it vertically inverts the shading cue that separates a leaf from its own
      shadow. Horizontal flip is safe because a row looks the same driven
      either way.

      Rotation beyond small angles. Onion rows are near-parallel in every
      frame, and that regularity is signal the model may legitimately use.

    Presets:
      none      identity (returns None)
      flip      horizontal + vertical flip, the historical behaviour
      standard  horizontal flip, small rotation, scale jitter, photometric
      strong    as standard with wider ranges, for when overfitting is clear
    """
    if preset in (None, "none"):
        return None
    if preset not in AUG_PRESETS:
        raise ValueError(f"augment preset must be one of {AUG_PRESETS}, "
                         f"got {preset!r}")
    from torchvision.transforms import v2

    if preset == "flip":
        steps = [v2.RandomHorizontalFlip(0.5), v2.RandomVerticalFlip(0.5)]
    else:
        strong = preset == "strong"
        steps = [
            v2.RandomHorizontalFlip(0.5),
            # Photometric first, on the un-warped image. Field light varies far
            # more between a bright noon pass and an overcast one than any
            # geometric change, and this is the cheapest transfer the model can
            # be taught with 62 frames.
            v2.RandomPhotometricDistort(
                brightness=(0.6, 1.5) if strong else (0.75, 1.3),
                contrast=(0.6, 1.5) if strong else (0.75, 1.3),
                saturation=(0.6, 1.5) if strong else (0.75, 1.3),
                hue=(-0.05, 0.05) if strong else (-0.025, 0.025),
                p=0.8 if strong else 0.5),
            # Scale jitter is the one that targets small-weed recall directly:
            # it shows the same plant at several apparent sizes, which is what
            # a varying camera height and a growing crop produce anyway.
            #
            # RandomAffine, NOT ScaleJitter. ScaleJitter resizes the CANVAS to
            # target_size x scale; against a 2208x1242 ZED frame a target of
            # 1024 meant every training image arrived at 24-73% of native.
            # Mask R-CNN's own transform then upsampled it back to min_size, so
            # raising min_size bought nothing - the detail was already gone and
            # the model trained on blur. The augmentation was silently undoing
            # the single biggest lever on the metric it was added to improve.
            # RandomAffine scales the CONTENT and leaves the canvas alone, which
            # is the augmentation that was intended all along, and it folds the
            # rotation into the same resample instead of blurring twice.
            v2.RandomAffine(degrees=15 if strong else 7,
                            scale=(0.5, 1.6) if strong else (0.7, 1.4)),
        ]
    # ALWAYS last: an instance whose box has been rotated or scaled out of the
    # frame must lose its box, its mask and its label together. torchvision
    # rejects a degenerate box outright, and a mask surviving without its label
    # would silently mislabel a plant.
    steps.append(v2.SanitizeBoundingBoxes(min_size=min_box_px))
    return v2.Compose(steps)


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
                 min_area_px=16, augment=True, seed=0, aug_preset="standard"):
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.frames = [f for f in manifest["frames"]
                       if not split or f.get("split") == split]
        # A single path or a list of them - resolve_image tries every root.
        # NOT collapsed to one Path here: a merged multi-source build's
        # sessions can live under more than one sessions folder.
        self.images_root = images_root or manifest.get("images_root") or "."
        # The manifest's own class list, which may be a REDUCED active set.
        # Indexing into the full ontology here would shift every class above a
        # dropped one.
        self.classes = list(manifest.get("classes") or CLASSES)
        self.min_area_px = min_area_px
        self.augment = augment
        self.aug_preset = aug_preset
        self._aug = build_augmentation(aug_preset) if augment else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.frames)

    def _resolve(self, rel, session_id=None, export_dir=None):
        return resolve_image(rel, self.images_root, session_id, export_dir)

    def __getitem__(self, i):
        rec = self.frames[i]
        path = self._resolve(rec["image_path"], rec.get("session_id"),
                             rec.get("export_dir"))
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

        img = torch.from_numpy(
            bgr[:, :, ::-1].copy().astype(np.float32) / 255.0).permute(2, 0, 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": (torch.from_numpy(np.stack(masks)).to(torch.uint8)
                      if masks else torch.zeros((0, h, w), dtype=torch.uint8)),
            "image_id": torch.tensor(i),
        }
        if self.augment and self._aug is not None and len(labels):
            # v2 transforms dispatch on tv_tensor SUBCLASSES, not on dict keys.
            # A plain tensor of boxes is silently treated as generic data and
            # left untransformed while the image is warped - masks and boxes
            # then describe a picture that no longer exists.
            from torchvision import tv_tensors
            h_, w_ = img.shape[-2:]
            wrapped = {
                "boxes": tv_tensors.BoundingBoxes(
                    target["boxes"], format="XYXY", canvas_size=(h_, w_)),
                "masks": tv_tensors.Mask(target["masks"]),
                "labels": target["labels"],
            }
            aug_img, aug = self._aug(tv_tensors.Image(img), wrapped)
            # Augmentation can legitimately empty a frame (every plant rotated
            # out). Keep the original rather than hand the trainer a sample it
            # has to skip.
            if len(aug["labels"]):
                m = aug["masks"].as_subclass(torch.Tensor).to(torch.uint8)
                # Recompute boxes FROM the transformed masks rather than take
                # v2's. Under rotation v2 transports a box by rotating its four
                # corners and taking the axis-aligned hull, which is strictly
                # larger than the rotated mask - a 45-degree rotation inflates
                # a square box by ~41% per side. Here the mask is the ground
                # truth, so deriving the box from it keeps the two consistent
                # and keeps the box tight.
                # bool, not uint8: torch treats a uint8 index as POSITIONAL,
                # so `labels[keep]` would silently select labels 0 and 1 by
                # position instead of masking - a wrong-label bug that only
                # shows up as poor accuracy.
                keep = m.flatten(1).any(1).bool()
                if keep.any():
                    from torchvision.ops import masks_to_boxes
                    m = m[keep]
                    b = masks_to_boxes(m)
                    # A non-empty mask is NOT enough. Rotation and scale jitter
                    # can leave a plant as a single row or column of pixels, and
                    # masks_to_boxes takes inclusive min/max - so x1 == x2 and
                    # the box has zero area. torchvision asserts on that inside
                    # forward() ("All bounding boxes should have positive height
                    # and width"), killing the run minutes in with no clue that
                    # augmentation caused it. The pre-augmentation path above
                    # already drops these; the augmented path must too, or the
                    # guard only holds for the transform that cannot break it.
                    solid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
                    if solid.any():
                        m = m[solid]
                        img = aug_img.as_subclass(torch.Tensor)
                        target = {
                            "boxes": b[solid].float(),
                            "labels": aug["labels"][keep][solid],
                            "masks": m,
                            "image_id": target["image_id"],
                        }
        return img, target


def collate(batch):
    """torchvision detection models take lists, not stacked tensors, because
    images may differ in size."""
    return tuple(zip(*batch))
