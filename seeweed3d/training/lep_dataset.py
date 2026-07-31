#!/usr/bin/env python3
"""
SeeWeed3D - torch Dataset for LEPRoiNet, driven by the LEP manifest.

Each row is one verified weed instance. The image is loaded and cropped on
demand from the ORIGINAL frame, so no derived ROI dataset is materialised on
disk; a manifest plus the existing images is the whole input.

The full per-sample chain is assembled here in one place so training and
evaluation cannot drift: crop -> depth degradation -> joint augmentation ->
geometry channels -> heatmap target -> supervision weight.
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
from common.depth_utils import load_depth_mm  # noqa: E402
from training import lep_targets as lt  # noqa: E402
from training import roi as roi_mod  # noqa: E402
from training.lep_roinet import geometry_channels  # noqa: E402

VISIBILITY = ["visible", "partially_occluded_inferable", "not_visible"]
TARGETABILITY = ["yes", "no", "uncertain"]


def polygons_to_mask(polygons, h, w):
    m = np.zeros((h, w), np.uint8)
    for p in polygons:
        a = np.asarray(p, np.float64).reshape(-1, 2)
        if len(a) >= 3:
            cv2.fillPoly(m, [np.round(a).astype(np.int32)], 1)
    return m.astype(bool)


class LEPRoiDataset(Dataset):
    """ROI samples for LEPRoiNet.

    manifest: path to lep_manifest.json, or the parsed dict.
    split: keep only rows from this split ("" keeps everything)."""

    def __init__(self, manifest, images_root=None, split="train", cfg=None,
                 augment=True, seed=0, depth_suffix="depth"):
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.rows = [r for r in manifest["rows"]
                     if not split or r.get("split") == split]
        self.images_root = Path(images_root or manifest.get("images_root", "."))
        self.cfg = cfg
        self.depth_suffix = depth_suffix
        self.rng = np.random.default_rng(seed)
        self.aug = lt.JointAugment(seed=seed) if augment else None

    def __len__(self):
        return len(self.rows)

    def _resolve(self, rel):
        """Manifest paths are posix-relative; resolve against the images root
        so the same manifest works on Windows and Linux."""
        p = Path(rel)
        if p.is_absolute() and p.exists():
            return p
        cand = self.images_root / p
        if cand.exists():
            return cand
        hits = list(self.images_root.rglob(p.name))
        if hits:
            return hits[0]
        raise FileNotFoundError(
            f"image {rel!r} not found under {self.images_root}. Pass the "
            f"correct --images-root; the manifest stores relative paths on "
            f"purpose so the dataset is not duplicated.")

    def _depth_for(self, img_path):
        d = img_path.parent.parent / self.depth_suffix / img_path.name
        if not d.exists():
            return None
        try:
            depth, _ = load_depth_mm(d)
            return depth
        except (ValueError, FileNotFoundError):
            return None

    def __getitem__(self, i):
        r = self.rows[i]
        cfgp = self.cfg
        img_path = self._resolve(r["image_path"])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {img_path}")
        h, w = bgr.shape[:2]

        mask = polygons_to_mask(r["polygons"], h, w)
        depth = self._depth_for(img_path) if cfgp.depth.use_depth else None
        if depth is not None:
            depth, _ = lt.simulate_depth_degradation(depth, cfgp.depth, self.rng)

        tf = roi_mod.make_transform(r["bbox_xywh"], cfgp.roi.out_size,
                                    cfgp.roi.expand_ratio, w, h,
                                    cfgp.roi.min_box_px)
        roi = roi_mod.extract_roi(bgr, mask, tf, depth, cfgp.roi.pad_value)
        uv_roi = tf.to_roi(r["lep_x"], r["lep_y"])

        rgb, m, uv, d, _ = (roi["rgb"], roi["mask"], uv_roi, roi["depth_mm"], None)
        if self.aug is not None:
            rgb, m, uv, d, _ = self.aug(rgb, m, uv, depth_mm=d)
        roi = {"rgb": rgb, "mask": m, "depth_mm": d,
               "depth_valid": np.isfinite(d) if d is not None
               else np.zeros(m.shape, bool)}

        n_geom = geometry_channels(cfgp.model.input_mode)
        geom = (roi_mod.build_geometry_channels(roi, cfgp.depth)[:n_geom]
                if n_geom else np.zeros((0,) + m.shape, np.float32))

        radius = float(np.sqrt(max(1.0, m.sum()) / np.pi))
        hm, uv_hm = lt.make_heatmap(uv, cfgp.roi.out_size, cfgp.heatmap, radius)
        weight = lt.target_weight(r.get("lep_visibility", "visible"),
                                  cfgp.heatmap)

        hs = lt.heatmap_size(cfgp.roi.out_size, cfgp.heatmap.stride)
        mask_hm = cv2.resize(m.astype(np.uint8), (hs, hs),
                             interpolation=cv2.INTER_NEAREST).astype(np.float32)

        return {
            "rgb": torch.from_numpy(
                np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0,
            "geom": torch.from_numpy(np.ascontiguousarray(geom)).float(),
            "heatmap": torch.from_numpy(hm).unsqueeze(0).float(),
            "coord": torch.tensor(uv_hm, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "mask": torch.from_numpy(mask_hm).unsqueeze(0).float(),
            "visibility": torch.tensor(
                VISIBILITY.index(r.get("lep_visibility", "visible"))
                if r.get("lep_visibility") in VISIBILITY else 0),
            "targetability": torch.tensor(
                TARGETABILITY.index(r.get("targetable", "yes"))
                if r.get("targetable") in TARGETABILITY else 0),
        }
