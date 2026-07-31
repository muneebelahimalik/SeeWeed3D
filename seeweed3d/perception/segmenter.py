#!/usr/bin/env python3
"""
SeeWeed3D - thin, version-pinned wrapper around Ultralytics instance segmentation.

WHY A WRAPPER AND NOT A FORK
----------------------------
Vendoring Ultralytics would mean owning its update path forever and would make
the AGPL-3.0 obligations harder to reason about, not easier. This module is a
boundary instead: everything downstream sees `Detections`, a plain numpy
structure, so swapping YOLO26 for another segmenter later touches one file.

It also means Ultralytics is an OPTIONAL dependency. Importing this module never
imports ultralytics; that happens inside `load()`, so the unit suite runs on a
machine with no training stack. `MockSegmenter` provides the same interface for
tests and for exercising the pipeline without weights.

LICENSING - READ BEFORE SHIPPING
--------------------------------
Ultralytics is AGPL-3.0. Under AGPL, distributing (or in some readings, merely
operating) a derivative work obliges you to release the complete corresponding
source of the LARGER application, including this repository and potentially the
trained weights. Ultralytics states that proprietary or commercial use requires
an Enterprise License. A commercial laser weeder is exactly that case. The
segmentation stage is therefore isolated behind this interface so it can be
replaced without touching the rest of the system if licensing forces it. See
docs/supervised_perception_baseline.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402


@dataclass
class Detections:
    """Segmenter output for one frame, framework-independent.

    masks: (N, H, W) bool. boxes: (N, 4) xywh. classes: (N,) int index into
    CLASSES. scores: (N,) float."""
    masks: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray
    width: int = 0
    height: int = 0
    names: list = field(default_factory=lambda: list(CLASSES))

    def __len__(self):
        return int(len(self.scores))

    def class_name(self, i):
        return self.names[int(self.classes[i])]

    def onion_safety_mask(self):
        """Union of every predicted crop mask.

        The union, not the individual instances, is the safety output: for
        crop protection it does not matter which onion a pixel belongs to, only
        that it is onion. Instances stay separate for training and metrics, but
        one conservative mask is what the laser decision consults."""
        if self.height and self.width:
            out = np.zeros((self.height, self.width), bool)
        elif len(self.masks):
            out = np.zeros(self.masks[0].shape, bool)
        else:
            return None
        crop_idx = CLASSES.index(CROP_CLASS)
        for i in range(len(self)):
            if int(self.classes[i]) == crop_idx:
                out |= self.masks[i].astype(bool)
        return out

    def weed_indices(self):
        crop_idx = CLASSES.index(CROP_CLASS)
        return [i for i in range(len(self)) if int(self.classes[i]) != crop_idx]


class UltralyticsSegmenter:
    """Loads a YOLO26-seg checkpoint and returns Detections."""

    def __init__(self, weights, conf=0.25, iou=0.60, device="cpu",
                 imgsz=None, max_det=300):
        self.weights = str(weights)
        self.conf, self.iou, self.device = conf, iou, device
        self.imgsz, self.max_det = imgsz, max_det
        self._model = None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. It is an OPTIONAL training/"
                "deployment dependency:\n"
                "    python -m pip install -r requirements-training.txt\n"
                "The data-pipeline and unit tests do not need it."
            ) from e
        if not Path(self.weights).exists() and not str(self.weights).startswith(
                ("yolo", "http")):
            raise FileNotFoundError(
                f"segmentation weights not found: {self.weights}. Train first "
                f"(seeweed3d/training/train_seg.py) or pass a pretrained name "
                f"such as 'yolo26n-seg.pt'.")
        self._model = YOLO(self.weights)
        return self

    def __call__(self, bgr):
        if self._model is None:
            self.load()
        kw = {"conf": self.conf, "iou": self.iou, "device": self.device,
              "max_det": self.max_det, "verbose": False}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        res = self._model.predict(bgr, **kw)[0]
        h, w = bgr.shape[:2]
        if res.masks is None or len(res.boxes) == 0:
            return Detections(np.zeros((0, h, w), bool), np.zeros((0, 4)),
                              np.zeros((0,), int), np.zeros((0,)), w, h)

        import cv2
        raw = res.masks.data.cpu().numpy()
        masks = np.zeros((raw.shape[0], h, w), bool)
        for i in range(raw.shape[0]):
            m = raw[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            masks[i] = m > 0.5
        xyxy = res.boxes.xyxy.cpu().numpy()
        boxes = np.stack([xyxy[:, 0], xyxy[:, 1],
                          xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
        return Detections(masks, boxes,
                          res.boxes.cls.cpu().numpy().astype(int),
                          res.boxes.conf.cpu().numpy().astype(float), w, h)


class MockSegmenter:
    """Fixed Detections, for tests and for running the pipeline with no weights.

    Exists so the end-to-end pipeline, including every safety path, is testable
    without a GPU or a trained checkpoint - the segmenter is the only component
    that genuinely needs them."""

    def __init__(self, detections):
        self.detections = detections

    def __call__(self, bgr):
        return self.detections
