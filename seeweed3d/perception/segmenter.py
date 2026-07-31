#!/usr/bin/env python3
"""
SeeWeed3D - pluggable instance-segmentation backends behind one interface.

Everything downstream sees `Detections`, a plain numpy structure, so the
segmenter is the ONLY component that knows which model produced it. That
boundary exists because of licensing, not merely tidiness (see below).

BACKENDS AND THEIR LICENCES
---------------------------
    name           licence        notes
    ------------   ------------   ----------------------------------------
    maskrcnn       BSD-3-Clause   torchvision. DEFAULT. No new dependency
                                  (torchvision is already required for
                                  Stage B), mature, easy to fine-tune on a
                                  small set. Not real-time; correct choice
                                  for a prototype.
    rfdetr         Apache-2.0     Roboflow RF-DETR-Seg. Real-time, TensorRT
                                  FP16, nano..2XL, built for fine-tuning.
                                  The upgrade path once the prototype works.
    rtmdet         Apache-2.0     OpenMMLab RTMDet-Ins. Real-time, long
                                  TensorRT track record; pulls in the
                                  mmengine/mmcv stack, which is
                                  version-fragile.
    ultralytics    AGPL-3.0 (!)   YOLO26-seg. Strong, but AGPL: commercial
                                  or proprietary use needs an Ultralytics
                                  Enterprise Licence, and AGPL otherwise
                                  obliges releasing the complete source of
                                  the larger work. NOT the default. Only
                                  reachable by asking for it explicitly.

The default is deliberately a permissively-licensed backend so that nothing in
the normal path can create a licensing obligation by accident. A commercial
laser weeder is exactly the case an AGPL dependency makes expensive, and unlike
a code bug it cannot be fixed after the fact by editing the source.

Every backend is an OPTIONAL dependency: importing this module imports none of
them. The heavy import happens inside `load()`, so the unit suite runs on a
machine with no training stack. `MockSegmenter` provides the same interface for
tests and for exercising the pipeline without weights.
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


class MaskRCNNSegmenter:
    """torchvision Mask R-CNN. BSD-3-Clause. The DEFAULT backend.

    Chosen as the default for a prototype for three reasons that matter more
    than raw speed:

      * No new dependency and no licence question. torchvision is already
        required for Stage B, and BSD-3 imposes nothing on a commercial
        product.
      * Fine-tunes well on a small verified set, which is the situation here -
        a few hundred annotated frames, not a public benchmark.
      * Mature, stable ONNX export.

    It is NOT real-time on a Jetson Orin. That is an accepted trade for the
    prototype; `rfdetr` is the drop-in upgrade once the pipeline is proven,
    and swapping backends does not touch anything downstream of this file.

    Class indices are the ontology order shifted by one, because torchvision
    detection models reserve index 0 for background."""

    def __init__(self, weights, conf=0.25, device="cpu", max_det=300,
                 mask_threshold=0.5):
        self.weights = str(weights)
        self.conf, self.device = conf, device
        self.max_det, self.mask_threshold = max_det, mask_threshold
        self._model = None

    @staticmethod
    def build(num_classes=None, pretrained=False):
        """A Mask R-CNN head sized for this ontology (+1 for background)."""
        try:
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        except ImportError as e:
            raise ImportError(
                "torchvision is required for the maskrcnn backend:\n"
                "    python -m pip install -r requirements-training.txt"
            ) from e
        n = (num_classes or len(CLASSES)) + 1          # +1 background
        weights = "DEFAULT" if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=weights, weights_backbone=None)
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, n)
        in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, n)
        return model

    def load(self):
        import torch
        p = Path(self.weights)
        if not p.exists():
            raise FileNotFoundError(
                f"segmentation weights not found: {p}\n"
                f"Train first:\n"
                f"  python -m seeweed3d.training.train_seg_torchvision "
                f"--dataset <prepared> --images-root <sessions> --out <dir>")
        blob = torch.load(p, map_location=self.device, weights_only=False)
        self._model = self.build(len(blob.get("classes", CLASSES)))
        self._model.load_state_dict(blob["model"])
        self._model.to(self.device).eval()
        return self

    def __call__(self, bgr):
        import torch
        if self._model is None:
            self.load()
        rgb = bgr[:, :, ::-1].copy().astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        with torch.no_grad():
            out = self._model([t])[0]

        h, w = bgr.shape[:2]
        keep = out["scores"] >= self.conf
        scores = out["scores"][keep][: self.max_det].cpu().numpy()
        if not len(scores):
            return Detections(np.zeros((0, h, w), bool), np.zeros((0, 4)),
                              np.zeros((0,), int), np.zeros((0,)), w, h)
        masks = (out["masks"][keep][: self.max_det, 0]
                 >= self.mask_threshold).cpu().numpy()
        xyxy = out["boxes"][keep][: self.max_det].cpu().numpy()
        boxes = np.stack([xyxy[:, 0], xyxy[:, 1],
                          xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
        # Undo the background offset so downstream indices are ontology order.
        labels = out["labels"][keep][: self.max_det].cpu().numpy() - 1
        return Detections(masks, boxes, labels.astype(int),
                          scores.astype(float), w, h)


class RFDETRSegmenter:
    """Roboflow RF-DETR-Seg. Apache-2.0. The real-time upgrade path.

    Apache-2.0 imposes no source-release obligation, so unlike the Ultralytics
    backend this one can ship in a commercial product as-is. Real-time with
    TensorRT FP16 and sized nano..2XL, so a model can be matched to the Orin's
    budget rather than the other way round.

    Not the default only because it is an extra dependency the prototype does
    not need; promote it by passing backend='rfdetr' once the pipeline is
    proven end to end."""

    def __init__(self, weights, conf=0.25, device="cpu", max_det=300,
                 resolution=None):
        self.weights = str(weights)
        self.conf, self.device = conf, device
        self.max_det, self.resolution = max_det, resolution
        self._model = None

    def load(self):
        try:
            from rfdetr import RFDETRSegPreview  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "rfdetr is not installed. It is OPTIONAL:\n"
                "    python -m pip install rfdetr\n"
                "RF-DETR-Seg is Apache-2.0, so it carries no source-release "
                "obligation - unlike the ultralytics backend."
            ) from e
        from rfdetr import RFDETRSegPreview
        kw = {"pretrain_weights": self.weights} if Path(self.weights).exists() \
            else {}
        if self.resolution:
            kw["resolution"] = self.resolution
        self._model = RFDETRSegPreview(**kw)
        return self

    def __call__(self, bgr):
        if self._model is None:
            self.load()
        import cv2
        h, w = bgr.shape[:2]
        det = self._model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                  threshold=self.conf)
        # rfdetr returns a supervision.Detections; normalise it to ours.
        masks = getattr(det, "mask", None)
        if masks is None or not len(det):
            return Detections(np.zeros((0, h, w), bool), np.zeros((0, 4)),
                              np.zeros((0,), int), np.zeros((0,)), w, h)
        xyxy = np.asarray(det.xyxy, float)
        boxes = np.stack([xyxy[:, 0], xyxy[:, 1],
                          xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
        return Detections(np.asarray(masks, bool), boxes,
                          np.asarray(det.class_id, int),
                          np.asarray(det.confidence, float), w, h)


# Backend registry. `ultralytics` is present but never selected by default -
# choosing it is an explicit, licence-relevant decision.
BACKENDS = {
    "maskrcnn": (MaskRCNNSegmenter, "BSD-3-Clause"),
    "rfdetr": (RFDETRSegmenter, "Apache-2.0"),
    "ultralytics": (UltralyticsSegmenter, "AGPL-3.0"),
}
DEFAULT_BACKEND = "maskrcnn"
PERMISSIVE_BACKENDS = ("maskrcnn", "rfdetr", "rtmdet")


def build_segmenter(backend=DEFAULT_BACKEND, weights=None, **kw):
    """Construct a backend by name, warning loudly about a copyleft choice.

    The warning is printed rather than raised: using Ultralytics is legitimate
    for research or for an AGPL-licensed project, and refusing outright would
    be making the user's licensing decision for them. Making it silent, though,
    would let an obligation attach to a product without anyone noticing."""
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown segmentation backend {backend!r}. Available: "
            f"{sorted(BACKENDS)} (default {DEFAULT_BACKEND!r}).")
    cls, licence = BACKENDS[backend]
    if backend not in PERMISSIVE_BACKENDS:
        print(f"[LICENCE WARNING] segmentation backend '{backend}' is "
              f"{licence}. Commercial or proprietary use requires a separate "
              f"licence from its authors, and AGPL otherwise obliges releasing "
              f"the complete source of the larger work. Permissive "
              f"alternatives: {', '.join(PERMISSIVE_BACKENDS)}.")
    return cls(weights, **kw)


class MockSegmenter:
    """Fixed Detections, for tests and for running the pipeline with no weights.

    Exists so the end-to-end pipeline, including every safety path, is testable
    without a GPU or a trained checkpoint - the segmenter is the only component
    that genuinely needs them."""

    def __init__(self, detections):
        self.detections = detections

    def __call__(self, bgr):
        return self.detections
