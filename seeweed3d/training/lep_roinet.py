#!/usr/bin/env python3
"""
SeeWeed3D - LEPRoiNet: per-instance LEP heatmap + visibility + targetability.

Runs on a BATCH of weed ROIs from one frame, not once per weed in a Python
loop: a dense frame holds 30-60 weeds, and per-instance kernel launches would
dominate the latency budget on a Jetson far more than the arithmetic does.

ARCHITECTURE AND WHY
--------------------
A small convolutional encoder-decoder, deliberately plain:

  * RGB encoder - depthwise-separable inverted-residual blocks, the
    MobileNetV3-style pattern. Chosen over a transformer for TensorRT: these
    ops have long-standing, well-optimised kernels, whereas attention still
    needs care to convert cleanly. This is the deployment constraint driving
    the design, not accuracy on a benchmark.
  * A SEPARATE geometry branch for [mask, height, depth_valid]. Depth is not
    concatenated as a fourth RGB channel: the first RGB conv would then mix a
    metric quantity into colour features at full resolution, and the network
    could not represent "depth is missing here" distinctly from "height is
    zero here". A separate branch fused at low resolution keeps the two
    modalities separable, which is what makes depth dropout survivable.
  * A light decoder producing ONE heatmap channel, and global heads for
    visibility and targetability.

The heatmap output is intentionally left as raw logits/activations, decoded by
`lep_targets.decode_lep()`. Keeping soft-argmax out of the graph keeps the
export ONNX/TensorRT-friendly and means the deployed decode is the same code
path as the training-time decode.

torch is imported lazily by the caller (this module needs it), so the rest of
the package stays importable in the lightweight environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _in_channels(input_mode):
    """RGB channels only. Geometry goes to its own branch."""
    return 3


def geometry_channels(input_mode):
    """How many geometry channels the configured ablation consumes.

    rgb            -> 0 (no geometry at all)
    rgb_mask       -> 1 (ownership mask only; works with no depth stream)
    rgb_mask_geom  -> 3 (mask, normalised height, depth validity)
    """
    return {"rgb": 0, "rgb_mask": 1, "rgb_mask_geom": 3}[str(input_mode)]


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.Hardswish(inplace=True) if act else nn.Identity())


class InvertedResidual(nn.Module):
    """MobileNetV3-style block: expand -> depthwise -> project.

    The residual is only added when shape permits, which is the standard
    formulation and keeps the export graph simple."""

    def __init__(self, cin, cout, stride=1, expand=4):
        super().__init__()
        mid = cin * expand
        self.use_res = (stride == 1 and cin == cout)
        self.block = nn.Sequential(
            ConvBNAct(cin, mid, 1),
            ConvBNAct(mid, mid, 3, stride, groups=mid),
            ConvBNAct(mid, cout, 1, act=False))

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class LEPRoiNet(nn.Module):
    """Per-ROI LEP localiser with visibility and targetability heads.

    Input:  rgb  (B, 3, S, S) float in [0,1]
            geom (B, G, S, S) or None, G = geometry_channels(input_mode)
    Output: dict with
            heatmap       (B, 1, S/stride, S/stride)
            visibility    (B, n_visibility)   logits
            targetability (B, n_targetability) logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.input_mode = str(cfg.input_mode)
        self.stride = int(cfg.heatmap_stride)
        w = int(cfg.width)
        self.n_geom = geometry_channels(self.input_mode)

        # RGB encoder: S -> S/2 -> S/4 -> S/8
        self.stem = ConvBNAct(_in_channels(self.input_mode), w, 3, 2)
        self.enc1 = nn.Sequential(InvertedResidual(w, w), InvertedResidual(w, w))
        self.down1 = InvertedResidual(w, w * 2, stride=2)
        self.enc2 = nn.Sequential(InvertedResidual(w * 2, w * 2),
                                  InvertedResidual(w * 2, w * 2))
        self.down2 = InvertedResidual(w * 2, w * 4, stride=2)
        self.enc3 = nn.Sequential(InvertedResidual(w * 4, w * 4),
                                  InvertedResidual(w * 4, w * 4))

        # Geometry branch, fused at S/8 where it is cheap and where the RGB
        # features are already semantic rather than raw colour.
        if self.n_geom:
            self.geom = nn.Sequential(
                ConvBNAct(self.n_geom, w, 3, 2),
                InvertedResidual(w, w * 2, stride=2),
                InvertedResidual(w * 2, w * 4, stride=2))
            self.fuse = ConvBNAct(w * 8, w * 4, 1)
        else:
            self.geom = None
            self.fuse = None

        # Decoder back up to S/stride.
        self.up1 = ConvBNAct(w * 4, w * 2, 3)
        self.up2 = ConvBNAct(w * 2 + w * 2, w * 2, 3)     # skip from enc2
        self.head = nn.Conv2d(w * 2, 1, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.vis_head = nn.Sequential(nn.Flatten(), nn.Linear(w * 4, w * 2),
                                      nn.Hardswish(inplace=True),
                                      nn.Linear(w * 2, int(cfg.n_visibility)))
        self.tgt_head = nn.Sequential(nn.Flatten(), nn.Linear(w * 4, w * 2),
                                      nn.Hardswish(inplace=True),
                                      nn.Linear(w * 2, int(cfg.n_targetability)))

    def forward(self, rgb, geom=None):
        x = self.stem(rgb)             # S/2
        x = self.enc1(x)
        x2 = self.enc2(self.down1(x))  # S/4
        x3 = self.enc3(self.down2(x2))  # S/8

        if self.geom is not None:
            if geom is None:
                # Depth/mask stream absent at runtime: feed zeros so the graph
                # is static (TensorRT wants fixed shapes) and the depth-dropout
                # training makes this a case the network has already seen.
                geom = rgb.new_zeros((rgb.shape[0], self.n_geom,
                                      rgb.shape[2], rgb.shape[3]))
            g = self.geom(geom)
            if g.shape[-2:] != x3.shape[-2:]:
                g = F.interpolate(g, size=x3.shape[-2:], mode="nearest")
            x3 = self.fuse(torch.cat([x3, g], 1))

        pooled = self.pool(x3)
        vis = self.vis_head(pooled)
        tgt = self.tgt_head(pooled)

        y = F.interpolate(self.up1(x3), size=x2.shape[-2:], mode="nearest")
        y = self.up2(torch.cat([y, x2], 1))
        heat = self.head(y)            # S/4

        target_hw = rgb.shape[-1] // self.stride
        if heat.shape[-1] != target_hw:
            heat = F.interpolate(heat, size=(target_hw, target_hw),
                                 mode="bilinear", align_corners=False)
        return {"heatmap": heat, "visibility": vis, "targetability": tgt}

    @torch.no_grad()
    def predict(self, rgb, geom=None):
        """Inference convenience: sigmoid on the heatmap, softmax on the heads."""
        out = self(rgb, geom)
        return {"heatmap": torch.sigmoid(out["heatmap"]),
                "visibility": torch.softmax(out["visibility"], -1),
                "targetability": torch.softmax(out["targetability"], -1)}


def build_model(cfg):
    return LEPRoiNet(cfg)
