#!/usr/bin/env python3
"""
SeeWeed3D - multitask losses for LEPRoiNet.

Five terms, each with a stated job. Deliberately few: a long list of
unvalidated auxiliary losses makes a regression impossible to attribute, and
every weight here has to be defensible in a paper.

  heatmap       Per-pixel localisation on the Gaussian target. The main signal.
  soft_argmax   Direct error on the decoded coordinate. The heatmap loss alone
                is happy with a slightly asymmetric blob whose centre of mass is
                off; this term penalises exactly the quantity that is actually
                used at inference.
  visibility    3-way classification, and the gate for abstention.
  targetability 3-way classification, the annotator's own treat/don't verdict.
  outside_mask  Probability mass placed outside the owning instance. This is
                what teaches "the growth point belongs to THIS plant" - the
                difference between a safe target and the neighbour's crown.

Per-sample weights (from `lep_targets.target_weight`) mask the localisation
terms for `not_visible` samples: there is no LEP to regress, and supervising
one would be inventing ground truth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_argmax_2d(heatmap, normalize=True, eps=1e-8):
    """Differentiable expected coordinate of each heatmap in a batch.

    heatmap: (B, 1, H, W) probabilities. Returns (B, 2) as (x, y) in heatmap
    pixels. Mirrors `lep_targets.soft_argmax` so the training objective and the
    deployed decode agree by construction."""
    b, _, h, w = heatmap.shape
    flat = heatmap.reshape(b, -1)
    if normalize:
        flat = flat.clamp_min(0.0)
        flat = flat / (flat.sum(1, keepdim=True) + eps)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=heatmap.device, dtype=heatmap.dtype),
        torch.arange(w, device=heatmap.device, dtype=heatmap.dtype),
        indexing="ij")
    x = (flat * xs.reshape(1, -1)).sum(1)
    y = (flat * ys.reshape(1, -1)).sum(1)
    return torch.stack([x, y], 1)


class LEPLoss(nn.Module):
    """Weighted sum of the five terms. Returns (total, parts) so each can be
    logged separately - a total alone hides which task regressed."""

    def __init__(self, weights):
        super().__init__()
        self.w = weights

    def forward(self, outputs, targets):
        """
        outputs: heatmap (B,1,h,w) logits, visibility (B,Cv), targetability (B,Ct)
        targets: heatmap (B,1,h,w), coord (B,2) in heatmap px, weight (B,),
                 visibility (B,) long, targetability (B,) long,
                 mask (B,1,h,w) owning-instance mask at heatmap resolution
        """
        parts = {}
        logits = outputs["heatmap"]
        prob = torch.sigmoid(logits)
        w = targets["weight"].reshape(-1, 1, 1, 1).to(prob.dtype)
        denom = w.sum().clamp_min(1e-6)

        # Localisation: MSE on the Gaussian, weighted per sample.
        se = (prob - targets["heatmap"]) ** 2
        parts["heatmap"] = (se * w).sum() / (denom * se.shape[-1] * se.shape[-2])

        # Coordinate error on the decoded point.
        pred_xy = soft_argmax_2d(prob)
        l1 = F.smooth_l1_loss(pred_xy, targets["coord"], reduction="none").sum(1)
        parts["soft_argmax"] = (l1 * targets["weight"].to(l1.dtype)).sum() / \
            targets["weight"].sum().clamp_min(1e-6)

        # Classification heads: supervised for every sample, including
        # not_visible - that verdict is exactly what the head must learn.
        parts["visibility"] = F.cross_entropy(outputs["visibility"],
                                              targets["visibility"])
        parts["targetability"] = F.cross_entropy(outputs["targetability"],
                                                 targets["targetability"])

        # Ownership: penalise probability outside the owning mask.
        if "mask" in targets and targets["mask"] is not None:
            outside = prob * (1.0 - targets["mask"].to(prob.dtype))
            parts["outside_mask"] = (outside * w).sum() / denom
        else:
            parts["outside_mask"] = prob.sum() * 0.0

        total = (self.w.heatmap * parts["heatmap"]
                 + self.w.soft_argmax * parts["soft_argmax"]
                 + self.w.visibility * parts["visibility"]
                 + self.w.targetability * parts["targetability"]
                 + self.w.outside_mask * parts["outside_mask"])
        return total, {k: float(v.detach()) for k, v in parts.items()}
