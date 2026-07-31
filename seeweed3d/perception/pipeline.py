#!/usr/bin/env python3
"""
SeeWeed3D - full RGB-D inference: segmentation -> batched LEP -> 3D -> safety.

ORDER OF OPERATIONS, AND WHY
----------------------------
1. Segment the whole frame ONCE.
2. Union every predicted crop mask into the onion safety mask BEFORE any weed
   is considered. Crop safety is a frame-level property; deciding it per weed
   would let an onion detected late in the loop fail to protect a weed decided
   early.
3. Build every weed ROI, then run the LEP network on the WHOLE BATCH. A dense
   frame holds 30-60 weeds; per-instance forward passes would spend most of the
   latency budget on kernel-launch overhead rather than arithmetic.
4. Localise in 3D and decide safety per weed.

The LEP model is optional. With none loaded the pipeline falls back to the
hand-engineered `perception/lep.py` estimator, so the system is runnable - and
comparable - before any learned model exists. That fallback is also the
baseline the learned model must beat.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402
from perception import safety as safety_mod  # noqa: E402
from perception.depth3d import localize_lep_3d  # noqa: E402
from perception.schema import (FrameResult, WeedTarget,  # noqa: E402
                               STATUS_ABSTAIN, STATUS_CANDIDATE)
from training import roi as roi_mod  # noqa: E402
from training.lep_targets import decode_lep  # noqa: E402


class InferencePipeline:
    """Stage A + Stage B + 3D + safety for one frame."""

    def __init__(self, segmenter, cfg, lep_model=None, torch_device="cpu",
                 fallback_estimator=None):
        self.segmenter = segmenter
        self.cfg = cfg
        self.lep_model = lep_model
        self.device = torch_device
        self._fallback = fallback_estimator

    # -- Stage B input assembly ------------------------------------------- #
    def _build_rois(self, bgr, det, idxs, depth_mm):
        h, w = bgr.shape[:2]
        rois, tfs = [], []
        for i in idxs:
            tf = roi_mod.make_transform(det.boxes[i], self.cfg.roi.out_size,
                                        self.cfg.roi.expand_ratio, w, h,
                                        self.cfg.roi.min_box_px)
            rois.append(roi_mod.extract_roi(bgr, det.masks[i], tf, depth_mm,
                                            self.cfg.roi.pad_value))
            tfs.append(tf)
        return rois, tfs

    def _batch_lep(self, rois):
        """One forward pass for every weed ROI in the frame."""
        import torch
        from training.lep_roinet import geometry_channels

        rgb = np.stack([r["rgb"] for r in rois]).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device)

        n_geom = geometry_channels(self.cfg.model.input_mode)
        geom = None
        if n_geom:
            stacks = [roi_mod.build_geometry_channels(r, self.cfg.depth)[:n_geom]
                      for r in rois]
            geom = torch.from_numpy(np.stack(stacks).astype(np.float32)).to(
                self.device)

        self.lep_model.eval()
        with torch.no_grad():
            out = self.lep_model.predict(rgb, geom)
        return (out["heatmap"].cpu().numpy()[:, 0],
                out["visibility"].cpu().numpy(),
                out["targetability"].cpu().numpy())

    def _fallback_lep(self, bgr, det, i, depth_mm, tf):
        """Hand-engineered estimator, used when no learned model is loaded."""
        from perception.lep import LEPEstimator, crop_context
        if self._fallback is None:
            self._fallback = LEPEstimator()
        x, y, bw, bh = [int(round(v)) for v in det.boxes[i]]
        ctx = crop_context(det.masks[i], bgr, (x, y, max(1, bw), max(1, bh)),
                           depth_full=depth_mm, pad=10,
                           class_name=det.class_name(i))
        r = self._fallback.estimate(ctx)
        if r is None:
            return None
        return {"uv_full": (float(r.uv[0]), float(r.uv[1])),
                "uv_roi": None, "uv_heatmap": None,
                "peak": float(r.confidence), "sigma_px": float(r.sigma_px),
                "covariance": r.covariance,
                "visibility": r.visibility}

    # -- main -------------------------------------------------------------- #
    def run(self, bgr, depth_mm=None, depth_valid=None, K=None,
            session_id="", frame_id=""):
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        det = self.segmenter(bgr)
        t_seg = time.perf_counter()

        onion = det.onion_safety_mask()
        result = FrameResult(session_id=session_id, frame_id=frame_id,
                             width=w, height=h, n_instances=len(det),
                             onion_mask_ref="onion_union" if onion is not None
                             else None,
                             onion_area_px=int(onion.sum()) if onion is not None
                             else 0)

        idxs = det.weed_indices()[: self.cfg.max_instances]
        if not idxs:
            result.timings_ms = {"segmentation": (t_seg - t0) * 1e3,
                                 "total": (time.perf_counter() - t0) * 1e3}
            return result

        rois, tfs = self._build_rois(bgr, det, idxs, depth_mm)
        t_roi = time.perf_counter()

        heatmaps = vis_p = tgt_p = None
        if self.lep_model is not None:
            heatmaps, vis_p, tgt_p = self._batch_lep(rois)
        t_lep = time.perf_counter()

        vis_names = ["visible", "partially_occluded_inferable", "not_visible"]
        tgt_names = ["yes", "no", "uncertain"]

        for k, i in enumerate(idxs):
            name = det.class_name(i)
            tgt = WeedTarget(
                session_id=session_id, frame_id=frame_id, instance_index=int(i),
                class_name=name, class_index=int(det.classes[i]),
                class_confidence=float(det.scores[i]),
                bbox_xywh=[float(v) for v in det.boxes[i]],
                mask_ref=f"inst_{int(i)}",
                mask_area_px=int(det.masks[i].sum()))

            if heatmaps is not None:
                lep = decode_lep(heatmaps[k], tfs[k], self.cfg.heatmap)
                vi = int(np.argmax(vis_p[k]))
                ti = int(np.argmax(tgt_p[k]))
                tgt.visibility = vis_names[vi] if vi < len(vis_names) else "unknown"
                tgt.targetable = tgt_names[ti] if ti < len(tgt_names) else "unknown"
                tgt.visibility_probs = [float(v) for v in vis_p[k]]
                tgt.targetability_probs = [float(v) for v in tgt_p[k]]
                vis_conf = float(vis_p[k][vi])
                tgt_conf = float(tgt_p[k][ti])
                weight_full = None
            else:
                lep = self._fallback_lep(bgr, det, i, depth_mm, tfs[k])
                tgt.visibility = (lep or {}).get("visibility", "unknown")
                tgt.targetable = "unknown"
                vis_conf = tgt_conf = None
                weight_full = None

            if lep is not None:
                tgt.lep_uv = [float(lep["uv_full"][0]), float(lep["uv_full"][1])]
                tgt.lep_peak = float(lep.get("peak", 0.0))
                tgt.lep_sigma_px = float(lep.get("sigma_px", 0.0))
                tgt.lep_covariance = lep.get("covariance")

            depth_res = None
            if depth_mm is not None and K is not None and lep is not None:
                valid = depth_valid if depth_valid is not None else (
                    np.isfinite(depth_mm) & (np.nan_to_num(depth_mm) > 0))
                depth_res = localize_lep_3d(
                    depth_mm, valid, lep["uv_full"], K,
                    weight_map=weight_full, mask=det.masks[i],
                    min_valid_fraction=self.cfg.safety.min_depth_valid_fraction,
                    max_spread_mm=self.cfg.safety.max_depth_spread_mm)
                tgt.used_depth = True
                st = depth_res.get("depth_stats", {}) or {}
                tgt.depth_stats = st
                tgt.depth_valid_fraction = float(st.get("valid_fraction", 0.0))
                tgt.depth_spread_mm = st.get("spread_mm")
                if depth_res.get("ok"):
                    tgt.xyz_mm = depth_res["xyz_mm"]
                    tgt.xyz_sigma_mm = depth_res.get("sigma_mm")
                    tgt.xyz_covariance = depth_res.get("covariance")

            decision = safety_mod.decide(
                class_name=name, class_confidence=float(det.scores[i]),
                lep=lep, instance_mask=det.masks[i], onion_mask=onion,
                cfg=self.cfg.safety, visibility=tgt.visibility,
                visibility_conf=vis_conf, targetable=tgt.targetable,
                targetable_conf=tgt_conf, depth_result=depth_res)

            tgt.safety_status = (STATUS_CANDIDATE if decision.is_candidate
                                 else STATUS_ABSTAIN)
            tgt.abstained = decision.abstained
            tgt.rejection_reasons = list(decision.reasons)
            tgt.safety_notes = dict(decision.notes)
            tgt.lep_snapped_px = float(decision.snapped_px)
            result.targets.append(tgt)

        t_end = time.perf_counter()
        result.timings_ms = {
            "segmentation": (t_seg - t0) * 1e3,
            "roi_build": (t_roi - t_seg) * 1e3,
            "lep_batch": (t_lep - t_roi) * 1e3,
            "depth_and_safety": (t_end - t_lep) * 1e3,
            "total": (t_end - t0) * 1e3,
            "n_weed_rois": len(idxs)}
        return result
