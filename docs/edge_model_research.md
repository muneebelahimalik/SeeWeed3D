# Edge model research — real-time segmentation + LEP on Jetson Orin

Research record, August 2026. Every figure is attributed; anything estimated is
labelled as such. **No number in this document was measured on our hardware or
our data.** They are vendor and paper figures used to rank candidates, not
predictions of what SeeWeed3D will do.

---

## 1. What the robot actually has to do

One frame, end to end, before the delta arm can fire:

```
ZED 2i capture → rectify + depth → instance segmentation (crop vs weed)
    → per-weed LEP (u,v) → depth sample at LEP → 3D point in camera frame
    → transform to arm frame → safety abstention → aim → dwell → next
```

The **latency budget** is set by ground speed and the laser's dwell time, not by
what looks fast in a benchmark. The governing constraint:

```
    perception_latency  <  (targeting_window_m / ground_speed_m_s) − dwell_s
```

A weed must still be inside the arm's reach when the aim command lands. At
1.0 m/s with a 0.4 m working window and 0.35 s of dwell per weed, perception has
**~50 ms** — and that is the *whole* chain, not just the segmentation forward
pass. Fill in your own numbers before treating any model below as fast enough.

**Measure, don't extrapolate.** `seeweed3d/deploy/benchmark.py` exists for
exactly this. Vendor latencies below are T4 numbers; a Jetson is a different
architecture with different memory bandwidth, and the mapping is not a constant
factor.

---

## 2. Current state, and why it is not the answer

`MaskRCNNSegmenter` (torchvision, BSD-3-Clause) is the default backend, chosen
for the prototype because it needs no new dependency, no licence question, and
fine-tunes well on a few dozen frames.

It is **not real-time on an Orin** and was never claimed to be. Two-stage
detectors with a per-ROI mask head are the wrong shape for this: cost scales
with instance count, which in a dense weed frame is precisely when you need it
to be fast. It stays as the training-wheels backend and the numerical reference.

---

## 3. Recommended upgrade: RF-DETR-Seg

Already wired as the `rfdetr` backend in `perception/segmenter.py`. The full
variant family — Nano through 2XL — was released **January 2026**; a preview
base model landed October 2025.

Roboflow's published checkpoints, COCO instance segmentation, latency on an
**NVIDIA T4 with TensorRT FP16**:

| Variant | COCO AP50 | AP50:95 | Latency | Params | Resolution |
|---|---|---|---|---|---|
| Seg-N | 63.0 | 40.3 | 3.4 ms | 33.6 M | 312² |
| Seg-S | 66.2 | 43.1 | 4.4 ms | 33.7 M | 384² |
| **Seg-M** | 68.4 | 45.3 | 5.9 ms | 35.7 M | 432² |
| Seg-L | 70.5 | 47.1 | 8.8 ms | 36.2 M | 504² |
| Seg-XL | 72.2 | 48.8 | 13.5 ms | 38.1 M | 624² |
| Seg-2XL | 73.1 | 49.9 | 21.8 ms | 38.6 M | 768² |

Architecture: DINOv2 ViT backbone, DETR-style set prediction, segmentation head
inspired by MaskDINO. End-to-end — **no NMS**, which removes a tuning knob and a
source of latency variance under dense instances. Cost is roughly flat in
instance count, the opposite of Mask R-CNN.

**Start at Seg-M or Seg-S.** The accuracy curve flattens well before 2XL while
latency more than triples, and resolution is the real lever for small weeds.

### Licence — verify before shipping

The `rfdetr` package and Apache-designated models are Apache-2.0. Sources
conflict on the top two segmentation variants: the release blog states Nano
through Large are Apache-2.0 with **XL and 2XL detection** models under Platform
Model License 1.0, while the repository description reads as though all
segmentation variants are Apache-2.0. **Confirm the licence on the specific
checkpoint you ship.** Nano/Small/Medium/Large are unambiguous, and those are
the ones worth using here anyway.

### Resolution is the catch

Seg-M runs at 432². ZED 2i frames are far larger, and a cotyledon-stage weed
that is 40 px across at full resolution is ~8 px at 432². Options, in order of
preference:

1. **Tile the frame.** Two or three overlapping crops at native scale, batched.
   Costs linearly but preserves the small weeds that matter most.
2. **Raise the variant** (Seg-XL at 624², Seg-2XL at 768²) and pay the latency.
3. **Narrow the field of view** by mounting the camera lower — changes the
   geometry, not the model.

Decide this by measuring small-weed recall (already reported by `eval_seg.py`),
not by eyeballing full-frame overlays.

---

## 4. The interesting finding: RF-DETR Keypoint

Roboflow released **RF-DETR Keypoint** as a preview: 71.8 COCO Keypoint AP at
576², 9.8 ms on T4 TensorRT FP16, beating YOLO26x-pose on both accuracy and
speed. A keypoint head built into the detection transformer predicts keypoints
per detected object in a **single forward pass** — no NMS, no heatmaps, no
post-hoc grouping.

Critically: **it is not limited to human skeletons. You can fine-tune on any
keypoint layout attached to any object class.**

An LEP is exactly that — one keypoint, on a weed. This is a direct match for a
consolidation that would remove an entire stage:

```
today:     segment (Stage A) → crop each weed to an ROI → LEPRoiNet (Stage B)
possible:  one network → boxes + masks + LEP, one forward pass
```

That would eliminate the ROI crop, the per-instance batching, and a whole second
model's worth of latency and failure modes.

### Why this is not the plan yet

**Segmentation and keypoints are separate model variants, not one checkpoint.**
The repository exposes `RFDETRSegNano…Seg2XLarge` and `RFDETRKeypointPreview` as
distinct classes. There is no shipped RF-DETR that emits masks and keypoints
together, so today the choice is:

- **masks + separate LEP stage** (what we have), or
- **boxes + keypoints, no masks** — and masks are not optional, because the
  crop-safety mask is what the laser decision consults

Running both models is worse than the current design: two transformer forward
passes instead of one CNN plus a 0.7 M-parameter ROI head.

**The recommendation is to watch this.** Roboflow states more keypoint
checkpoints are coming. If a seg+keypoint variant ships, it collapses Stage A
and Stage B into one model and it is worth restructuring for. Until then,
`LEPRoiNet` stays — and note it is only **701,447 parameters**, so Stage B is
not where the latency is.

---

## 5. Also evaluated

| Model | Licence | Verdict |
|---|---|---|
| **RTMDet-Ins** (MMDetection) | Apache-2.0 | Genuinely strong: 52.8 AP, 300+ FPS on a 3090. Real-time instance segmentation SOTA when published. Rejected on **integration cost** — MMDetection pulls in `mmcv`, `mmengine` and a config system, and mmcv's CUDA build is a recurring pain on Jetson. Reconsider if RF-DETR disappoints. |
| **YolactEdge** | permissive | 30.8 FPS on AGX Xavier at 550², explicitly designed for Jetson with TensorRT and temporal feature warping. Older and less accurate than the above; the temporal warping idea is worth stealing for a continuously-driving robot even if the model is not used. |
| **Ultralytics YOLO26-seg** | **AGPL-3.0** | Fast and easy. Already implemented as the `ultralytics` backend and **not the default, for licence reasons** — AGPL obligations attach to a commercial product. Research use only. |
| **SAM 3 / FastSAM / MobileSAM** | varies | Prompt-driven and class-agnostic. Right for the *annotation* pipeline, where SAM 3 already earns its place. Wrong for inference — the robot needs class decisions (crop vs weed), which is what the fine-tuned model provides. |

---

## 6. Deployment stack

**JetPack 7.2** brings the full JetPack 7 line to the entire Orin family — AGX
Orin, Orin NX, Orin Nano — on Jetson Linux 39.2, Linux kernel 6.8, Ubuntu 24.04
LTS, **CUDA 13.2.1 and TensorRT 10.16.2**.

For reference sizing: the Orin Nano Super delivers up to 67 TOPS with 8 GB
LPDDR5. On Orin NX, TensorRT FP16 has been reported at 27 ms/image (≈37 FPS) for
YOLOv8-class detection — useful only as an order-of-magnitude sanity check.

### Use FP16. Do not reach for INT8 on the transformer.

This is the least obvious finding in this document and it will save you days.

A developer reported INT8 quantization of a **ViT-S + DPT** model on Orin Nano
running **2.7× slower** than FP16 — 41.8 ms → 114.5 ms. Isolated, the ViT gained
only ~1.3% from INT8, indicating the underlying kernels get little benefit from
INT8 for transformer architectures on this hardware. NVIDIA identified a
TensorRT bug as part of it and pointed to a future release.

RF-DETR is a **DINOv2 ViT backbone**. Assume the same class of behaviour applies
until measured. **FP16 is the practical optimum for transformers on Orin**, and
INT8 should be treated as an experiment to be verified with a stopwatch, never
as an assumed speedup.

Convolutional heads may still benefit from INT8 — quantize selectively and
measure each time.

### Engines are not portable

A TensorRT engine is specific to the GPU architecture, TensorRT version and
JetPack version. **Build on the Orin itself**, never on the desktop. The runbook
already says this; it is repeated here because it is the single most common way
a deployment day gets lost.

---

## 7. Where depth belongs

Depth is currently used in Stage B (`rgb_mask_geom`) and in 3D localization, and
**not** in Stage A or the vegetation prior. Ranked by value per unit of risk:

### 7.1 Height gate on the vegetation prior — do this first
Soil and plant are separable by **height above the local soil plane** far more
reliably than by colour under changing light. This is the cheapest real win, and
it is a filter, not a learned component, so it cannot silently degrade.

Blocked on one measurement: `median_depth_valid_frac_veg`. The ZED 2i's depth is
unreliable on thin leaf edges and in bright sun, and gating on invalid depth
would delete exactly the small weeds that matter. **Measure the valid fraction
over vegetation pixels before enabling any depth gate**, and make the gate
fail-open where depth is invalid.

### 7.2 Depth as a 4th input channel to Stage A
The literature supports it — WeedsNet (Precision Agriculture, 2024) uses a dual
attention network on RGB-D for wheat-field weed detection; FuseNet/RedNet-style
architectures run RGB and depth through separate branches fused before
upsampling; CMFNet (2024) does transformer cross-modal fusion for field weed
mapping with RGB+NIR.

Two cautions before adopting it:

- **Encode height above local soil, not raw depth.** Raw depth encodes camera
  mount height, so a model trained on it will not transfer to a different rig.
  `roi.py::local_height_map()` already does this for Stage B; reuse it.
- **Train with depth dropout.** Stage B already applies p=0.25. Without it, the
  model learns to depend on a channel that fails in bright sun — and it will
  fail on exactly the sunny days you most want to be weeding.

Note this conflicts with using a pretrained RF-DETR backbone, which expects 3
channels. A separate depth branch fused later is the compatible pattern, at the
cost of a custom architecture and losing some pretrained benefit. **Not worth it
until Stage A's RGB-only ceiling is measured.**

### 7.3 Already in place
Bimodality-checked depth sampling around the LEP, MAD outlier rejection, plane
fallback, and pinhole uncertainty propagation live in `perception/depth3d.py`.
That is the part where depth errors turn into a laser pointed at the wrong
tissue, and it is the part that is most carefully guarded.

---

## 8. Recommended sequence

1. **Now** — train Mask R-CNN on the 45 frames. Establish that the data is
   learnable and get the eval table. Change nothing else.
2. **Next** — active learning to 150–200 frames across ≥3 sessions. Everything
   below is bottlenecked on data, not architecture. A held-out *session* becomes
   possible here, and only then do any numbers mean anything.
3. **Then** — swap in RF-DETR-Seg-M via the existing `rfdetr` backend. Compare
   against Mask R-CNN on the same split with `eval_seg.py`.
4. **Then** — export to ONNX, build a TensorRT FP16 engine **on the Orin**, and
   run `deploy/benchmark.py`. This is where the real latency number appears and
   where the resolution/tiling decision gets made on evidence.
5. **Measure** `median_depth_valid_frac_veg`, then decide on the height gate.
6. **Watch** RF-DETR Keypoint for a seg+keypoint variant. If it ships, Stage A
   and Stage B collapse into one model.

Step 2 is the one that matters. No architecture on this page fixes 45 frames
from a single session.

---

## Sources

- [RF-DETR documentation — segmentation benchmarks](https://rfdetr.roboflow.com/develop/)
- [New RF-DETR Segmentation Checkpoints from Nano to 2XLarge — Roboflow](https://blog.roboflow.com/rf-detr-segmentation/)
- [RF-DETR Keypoint Preview — Roboflow](https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/)
- [roboflow/rf-detr — GitHub](https://github.com/roboflow/rf-detr)
- [JetPack SDK downloads and release notes — NVIDIA](https://developer.nvidia.com/embedded/jetpack)
- [JetPack 7.2 and Yocto — Connect Tech](https://connecttech.com/jetpack-7-2-yocto/)
- [INT8 quantization performance regression on Jetson Orin Nano (ViT-S + DPT) — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835)
- [RTMDet: An Empirical Study of Designing Real-Time Object Detectors — arXiv 2212.07784](https://arxiv.org/abs/2212.07784)
- [YolactEdge: Real-time Instance Segmentation on the Edge — arXiv 2012.12259](https://arxiv.org/abs/2012.12259)
- [Benchmarking YOLOv8 Variants on Jetson Orin NX — MDPI Computers](https://www.mdpi.com/2073-431X/15/2/74)
- [Multi-Modal Deep Learning for Weeds Detection in Wheat Field Based on RGB-D Images — Frontiers in Plant Science](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.732968/full)
- [Beyond Color: Advanced RGB-D data augmentation for robust semantic segmentation in crop farming — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S016816992600027X)
