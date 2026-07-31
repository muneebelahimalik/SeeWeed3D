# SeeWeed3D

Perception and 3D growth-point localization for a robotic laser-weeding
platform in Vidalia onion production. The system distinguishes onions from
weeds, localizes each weed's biologically effective laser-treatment point
(the LEP/AMT), and turns it into a 3D coordinate in the robot frame — while
treating **crop safety** (never targeting onion tissue) as the primary outcome.

This repository currently covers the **data pipeline**: field capture, dataset
extraction, annotation-batch selection, and model-assisted onion prelabeling.
Perception, localization, tracking, and evaluation are the next stages.

## Repository layout

```
seeweed3d/
  capture/      zed_capture.py            ZED field capture (v2: SVO + depth + confidence + logs)
  extraction/   extract_sessions.py       recordings -> indexed, QC'd frame pool (v1 & v2)
                curate_pool.py             drop redundant/bad frames (manifest-only, reversible)
                select_batches.py          pool -> CVAT-ready annotation batches (holdout-safe)
  annotation/   prelabel_onions_sam3.py    SAM 3 onion prelabels for onion-only scenes
                prelabel_weeds_sam3.py     SAM 3 weed instances + morphology + LEP proposals
                cvat_roundtrip.py          CVAT export -> training masks + auto-vs-verified IoU
                regen_cvat_labels.py       refresh label schema files without re-running SAM 3
  perception/   lep.py                     multi-evidence LEP estimator (hand-engineered baseline)
                segmenter.py               pluggable seg backends -> framework-free Detections
                pipeline.py                full RGB-D inference: seg -> batched LEP -> 3D -> safety
                depth3d.py                 robust LEP depth sampling + 3D point with uncertainty
                safety.py                  treatment-candidate decision (can only ABSTAIN)
                schema.py                  structured WeedTarget / FrameResult output
  training/     config.py                  dataclass configuration for every stage
                datumaro_multitask.py      CVAT Datumaro 1.0 -> masks + grouped LEPs + report
                splits.py                  session-safe splits with leakage checks
                roi.py                     invertible ROI transform + local-height geometry
                lep_targets.py             Gaussian targets, soft-argmax, joint augmentation
                lep_roinet.py              LEPRoiNet (heatmap + visibility + targetability)
                losses.py                  multitask LEP losses
                lep_dataset.py             torch Dataset driven by the LEP manifest
                prepare_dataset.py         entry point: export -> trainable dataset
                seg_dataset.py             torchvision-format instance segmentation dataset
                train_seg_torchvision.py   Stage A training, BSD-3 backend (default)
                train_seg.py / train_lep.py  Ultralytics (AGPL, opt-in) / LEP training
  evaluation/   metrics.py                 segmentation, LEP, safety, 3D and latency metrics
  deploy/       export.py                  ONNX/TensorRT export with numerical parity checks
                benchmark.py               latency benchmarking (records the device)
  common/       ontology.py                class names + stable COCO ids (single source of truth)
                progress.py                dependency-free progress lines with rate + ETA
                vegetation.py              shared ExG vegetation prior + white balance
                depth_utils.py             canonical depth reader + robust 3D point sampling
  validation/   depth_data_validation.py   sanity-check a raw session's depth stream
docs/           pipeline, capture, and prelabeling guides
legacy/         superseded scripts, kept for provenance
tests/          synthetic end-to-end checks for the extraction + prelabel logic
```

Each runnable script has a clearly marked **CONFIG block at the top** — set the
input/output paths there and run it. No command-line flags are required.

## Pipeline

```bash
# 1. Capture (in the field, on the ZED rig)
python seeweed3d/capture/zed_capture.py

# 2. Extract recordings into an indexed, QC'd dataset
python seeweed3d/extraction/extract_sessions.py

# 3. (optional) Drop overlapping/bad frames - edits pool.csv only, never files
python seeweed3d/extraction/curate_pool.py

# 4. Select CVAT-ready annotation batches (whole-session holdout enforced)
python seeweed3d/extraction/select_batches.py

# 5. (onion-only scenes) Prelabel onions with SAM 3, then verify in CVAT
python seeweed3d/annotation/prelabel_onions_sam3.py
```

**→ [`docs/lep_localization_explained.md`](docs/lep_localization_explained.md)** explains LEP
localization in full technical detail — biology, both estimators, the math,
uncertainty, 3D conversion and the safety rules.

**→ [`docs/RUNBOOK.md`](docs/RUNBOOK.md) is the complete end-to-end guide**: every
step from raw recordings through SAM 3 prelabeling, CVAT annotation, merging
multiple CVAT tasks, training both stages, evaluation, inference and Jetson
deployment — with all commands.

See `docs/dataset_pipeline.md` for the full extraction/selection guide and
`docs/onion_prelabeling.md` for the SAM 3 workflow. Capture rationale and the
v1→v2 changes are in `docs/capture_changelog.md`.

## Supervised perception baseline — status

The supervised stage (crop-vs-weed segmentation, learned LEP localization, RGB-D
3D conversion, safety abstention) is implemented and unit-tested. **It has not
been trained.** Full design, commands and limitations:
`docs/supervised_perception_baseline.md`.

| | Status |
|---|---|
| Datumaro multitask ingestion, contract validation, session-safe splits | **implemented, unit-tested** |
| ROI transform, heatmap targets, joint augmentation, depth representation | **implemented, unit-tested** |
| `LEPRoiNet` + losses (forward, backward, CPU training step, ONNX export) | **implemented, unit-tested** (torch 2.13 CPU) |
| Depth→3D localization, safety abstention, structured output, full pipeline | **implemented, unit-tested** with mocked segmentation + synthetic RGB-D |
| Evaluation metrics (segmentation, LEP, safety, 3D, latency) | **implemented, unit-tested** on synthetic inputs |
| Stage A backends: `maskrcnn` (BSD-3, **default**), `rfdetr` (Apache-2.0), `ultralytics` (AGPL, opt-in) | **implemented, unit-tested**; Mask R-CNN train step + inference verified on CPU |
| Stage A / Stage B **trained weights** | **requires verified CVAT annotations** — none exist yet |
| Any accuracy / mAP / LEP-error number | **not measured** — nothing is trained |
| TensorRT engines, INT8 | **requires a GPU / Jetson Orin** — engines must be built on the device |
| Jetson latency, GPU memory | **not measured** — desktop numbers do not transfer |
| DINOv3 teacher / distillation, temporal tracking | **future work**, not started |

```bash
# Optional training/deployment stack. The unit suite does NOT need it.
python -m pip install -r requirements-training.txt
```

## Key invariants (read before touching data)

- **Depth is raw millimetres, `0` = invalid.** Never rescale by the old
  `depth_vis_max_mm` GUI constant. Use `seeweed3d/common/depth_utils.load_depth_mm()`.
- **Frames are paired by index, not time.** Capture can drop frames while the
  encoder assumes constant fps, so video time ≠ real time; all streams share the
  video frame index.
- **Split by whole session, never by frame.** Adjacent video frames are
  near-identical; a random split leaks the test set into training.
- **`vegetation == onion` only in onion-only scenes.** In mixed scenes a missed
  onion must never become a weed label.

## Requirements

- Python 3.10+, `ffmpeg`/`ffprobe` on PATH, and the packages in
  `requirements.txt`.
- Capture additionally needs the **ZED SDK** + `pyzed` (installed via the SDK,
  not pip).
- Onion prelabeling additionally needs Meta's official `sam3` package (`pip
  install "git+https://github.com/facebookresearch/sam3.git"`, which pulls in
  `torch`) and access to the gated `facebook/sam3` / `facebook/sam3.1` weights.

```bash
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/ -v
```
The tests build synthetic v1/v2 sessions with real FFV1 MKVs and run the
extraction and prelabel logic end-to-end (lossless depth round-trip, index
alignment, dropped-frame recovery, holdout isolation, reproducible selection,
and COCO validity). SAM 3 itself is stubbed — it needs a GPU and gated weights.
