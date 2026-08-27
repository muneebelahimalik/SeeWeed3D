# SeeWeed3D

Perception and 3D growth-point localization for a robotic laser-weeding
platform in Vidalia onion production. The system distinguishes onions from
weeds, localizes each weed's biologically effective laser-treatment point
(the LEP/AMT), and turns it into a 3D coordinate in the robot frame — while
treating **crop safety** (never targeting onion tissue) as the primary outcome.

The repository covers the pipeline end to end: field capture, extraction and
curation, SAM 3 prelabeling, CVAT round trip, dataset building, Stage A
segmentation training on two backends, evaluation, and inference. Stage B (the
learned LEP head) is implemented and not yet trained.

**→ [`CHANGELOG.md`](CHANGELOG.md) is the project history** — what changed, when,
and the problem that forced each change. Several settings here look arbitrary
until you know which failure produced them.

## Repository layout

```
seeweed3d/
  capture/      zed_capture.py             ZED field capture (v2: SVO + depth + confidence + logs)
  extraction/   extract_sessions.py        recordings -> indexed, QC'd frame pool (v1 & v2)
                curate_pool.py             drop redundant/bad frames (manifest-only, reversible)
                select_batches.py          pool -> CVAT-ready annotation batches (holdout-safe)
  annotation/   rank_by_contact.py         which mixed frames to annotate first (onion/weed contact)
                prelabel_onions_sam3.py    SAM 3 onion prelabels for onion-only scenes
                prelabel_weeds_sam3.py     SAM 3 weed instances + morphology + LEP proposals
                prelabel_mixed_sam3.py     MIXED scenes: precise masks, one class, no guessing
                mine_pool.py               model-in-the-loop: rank the pool, export the next batch
                cvat_roundtrip.py          CVAT export -> training masks + auto-vs-verified IoU
                regen_cvat_labels.py       refresh label schema files without re-running SAM 3
                fix_coco_categories.py     repair pre-rename COCO category names before import
  perception/   lep.py                     multi-evidence LEP estimator (hand-engineered baseline)
                segmenter.py               pluggable seg backends -> framework-free Detections
                pipeline.py                full RGB-D inference: seg -> batched LEP -> 3D -> safety
                predict_images.py          run a checkpoint on a folder of unlabelled frames
                depth3d.py                 robust LEP depth sampling + 3D point with uncertainty
                safety.py                  treatment-candidate decision (can only ABSTAIN)
                schema.py                  structured WeedTarget / FrameResult output
  training/     make_dataset.py            config-block runner: CVAT exports -> training dataset
                datasets/                  one runner per dataset (weeds, ...), so several coexist
                train_model.py             config-block runner: Stage A on Mask R-CNN (default)
                train_model_rfdetr.py      config-block runner: Stage A on RF-DETR-Seg
                datumaro_multitask.py      CVAT Datumaro 1.0 -> masks + grouped LEPs + report
                splits.py                  session-safe splits with leakage checks
                seg_dataset.py             torchvision-format instance segmentation dataset
                coco_export.py             seg_manifest -> Roboflow-style COCO tree for rfdetr
                train_seg_torchvision.py   Stage A training, BSD-3 backend
                train_seg_rfdetr.py        Stage A training, Apache-2.0 real-time backend
                losses.py                  multitask LEP losses + Tversky / focal Tversky
                tracking.py                local-only TensorBoard + MLflow, with preview panels
                active_learning.py         rank the unlabelled pool: what to annotate NEXT
                roi.py lep_targets.py lep_roinet.py lep_dataset.py train_lep.py
                                           Stage B: ROI transform, targets, model, training
                train_seg.py               Ultralytics backend (AGPL, opt-in)
  evaluation/   bench_mixed.py             the mixed-scene ruler: crop asymmetry, merges, cluster over-use
                eval_seg.py                Stage A metrics + crop safety + confidence sweep
                metrics.py                 segmentation, LEP, safety, 3D and latency metrics
                report.py                  self-contained HTML report: misses, overlays, buckets
                plots.py                   training curves, per-class AP, sweep, recall-by-size
                analyze_run.py             one command: detect backend, emit every figure
  deploy/       export.py                  ONNX/TensorRT export with numerical parity checks
                benchmark.py               latency benchmarking (records the device)
  common/       ontology.py                class names + stable COCO ids (single source of truth)
                progress.py                dependency-free progress lines with rate + ETA
                vegetation.py              shared ExG vegetation prior + white balance
                depth_utils.py             canonical depth reader + robust 3D point sampling
                torch_utils.py             device checks that fail before anything expensive
  validation/   depth_data_validation.py   sanity-check a raw session's depth stream
                diagnose_blur.py           is blur MOTION or OPTICS? (decides from the frames)
docs/           runbook, pipeline, prelabeling and dataset-growth guides
legacy/         superseded scripts, kept for provenance
tests/          synthetic end-to-end checks for every stage above
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

# 5. Prelabel with SAM 3, then verify in CVAT. Pick the one that matches the scene:
python seeweed3d/annotation/prelabel_onions_sam3.py   # onion-only
python seeweed3d/annotation/prelabel_weeds_sam3.py    # weed-only
python seeweed3d/annotation/prelabel_mixed_sam3.py    # BOTH in frame

# 6. Build a training dataset from the corrected CVAT exports
python seeweed3d/training/make_dataset.py

# 7. Train Stage A (segmentation)
python seeweed3d/training/train_model.py              # Mask R-CNN (default)
python seeweed3d/training/train_model_rfdetr.py       # RF-DETR-Seg (real-time)

# 8. Every figure and metric for a finished run, in one command
python -m seeweed3d.evaluation.analyze_run <run_dir>

# 9. Once a model exists, this replaces step 5 for each subsequent round
python seeweed3d/annotation/mine_pool.py
```

**→ [`docs/lep_localization_explained.md`](docs/lep_localization_explained.md)** explains LEP
localization in full technical detail — biology, both estimators, the math,
uncertainty, 3D conversion and the safety rules.

**→ [`docs/system_readiness.md`](docs/system_readiness.md) — is the whole system
ready?** What each stage needs from the last, what goes wrong when it doesn't
have it, and why a weed-only model cannot produce a single treatment candidate.
Every failure it describes produces output that looks completely ordinary.
Check with `python -m seeweed3d.perception.preflight`.

**→ [`docs/RUNBOOK.md`](docs/RUNBOOK.md) is the complete end-to-end guide**: every
step from raw recordings through SAM 3 prelabeling, CVAT annotation, merging
multiple CVAT tasks, training both stages, evaluation, inference and Jetson
deployment — with all commands.

Other guides: `docs/extraction_quality.md` (**extraction fidelity — the
lossless guarantees, what was measured, and what is not claimed**),
`docs/dataset_pipeline.md` (extraction and selection),
`docs/onion_prelabeling.md` and `docs/weed_prelabeling.md` (the SAM 3
workflows), **`docs/sam_prelabeling.md` (the complete record of the prelabeling
pipeline — every technique, its config key, its current value, and the
measurement behind it, including everything that was built and then turned
off)**, `docs/mixed_prelabeling.md` (scenes with both, and the mask logic),
`docs/dataset_growth.md` (active learning, and how to grow the dataset well),
**`docs/mixed_dataset_strategy.md` (how the mixed-scene dataset gets built, and
why instance identity — not boundary quality — is the bottleneck)**,
**`docs/training.md` (the complete record of Stage A training - which model,
every setting changed from its default and why, and the defects that only
appeared by running it)**,
**`docs/stage_a_improvements.md` (what actually improves Stage A, and why an
architecture swap today would measure the teacher)**,
**`docs/new_machine_setup.md` (standing up a second training machine — and why
you copy the sessions, not the built dataset)**,
**`docs/self_training.md` (scoring the model's own predictions against an
independent witness, and why confidence is the wrong ranking signal)**,
**`docs/weed_active_learning.md` (the round-by-round loop, and why you annotate
the frames the model gets WRONG)**,
`docs/experiment_tracking.md`, `docs/edge_model_research.md`, and
`docs/capture_changelog.md` (capture rationale, v1→v2).

## Status

Full design, commands and limitations for the supervised stage:
`docs/supervised_perception_baseline.md`. Chronology and reasoning for every
change: [`CHANGELOG.md`](CHANGELOG.md).

| | Status |
|---|---|
| Capture, extraction, curation, CVAT round trip | **in use on real field recordings** |
| SAM 3 prelabeling: onion-only, weed-only, mixed | **in use** |
| Dataset building from multiple CVAT exports | **in use** |
| Stage A backends: `maskrcnn` (BSD-3, **default**), `rfdetr` (Apache-2.0), `ultralytics` (AGPL, opt-in) | **implemented; the first two trained on real data** |
| Stage A **trained weights** | **exist** — Mask R-CNN and RF-DETR-Seg runs, evaluated and compared |
| Stage A accuracy / mAP / crop-safety numbers | **measured**, but see the caveat below |
| Model-in-the-loop mining + active-learning ranking | **implemented, wired end to end** |
| `LEPRoiNet` + losses, ROI transform, heatmap targets, depth→3D, safety abstention | **implemented, unit-tested**; mocked segmentation + synthetic RGB-D |
| Stage B (LEP) **trained weights** | **none** — needs human LEP annotations |
| TensorRT engines, INT8 | **requires a GPU / Jetson Orin** — engines must be built on the device |
| Jetson latency, GPU memory | **not measured** — desktop numbers do not transfer |
| Teacher–student / distillation, temporal tracking | **deliberately not started** — see `docs/dataset_growth.md` for why pseudo-labelling the crop is unsafe here |

> ⚠️ **Every Stage A number so far is same-session validation.** Adjacent frames
> from one drive are near-identical, so the val split shares ground with train
> even under a session-safe split when only a few sessions exist. Until a whole
> session is held out and annotated, the metrics say the model fits this field
> on this day — not that it generalises. This is the highest-value 40 frames
> available, and it is the first recommendation in `docs/dataset_growth.md`.

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
- **`vegetation == onion` only in onion-only scenes**, and `vegetation == weed`
  only in weed-only ones. In a mixed scene neither holds — use
  `prelabel_mixed_sam3.py`, which proposes no class at all rather than a wrong
  one. A missed onion must never become a weed label.
- **The operating confidence is part of any result.** The same weights scored
  small-weed recall 0.277 at conf 0.5 and 0.732 at conf 0.25. Always report the
  threshold, and choose it from `eval_seg --sweep`.
- **Never pseudo-label the crop.** The model is already confidently wrong about
  grass-as-onion; feeding that back improves every metric while making the
  real failure worse.

## Requirements

- Python 3.10+, `ffmpeg`/`ffprobe` on PATH, and the packages in
  `requirements.txt`.
- Capture additionally needs the **ZED SDK** + `pyzed` (installed via the SDK,
  not pip).
- Prelabeling additionally needs Meta's official `sam3` package (`pip
  install "git+https://github.com/facebookresearch/sam3.git"`, which pulls in
  `torch`) and access to the gated `facebook/sam3` / `facebook/sam3.1` weights.
- Training and deployment live in a **separate environment** on purpose, so a
  training dependency can never break SAM 3's `numpy>=1.26,<2` pin. That pin is
  load-bearing — see `docs/RUNBOOK.md` §0.

```bash
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/ -v
```
The tests build synthetic v1/v2 sessions with real FFV1 MKVs and run each stage
end-to-end: lossless depth round-trip, index alignment, dropped-frame recovery,
holdout isolation, reproducible selection, COCO validity, dataset splits, the
segmentation training step, the metrics and the prelabel mask logic. SAM 3 and
CUDA are stubbed — both need a GPU, and SAM 3 needs gated weights.

Regression tests here are named after the failure they pin, not the function
they call. If one breaks, its docstring says what went wrong the first time.
