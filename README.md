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
                select_batches.py          pool -> CVAT-ready annotation batches (holdout-safe)
  annotation/   prelabel_onions_sam3.py    SAM 3 onion prelabels for onion-only scenes
  validation/   depth_data_validation.py   sanity-check a raw session's depth stream
  common/       depth_utils.py             canonical depth reader + robust 3D point sampling
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

# 3. Select CVAT-ready annotation batches (whole-session holdout enforced)
python seeweed3d/extraction/select_batches.py

# 4. (onion-only scenes) Prelabel onions with SAM 3, then verify in CVAT
python seeweed3d/annotation/prelabel_onions_sam3.py
```

See `docs/dataset_pipeline.md` for the full extraction/selection guide and
`docs/onion_prelabeling.md` for the SAM 3 workflow. Capture rationale and the
v1→v2 changes are in `docs/capture_changelog.md`.

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
- Onion prelabeling additionally needs `transformers` + `torch` and access to
  the gated `facebook/sam3` model.

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
