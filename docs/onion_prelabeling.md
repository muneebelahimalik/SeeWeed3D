# Onion prelabeling with SAM 3 (`annotation/prelabel_onions_sam3.py`)

Auto-generates a high-recall onion **safety mask** for every pooled frame and
writes it as CVAT-importable COCO, so annotators **verify** masks instead of
drawing them. For **onion-only** recordings only.

## Why this is safe here (and nowhere else)

In a mixed scene you must never assume `vegetation == onion` — one missed onion
becomes a dangerous weed label. In an onion-only field there are no weeds to
confuse, so the green vegetation prior is a legitimate, high-recall onion
signal. **Do not point this script at mixed or weed scenes.**

## The technique

1. **SAM 3** concept segmentation (`onion` text, or image-exemplar boxes) gives
   clean boundaries on thin, crossing leaves and returns every instance at once.
2. An **Excess-Green (ExG)** vegetation prior validates SAM masks (drops
   anything not on green tissue) and recovers onion tissue SAM missed. If SAM
   returns nothing usable, the vegetation mask is the fallback — a frame is
   never silently dropped.
3. The fused per-frame result is a **semantic** onion mask (coverage over
   instance separation — exactly what the laser must avoid), exported as
   polygons under the `onion plant` label.

Everything except the SAM 3 call is plain OpenCV/NumPy; the SAM 3 call is
isolated in `load_sam3()` / `sam3_masks()`, so the rest is unit-testable and the
model backend can be swapped without touching the pipeline.

## Prerequisites

```bash
pip install pillow opencv-python numpy
pip install "git+https://github.com/facebookresearch/sam3.git"   # pulls in torch
```
SAM 3 is loaded through Meta's **official `sam3` package**
(`build_sam3_image_model` + `Sam3Processor`), which loads a `.pt` checkpoint and
supports **both SAM 3 and the faster SAM 3.1**. This deliberately avoids
depending on a `transformers` release that bundles SAM 3.

`facebook/sam3` and `facebook/sam3.1` are **gated** — request access on their
Hugging Face pages, then either:
- log in once with `huggingface-cli login` and leave `SAM_CHECKPOINT = None`
  (the chosen `SAM_VERSION` auto-downloads and caches), **or**
- download the checkpoint yourself and point `SAM_CHECKPOINT` at the `.pt` file
  (`sam3.pt` for SAM 3, `sam3.1_multiplex.pt` for SAM 3.1).

### Which checkpoint / where to put it
Checkpoints are multi-GB — keep them **outside the git repo** (gitignored
anyway). Only the `.pt` is needed for this route (the `model.safetensors` +
tokenizer files are for the separate transformers route and are not used here).

| `SAM_VERSION` | Checkpoint file | Notes |
|---|---|---|
| `"sam3"` | `sam3.pt` | base model |
| `"sam3.1"` | `sam3.1_multiplex.pt` | faster variant (default here) |

```python
SAM_CHECKPOINT = r"C:\Users\mm17889\models\sam3\sam3.1_multiplex.pt"
```

A CUDA GPU is expected (`DEVICE="cuda"`); `"cpu"` works but is slow.

## Run

Set the values at the top of the script:
```python
DATASET_ROOT   = r"E:\Dataset_Vidalia"   # = OUTPUT_ROOT from extract_sessions.py
SAM_VERSION    = "sam3.1"                # "sam3" | "sam3.1"
SAM_CHECKPOINT = None                    # None auto-downloads; or a local .pt path
```
Trial on a few frames first (`CONFIG["LIMIT_PER_SESSION"] = 20`), eyeball the
overlays, then set it back to `None` for the full pool:
```bash
python seeweed3d/annotation/prelabel_onions_sam3.py
```

Output under `DATASET_ROOT/auto_labels_onion/<session_id>/`:

| Item | Purpose |
|---|---|
| `instances_default.json` | COCO, import into CVAT (category `onion plant`) |
| `masks/<frame>.png` | binary onion mask (255 = onion) |
| `preview/<frame>.jpg` | overlay for fast QC / FiftyOne review |

## Into CVAT

1. Create a task from `sessions/<session_id>/rgb/`.
2. Paste `cvat_labels.json` (from `select_batches.py`) into the **Raw** label editor.
3. Import that session's `instances_default.json` as **COCO 1.0**.
4. **Verify**: fix leaf edges, delete the rare stray weed. Prioritize coverage
   over splitting overlapping leaves — this is a safety mask.
5. Export corrected masks as your training labels.

## Tuning (all in `CONFIG`)

| Key | Effect |
|---|---|
| `WHITE_BALANCE` / `WB_CAST_RATIO` | gray-world correction to recover green colour-cast frames (neutral frames untouched) |
| `SAM_PROMPT_MODE` | `auto_exemplar` (default: onion boxes from veg blobs), `text`, or `manual` |
| `EXEMPLAR_MIN_AREA_PX` / `EXEMPLAR_MAX_BOXES` / `EXEMPLAR_PAD_PX` | auto-exemplar box derivation |
| `SAM_TEXT_PROMPTS` / `EXEMPLARS` | text concepts (unioned) vs. per-session hand-drawn exemplar boxes |
| `SAM_VERSION` / `SAM_CHECKPOINT` | choose SAM 3 vs 3.1, and where the `.pt` lives |
| `SAM_CONF` | SAM 3 confidence threshold (detections below are dropped) |
| `EXG_THRESHOLD` | lower = more permissive vegetation prior |
| `SAM_VEG_OVERLAP_MIN` | how much of a SAM mask must sit on vegetation to be kept |
| `RECOVER_VEG_MIN_PX` | min size of a veg region SAM missed before it is added back |
| `POLY_MIN_AREA_PX` / `POLY_APPROX_EPS` | drop tiny polygons / simplify boundaries |

## Scaling beyond this (teacher–student)

Once a few hundred verified frames exist, train a small single-class onion
segmenter, pseudo-label the rest, keep only high-confidence + temporally-stable
masks, verify a fraction, retrain (project plan §14). That compounds the
training set without hand-labeling everything.
