# Stage 3 — Onion prelabeling with SAM 3 (`03_prelabel_onions_sam3.py`)

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
isolated in `load_sam3()` / `sam3_masks()`.

## Prerequisites

```bash
pip install ultralytics opencv-python numpy
```
`sam3.pt` (3.45 GB) is gated: request access on the SAM 3 Hugging Face page,
download it, and point `SAM3_MODEL` at it. A CUDA GPU is expected
(`DEVICE="cuda:0"`); `"cpu"` works but is slow.

## Run

Set the two values at the top:
```python
DATASET_ROOT = r"E:\Dataset_Vidalia"   # = OUTPUT_ROOT from stage 1
SAM3_MODEL   = r"C:\models\sam3.pt"
```
Trial on a few frames first (`CONFIG["LIMIT_PER_SESSION"] = 20`), eyeball the
overlays, then set it back to `None` for the full pool:
```bash
python data_extraction/03_prelabel_onions_sam3.py
```

Output under `DATASET_ROOT/auto_labels_onion/<session_id>/`:

| Item | Purpose |
|---|---|
| `instances_default.json` | COCO, import into CVAT (category `onion plant`) |
| `masks/<frame>.png` | binary onion mask (255 = onion) |
| `preview/<frame>.jpg` | overlay for fast QC / FiftyOne review |

## Into CVAT

1. Create a task from `sessions/<session_id>/rgb/`.
2. Paste `cvat_labels.json` (from stage 2) into the **Raw** label editor.
3. Import that session's `instances_default.json` as **COCO 1.0**.
4. **Verify**: fix leaf edges, delete the rare stray weed. Prioritize coverage
   over splitting overlapping leaves — this is a safety mask.
5. Export corrected masks as your training labels.

## Tuning (all in `CONFIG`)

| Key | Effect |
|---|---|
| `SAM_TEXT_PROMPTS` / `EXEMPLARS` | text concepts vs. per-session exemplar boxes (more reliable if text under-segments) |
| `EXG_THRESHOLD` | lower = more permissive vegetation prior |
| `SAM_VEG_OVERLAP_MIN` | how much of a SAM mask must sit on vegetation to be kept |
| `RECOVER_VEG_MIN_PX` | min size of a veg region SAM missed before it is added back |
| `POLY_MIN_AREA_PX` / `POLY_APPROX_EPS` | drop tiny polygons / simplify boundaries |

## Scaling beyond this (teacher–student)

Once a few hundred verified frames exist, train a small single-class onion
segmenter, pseudo-label the rest, keep only high-confidence + temporally-stable
masks, verify a fraction, retrain (project plan §14). That compounds the
training set without hand-labeling everything.
