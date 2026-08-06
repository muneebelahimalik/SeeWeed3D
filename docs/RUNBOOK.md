# SeeWeed3D — complete runbook

Every step from raw ZED recordings to running inference, with the exact
commands. Windows/PowerShell throughout; on Linux swap `` ` `` line-continuations
for `\` and `D:/...` for your paths.

**Use `python -m pip`, never bare `pip`** — that guarantees packages land in the
interpreter you are actually running (your `dl` conda env), not some other one
on PATH.

---

## Contents

| Stage | What it does | Needs GPU? |
|---|---|---|
| [0. Install](#0-install) | environments | no |
| [1. Extract](#1-extract-recordings) | recordings → indexed frame pool | no |
| [2. Curate](#2-curate-the-pool-optional) | drop redundant/bad frames | no |
| [3. Prelabel with SAM 3](#3-prelabel-with-sam-3) | auto-annotations to correct | **yes** |
| [4. CVAT](#4-annotate-and-correct-in-cvat) | human verification | no |
| [5. Merge + prepare](#5-merge-exports-and-build-the-training-dataset) | many CVAT tasks → one dataset | no |
| [6. Train Stage A](#6-train-stage-a-segmentation) | crop-vs-weed segmentation | **yes** |
| [7. Train Stage B](#7-train-stage-b-lep) | LEP localization | **yes** |
| [8. Evaluate](#8-evaluate) | metrics by session | no |
| [9. Inference](#9-run-inference) | full RGB-D pipeline | yes (practically) |
| [10. Deploy](#10-export-and-benchmark-jetson) | ONNX/TensorRT + benchmark | on Jetson |

**Related:** [experiment tracking](experiment_tracking.md) ·
[edge model research](edge_model_research.md) ·
[LEP localization explained](lep_localization_explained.md)

---

## 0. Install

Two environments, deliberately separate.

```powershell
# --- Environment A: data pipeline + SAM 3 prelabeling -----------------------
conda activate dl
python -m pip install -r requirements.txt

# SAM 3 (only needed for stage 3). Pulls torch.
python -m pip install "git+https://github.com/facebookresearch/sam3.git"
# Windows + torch 2.5:
python -m pip install "triton-windows>=3.1,<3.2"
```

> **Do not upgrade numpy past 2.0 in this environment.** `requirements.txt`
> pins `numpy>=1.26,<2` because SAM 3 requires it. Breaking that pin breaks
> prelabeling.

```powershell
# --- Environment B: training + deployment ----------------------------------
# Separate env so a training dependency can never break SAM 3's numpy pin.
conda create -n sw-train python=3.11 -y
conda activate sw-train
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt

# CUDA torch matching your driver (check with: nvidia-smi)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pytest tests/ -q
```

**Licence note:** the default segmentation backend is torchvision Mask R-CNN
(BSD-3-Clause) — nothing to install beyond the above, and no obligation on a
commercial product. Ultralytics is **not** installed by default because it is
AGPL-3.0. See [§6](#6-train-stage-a-segmentation).

---

## 1. Extract recordings

Turns raw ZED recordings into an indexed, QC'd frame pool.

Edit the CONFIG block at the top of `seeweed3d/extraction/extract_sessions.py`:

```python
INPUT_ROOTS = [r"E:\Field_Recordings\2026-01-08"]   # one entry per visit
OUTPUT_ROOT = r"E:\Dataset_Vidalia"
DRY_RUN     = True                                   # run this first
```

```powershell
conda activate dl
python seeweed3d/extraction/extract_sessions.py     # DRY_RUN=True: inspect
# then set DRY_RUN = False and run again
python seeweed3d/extraction/extract_sessions.py
```

**Produces** `E:\Dataset_Vidalia\sessions\<session_id>\` containing `rgb/`,
`depth/`, `conf/`, `meta/pool.csv`, `meta/frames_index.csv`,
`meta/calibration.json`, plus a top-level `registry.csv`.

Read `registry.csv` before continuing — check `median_depth_valid_frac_veg` and
pick which sessions you will hold out for testing.

> **Depth is raw uint16 millimetres, `0` = invalid.** Never rescale it. Always
> read it via `common/depth_utils.load_depth_mm()`.

---

## 2. Curate the pool (optional)

Drops near-duplicate frames (from driving slowly) and bad frames. **Edits
`pool.csv` only — no image is ever deleted or renamed**, and `RESTORE_ALL`
undoes everything.

Edit the top of `seeweed3d/extraction/curate_pool.py`:

```python
DATASET_ROOT = r"E:\Dataset_Vidalia"
DRY_RUN      = True
CONFIG["MIN_SHIFT_FRAC"] = 0.6      # see the sweep table it prints
```

```powershell
python seeweed3d/extraction/curate_pool.py          # dry run: read the sweep
# choose MIN_SHIFT_FRAC from the "overlap between kept frames" column,
# then set DRY_RUN = False and run again
python seeweed3d/extraction/curate_pool.py
```

Pick the threshold from the **overlap** column, not the drop percentage: at
`0.15`, 85% of every kept frame is ground you already annotated in the previous
one. Since you hand-correct everything in CVAT, annotation cost dominates —
`0.4–0.6` is usually the right range.

To undo: set `RESTORE_ALL = True` and re-run.

---

## 3. Prelabel with SAM 3

Auto-generates annotations so CVAT is *correction*, not annotation from scratch.

**Get the weights** (gated — request access on HuggingFace first):

```powershell
huggingface-cli login
huggingface-cli download facebook/sam3 --local-dir E:\Models\sam3
```

Edit the top of the prelabeler you need:

```python
DATASET_ROOT   = r"E:\Dataset_Vidalia"
SAM_VERSION    = "sam3"
SAM_CHECKPOINT = r"E:\Models\sam3\sam3.pt"
CONFIG["ONLY_SESSIONS"]      = ["vid3_20260108_103135"]   # IMPORTANT
CONFIG["LIMIT_PER_SESSION"]  = 20                          # trial first, then None
```

```powershell
conda activate dl

# Weed-only scenes
python seeweed3d/annotation/prelabel_weeds_sam3.py

# Onion-only scenes
python seeweed3d/annotation/prelabel_onions_sam3.py
```

**Check `auto_labels_weeds/<session>/preview/` before doing a full run.** Look
for coloured dots on bare ground — if you see them, the vegetation prior is
false-positiving on your substrate and you should tighten
`RECOVER_MIN_VEG_SCORE` / leave `RECOVER_MISSED_PLANTS` off.

**Produces** per session:

| Item | Use |
|---|---|
| `cvat_ready/` | **upload this folder to CVAT** |
| `instances_default.json` | COCO import for CVAT |
| `weed_cvat_labels.json` | paste into CVAT's Raw label editor |
| `preview/` | sanity-check before committing to a full run |
| `flagged_rgb/` | colour-cast/glare frames — annotate separately or skip |

> `vegetation == weed` holds only in **weed-only** recordings, and
> `vegetation == onion` only in **onion-only** ones. Never point a prelabeler at
> a mixed scene.

If you change the ontology later, you do **not** need to re-run SAM 3:

```powershell
python seeweed3d/annotation/regen_cvat_labels.py    # rewrites label JSONs in seconds
```

---

## 4. Annotate and correct in CVAT

**One CVAT task per session.** Tasks stay manageable, and a session is the unit
that train/val/test splits must respect anyway. They get merged in [§5](#5-merge-exports-and-build-the-training-dataset).

### 4.1 Create the task

1. **Tasks → + → Create new task**
2. Name it **exactly the session id** (e.g. `vid3_20260108_103135`) — this is
   how sessions are traced later.
3. **Select files** → upload `auto_labels_weeds/<session>/cvat_ready/`
4. Open the **Raw** tab in the label editor and paste the entire contents of
   `weed_cvat_labels.json`.
5. Submit.

### 4.2 Import the prelabels

**Task menu → Upload annotations → COCO 1.0** → `instances_default.json`.

*(COCO is fine for this direction — it only has to carry masks in. The
export direction must be Datumaro, because COCO cannot represent the mask↔LEP
grouping.)*

### 4.3 Correct the annotations

For every plant:

1. **Fix the class.** SAM proposes only `grass_weed`, `weed_cluster` or
   `other_weed`. `cutleaf_evening_primrose` vs `wild_radish` is an *appearance*
   question that shape cannot answer — those are always yours to assign.
2. **Fix the mask boundary** where it is wrong.
3. **Place the `weed_LEP` point** at the centre of the youngest emerging tissue
   (the crown / apical meristem). *Optional on a first pass — see
   "LEPs are optional" below.*
4. **⚠️ GROUP the LEP with its weed mask.** Select both shapes → press **`G`**.

> **Grouping is the single most important step.** The mask↔LEP link is carried
> *only* by the group id. An ungrouped LEP is **rejected** by the importer — it
> is never auto-assigned to the nearest weed, because in a dense frame the
> nearest crown is frequently the neighbouring plant, and a wrong owner aims a
> 60 W laser at the wrong tissue.

5. Set the attributes: `lep_visibility`, `targetable`, `growth_stage`.

**Rules the importer enforces** (violations are listed for you to fix):

| Rule | Why |
|---|---|
| A `visible` + `targetable=yes` weed has exactly one grouped LEP | otherwise there is nothing to train on |
| A `not_visible` weed needs **no** LEP | absence is legitimate; never invent one |
| `weed_cluster` must **not** have an LEP | a cluster has no single growth point |
| `onion_plant` never has an LEP | the crop is never a target |
| The LEP lies inside its own mask (±4 px) | otherwise it is grouped with the wrong plant |
| Each plant has its own group id | duplicate group ids make ownership ambiguous |

Use `ignore_region` for areas that should be excluded from training entirely
(severe blur, glare, the rig in frame).

### LEPs are optional — you can annotate masks first

**Stage A (crop-vs-weed segmentation) does not use LEPs at all.** It trains from
masks and classes only. Stage B (`LEPRoiNet`) is the only thing that needs them,
and until it exists the pipeline uses the hand-engineered estimator in
`perception/lep.py`, which needs no training data whatsoever.

So a mask-only annotation round is a legitimate, complete deliverable.
`prepare_dataset.py` detects it automatically:

| LEP points in the export | What happens |
|---|---|
| **zero**, with weeds present | `dataset_kind: segmentation_only`. The LEP requirement is lifted, no `missing_lep` errors, no `lep_manifest.json`. Stage A trains normally. |
| **some but not all** | A real contract gap — every weed still missing one is reported in `annotations_needing_correction.json`. Half-labelled is worse than not labelled, because Stage B would learn from a biased subset. |
| **all of them** | `dataset_kind: multitask`. Both stages train. |

Pass `--no-require-lep` to force segmentation-only even when a few LEPs exist
(they are kept in the records, just not required).

> **Note on prelabels:** the SAM 3 prelabeler estimates an LEP per weed, but the
> CVAT import in §4.2 is **COCO**, and COCO has no point annotation type — so
> those proposals do not reach CVAT and every LEP placed there is placed by hand.
> That is why deferring them is reasonable: do the masks now, and add LEPs in a
> later pass on the frames you actually intend to train Stage B on.

### 4.4 Export

**Task menu → Export annotations → `Datumaro 1.0`** → download → **unzip**.

> **Must be Datumaro 1.0, not COCO.** COCO has no representation for shape
> groups, so exporting as COCO silently destroys every mask↔LEP association you
> just created.

Put every unzipped export under one parent folder:

```
E:\CVAT_exports\
  vid3_20260108_103135\annotations\default.json
  vid3_20260108_110444\annotations\default.json
  onion1_20260112_090000\annotations\default.json
```

---

## 5. Merge exports and build the training dataset

This is where the separate CVAT tasks become one dataset.

```powershell
conda activate sw-train

# Point at the PARENT folder - every export under it is found and merged
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root  E:/CVAT_exports `
    --images-root    E:/Dataset_Vidalia/sessions `
    --out            E:/Dataset_Vidalia/training/mixed_v1 `
    --holdout-test   vid3_20260108_103135 `
    --val-fraction 0.2 --test-fraction 0.2 --seed 1234
```

Or list exports explicitly:

```powershell
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root  E:/CVAT_exports/vid3_20260108_103135 `
                     E:/CVAT_exports/vid3_20260108_110444 `
    --images-root    E:/Dataset_Vidalia/sessions `
    --out            E:/Dataset_Vidalia/training/mixed_v1
```

### 5.1 ⚠️ If the CVAT task was pre-loaded with SAM prelabels

Then **every frame in the export has annotations**, including the ones you never
reached. Those frames are *not* empty — they carry SAM's masks with whatever
class it guessed. Nothing else in the pipeline filters them, and the
un-annotated-frame exclusion will not catch them.

Training on them is **worse than having no data**: the mask geometry is
basically correct and only the label is wrong, so the loss is confident and
consistent, and the model reliably learns the wrong class.

Select the frames you actually verified. **List first:**

```powershell
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root E:/CVAT_exports --list-frames
```

That prints a numbered table — position, item id, instance count, class
breakdown. Positions are **1-based** over item-id order. Check the numbering
against what you remember from CVAT *before* selecting, rather than discovering
a one-off in a trained model.

Then build with the selection:

```powershell
    --include-frames "1-26,28-36,50-59"
```

| Syntax | Meaning |
|---|---|
| `1-26` | inclusive position range |
| `12` | a single position |
| `frame_0007` | a literal item id |
| `*_001*` | fnmatch glob over item ids |
| `@keep.txt` | read the list from a file, one token per line, `#` comments |
| `--exclude-frames` | same syntax, applied **after** `--include-frames` |

Selection happens **before** validation, so the frames you discarded cannot fill
`annotations_needing_correction.json` with errors about annotations you are
throwing away. A position beyond the end of the export is an **error**, not a
silent no-op — quietly ignoring it would build the wrong dataset.

**Merging is safe across tasks:** each export is resolved through its *own*
`categories` block, so two CVAT tasks that happen to order their labels
differently still merge correctly — only the label **name** is carried forward.
The same frame appearing in two exports is reported as an error, because it
would be trained on twice and could land in two splits.

**Produces:**

| File | Purpose |
|---|---|
| `seg_manifest.json` | Stage A training (permissive backend) |
| `lep_manifest.json` | Stage B training — **omitted for a segmentation-only dataset** |
| `data.yaml` + `labels/` | Stage A training (Ultralytics backend only) |
| `splits/{train,val,test}_sessions.txt` | which session went where |
| `splits/splits_summary.json` | per-split session/class counts |
| `dataset_report.json` | full integrity report |
| **`annotations_needing_correction.json`** | **read this** |

**If it exits with errors**, open `annotations_needing_correction.json`, fix
those shapes in CVAT, re-export, and re-run. To inspect without blocking, add
`--allow-errors` — but do **not** train on the result.

Images are never copied. Manifests reference your existing files.

---

## 6. Train Stage A (segmentation)

**Default backend: torchvision Mask R-CNN (BSD-3-Clause).** No extra install, no
licence obligation.

```powershell
conda activate sw-train
python -m seeweed3d.training.train_seg_torchvision `
    --dataset     E:/Dataset_Vidalia/training/mixed_v1 `
    --images-root E:/Dataset_Vidalia/sessions `
    --out         E:/Dataset_Vidalia/runs/seg_v1 `
    --epochs 20 --batch 2 --device cuda `
    --track all --preview-every 5
```

Produces `runs/seg_v1/best.pt`, `last.pt`, `history.json`, `params.json`.

### Watching the run

`--track` enables **local** experiment tracking — nothing is uploaded and no
account is needed. `auto` (the default) uses whatever is installed and never
fails; `all` requires both and errors if either is missing.

```powershell
python -m pip install tensorboard mlflow

tensorboard --logdir E:/Dataset_Vidalia/runs/seg_v1/tb
mlflow ui --backend-store-uri E:/Dataset_Vidalia/runs/mlruns
```

| Flag | Effect |
|---|---|
| `--preview-every N` | **GT-vs-prediction overlay panels** every N epochs on a fixed sample of val frames |
| `--eval-every N` | val `mAP@50`, `mAP@50:95` and `missed_onion_fraction` every N epochs (slow — a full pass over val) |

The preview panels are the highest-value signal on a small dataset. A loss curve
cannot tell you that every mask is one plant too large, or that the model calls
every onion a weed; ten seconds looking at the overlays will. Full rationale and
the tools that were rejected: [experiment tracking](experiment_tracking.md).

### Choosing a different backend

| Backend | Licence | Real-time? | When |
|---|---|---|---|
| **`maskrcnn`** (default) | **BSD-3** | no | **prototyping — start here** |
| `rfdetr` | **Apache-2.0** | **yes** | the real-time upgrade, ships commercially |
| `ultralytics` | **AGPL-3.0** | yes | research only — see below |

`rtmdet` is **not** implemented. RTMDet-Ins is a credible Apache-2.0 alternative
on paper, but mmcv's CUDA build is a recurring problem on Jetson, so it was not
wired up. Don't pass `backend="rtmdet"` expecting it to work.

```powershell
# Real-time upgrade once the prototype works (Apache-2.0, no obligation)
python -m pip install rfdetr
```

RF-DETR-Seg's full Nano..2XL family shipped January 2026 — DINOv2 ViT backbone,
MaskDINO-style mask head, no NMS, ONNX/TensorRT export. Sizing, the licence
caveat on the XL variants, the resolution problem for small weeds, and why
**INT8 is the wrong move on Orin for a transformer**:
[edge model research](edge_model_research.md).

> **Ultralytics is AGPL-3.0.** Commercial or proprietary use requires an
> Ultralytics Enterprise Licence — they state this applies even to internal
> company R&D unless the whole project is released under AGPL. For a commercial
> laser weeder that is a real cost, and unlike a code defect it cannot be fixed
> after distribution. It is not installed by default; `build_segmenter()` prints
> a loud warning if you select it.

---

## 7. Train Stage B (LEP)

> **Skip this section if your dataset is `segmentation_only`.** With no LEP
> annotations there is nothing to fit, and `perception/lep.py` supplies growth
> points at inference time without any training. Come back here after an
> annotation round that places LEPs.

```powershell
python -m seeweed3d.training.train_lep `
    --manifest    E:/Dataset_Vidalia/training/mixed_v1/lep_manifest.json `
    --images-root E:/Dataset_Vidalia/sessions `
    --out         E:/Dataset_Vidalia/runs/lep_v1 `
    --input-mode rgb_mask_geom --device cuda --epochs 50
```

`--input-mode` selects the ablation:

| Mode | Inputs | Use |
|---|---|---|
| `rgb` | RGB only | baseline; works with no depth at all |
| `rgb_mask` | + owning instance mask | no depth stream available |
| `rgb_mask_geom` | + normalised height + depth validity | **default**, full model |

Run all three to report the ablation. Depth dropout (p=0.25) is applied during
training, so the full model still works when depth fails at runtime.

---

## 8. Evaluate

### 8.1 Stage A

Training prints loss, which is not a model-quality number — it is not comparable
across runs with different class counts, and it says nothing about whether small
weeds are found or onions are protected. Run this instead:

```powershell
python -m seeweed3d.evaluation.eval_seg `
    --checkpoint  E:/Dataset_Vidalia/runs/seg_v1/best.pt `
    --dataset     E:/Dataset_Vidalia/training/mixed_v1 `
    --images-root E:/Dataset_Vidalia/sessions `
    --split val --device cuda --conf 0.5
```

Three tables, deliberately not combined into one score:

| Table | What it answers |
|---|---|
| **detection** | `mAP@50`, `mAP@50:95` per class — score-ranked, 101-point interpolated, so comparable with published numbers |
| **operating point** | precision/recall/IoU at the confidence you would actually deploy at. mAP is threshold-free; a robot is not |
| **crop safety** | **missed onion pixels** — crop pixels the system believes are safe to fire at |

Crop safety is reported separately on purpose. A model can post excellent mAP and
still miss the one onion that matters, and averaging that into a headline number
hides it. A class with no ground truth in the split reports AP as `-` (undefined)
and is excluded from the mean, rather than counted as zero.

Writes `metrics_<split>.json` next to the checkpoint.

### 8.2 Everything else

Metrics live in `seeweed3d/evaluation/metrics.py` and are computed from stored
results, so you can re-evaluate without re-running inference.

```python
from seeweed3d.evaluation import metrics as m

m.segmentation_metrics(pred_masks, pred_classes, gt_masks, gt_classes)
m.onion_safety_metrics(pred_onion_union, gt_onion_union)   # crop safety
m.lep_errors(pred_uv, gt_uv, plant_radius_px)
m.compare_lep_methods(gt_uv, {"bbox_center": ..., "centroid": ...,
                              "dt_peak": ..., "lep_py": ..., "learned": ...})
m.uncertainty_calibration(sigmas, errors)
m.safety_metrics(targets, gt_lookup)
```

**Evaluate by whole session.** Watch **onion recall** and **missed onion pixels**
separately from IoU — missing crop tissue can destroy the crop, while a false
onion merely skips a weed, and IoU averages those two very different errors
together.

The learned LEP must be compared against all four baselines (bbox centre,
centroid, DT peak, and the hand-engineered `perception/lep.py`). That estimator
is preserved and is also the runtime fallback.

---

## 9. Run inference

```python
import cv2, torch
from seeweed3d.perception.segmenter import MaskRCNNSegmenter
from seeweed3d.perception.pipeline import InferencePipeline
from seeweed3d.training.config import PipelineConfig
from seeweed3d.training.lep_roinet import build_model
from seeweed3d.common.depth_utils import load_depth_mm, load_intrinsics

cfg = PipelineConfig()
seg = MaskRCNNSegmenter(r"E:/Dataset_Vidalia/runs/seg_v1/best.pt",
                        conf=0.25, device="cuda")

blob = torch.load(r"E:/Dataset_Vidalia/runs/lep_v1/best.pt", weights_only=False)
lep = build_model(cfg.model); lep.load_state_dict(blob["model"]); lep.to("cuda")

pipe = InferencePipeline(seg, cfg, lep_model=lep, torch_device="cuda")

session = r"E:/Dataset_Vidalia/sessions/vid3_20260108_103135"
bgr = cv2.imread(f"{session}/rgb/vid3_20260108_103135_000100.png")
depth, valid = load_depth_mm(f"{session}/depth/vid3_20260108_103135_000100.png")
K = load_intrinsics(session)

result = pipe.run(bgr, depth, valid, K,
                  session_id="vid3_20260108_103135", frame_id="000100")

for t in result.candidates:                    # SAFE to consider for treatment
    print(t.class_name, t.lep_uv, t.xyz_mm, "sigma", t.xyz_sigma_mm)

for t in result.abstentions:                   # refused, with reasons
    print("SKIP", t.class_name, t.rejection_reasons)

print(result.timings_ms, result.reason_counts())
```

**The pipeline produces candidates only.** `perception/safety.py` has no
actuator surface and never commands anything. Turning a candidate into a laser
command is a separate, deliberate act by your control layer — and it must
consult `t.safety_status == "candidate"`, never just the presence of `lep_uv`.

Without a trained LEP model, `lep_model=None` falls back to the hand-engineered
estimator, so the pipeline runs end-to-end before Stage B exists.

---

## 10. Export and benchmark (Jetson)

```powershell
# ONNX + PyTorch-vs-ONNX numerical parity check
python -m seeweed3d.deploy.export `
    --checkpoint E:/Dataset_Vidalia/runs/lep_v1/best.pt `
    --out        E:/Dataset_Vidalia/runs/lep_v1/export `
    --precision fp16
```

```bash
# ON THE JETSON ORIN - engines are architecture- and version-specific,
# so one built on your desktop GPU will not work here.
python -m seeweed3d.deploy.benchmark \
    --checkpoint runs/lep_v1/best.pt --device cuda --out jetson_bench.json
```

> A successful export is **not** a correct export. ONNX export succeeding only
> means the graph was traceable. Check `export_report.json` for
> `torch_vs_onnx.passed` before trusting the artefact, and verify FP16 accuracy
> before attempting INT8 — calibrating on an already-wrong engine bakes the
> error in.

Desktop numbers do not transfer to a Jetson. `benchmark.py` records the device
in every result and warns if it is not running on one.

---

## Quick reference

```powershell
# Environment A (dl): data pipeline + SAM 3
python seeweed3d/extraction/extract_sessions.py
python seeweed3d/extraction/curate_pool.py
python seeweed3d/annotation/prelabel_weeds_sam3.py
python seeweed3d/annotation/prelabel_onions_sam3.py
python seeweed3d/annotation/regen_cvat_labels.py

# --- CVAT: one task per session, group LEP with mask (G), export Datumaro 1.0 ---

# Environment B (sw-train): training + deployment
python -m seeweed3d.training.prepare_dataset --datumaro-root <exports> --images-root <sessions> --out <dir>
python -m seeweed3d.training.train_seg_torchvision --dataset <dir> --images-root <sessions> --out <run>
python -m seeweed3d.evaluation.eval_seg --checkpoint <run>/best.pt --dataset <dir> --images-root <sessions> --split val
python -m seeweed3d.training.train_lep --manifest <dir>/lep_manifest.json --images-root <sessions> --out <run>
python -m seeweed3d.deploy.export --checkpoint <run>/best.pt --out <run>/export --precision fp16
python -m seeweed3d.deploy.benchmark --checkpoint <run>/best.pt --device cuda
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no Datumaro JSON found` | Exported as COCO instead of Datumaro 1.0, or didn't unzip. |
| `ungrouped_lep` errors | LEP points not grouped with their masks in CVAT — select both, press **G**. |
| `missing_lep` errors | Some weeds have LEPs and some don't. Finish the pass, or pass `--no-require-lep` to build a segmentation-only dataset. (A dataset with *zero* LEPs never raises this.) |
| `unknown label 'X'` | CVAT schema drifted from `common/ontology.py`. Run `regen_cvat_labels.py` and re-paste. |
| `duplicate_frame_across_exports` | The same frame is in two CVAT tasks. Delete one. |
| `compressed RLE ... needs pycocotools` | `python -m pip install pycocotools`, or re-export with polygons. |
| `unresolvable_session` | Task not named after the session id. Rename the CVAT task and re-export. |
| `training split has no rows` | Too few sessions for the requested fractions. Lower them or annotate more sessions. |
| SAM 3 `numpy` errors | numpy was upgraded past 2.0 in env A. `python -m pip install "numpy<2"`. |
| Phantom detections on bare ground | Vegetation prior false-positiving. Keep `RECOVER_MISSED_PLANTS=False`. |
