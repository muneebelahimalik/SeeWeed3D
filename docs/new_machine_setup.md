# Setting up a training machine

Getting a second PC to the point where it can train Stage A overnight, and the
one thing that will silently break if you skip it.

**Related:** [runbook](RUNBOOK.md) · [dataset assembly](dataset_assembly.md) ·
[Stage A improvements](stage_a_improvements.md) ·
[experiment tracking](experiment_tracking.md)

---

## The one that will bite you

**`seg_manifest.json` records ABSOLUTE image paths.** The dataset build does not
copy images — it writes down where they are:

```json
"images_root": ["E:/Dataset_Vidalia/onions_20260108_1/sessions", ...]
```

Copy that folder to `D:\data\...` on another machine and every path in it is
wrong. The failure is not subtle, but it comes hours into a run: COCO export
opens the first image and raises `FileNotFoundError`.

**Do not copy the built dataset. Copy the sessions and rebuild.** The build
takes seconds, writes manifests only, and produces paths that are correct for
the machine it ran on.

---

## 1. What to copy

The **session folders**, not the dataset build:

```
E:\Dataset_Vidalia\onions_20260108_1\sessions\        (3 sessions)
E:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions\  (3 sessions)
```

Each session holds `annotations/`, `rgb/`, `depth/` and `meta/`. The annotations
travel with the images, which is why campaign-level `SOURCES` works.

**If the flash drive is tight, `depth/` can stay behind.** Stage A trains on RGB
only — the detector never opens a depth frame. You need `depth/` for the LEP 3D
stage and for any of the depth work, so copy it when you can, but it is not
required to start training tonight.

| Folder | Needed to train Stage A |
|---|---|
| `rgb/` | **yes** |
| `annotations/` | **yes** |
| `meta/` | yes — small, and carries `session.json` for stratification |
| `depth/` | no (Stage B and depth work only) |

**Easiest option of all: use the same drive letter and path.** If the new
machine can host `E:\Dataset_Vidalia\...`, nothing needs changing and you can
even copy the built dataset directly.

---

## 2. Install

### System tools

- **git**
- **ffmpeg / ffprobe on `PATH`** — only needed for *extraction*, not training.
  Install it anyway; the check that fails without it fails late.
- **NVIDIA driver** — check with `nvidia-smi`. The CUDA version it prints is the
  *maximum* the driver supports, which is what picks the torch wheel below.

### Python

3.11 is what this repo runs on. Conda keeps the environment separate from
whatever else is on the machine:

```powershell
conda create -n sw-train python=3.11
conda activate sw-train
```

### Torch — match the driver, do not take the default

```powershell
# Read what nvidia-smi printed, then pick ONE:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# or cu124 / cu126 for a newer driver
```

The plain `pip install torch` gives a CPU build on Windows. It trains — at
roughly a hundredth of the speed — and nothing warns you. Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True` and a GPU name, or stop and fix it before going further.

### The project

```powershell
git clone https://github.com/muneebelahimalik/SeeWeed3D.git
cd SeeWeed3D
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt
python -m pip install rfdetr
```

Then confirm the install is sound before trusting a run to it:

```powershell
python -m pytest tests/ -q
```

Expect **1137 passed, 1 skipped**. The suite needs no GPU and no dataset; a
failure here is an environment problem, not a data one.

### What you do NOT need on this machine

**SAM 3.** It is only used for prelabeling. Skipping it also means you are not
bound by the `numpy>=1.26,<2` pin that exists solely for SAM 3 — though leaving
numpy where `requirements.txt` puts it is fine and keeps the two machines
comparable.

**pyzed / the ZED SDK.** Capture only.

> Keep prelabeling and training in **separate environments** if you ever do both
> here. `requirements-training.txt` says why: the two stages never run in one
> process, and a training dependency that wants numpy 2 would break SAM 3
> silently.

---

## 3. Point the config at the new paths

Two files, four lines. In `seeweed3d/training/make_dataset.py`:

```python
"SOURCES": [
    {"DATUMARO_ROOT": r"D:\Dataset_Vidalia\onions_20260108_1\sessions",
     "IMAGES_ROOT":   r"D:\Dataset_Vidalia\onions_20260108_1\sessions"},
    {"DATUMARO_ROOT": r"D:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions",
     "IMAGES_ROOT":   r"D:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions"},
],
"OUT_DIR": r"D:\Dataset_Vidalia\training_onion_all_sessions",
```

And in `seeweed3d/training/train_model_rfdetr.py`, `DATASET_DIR` and `RUN_DIR`
must match that `OUT_DIR`. **A mismatch here is the single most common way a run
dies at minute one** — it has happened twice on this project.

---

## 4. Build and train

```powershell
python seeweed3d/training/make_dataset.py
python seeweed3d/training/train_model_rfdetr.py
```

The build should report **6 sessions**. Fewer means a `SOURCES` path is wrong.

Preflight runs before training and writes `RUN_DIR/preflight.json`. It will
report that the labels are unreviewed prelabels — that is expected and is not a
failure. Do not set `SKIP_PREFLIGHT` to silence an error-level finding.

---

## 5. Settings worth raising on a better GPU

The defaults were tuned for a small card. Change **one thing at a time** and
compare against the previous run, or you will not know which change did what.

| Setting | Default | On more VRAM |
|---|---|---|
| `RESOLUTION` | 1008 | **1248 or 1344** — the highest-value change on this data. Must be a multiple of 24 for medium/large |
| `BATCH` × `GRAD_ACCUM` | 2 × 8 | keep the product near 16: `4 × 4` or `8 × 2` |
| `WORKERS` | 0 | **0 for the first run.** On Windows a lightning dataloader worker that fails to start kills the process with no traceback — training simply stops after the model summary. Raise to 2–4 once a run is known to work |
| `VARIANT` | `medium` | `large` only after resolution has been pushed; capacity is rarely the binding constraint here |

**Raise `GRAD_ACCUM` rather than trading resolution for batch size.** That trade
is what threw away small weeds before — it is documented in the CHANGELOG as its
own era.

Overnight is enough for far more than 60 epochs at this dataset size, so
`EPOCHS` can go up. `EARLY_STOPPING` with `PATIENCE: 25` will stop it when the
val score stops moving, and the patience is deliberately higher than rfdetr's
default of 10 because a re-initialised head leaves classes at AP 0.000 for the
first several epochs.

---

## 6. Getting results back

`RUN_DIR` holds everything: checkpoints, `preflight.json`, the metrics and the
figures. Copy the whole folder.

**`checkpoint_best_total.pth`, not `_ema`.** rfdetr writes three files —
`_regular`, `_ema` and `_total`, the last copied from whichever actually won.
Scoring `_ema` silently reports the loser.

To evaluate:

```powershell
python -m seeweed3d.evaluation.eval_seg --backend rfdetr `
    --checkpoint <RUN_DIR>/checkpoint_best_total.pth `
    --dataset <OUT_DIR> --split test --device cuda --sweep
```

`--sweep` is not optional in spirit: a single-threshold table is a claim about a
threshold, not about a model. This project has seen small-weed recall move from
0.28 to 0.73 on unchanged weights.

MLflow and TensorBoard write to a local directory and upload nothing, so the
results are portable with the folder.
