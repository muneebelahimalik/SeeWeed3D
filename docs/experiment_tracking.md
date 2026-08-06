# Experiment tracking and training analytics

Research and decision record. Implemented in `seeweed3d/training/tracking.py`.

---

## 1. The constraint that decided this

Field imagery from a commercial Vidalia operation, plus the geometry of a
customer's rows, is not data to upload to a vendor's cloud in exchange for a
nicer chart. **Both selected backends are local-only and need no account.**

That rules out the tool most people reach for first. Weights & Biases has an
excellent free personal tier, but the free tier is *hosted* — runs, config, and
any logged preview image land on their servers. W&B does offer self-hosting, but
only on paid plans. If the imagery ever stops being yours to publish, the
decision has already been made for you, so it is made here instead.

---

## 2. What was chosen

| Tool | Licence | Where data lives | Job it does |
|---|---|---|---|
| **TensorBoard** | Apache-2.0 | `<run>/tb/` | scalar curves + image previews, live during training |
| **MLflow** | Apache-2.0 | `./mlruns/` | run **comparison** table: params ↔ metrics across every run |

They are not redundant. TensorBoard answers *"is this run learning?"* MLflow
answers *"which of my eleven runs was best, and what was different about it?"*
On a 45-frame dataset where you will sweep learning rate, epochs, class sets and
augmentation, the second question is the one that eats the time.

### Considered and rejected

| Tool | Why not |
|---|---|
| **Weights & Biases** | Free tier is cloud-hosted; self-hosting is paid. Best-in-class UI otherwise. |
| **ClearML** | Capable, but bundles orchestration, data management and deployment you don't need; self-hosting is a server + MongoDB + Elasticsearch to run. |
| **Aim** | Genuinely fast UI, and it can even read an MLflow store. Rejected only because it adds a second tool for a UI preference, not a capability. Reconsider if MLflow's UI becomes the bottleneck. |
| **Neptune / Comet** | Commercial-first, cloud-first. |
| **DVC** | Solves data versioning, not experiment tracking. Worth adding later for the dataset itself — a different problem. |

---

## 3. What is logged

### Scalars, per epoch
`train_loss`, `val_loss`, `lr`, and — when `--eval-every` is on — `val_map50`,
`val_map50_95`, `missed_onion_fraction`.

The mAP metrics are computed on the **current weights**, not on the current
best checkpoint. Evaluating the best one is circular: it cannot say whether
*this* epoch improved, which is exactly what `--select-by map50_95` needs.

### Parameters, once
Learning rate, batch, epochs, seed, class list, train/val counts, plus
`dataset_kind` and `split_strategy` from the manifest. Those last two matter
more than they look: a run against a `frame_block` split is not comparable with
one against a held-out session, and six months from now the run table is the
only thing that will remember which was which.

Also written to `<run>/params.json`, so the record survives even with
`--track none`.

### Provenance, once — what makes a number explainable later

Logged automatically by `tracking.environment_params()`:

| Field | Why |
|---|---|
| `git_commit` | with **`-dirty`** appended when the working tree had uncommitted changes. The flag is the important half: a bare hash *claims* the run is reproducible, and if the tree differed from that commit it is a false claim |
| `seg_manifest_sha256`, `class_mapping_sha256` | `OUT_DIR` is reused between rebuilds, so a dataset **path** is not an identity. The hash ties a run to the exact dataset it saw |
| `torch_version`, `torchvision_version`, `cuda_version`, `cudnn_version` | a latency or accuracy change after an environment upgrade is otherwise unattributable |
| `gpu_name`, `gpu_count`, `gpu_memory_gb` | desktop-vs-Jetson comparisons are meaningless without it |
| `python_version`, `platform` | |

Every field is best-effort. No git binary, no CUDA build, no repo — none of it
stops a training run. Fields that cannot be determined are **omitted**, not
logged as `None`, because a parameter reading `"None"` is indistinguishable
from one genuinely set to that string.

### System metrics — is it the model or the dataloader?

MLflow runs start with `log_system_metrics=True`, sampling GPU utilisation, GPU
memory, CPU, RAM, disk and network throughout the run.

This answers a question a loss curve cannot: **"the epoch is slow"** has
completely different fixes depending on whether the GPU is pinned at 100% (the
model is the bottleneck — smaller backbone, lower resolution, AMP) or idling at
20% (the dataloader is starving it — more workers, smaller decode, cached
frames). Guessing wrong costs a day.

Needs `psutil` (CPU/RAM/disk) and `nvidia-ml-py` (GPU), both in
`requirements-training.txt`. Their absence is **checked before** the run
starts, not caught afterwards: MLflow creates the run in the store and only
then launches the metrics monitor, so a failure there strands an orphan run
that `mlflow.active_run()` no longer reports and nothing can delete — an empty
row in exactly the comparison table MLflow is here for. Without them everything
else is still logged and a warning names the install command.

### Storage: SQLite, not `./mlruns`

**MLflow 3 refuses the plain filesystem backend** — `FileStore` raises *"in
maintenance mode"* on start. The tracking URI is therefore
`sqlite:///<runs>/mlruns/mlflow.db`, with artifacts under
`<runs>/mlruns/artifacts/`. Still one local folder, still no server, still
nothing uploaded.

Two consequences worth knowing:

- The `mlflow ui` command printed at the end of training passes that **URI**.
  Pointing `--backend-store-uri` at the *directory* opens a store MLflow
  refuses.
- With a database backend the artifact root is not implied by the tracking URI.
  Left unset it resolves to `./mlruns` relative to your **working directory**,
  so preview images would land wherever you happened to run `python` from. The
  experiment is created with an explicit `artifact_location` to prevent that.

`MLFLOW_TRACKING_URI` still overrides everything if you want a shared server.

### Prediction previews — the part that actually helps
Every `--preview-every` epochs, a **side-by-side GT vs prediction** panel on a
fixed sample of val frames. Crop instances are outlined thicker than weeds.

This is the highest-value signal at your data volume, and no scalar can
substitute for it. A loss curve cannot tell you that every mask is one plant too
large, or that the model has learned to call every onion a weed — ten seconds
looking at eight overlays will.

The frames are **evenly spaced and fixed**, not resampled each epoch: a preview
that shows different ground every epoch cannot show you whether anything
improved.

Panels are two images with a divider rather than one blended overlay, because
overlapping tints of a correct and an incorrect mask are indistinguishable from
a single mask of a third colour — which is exactly the case you are looking for.

---

## 4. Usage

```powershell
python -m pip install tensorboard mlflow psutil nvidia-ml-py

python -m seeweed3d.training.train_seg_torchvision `
    --dataset     E:/Dataset_Vidalia/training/subset45 `
    --images-root E:/Dataset_Vidalia/sessions `
    --out         E:/Dataset_Vidalia/runs/seg_v1 `
    --epochs 60 --batch 2 --lr 2.5e-3 --device cuda --workers 4 `
    --track all --preview-every 5 --eval-every 20
```

Then, in a second terminal:

```powershell
tensorboard --logdir E:/Dataset_Vidalia/runs/seg_v1/tb
# the URI, not the directory - training prints the exact line to use
mlflow ui --backend-store-uri sqlite:///E:/Dataset_Vidalia/runs/mlruns/mlflow.db
```

| Flag | Meaning |
|---|---|
| `--track auto` | **default** — uses whatever is installed, never fails |
| `--track all` | TensorBoard + MLflow; **errors** if either is missing |
| `--track none` | no-op; `params.json` and `history.json` are still written |
| `--preview-every N` | overlay panels every N epochs (`0` disables) |
| `--eval-every N` | val mAP every N epochs (`0` disables — it is slow; a full pass over val) |
| `--select-by` | what `best.pt` is chosen on: `val_loss` (default), `map50`, `map50_95` |

`val_loss` is the default only because it is free. It is a poor proxy for a
detector — it sums classification, box and mask terms whose scales have nothing
to do with whether a plant was found — and on a small set it often bottoms out
long before detection quality peaks. Training now prints which epoch won and
flags an early peak; if the winner is in the first 40% of the schedule, switch
to `map50_95` and compare.

---

## 7. Where this stops, and what replaces it

Experiment tracking answers *"which training run was best?"*. It cannot answer
*"is the deployed model now seeing field conditions unlike its training data?"*
— a different question needing a different tool ([Evidently](https://www.evidentlyai.com/),
Apache-2.0, also self-hostable).

That is deliberately **not** installed yet. Drift monitoring needs a trained
model, a locked test set, a repeatable inference pipeline and several field
sessions to compare against a reference distribution. With one annotated
session there is no reference distribution to drift from, so it would produce
confident-looking output with nothing behind it. Revisit after the first real
field deployment.

Note the shape of the eventual integration when you get there: drift on raw
pixels is meaningless, so the comparison runs over derived per-frame features —
brightness, ExG distribution, blur score, depth-valid fraction, weed density,
abstention rate, p95 latency — most of which this pipeline already computes.

---

## 5. Failure behaviour, stated deliberately

**Tracking can never kill a training run.** Preview rendering, mid-training
evaluation, and MLflow calls are all wrapped — a failure prints a warning and
training continues. Losing a 3-hour run to a charting library would be absurd.

**One exception**: `--track mlflow` or `--track all` with the package missing
exits immediately with the install command. Silently downgrading an explicitly
requested backend to a no-op would let you finish a run believing it was logged.
`--track auto` never fails, because you didn't ask for anything specific.

`None` and `NaN` metrics are dropped rather than logged — a missing metric
plotted as `0.0` reads as a catastrophic result.

---

## 6. What this does not give you

Tracking is bookkeeping, not evidence. With 32 training frames from **one
session**, a beautiful MLflow table comparing twenty runs still compares twenty
numbers measured on the same drive, in the same light, on many of the same
individual plants. It will reliably tell you which hyperparameters fit *this
recording* best. It cannot tell you which will work in a different field.

Only a held-out **session** can support that claim, and the runbook's
`--holdout-test` flag exists for the day you have one.

---

## Sources

- [MLflow Tracking documentation](https://mlflow.org/docs/latest/ml/tracking/)
- [A Comprehensive Comparison of ML Experiment Tracking Tools — Towards Data Science](https://towardsdatascience.com/a-comprehensive-comparison-of-ml-experiment-tracking-tools-9f0192543feb/)
- [We Tested 9 MLflow Alternatives for MLOps — ZenML](https://www.zenml.io/blog/mlflow-alternatives)
- [8 Best Neptune AI Alternatives — ZenML](https://www.zenml.io/blog/neptune-ai-alternatives)
- [awesome-ml-experiment-management](https://github.com/awesome-mlops/awesome-ml-experiment-management)
