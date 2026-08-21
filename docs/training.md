# Stage A training: which model, and what was changed

The complete record of the segmentation training path — the two backends, every
setting that was changed from its default and why, and the defects that were
only found by running it.

**Related:** [SAM 3 prelabeling](sam_prelabeling.md) ·
[weed active learning](weed_active_learning.md) ·
[Stage A improvements](stage_a_improvements.md) ·
[experiment tracking](experiment_tracking.md) · [CHANGELOG](../CHANGELOG.md)

---

## Two backends, one interface

Everything downstream sees `Detections`, a plain numpy structure, so the
segmenter is the only component that knows which model produced it.

| backend | licence | role |
|---|---|---|
| `maskrcnn` | BSD-3-Clause | torchvision. The **prototype default** — no new dependency, mature, easy to fine-tune on a small set. Not real-time. |
| **`rfdetr`** | **Apache-2.0** | Roboflow **RF-DETR-Seg**. What both the onion and weed models are trained on now. Real-time, TensorRT FP16, ONNX export for the Jetson Orin. |
| `rtmdet` | Apache-2.0 | documented as viable, **not implemented** — the mmengine/mmcv stack is version-fragile |
| `ultralytics` | **AGPL-3.0** | strong, but never the default and only reachable by asking for it explicitly |

> **That boundary exists for licensing, not tidiness.** A commercial laser
> weeder is exactly the case an AGPL dependency makes expensive, and unlike a
> code bug it cannot be fixed after the fact by editing the source. Only
> RF-DETR-Seg **Nano–Large** are offered: XLarge/2XLarge may fall under
> Roboflow's Platform Model License.

**Onions and weeds use the same architecture and the same runner.** They differ
only in dataset, class list and run directory — which is deliberate, because a
scene-specific architecture would make the two sets of numbers incomparable.

---

## Why RF-DETR-Seg, concretely

Verified against `rfdetr 1.9.1` rather than assumed. Things it exposes that
torchvision's Mask R-CNN does not, without forking its ROI heads:

| capability | why it matters here |
|---|---|
| `mask_ce_loss_coef` / `mask_dice_loss_coef` | mask loss is weightable — the entry point for Tversky |
| `use_ema` / `ema_decay` | EMA weights, free variance reduction on a tiny dataset |
| `early_stopping{,_patience,_min_delta}` | built in |
| `multi_scale` / `expanded_scales` | multi-scale training |
| `grad_accum_steps` | effective batch 16 on 24 GB VRAM at 1008 px |
| `tensorboard` / `mlflow` | native logging |

And the structural reason: **DETR-style set prediction has no anchors**, so the
anchor/resolution trap that cost the Mask R-CNN path its small weeds cannot
recur in the same form.

---

## The single largest measured gain in the project

**Resolution (#49).** torchvision's default resize took 2208×1242 ZED frames
down to 1333×749, shrinking a 250 px weed to **9.5 px — below the smallest 32 px
RPN anchor, so the proposal network could never propose it.**

`min_size`, `max_size` and `anchor_sizes` became configurable and are stored in
the checkpoint, so inference matches training.

### Augmentation was quietly undoing it (#53)

`v2.ScaleJitter(target_size=...)` resizes the **canvas**; only
`v2.RandomAffine(scale=...)` scales the **content**. Measured, the augmented
frames were **24–73% of native** — so `MIN_SIZE: 1000` was doing nothing at all.

```python
# seg_dataset.py - RandomAffine, NOT ScaleJitter
v2.RandomAffine(degrees=15 if strong else 7, ...)
```

The same PR fixed a crash three minutes into a run:

```
AssertionError: All bounding boxes should have positive height and width.
Found invalid box [1432.63, 997.66, 1436.67, 997.66]
```

`masks_to_boxes` uses **inclusive** min/max, so a 1-px-tall mask yields a
zero-height box. The guard existed before augmentation but not after it.

### The same trap, in the new backend's default

RF-DETR-Seg's **default is 432×432**. Left alone, switching to it would have
**regressed small-weed recall while looking like an upgrade**.

So the runner *refuses* to train at the default without
`--allow-default-resolution`, and validates that the value is a multiple of
`patch_size × num_windows` — read out of the package's own config rather than
hard-coded as 24, so it cannot drift.

---

## Current training settings, and why each is not the default

`training/train_model_rfdetr.py`, with the weed runner overriding only the paths
and the round.

| setting | value | reason |
|---|---|---|
| `VARIANT` | `medium` | Nano–Large only; medium is the largest that trains at 1008 px on 24 GB |
| `RESOLUTION` | `1008` | **not** 432. Multiple of 24. Resolution has bought more than capacity on this data every time |
| `BATCH` / `GRAD_ACCUM` | `2` × `8` | effective batch 16. VRAM grows with the *square* of resolution, and an OOM eight hours into an overnight run costs the night |
| `LR` | `1e-4` | |
| `LR_SCHEDULER` | `cosine` | **not** rfdetr's `step` — see below |
| `WARMUP_EPOCHS` | `1.0` | |
| `USE_EMA` | `True` | |
| `MULTI_SCALE` | `True` | |
| `EARLY_STOPPING` / `PATIENCE` | `True` / `25` | higher than rfdetr's default 10: a re-initialised head leaves whole classes at AP 0.000 for the first several epochs |
| `WORKERS` | `0` | on Windows a dataloader worker that fails to start kills the process after the model-summary table **with no traceback** |
| `TVERSKY_ALPHA` / `TVERSKY_BETA` | `0.3` / `0.7` | recall-leaning — see loss shaping |
| `FOCAL_GAMMA` | `1.0` | |
| `TRACK` | `auto` | |

### The scheduler was the sharpest of these (#58)

rfdetr defaults `lr_scheduler="step"` with `lr_drop=100`, so **on any run
shorter than 100 epochs the step never fires and the learning rate is constant
from start to finish.** The first run ended at the same 1e-4 it began with.
Nothing warns about this.

---

## Loss shaping (#63)

Dice weights a false positive and a false negative equally. **Nothing about this
system does:** a missed weed survives to set seed, a spurious one costs one laser
pulse, and onion the model fails to mark is onion the targeting stage has no
reason to protect.

```
TI = TP / (TP + ALPHA·FP + BETA·FN)          loss = (1 − TI)^GAMMA
```

`alpha = beta = 0.5` is **exactly** Dice — asserted against rfdetr's own
`dice_loss` to 1e-6 on random logits. That is what keeps a Tversky run
comparable with every Dice run already recorded.

At `alpha 0.3 / beta 0.7` a false negative costs more than twice a false
positive, which is the asymmetry the field imposes.

**It installs by rebinding rfdetr's module-level `dice_loss_jit`**, and *raises*
if that symbol is absent rather than silently training with Dice while the
config claims Tversky.

> **Watch precision in the confidence sweep.** If weed precision falls further
> than small-weed recall rises, the answer is a higher operating confidence, not
> a higher BETA.

### Provenance for a runtime patch (#64)

The loss is patched in at runtime, so nothing rfdetr writes records it — two
runs differing only in alpha/beta shipped **byte-identical sidecars**.
`rfdetr_train_config.json` now carries a `mask_loss` block, **on Dice runs too**,
so an absent block never has to be read as "probably the default".

---

## Preflight: what to check before committing hours

`training/preflight.py` runs before the first epoch. **Every finding is
something that makes a run finish successfully, print a plausible metric, and
mean nothing.** None of them raise during training, which is exactly why they
need a pass of their own.

| # | finding | consequence if unchecked |
|---|---|---|
| 1 | a class under ~20 instances | contributes an AP near zero that drags the mean down for a reason unrelated to the model |
| 2 | a class in train with **zero** in val | no validation signal at all — early stopping and best-checkpoint selection are blind to it. When that class is the crop, "best" may be the checkpoint that segments onions *worst* |
| 3 | a class missing from **training** | the model can never predict it, and a class it never predicts reports an empty mask — indistinguishable from "looked and found nothing" |
| 4 | epochs vs **steps** | 60 epochs over 60 frames is 1,800 optimiser steps; over 600 frames it is 18,000. Steps is what a schedule and a patience are really denominated in |
| 5 | patience longer than the run | early stopping cannot fire, so the run has none regardless of the config |
| 6 | val below ~10 frames | epoch-to-epoch noise exceeds the differences between checkpoints, so "best" is chosen by chance |

Findings 4 and 6 both fired on the weed round-0 run and were correct: 42 frames
at effective batch 16 is **3 steps/epoch**, and val was 10 frames.

---

## Five defects found only by running it

Each of these produced a run that completed and reported a number.

- **(#52) MLflow's tracking URI is read at import.** rfdetr builds its
  `MLFlowLogger` without one, and Lightning reads `MLFLOW_TRACKING_URI` as a
  **default argument evaluated at module import** — so the environment variable
  must be set *before* rfdetr is imported. Same PR: `--backend` was parsed and
  dropped in `eval_seg`, which would have scored an RF-DETR checkpoint through
  the Mask R-CNN builder.

- **(#57) The command printed to score the first run would have produced
  nonsense** — wrong architecture, wrong resolution, wrong class list. None of
  the three raise. All now travel with the run in `rfdetr_train_config.json`.

- **(#58) The guard from #57 fired immediately**: *"predicted class id 4 but was
  loaded with 4 classes"*. It was right to stop.

- **(#59) The label mapping was wrong twice**, so the third time it was read out
  of the package rather than reasoned about. `PostProcess` computes
  `labels = topk_indexes % out_logits.shape[2]` — **raw 0-based labels** over
  ascending category-id order. And LW-DETR allocates `num_classes + 1`
  classifier outputs, where the extra slot has no training target and must be
  dropped.

- **(#63) `checkpoint_best_total.pth`, not `_ema`.** rfdetr keeps three files and
  copies `_total` from whichever actually won, so naming the EMA file silently
  scores the loser whenever the regular weights were better — **which had
  already happened once.**

---

## Evaluation built alongside it

Training numbers are only as good as what reads them.

**Crop burn, not just crop misses (#51).** `missed_onion_fraction = 0.181` gives
no way to tell whether the model is *ignoring* 18% of the crop or *aiming a 60 W
laser at it*. Those are not the same failure and do not have the same fix.
`weed_on_crop_px` counts ground-truth onion pixels a **weed** prediction claims;
pixels claimed by both crop and weed predictions are excluded, because the
onion-conflict check suppresses that shot.

**The operating point is part of the result (#55).** run4 scored small-weed
recall **0.277 at conf 0.5 and 0.732 at conf 0.25 on unchanged weights.** The
number was measuring the threshold, not the model. `--sweep` reports the full
curve so the deployment confidence is *chosen* rather than inherited.

**Evaluation ran long enough to be killed by hand (#60).** Not the model — mask
IoU at 3.84 ms/pair on 2208×1242 masks. A bbox pre-filter plus an
intersection-box AND, with the matrix reused across thresholds and confidences:
**120× faster.**

**A refactor regression (#56).** Extracting the operating point into
`_summarise()` dropped `operating = {"conf": conf}`, so `--sweep` raised
`KeyError: 'conf'` on the first real run. The deeper gap: every `format_report`
test used hand-built dicts, so **none of them exercised real output.**

**Inference on unlabelled frames (#54).** Everything that drew a prediction
needed ground truth, so nothing could be pointed at a held-out session, a new
field, or tomorrow's drive — the frames that actually decide whether the model
is worth deploying. That is `perception/predict_images.py`, and
`training/datasets/weeds_look.py` wires it to the weed paths.

**One plant, one detection (#111).** RF-DETR predicts a *set*: every query
proposes independently, and nothing makes two queries that found the same plant
agree on what it is. The same mask came back under two class labels at IoU
**1.000** in 6 of 16 frames, with no NMS anywhere in the inference path.
Suppression is class-agnostic and mask-based — see `common/dedup.py`.

---

## Tracking

Backends are chosen for one property above all: **nothing leaves the machine.**
Field imagery and the geometry of a commercial onion operation are not data to
upload to a vendor's cloud for the convenience of a nicer chart.

| backend | answers |
|---|---|
| `tensorboard` | *is this run learning* |
| `mlflow` | *which of my eleven runs was best, and what was different about it* |

Both write to a local directory and neither needs an account. Every backend is
optional and `Tracker(backend="none")` is a working no-op — but an **explicitly
requested** backend that is missing *raises*, because silently degrading it to a
no-op would let you finish a run believing it was logged.

**Both backends produce the same analytics (#61).** RF-DETR was being handed to
its own logger, which produces MLflow scalars and nothing else, while the richer
`Tracker` path stayed on the Mask R-CNN side — so one backend left preview
panels behind and the other left a CSV, and the two were not comparable in any
single view. `evaluation/analyze_run.py` now generates training curves,
per-class AP, the confidence sweep, the crop-safety curve and recall-by-size for
either backend, detecting which from what the run directory contains.

**Visual report (#43).** One self-contained HTML file: recall bucketed by
instance size, crops of every missed weed smallest first, per-frame ground truth
beside prediction coloured **by outcome rather than by class**, and crop safety
kept out of every averaged score.

---

## What has NOT been done, deliberately

| | why |
|---|---|
| **XLarge / 2XLarge variants** | may fall under Roboflow's Platform Model License |
| **Teacher–student / distillation** | pseudo-labelling the crop is unsafe here — a confident wrong onion label becomes training truth |
| **Temporal tracking across frames** | not started; see `dataset_growth.md` |
| **Architecture comparison** | the current limit is **label provenance**, not architecture. A comparison run on unreviewed prelabels measures agreement with SAM, not accuracy — see [`stage_a_improvements.md`](stage_a_improvements.md) |
| **Dedup in `"full"` mode** | that is the crop-safety path, and it will not be changed as a side effect of a display fix |

---

## The lesson that shaped this path

**A silent wrong answer beats a loud one every time — so make it loud.**

Wrong architecture, wrong class list, wrong resolution, a dropped `--backend`, a
learning rate that never decayed, a label mapping off by one, an EMA checkpoint
scored instead of the winner: **none of these raise.** Every one produced a
completed run and a plausible number. Most of the guards in the training path
exist because one of them didn't.
