# The weed active-learning loop

Turning a few dozen hand-annotated weed frames into a dataset worth training
on, without annotating the whole pool.

**Related:** [dataset growth](dataset_growth.md) · [dataset assembly](dataset_assembly.md) ·
[new machine setup](new_machine_setup.md) · [runbook](RUNBOOK.md)

---

## The loop, and the one thing to get right about it

```
   ┌─ train on what you have
   │        ↓
   │  mine the pool ──── rank by what the model gets WRONG
   │        ↓
   │  correct in CVAT ── a human, always
   │        ↓
   │  rebuild + measure
   └────────┘
```

**Select the frames the model gets wrong, not the ones it gets right.**

The intuitive version — run the model, keep the frames it did well on, add
those to training — teaches almost nothing. Where the model is already right
the loss is already near zero, so there is no gradient and nothing to learn.
All that changes is that the model grows more confident about what it already
knew while its blind spots stay exactly as blind.

That version also feeds the model's own predictions back as ground truth, so
its confident errors become training truth and compound. Nothing in this
pipeline writes a prediction into a manifest: every exported frame goes to a
person first.

> **The export is CORRECTION, not annotation.** Because the ranking model is
> *yours*, it already knows this ontology — so the classes are usually right
> and the work is fixing boundaries and adding what was missed. That is much
> faster than drawing from nothing, which is the entire economic case for the
> loop.

---

## Before round 0: pin a holdout

Name one or two sessions in **both** places, and never annotate them:

```python
# training/datasets/weeds.py
HOLDOUT_TEST = ["vid2_20260108_122731"]
```

`weeds_mine.py` reads the same list, and a test asserts they agree — because
they are separate settings in separate files and one being right does not make
the other right.

**This matters more here than anywhere else in the project.** Mining selects
the frames the model finds *hardest*, which are precisely the frames it would
most benefit from having seen. Mining a test session does not merely leak it —
it leaks it in the most flattering possible direction, and every later round
will look like progress.

Without a holdout, round 3 scoring better than round 1 tells you nothing: you
cannot separate a better model from an easier test set.

---

## Round 0 — the starting model

```powershell
# 1. Build from what is already annotated
python -m seeweed3d.training.datasets.weeds

# 2. Train
python -m seeweed3d.training.datasets.weeds_train
```

Check the build reports the sessions you expect, and that preflight does not
report an error-level finding. It will note the label provenance — for this
build that is `hand_corrected`, which is what makes these scores mean something.

**Do not tune anything after this.** Settle resolution, variant and seed on
round 0 and then leave them alone: changing the architecture in the same step
that adds data means the improvement cannot be attributed, and attributing it
is the only reason to run a loop rather than annotate at random.

---

## Round N — mine, correct, rebuild

### 1. Mine

```powershell
# set ROUND = N and CHECKPOINT to round N-1's run, then:
python -m seeweed3d.training.datasets.weeds_mine
```

Writes a CVAT-ready folder with the model's predictions already in it, and
records the batch in `al_rounds.json` so the same frames are not selected again
next round while they sit unfinished in CVAT.

**Round 0→1 is different.** Uncertainty ranking needs a model good enough for
its uncertainty to mean something, and one trained on a few dozen frames is
not. For the first batch lean on **coverage** — the diversity pass already
spreads the selection across sessions — and start trusting the ranking from
round 2.

### 2. Correct in CVAT

Upload the batch, correct it, export **Datumaro 1.0** back into the session's
`annotations/` folder.

> **Watch for what is ABSENT.** A pre-labelled frame biases you toward
> accepting what is there and not noticing what is missing — and missed weeds
> are this project's failure mode. The batch report prints the model's own
> recall at the export threshold as a reminder of roughly how many instances
> per frame it is expected to have left out.

### 3. Mark the round returned

```powershell
python -m seeweed3d.training.al_round merge --dataset <weeds_v1>
```

If a batch will never be finished, release its frames instead of leaving them
blocked:

```powershell
python -m seeweed3d.training.al_round abandon --dataset <weeds_v1> --round N
```

### 4. Rebuild and retrain

```powershell
python -m seeweed3d.training.datasets.weeds
# bump ROUND in weeds_train.py, then:
python -m seeweed3d.training.datasets.weeds_train
```

### 5. Measure, and record it against the round

```powershell
python -m seeweed3d.evaluation.eval_seg --backend rfdetr `
    --checkpoint <runs>/weeds_rN/checkpoint_best_total.pth `
    --dataset <weeds_v1> --split test --device cuda:1 --sweep

python -m seeweed3d.training.al_round metrics --dataset <weeds_v1> --round N ...
python -m seeweed3d.training.al_round status  --dataset <weeds_v1>
```

`status` prints the round history with deltas — which is the only thing that
answers *"is this working?"*.

**`checkpoint_best_total.pth`, not `_ema`.** rfdetr keeps three files and
`_total` is copied from whichever actually won; scoring `_ema` silently reports
the loser.

**`--sweep` is not optional in spirit.** A single-threshold table is a claim
about a threshold, not a model — this project has seen small-weed recall move
from 0.28 to 0.73 on unchanged weights.

---

## When to stop

Stop when a round stops paying. Concretely: when test-set weed recall moves by
less than its round-to-round noise, more annotation of the *same kind* has
stopped buying anything, and the constraint has moved somewhere else — usually
to scenes the pool does not contain at all.

That is a result, not a failure: it tells you to go and record different
ground rather than label more of the same.

---

## Why weed-only frames are worth more per frame than mixed

The class label is **free and certain**. Every plant in a weed-only drive is a
weed by construction of the recording, not by anyone's judgment. The expensive
question in a mixed scene — *which* plant is this — does not arise.

What is not free is **instance identity**: which pixels belong to which plant.
That is the entire cost of annotating these frames, and it is what the loop is
buying.

It is also why `weed_cluster` deserves care here. It exists for weeds with no
separable crown, and using it because separation is *tedious* rather than
*impossible* teaches the model to do the same at runtime — where it means weeds
that never receive an individual LEP and never get targeted. Its rate per round
is worth watching for exactly that reason. See
[the dataset strategy](mixed_dataset_strategy.md#weed_cluster-is-defined-by-crowns-not-by-foliage).

---

## Settings you may want to change, and ones you should not

| Setting | Where | Change it? |
|---|---|---|
| `ROUND` | `weeds_train.py`, `weeds_mine.py` | **every round** |
| `CHECKPOINT` | `weeds_mine.py` | every round — point at N−1 |
| `BATCH_SIZE` | `weeds_mine.py` | size it to what you will actually annotate |
| `HOLDOUT_TEST` | `weeds.py` | once, before round 0 |
| `SEED` | `weeds.py` | **never** — a new seed re-draws every split |
| `RESOLUTION`, `VARIANT` | `weeds_train.py` | settle on round 0, then leave alone |

An over-large batch goes stale, and a stale batch blocks its frames from being
re-selected until it is merged or abandoned. Forty to sixty is what the ledger
assumes.
