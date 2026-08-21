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

## What you edit, and what you can ignore for now

| File | Edit | Needed for round 0 |
|---|---|---|
| `datasets/weeds.py` | `WEED_SESSIONS` — full path(s) to your annotated session folder(s) | **yes** |
| `datasets/weeds_train.py` | `ROUND` | **yes** |
| `datasets/weeds_mine.py` | `ROUND`, `CHECKPOINT` | no — round 1+ |
| `datasets/common.py` | `DATA_ROOT`, or set `SEEWEED3D_DATA_ROOT` | only decides where builds and runs are written |

`WEED_SESSIONS` takes either form, and both are found the same way:

```python
WEED_SESSIONS = [
    # ONE session folder holding annotations/ + rgb/ + depth/ + meta/
    r"D:\...\sessions\vid2_20260108_122731",
    # or a folder whose CHILDREN are session folders - every one is discovered
    # r"D:\...\sessions",
]
```

With a single session, name it directly. The `sessions` form only earns its
keep once several sit under one parent.

### At 60 frames the split settings are not the ones a big build uses

Carrying the onion build's settings over (3 blocks, a 12-frame gap, 15/15) spent
**24 of 60 frames on buffers** and produced `val=5 test=5` — two splits too
small for any number computed on them to mean anything, and a training set of
26. One block has two seams instead of six, and that is the whole difference:

| blocks | gap | test | train | val | test | binned |
|---|---|---|---|---|---|---|
| 3 | 12 | 0.15 | 26 | 5 | 5 | **24** |
| 1 | 8 | 0.0 | **42** | **10** | 0 | **8** |

**`TEST_FRACTION` is 0 on purpose.** A 9-frame test set has error bars wider
than the number it reports — not a weaker measurement, a misleading one. Val
does double duty: it selects the checkpoint *and* tracks round-to-round change,
which is what the loop needs. Optimistic as an absolute score, consistent as a
relative one.

### A contiguous block takes whatever the drive put there

Round 0 trained cleanly and reported a per-class table that looked like one
weak class:

| class | AP 50:95 | IoU |
|---|---|---|
| cutleaf_evening_primrose | 0.557 | 0.808 |
| grass_weed | 0.458 | 0.795 |
| **other_weed** | **0.214** | 0.707 |

It was not one weak class. It was one badly placed one:

| class | train | val | val share of instances |
|---|---|---|---|
| cutleaf_evening_primrose | 642 | 73 | 10% |
| grass_weed | 327 | 26 | 7% |
| **other_weed** | **145** | **74** | **34%** |
| *frames* | *42* | *10* | ***19%*** |

Val holds 19% of the frames and 34% of every `other_weed` instance in the
dataset. The class is **starved in training and over-weighted in the score at
the same time**, and the mean AP carries the difference.

**None of this is visible in an AP table** — the table reports the score, never
the split that produced it. So the build now prints the class balance next to
the frame split, and flags any class whose share drifts by more than 1.5×.

> The tolerance is set from this case, not from taste. The skew above is a
> ratio of 1.79; a rounder-looking 2.0 would have said nothing about it.

**The layout is now chosen rather than hashed.** With one block there are
exactly three possible layouts, and the rotation used to be picked by a CRC32
of the session key — blind to what was in the frames. It now lays out all three
and keeps whichever puts each class's val share closest to the frame share.
Only ground-truth label counts are consulted, never a model or a score, so this
is ordinary stratification applied to the one axis a contiguous split still has
free. Ties keep the layout the seed would have picked.

### When balance and separation compete, balance wins here

A class confined to one stretch of the drive has *no* layout that fixes it —
three rotations cannot un-cluster it. The lever is **more blocks, paid for with
a smaller gap**: each block samples a different stretch, so a clustered class
stops being all-or-nothing.

On a synthetic reproduction of this exact skew:

| blocks | gap | train | val | binned | `other_weed` val share (want ~19%) |
|---|---|---|---|---|---|
| 1 | 8 | 42 | 10 | 8 | **4%** |
| 2 | 4 | 39 | 9 | 12 | 4% |
| 3 | 4 | 36 | 8 | 16 | **24%** |

Blocks cost one seam each, so buying them without lowering `GAP_FRAMES` spends
the dataset on buffers instead — which is exactly the mistake the 3/12 settings
made the first time.

**Here that trade is nearly free**, for the reason in the next section: the
separation floor is cleared at a very small gap on this session, so gap frames
past about 3 buy nothing and can be spent on blocks instead.

The real build, at 3 blocks and a 3-frame gap, came out:

| class | train | val | val share |
|---|---|---|---|
| cutleaf_evening_primrose | 604 | 92 | 13% |
| grass_weed | 232 | 63 | **21%** |
| other_weed | **199** | 18 | 8% |
| *frames* | *39* | *9* | ***19%*** |

`grass_weed` is now balanced, `other_weed` moved from 34% *over* to 8% *under*,
and — the part that matters — its training instances went from **145 to 199**.
That is the trade the layout is built to make: a worse *estimate* of a
better-trained class beats the reverse.

### The two directions of skew are not equally bad

A class **concentrated** in val is starved in training *and* over-weighted in
the score. A class **thin** in val trains on nearly everything and merely gets a
noisy AP. So `OVER_REPRESENTATION_PENALTY` weights the first direction 2× when
the layout is chosen, and the build's warning names which one it hit.

This is the same rule the project already applies to crop safety: an asymmetric
pair of errors is never averaged into one number.

### The separation floor, and a stride estimate that was wrong

The build warns when a val frame sits closer than 60 **video** frames to a
training frame. Measured on this session:

| `GAP_FRAMES` | nearest train frame |
|---|---|
| 8 | 240 |
| 3 | **95** |

That is ~30 video frames per pool frame, so `GAP_FRAMES=2` already clears the
floor. An earlier note here estimated the curation stride at ~2 and concluded
the floor was unreachable without spending half the dataset — that was wrong,
and the measured numbers above replace it. Gap frames are **cheap** on this
pool, which is why 3 blocks at gap 3 costs only 12 buffered frames.

Clearing the floor still only means val is not the *same photograph* as train.
It shares the session's light, soil and growth stage regardless. So read the val
score as *"training is working"*, never as *"this is how it performs"* — the fix
for that is a second weed session, not a bigger gap.

### With one session, expect a frame-block split

A session-level split needs at least two sessions, so the build falls back to
contiguous blocks *within* the one you have, and says so. That is the honest
fallback and its scores are an **upper bound**: val and test share the
session's light, soil and growth stage.

`HOLDOUT_TEST` must stay empty until you have a second weed session — holding
out your only one leaves nothing to train on. A test asserts that, so it cannot
be set by accident.

---

## Before round 0: pin a holdout

**Once you have two or more weed sessions**, name one in `HOLDOUT_TEST` and
never annotate it:

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
    --dataset <weeds_v1> --split test --device cuda --sweep

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
