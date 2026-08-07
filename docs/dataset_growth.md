# Growing the dataset

The dataset is the limit. Two runs said so with numbers, not intuition:

- at conf 0.15 — the most permissive setting crop safety allows — **18% of small
  weeds are never found at any threshold**
- `other_weed` sits at **AP@50:95 = 0.25 over 86 instances**, the worst class on
  every axis, and it is a catch-all so it is visually heterogeneous *by
  definition*
- the model predicted `onion_plant` **10 times in two frames of a weeds-only
  session** — thin grass blades read as onion leaves, because it has never seen
  the two in the same frame

None of that is a hyperparameter. All of it is data.

This document is the plan for the next few hundred frames, ordered by **learning
per hour of annotation**, which is the only budget that matters at 78 labelled
frames.

---

## Contents

| | |
|---|---|
| [0. Do this first](#0-do-this-first-a-held-out-session) | a held-out session, before anything else |
| [1. The loop](#1-the-loop) | mine → correct → merge → retrain |
| [2. Active learning](#2-active-learning-which-frames) | which frames, and why the obvious answer is wrong |
| [3. Model-in-the-loop prelabels](#3-model-in-the-loop-prelabels) | correction instead of annotation |
| [4. Targeting the failures](#4-targeting-the-specific-failures) | the three named gaps above |
| [5. Semi-supervised](#5-semi-supervised-teacher-student) | what it buys, and where it is dangerous here |
| [6. Not worth it yet](#6-techniques-that-are-not-worth-it-yet) | synthetic data, SAM refinement, MixUp |
| [7. Annotation quality](#7-annotation-quality) | the failure that survives every model change |

---

## 0. Do this first: a held-out session

**Every number you have is from same-session validation.** `val` is 16 frames
from the same two drives as `train` — same light, same soil, often the *same
plants*. Those numbers show training works. They are not evidence the model
generalises, and until that changes you cannot tell whether new data helped.

Pick one session you have **not** annotated, from a different drive — ideally a
different time of day. Annotate ~40 frames of it. Never train on it.

```python
# mine_pool.py — so it can never be drawn into an annotation batch
"HOLDOUT_SESSIONS": ["vid3_20260108_110444"],
```

Then `make_dataset.py` puts it in `test`, and `analyze_run.py --split test`
gives you a number that means something. Everything below is measured against
it; without it you are tuning against a mirror.

**This is 40 frames that do not improve the model at all**, and it is still the
highest-value annotation you can do right now, because it is what tells you
whether the next 300 frames worked.

---

## 1. The loop

```
   trained model
        │
        ▼
   mine_pool.py ──────►  cvat_ready/ + instances_default.json
        │                        │
        │                        ▼
        │                   CVAT: correct
        │                        │
        │                        ▼
        │                  export Datumaro 1.0
        │                        │
        │                        ▼
        │                 make_dataset.py  (merges every export)
        │                        │
        │                        ▼
        └──────────────  train_model.py / train_model_rfdetr.py
                                 │
                                 ▼
                          analyze_run.py --split test
```

One round is 60 frames. Retrain after each round rather than annotating 300 in
one go: the ranking is only as good as the model doing it, so a mid-course
model picks better frames than a stale one. Three rounds of 60 beats one round
of 180.

```powershell
python seeweed3d/annotation/mine_pool.py     # edit the config block first
```

---

## 2. Active learning: which frames

`training/active_learning.py` scores every frame on four signals, then does a
diversity pass. All four are there for a reason.

| Signal | What it finds | Weight |
|---|---|---|
| **uncertainty** | detections in the ambiguous confidence band (0.25–0.70) | 1.0 |
| **rarity** | frames predicted to contain scarce classes, `1/√(1+n)` | **1.2** |
| **crop risk** | a weed candidate within 40 px of predicted onion | 0.6 |
| **LEP quality** | poorly-localised growth points | 0.8 (inert until Stage B is trained) |

Rarity is weighted highest on purpose. With `wild_radish` and `weed_cluster` at
or near zero instances, **class coverage is worth more than refining a boundary
the model already roughly knows** — a class with no examples can never be
learned at all.

### Why pure uncertainty sampling fails here

It is the classic trap and it would bite hard on this data. If one frame
confuses the model, the frames either side of it confuse it *identically* —
consecutive ZED frames are near-duplicates. A pure uncertainty ranking returns
twenty pictures of one difficult patch, and you would spend a full annotation
round teaching the model about a single plant.

So the final selection is **farthest-point on an appearance descriptor**, seeded
by score. You get the informative frames, spread across the pool. `STRIDE` in
the config attacks the same problem earlier by not even scanning every frame.

### Cold start

With no model there is nothing to be uncertain about, and
`select_cold_start()` falls back to pure appearance diversity. That is what the
*first* batch should have been. You are past this now.

---

## 3. Model-in-the-loop prelabels

The SAM 3 prelabelers propose masks with **generic** classes, which is why
earlier CVAT tasks arrived full of confidently wrong labels and every frame
needed its classes rewritten. Your own checkpoint knows this ontology, so
`mine_pool.py` exports **your model's predictions** as the prelabels. The work
becomes correction: fix boundaries, fix the occasional class, add what was
missed.

Rough economics on this data: annotating a dense frame from scratch is 10–20
minutes; correcting a good prelabel is 3–5. That is the difference between 60
frames a week and 60 frames a day.

### The bias this introduces, and what to do about it

**A pre-labelled frame makes you accept what is there and stop looking for what
is not.** Missed weeds are precisely this project's failure mode, so the
prelabels bias the annotator toward the model's existing blind spot.

Three mitigations, all cheap:

1. `CONF: 0.20` — export prelabels *below* your deployment threshold. A spurious
   mask costs one keystroke to delete; a missing one costs you noticing an
   absence, which is much harder.
2. Before each round, open `analysis/report.html` and look at the **missed-weed
   gallery**. Five minutes of seeing what the model misses re-tunes your eye for
   the whole batch.
3. Sweep each frame once with the prelabels **hidden** before submitting.

---

## 4. Targeting the specific failures

Generic mining will not fix three named problems. Each needs deliberate frames.

### `other_weed` — the worst class

86 instances, AP@50:95 = 0.25, recall 0.128 at conf 0.5. It is a **catch-all**,
so it is not one visual thing and no amount of data makes it one.

Two options, and the second is better:

- **more of it** — the rarity term already favours it as its count is low
  relative to `cutleaf_evening_primrose`
- **split it** — if two or three species dominate what is currently
  `other_weed`, giving them their own classes converts an unlearnable
  catch-all into learnable categories. Look at the report gallery and count
  what is actually in there. This is an *ontology* change, so do it before the
  next few hundred frames rather than after.

Keep `other_weed` as the genuine remainder either way — the laser treats every
weed alike, so a wrong *weed* class costs nothing operationally.

### Grass read as onion

The model predicts `onion_plant` on thin grass blades in weeds-only sessions.
Both are narrow monocot blades, and **it has never seen them in the same
frame** — your onion frames came from onion-only sessions, your weed frames
from weed-only ones. It has no reason to have learned what separates them.

The fix is **mixed scenes**: frames containing onions *and* grass together. If
your recordings have crop rows with grass between them, those are the frames
worth the most in the entire pool. Mine them specifically:

```python
"ONLY_SESSIONS": ["<a session with both>"],
```

This is an efficacy failure, not a safety one — a grass weed called onion never
gets treated — but it is a *silent* one: it will not show up in
`missed_onion_fraction` or `weed_on_crop_fraction`, because a weeds-only
session has no onion ground truth to measure against.

### Small weeds

18% never found at any threshold. Cotyledon-stage plants are a few hundred
pixels. The resolution lever is spent (1777×1000 already); the remaining
levers are more small-weed examples, and eventually **tiling** — training on
crops of the frame at native resolution rather than the whole downscaled frame.
Tiling is a bigger change and is not built; revisit it if small-weed recall is
still the binding constraint after another 200 frames.

---

## 5. Semi-supervised: teacher–student

The literature is real — STAC, Unbiased Teacher, Soft Teacher all report solid
gains on COCO with 1–10% labels, which is roughly your regime. A teacher
(EMA weights) pseudo-labels unlabelled frames; a student trains on labels plus
confident pseudo-labels under strong augmentation.

**RF-DETR already gives you the teacher for free** — `USE_EMA: True` maintains
exactly those averaged weights.

### Why it is not wired up, and the condition for changing that

A pseudo-label is the model's current belief taught back to itself. For weeds
that is mostly harmless: a wrong weed label costs a wasted laser pulse, and the
errors are diverse enough to average out.

**For the crop it is not.** The model already calls grass `onion_plant` with
0.91 confidence. Feed that back as training data and you teach it to be
confidently wrong about the crop — and every metric on the page *improves*,
because the model now agrees with itself. A crop-safety failure that makes
validation look better is the worst failure mode this project can have.

So if you do this, the rule is:

> **Pseudo-label weeds. Never pseudo-label the crop.**
> Onion masks come only from a human.

That is implementable — pseudo-labels for weed classes, `ignore_region` for
everything the teacher is unsure about, and real annotations for onion — but it
is a substantial piece of work whose benefit is measured *against the held-out
session you do not have yet*. Ordering matters: **§0, then two or three rounds
of §1, and only then this.** Ask and I will build it.

### The cheap 80%

`mine_pool.py` already gives you most of the practical benefit — the model's
predictions do the bulk of the labelling work — with a human on every frame, so
no error can be laundered into truth. That is the right trade at this dataset
size.

---

## 6. Techniques that are not worth it yet

| Technique | Verdict |
|---|---|
| **Synthetic / copy-paste** | Deliberately absent from every augmentation preset, and it stays absent. Pasting an onion between frames fabricates crop geometry no field produced, and this is a crop-**safety** model. The same argument keeps Mosaic and MixUp out. |
| **SAM-refined boundaries** | SAM 3 could tighten human polygons. But `onion_boundary_f = 0.78` and mask IoU is not what limits you — *recall* is. Better boundaries on plants you already find buys almost nothing. |
| **Self-training on the crop** | See §5. Not a scheduling question — a correctness one. |
| **Tiling for small objects** | Genuinely promising, genuinely a bigger change. Revisit when small-weed recall is still binding after another 200 frames. |
| **More augmentation** | `strong` exists and is one config value away. Cheap to try, but it does not add information — it only reuses what you have. |

---

## 7. Annotation quality

The one failure that survives every model change is **inconsistent ground
truth**, and it is invisible in the metrics: the model learns the
inconsistency, and validation — annotated the same inconsistent way — rewards
it.

Four rules worth writing down before the next few hundred frames:

1. **One decision per ambiguous case, recorded.** Does a two-leaf seedling
   count? Does a weed half out of frame? Write the answer down the first time
   and follow it. A rule you can state is better than a better rule you cannot
   remember.
2. **`ignore_region` is the escape hatch.** Genuinely undecidable areas —
   motion blur, deep shadow, a tangle you cannot separate — get marked ignored,
   not guessed. A guess becomes a wrong gradient; an ignore becomes nothing.
3. **Onion boundaries decide crop safety.** A sloppy onion mask directly moves
   `missed_onion_fraction`, and you cannot tell a model error from a labelling
   error afterwards. Spend the extra seconds on the crop.
4. **Re-annotate 10 frames from an early batch, blind, after 200 more frames.**
   Compare. That measures your own drift, and it is the only way to find out
   whether your standard moved.

Re-annotating an already-labelled frame is worse than wasted effort — two
versions of the truth and nothing downstream knows which to believe — so
`mine_pool.py` excludes labelled `item_id`s from the pool automatically.

---

## The order

1. **§0** — hold out a session, annotate ~40 frames of it. Nothing else is
   measurable until this exists.
2. Decide the `other_weed` split (§4) — an ontology change is cheapest *before*
   the next few hundred frames.
3. Round 1: `mine_pool.py` → 60 frames → correct → retrain → `analyze_run.py
   --split test`.
4. Round 2, targeting **mixed onion+grass scenes** specifically.
5. Round 3, whatever the test-set numbers now say is worst.
6. Re-read §5 with a real test number in hand.
