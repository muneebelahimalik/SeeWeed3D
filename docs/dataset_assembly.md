# Building the training dataset from many CVAT tasks

How several CVAT exports become one dataset, and how that dataset is split so
the numbers it produces mean what they appear to mean.

**Related:** [runbook](RUNBOOK.md) · [growing the dataset](dataset_growth.md) ·
[mixed-scene prelabeling](mixed_prelabeling.md)

---

## Contents

| Section | Question it answers |
|---|---|
| [1. Merging tasks](#1-merging-many-cvat-tasks) | How do several exports become one dataset? |
| [2. Why not a random split](#2-why-the-split-is-not-random) | Why can't I just shuffle the frames? |
| [3. Scene stratification](#3-scene-stratification) | How do onion-only, weed-only and mixed stay represented? |
| [4. The pinned test set](#4-the-pinned-test-set) | How do I get a number I can compare across rounds? |
| [5. Reading the output](#5-reading-what-it-prints) | What do I check before training on it? |
| [6. Recipes](#6-recipes) | What do I actually type? |

---

## 1. Merging many CVAT tasks

One session per CVAT task is good practice: tasks stay small, and a session is
the unit a split has to respect anyway. `make_dataset.py` merges any number of
them.

```python
"SOURCES": [
    {"DATUMARO_ROOT": r"E:\exports\onions_vid3_132749",
     "IMAGES_ROOT":   r"E:\Dataset_Vidalia\onions_20260108_1\sessions"},
    {"DATUMARO_ROOT": r"E:\exports\weeds_vid2_122731",
     "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Weeds_3_good\sessions"},
    {"DATUMARO_ROOT": r"E:\exports\mixed_vid1_101500",
     "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Mixed_1\sessions"},
],
```

Sources may share an `IMAGES_ROOT` or point at entirely different folders. A
`DATUMARO_ROOT` may also be one *parent* folder holding several unzipped
exports — they are all found and merged.

**Merging is keyed on the label NAME, never on `label_id`.** Each export is
resolved through its own `categories` block, because `label_id` 2 can
legitimately mean different classes in two tasks. A merge keyed on the id would
silently relabel half the dataset, and nothing downstream would notice.

**Images are never copied.** Manifests reference the files where they already
live, so a 40 GB dataset is not duplicated and the same export can be rebuilt
into several datasets cheaply.

> **Export `Datumaro 1.0`, not COCO.** COCO cannot carry shape groups, so a
> COCO export silently discards every mask-to-LEP link. `make_dataset.py`
> sniffs the file and tells you if you pointed it at COCO.

### Only the frames you actually corrected

A task pre-loaded with SAM 3 prelabels has annotations on every frame, so "has
annotations" does not mean "verified". Run once with `LIST_FRAMES = True`, note
the positions you corrected, and put them in `INCLUDE_FRAMES`:

```python
"INCLUDE_FRAMES": "vid2_20260108_122731:1-27,vid2_20260108_122731:51-60,"
                  "vid3_20260108_132749:1-36,vid1_20260108_101500:*",
```

Positions restart at 1 **in each session**, so a second export can never
renumber the first and quietly redirect a carefully-checked selection.

---

## 2. Why the split is not random

**Adjacent video frames are near-identical.** A random frame split puts a frame
and its neighbour on opposite sides of the train/test boundary, so the test set
measures memorisation and reports an accuracy the field will not reproduce. At
30 fps and walking pace, frames 400 and 401 are the same plants from 3 cm away.

So the unit is the **whole session**, and the allocator enforces it rather than
leaving it as a convention someone has to remember.

Sessions that are *near-duplicates of each other* are kept together too: same
date, same field, same camera. Two drives of the same bed on the same morning
leak exactly as a frame split does. That metadata comes from each session's own
`meta/session.json`, written by `extract_sessions.py`.

**When a session split is impossible** — one session, or a class living only in
a held-out session — the build falls back to contiguous **frame blocks** with a
discarded gap at each boundary, and says loudly that it did. Blocks put train
and val in different parts of the drive, which is far weaker than different
drives but much better than shuffling. Treat those scores as a sanity check
that training works, not as evidence of generalisation.

### Shuffling

Order within a split does not matter and is not what makes a split good — the
training loader shuffles every epoch. What matters is *which sessions* land
where, and that is what the allocator decides. A dataset that was "properly
shuffled" but split by frame is still measuring memorisation.

---

## 3. Scene stratification

Each session carries a `scene_hint` — `onion_only`, `weed_only`, `mixed` or
`unknown` — set per input folder in `extract_sessions.py`:

```python
INPUT_ROOTS = [
    {"path": r"...\just_onions_jan_2026", "trip": "Visit1",
     "site": "vidalia_1", "field": "field_A",
     "scene_hint": "onion_only", "notes": "..."},
]
```

`STRATIFY_BY_SCENE` (default **on**) allocates sessions separately *within*
each scene, so train, val and test each get their share of all three.

**What it prevents.** Without it the allocator is blind to what a session
contains. Measured on three onion-only and three mixed sessions across 40
seeds, validation ended up holding a single scene on **10 of them** — a
one-in-four chance of a val set that never once exercises the crop-vs-weed
decision. With stratification, zero. Nothing in any printed metric would have
told you which case you were in.

What each scene is the only one that can measure:

| Scene | What only it measures |
|---|---|
| `mixed` | the crop-vs-weed decision — the one that decides whether the laser fires on a weed or on the crop |
| `weed_only` | weed recall in isolation, with no crop to confuse it |
| `onion_only` | crop segmentation, and the false-positive rate on a frame with no weeds in it |

Sessions whose scene is `unknown` form their own stratum — they carry no
evidence about what they contain, so they are not assumed to match anything.
The build prints which sessions those are, because an unrecognised
`scene_hint` (`"onions"` instead of `"onion_only"`, say) removes that session
from stratification silently.

A group spanning several scenes is labelled `mixed`: the group is indivisible,
so whichever split receives it receives every scene in it, and calling it
`onion_only` would overstate what that split holds.

Set `STRATIFY_BY_SCENE = False` to get the previous, scene-blind behaviour.

---

## 4. The pinned test set

```python
"HOLDOUT_TEST_SESSIONS": ["vid3_20260108_110444"],
"HOLDOUT_VAL_SESSIONS":  [],
```

A pinned session goes to that split on **every rebuild**. A `TEST_FRACTION`
does not: it re-draws the test set as the dataset grows, so round 3's score and
round 4's score are computed on different rulers and no improvement can be
attributed to anything.

Pin the same session in `mine_pool.py`'s `HOLDOUT_SESSIONS`, which is what
stops active learning from annotating it into training. See
[dataset_growth.md](dataset_growth.md#0-do-this-first-a-held-out-session) —
including why mining the test set is the most flattering possible way to
destroy it.

Holdouts are honoured **before** any fraction and win unconditionally. They
guarantee *placement*, not exclusivity — the quota may put other sessions in
test alongside a pinned one.

---

## 5. Reading what it prints

```
  Scene representation (sessions per split):
      train onion_only=3, weed_only=2, mixed=2
      val   onion_only=1, mixed=1
      test  mixed=1
  [!] val contains no weed_only session - weed recall is never measured there.
```

That warning is not an error. With few sessions it can be unavoidable. What it
does is bound what the number means, and that bound is invisible in any metric
the training run will print.

Other things worth checking before you train:

| Output | Check |
|---|---|
| `splits/splits_summary.json` | per-split session list, frame counts, class counts |
| `dataset_report.json` | per-class instance counts — a class under ~20 instances is not learnable and will drag the mean |
| `annotations_needing_correction.json` | contract violations; a strict build refuses to write while any remain |
| the `SESSION-LEVEL SPLIT NOT USED` banner | you got frame blocks, so val shares each session's light, soil and often its individual plants |

---

## 6. Recipes

**First real dataset, three scene types, one pinned test session:**

```python
"HOLDOUT_TEST_SESSIONS": ["vid3_20260108_110444"],
"VAL_FRACTION":  0.2,
"TEST_FRACTION": 0.0,          # the pinned session IS the test set
"STRATIFY_BY_SCENE": True,
"DROP_CLASSES": ["wild_radish", "weed_cluster"],   # too few instances yet
```

**Merging a new active-learning round into an existing dataset:** add its
export to `SOURCES` and rebuild. The split is deterministic for a seed, so
sessions that were in test stay in test — as long as you do not change `SEED`.
Changing the seed re-draws every split and invalidates comparison with every
earlier run.

**One session only:** you will get frame blocks and a warning. That is correct
and there is nothing to fix except recording another session.
