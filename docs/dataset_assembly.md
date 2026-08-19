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
| [2b. No session to spare](#when-you-have-no-session-to-spare) | How do I get a representative test set from all sessions? |
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

### When you have no session to spare

`SPLIT_MODE = "frame_block"` splits *within* every session instead, so the test
set is drawn from all of your data — every session, every scene, in proportion.

```python
"SPLIT_MODE": "frame_block",
"BLOCKS_PER_SESSION": 4,
"GAP_FRAMES": 8,
"VAL_FRACTION":  0.15,
"TEST_FRACTION": 0.15,
```

**Be clear about what this buys and what it does not.** It gives you a test set
that is *representative of the data you have*. It cannot tell you the model
works on a **new drive**, because every test frame shares its session's light,
soil, growth stage and field with training frames. Read the resulting score as
an **upper bound** on field performance, not an estimate of it. A held-out
session remains strictly better and nothing here replaces it.

Three things make it as honest as it can be:

**Blocks stay contiguous.** Never a random frame split — adjacent frames are
near-identical, so a shuffle puts a frame and its own near-duplicate on both
sides of the boundary.

**Several blocks per session.** With one block, test is always the *last*
stretch of every recording, which on this data is systematically different:
later in the day, further along the bed, often the headland where the rig
turns. A test set made only of drive-ends measures drive-ends.
`BLOCKS_PER_SESSION = 4` samples from four places along each drive. The request
is a ceiling — a session too short to hold that many blocks gets fewer, rather
than sending every frame to train and leaving val empty.

**The separation is measured, not assumed.** `GAP_FRAMES` is counted in *pool*
frames, and the pool is usually strided — so `GAP_FRAMES = 2` after a stride-5
curation buys 10 video frames, about a third of a second. Whether that is
separation or self-deception is invisible in the config, so the build measures
the real distance and prints it:

```
      Nearest TRAIN frame, in VIDEO frames:
        val   vid1_20260108_101500              40
        test  vid1_20260108_101500              80
        test  vid2_20260108_122731              40
  [!] 1 split/session pair(s) sit closer than 60 video frames to training data.
```

Raise `GAP_FRAMES` until that warning clears. The cost is a handful of
annotated frames — far cheaper than a test score wrong in the optimistic
direction. A seam exists at every block boundary *and* between one block's test
tail and the next block's train head; both are buffered.

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

**No session to spare — representative test set from everything:**

```python
"SPLIT_MODE": "frame_block",
"BLOCKS_PER_SESSION": 4,
"GAP_FRAMES": 8,              # raise until the separation warning clears
"VAL_FRACTION":  0.15,
"TEST_FRACTION": 0.15,
"HOLDOUT_TEST_SESSIONS": [],  # nothing pinned; every session contributes
```

Read the score as an upper bound. Record a fresh session when you can and
rebuild with `HOLDOUT_TEST_SESSIONS` — that is the only configuration that can
report generalisation.

**First real dataset, three scene types, one pinned test session:**

```python
"HOLDOUT_TEST_SESSIONS": ["vid3_20260108_110444"],
"VAL_FRACTION":  0.2,
"TEST_FRACTION": 0.0,          # the pinned session IS the test set
"STRATIFY_BY_SCENE": True,
"DROP_CLASSES": ["wild_radish", "weed_cluster"],   # too few instances yet
```

**Every onion-only session, as a crop-only detector:**

```python
"SOURCES": [ ... one entry per onion CVAT export ... ],
"KEEP_CLASSES": ["onion_plant"],   # not DROP_CLASSES: see below
"SPLIT_MODE": "frame_block",       # unless you can pin an onion session
"BLOCKS_PER_SESSION": 4,
"STRATIFY_BY_SCENE": True,         # a no-op when every session is onion_only
```

`KEEP_CLASSES` names what the build **is**; `DROP_CLASSES` names what it lacks.
For a crop-only build the two pick the same classes today and diverge the moment
a class is appended to the ontology — which is
[the documented way it grows](../seeweed3d/common/ontology.py). A deny-list
listing the five current weeds would silently admit the sixth into a dataset
that called itself onion-only; the allow-list keeps meaning what it said.
`class_mapping.json` records which spelling was used, so a later reader can tell
*onion-only by intent* from *these happened to be dropped that day*. Both may be
combined — `DROP_CLASSES` applies on top, which narrows an allow-list without
rewriting it.

> **What a crop-only model can and cannot tell you.** Trained on onion-only
> scenes it has never seen a weed, so it has not learned *not everything green
> is an onion* and will over-fire on weeds in a mixed field. That is the
> expected result, not a bug: this build measures **crop detection**, and its
> value is as the onion half of the complement approach
> (`prelabel_complement_sam3.py`, where "not onion" becomes the weed proposal)
> and as a baseline the mixed-scene model must beat on onions. Do not read its
> AP as field performance on a mixed bed.

### How the splits are actually made

Three decisions, each of which was a defect before it was fixed.

**What unit is held out** — `SPLIT_GRANULARITY`. `group` holds out a whole
date+field+camera, the only split that estimates generalisation. `session`
holds out a whole recording from a date training also saw: no frame leaks, but
the conditions are shared, so the score is an upper bound. `auto` tries `group`,
falls back, and **says which it used** — the manifest records it, because a run
against one is not comparable with a run against the other.

Six sessions recorded on two days are only **two** groups, and two indivisible
units cannot fill three splits: test comes out empty and training sees one date.

**How much lands in each split** — best fit, not first fit. The obvious loop
("add this group if the split is still under quota") checks the quota *before*
adding and overshoots by the whole size of the group. Asking for 20% test from
groups of 1, 2 and 3 sessions put all three of the last group in test — 50% of
the data, with its entire capture date absent from training. A group is now
taken only if it *fits*; when nothing fits, the split takes the **smallest**
remaining group rather than the first one it happens to see. Sizes are counted
in **frames**, not sessions: a 500-frame and a 50-frame session are not
interchangeable, and counting sessions makes a 20% split mean anything between
5% and 60% of the data.

**Where along each drive the splits sit** — the block layout rotates.
`[train | gap | val | gap | test]` used to be fixed, so val was the middle of
every stretch of every session and test was the end of every one. A test set
made entirely of drive-ends measures drive-ends: later in the pass, further
along the bed, often the headland where the rig turns. The order now rotates
deterministically per session and per block, so test is drawn from beginnings,
middles and ends.

> This is **not** a random frame shuffle. Adjacent video frames are
> near-duplicates; shuffling frames puts a frame and its own near-duplicate on
> opposite sides of the split boundary, and the score then measures
> memorisation. Blocks stay contiguous — only their *order* rotates.

**The fractions describe what is built, not what was asked for.** The buffers
come off the block first; the fractions then apply to what survives. Sizing val
and test from the raw block and letting train absorb every buffered frame turned
a requested 70/15/15 into **56/22/22** on real data — the shortfall landing
entirely on the split that needed the frames most.

**What the gaps cost, and which ones are free.** A buffer is charged at every
seam where the split changes. With the layout rotated, two chunks often meet at
the *same* split, and dropping frames to separate train from train buys nothing
— that seam is now free. The internal seams within a block are all real and are
still charged.

The remaining cost is unavoidable and worth knowing: at `GAP_FRAMES: 12` and
`BLOCKS_PER_SESSION: 3`, roughly **a quarter to a third of frames are dropped**
as buffers. `GAP_FRAMES` is set by the separation floor — 12 pool frames at a
stride of 5 is 60 video frames, exactly `MIN_SEAM_SEPARATION` — so it cannot be
lowered without the build's own separation warning firing. More blocks means a
more representative test set and more seams to pay for; three is the balance,
and the build prints what it actually dropped.

**Merging a new active-learning round into an existing dataset:** add its
export to `SOURCES` and rebuild. The split is deterministic for a seed, so
sessions that were in test stay in test — as long as you do not change `SEED`.
Changing the seed re-draws every split and invalidates comparison with every
earlier run.

**One session only:** you will get frame blocks and a warning. That is correct
and there is nothing to fix except recording another session.
