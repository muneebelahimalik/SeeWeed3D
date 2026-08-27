# Is the system ready?

> The whole pipeline, what each stage needs from the last, and what goes wrong
> when it doesn't have it.
>
> Run the check: `python -m seeweed3d.perception.preflight`

Every failure described here **produces output that looks completely ordinary**.
Nothing raises. That is why the check exists and why each finding carries its
consequence rather than just its name.

---

## The pipeline, and what each stage needs

| Stage | Needs | Without it |
|---|---|---|
| **A — segment** | a checkpoint **and its recorded class list** | labels map through the full ontology; every plant is confidently mislabelled |
| **crop safety** | an `onion_plant` class in that model | **every candidate is rejected** — see below |
| **B — LEP** | a LEP checkpoint, or the fallback | the hand-engineered estimator runs; it is the baseline, not the deployed stage |
| **3D** | `depth/` **and** `meta/calibration.json` | every target returns `xyz_mm: null` — nothing to aim at |
| **decide** | thresholds in `SafetyConfig` | — |

The order matters: crop safety is decided for the **whole frame** before any
weed is considered, because deciding it per weed would let an onion found late
in the loop fail to protect a weed decided early.

---

## The one that will surprise you

**A weed-only model cannot produce a single candidate.**

`SafetyConfig.allow_missing_crop_mask` defaults to `False`. A model with no
`onion_plant` class returns no crop mask at all, and the decision function
refuses to read that as "there is no crop here". So every target abstains, the
run completes, and the JSON reports zero candidates — which is *exactly* what a
clean frame with no weeds looks like.

That default is deliberate. "This model cannot see onions" must never be read as
"there are no onions here", because the difference is a laser and a crop.

**The fix is the mixed build**, not the flag:

```
python -m seeweed3d.training.datasets.mixed
python -m seeweed3d.training.datasets.mixed_train
```

Setting `allow_missing_crop_mask = True` is a claim *about the field* — that
there is no crop in front of the camera — and only whoever is standing in it can
make that claim. It is recorded in every decision either way.

---

## The three builds

| Runner | Classes | Provenance | What it is for |
|---|---|---|---|
| `datasets/weeds.py` | weeds only | `hand_corrected` | Stage A quality, and the self-training loop |
| `datasets/onions.py` | `onion_plant` | `prelabel_unreviewed` | the crop class, before anyone has corrected one |
| `datasets/mixed.py` | **all six** | `mixed` | **the deployable model** |

They are separate runners on purpose. One shared CONFIG edited back and forth
between an onion build and a weed build is how a stale `DATASET_DIR` reaches a
training run — which has happened twice.

`mixed.py` imports its sources from the other two rather than repeating them, so
a session added to the weed build reaches the mixed build without a second edit.

### The labels are not all the same kind

The weed sessions are hand corrected. The onion sessions are SAM 3 prelabels
nobody opened. Merging them doesn't average into something in between — it makes
a dataset where the weed classes are *measured* and the crop class is *agreement
with a prelabeler*. `LABEL_PROVENANCE = "mixed"` is what keeps that recoverable
six months from now.

It matters asymmetrically. An unreviewed weed mask costs a slightly wrong
boundary. An unreviewed **crop** mask decides whether the laser fires. The crop
class is the one whose labels most deserve a human and currently has had the
least — correct crop frames *before* trusting a crop-safety number.

### What to drop, and why not to copy the list

`weeds.py` drops `weed_cluster` (2 instances) and `wild_radish` (0). Those counts
are properties of *that* build. Merging sessions changes them, so `mixed.py`
starts with `DROP_CLASSES = []` and the build's own class report decides.

---

## Running the whole thing

```
python -m seeweed3d.perception.run_full
```

Preflight runs first, before the GPU. `STOP_ON_BLOCKING = True` ends the run on
a blocking finding; set it `False` when you are deliberately looking at a model
you already know is incomplete.

Per frame you get an overlay and a record:

| field | meaning |
|---|---|
| `class_name` / `score` | Stage A |
| `lep_uv` | growth point, full-frame pixels |
| `xyz_mm` | that point in the camera frame, millimetres |
| `xyz_sigma_mm` | how sure — the delta robot's tolerance lives here |
| `safety_status` | `candidate` or `abstain` |
| `rejection_reasons` | **why** it abstained, one string per failed threshold |

**Read `rejection_reasons` first on a disappointing run.** Zero candidates with
reasons dominated by one code is a *threshold* problem. Zero candidates with no
targets at all is a *Stage A* problem. They are fixed in completely different
places, and the top-line number looks identical.

---

## Duplicate detections

RF-DETR is a set-prediction model: every query proposes independently, and
nothing makes two queries that found the same plant agree about what it is. On a
real weed session **14% of detections were duplicates** at IoU ≥ 0.85, observed
at box IoU 1.000 in 6 of 16 frames.

The pipeline suppresses them before anything reads the detections — including
the crop-safety mask. Order is the whole point: a duplicate labelled `onion` puts
those pixels in the safety mask while its twin sits on the target list, so the
same plant is both protected and fired at.

`PipelineConfig.dedup_iou = 0` disables it, and preflight says so.

---

## What a score means

Two facts change how every number should be read, and both are reported:

- **No holdout test session.** `val` is contiguous blocks *inside* the training
  sessions, so it shares their light, soil, growth stage and often the individual
  plants. Every score is an upper bound and no round is comparable with the last.
  Cut one drive out and correct 20–30 frames from it.
- **Machine labels.** Any provenance other than `hand_corrected` means the score
  measures agreement with a prelabeler, or with the model's own earlier output —
  not correctness.

When adding the crop class, **watch weed recall, not mean AP.** Onions are large,
regular and easy, so mAP rises almost regardless of what happens to the weeds. A
model that got better at onions while getting worse at small weeds is a worse
weeder with a better number. `eval_seg`'s recall-by-size is the column that
answers it.

---

**See also:** [runbook](RUNBOOK.md) ·
[self-training loop](self_training.md) ·
[dataset assembly + splits](dataset_assembly.md) ·
[LEP localization](lep_localization_explained.md) ·
[training](training.md)
