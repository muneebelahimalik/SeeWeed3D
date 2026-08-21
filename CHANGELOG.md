# SeeWeed3D — project history

What changed, when, and **why**. Every entry names the problem that forced the
change, because the reason is the part that stops it being undone later.

Numbers in parentheses are pull requests. The pipeline stages referenced here
are the ones in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

**Reading this as a newcomer:** the [Lessons](#lessons-this-project-paid-for)
section at the end is the short version. Several settings in this repo look
arbitrary or over-cautious until you know which failure produced them.

---

## Contents

| Phase | What it established |
|---|---|
| [1. Capture and extraction](#1-capture-and-extraction) | recordings that can still be re-processed a year later |
| [2. SAM 3 prelabeling](#2-sam-3-prelabeling) | CVAT as correction rather than annotation from scratch |
| [3. Ontology and the CVAT round trip](#3-ontology-and-the-cvat-round-trip) | one spelling of every class, everywhere |
| [4. Boundary quality, and reverting it](#4-boundary-quality-and-reverting-it) | the field is the arbiter, not the metric |
| [5. Pool curation](#5-pool-curation) | annotation effort spent on new ground |
| [6. Perception baseline](#6-perception-baseline) | segmentation → LEP → 3D → safety, end to end |
| [7. Dataset building](#7-dataset-building) | many CVAT tasks → one trainable dataset, safely |
| [8. Experiment tracking](#8-experiment-tracking) | runs that can be compared to each other |
| [9. Training: the resolution era](#9-training-the-resolution-era) | small weeds stopped being thrown away |
| [10. RF-DETR-Seg backend](#10-rf-detr-seg-backend) | a second, real-time backend, correctly wired |
| [11. Evaluation that measures the right thing](#11-evaluation-that-measures-the-right-thing) | crop burn, and the operating point as a choice |
| [12. Analytics for both backends](#12-analytics-for-both-backends) | one command, all the figures |
| [13. Growing the dataset](#13-growing-the-dataset) | active learning, reachable at last |
| [14. Loss shaping](#14-loss-shaping) | Dice's symmetry broken deliberately |
| [15. Mixed scenes](#15-mixed-scenes) | onions and weeds in one frame |
| [16. The Feb 2025 visit](#16-the-feb-2025-visit-and-depth-that-never-existed) | depth that never existed, and how it was proven |
| [17. The mixed-scene dataset strategy](#17-the-mixed-scene-dataset-strategy) | identity is the bottleneck, and laziness ships |

---

## 1. Capture and extraction

**The v1 → v2 capture rewrite** (see [`docs/capture_changelog.md`](docs/capture_changelog.md)
for the full table). Every change existed because something in the v1 recordings
could not be recovered afterwards: only the left view was saved, so stereo could
never be re-processed; exposure tracked the sun with nothing logged, so
appearance drifted with the weather rather than the plants; frame drops were
silent, so video time stopped matching real time. v2 records an SVO2 archive,
the confidence map, pose and IMU, three clocks per frame, and a full calibration
dump.

**Extraction hardening.** `INPUT_ROOTS`/`OUTPUT_ROOT` promoted to documented
config blocks supporting multiple input directories with v1 and v2 formats
mixed; `ffmpeg`/`ffprobe` resolved from `PATH` cross-platform.

**Package layout** (#2). History preserved through git renames; everything
grouped by pipeline stage under `seeweed3d/`.

> **Depth is raw uint16 millimetres, `0` = invalid.** Never rescaled. A
> `depth_vis_max_mm` field in v1 metadata was being read as a data scale and
> corrupting depth; it was removed and the encoding stated explicitly.

---

## 2. SAM 3 prelabeling

The goal throughout: **an annotator corrects masks instead of drawing them.**

**Onion prelabeling** (#1, #3). One high-recall semantic safety mask per frame
for onion-only scenes, exported as CVAT-importable COCO. Moved from the
`transformers` backend to Meta's official `sam3` package.

**Three defects found by running it on real frames:**

- (#4) SAM 3's vision backbone mixes bf16 activations with fp32 weights, raising
  a dtype error outside an autocast context. Inference now runs under
  `torch.autocast`.
- (#5) Real SAM output tripped an OpenCV resize assertion. Mask shapes are
  normalised from whatever nesting SAM returns into 2-D bool masks.
- (#6) **SAM text prompts returned ~0 masks on top-down field imagery**, so the
  output was the Excess-Green prior alone — which masked whole frames where the
  soil had a green cast. ExG is now gated by green dominance and saturation.

**Exemplar prompting** (#7). Since the text concept "onion" does not ground on
this imagery, boxes derived from vegetation blobs are fed to SAM 3 as positive
exemplars instead. This is still the default prompt mode everywhere.

**Colour-cast recovery** (#8). Some ZED frames have a severe green white-balance
error and were being flagged and lost. The structure is intact underneath, so a
gray-world white balance recovers them.

**Weed instance prelabeling** (#11). The opposite problem from onions: every
plant separated, classified by morphology, with a proposed LEP growth point.

**Multi-evidence LEP estimator** (#14). The LEP is the research target, so it
has to be defensible as the apical meristem rather than a shape summary that
happens to correlate with it — a centroid or bbox centre drifts off the meristem
on an asymmetric plant with no way to notice. Petiole convergence, radial
isotropy, young-tissue chromatics, canopy height and medial-axis interiority.

**Progress reporting** (#18) — long runs gave no feedback until a session
finished.

---

## 3. Ontology and the CVAT round trip

**One shared ontology** (#15). Class names had been defined per-module and had
drifted — generic `brassica`, `primrose`, `grass`, and names with spaces in
them. `common/ontology.py` became the single source of truth, with **stable
category ids** so weed, onion and future mixed datasets merge without remapping
a single annotation.

**What shape can and cannot decide** (#13). The prelabeler proposes only what
morphology supports: grass by elongation, clusters by multiple growth points.
**Species are never auto-assigned** — `cutleaf_evening_primrose` vs
`wild_radish` is an appearance question and both are rosettes.

**The CVAT round trip** (#9). Verified export → rasterised training masks →
auto-vs-verified IoU, so prelabel quality is measured rather than assumed.

**Three separate schema failures**, all of which fail *silently* in CVAT:

- (#19) CVAT's Raw editor requires a non-empty `values` array for **every**
  attribute, including text ones. Two shipped without it, and CVAT rejected the
  whole schema.
- (#39) The onion schema used type `mask`, which CVAT exports as RLE needing
  pycocotools. Changed to `polygon` to match the working weed round trip.
- (#40) A prelabeling run from *before* the snake_case rename still wrote
  `"onion plant"` as a COCO category name. CVAT matches by name, so this
  silently creates a **duplicate label** instead of filling the one the
  prelabels were meant to correct. `fix_coco_categories.py` repairs it and
  refuses to write rather than guess at an unknown name.

**Refresh without re-running SAM 3** (#20). The label schema is pure data
derived from the ontology, so a rename should not cost a multi-hour inference
pass. `regen_cvat_labels.py` rebuilds it in seconds.

**Flagged frames separated from the upload set** (#10) — a blank-mask frame
mixed into the COCO could mismatch the CVAT upload.

**`instances.csv` schema fix** (#17). Rows do not all share a key set — a
`weed_cluster` carries no `lep_*` columns — so taking the header from row 0
crashed as soon as the first instance was a cluster.

---

## 4. Boundary quality, and reverting it

This sequence is worth reading in order. It is the clearest case in the project
of a metric-driven improvement being wrong.

- (#16) `MIN_INSTANCE_AREA_PX` had been raised to 700 on the reading that dense
  frames showed speck over-detection. **The imagery showed those were real
  cotyledon-stage weeds.** Reverted to 250.
- (#21) Edge snapping, plant splitting and full multi-part polygons, on the
  reasoning that prelabel boundary quality becomes the training target's
  quality.
- (#22) A recall backstop: unclaimed vegetation becomes an instance, because a
  missed weed is invisible and enters the training target as background.
- (#23) Anti-aliasing, since per-pixel edge decisions leave staircase jaggies
  that no real leaf margin has — and the skeleton-based LEP evidence is
  sensitive to exactly that noise.
- (#24) Real runs showed **dozens of phantom detections on bare, pale, mottled
  ground.** The vegetation prior is a colour index and colour indices
  false-positive on green-tinted mineral. Confidence gating added.
- (#25) Still not clean. Reverted the exemplar loosening entirely and **disabled
  the recall backstop by default.** A single low-quality exemplar does not risk
  one bad box — SAM hunts the whole frame for more of "that concept".
- (#29) **Restored the PR #11/#12 mask profile wholesale.** Field comparison
  judged those masks best, with a specific observed failure in the newer build:
  one weed exported as several instances. `split_touching_instances` was the
  cause — a leaf reaching away from the crown raises a second
  distance-transform peak, and that was enough to split one plant.

The post-#12 code is all still present and still tested; it is simply off by
default, each entry independently re-enablable.

---

## 5. Pool curation

**`curate_pool.py`** (#26). Driving slowly produces consecutive frames of almost
the same ground. Doing this by hand means deleting from `rgb/`, then `depth/`,
then `conf/`, then fixing whatever references the names — so curation **edits
`pool.csv` only** and never touches an image file. `RESTORE_ALL` undoes
everything.

**Jitter was accumulating as travel** (#27). `mark_redundant` summed the
*magnitude* of each phase-correlation step. Magnitudes are strictly positive, so
a stationary camera's jitter integrated into apparent forward motion.

**The threshold is a choice, not a guess** (#28). The first dry runs showed a
near-flat ~50% drop rate, which the hint text mislabelled as "too aggressive". A
flat histogram means the camera moved at a steady speed, so the threshold alone
sets the sampling rate. It now prints a sweep, and you pick from the *overlap
between kept frames* column rather than the drop percentage.

---

## 6. Perception baseline

**Supervised perception baseline** (#30). Segmentation → learned LEP → 3D →
safety abstention, end to end. Nothing trained: no accuracy, latency or 3D
number appeared anywhere, because no verified annotations existed yet.

**Pluggable segmentation backends** behind one framework-free `Detections`
structure. That boundary exists for **licensing**, not tidiness: the default is
permissively licensed (torchvision, BSD-3) so nothing in the normal path can
create an AGPL obligation by accident.

**Two crop-safety fixes, both silent failures:**

- (#32) `Detections` resolved the crop class against the *full* ontology while a
  `--drop-classes` checkpoint emits indices into a *reduced* set. Dropping any
  class below `onion_plant` made the safety mask empty and handed every onion to
  the targeting stage. Class indices now resolve through the model's own list,
  recorded in the checkpoint.
- (#37) A missing crop mask and an empty one were both treated as "no conflict".
  An empty mask means the segmenter looked and found nothing; `None` means it
  has no crop class and **cannot look**. A model trained from an export with no
  onion instances produces the latter, and it cleared every shot in a crop row.

---

## 7. Dataset building

**From a 35-frame CVAT subset** (#31). `--drop-classes` removes a class from a
single build without editing the ontology, remapping indices contiguously and
recording the active set.

**Frame selection** (#34). A CVAT task pre-loaded with SAM proposals has
annotations on *every* frame, so frames the annotator never reached are not
empty and no existing filter removed them. `--include-frames` /
`--exclude-frames` applied before duplicate detection.

**Positions are per session** (#38) — resolved against the merged order, adding
a second export renumbered the first and redirected a checked selection at
different frames.

**Config-block runners** (#35) so no stage needs a long command line, matching
the rest of the pipeline.

**Frame resolution through the canonical layout** (#36). The manifest stores a
bare filename because CVAT uploads are flat, and the old fallback searched the
whole dataset root. **The depth PNG carries the same filename as its RGB frame**,
so that search could return a depth map to be used as a training image — on
every `__getitem__`.

**Multi-root merging** (#41). A weed capture campaign and a separately-recorded
onion campaign are not generally stored under one parent.

**Two builds that silently discarded data:**

- (#42) A repeated `INCLUDE_FRAMES` key in a CONFIG dict meant Python kept only
  the last, so an entire onion export contributed zero frames and **a full
  training run completed without ever seeing the crop class.**
- (#45) Two sessions at `val_fraction 0.2` rounds to zero val sessions, so the
  build produced no validation set and training saved no checkpoint. Raising the
  fraction instead put every weed in train and every onion in val, since the two
  recordings partition the ontology perfectly.

**Split summary counted sessions, not frames** (#46) — it printed `val=0` while
sixteen validation frames existed.

---

## 8. Experiment tracking

**Local-only tracking** (#33) over TensorBoard and MLflow, both Apache-2.0.
Local-only is a requirement, not a preference: **field imagery and a customer's
row geometry must never be uploaded to a vendor cloud.** That is why W&B and
Comet were evaluated and rejected. Plus side-by-side GT/prediction overlays on a
fixed sample of val frames.

**MLflow 3 refuses the plain file store** (#44), and `_start_mlflow` only caught
`ImportError`, so that exception would have killed a training run. Now
`sqlite:///<runs>/mlruns/mlflow.db` with an explicit artifact location, and a
broken backend disables itself instead of raising.

**Two error messages made actionable** with two conda environments in play:

- (#47) "pip install mlflow" is not actionable when following it inside the
  SAM 3 environment breaks its numpy pin. The error now names `sys.executable`
  and its environment.
- (#48) A CPU-only torch failed inside `Module.to()` half a minute into a run,
  after the dataset loaded and MLflow opened a run, with an `AssertionError`
  naming neither cause nor fix. `require_device()` now runs first.

**numpy pinned `>=1.26,<2`** (#12) — unpinned, `pip install -r requirements.txt`
could upgrade an environment and break SAM 3. The pin is load-bearing.

---

## 9. Training: the resolution era

**#49 is the single largest measured gain in the project.**

torchvision's default resize downscaled 2208×1242 ZED frames to 1333×749,
shrinking a 250 px weed to 9.5 px — **below the smallest 32 px RPN anchor, so
the proposal network could never propose it.** `min_size`, `max_size` and
`anchor_sizes` became configurable and are stored in the checkpoint so inference
matches training.

**Augmentation was undoing it** (#53). `v2.ScaleJitter(target_size=...)` resizes
the *canvas*; only `v2.RandomAffine(scale=...)` scales *content*. Measured, the
augmented frames were 24–73% of native — so `MIN_SIZE: 1000` was doing nothing
at all. Replaced with `RandomAffine`.

The same PR fixed a crash three minutes into a run:

```
AssertionError: All bounding boxes should have positive height and width.
Found invalid box [1432.63, 997.66, 1436.67, 997.66]
```

`masks_to_boxes` uses **inclusive** min/max, so a 1-px-tall mask yields a
zero-height box. The guard existed before augmentation but not after it.

---

## 10. RF-DETR-Seg backend

**Added as a fully wired second backend** (#50), Apache-2.0 — Mask R-CNN stays
the default. Two things it buys that torchvision cannot expose without forking
its ROI heads: **configurable mask losses**, and **no anchors at all** (DETR set
prediction, so the anchor-size trap of #49 cannot recur in the same form).

The runner **refuses to train at the model's own 432×432 default**, naming the
valid resolutions for the variant. Accepting it silently would have regressed
small-weed recall while looking like an upgrade.

Only Nano–Large are offered: XLarge/2XLarge may fall under Roboflow's Platform
Model License, which is not one a commercial laser weeder can ship under
unchecked.

**Then five PRs of things that were silently wrong**, each found by actually
running it:

- (#52) rfdetr builds its `MLFlowLogger` without a tracking URI, and Lightning
  reads `MLFLOW_TRACKING_URI` as a **default argument evaluated at module
  import** — so the environment variable has to be set *before* rfdetr is
  imported. Same PR: `--backend` was being parsed and dropped in `eval_seg`,
  which would have scored an RF-DETR checkpoint through the Mask R-CNN builder.
- (#57) The first run finished and the command printed to score it would have
  produced nonsense: wrong architecture, wrong resolution, wrong class list.
  None of the three raise. All now travel with the run in
  `rfdetr_train_config.json`.
- (#58) The guard added in #57 fired immediately: *"predicted class id 4 but was
  loaded with 4 classes"*. It was right to stop. Same PR: rfdetr defaults
  `lr_scheduler="step"` with `lr_drop=100`, so **on any run shorter than 100
  epochs the step never fires and the learning rate is constant start to
  finish** — the first run ended at the same 1e-4 it began with.
- (#59) The label mapping, wrong twice, so this time it was read out of the
  package rather than reasoned about. `PostProcess` computes
  `labels = topk_indexes % out_logits.shape[2]` — **raw 0-based labels** over
  ascending category-id order. And LW-DETR allocates `num_classes + 1`
  classifier outputs, where the extra slot has no training target and must be
  dropped.
- (#63, in part) `checkpoint_best_total.pth`, not `_ema`. rfdetr keeps three and
  copies `_total` from whichever actually won, so naming the EMA file silently
  scores the loser whenever the regular weights were better — which had already
  happened once.

---

## 11. Evaluation that measures the right thing

**Crop burn, not just crop misses** (#51). `missed_onion_fraction = 0.181` gives
no way to tell whether the model is *ignoring* 18% of the crop or *aiming a 60 W
laser at it*. Those are not the same failure and do not have the same fix.
`weed_on_crop_px` counts ground-truth onion pixels a **weed** prediction claims;
pixels claimed by both crop and weed predictions are excluded, because the
onion-conflict check suppresses that shot.

**The operating point is a choice** (#55). run4 scored small-weed recall
**0.277 at conf 0.5 and 0.732 at conf 0.25 on unchanged weights.** The number
was measuring the threshold, not the model. `--sweep` reports the full curve so
the deployment confidence is chosen rather than inherited.

Same PR: overlays coloured by class, so cutleaf evening primrose is
distinguishable from everything else at a glance.

**A refactor regression** (#56). Extracting the operating point into
`_summarise()` dropped `operating = {"conf": conf}`, so `--sweep` raised
`KeyError: 'conf'` on the first real run. The deeper gap: every `format_report`
test used hand-built dicts, so none of them exercised real output.

**Evaluation ran long enough to be killed by hand** (#60). It was not the model
— it was mask IoU at 3.84 ms/pair on 2208×1242 masks. A bbox pre-filter plus an
intersection-box AND, with the matrix reused across thresholds and confidences:
**120× faster.**

**Visual report** (#43). One self-contained HTML file: recall bucketed by
instance size, crops of every missed weed smallest first, per-frame GT beside
prediction coloured **by outcome rather than by class**, and crop safety kept
out of every averaged score.

**Inference on unlabelled frames** (#54). Everything that drew a prediction
needed ground truth, so nothing could be pointed at a held-out session, a new
field, or tomorrow's drive — the frames that actually decide whether the model
is worth deploying.

---

## 12. Analytics for both backends

**#61.** RF-DETR was being handed to its own logger, which produces MLflow
scalars and nothing else, while the richer `Tracker` path stayed on the Mask
R-CNN side. So one backend left preview panels behind and the other left a CSV,
and the two were not comparable in any single view.

`evaluation/plots.py` and `evaluation/analyze_run.py`: training curves, per-class
AP, the confidence sweep, the crop-safety curve and recall-by-size, generated
for either backend by pointing `analyze_run` at a run directory. It detects the
backend from what the directory contains.

---

## 13. Growing the dataset

**#62.** `active_learning.py` had been in the repo scoring frames well, and
nothing could call it: it consumes stored `FrameResult` dicts and nothing
produced those over the unlabelled pool. Good code with no way in.

`mine_pool.py` is the way in — runs a checkpoint over the pool, ranks by
uncertainty × rarity × crop-risk, does a farthest-point diversity pass, and
writes the same CVAT-ready layout the SAM 3 prelabelers produce.

**Model-in-the-loop rather than SAM 3**, because a trained checkpoint knows this
ontology and the SAM 3 prelabelers propose generic classes — which is why
earlier CVAT tasks arrived full of confidently wrong labels. The trade has a
cost the module states rather than sells: **a prelabelled frame biases an
annotator toward accepting what is present and not noticing what is absent**,
and missed weeds are this project's failure mode. Hence a mining confidence of
0.20, below deployment.

What it refuses to do: frames already in `seg_manifest.json` are excluded (the
same frame in two CVAT tasks produces two versions of the truth), and
`HOLDOUT_SESSIONS` are excluded because **a session intended as a test set stops
being one the moment it is annotated into training**, and this is the last point
where that is preventable.

[`docs/dataset_growth.md`](docs/dataset_growth.md) is the comprehensive
treatment: why pure uncertainty sampling would return twenty pictures of one
plant here, the three named failure modes and what each actually needs, and what
is not worth doing yet.

**On teacher–student:** RF-DETR's EMA already provides the teacher and the
papers are real, but it is deliberately not wired up. A pseudo-label is the
model's belief taught back to itself. For weeds that is survivable. **For the
crop it is not** — the model already calls grass `onion_plant` at 0.91
confidence, and feeding that back teaches it to be confidently wrong about the
crop while every metric *improves*, because the model now agrees with itself. If
it is ever built: weeds only, onion from a human.

The document's first recommendation is 40 frames that improve the model not at
all — **a held-out session.** Every number so far is same-session validation, so
nothing can currently tell whether new data helped.

---

## 14. Loss shaping

**Tversky** (#63). Dice weights a false positive and a false negative equally.
Nothing about this system does: a missed weed survives to set seed, a spurious
one costs one laser pulse, and onion the model fails to mark is onion the
targeting stage has no reason to protect.

```
TI = TP / (TP + ALPHA*FP + BETA*FN)          loss = (1 - TI)^GAMMA
```

`alpha = beta = 0.5` is **exactly** Dice — asserted against rfdetr's own
`dice_loss` to 1e-6 on random logits, which is what keeps a Tversky run
comparable with every Dice run already recorded. It installs by rebinding
rfdetr's module-level `dice_loss_jit`, and raises if that symbol is absent
rather than silently training with Dice while the config claims Tversky.

**Connected components** (#63). `drop_fragments` keeps the largest component and
anything comparable to it. The threshold is a **fraction** of the largest
component, not an absolute area: an absolute floor would delete genuine
cotyledons, which are exactly the instances this project is already losing. Two
parts of one plant separated by an occluding leaf are both kept.

Wired into the prelabel export, where the saving is annotation time — **not**
applied to the model's own output during evaluation, since a silent cleanup step
would change a metric being tracked across runs.

**Provenance for the loss** (#64). The loss is patched in at runtime, so nothing
rfdetr writes records it, and two runs differing only in alpha/beta shipped
byte-identical sidecars. `rfdetr_train_config.json` now carries a `mask_loss`
block — on Dice runs too, so an absent block never has to be read as "probably
the default".

---

## 15. Mixed scenes

**#64.** Neither prelabeler could be pointed at a scene containing both crops
and weeds: one assumes `vegetation == weed` and would call every onion a weed,
the other assumes the reverse.

`prelabel_mixed_sam3.py` drops the class proposal entirely and does one job —
**one precise mask per plant** — emitting a single class, `plant`, reassigned in
CVAT with one keystroke per shape. That is the correct call, not a shortcut: the
one morphology decision the weed prelabeler makes confidently is *elongation →
`grass_weed`*, and **an onion is a blade.**

**Vegetation owns the boundary; SAM owns the identity.** Where a plant ends is
answerable per pixel at full resolution by a colour index; which plant a green
pixel belongs to is not. SAM is the reverse. So SAM proposals are reduced to
seeds, and a marker-controlled watershed over the vegetation mask — flooding the
inverted distance transform — assigns every green pixel to exactly one seed.

The peak-based split that fragmented plants in #29 cannot recur here: the
markers come from SAM, so a blob divides only where SAM saw two plants. The
watershed decides *where* the cut falls, never *whether* there is one.

Two things measured rather than assumed: seeding the soil as a background marker
**ate four of six arm tips** on a synthetic rosette (soil is flat ground at the
top of the relief, so the background flood beats the one coming up the arm), and
watershed ridge pixels shave an edge off every instance unless reclaimed where
the neighbourhood is unambiguous.

Full treatment in [`docs/mixed_prelabeling.md`](docs/mixed_prelabeling.md).

**Then the first real run showed onion coming out as a scatter of tiny
speckle instances instead of one mask per leaf.** Not a SAM problem — the
vegetation prior was fragmenting a single onion leaf into a handful of
disconnected slivers before SAM ever saw it, and everything downstream
inherited the damage (exemplar boxes per fragment instead of per leaf, the
watershed unable to flood across the gaps, fragment-dropping treating the
slivers as soil-texture speckle).

The cause is onion's leaf surface itself: a glossy, waxy, glaucous
blue-green tube. Its wax bloom lifts blue reflectance past green, failing the
prior's `g >= b` gate outright over parts of the leaf, and its glossy curve
throws specular highlights carrying no leaf colour at all — a highlight *is*
the light source's colour, not the tissue's, so no threshold admits it. Both
defects cut across the leaf's **width**, so the gap touches soil on both
sides, and `fill_holes()` — which only fills gaps fully *enclosed* by
vegetation, by design — cannot reach either one.

`recover_glaucous_pixels()` fixes it without touching the shared
`vegetation.py`, which the onion and weed prelabelers are tuned against.
It relaxes the green-dominance and saturation gates, but only within a small
halo around pixels the strict gate already accepted, so a bare patch of pale
soil earns nothing while a highlight sitting inside a confirmed leaf does.

A second, purely geometric closing step (`close_thin_gaps`) was tried as a
belt-and-braces addition and **shipped off by default** after it broke an
existing regression test: measured against two plants touching at a 4px
overlap, even a small closing kernel smoothed over the concave neck between
them exactly the way it smoothed over a gap within one leaf, because a
shape-only operator cannot tell the two apart. The colour-aware bridge has no
such failure mode — the soil around a genuine neck fails its colour test —
and was measured sufficient on its own.

**Reconnecting the mask turned out not to be enough on its own.** Real
sessions still came back with empty masks and empty previews on most frames —
sparse ones and dense, overlapping ones alike — even with the mask itself
fixed. Two more places assumed a clean, unfragmented prior.

First, the size floor was running *before* bridging: `vegetation_mask()`
deletes any component under `VEG_MIN_COMPONENT_PX` before returning, and on a
thin, heavily afflicted leaf every individual fragment can be smaller than
that floor even though the leaf as a whole is not. Pruning first deletes every
fragment before bridging ever sees them — and dilating an already-empty mask
stays empty, so the whole plant vanished. `strict_vegetation()` now computes
the strict gate with no size floor at all, bridging reconnects every surviving
speck, and the real floor is applied once, at the very end, to the
**reconnected** result.

Second, and the larger effect in practice: the recall backstop and the
exemplar-confidence gate both scored a component with `vegetation_score()` —
the exact strict colour rule that fragmented the leaf in the first place. A
bridged pixel scores near zero on the rule it was bridged *for failing*, so a
component substantially made of bridged pixels — measured at 0.62–0.71 on a
synthetic afflicted leaf — scored comfortably under `RECOVER_MIN_VEG_SCORE`
(0.90) and `EXEMPLAR_MIN_VEG_SCORE` (0.85), rejected by the exact gates meant
to keep noise out. This fired on **every** frame where bridging did meaningful
work, not a rare edge case — in practice most onion-containing frames, which
is why both sparse and dense sessions collapsed to empty output. It was also
invisible to the tests that shipped with the original fix, because they fed
hand-built SAM masks directly into `analyze_frame()`, never exercising the
real `component_boxes(confidence=...)` path `prelabel_session()` actually
uses to prompt SAM. `plant_confidence()` scores a bridged pixel by the best
strict-gate score found within `VEG_BRIDGE_PX` of it — the same context that
already justified admitting the pixel — instead of re-litigating it against
the rule that failed in the first place.

---

## 16. The Feb 2025 visit, and depth that never existed

Extracting an earlier campaign turned up nine sessions where seven were AVI and
two MKV, under one parent folder. Three things came out of it.

**The AVI sessions were invisible, not skipped** (#65). The discovery patterns
carried `.mkv`, and the "this looks like a recording" fallback searched for
depth with the same extension list that had just failed — so `discover()`
returned nothing for them *and printed nothing*. A run over the folder reported
success after extracting two of nine.

**Depth is now checked before it is decoded** (#65). ffmpeg will produce
`gray16le` from an 8-bit source by scaling up values it invented, and the
result is a valid PNG full of plausible millimetres that are fiction. The QC
fractions compute, the range looks sane, and the LEP's canopy-height channel
reads the noise as terrain.

**Then a chain of tools to answer what the 8-bit files actually were**
(#66–#70): what is in the file, how to recover a preview if it is one, how to
measure the scale rather than assume it, and a control to say whether a failed
calibration is the preview or just two sessions of different ground.

The answer, in the end, came from the capture source: those builds wrote
`depth / depth.max() * 255` with the max recomputed **every frame**. The scale
is a property of each individual frame, so no constant could ever have
recovered it — the calibration was right to refuse, and now says so on sight
rather than after a fit that drifts.

That visit's depth was destroyed at capture time. Its RGB is fine, and RGB is
what segmentation needs, so the seven sessions are fully usable for everything
currently being built.

---

## 17. The mixed-scene dataset strategy

The height veto was field-tested and reverted (#93), and the onion build made it
clear the exports carry **unreviewed** SAM output rather than corrected
annotation — recorded from then on as `label_provenance`, and restated by
preflight at train time, because it decides whether a reported AP measures
performance or agreement with the prelabeler.

That forced the question of how a mixed-scene dataset actually gets built.
[`docs/mixed_dataset_strategy.md`](docs/mixed_dataset_strategy.md) is the record.
The findings that changed the plan:

**The bottleneck is instance identity, not boundary quality.** The mixed
prelabeler's marker-and-watershed design is sound, but a watershed is only as
good as its markers, and zero-shot SAM gets identity wrong in dense mixed scenes.
Boundary refinement improves the half that already works. Identity is domain
knowledge — that two touching onion *leaves* are one plant while two touching
onion *plants* are two — and only data teaches it.

**`weed_cluster` is not an escape hatch.** Annotation policy becomes deployed
policy: reach for the cluster class whenever separation is *tedious* rather than
*impossible*, and the model reproduces that at runtime as weeds that never get an
individual LEP. The ontology's criterion was always crown-based — *"no separable
single LEP"* — not foliage-based. Its **rate** is therefore a process metric, and
a rising one is an early warning that effort is decaying.

**Merges are not equally costly, but that ranks effort rather than licensing
skips.** weed+weed loses a target; onion+onion barely matters because protection
is a union and onions are never targeted; onion+weed is the dangerous one. For
onions the risk is mask *extent*, not count — err large, without training the
model to be indiscriminate.

**Mean IoU would hide exactly the failure that matters**, averaging the
catastrophic direction into many harmless ones. The mixed benchmark reports
onion-labelled-weed separately from weed-labelled-onion, the same reasoning that
already keeps `onion_recall` apart from IoU in `onion_safety_metrics`.

**Separation gets cheaper, not optional.** Prelabel-then-correct only beats
annotating from scratch when the prelabels are good; repairing a merged mask is
often slower than one click. The crown click *is* the instance definition, and
yields identity, the LEP and a SAM prompt in one gesture.

**The ruler, and what to annotate next.** `evaluation/bench_mixed.py` scores any
prelabeler or model against the hand-annotated mixed frames, reporting the three
groups above without combining them, and flags a small frame count next to the
numbers rather than letting a per-frame fraction from ten frames look sturdy.
Crop error pools by PIXEL, so a frame holding four onions does not weigh the
same as one holding forty. `annotation/rank_by_contact.py` ranks frames by how
much onion/weed boundary they contain - the decision that matters - and runs on
whatever prelabels exist, before anything is trained. It caps per session by
default, because ranking on one signal and taking the top N returns the same
stretch of one drive.

[`docs/stage_a_improvements.md`](docs/stage_a_improvements.md) records the Stage A
analysis: that the current limit is label provenance rather than architecture,
so an architecture comparison run today measures agreement with SAM; that
resolution has historically bought more than capacity on this data; and that the
two-stage arrangement - trained model for identity, SAM for boundaries - is the
only route to output better than the masks trained on.

---

## 18. Round 0 of the weed loop, and a low AP that was the split's fault

The first weed model trained on 42 frames, early-stopped at epoch 73, and
reported `mAP@50 = 0.701`, `mAP@50:95 = 0.410`, small-weed recall 0.767 at
conf 0.25. Two classes scored 0.557 and 0.458; `other_weed` scored **0.214**.

`other_weed` was not harder. It was in the wrong place:

| class | train | val | val share of instances |
|---|---|---|---|
| cutleaf_evening_primrose | 642 | 73 | 10% |
| grass_weed | 327 | 26 | 7% |
| **other_weed** | **145** | **74** | **34%** |
| *frames* | *42* | *10* | ***19%*** |

The val block landed on an `other_weed`-dense stretch of the drive, so the
class was **starved in training and over-weighted in the score at once**, and
the mean AP carried the difference. Nothing in a per-class AP table can show
this: the table reports the score, never the split that produced it.

The cause was that the block layout is rotated by a CRC32 of the session key.
With one block there are exactly three layouts and nothing was choosing the
balanced one — the rotation existed to stop the test set being made entirely of
drive-ends (#split rotation), and it solved that while staying blind to
contents.

`splits.py` now lays out all three rotations and keeps whichever puts each
class's val share closest to that split's share of the frames, consulting
ground-truth label counts only — never a model, never a score. Ties keep the
layout the seed would have picked, so builds with nothing wrong with them do
not silently re-draw. `prepare_dataset` prints the balance table beside the
frame split and flags any class drifting more than 1.5×.

**The tolerance came from the case, not from taste.** The skew above is a ratio
of 1.79. A rounder-looking 2.0 would have said nothing about it, which is the
only test a threshold has to pass.

A class confined to one stretch has no layout that fixes it, and the build says
so rather than pretending: the lever is **more blocks paid for with a smaller
gap**, since blocks cost one seam each.

Rebuilt at 3 blocks and a 3-frame gap, `grass_weed` landed at 21% against a 19%
target and `other_weed` moved from 34% *over* to 8% *under* — with its training
instances up from **145 to 199**.

That flip is the reason the objective is asymmetric. A class concentrated in val
is starved in training *and* over-weighted in the score; a class thin in val
trains on nearly everything and merely gets a noisy AP. So over-representation
is weighted 2× when the layout is chosen, and the warning names which direction
it hit. Same rule this project already applies to crop safety: an asymmetric
pair of errors is never averaged into one number.

**A stride estimate here was also wrong.** The docs had the pool curated at
~stride 2, concluding the 60-video-frame separation floor was unreachable
without spending half the dataset. Measured: gap 8 gives 240, gap 3 gives 95 —
about 30 video frames per pool frame, so `GAP_FRAMES=2` already clears it. Gap
frames are cheap on this pool, which is why 3 blocks at gap 3 costs only 12
buffered frames. Clearing the floor still only means val is not the same
photograph as train; it shares the session's light and soil regardless.

[`docs/weed_active_learning.md`](docs/weed_active_learning.md) carries the
numbers and the blocks-vs-gap table.

---

## Lessons this project paid for

Each of these cost a run, a dataset, or a batch of annotation time.

1. **A silent wrong answer beats a loud one every time — so make it loud.**
   Wrong architecture, wrong class list, wrong resolution, a dropped
   `--backend`, an empty crop mask, a duplicate CVAT label: none of these raise.
   Most of the guards in this repo exist because one of them didn't.

2. **Check the imagery before believing the metric.** Small detections read as
   noise and were real cotyledons (#16). A "50% drop rate" read as too
   aggressive and was just a steady driving speed (#28). A boundary pipeline
   that improved every number produced worse masks in the field (#29).

3. **The operating point is part of the result.** 0.277 and 0.732 small-weed
   recall on identical weights (#55). Any single-threshold table is a claim
   about a threshold, not a model.

4. **Recall and precision are not symmetric here.** A missed weed sets seed and
   is invisible; a spurious one costs one delete or one laser pulse. That
   asymmetry drives the mining confidence, the fragment threshold, the Tversky
   weighting and the mixed-scene recall backstop.

5. **Never pseudo-label the crop.** The model is already confidently wrong about
   grass-as-onion. Teaching that back improves every metric while making the
   failure worse.

6. **Same-session validation is not validation.** Still open, and still the
   highest-value 40 frames available.

7. **Read the package, don't reason about it.** The RF-DETR label mapping was
   wrong twice from plausible reasoning and right once from reading
   `PostProcess`. The same held for the Feb 2025 depth: four tools of
   increasingly careful inference, and the capture source settled it in one
   line.

8. **Annotation policy becomes deployed policy.** A cluster class used when
   separation is merely tedious teaches the model to do the same at runtime,
   where it means weeds that are never targeted. Systematic label bias is
   reproduced, not averaged out.

9. **A pipeline that skips input silently is worse than one that crashes.**
   Seven of nine sessions vanished with no output at all, and the run looked
   like a success.

10. **A low per-class AP accuses the model, and the split is often guilty.**
    `other_weed` scored 0.214 while holding 34% of its instances in a 19% val
    split — starved in training and over-weighted in the score at the same
    time. An AP table reports the score and never the split that produced it,
    so the check has to run at build time, where the split can still change.
