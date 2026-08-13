# Weed instance prelabeling (`annotation/prelabel_weeds_sam3.py`)

Builds a **multi-class weed instance** dataset from weed-only recordings, with a
proposed **LEP/AMT growth point** per instance, exported for correction in CVAT.

For **weed-only** recordings. In a weed-only field there is no crop to confuse,
so `vegetation == weed` is a legitimate high-recall prior — exactly as
`vegetation == onion` is in an onion-only field. **Do not point this at mixed
scenes.**

## How this differs from the onion prelabeler

| | Onions | Weeds |
|---|---|---|
| Goal | one high-recall **semantic** safety mask | every **individual** plant separated |
| Priority | coverage over separation | separation + class + treatment point |
| Classes | one (`onion_plant`) | `cutleaf_evening_primrose`, `wild_radish`, `grass_weed`, `weed_cluster`, `other_weed` |
| Growth point | not needed | **LEP/AMT proposed per instance** |

## Pipeline

1. White balance → vegetation prior (ExG + green dominance + saturation), shared
   with the onion path via `common/vegetation.py`.
2. Vegetation blobs become SAM 3 exemplar boxes. SAM 3 concept segmentation
   returns **every matching instance as its own mask** — that is the instance
   segmentation. (Text concepts do not ground on top-down field imagery, so
   exemplars are the default, as established for onions.)
3. Instances are validated against the vegetation prior and de-duplicated by
   **mask NMS**; too-small and whole-frame masks are rejected.
4. **Recall backstop (off by default)**: optionally, vegetation no instance
   claimed can be recovered as extra instances (see below for why this isn't
   on by default and when to turn it on for a specific session).
5. Per instance: shape descriptors → provisional morphology class, growth-stage
   estimate, and **three candidate treatment points**.
6. Export COCO (categories = morphology classes) + per-instance CSV + instance
   crops + previews.

## The three treatment points (why all three)

| Point | Meaning |
|---|---|
| `lep_dt` | **distance-transform peak** — the deepest interior point of the mask. For a rosette weed this lands at the centre where the youngest leaves emerge, i.e. the growth point. The strongest purely geometric prior for the LEP/AMT before a trained heatmap model exists. |
| `centroid` | mask centroid (project plan baseline) |
| `bbox_ctr` | bounding-box centre (project plan baseline) |

Storing all three means the plan's **LEP-method comparison** (§15/§22 — bbox
centre vs mask centroid vs learned heatmap) can be computed directly the moment
human LEP labels exist, with no re-processing.

`dt_radius_px` (inscribed-circle radius at the peak) and `lep_centroid_dist_px`
are recorded too — large centroid distance flags asymmetric or occluded plants
where the LEP proposal is less reliable, which is useful active-learning signal.

## What the prelabeler can and cannot decide

| Class | Auto-assigned? | Why |
|---|---|---|
| `grass_weed` | **yes**, by elongation | Measured: a blade has aspect ratio ~20, a rosette ~1. |
| `weed_cluster` | **yes**, high threshold | Several distinct growth-point peaks inside one large mask, i.e. individual LEPs genuinely cannot be assigned. |
| `other_weed` | fallback | Everything else, with **zero confidence**. |
| `cutleaf_evening_primrose`, `wild_radish` | **never** | Both are rosette-forming, so shape cannot separate them. Species is an *appearance* question. |
| `onion_plant` | **never** | The crop. Present in the CVAT schema only so that an onion appearing in a weed scene can be labelled correctly instead of being forced into a weed class - a crop-safety protection. |

Class names and COCO category IDs come from `common/ontology.py`, the single
source of truth, so a class can never be spelled two ways and the weed, onion
and future mixed datasets merge without remapping annotations.

**This is the key limitation to understand:** no threshold tuning will make the
prelabeler tell cutleaf evening primrose from wild radish — both are
rosettes, and aspect ratio and solidity describe
*form*, not species. Two things resolve it: you assigning species in CVAT, and
(much faster) the DINOv2 cluster-then-label stage below.

What the shape data actually showed (measured, not assumed):

- **Elongation is the only reliable shape discriminator** — grass ~20 vs rosette ~1.
- **Circularity is useless here** — a rosette's radiating leaves give it a long,
  spiky perimeter, so its circularity (~0.15) is as low as a blade's (~0.13).
- **Solidity cannot flag grass** — a straight blade is nearly its own convex
  hull (solidity ~0.9).

### Cluster detection

`weed_cluster` fires only when a mask has at least `CLUSTER_MIN_PEAKS` distinct
growth-point peaks **and** exceeds `CLUSTER_MIN_AREA_PX`. Raise either to make it
rarer. A cluster gets **no single LEP** (`lep_valid=0` in `instances.csv`); the
preview marks each detected growth point with a cross instead of one dot.

## Recall vs precision: what shipped, what got reverted, and why

**A weed the prelabeler misses is worse than a weed it labels badly** in
principle — a bad label gets corrected in CVAT in seconds, a missing one
silently becomes *background* in the training target. That reasoning motivated
two recall-boosting changes: prompting SAM with more, smaller exemplars
(`EXEMPLAR_MAX_BOXES` 30→60, `EXEMPLAR_MIN_AREA_PX` 300→200), and a backstop
(`recover_missed_plants()`) that turns any unclaimed vegetation component into
an instance outright.

**Real runs on a pale, textured field surface showed why that reasoning was
incomplete**: dozens of phantom point detections scattered across bare ground
with no plant present. A dataset full of phantom detections is not trainable
at all, which in practice is a worse failure than the recall problem this was
meant to fix. Two rounds of tightening (a claimed-fraction guard, then a
vegetation-confidence gate) reduced but never eliminated it on real imagery,
because the underlying signal — a plain ExG + saturation + green-dominance
colour index — is a documented weak spot against green-tinted mineral flecks,
lichen, and shadow reading cooler/greener than sunlit ground. Colour evidence
alone is sometimes genuinely ambiguous; no amount of threshold tuning makes it
always separable.

**Current defaults, verified clean:**

| Setting | Value | |
|---|---|---|
| `EXEMPLAR_MAX_BOXES` / `EXEMPLAR_MIN_AREA_PX` | 30 / 300 | reverted to the original, verified-clean values |
| `RECOVER_MISSED_PLANTS` | **False** | off by default |
| `EXEMPLAR_MIN_VEG_SCORE` | 0.85 | kept as extra protection on the (now smaller) exemplar set |
| `RECOVER_MIN_VEG_SCORE` | 0.9 | kept for when recovery is re-enabled |

### Why more exemplars hurt rather than helped

Exemplar boxes are submitted in **one forward pass**, so more of them is cheap
in *compute* — but SAM 3 concept segmentation is asked to find *every instance
of the same concept* as the exemplars, **across the whole frame**. A single
low-quality exemplar (gravel, lichen, a shadowed pit that only marginally
passed the vegetation prior) does not just risk one bad box — it can teach SAM
a bad concept and cause it to propose several more false positives elsewhere in
the frame that resemble it. Lowering the area floor and raising the box count
made this measurably more likely, not less.

### The backstop (`recover_missed_plants()`, off by default)

In a weed-only scene `vegetation == weed` — the same prior the onion path
relies on — so any substantial connected vegetation component that **no
instance claims** can be recovered as an instance in its own right. The
recovered mask goes through *exactly the same* boundary refinement and
touching-plant splitting as a detected one, indistinguishable in the output
except for its recorded `source`.

Two guards exist so it does not fabricate duplicates or trust weak evidence:

- **`RECOVER_MAX_CLAIMED_FRAC`** (0.30): not the size of the unclaimed residual
  but how much of the **whole vegetation blob** it belongs to is already
  claimed. A leaf tip clipped off a good detection sits on a blob ~85%
  claimed and is rejected; a plant SAM never saw sits on a blob 0% claimed and
  is recovered.
- **`RECOVER_MIN_VEG_SCORE`** (0.9): the component's **mean** continuous
  `vegetation_score()` must clear the floor, not merely cross the binary
  threshold once. Measured real plant colour, even degraded (shade, a pale
  cotyledon, a grazing-angle leaf edge), scores ≥ 0.985 - comfortable margin.
  Measured on a synthetic adversarial gravel/lichen/shadow texture: some
  components still scored up to 0.971, which is why this remains off by
  default rather than trusted as a solved problem.

This has **no SAM corroboration at all** — unlike an exemplar, which SAM still
gets to independently confirm or reject, a recovered instance is accepted
purely on the vegetation prior's say-so. That is what makes it the most
exposed mechanism in this pipeline, and why it needs to be turned on
deliberately, per session, rather than trusted as a default.

**When to re-enable it**: if a specific session's previews show real,
confirmed under-detection (not just an assumption) and spot-checking a run
with `RECOVER_MISSED_PLANTS: True` shows clean recovered instances (no white
halos on bare ground), it's a reasonable per-session choice. Don't leave it on
by default across unseen substrates.

### Seeing it work

Every session prints a recall line:

```
      recall: 91.2% of vegetation inside an exported instance | 1642 from SAM + 0 recovered
```

With recovery off by default, `recovered` will normally read 0 — that's
expected, not a bug. Watch the SAM-only recall percentage instead: a low number
on a session that visibly has plenty of plants in it is the signal to go look
at previews and consider enabling the backstop *for that session specifically*.
If you do enable it, a recovered instance is drawn with a **white halo** around
its class-coloured outline, so you can judge directly from the images whether
it's finding real plants or fabricating them. `instances.csv` always carries a
`source` column (`sam` / `vegetation`) so the two populations stay separable.

## Boundary quality

> **All of the post-processing in this section is OFF by default.** A field
> comparison judged the PR #11/#12 masks — plain SAM output, no post-processing
> — better than the processed ones, so that is the shipped default. The code is
> intact, tested, and each piece is one config flag away; this section explains
> what each does so you can re-enable them individually and judge for yourself.
>
> The decisive one is **splitting** (§3): it is the only thing in the pipeline
> that can turn one SAM mask into several instances, and on real plants a leaf
> reaching away from the crown produces a second distance-transform peak, which
> was enough to cut one weed into two or three separate annotations. A false
> split is worse than a missed one — an over-segmented plant teaches the model
> that half a rosette is a whole instance, and the annotator has to merge shapes
> by hand instead of just drawing one boundary. With it off, intergrown plants
> stay one blob and become `weed_cluster`, which is the honest label for tissue
> that genuinely cannot be separated.

| Feature | Flag | Default | Enable with |
|---|---|---|---|
| Edge snapping | `BOUNDARY_REFINE_BAND_PX` | `0` | `3` |
| Anti-aliasing | `BOUNDARY_SMOOTH_SIGMA_PX` | `0.0` | `0.7` |
| Splitting touching plants | `SPLIT_TOUCHING_INSTANCES` | `False` | `True` |
| Multi-part polygons | `POLY_ALL_PARTS` | `False` | `True` |
| Multi-evidence LEP | `USE_FUSED_LEP` | `False` | `True` |

Four things determine how good the exported boundaries are as a *training
target*, and all four are handled explicitly (when enabled).

### 1. Edge snapping (`BOUNDARY_REFINE_BAND_PX`)

SAM gives excellent structure but its edge can sit a few pixels off the true
leaf margin - bleeding onto soil, or clipping a thin leaf tip. Only the narrow
band around each boundary is re-decided, using a **continuous** vegetation score
(`vegetation_score()`, the soft counterpart of the binary prior): SAM decides
*what* the object is, the image decides exactly *where it ends*. The interior
and the overall shape are never touched, and added pixels must stay connected to
the original core so refinement cannot absorb a neighbouring plant. Set the band
to 0 to disable.

### 2. Anti-aliasing (`BOUNDARY_SMOOTH_SIGMA_PX`)

Edge snapping still makes a hard, independent per-pixel decision, so the
resulting boundary keeps single-pixel staircase noise that no real leaf margin
has. `smooth_boundary()` blurs the mask and re-thresholds at 0.5 - the standard
way to anti-alias a binary mask - immediately after edge snapping, before
anything downstream sees the boundary.

This isn't just cosmetic. The skeleton-based `PetioleConvergence` LEP evidence
(`perception/lep.py`) finds the growth point at skeleton **junctions** where
petiole axes meet; every boundary jag fabricates a short spurious branch and a
spurious junction, i.e. noise injected directly into the growth-point estimate.
Measured on a synthetic noisy disc: smoothing cuts the perimeter **11%** (the
staircase noise) while area moves by **0.09%** (7 px of 7708), and it reduces
skeleton junction count on the same shape. On a grass blade, aspect ratio and
area are unchanged to the pixel - sigma is deliberately sub-pixel (default
`0.7`), far smaller than any real leaf structure, so elongation survives
exactly.

The one failure mode - blurring through a genuinely thin neck between two
lobes - is guarded the same way `split_touching_instances()` guards its own
split: the smoothed result is kept only if it retained
`BOUNDARY_SMOOTH_MIN_RETAINED_FRAC` (default 85%) of the original area *and*
stayed a single connected blob; otherwise the unsmoothed mask is returned
rather than tissue silently disappearing. Set sigma to 0 to disable.

### 3. Splitting touching plants (`SPLIT_TOUCHING_INSTANCES`)

Two rosettes growing into each other are **one connected blob**, so SAM returns
them as a single instance with no boundary between them. Each blob containing
several detected growth points is split into one mask per plant by **geodesic
assignment**: every pixel goes to the growth point it reaches by the shortest
path *through the plant*.

That is the correct semantics - a leaf belongs to the plant whose crown it
physically joins - and it is much more robust than a distance-transform
watershed, which on spindly plants has a flat ridge along thin leaves and so
places the cut arbitrarily. Measured on two overlapping synthetic rosettes:
watershed gave parts of 7042/1848 px with one part swallowing 2222 px of its
neighbour, while geodesic assignment gives 5216/5228 px, 100% coverage, zero
overlap, and each part claiming its own plant's tissue exclusively.

The split falls back to the unsplit mask whenever it is not clean (fewer than
two viable parts, or less than `SPLIT_MIN_COVERAGE` of the blob retained), so a
doubtful split can never fabricate a boundary. `weed_cluster` is then reserved
for what genuinely cannot be separated.

### 4. Polygon fidelity

- **`POLY_ALL_PARTS` (default off) exports every part**, not just the largest
  contour. This does *not* create extra instances — a multi-part instance is
  still one COCO annotation — but it draws one outline per part in the preview,
  which can read as fragmentation even though it isn't. PR #12 exported the
  largest contour only, so that is the default.
  **The trade-off is real:** largest-only *silently drops* tissue whenever a
  leaf is separated from the crown by an occluding leaf, so that plant tissue
  enters the training target as background. If previews look clean but plants
  have visibly missing leaves, turn this on.
- **Tolerance scales with instance size** (`POLY_APPROX_EPS_FRAC`, clamped by
  `_MIN`/`_MAX`). A fixed tolerance erases real shape on a cotyledon seedling
  and leaves thousands of near-duplicate vertices on a large rosette; scaling
  with the square root of area keeps roughly constant *relative* fidelity.
- **`area` is the true mask area**, not the bbox area, which badly overstates a
  thin or lobed plant.

### Controlling over-detection

`MIN_INSTANCE_AREA_PX` (default **250**, ~16x16 px at 2208x1242) sets how small a
detection may be.

It was briefly raised to 700 on the assumption that the many small detections in
dense frames were noise. Checking the imagery showed they are **real
cotyledon-stage weeds**: at 700 every plant under ~29 px diameter was silently
dropped, which removed the entire cotyledon and 2-leaf population. For a laser
weeder a missed small weed is a worse failure than an extra instance the
annotator deletes in one click, so the default keeps them. Raise it only if
previews show detections sitting on bare soil rather than on real seedlings.

## Run

```python
DATASET_ROOT   = r"E:\Dataset_Vidalia"
SAM_VERSION    = "sam3"
SAM_CHECKPOINT = r"E:\Models\sam3.pt"
CONFIG["ONLY_SESSIONS"] = ["<your weed-only session ids>"]   # IMPORTANT
CONFIG["LIMIT_PER_SESSION"] = 20                              # trial first
```
```bash
python seeweed3d/annotation/prelabel_weeds_sam3.py
```

**If a session is weed-only for only part of its length** — the drive passes out
of the weedy stretch and into clean onion rows — restrict this run to the weed
half rather than skipping the session or sending it to the mixed prelabeler:

```python
CONFIG["ONLY_FRAMES"] = {"vid3_20260108_103135": ["0-1200"]}
```

Tokens are the same ones curation's `MANUAL_DROPS` takes: `1187`, `0-250`,
open-ended `1500-` or `-250`, or a filename pasted out of `preview/` (`.jpg`
accepted). It is applied *before* `LIMIT_PER_SESSION`, so a 20-frame trial
samples the stretch you named. Leave a gap at the transition —
`vegetation == weed` is exactly the assumption that turns an onion into a target
there. Full procedure: RUNBOOK §3c.

While running, each session shows a live progress line with rate and ETA:

```
  [weed1_20260108_143022] 128/400 frames  32.0% | 3.49 frames/s | elapsed 00:37 | ETA 01:18 | 1794 instances, 2 flagged
```

and finishes with a recall readout:

```
  [weed1_20260108_143022] 400 frames | 5612 weed instances | 2 flagged | 0 with no instances
      provisional classes: grass_weed=812 weed_cluster=61 other_weed=4739
      recall: 91.4% of vegetation inside an exported instance | 5612 from SAM + 0 recovered
      LEP: median confidence 0.74 | median channel agreement 3.1px | visible=5401 ...
```

When output is redirected to a file it switches to occasional whole lines
instead of a self-updating one, so logs stay readable.

Output under `DATASET_ROOT/auto_labels_weeds/<session_id>/`:

| Item | Purpose |
|---|---|
| `cvat_ready/` | **upload this folder to CVAT** — matches `instances_default.json` exactly |
| `instances_default.json` | COCO instance segmentation, one category per morphology class |
| `weed_cvat_labels.json` | label schema (all ontology classes + `weed_LEP` points + `ignore_region`) |
| `instances.csv` | per-instance class, confidence, `source` (`sam`/`vegetation`), growth stage, **all three treatment points**, and shape descriptors |
| `crops/` | per-instance image crops — the input for DINO cluster-then-label |
| `flagged_rgb/` + `flagged_for_manual.txt` | colour-cast/glare frames, no auto-labels, separate manual task |
| `masks/`, `preview/` | union mask and overlay (instance outlines, class, LEP dot; white halo = recovered) |

## Into CVAT

1. Task from `auto_labels_weeds/<sid>/cvat_ready/`.
2. Paste `weed_cvat_labels.json` into the **Raw** label editor.
3. Import `instances_default.json` as **COCO 1.0**.
4. Correct the **class** of each instance and place/drag the **`weed_LEP`**
   point (`weed_LEP`) at the centre of the youngest emerging tissue. Set `lep_visibility` and
   `targetable`; use `weed_cluster` where individual weeds cannot be
   separated, and do **not** place an LEP where it is not visually identifiable.
5. Export as **COCO 1.0**.

## Refreshing labels without re-running SAM 3

The label schema (`weed_cvat_labels.json`) is pure data derived from
`common/ontology.py` - it never depends on per-frame inference. If the
ontology changes after you've already run a full (multi-hour) prelabeling
pass, don't re-run it just to get a fresh label file:

```bash
python seeweed3d/annotation/regen_cvat_labels.py
```

Rewrites `weed_cvat_labels.json` (and `onion_cvat_labels.json`, if that tree
exists) inside every **already-processed** session folder, in seconds.
`instances_default.json`, `masks/`, `preview/` and `cvat_ready/` are never
touched, and folders with no `instances_default.json` (never prelabeled) are
skipped rather than created.

## Scaling: cluster-then-label (next stage)

Correcting the class of thousands of instances one by one is the bottleneck.
`crops/` exists so the next stage can embed every instance (DINOv2), cluster
them, and let you assign a class **per cluster** (~20 decisions instead of
thousands), with outliers flagged for individual attention. Then train an
instance-segmentation model on the verified set, pseudo-label the rest, and
re-verify only the low-confidence and disagreeing cases (plan §13/§14).
