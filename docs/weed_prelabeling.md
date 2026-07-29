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
4. **Recall backstop**: vegetation no instance claimed is recovered as extra
   instances, so a plant SAM missed still reaches the annotator (see below).
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

## Recall: never silently drop a weed

**A weed the prelabeler misses is worse than a weed it labels badly.** A bad
label gets corrected in CVAT in seconds. A missing one is invisible — it enters
the training target as *background*, actively teaching the model that such
plants are not weeds. That error then survives every later stage.

SAM only reports what it detects, so anything it does not return was previously
dropped without trace. Two changes close that gap.

### 1. Prompt SAM on more plants

Exemplar boxes are all submitted in **one forward pass**, so prompting with more
of them costs almost nothing. The old caps (`EXEMPLAR_MAX_BOXES` 30,
`EXEMPLAR_MIN_AREA_PX` 300) meant that in a dense frame only the largest ~30
blobs ever became exemplars; a small plant unlike any of them had no reason to
be returned. Now **60 boxes down to 200 px**.

### 2. The backstop (`RECOVER_MISSED_PLANTS`)

In a weed-only scene `vegetation == weed` — the same prior the onion path
already relies on. So any substantial connected vegetation component that **no
instance claims** is recovered and becomes an instance in its own right.

The recovered mask is then put through *exactly the same* boundary refinement
and touching-plant splitting as a detected one, so it is indistinguishable in
the output except for its recorded `source`.

The one thing this must not do is fabricate duplicates. A leaf tip poking past
the edge of an otherwise-good detection is also unclaimed vegetation, and is
easily larger than `MIN_INSTANCE_AREA_PX`. The guard is therefore not the size
of the residual but **how much of the whole vegetation blob it belongs to is
already claimed** (`RECOVER_MAX_CLAIMED_FRAC`, default 0.30). A clipped leaf tip
sits on a blob that is ~85% claimed and is rejected; a plant SAM never saw sits
on a blob that is 0% claimed and is recovered. `RECOVER_COVERED_DILATE_PX`
absorbs the few pixels of disagreement between a SAM edge and the vegetation
prior so a thin rim never registers as a plant.

### 3. Confidence, not just area (`EXEMPLAR_MIN_VEG_SCORE` / `RECOVER_MIN_VEG_SCORE`)

Both mechanisms above only had an **area** floor, and area cannot tell a small
real plant from a same-sized patch of noise. In the field this showed up as
dozens of phantom point detections scattered across bare, pale, mottled
ground - because the vegetation prior is a plain colour index (ExG +
saturation + green dominance), and colour indices are a documented weak spot
against green-tinted mineral flecks, lichen, and shadow that reads
cooler/greener than sunlit ground. That failure mode is invisible to an area
check: the false-positive component clears `MIN_INSTANCE_AREA_PX` exactly
like a real seedling would.

The fix asks a different question - not "how big" but "how sure" - using
`vegetation_score()`, the same continuous signal boundary refinement already
uses, instead of the binary yes/no prior. A component must clear a **mean
score**, not merely cross the threshold once:

- `EXEMPLAR_MIN_VEG_SCORE` (default **0.85**) gates which vegetation
  components become SAM exemplars. This matters beyond just that one box: SAM
  3 concept segmentation is asked to find *every instance of the same concept*
  across the whole frame, so a single noise exemplar can seed a search for
  more false positives elsewhere, not just fail at its own location. SAM's own
  confirmation is still the primary defence here - this only keeps the worst
  candidates out of the prompt set.
- `RECOVER_MIN_VEG_SCORE` (default **0.9**, stricter) gates
  `recover_missed_plants()`, which has **no SAM corroboration at all** and is
  therefore the most exposed path.

Measured on real plant colour, even degraded (partial shade, a pale young
cotyledon, a leaf edge at a grazing angle): mean score **≥ 0.985**, comfortably
clear of both floors. Measured on a synthetic adversarial gravel/lichen/shadow
texture built specifically to probe this: components scored up to 0.971 - so no
threshold cleanly separates the worst case, color evidence alone is
fundamentally ambiguous some of the time - but the defaults still cut that
texture's exemplar candidates by ~40% and recovered instances by ~62%, at zero
measured cost to a properly-sized real plant in the same tests.

If phantom detections persist on a particular field's substrate even after
this, `RECOVER_MISSED_PLANTS: False` remains the full escape hatch, trading
recall back for precision.

### Seeing it work

Every session now prints a recall line:

```
      recall: 98.7% of vegetation inside an exported instance | 1642 from SAM + 152 recovered
```

Watch **that** number rather than the instance count — a run can look productive
while missing half the plants. In the previews, a recovered instance is drawn
with a **white halo** around its class-coloured outline, so you can judge from
the images alone whether the backstop is earning its keep. `instances.csv` has a
`source` column (`sam` / `vegetation`) so the two populations stay separable
when auditing or weighting the data.

If this percentage drops on a noisy/textured session after upgrading, that is
expected and is the metric becoming *more honest*, not new missed plants: it was
previously possible for the backstop to recover a false-positive vegetation
component and then get credited for "covering" the very noise it fabricated.
With the confidence gate, that noise is excluded from both the numerator and
the count of recovered instances - the percentage now reflects real plant
coverage, not the pipeline grading its own hallucinations.

Set `RECOVER_MISSED_PLANTS: False` to turn it off (for a pure SAM ablation).

## Boundary quality

Four things determine how good the exported boundaries are as a *training
target*, and all four are now handled explicitly.

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

- **All parts are exported.** Previously only the largest contour survived, so
  any tissue separated by an occluding leaf was silently dropped from the
  training target. COCO's `segmentation` is a list precisely so a multi-part
  instance can be represented.
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

While running, each session shows a live progress line with rate and ETA:

```
  [weed1_20260108_143022] 128/400 frames  32.0% | 3.49 frames/s | elapsed 00:37 | ETA 01:18 | 1794 instances, 2 flagged
```

and finishes with a recall readout:

```
  [weed1_20260108_143022] 400 frames | 5612 weed instances | 2 flagged | 0 with no instances
      provisional classes: grass_weed=812 weed_cluster=61 other_weed=4739
      recall: 98.7% of vegetation inside an exported instance | 5160 from SAM + 452 recovered
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
