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
4. Per instance: shape descriptors → provisional morphology class, growth-stage
   estimate, and **three candidate treatment points**.
5. Export COCO (categories = morphology classes) + per-instance CSV + instance
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

When output is redirected to a file it switches to occasional whole lines
instead of a self-updating one, so logs stay readable.

Output under `DATASET_ROOT/auto_labels_weeds/<session_id>/`:

| Item | Purpose |
|---|---|
| `cvat_ready/` | **upload this folder to CVAT** — matches `instances_default.json` exactly |
| `instances_default.json` | COCO instance segmentation, one category per morphology class |
| `weed_cvat_labels.json` | label schema (all ontology classes + `weed_LEP` points + `ignore_region`) |
| `instances.csv` | per-instance class, confidence, growth stage, **all three treatment points**, and shape descriptors |
| `crops/` | per-instance image crops — the input for DINO cluster-then-label |
| `flagged_rgb/` + `flagged_for_manual.txt` | colour-cast/glare frames, no auto-labels, separate manual task |
| `masks/`, `preview/` | union mask and overlay (instance outlines, class, LEP dot) |

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
