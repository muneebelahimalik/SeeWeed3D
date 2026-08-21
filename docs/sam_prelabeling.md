# SAM 3 prelabeling: what was built, and what was measured

The complete record of the prelabeling pipeline — every technique, its config
key, its current value, and the evidence behind it.

**Related:** [mixed prelabeling](mixed_prelabeling.md) ·
[weed active learning](weed_active_learning.md) ·
[dataset assembly](dataset_assembly.md) · [CHANGELOG](../CHANGELOG.md)

---

## The goal, stated once

**An annotator corrects masks instead of drawing them.**

Everything below follows from that. A prelabel that is *nearly* right saves most
of the work; a prelabel that is confidently wrong costs more than a blank frame,
because deleting is slower than drawing and accepting is faster than either.

That asymmetry is why so much of this document is about things that were built,
measured, and then **turned off**.

> **Prelabels become the training target, and the training target is the ceiling
> on every model trained from it.** A systematic bias in the prelabeler is
> reproduced by the model, not averaged out. This is the single idea that
> decides every trade-off here.

---

## The pipeline

```
frame
  │
  ├─ 1. gray-world white balance            (only if the frame is cast)
  │
  ├─ 2. vegetation prior  ExG + green dominance + saturation  ──► veg
  │
  ├─ 3. veg blobs ──► exemplar boxes ──► SAM 3 concept segmentation
  │                                          ──► instance masks
  │
  ├─ 4. filter: area, veg overlap, mask NMS
  │
  ├─ 5. per instance: edge snap (small only) ──► shape descriptors
  │                   ──► morphology class ──► growth stage
  │                   ──► 3 candidate treatment points
  │
  └─ 6. export: COCO polygons + CVAT label schema + instances.csv
                + per-instance crops + previews
```

Three prelabelers share this skeleton and differ in step 3–5:

| module | scene | what it optimises |
|---|---|---|
| `prelabel_onions_sam3.py` | onion-only | one high-recall **semantic** safety mask — coverage over separation |
| `prelabel_weeds_sam3.py` | weed-only | every plant **separated**, classified, with a growth point |
| `prelabel_mixed_sam3.py` | mixed | instance identity where crop and weed touch |

`prelabel_mixed_sam3.py` imports `sam3_instances` and `load_sam3` from the weed
module, so backend fixes reach all three.

---

## 1 · Making SAM 3 work on this imagery at all

Four defects, each found only by running on real frames.

### Text prompts return ~0 masks (#6)

`SAM_TEXT_PROMPTS` — `["plant", "weed", "green plant"]` — do not ground on
top-down field imagery. The output collapsed to the Excess-Green prior alone,
which masks whole frames wherever the soil has a green cast.

**Fix: exemplar prompting (#7).** Vegetation blobs become boxes, boxes become
positive exemplars, SAM 3 returns every instance of "the same concept". This is
still the default prompt mode in all three modules.

```python
"SAM_PROMPT_MODE": "auto_exemplar",   # auto_exemplar | text | manual
```

### bf16 activations against fp32 weights (#4)

SAM 3's vision backbone raises a dtype error outside an autocast context.
Inference runs under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.

### Mask nesting varies (#5)

Real SAM output tripped an OpenCV resize assertion. `_state_masks()` normalises
whatever nesting SAM returns — `[N,1,H,W]`, `[N,H,W]`, `[H,W]` — into a list of
2-D bool masks.

### The threshold nobody chose (#105)

Extraction ended with `arr.astype(bool)`, which is `True` for **every non-zero
value**. Exact on a binary mask; on a probability or logit mask it thresholds at
0.0 instead of 0.5 and every mask inflates outward through its own soft edge.

`common/masks.py` now classifies the array by its **values** and reports once
per run:

| array holds | read as | cut at |
|---|---|---|
| `bool`, 0/1 int, 0/255 int | binary | untouched |
| float in [0, 1] | probability | **0.5** |
| float with negatives | logit | **0.0** |
| float ≥ 0 with max > 1 | **ambiguous** | old behaviour, flagged loudly |

**Measured on this data: `dtype=bool range=[0,1]` → BINARY.** The hypothesis was
wrong and the guard stays, because an assumption that is pinned costs nothing
and an assumption that is assumed costs a dataset.

---

## 2 · The vegetation prior

ExG alone masks bare soil whenever a frame has a green cast. Two extra gates fix
it: a pixel is vegetation only if green **dominates** both other channels *and*
it is saturated enough — real leaves are saturated green, cast soil is not.

| key | value | why |
|---|---|---|
| `EXG_THRESHOLD` | `0.05` | excess-green cut |
| `VEG_MIN_SATURATION` | `40` | HSV S; rejects desaturated cast soil |
| `VEG_MORPH_KERNEL` | `3` | close/open tidy |
| `VEG_MIN_COMPONENT_PX` | `150` | drop specks |
| `MAX_MASK_FRACTION` | `0.5` | veg above this ⇒ colour-cast/glare failure, frame flagged |

ExG is computed on **chromaticity** (`2g/s − r/s − b/s`), so it is largely
shadow-invariant. Shadow's real damage is not darkness — it is noisy
chromaticity on dark soil producing *false* vegetation.

### Colour-cast recovery (#8)

Some ZED frames have a severe green white-balance error and were being flagged
and lost entirely. The structure underneath is intact, so a gray-world white
balance recovers them — applied **only** when a frame is actually cast:

```python
"WHITE_BALANCE": True,
"WB_CAST_RATIO": 1.15,    # apply only if max/min channel-mean exceeds this
```

Soil-dominant frames are barely touched; green-cast frames are recovered so both
ExG and SAM work on them.

---

## 3 · Exemplar quality is a safety property

SAM 3 concept segmentation searches the **whole frame** for more of "that
concept". So a single low-quality exemplar — gravel, lichen, a shadowed pit that
only marginally passes the vegetation prior — does not risk one bad box. **It
teaches SAM a bad concept**, and dozens of phantom detections appear on bare
ground that resembles it.

This was observed on a real pale, textured field surface (#24).

| key | value | note |
|---|---|---|
| `EXEMPLAR_MIN_AREA_PX` | `300` | briefly 200 — reverted (#25) |
| `EXEMPLAR_MAX_BOXES` | `30` | briefly 60 — reverted (#25) |
| `EXEMPLAR_PAD_PX` | `8` | so thin leaf tips are included |
| `EXEMPLAR_MIN_VEG_SCORE` | `0.85` | confidence gate (#24) |
| `SAM_CONF` | `0.25` | SAM's own detection threshold |

**The 0.85 figure is measured, not chosen.** Real plant colour — even degraded
by partial shade, a pale young cotyledon, or a grazing-angle leaf edge — scores
**≥ 0.985**. A synthetic adversarial gravel/lichen/shadow texture built to probe
this had components scoring up to **0.971**.

So no threshold cleanly separates the worst case. 0.85 keeps a real margin below
the real-plant floor while cutting the adversarial case's exemplar candidates by
**~40%**. SAM's own concept confirmation remains the primary defence; this is a
supplementary precaution.

---

## 4 · Instance filtering

| key | value | why |
|---|---|---|
| `MIN_INSTANCE_AREA_PX` | `250` | ~16×16 px at 2208×1242 |
| `MAX_INSTANCE_FRAC` | `0.25` | one weed covering >25% of frame = failure |
| `INSTANCE_VEG_OVERLAP_MIN` | `0.35` | instance must sit on vegetation |
| `NMS_IOU` | `0.65` | de-duplicate overlapping SAM instances |

**`MIN_INSTANCE_AREA_PX` is the clearest single lesson in this pipeline (#16).**
It was raised to 700 on the reading that the many small detections in dense
frames were speck over-detection. Checking the imagery showed **they were real
cotyledon-stage weeds** — 700 silently deleted genuine plants.

Reverted to 250. For a laser weeder, a missed small weed is a worse failure than
an extra instance the annotator deletes in one click.

> Raise it only if previews show detections on **bare soil**, not on seedlings.

---

## 5 · What shape can and cannot decide (#13)

| question | shape can answer? | evidence |
|---|---|---|
| grass vs rosette | **yes** | aspect ratio: blade ≈ 20, rosette ≈ 1 |
| intermingled cluster | **yes** | multiple growth-point peaks in one mask |
| *species* | **no** | both named species are rosettes |

Circularity **cannot** separate grass from rosette: a rosette's radiating leaves
give it a spiky perimeter (≈ 0.15), as low as a blade's (≈ 0.13). Solidity
cannot flag grass either — a straight blade is nearly its own convex hull
(≈ 0.9). Elongation is the only descriptor that separates them cleanly.

```python
"GRASS_MIN_ASPECT": 3.0,
"DEFAULT_SPECIES_CLASS": "other_weed",   # confidence 0.0 — "you decide"
```

**Species are never auto-assigned.** `cutleaf_evening_primrose` vs
`wild_radish` is an appearance question, not a shape one. On a real run this
means ~92% of instances arrive as `other_weed` with zero confidence — which is
the classifier being honest, not failing.

### Cluster detection is deliberately hard to trigger

```python
"CLUSTER_MIN_PEAKS": 3,          # distinct growth-point peaks in one mask
"CLUSTER_MIN_AREA_PX": 20000,    # AND it must be large
"PEAK_REL_THRESHOLD": 0.5,       # peak counts if >= this fraction of max radius
"PEAK_MIN_SEPARATION_PX": 15,
```

`weed_cluster` means *"no separable single LEP"* — every instance carrying it is
a plant that never gets targeted individually.

**The rate is now reported, because it is a policy number (#106):**

| source | clusters | instances | share |
|---|---|---|---|
| prelabel `vid3_..._103135` | 800 | 11,816 | **6.8%** |
| prelabel `vid3_..._110444` | 380 | 9,592 | **4.0%** |
| hand-corrected `weeds_v2` | 2 | 1,444 | 0.1% |

A prelabel arriving already marked as a cluster biases the annotator toward
accepting it, so the rate proposed here becomes the rate the dataset carries,
which becomes the rate the model predicts at runtime. The run warns above 2%.

---

## 6 · Three treatment points, stored for free

Per instance the pipeline stores all three candidates:

| field | what it is |
|---|---|
| `lep_dt` | **distance-transform peak** — deepest interior point. For a rosette this *is* the crown |
| `centroid` | mask centroid — project-plan baseline |
| `bbox_ctr` | bounding-box centre — project-plan baseline |

Storing all three gives the LEP-method comparison for free once human LEP labels
exist, with no re-run.

### The multi-evidence LEP estimator (#14) — built, and off

`perception/lep.py` fuses petiole convergence, radial isotropy, young-tissue
chromatics, canopy height and medial-axis interiority. It exists because the LEP
is the research target and has to be defensible as the apical meristem rather
than a shape summary that happens to correlate with it — a centroid drifts off
the meristem on an asymmetric plant with no way to notice.

```python
"USE_FUSED_LEP": False,
```

**Off by default**, for three reasons: the plain DT peak lands at the crown for
a rosette and has no dependence on boundary detail; the fused estimator's
skeleton channel is sensitive to exactly the boundary noise that the disabled
post-processing used to clean up, so running it on unrefined masks is not the
configuration it was tuned for; and it is by far the most expensive per-instance
step. Turn it on when the LEP itself is the object of study.

A related saving: depth feeds the canopy-height channel and **nothing else**, so
it is read only when that estimator runs. It used to decode a 16-bit depth PNG
per frame and discard it.

---

## 7 · Boundary quality — the sequence worth reading in order

This is the clearest case in the project of a metric-driven improvement being
wrong, and it is why the defaults look conservative.

| PR | what | outcome |
|---|---|---|
| #21 | edge snapping, plant splitting, multi-part polygons | added on the reasoning that prelabel boundary quality becomes the training target's quality |
| #22 | recall backstop — unclaimed vegetation becomes an instance | added: a missed weed is invisible and enters the target as background |
| #23 | anti-aliasing — per-pixel edges leave staircase jaggies no leaf margin has | added |
| #24 | **dozens of phantom detections on bare, pale, mottled ground** | confidence gating added |
| #25 | still not clean | exemplar loosening reverted; recall backstop **disabled by default** |
| #29 | **restored the #11/#12 mask profile wholesale** | field comparison judged those masks best |

**#29's named culprit was `split_touching_instances`** — a leaf reaching away
from the crown raises a second distance-transform peak, and that was enough to
split one plant into several annotations, each with its own outline, class and
LEP dot.

The post-#12 code is all still present and still tested. It is simply off by
default, **each entry independently re-enablable** — which matters, because one
of them has since been turned back on.

### Current state of the post-#12 block

```python
"SPLIT_TOUCHING_INSTANCES": False,     # the #29 culprit — never drift this
"BOUNDARY_SMOOTH_SIGMA_PX": 0.0,       # anti-aliasing, off
"USE_FUSED_LEP": False,
"POLY_ALL_PARTS": False,
"RECOVER_MISSED_PLANTS": False,
"USE_DEPTH_HEIGHT": False,

"BOUNDARY_REFINE_BAND_PX": 2,          # ← ON, deliberately
"BOUNDARY_REFINE_VEG_MIN": 0.5,
"BOUNDARY_REFINE_MAX_AREA_PX": 1500,   # ← and gated
```

---

## 8 · Size-gated edge snapping (#107)

**The observation:** big weeds already have correct boundaries and need no
correction; small ones do not.

That asymmetry is the diagnosis, and it rules out most candidate causes — if SAM
were including inter-leaf soil, big rosettes would be the *worst* case.

It is a **pixels-per-plant** problem. SAM decodes masks on a fixed grid spread
over the whole frame, so a rosette gets many cells across it and a seedling gets
a handful. A fixed-size boundary error is then a large *fraction* of a small
plant:

| plant | area px | error from a 2 px overshoot |
|---|---|---|
| 20 px | 400 | **44%** |
| 30 px | 900 | 28% |
| 60 px | 3,600 | 14% |
| 120 px | 14,400 | 7% |

`refine_boundary()` re-decides a ±2 px ring around each mask using the
**continuous** vegetation score:

> SAM decides *what* the object is; the image decides exactly *where it ends*.

It never eats the interior, and added pixels must stay connected to the original
core so refinement cannot bridge to a neighbouring plant.

**`BOUNDARY_REFINE_MAX_AREA_PX = 1500` skips bigger instances byte-for-byte.**
1500 px is the project's own definition of a small weed in `eval_seg.py`, so the
prelabeler and the metric agree on what "small" means. Snapping on *without* a
gate is free rein over the boundaries field judgement says are already right —
which is #29 exactly — so a test pins the pair rather than either value.

---

## 9 · Polygon export

```python
"POLY_APPROX_EPS_FRAC": 0.010,   # Douglas-Peucker tolerance = 0.010 * sqrt(area)
"POLY_APPROX_EPS_MIN": 0.5,
"POLY_APPROX_EPS_MAX": 1.5,
"POLY_MIN_PART_AREA_PX": 60,
"POLY_ALL_PARTS": False,
```

Tolerance **scales with instance size** for roughly constant relative fidelity —
a fixed value erases shape on seedlings and bloats vertex counts on rosettes.

Two things worth knowing:

- The clamp is mildly **size-biased against small weeds**. A 400 px weed hits the
  0.5 floor — 2.5% of its 20 px extent; a 14,400 px weed hits the 1.5 cap —
  0.75% of its extent. Simplification is ~3× more aggressive, relatively, on the
  instances that can least afford it.
- `POLY_ALL_PARTS: False` matches #12 and exports only the largest contour. It
  **silently drops tissue** whenever a leaf is separated from the crown by an
  occluding leaf, and that tissue enters the training target as background. If
  previews look clean but plants have visibly missing leaves, set it `True`.

---

## 10 · Export and the CVAT round trip

Per session the run writes:

| output | purpose |
|---|---|
| `cvat_ready/` | exactly the frames with a usable prelabel — upload this folder |
| `instances_default.json` | COCO, covering exactly those frames |
| `weed_cvat_labels.json` | label schema for CVAT's Raw editor |
| `instances.csv` | per-instance descriptors, all three treatment points, LEP columns |
| `crops/` | per-instance crops, ready for DINO cluster-then-label |
| `preview/` | overlays for judging the run |
| `masks/` | the exported union per frame |
| `flagged_for_manual.txt` | frames blanked by the `MAX_MASK_FRACTION` cap |

**Flagged frames are kept in a separate folder (#10)** — a blank-mask frame mixed
into the COCO could mismatch the CVAT upload.

### Four schema failures, all silent in CVAT

| PR | failure |
|---|---|
| #15 | class names had drifted per-module (`brassica`, `primrose`, names with spaces). `common/ontology.py` became the single source of truth, with **stable category ids** so weed, onion and mixed datasets merge without remapping an annotation |
| #19 | CVAT's Raw editor requires a non-empty `values` array for **every** attribute, including text ones. Two shipped without it and CVAT rejected the whole schema |
| #39 | the onion schema used type `mask`, which CVAT exports as RLE needing pycocotools. Changed to `polygon` to match the working weed round trip |
| #40 | a run from before the snake_case rename wrote `"onion plant"` as a category name. CVAT matches by **name**, so this silently creates a duplicate label instead of filling the one the prelabels were meant to correct. `fix_coco_categories.py` repairs it, and refuses to write rather than guess at an unknown name |

**Schema refresh without re-running SAM 3 (#20).** The label schema is pure data
derived from the ontology, so a rename should not cost a multi-hour inference
pass. `regen_cvat_labels.py` rebuilds it in seconds.

**Round-trip measurement (#9).** Verified export → rasterised training masks →
auto-vs-verified IoU, so prelabel quality is measured rather than assumed.

**`instances.csv` schema fix (#17).** Rows do not all share a key set — a
`weed_cluster` carries no `lep_*` columns — so taking the header from row 0
crashed as soon as the first instance was a cluster. The header is the union of
every row's keys.

---

## 11 · What the run tells you about itself

Reporting is a feature here, not decoration: every number below exists because
its absence hid a real problem.

```
[i] SAM 3 (weeds) raw masks: shape=(78, 1242, 2208) dtype=bool range=[0, 1]
    -> read as BINARY (thresholding is exact; nothing to decide).
[vid3_20260108_103135] 391 frames | 11816 weed instances | 0 flagged | 0 empty
    provisional classes: grass_weed=169 weed_cluster=800 other_weed=10847
    weed_cluster: 800 of 11816 instances (6.8%)
[!] That is a high cluster rate. Each one is a plant that gets no individual LEP…
    recall: 93.3% of vegetation inside an exported instance | 11816 from SAM + 0 recovered
[!] 9 frame(s) in pool.csv could not be read from rgb/ and were skipped: …
```

| readout | why it exists |
|---|---|
| **mask encoding** (#105) | the threshold was an assumption nobody had checked |
| **vegetation recall** | in a weed-only scene vegetation *is* plants, so veg left outside every instance is weeds the annotator never sees. A pipeline can look productive while quietly missing plants |
| **SAM vs recovered split** | keeps the two populations separable when auditing |
| **cluster share** (#106) | a policy number — see §5 |
| **unreadable frames** (#106) | 391 of a 400-frame pool, with nothing saying why. *A pipeline that skips input silently is worse than one that crashes* |
| **edge-snapping delta** (#107) | how many masks moved and by how much — reports the *size* of the change, never that it was good |
| **progress** (#18) | long runs gave no feedback until a session finished |

---

## 12 · Run control

```python
"LIMIT_PER_SESSION": None,     # start small; None for the full pool
"ONLY_SESSIONS": [],           # empty = every session under sessions/
"ONLY_FRAMES": {},             # index ranges — see below
"SAVE_PREVIEWS": True,
"PREVIEW_SCALE": 0.5,
"PREVIEW_OUTLINE_BGR": (0, 0, 255),
```

### Splitting one drive between prelabelers

A single drive can pass from one crop zone into another — weeds for the first
stretch, then onions. That is **not** a mixed scene; it is two single-class
scenes end to end, and the specialised prelabelers give correct classes for free
on each stretch, which the mixed one cannot.

```python
"ONLY_FRAMES": {"vid1_20250221_131902": ["0-1200"]}   # inclusive range
                                        # ["1500-"]   # open-ended
```

> **Leave a gap at the transition.** The frames where one zone becomes the other
> are exactly where a single-class assumption is most dangerous, and calling an
> onion a weed is the worst error this project can make.

### Preview outlines are one colour on purpose

The per-class palette is the wrong tool for judging a boundary: nearly every
instance comes out as `other_weed`, whose ontology colour is a mid grey that
vanishes against dry soil, stubble and shadow — so the thing you are trying to
inspect is the thing you cannot see. Red belongs to neither plant nor soil
anywhere in this imagery. Nothing here reaches CVAT.

---

## 13 · The mixed-scene path is a different algorithm

Where crop and weed touch, neither signal is sufficient alone:

- **The vegetation prior owns the boundary.** It is per-pixel and exact, with no
  model. It cannot say which green pixel belongs to *which* plant — it sees one
  blob where two rosettes touch.
- **SAM 3 owns the identity.** Concept segmentation returns instances. But its
  boundaries are a learned prior at its own working resolution: it clips leaf
  tips, rounds dissected margins, and bleeds onto soil.

Intersecting is not enough — a clipped tip is already gone before the
intersection happens. So:

1. Vegetation prior → `veg`, the set of plant pixels. **Boundary, done.**
2. SAM 3 → proposals, each reduced to a **seed**: its overlap with `veg`, eroded
   to its confident interior. The seed asserts *"a distinct plant is here"* and
   nothing about where it ends.
3. **Marker-controlled watershed** over `veg`, seeded by those markers, flooding
   the inverted distance transform.

Every vegetation pixel is assigned to exactly one seed, and surfaces meet at the
**neck** between plants. A seed that undershot grows out to the true leaf edge;
a proposal that bled onto soil is cut back to `veg` by construction; two plants
sharing one blob are cut where the blob is thinnest — which is where they
actually touch.

### Why splitting is safe here when it was not for weeds

`SPLIT_TOUCHING_INSTANCES` fragmented single plants because it split on
distance-transform **peaks**, and a leaf reaching away from the crown raises a
second peak.

Nothing on the mixed path splits on peaks. The markers come from **SAM**, so a
blob is divided only where SAM saw two distinct plants — and a leaf reaching
away from a crown is not a second SAM detection. The watershed decides only
*where* the cut falls, never *whether* there is one. The failure mode that
closed that door does not exist on this path.

### Onion's glaucous leaf defeats the colour prior

Onion's leaf is a glossy, glaucous (waxy blue-green) tube, and that surface
defeats the prior in two ways at once:

1. **The wax bloom shifts colour toward blue.** `vegetation_mask()` requires
   green to dominate red *and* blue at every pixel. A matte broadleaf weed clears
   that easily; onion's coating measurably lifts blue reflectance, so stretches
   of a real leaf fail `g ≥ b` outright.
2. **The glossy curve throws specular highlights.** A highlight has no leaf
   colour to detect — it *is* the light source's colour — so it fails ExG, green
   dominance and saturation simultaneously. No threshold tuning admits it,
   because there is no green signal there at all.

Both cut **straight across** a leaf's width, so the gap touches soil on both
sides — and `fill_holes()` deliberately fills only holes fully *enclosed* by
vegetation, so a genuine gap between two adjacent leaves stays soil. One onion
leaf can emerge broken into a dozen slivers, and every downstream step inherits
the damage.

`recover_glaucous_pixels()` relaxes the green-dominance and saturation gates
**only within a small halo around pixels the strict gate already accepted** — so
bare pale soil earns nothing, while a highlight sitting inside a confirmed leaf
does. It does not touch shared `vegetation.py`, because the onion and weed
prelabelers are tuned against its current behaviour.

---

## 14 · Measured throughput

| metric | value |
|---|---|
| speed | 0.16–0.17 frames/s (~6 s/frame, RTX 4090) |
| one 400-frame session | ~40 minutes |
| vegetation recall | **93.3%** and **94.9%** across two sessions |
| instances found | 11,816 and 9,592 |
| instances/frame | ~30 and ~24 |
| recovered by backstop | 0 (disabled) |

---

## The three lessons this pipeline paid for

1. **Check the imagery before believing the metric.** Small detections read as
   noise and were real cotyledons (#16). A boundary pipeline that improved every
   number produced worse masks in the field (#29).

2. **A silent wrong answer beats a loud one — so make it loud.** A duplicate
   CVAT label, an empty crop mask, a skipped frame, a threshold nobody chose:
   none of these raise. Most of the guards here exist because one of them
   didn't.

3. **Annotation policy becomes deployed policy.** A cluster class used when
   separation is merely tedious teaches the model to do the same at runtime,
   where it means weeds that are never targeted. Systematic label bias is
   reproduced, not averaged out.
