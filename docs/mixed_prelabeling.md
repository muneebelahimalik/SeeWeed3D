# Mixed-scene prelabeling — precise masks, classes assigned by hand

For building a dataset of scenes that contain **both onions and weeds**, which
is what the field actually looks like and what neither existing prelabeler can
be pointed at.

`seeweed3d/annotation/prelabel_mixed_sam3.py` has exactly one job: **one precise
mask per plant**. It emits a single class, `plant`, and you assign the real
classes in CVAT with one keystroke per shape.

---

## Why a separate module rather than a flag on the weed prelabeler

The two prelabelers already in the repo both rest on an assumption that a mixed
scene breaks:

| Module | Assumption | In a mixed scene |
|---|---|---|
| `prelabel_weeds_sam3.py` | vegetation == weed | every onion becomes a weed |
| `prelabel_onions_sam3.py` | vegetation == onion | every weed becomes crop |

Turning that off is not a flag, it is a different pipeline: it removes the class
proposal entirely and moves all of the effort onto separation and boundaries.

### What "mixed" means here — and what it does not

Mixed means **both classes in the same frame**. That is the only condition that
makes the class proposal unsafe, and therefore the only condition that justifies
paying for this module's cost: you give up a free correct class and buy it back
with one keystroke per shape in CVAT.

A drive that is weed-only for its first stretch and onion-only after it does
**not** meet that condition. Every one of its frames still contains exactly one
class — the class just changes partway through the session. Sent through this
module, such a session would throw away a class label that was available for
free on both halves, and hand the annotator hundreds of reassignments that the
data itself already answered.

Split it by frame index instead and run the prelabeler whose assumption holds on
each stretch. Every prelabeler in the repo takes `CONFIG["ONLY_FRAMES"]`, keyed
by session id, using the same tokens curation's `MANUAL_DROPS` accepts:

```python
# prelabel_weeds_sam3.py      # prelabel_onions_sam3.py
CONFIG["ONLY_FRAMES"] = {     CONFIG["ONLY_FRAMES"] = {
    "vid3_20260108_103135": ["0-1200"]}   "vid3_20260108_103135": ["1500-"]}
```

Tokens: a bare index `1187`, an inclusive range `0-250`, an open-ended `1500-`
or `-250`, or a filename pasted straight out of `preview/` — the `.jpg` preview
name is accepted, not just the `.png` source. `ONLY_FRAMES` is applied *before*
`LIMIT_PER_SESSION`, so a 20-frame trial samples the stretch you selected rather
than the first 20 frames of the whole session, which on a split drive would be
entirely the wrong zone.

**This module still has a job on such a session: the transition.** Leave a gap
between the two single-class ranges covering every frame where you cannot tell
from a preview which crop you are looking at, or where both are genuinely in
shot, and give that gap to `prelabel_mixed_sam3.py`:

```python
CONFIG["ONLY_FRAMES"] = {"vid3_20260108_103135": ["1201-1499"]}
```

A single-class prelabeler run over that stretch states a class confidently and
wrongly, and **calling an onion a weed is the worst error this project can
make** — it is the one that ends with a laser on the crop. Excluding the stretch
from this round is also a legitimate answer; prelabelling it under an assumption
you know is broken is not.

Never let two ranges overlap. A frame prelabelled under both assumptions reaches
CVAT twice carrying contradictory classes, and nothing downstream can tell you
which copy you corrected.

## Why one homogeneous class is the right answer, not a shortcut

This was your call and it holds up for three independent reasons:

1. **Shape cannot tell an onion from a grass weed — and would get it backwards.**
   The one morphology call the weed prelabeler makes confidently is
   *elongation → `grass_weed`* (a blade has aspect ~20, a rosette ~1). An onion
   *is* a blade. Auto-classifying in a mixed scene would systematically label
   the crop as a weed, which is the single worst error this project can make.

2. **A plausible wrong label survives review; a blank one does not.** Annotators
   confirm what looks right and re-examine what looks empty. A neutral class
   forces a decision on every shape.

3. **The cheap half is the half being automated the wrong way round.**
   Reassigning a class is one keystroke. Fixing a boundary is a minute of mouse
   work. Spend the pipeline on boundaries.

`plant` deliberately sits **outside** `common/ontology.CLASSES`, with category id
`100`. So it can never be trained on by accident, and any shape still carrying
it is a shape nobody has reviewed — which is a number you can count.

---

## The mask logic: vegetation owns the boundary, SAM owns the identity

These are two different questions and the pipeline is bad at them in opposite
directions.

**Where a plant ends** is nearly free. Top-down field imagery is green tissue on
brown soil; a colour index answers that per pixel, at full resolution, with no
model. What it cannot do is say *which* green pixel belongs to *which* plant — a
colour index sees one blob where two rosettes touch.

**SAM 3 is the reverse.** Concept segmentation returns instances, so identity is
what it is good at. Its boundaries are a learned prior at its own working
resolution: it clips leaf tips, rounds off dissected margins, and bleeds onto
soil. Intersecting is not enough on its own either — a clipped tip is already
gone before the intersection happens.

So each is used only for what it is good at:

```
1. vegetation prior          -> veg          the set of plant pixels
2. SAM 3 proposals           -> seeds        proposal ∩ veg, eroded
3. marker-controlled watershed over veg, seeded by those markers,
   flooding the INVERTED DISTANCE TRANSFORM
```

Step 3 is what produces the masks:

- a seed that **undershot** grows out to the true leaf edge
- a proposal that **bled onto soil** is cut back to `veg` by construction, and
  can never re-acquire soil because the flood runs only inside `veg`
- two plants sharing **one green blob** are cut apart where the blob is
  thinnest, which is where they actually touch

Flooding the *image gradient* instead sounds more principled and is worse:
inside a canopy the strongest gradients are leaf veins, shadow edges and
specular highlights, so the cut chases texture within one plant instead of the
join between two.

### Why this split is safe when the weed prelabeler's was not

`prelabel_weeds_sam3.py` ships `SPLIT_TOUCHING_INSTANCES: False` because
splitting on distance-transform **peaks** fragmented single plants — a leaf
reaching away from the crown raises a second peak, and one rosette became two
annotations.

Nothing here splits on peaks. **The markers come from SAM**, so a blob is
divided only where SAM saw two distinct plants, and a leaf reaching away from a
crown is not a second SAM detection. The watershed decides only *where* the cut
falls, never *whether* there is one. A test pins this: a six-armed rosette with
one seed comes out as one instance.

### Two implementation details that cost real tissue

Both were found by measurement and are guarded by regression tests.

**No background marker.** Seeding the soil as a background label looks like the
careful thing to do and destroys thin tissue: soil is flat ground at the top of
the relief, so a background flood reaches a near-flat arm tip before the flood
coming up the arm from the crown does. On a synthetic six-armed rosette that
marker ate **four of the six tips**. Plant-versus-soil is not the watershed's
question anyway — the prior already answered it, and intersecting with `veg`
enforces it exactly.

**Ridge reclamation.** Watershed marks its boundary pixels `-1`, and OpenCV also
stamps the image border. Left alone that shaves a pixel off every instance —
along the join between two plants, where it belongs, but also off any plant
running to the frame edge, where it does not. Ridge pixels are reclaimed only
where unambiguous: a ridge pixel whose 8-neighbourhood touches exactly **one**
instance is an outer edge and goes to it; one touching **two** is a real join
and stays unassigned. So adjacent instances end up separated by a one-pixel line
and **no pixel belongs to two instances**.

---

## Onion's glaucous leaf, and why it fragments the prior

Real field frames add a third problem, on top of the two above: a single
onion leaf can come out of the vegetation prior broken into a handful of
disconnected slivers, and every step downstream inherits the damage — SAM's
exemplar boxes end up one per fragment instead of one per leaf, the watershed
can only flood within `veg` so it cannot cross the gaps either, and the
fragment-dropping cleanup throws the smaller slivers away as if they were
soil-texture speckle, because from its side of the pipeline that is exactly
what they look like. The result is a scatter of tiny instances instead of one
clean leaf, and onion massively under-represented relative to the weeds
around it.

**The cause is onion's leaf surface, not a bug in the watershed.** Onion is a
glossy, waxy, blue-green tube, and that surface defeats
`vegetation_mask()`'s colour gates two ways at once:

1. **The wax bloom lifts blue reflectance.** The gate requires green to
   dominate both red *and* blue at every pixel. A matte broadleaf weed clears
   that easily; the glaucous coating on onion measurably shifts colour toward
   blue, so stretches of a real leaf fail `g >= b` outright.
2. **The glossy curve throws specular highlights.** A round, wet-looking
   surface catches direct light as a near-white streak. A highlight has no
   *leaf* colour to detect — it **is** the light source's colour — so it fails
   ExG, green-dominance and saturation all at once. No threshold admits it,
   because there is no green signal there to threshold.

Both cut straight across the leaf's *width*, so the gap connects to the
surrounding soil on both sides. `fill_holes()` cannot help: it only fills gaps
fully **enclosed** by vegetation, by design, so a genuine gap between two
adjacent leaves — which is soil, and must stay soil — is never touched. A gap
that severs a thin leaf is indistinguishable from that case by shape alone.

**`recover_glaucous_pixels()`** fixes it without touching the shared
`vegetation.py` — the onion and weed prelabelers are tuned against its current
behaviour, and this project has already reverted one boundary "improvement"
that looked better on paper and was worse in the field (see the
`CHANGELOG.md` phase on boundary quality). It relaxes the green-dominance and
saturation gates, but **only** within a small halo (`VEG_BRIDGE_PX`) around
pixels the strict gate already accepted. A bare patch of pale soil sitting on
its own earns nothing from either relaxed rule, however low its saturation —
only a highlight or blue-shifted stretch sitting *inside* an already-confirmed
leaf does.

`close_thin_gaps()` is a second, purely geometric closing step for whatever
the colour-aware pass still misses — and it ships **off by default**
(`VEG_CLOSE_KERNEL_PX: 0`). Measured against two plants touching at a 4px
overlap (an existing regression scene in the test suite), even a small closing
kernel smooths over the concave *neck* between them exactly the way it smooths
over a gap within one leaf, because a shape-only operator cannot tell the two
apart. The colour-aware bridge has no such failure mode — the soil around a
genuine neck fails its colour test — and measured against a synthetic
afflicted leaf it is sufficient on its own. Only raise `VEG_CLOSE_KERNEL_PX` if
bridging alone still leaves gaps after `EXG_THRESHOLD` and
`VEG_BLUE_TOLERANCE` have been tuned, and re-check a **touching-plants**
preview afterward, not just a single-leaf one.

| Setting | What it does |
|---|---|
| `VEG_BRIDGE_PX` | How far past confirmed vegetation the relaxed gate and the highlight rule may look. `0` disables both |
| `VEG_BRIDGE_EXG_RELAX` | How far below `EXG_THRESHOLD` a bridge-halo pixel may score and still be admitted |
| `VEG_BLUE_TOLERANCE` | How far blue may exceed green in the bridge halo — the knob that exists specifically for onion's wax bloom |
| `VEG_BRIDGE_HIGHLIGHT_SAT` | HSV saturation at or below which a halo pixel is treated as a specular highlight and admitted regardless of hue |
| `VEG_CLOSE_KERNEL_PX` | Off by default. Pure-shape safety net; smooths plant-to-plant necks as readily as leaf gaps |

### Reconnecting the mask is not enough on its own

Two more places assume a clean, unfragmented prior, and missing either one
meant real field sessions still came back with **empty masks and empty
previews on most frames** even after the mask itself was reconnected — sparse
frames and dense, overlapping ones alike.

**The size floor has to run last.** `vegetation_mask()` deletes any component
under `VEG_MIN_COMPONENT_PX` before it ever returns — and on a thin, heavily
afflicted leaf, every *individual* fragment can be smaller than that floor
even though the leaf as a whole is not. Calling it with the real floor already
applied prunes every fragment before bridging ever sees them, and dilating an
already-empty mask stays empty: the whole plant vanishes. `strict_vegetation()`
now calls it with `min_component_px=0`, so bridging gets every surviving speck
to reconnect, and the real floor is applied once, at the very end, to the
**reconnected** result.

**Confidence has to know what was bridged.** The recall backstop and the
exemplar-confidence gate both score a component with `vegetation_score()` —
the exact strict colour rule that fragmented the leaf in the first place. A
bridged highlight or blue-shifted pixel scores near zero on it, so a component
substantially made of bridged pixels can score well under
`RECOVER_MIN_VEG_SCORE` (0.90) or `EXEMPLAR_MIN_VEG_SCORE` (0.85) even though
it is real, reconnected leaf tissue — measured at 0.62–0.71 on a synthetic
afflicted leaf, comfortably real vegetation, comfortably rejected. That fires
on **every** frame where bridging did meaningful work, which in practice is
most onion-containing frames — not a rare edge case. `plant_confidence()`
scores a bridged pixel by the best `vegetation_score()` found within
`VEG_BRIDGE_PX` of it among pixels that passed the strict gate directly — the
same context that already justified admitting the pixel, reused rather than
re-litigated. `analyze_frame()` and the exemplar-prompting step in
`prelabel_session()` both use it in place of the raw score now.

---

## Recall: unclaimed green becomes an instance

A vegetation component containing no seed is a plant SAM missed entirely. Those
are recovered as instances of their own, gated on mean vegetation score
(`RECOVER_MIN_VEG_SCORE`).

The weed prelabeler disables its equivalent backstop. That reasoning does not
carry over: **there**, a recovered instance also had to be given a class from
colour alone, so a green-tinted mineral fleck entered the training set as a
labelled weed. **Here** every instance is `plant` and every instance is reviewed
before it becomes training data, so a phantom costs one keypress to delete while
a missed plant is a plant the annotator is never shown — in a prelabelled task
nobody draws what is not already there.

Read the previews for the first session anyway. If you see instances on bare
ground, raise `RECOVER_MIN_VEG_SCORE` or set `RECOVER_UNSEEDED: False`.

---

## Running it

```powershell
conda activate dl

# edit the top of the file
#   DATASET_ROOT  = r"E:\Dataset_Vidalia\Mixed_1"
#   CONFIG["ONLY_SESSIONS"]     = ["vid1_20260108_101500"]
#   CONFIG["LIMIT_PER_SESSION"] = 20        # trial first, then None
#   CONFIG["ONLY_FRAMES"]       = {}        # {} = whole session; see above for
#                                           # split-zone drives

python seeweed3d/annotation/prelabel_mixed_sam3.py
```

**Do a 20-frame trial and look at `preview/` before committing to a full run.**
Previews are coloured **by instance index**, not by class — every instance here
has the same class, so a per-class palette would paint the frame one colour and
show nothing. Two adjacent plants in two different colours is the whole signal.
A white halo marks an instance SAM never proposed (the recall backstop).

### Output, per session under `auto_labels_mixed/<session>/`

| Item | Use |
|---|---|
| `cvat_ready/` | **upload this folder to CVAT** |
| `instances_default.json` | COCO import for CVAT (one category, `plant`) |
| `mixed_cvat_labels.json` | paste into CVAT's Raw label editor |
| `preview/` | judge separation and boundaries before a full run |
| `review_first.txt` | frames whose numbers say the masks are wrong — **do these first** |
| `flagged_rgb/` | colour-cast/glare frames, not exported |
| `instances.csv` | per-instance area, seed area, growth factor, source |
| `masks/` | the exported union, for a quick visual diff |

Same layout as the weed and onion prelabelers, so anything already built around
those directories keeps working.

### The console readout to actually watch

```
[sess] 62 frames | 431 instances | 0 flagged | 1 with none
    coverage: 97.3% of vegetation inside an instance | 402 from SAM + 29 recovered
    3 frame(s) to review first -> review_first.txt
```

**Coverage, not instance count.** Vegetation left outside every instance is
plant the annotator will never be shown. A pipeline can look productive while
quietly missing plants; the instance count cannot tell you that and coverage
can.

---

## Correcting in CVAT

1. Create the task from `cvat_ready/`.
2. Paste `mixed_cvat_labels.json` into the **Raw** label editor.
3. Import `instances_default.json` as **COCO 1.0**.
4. Work through `review_first.txt` first, then the rest.

Every shape arrives labelled `plant`. Reassign with the number keys — the schema
puts the real classes **first** precisely because CVAT assigns label shortcuts in
list order:

| Key | Class |
|---|---|
| 1 | `cutleaf_evening_primrose` |
| 2 | `wild_radish` |
| 3 | `grass_weed` |
| 4 | `weed_cluster` |
| 5 | `other_weed` |
| 6 | `onion_plant` |

`plant` sits last and is never chosen, only cleared. **Anything still labelled
`plant` when you export is unreviewed** — that is the sentinel's second job, and
it is worth grepping the export for before merging.

No LEP point label is offered. A shape type nobody draws still costs a slot in
that shortcut list, and LEP is better estimated over corrected masks anyway — an
LEP computed on a mask that turns out to be two plants is wasted work.

### Merging into the training set

`prepare_dataset.py` reads the CVAT export as usual. Nothing special is needed —
by the time it is exported, every shape carries a real ontology class. If a
`plant` shape survives, the category id `100` shows up as an unknown class
rather than silently becoming class 1.

---

## The two failures on pale, pebbly ground

Both were reported from the same real session, and they have nothing to do with
each other. The weed-only and onion-only prelabelers were clean on the same
imagery, which is itself the clue.

### Hundreds of tiny false masks on the gravel

**Cause: this path was far more permissive than the weed prelabeler, on
substrate where a colour index cannot win.** A pale, pebbly surface has
brown-olive stones that clear ExG, green-dominance and saturation. The weed
prelabeler ships its own recall backstop **off** for exactly this reason —
"a colour-only signal fundamentally cannot always tell green organic matter
from green-tinted mineral" — while this module runs its backstop on, and ran it
at a much lower floor:

| | weed prelabeler | mixed (before) | mixed (now) |
|---|---|---|---|
| `MIN_INSTANCE_AREA_PX` | 250 | 120 | **250** |
| recall backstop | off | on at 150 px | on at **600 px** |

`RECOVER_MIN_VEG_SCORE` looks like the defence and is not: `vegetation_score()`
is a logistic ramp times two gates, so it **saturates at 1.0** for anything
comfortably past the threshold. An olive pebble and a leaf both score ~1.0, and
a 0.90 floor rejects neither.

So **size is the discriminator that is left**, and its right value depends on
your mount height. `LOG_INSTANCE_AREAS` (on by default) prints the distribution
at the end of a run:

```
  instance area px: p10=180 p50=214 p90=5900 max=18400 | 74 of 96 (77%) under
  4x the 250px floor - suspect speckle, raise MIN_INSTANCE_AREA_PX and
  RECOVER_MIN_AREA_PX
```

Two populations an order of magnitude apart is what gravel-plus-plants looks
like. Put the floor in the gap between them rather than inheriting a number
measured on somebody else's rig.

### Several plants coming out as one instance

**Cause: the watershed was given one marker for a region containing four
plants.** When canopies grow into each other they are one connected vegetation
component, so if SAM proposed a mask for only one of them, that single marker
inherits the whole group. `review_first.txt` reports it exactly:

```
Visit1_..._000190.png   a seed grew 91x - possible merge
Visit1_..._000220.png   a seed grew 58x - possible merge; one instance covers 16% of the frame
```

The watershed is not at fault and needs no change — it was simply under-seeded.
`peak_seeds()` adds a marker at each distance-transform crown, and only for
components whose flood would exceed `SPLIT_GROWTH_TRIGGER` (8×, the same ratio
the review line prints). A component SAM seeded properly is never touched.

> **Why this is safe here when peak-splitting was abandoned in the weed
> prelabeler.** There, a *finished mask* was cut in half on peak evidence, and a
> leaf reaching away from the crown severed one plant into two annotations.
> Here the peaks only add **markers before the flood**, so a doubtful peak
> produces two markers the watershed may divide almost evenly rather than a
> guaranteed cut.
>
> That alone was not enough. **A long leaf has a flat distance ridge** — every
> point along a ribbon of constant width has the same inscribed radius — so the
> first version of this shattered a synthetic onion leaf into **11 instances**.
> Two guards fix it:
>
> - **Elongation ceiling** (`SPLIT_MAX_SPREAD`, default 8). Component area over
>   the area of its own largest inscribed disc. Measured: one rosette **1.0**,
>   two merged **2.1**, four merged **4.1**, a straight leaf **12.1**, a curved
>   one **27.3**. Above the ceiling the component is a ribbon and is left alone.
> - **Saddle test** (`SPLIT_PEAK_SADDLE_DROP`, default 0.7). A candidate crown
>   is kept only if it is still separated from every accepted crown at a level
>   below its own radius — between two real crowns the transform dips at the
>   neck; along one leaf it does not. Measured on a 240 px ribbon: 10 peaks
>   without it, **1** with it, while two rosettes joined by a thin neck still
>   give 2.
>
> **The accepted cost: two merged onions are not split either** — two ribbons
> are still a ribbon. That is the safe direction. A missed split leaves one
> shape to divide by hand; a false split teaches the model that a fragment of a
> leaf is a whole plant.

Set `SPLIT_UNDERSEEDED = False` to restore the previous behaviour, where only
SAM could ever separate two plants. The console now counts the three recall
paths separately, so you can see which is doing the work:

```
  coverage: 96.4% of vegetation inside an instance
  | 88 from SAM + 14 split from a merge + 6 recovered
```

---

## Tuning for mask quality

In rough order of how much they matter.

| Setting | When to change it |
|---|---|
| `EXG_THRESHOLD` | **The first thing to tune.** It decides every boundary. Onion is *bluer* than most broadleaf weeds — if previews show onion blades eroded while weeds look right, this is the cause, not SAM. Lower it to catch pale or shadowed tissue, at the cost of soil speckle |
| `VEG_MIN_COMPONENT_PX` | 60 here vs 150 in the weed prelabeler. This floor deletes tissue outright and nothing downstream recovers it. Raise it only if lowering `EXG_THRESHOLD` brought in speckle |
| `VEG_FILL_HOLES_MAX_PX` | Specular highlights read as non-green and punch holes through the prior, which come out as holes in the polygon. Raise if you see them; a genuine gap between leaves is soil and is protected by the area bound |
| `VEG_BLUE_TOLERANCE` / `VEG_BRIDGE_PX` | **If a whole onion leaf shows up as several small speckle instances instead of one**, this is the cause — see "Onion's glaucous leaf" above. Raise `VEG_BLUE_TOLERANCE` first; raise `VEG_BRIDGE_PX` only if the gaps are wide |
| `VEG_CLOSE_KERNEL_PX` | Off by default. Last resort if bridging alone still leaves gaps — check a **touching-plants** preview afterward, since this one can smooth over a real neck between two different plants |
| `SEED_NMS_IOU` | Lower merges more duplicate proposals into one plant; raise keeps more distinct instances. Symptom of too high: one plant with two overlapping outlines |
| `SEED_ERODE_PX` | Only affects where two instances meet. Raise if the cut between touching plants looks like SAM's boundary rather than the neck |
| `MIN_INSTANCE_AREA_PX` / `RECOVER_MIN_AREA_PX` | **The speckle controls.** On pebbly ground these are the only thing separating a cotyledon from a stone — `RECOVER_MIN_VEG_SCORE` saturates and cannot. Calibrate from `LOG_INSTANCE_AREAS`, not from a number measured elsewhere |
| `SPLIT_GROWTH_TRIGGER` | Raise if single plants come apart; lower if merges survive. Denominated in the same ratio `review_first.txt` prints |
| `SPLIT_MAX_SPREAD` | The elongation ceiling that keeps splitting off leaves. Lower to be more conservative; 0 would disable splitting entirely — use `SPLIT_UNDERSEEDED` for that instead |
| `RECOVER_MIN_VEG_SCORE` | Weaker than it looks — the score saturates at 1.0 for anything past the threshold, so gravel and leaf both clear 0.90. Use the area floors |
| `REVIEW_*` | Triage thresholds only — they change which frames get listed, never a mask |

### `instances.csv` tells you which failure you have

`growth` is the ratio of final area to seed area — how far the watershed had to
carry each seed.

- **`growth` near 1 everywhere** — SAM already had the plants; the watershed is
  doing nothing. Fine, but check that boundaries actually look right.
- **`growth` very large on a few instances** — the signature of two plants
  merged into one instance, because SAM only proposed one of them. These are
  flagged into `review_first.txt` by `REVIEW_MAX_GROWTH`. The pipeline cannot
  fix this itself; splitting the shape in CVAT is the correction.
- **many `source: vegetation` rows** — SAM is under-detecting. Check the
  exemplar gates before trusting the backstop to carry the session.

---

## What this deliberately does not do

No LEP, no growth stage, no instance crops, no depth, no species proposal. All
of it belongs to the weed prelabeler and can be run later over **corrected**
masks, which is the better order anyway.
