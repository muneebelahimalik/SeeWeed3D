# Depth-assisted masking

Using metric stereo depth to do the one thing a colour index fundamentally
cannot: tell a green-tinted pebble from a cotyledon.

**Related:** [runbook](RUNBOOK.md) · [mixed-scene prelabeling](mixed_prelabeling.md) ·
[extraction fidelity](extraction_quality.md)

---

## Status: the height veto is OFF by default

`USE_DEPTH_HEIGHT` ships `False` in all four prelabelers. A field trial on a
real metric-depth onion session (dense transplanted seedlings, ~885mm boom
height) showed it removing genuine onion tissue — a thin leaf next to a
taller one in the same cluster lost its mask entirely, not just gravel or
soil speckle as intended.

The likely mechanism: dense clusters leave few bare-soil pixels per
`GROUND_TILE_PX` tile, so a neighbouring tile's estimate — pulled toward a
**taller** nearby plant — gets borrowed by nearest-neighbour fill, and the
shorter, thinner leaf reads as below that borrowed ground level. This is
offered as the likely explanation, not a confirmed one.

A missed onion is a worse failure than a phantom speckle surviving — this
project already reverted `RECOVER_MISSED_PLANTS` and the weed
boundary-refinement block on exactly this trade-off, and the same judgment
applies here. The infrastructure below (`perception/ground.py`) is unchanged
and worth revisiting for a use that does not delete on this evidence alone —
e.g. restricting the veto to backstop-only "recovered" instances that have no
other corroboration, or making it advisory (route to `review_first.txt` for
human review) rather than a silent drop. You can still opt back in per run via
`"USE_DEPTH_HEIGHT": True` or `"auto"` with this caveat in mind — the rest of
this document describes how it works when enabled.

---

## What depth is good at here, and what it is not

Stereo answers one question well on this imagery: **is this raised above the
ground?** A pebble sits at 0 mm. A cotyledon sits at 5–20 mm. Colour cannot
separate those at all — that is why the weed prelabeler ships its recall
backstop off, and why the mixed one had to fall back on a pixel-area floor.

It is **bad at the question it looks like it should answer**: *where exactly
does this leaf end?* Block matching fails on thin, low-texture structures and
fringes at every depth discontinuity, so leaf margins come back holed and
haloed — at precisely the boundary a mask is deciding.

So depth is used here as a **veto and a scale reference, never a boundary
source**. Colour keeps every boundary. Depth deletes things lying on the dirt
and converts pixels to millimetres.

## Which prelabeler gets it

All four, since a single-class scene has the same pebbly ground as a mixed one.

| Prelabeler | `HEIGHT_MIN_MM` | Note |
|---|---|---|
| `prelabel_onions_sam3.py` | 6.0 | transplanted onions stand up — the safest case |
| `prelabel_mixed_sam3.py` | 6.0 | per instance, before polygons |
| `prelabel_complement_sam3.py` | 6.0 | applied to the **weed** side only |
| `prelabel_weeds_sam3.py` | **4.0** | see the caution below |

> **Weeds are not the same problem.** Some broadleaf weeds grow **prostrate**,
> pressed flat against the soil — real targets that genuinely have almost no
> height, and a height gate is exactly the wrong instrument for them. Onions
> stand up. The weed's `HEIGHT_MIN_MM` is lower for that reason, should you
> re-enable the veto (see [Status](#status-the-height-veto-is-off-by-default)
> above — it is off by default in all four prelabelers now); trial it on 20
> frames and look for rosettes disappearing before trusting it.

## Which sessions can use it

Only captures with real 16-bit depth. Check per session:

```powershell
type <sessions>\<sid>\meta\session.json | findstr depth_kind
```

| `depth_kind` | Meaning |
|---|---|
| `metric` | real millimetres — everything here applies |
| `preview` | an 8-bit preview normalised **per frame** at capture. No constant relates it to millimetres; unrecoverable |
| `none` | no depth stream |

### If `depth_kind` says `unknown`

`depth_kind` was only added to the extractor in **#80**, so any session
extracted before that has no such field — it reads as `unknown` and the veto
correctly refuses to run, even when the depth PNGs are perfectly good.

Re-extracting answers it at the cost of hours and of whatever curation state
the pool carries. This reads the frames instead and classifies them:

```powershell
python -m seeweed3d.validation.check_extracted_depth --sessions E:\...\sessions
```

```
  Visit1_20260108_132749
    depth_kind : metric
    because    : median 1187 is a plausible camera distance in millimetres, and
                 the per-frame maxima vary with scene content rather than being pinned
    median 1187.0 | max 2410.0 | max spread 380.0 | peak saturated 0.0001
```

**If it reports `uncertain`**, read the printed median before assuming
anything is wrong. `PLAUSIBLE_MM` defaults to 250-6000 mm, which is a guess
about a typical boom — a delta-robot rig mounted close to the canopy can
legitimately sit under that. If the median is your real mount height, say so:

```powershell
python -m seeweed3d.validation.check_extracted_depth --sessions ... --min-mm 100 --max-mm 400 --write
```

Only widen it because you know the number — not to make the warning go away.
A median in the tens of thousands, or near zero, is not a mount-height
question; that is real evidence something in the capture or decode is wrong,
and re-extracting is the next step, not a wider range.

Add `--write` to record it in each `meta/session.json`. It **refuses to write
an uncertain classification** — a 16-bit PNG is not by itself evidence that the
numbers are millimetres, since ffmpeg produces `gray16le` from any source by
scaling values it invented, and guessing here would recreate by hand exactly
the fiction `REQUIRE_16BIT_DEPTH` exists to prevent.

Your v1 (AVI) captures are `preview` or `none`. Your v2 (MKV/FFV1) captures are
`metric`. `USE_DEPTH_HEIGHT` ships `False` (see
[Status](#status-the-height-veto-is-off-by-default) above). Set it to `"auto"`
to opt back in and use depth where it exists, behaving exactly as `False`
where it does not, so a mixed set of sessions needs no special handling. Set
it to `True` to make a session **without** metric depth a hard error, when you
intend a run to be depth-gated and want to know if it isn't.

---

## Why a local soil surface and not a fitted plane

Vidalia onions grow on **raised beds with furrows**. A single plane through the
frame reports the furrow floor as below ground and the bed top as plant —
inventing height where there is none and hiding it where there is.

`soil_surface()` estimates the ground **locally**, as a high percentile of
depth in a neighbourhood: the camera looks down, so soil is the *farthest*
surface and anything nearer is standing on it. That follows a furrow, a slope
and a tilted camera for free, because it never assumes the ground is flat —
only that it is smooth at the scale of the window.

### `GROUND_TILE_PX`, and why passing vegetation decides it

Two bounds pull opposite ways. The window must be **larger than the biggest
plant**, or a big plant defines its own soil and measures itself as flat. It
must be **smaller than the terrain undulation**, or a furrow is averaged into
the bed beside it.

Passing the vegetation mask dissolves the first bound. Measured on a 140 px
plant standing 25 mm proud of flat ground:

| `tile_px` | 32 | 48 | 96 | 160 |
|---|---|---|---|---|
| **veg given** | 25.0 | 25.0 | 25.0 | 25.0 |
| veg absent | 12.1 | 17.9 | 17.3 | 25.0 |

The prelabeler always passes it, so small tiles are strictly better — and the
second bound is the one that bites. Measured on 160 px beds with 60 mm furrows
under a tilted camera, two 18 mm plants (one on a bed / one in a furrow):

| `tile_px` | 32 | 48 | 64 | 96 |
|---|---|---|---|---|
| **smooth 0** | **20.0 / 20.0** | 20.8 / 20.3 | 35.9 / 24.6 | 61.0 / 21.9 |
| smooth 1 | 19.8 / 19.8 | 38.2 / 20.2 | 47.8 / 15.4 | 60.5 / 14.3 |

A tile straddling a bed and a furrow takes its soil from the **furrow**, so the
bed plant reads as bed height plus plant height — 61 mm for an 18 mm plant.

> **The error is not symmetric, which is why the defaults are conservative.** A
> window too large *inflates* plants on high ground and *deflates* plants in
> the low ground. A height veto deletes short things, so the plants it would
> silently drop are the ones in the furrows.

---

## The safety property: abstention

**An instance whose height cannot be measured is kept.** Stereo drops out on
thin, low-texture tissue and at every depth discontinuity — which is to say, on
exactly the small plants a height gate would otherwise delete. The veto fires
only on positive evidence that something is flat, never on absence of evidence
that it is not.

`HEIGHT_MIN_MEASURED_FRAC` (default 0.25) is the threshold. Below it the
instance survives on the colour and size gates exactly as before, and the run
counts it as an abstention rather than a decision.

```
  depth: 61% of pixels measured | ground relief 47 mm
       | dropped 38 flat + 4 undersized, abstained on 112
```

Abstaining on more instances than it keeps is **expected on seedlings** and the
run says so. It means the gate is mostly not acting — not that it is broken.

## Two things that are never guessed

**Invalid depth is not zero depth.** The extractor writes 0 as the invalid
sentinel. Read naively that is "touching the lens", nearer than everything, and
would read as the tallest object in the frame. Depth is loaded with `NaN` for
invalid and every function returns a `measured` mask alongside its answer.

**Confidence polarity.** v2 captures write a confidence map and `session.json`
records which direction means "good". Guessing that backwards does not degrade
gracefully — it keeps *precisely* the pixels it should have dropped, and the
result looks like a cleaner depth map. An unknown polarity therefore disables
confidence gating entirely, and the run prints that it did.

---

## Metric size floors

With depth and `calibration.json` you know mm per pixel, so
`MIN_INSTANCE_AREA_MM2` replaces `MIN_INSTANCE_AREA_PX` where it can be
computed. That removes the whole "calibrate the floor to your mount height"
problem: a session recorded at a different boom height stops needing its own
numbers.

Without calibration the metric floor is inert and the pixel floor applies, so a
missing `calibration.json` cannot silently delete everything.

---

## Settings

```python
"USE_DEPTH_HEIGHT": False,       # "auto" | True | False - ships False, see Status above
"HEIGHT_MIN_MM": 6.0,            # below this above local soil = not a plant
"HEIGHT_MIN_MEASURED_FRAC": 0.25,
"HEIGHT_PERCENTILE": 75.0,       # of the instance body, not its edge
"GROUND_TILE_PX": 32,
"GROUND_PERCENTILE": 80.0,
"DEPTH_MIN_CONFIDENCE": 0.30,
"MIN_INSTANCE_AREA_MM2": 40.0,   # None = keep using pixels
```

| Symptom | Setting |
|---|---|
| real seedlings disappearing | **lower `HEIGHT_MIN_MM`** first — it deletes, and a deleted cotyledon never reaches the annotator |
| ground clutter surviving | raise `HEIGHT_MIN_MM`, then check `ground_relief_mm` is plausible for your field |
| almost everything abstains | expected on small plants; lower `HEIGHT_MIN_MEASURED_FRAC` only if previews show clutter getting through |
| plants in furrows disappearing | `GROUND_TILE_PX` is too large for your bed spacing — halve it |
| heights implausible everywhere | check `median_depth_mm` against your real boom height before touching anything else |

`height_mm`, `height_measured_frac` and `area_mm2` are written per instance to
`instances.csv`, empty where unmeasured — because "0 mm" is a claim that
something is flat and "we could not tell" is a different statement.

---

## The complement prelabeler

`annotation/prelabel_complement_sam3.py` runs a trained **onion** model over a
mixed scene and labels the vegetation it did not claim. Onion-only sessions are
the cheapest thing to annotate — one class, no ambiguity — so a model trained on
them can be spent on the expensive scenes.

### Why it is a prelabeler and must never be a deployment rule

"Everything that is not onion is a weed" is arithmetic, and the arithmetic is
right. What is wrong is **the direction the errors point**.

When the onion model misses an onion — occluded, an unusual growth stage, the
edge of the frame, motion blur — those pixels are vegetation, no onion mask
covers them, and the complement calls them weed. Deployed, that is a laser
fired at the crop. It is not an exotic failure either: a missed detection is
the most common thing a detector does, recall never reaches 1.0, and every
onion in the residue becomes a target.

A two-class model's failure mode is "unsure", which a confidence threshold can
act on. The complement has no such handle — uncertainty defaults to weed, and
weed means fire.

### The third outcome is the whole design

| Region | Label |
|---|---|
| vegetation confidently claimed by the model | `onion_plant` |
| vegetation far from any onion, on confident ground | `other_weed` |
| everything else — near an onion, small, ambiguous | `ignore_region` |

`ignore_region` is an annotation-only label the dataset builder excludes from
training, so the complement's most common error costs an annotator one look and
teaches the model nothing.

**The halo does the real work.** A missed leaf of a *detected* onion is nearly
always adjacent to the part that was detected, so the onion region is widened
by `ONION_HALO_MM` before the complement is taken. Everything in that band that
was not claimed outright becomes `ignore_region`, never weed. With depth and
calibration the band is a fixed distance **on the ground** rather than in the
image, so it means the same thing at any boom height.

**Two escalations, both printed:**

- A frame with **no onion detected at all** does not become a frame full of
  weeds. In a mixed scene that is far more likely a model failure, so every
  plant in it is written as `ignore_region`.
- Over 90% of vegetation called weed prints a warning: an onion model failing
  on a session looks exactly like a session full of weeds.

`ONION_CONF` defaults to **0.15** — deliberately below anything you would
deploy. A generous onion mask costs one correction; a stingy one puts crop
tissue on the weed side.

### Running it

```python
"CHECKPOINT": r"E:\Dataset_Vidalia\training1\rfdetr_v4\best.pt",
"BACKEND": "rfdetr",
"DATASET_ROOT": r"E:\Dataset_Vidalia\Mixed_1",
"ONLY_SESSIONS": ["vid1_20260108_101500"],
"LIMIT_PER_SESSION": 20,          # trial first
```

```powershell
python seeweed3d/annotation/prelabel_complement_sam3.py
```

```
  [vid1_20260108_101500] 20 frames | 46 onion, 118 weed, 39 ignore
      vegetation: 41% onion | 44% weed | 15% uncertain
```

Correct the `ignore_region` shapes first — they are where the crop/weed
boundary actually lives, and they are the frames worth mining next.

> **This needs a trained onion model, which does not exist yet.** `rfdetr_v4`
> has been configured but never run. Until it has, use
> `prelabel_mixed_sam3.py` for mixed scenes.
