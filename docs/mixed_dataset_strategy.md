# Building the mixed-scene dataset

How to get from the data that exists to a multi-class instance-segmentation
dataset good enough to train on — and which parts of that are annotation cost
that cannot be engineered away.

**Related:** [runbook](RUNBOOK.md) · [mixed-scene prelabeling](mixed_prelabeling.md) ·
[dataset growth](dataset_growth.md) · [dataset assembly](dataset_assembly.md) ·
[depth-assisted masking](depth_assisted_masking.md)

---

## The problem, stated precisely

Zero-shot SAM 3 produces usable instances in **single-class** scenes and poor
ones in **mixed** scenes. The reflex is to call this a boundary-quality problem.
It is not.

The mixed prelabeler already uses SAM detections as watershed **markers** and
lets the flood decide only *where the cut falls*, constrained to the vegetation
mask. That design is sound. But a watershed can be no better than its markers,
and in a dense mixed scene zero-shot SAM gets **instance identity** wrong — it
merges two plants, or shatters one. Nothing downstream recovers from that.

Identity is also not something a general prior can supply. Knowing that two
touching onion *leaves* are one plant while two touching onion *plants* are two
is domain knowledge. It has to be learned from data.

> **The bottleneck is identity, not boundaries.** Effort spent on boundary
> refinement is spent on the half that already works.

## What the data can and cannot do

| Asset | What only it provides | Role |
|---|---|---|
| Onion-only prelabels (~17k instances, metric depth) | onion appearance and identity at scale; real soil backgrounds; reliable depth | train onion recall; composite **backgrounds** |
| Weed-only, **manually annotated** (~50 frames) | true weed masks **and** LEPs | composite **sources**; LEP-stage training |
| Mixed, **manually annotated** (~10 frames) | onion/weed **co-occurrence and contact** | **test only — never train** |
| Unlabelled pool | scale | mine, ranked by contact |

Everything here is strong except one thing: **only ten frames contain onion and
weed in contact.** That single gap is what the strategy below exists to close.

The onion prelabels are SAM output that made a round trip through CVAT without
correction — see `label_provenance` in
[dataset assembly](dataset_assembly.md). They are useful at scale and they are
not ground truth; a model trained on them inherits SAM's boundary conventions
and cannot exceed them.

## Stage B is not blocked by any of this

The system is instance segmentation → LEP localization. **The LEP stage does not
need mixed scenes.** It operates on ROI crops of individual weeds and neither
knows nor cares whether an onion was elsewhere in the frame.

The ~50 manually annotated weed-only frames already carry masks *and* grouped
LEPs. That is the LEP stage's training set, available now, and it is the one
part of the pipeline nothing is waiting on. Train and ablate it (`rgb`,
`rgb_mask`, `rgb_mask_geom`) in parallel with everything below.

---

## Principles this project has committed to

### Annotation policy becomes deployed policy

Models do not average out systematic label bias; they reproduce it. If
annotators reach for `weed_cluster` whenever separation is *tedious* rather than
*impossible*, the model learns that tangled foliage means cluster — and at
runtime that is not a labelling shortcut, it is weeds that never receive an
individual LEP and never get targeted. The laziness ships.

This makes `weed_cluster` **rate** a process metric, not just a dataset
statistic. Track it per session and per annotator. A rising cluster rate is an
early warning that annotation effort is decaying, visible long before it appears
as missed weeds in the field.

### `weed_cluster` is defined by crowns, not by foliage

The ontology already says it: *"intermingled weeds, no separable single LEP"*,
and the contract enforces the corollary that a cluster carries no LEP.

The criterion is therefore **"is there a distinguishable crown?"** — not "is the
foliage hard to trace?". Two plants whose leaves are thoroughly interwoven but
whose crowns are both visible are two instances, however annoying they are to
separate. The definition is correct as written; the risk is **drift** from it
late in a long annotation session.

### Not every merge costs the same

| Merge | Cost | Separation required |
|---|---|---|
| weed + weed | one target; different targeting strategy | only when no crown is separable |
| onion + onion | protection is a union; onions are never targeted | count does not matter — **extent** does |
| onion + weed | fire on crop, or miss a weed beside crop | **yes — this is the whole problem** |

This ranks effort; it does not license skipping. For **weeds**, instance count
has a direct consumer — each instance becomes a shot — so under-splitting costs
targets. For **onions**, nothing consumes the count, and the real risk is a
blobby mask that swallows an adjacent weed.

> Err large on onion masks, but do not train the model to be indiscriminate
> about it.

### Zero-shot has a fixed ceiling; a trained model does not

Zero-shot correction cost per frame is the same on frame 1,000 as on frame 1. A
model trained on in-domain instances improves every round, and the single-class
sessions are precisely where SAM gets identity *right* — training on them
transfers that competence into the dense scenes where it fails.

That is distillation from an easy regime into a hard one, not circular
reasoning. It is also why the two-stage arrangement is preferred over choosing
between them: **trained model for identity, SAM for boundary refinement from its
prompts.**

---

## Measure the asymmetry, not the mean

A mean IoU over all instances buries the one failure that matters among many
harmless ones — a model can look excellent while making exactly the dangerous
error. The mixed benchmark reports, separately:

* **onion tissue labelled weed** — catastrophic (laser on crop)
* **weed tissue labelled onion** — annoying (missed weed)
* boundary error specifically **at onion/weed contacts**
* `weed_cluster` predicted where ground truth has separable instances

This is the mask-level version of the philosophy already in
`onion_safety_metrics`, which keeps `onion_recall` and `missed_onion_px`
separate from IoU for the same reason.

---

## Make separation cheaper, not optional

Separation is the product; it cannot be engineered away. It can be made much
cheaper by changing **what the human does**.

Prelabel-then-correct beats annotating from scratch only when the prelabels are
good. In mixed scenes they are not, and repairing a merged or fragmented mask —
deleting, redrawing, splitting — is often *slower* than producing a good mask
from a click.

So split the labour along the line where each side is strong:

| Who | Job | Cost |
|---|---|---|
| Human | **one click per crown** | seconds; no boundary work |
| SAM | click → mask | its strongest mode |
| Watershed / vegetation mask | constrain growth | already implemented |

The click *is* the instance definition — the same crown criterion that
distinguishes an instance from a cluster. One gesture yields instance identity,
the LEP, and the mask prompt together, and the annotation contract already binds
a point to a mask by group id.

---

## Close the co-occurrence gap by compositing

Paste weed instances cut from the **manually annotated** weed frames into
onion-only frames. This is unusually well suited here:

* the cut-outs are human-quality, so pasted instances are **true** masks
* **their LEPs travel with them**, so composites train Stage A *and* Stage B
* backgrounds are real onion scenes — real soil, real lighting, real onions
* `mm_per_px` allows pasting at physically correct scale
* **contact is controllable** — weeds can be placed deliberately touching and
  overlapping onions, the case that is rarest in real data and most dangerous in
  the field

Composite in **height-above-local-soil** space rather than raw depth: relative
height transfers between scenes, absolute Z does not. `perception/ground.py`
already computes it, so pasted instances keep valid geometry channels.

**What composites do not give you:** real occlusion where plants grow *through*
each other, real shadow interaction, real co-adapted growth. They bootstrap the
contact case; they do not retire it. **Never validate on them.**

### The background screen decides whether this helps or hurts

Implemented in `annotation/compose_mixed.py`. One gate matters more than every
compositing detail combined.

A background is usable only if **everything green in it is already labelled**.
Paste a labelled weed into a frame that also holds an *unlabelled* one and the
composite teaches the exact confusion it was built to remove: two visually
identical plants, one a target, one background. That is worse than generating
nothing.

This is not hypothetical — the mixed build found three `Mix_2_Visit_2` drives
that are mixed scenes annotated for the crop alone. So every candidate
background is screened against the vegetation prior first, and vegetation not
covered by an annotated mask rejects the frame (`UNCLAIMED_VEG_MAX`).

Contact is **measured, not assumed**: each composite targets a band
(`isolated` → `near` → `very_near` → `touching` → `overlap`), and the band it
achieved is computed from the finished masks and recorded per instance. A run
that cannot report its achieved distribution cannot be ablated against.

Note also that the onion masks these are composited against are SAM prelabels,
so "distance to crop" is a distance to a machine label — a bound on what a
contact band means here, not a reason to skip it.

### One paste, one plant

Cut-outs are drawn **without replacement** — every hand-drawn plant is used
once before any is used twice. Sampling with replacement looks harmless and
isn't: a reused cut-out is the *same pixels*, moved and rotated, and the
frame-block split separates by frame **index**, which a composite set has no
video order to give meaning to. So duplicates scatter at random across train,
val and test, and a weed memorised in training becomes part of what the model
is scored on. The split's own seam-distance warning can't see it, because it
isn't a question about frame ordering.

The first run made this concrete: 506 pastes from a bank capped at 600 were
about 340 distinct plants, ~120 of them appearing in several composites.

Reuse now begins only when the pastes outnumber the bank, and the run says so
when it does. Distinct cut-outs is still an **upper** bound on distinct plants:
the weed drives are video, so one weed recurs in every frame it was driven past,
and 3,842 cut-outs came from 131 source frames. The report prints both numbers,
because no id in this pipeline can tell which instances are the same plant. Two knobs decide it — `BANK_MAX` (0 = every cut-out there is;
the two source drives hold ~3,300) and `WEEDS_PER_IMAGE`. Every pasted instance
also carries a `source_instance` attribute into the annotations, so a split can
honour provenance directly rather than inferring it from frame order.

`WEEDS_PER_IMAGE` has a second job: a composite carries every onion its
background held — about 18 — so a low weed count is what made the built dataset
30:1 crop-to-weed, with the thin weed classes unmeasurable. Don't overcorrect
past what a field looks like; the model learns scene statistics as well as
shapes.

### Looking at what came out

`compose_mixed` writes RGB and Datumaro, no pictures. To see a finished run's
own annotations drawn on its frames:

```
python -m seeweed3d.annotation.compose_mixed --overlays <run folder> --stride 10
```

It reads a run that already exists, so looking is never a reason to regenerate.
Weeds are coloured by the **contact band they achieved**, not by class — the
cut-out already carries a hand-drawn class, so the question a composite raises
is whether the placement the report claims is the one you can see. A weed
labelled `touching` sitting alone in bare soil is a generator bug no number in
the report would show.

Three things only a person can check: an outline that doesn't follow the plant,
a band label that disagrees with the picture, and light or shadow on the pasted
weed that doesn't match the onions around it.

### Wiring a run into the build

Every `compose_mixed.py` run writes a **new stamped folder**, so the mixed
build never picks "whatever is newest" — that is exactly the drift these
runners exist to prevent. Point `SYNTH_RUN` in
`training/datasets/mixed.py` at a run you have **looked at**; `SYNTH_RUN = ""`
builds without composites.

One constant, because the run has to be named in three places and they are
**three different strings**:

| where | what it needs | example |
|---|---|---|
| `MIXED_SESSIONS` / `SOURCES_ROOTS` | the **folder** | `synth_mixed_20260904_0156` |
| `INCLUDE_FRAMES` | the **session id** | `synth_20260904_015600` |
| `SCENE_HINTS` | the same session id → `"mixed"` | |

The session id comes from the **filenames** (`synth_<stamp>00_*.png`), not the
folder — a session id needs seconds and the folder stamp has none. So
`SYNTH_SESSION` is derived by `compose_mixed.session_id_for()`, the writer's own
function, and never typed. Getting it wrong fails *silently*: the wrong string
in `INCLUDE_FRAMES` matches nothing and 200 composites vanish from the build,
and in `SCENE_HINTS` it hints a session that does not exist, so the frames train
with no scene and the unmeasurable-class warning never fires.

Composites are **never** a holdout, and that is enforced rather than advised:
`TRAIN_ONLY_SESSIONS` pins the synth session to train, because the frame-block
splitter otherwise puts a share of *every* session into val exactly as designed
— it did, 15 val and 15 test, carrying most of val's weed instances. Held out, they would measure the compositor
— whether pasted weeds get found on backgrounds the compositor itself screened
— and report it as a crop-safety number for contact nobody observed. Worse, the
instances come from drives that also train, so the same plant would sit on both
sides of the split.

### Copy-paste as augmentation is still refused

[Dataset growth](dataset_growth.md) rules copy-paste out of every augmentation
preset. That rule is about pasting the **crop**, which would fabricate row
spacing and planting geometry no field produced. Weed-into-onion compositing
fabricates no crop geometry: every onion, furrow and shadow is the one the
camera recorded. The two positions are consistent, and the direction is the
reason.

---

## The sequence

1. **Benchmark on the 10 mixed frames**, with the asymmetric metrics above. The
   only ruler; everything after it is judged by it.
2. **Train the LEP stage** on the 50 weed frames. Independent and unblocked.
3. **Composite** weed instances into onion scenes, with deliberate contact.
4. **Train Stage A** on composites + onion prelabels + the 50 weed frames.
5. **Prelabel real mixed** with that model — three buckets, ranked by onion/weed
   contact perimeter.
6. **Crown-click annotate** the highest-contact frames.
7. **Retrain on real + composite**, re-measure on the 10, iterate.

Steps 5–7 are the loop. Each turn improves the prelabels and lowers the
annotation cost of the next.

### Ranking frames by contact needs no model

Shared perimeter between predicted onion and predicted weed is computable from
any prelabel available today. High contact means the dangerous decision is
actually exercised; well-separated frames are cheap and teach little.

The active learner already applies this idea at the LEP level — `_crop_risk_score`
scores a weed by its distance to onion — so applying it to prelabels for
annotation selection extends something that exists rather than inventing one.

---

## Allocation decisions

**All 10 mixed frames stay in test.** Ten frames will not move a model but are
barely enough for a metric. Splitting them leaves neither a training signal nor
a usable ruler.

**Start 2-class, not multi-class.** Species classes (`cutleaf_evening_primrose`,
`wild_radish`, `grass_weed`) will fall far under the 20-instance floor
`preflight.py` already checks, at ~50 weed frames. Train onion-vs-weed first and
add species as counts justify it; otherwise most of the reported AP is noise
from classes with three examples.

---

## Deliberately not doing

* **Reviving the global height veto.** Field-tested and reverted — it removed
  genuine thin onion tissue. See [depth-assisted masking](depth_assisted_masking.md).
* **Pseudo-labelling the crop.** Long-standing project rule; the model is
  already confidently wrong about grass-as-onion.
* **Using depth as a mask boundary.** Stereo fails at exactly the leaf margins a
  mask is deciding.
* **Treating `weed_cluster` as an escape hatch.** See above — it is a last
  resort defined by crowns, and its rate is monitored.
* **Validating on composites.** They share the generator's blind spots.
