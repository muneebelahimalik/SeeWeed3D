# LEP localization — complete technical explanation

Everything about how the Leaf Emergence Point is found, from biology through to
a 3D coordinate and a safety verdict.

---

## 1. What the LEP is, and why the whole system is built around it

The **Leaf Emergence Point (LEP)**, also called the **apical meristem** or growth
point, is the small region at the centre of a plant where new leaves are
produced. In a rosette weed it is the crown at the centre of the leaf whorl; in
a grass it is the basal point where tillers emerge.

It matters because of how laser weeding actually kills a plant:

- **Burning a leaf does not kill the plant.** The plant regrows from the
  meristem. You have destroyed tissue the plant can afford to lose.
- **Destroying the meristem causes irreversible growth failure.** There is no
  tissue left to produce new leaves.

So the energy budget of the whole machine depends on hitting a target a few
millimetres across. A 60 W laser aimed 10 mm off may do nothing at all. This is
why LEP localization is not "nice extra precision" — it is the difference
between a weeder that works and one that scorches leaves.

Three consequences that shape every design decision below:

1. **A confidently wrong LEP is worse than no LEP.** Firing at the wrong point
   wastes energy, wastes time, and may damage a crop plant. Not firing costs one
   weed. Every component therefore abstains rather than guesses.
2. **The estimate must belong to the *right plant*.** In a dense frame the
   nearest crown is frequently a *neighbouring* plant. Ownership is enforced
   explicitly at every stage.
3. **It must be defensible, not just accurate.** For a paper, "the deepest point
   of the silhouette" is a property of the *shape*, not evidence about the
   *plant*. The estimator is built so each piece of evidence has a biological
   justification.

---

## 2. Two independent implementations

The repository contains **two** LEP estimators, and this is deliberate.

| | **A. Multi-evidence estimator** | **B. Learned LEPRoiNet** |
|---|---|---|
| File | `seeweed3d/perception/lep.py` | `seeweed3d/training/lep_roinet.py` |
| Type | Hand-engineered, no training | CNN, supervised |
| Needs labels? | **No** | Yes (verified CVAT LEPs) |
| Used for | SAM 3 prelabeling; runtime fallback; **comparison baseline** | Runtime, once trained |
| Status | Working now | Implemented, **not yet trained** |

**A is not throwaway code.** It:
- produces the LEP proposals that go into CVAT, so annotators *correct* rather
  than place every point from scratch;
- runs at inference when no learned model is loaded, so the full pipeline works
  today;
- is one of the four baselines the learned model must beat (§8).

---

## 3. Estimator A — the multi-evidence LEP estimator

### 3.1 Core idea

Rather than one geometric trick, compute **five independent score maps** over
the plant, each justified by a different aspect of plant biology or physics,
then fuse them. Because the channels are independent, **their disagreement is a
direct measure of uncertainty** — which is what makes principled abstention
possible.

Input is a `PlantContext`: the instance mask, the BGR crop, optional depth, the
crop origin, and the class name.

### 3.2 The five evidence channels

Every channel returns a map normalised to `[0, 1]` **inside the mask only**
(`_norm()`), so channels are comparable regardless of their internal units.

---

#### Channel 1 — PetioleConvergence (strongest structural evidence)

**Biology:** every leaf of a rosette radiates *from* the shoot apical meristem,
so the petiole (leaf-stalk) axes all converge on it. This is literally what a
botanist uses by eye.

**Method:**
1. Skeletonise the mask to a 1-pixel-wide medial skeleton using **Zhang–Suen
   thinning** (implemented in-repo so there is no `opencv-contrib` dependency
   and the result is deterministic across machines).
2. Find **junctions**: skeleton pixels whose 8-neighbourhood contains ≥3 other
   skeleton pixels. Junction strength = `max(0, degree − 2)`.
3. **Weight each junction by its inscribed radius** (`distance_transform / max`).
   This is the key refinement: the convergence point of a rosette is *thick* —
   many petioles overlapping — whereas leaf-tip junctions are thin. Radius
   weighting suppresses spurious tip junctions.
4. Gaussian blur, `σ = max(2.0, 0.12 · √area)`, so the score scales with plant
   size.

**Why it can fail:** heavily occluded plants, or plants whose leaves are so
overlapping that the skeleton has no clean junction.

---

#### Channel 2 — RadialIsotropy (phyllotaxy)

**Biology:** rosette phyllotaxy arranges leaves at regular angular intervals
about the meristem. **Seen from the meristem**, plant tissue is distributed
almost uniformly in angle. Seen from any off-centre point, it is not — tissue
bunches to one side.

**Method:** for each candidate point inside the plant, compute the **angular
entropy** of the directions to sampled plant pixels:

```
angles      θᵢ = atan2(yᵢ − y_c, xᵢ − x_c)
histogram   p  = normalised counts over 16 angular bins
score       H  = −Σ p·log(p) / log(16)        →  1.0 = perfectly isotropic
```

- Sample up to **600** plant pixels (deterministic RNG, seed 0).
- Evaluate candidates on a stride-4 grid, restricted to `dt > 0.15·dt_max` —
  **the meristem cannot lie on the silhouette boundary**, so boundary candidates
  are never scored.
- Vectorised: binning by integer arithmetic + one `bincount` over all
  candidates, chunked so peak memory stays bounded regardless of plant size.
- Dilate back to full resolution, Gaussian blur.

**Scale-free by construction** — entropy does not care how big the plant is or
how many leaves it has.

**Why it's downweighted for grasses:** a grass tiller is not a radial rosette,
so angular uniformity carries much less information (weight 0.9 → 0.3).

---

#### Channel 3 — YoungTissue (the only channel keyed on actual biology, not geometry)

**Biology:** the defining feature of the LEP is the **youngest emerging leaf
tissue**. Young expanding leaves have not yet accumulated full chlorophyll, so
they reflect more light and sit closer to yellow-green than mature leaves.

**Method:** convert the crop to **CIE Lab**, then

```
youth = 0.5 · norm(L*) + 0.5 · norm(b*)
```

- `L*` = lightness → young tissue is **lighter**
- `b*` = blue↔yellow axis → young tissue is **more yellow**

Blur with `σ = max(2.0, 0.08 · √area)`. Values clipped at the 99th percentile so
one specular highlight cannot dominate.

**This is the channel that makes the estimate defensible as a *leaf emergence
point*** rather than a shape centroid — it responds to the biological state of
the tissue.

---

#### Channel 4 — CanopyHeight (physical, needs depth)

**Biology:** leaves stack where they emerge, and the growing tip is elevated
relative to the surrounding expanded leaves and soil.

**Method:** from stereo depth, compute height above the local surface (nearer =
higher, since the camera looks down). Requires ≥25 % valid depth on the plant
(`min_valid_frac = 0.25`), otherwise the channel **abstains entirely** rather
than contributing noise.

**Optional by design** — the estimator works without depth, just with one fewer
line of evidence.

---

#### Channel 5 — MedialAxis (cheap regulariser)

**Method:** the plain distance transform of the mask — the deepest interior
point.

**Honest assessment:** this is the weakest evidence, because it is a property of
the *silhouette*, not the plant. It is included because it is cheap, extremely
stable, and a good regulariser when the other channels are ambiguous. **It is
deliberately the lowest-weighted channel** (0.4 for rosettes) — this is exactly
the "geometric trick" the multi-evidence design exists to avoid relying on.

---

### 3.3 Class-conditional weights

```python
DEFAULT_WEIGHTS = {
  "rosette": {petiole_convergence: 1.0, radial_isotropy: 0.9,
              young_tissue: 0.8, canopy_height: 0.6, medial_axis: 0.4},
  "grass":   {petiole_convergence: 1.0, radial_isotropy: 0.3,
              young_tissue: 0.8, canopy_height: 0.5, medial_axis: 0.5},
}
```

`ROSETTE_CLASSES` = `cutleaf_evening_primrose`, `wild_radish`, `other_weed`.
Everything else uses the grass profile. The only large difference is
**radial isotropy**, for the reason given above.

### 3.4 Fusion

```
fused = Σ (wᵢ · scoreᵢ) / Σ wᵢ
```

A channel is skipped if its weight is 0, if it reports itself unavailable
(e.g. depth missing), or if its map is degenerate. Each contributing channel's
**own argmax is recorded** — this is what makes the ablation ("which evidence
actually carried the estimate?") computable from stored results with no
reprocessing.

### 3.5 Sub-pixel localization

Not a single argmax pixel — that would be noise-sensitive and integer-valued.
Instead:

1. Take the **dominant peak region**: `fused ≥ max·(1 − top_frac)`, `top_frac = 0.25`
   (i.e. the top 25 % of the peak), intersected with the mask.
2. **Intensity-weighted centroid** of that region → sub-pixel `(cx, cy)`.
3. The same weighted region yields a **2×2 covariance** for free.

```
σ_px = √(largest eigenvalue of the covariance)
```

### 3.6 Uncertainty and confidence

Two independent uncertainty signals:

**Inter-channel agreement** — the median distance from each channel's own argmax
to the fused estimate. Small agreement means independent lines of evidence
coincide, which *is* the argument that the point is real.

Thresholds are expressed as a **fraction of plant radius** (`radius = √(area/π)`),
so they are scale-free — 5 px means very different things on a cotyledon and a
mature rosette:

```
good = 0.10 · radius      bad = 0.30 · radius
agree_score = clip((bad − agreement) / (bad − good), 0, 1)
```

**Peak sharpness** — a tight peak relative to plant size is decisive; a diffuse
one means the evidence is smeared over the plant:

```
sharp = clip(1 − σ_px / (0.6 · radius), 0, 1)
```

**Combined confidence:**

```
n_factor   = min(1, n_contributing_channels / 3)
confidence = clip(0.50·agree_score + 0.35·sharp + 0.15·n_factor, 0, 1)
```

**Visibility verdict (drives abstention):**

| Condition | Verdict |
|---|---|
| `confidence ≥ 0.60` **and** `agreement ≤ 1.5·good` | `visible` |
| `confidence ≥ 0.35` | `partially_occluded_inferable` |
| otherwise | `not_visible` → **abstain** |

### 3.7 The cost optimisation (and why it's safe)

Skeleton thinning and the radial-isotropy grid search both scale with crop
**area**, so a large rosette costs ~100× a seedling. Evidence is therefore
computed on a crop downscaled so its longest side is ≤ `max_work_px = 160`, and
the result is mapped back:

- lengths (`cx`, `cy`, `σ`, `agreement`) scale **linearly**
- second moments (covariance) scale **quadratically**

This is safe because the LEP is a *coarse location on a broad evidence peak* —
the peak is many pixels wide, so evaluating it at reduced resolution does not
move it. Measured speedup ≈ 3× with bit-identical output on the test shapes.

### 3.8 Measured accuracy (synthetic asymmetric rosettes)

| Method | Mean error |
|---|---|
| bbox centre | 28.0 px |
| mask centroid | 23.5 px |
| DT peak | 3.0 px |
| **fused multi-evidence** | **1.4 px** |

> These are **synthetic** shapes, not field data. They demonstrate the fusion
> improves on each single geometric baseline; they are **not** a field accuracy
> claim.

### 3.9 Output — `LEPResult`

```python
uv              (x, y) full-frame, sub-pixel
uv_local        (x, y) within the crop
confidence      [0, 1]
visibility      visible | partially_occluded_inferable | not_visible
agreement_px    spread of the per-channel argmaxes
covariance      2×2 spatial covariance
sigma_px        √(largest eigenvalue)
channels        {name: {uv, weight, peak}}   ← per-channel breakdown, for ablation
method_version  "seeweed3d/lep/1.0"
```

---

## 4. Estimator B — the learned LEPRoiNet

### 4.1 Why a second stage rather than one network

The LEP needs a **high-resolution view of one plant's crown**. Predicting it
from full-frame features at 1/8 resolution throws away exactly the pixels that
carry the phyllotactic structure. So:

- **Stage A** segments the whole frame → per-instance masks, boxes, classes.
- **Stage B** takes each weed as its own ROI and predicts its LEP.

### 4.2 ROI extraction (`training/roi.py`)

For each detected weed:

1. **Expand the box** by `expand_ratio = 1.35` about its centre and square it.
   The extra margin brings in the soil ring that makes local height meaningful,
   and context that says which plant a leaf belongs to.
2. **Letterbox** to `128 × 128`, preserving aspect ratio.

> **Aspect ratio is never stretched.** Stretching changes the apparent angles
> between leaves — and phyllotactic symmetry about the meristem is precisely the
> structure the network has to read.

3. **The same transform is applied to RGB, the owning mask, and depth.**

The transform is an explicit, **exactly invertible** object:

```
u_roi = (u_full − x0)·scale + pad_x
v_roi = (v_full − y0)·scale + pad_y
```

A single `warpAffine` implements it, so the image and the coordinates cannot
drift apart. **A systematic half-pixel error here would become a permanent
aiming bias that no amount of training corrects** — hence a round-trip test
asserting `to_full(to_roi(p)) == p` to 1e-9.

Interpolation: RGB uses linear; **mask and depth use NEAREST**. Interpolating
across a depth discontinuity invents a distance between a leaf and the soil
behind it — a value no surface occupies.

### 4.3 Geometry channels — and why depth is *not* a 4th RGB channel

Three geometry channels, fed to a **separate branch**:

| Channel | Meaning |
|---|---|
| `mask` | which pixels belong to **this** plant (ownership) |
| `height_norm` | elevation above the **local** soil reference, normalised |
| `depth_valid` | where `height_norm` may be believed at all |

**Why not concatenate depth to RGB?** Two reasons:
1. The first convolution would mix a metric quantity into colour features at
   full resolution.
2. It could not represent *"depth is missing here"* distinctly from
   *"height is zero here"*. Keeping validity as its own channel is what lets the
   network distinguish **flat ground** from **no measurement**.

**Why height above local soil, never raw depth?** Raw depth encodes the **camera
mount height**. A network fed raw depth learns "a plant is 900 mm away" and
silently fails the moment the rig is raised. Height above the soil *immediately
around this plant* is a property of the plant, transferable across cameras and
mount heights.

```
reference   = median depth of an annulus outside the instance
              (radii 1.10× → 1.60× the instance radius, so a seedling and a
               large rosette both sample soil just beyond their own canopy)
height      = clip(reference − plant_depth, −50 mm … +400 mm)
height_norm = (height − min) / (max − min)          → [0, 1]
```

If the ring has fewer than **40 valid pixels**, no reference exists and the
channel is reported **invalid** rather than guessed — an invented reference
biases every height on the plant.

*(A test verifies the normalised height is identical for the same plant at
1000 mm and 1500 mm mount heights — that is the camera-transferability property
made concrete.)*

### 4.4 Network architecture

```
RGB (B,3,128,128)
  stem      3×3 s2         → 64×64
  enc1      2× InvRes      → 64×64
  down1     InvRes s2      → 32×32
  enc2      2× InvRes      → 32×32      ─────────┐ skip
  down2     InvRes s2      → 16×16               │
  enc3      2× InvRes      → 16×16               │
                                                 │
geometry (B,3,128,128)                           │
  3× conv s2               → 16×16               │
                                                 │
  fuse: concat → 1×1 conv  → 16×16               │
        │                                        │
        ├── global avg pool → visibility (3)     │
        │                  → targetability (3)   │
        │                                        │
  up1   → interpolate to 32×32 ──── concat ──────┘
  up2   → 3×3 conv
  head  → 1×1 conv → HEATMAP (B,1,32,32)
```

**Blocks:** MobileNetV3-style inverted residuals (expand 1×1 → depthwise 3×3 →
project 1×1, Hardswish, residual when shape permits). Base width 32.

**Why convolutions and not a transformer?** TensorRT. These ops have
long-established, well-optimised kernels on Jetson; attention still needs care
to convert cleanly. This is a **deployment constraint driving the architecture**,
not a benchmark preference.

**Geometry fused at 1/8 resolution** — cheap, and by then the RGB features are
semantic rather than raw colour, so the two modalities stay separable. That
separability is what makes depth dropout survivable.

**Missing geometry at runtime:** zeros are fed instead, keeping the graph static
(TensorRT wants fixed shapes). Depth dropout during training means the network
has already seen this case.

### 4.5 Heatmap targets (`training/lep_targets.py`)

The network predicts a **32×32 heatmap** (stride 4 from the 128×128 ROI), not
coordinates directly.

**Why a heatmap?** It gives sub-pixel accuracy *and* a free uncertainty estimate
(the heatmap's spatial spread). Direct coordinate regression gives a point with
no confidence attached — and confidence is what abstention needs.

**Target generation** — a Gaussian rendered about the **exact float** location:

```
c = (u_roi + 0.5)/stride − 0.5          ← pixel-centre convention
G(x,y) = exp(−((x−cx)² + (y−cy)²) / (2σ²))
```

`σ = 2.0` heatmap pixels by default (clamped 1.0…6.0), optionally scaled with
plant size so a cotyledon and a large rosette are supervised at comparable
*relative* precision.

> **Sub-pixel truth is preserved.** Rounding the target to the heatmap grid would
> cap achievable accuracy at the stride (4 px) *before training starts*.

**Missing is not zero.** A `not_visible` weed gets supervision **weight 0**, not
a zero heatmap. Supervising zeros would teach the model "this plant has no
growth point", which is false — the point exists, the annotator could not see it.

| Visibility | Weight |
|---|---|
| `visible` | 1.0 |
| `partially_occluded_inferable` | 0.5 |
| `not_visible` | **0.0** |

### 4.6 Augmentation

Applied **identically** to RGB, mask, depth and the LEP point: h/v flips, 90°
rotations, ±15° rotation, ±10 % scale, ±20 % brightness. A test verifies the
point still lands on its own mask across many random seeds.

**Mosaic, MixUp, CutMix and copy-paste are refused with an exception**, not
merely unused. They composite pixels from *different plants* into one ROI, which
destroys the biological ownership the entire task depends on. A silently ignored
request would look like it had been applied.

### 4.7 Depth degradation during training

Field stereo depth is not the clean map a synthetic pipeline produces:

| Simulated | Value |
|---|---|
| Complete depth dropout | **p = 0.25** |
| Random holes | p = 0.15 |
| Range-dependent noise | 5 mm per metre |
| Quantisation | 1 mm |

The 25 % full dropout is the important one: it forces the RGB path to remain
self-sufficient, so the model degrades gracefully when depth fails in the field
instead of collapsing.

### 4.8 Losses (`training/losses.py`)

Five terms, each with a stated job. Deliberately few — a long list of
unvalidated auxiliary losses makes a regression impossible to attribute.

| Term | Weight | Job |
|---|---|---|
| `heatmap` | **1.0** | Per-pixel MSE on the Gaussian. Main localisation signal. |
| `soft_argmax` | 0.1 | Smooth-L1 on the **decoded** coordinate. The heatmap loss alone is content with a slightly asymmetric blob whose centre of mass is off; this penalises the quantity actually used at inference. |
| `visibility` | 0.5 | 3-way cross-entropy. The abstention gate. |
| `targetability` | 0.5 | 3-way cross-entropy. The annotator's treat/don't verdict. |
| `outside_mask` | 0.2 | Penalises probability mass outside the owning instance. |

**`outside_mask` is the ownership loss** — it is what teaches *"this plant's
crown, not the neighbour's"*, which is the difference between a safe target and
a wrong-instance error.

Classification heads are supervised for **every** sample including
`not_visible` — that verdict is exactly what the head must learn.

### 4.9 Decoding at inference

```python
uv_hm  = soft_argmax(heatmap)          # expected coordinate under the heatmap
cov    = spatial covariance of the heatmap
σ_hm   = √(largest eigenvalue)
uv_roi = (uv_hm + 0.5)·stride − 0.5
uv_full = transform.to_full(uv_roi)     # exact inverse
σ_px   = σ_hm · stride / transform.scale
```

**Soft-argmax, not integer argmax** — an integer argmax is floor-limited by the
stride (4 px). Soft-argmax is sub-pixel and differentiable.

**Decoding lives in one function** (`decode_lep`) used by both training and
deployment, so the two can never drift apart — a classic source of an offset
that only appears in the field.

**Uncertainty for free:** a confident prediction is a tight unimodal blob; an
ambiguous one is broad or bimodal, and both inflate the covariance. `σ_px` is
therefore a usable abstention signal costing nothing extra.

### 4.10 Batching

All weed ROIs from a frame are **stacked and run as one forward pass**. A dense
frame holds 30–60 weeds; per-instance forward passes would spend most of a
Jetson latency budget on kernel-launch overhead rather than arithmetic. A test
asserts the ROI model is called exactly **once per frame**.

---

## 5. From 2D LEP to a 3D camera-frame point (`perception/depth3d.py`)

A single stereo pixel at a plant crown is frequently wrong — it may be soil seen
through the whorl, an occluding neighbour leaf, or an unmatched pixel. Reading
one pixel and back-projecting gives a confident 3D point metres from the plant.

**The procedure:**

1. **Sample a disc** (radius 7 px) around the LEP.
2. **Intersect with the owning instance mask** — this is what stops a
   neighbouring plant's leaf, often at a very different depth, from contributing.
3. **Bimodality check on the RAW samples, before outlier rejection.** Sort the
   values, find the largest gap. If gap > 25 mm **and** the minority cluster
   holds ≥15 % of samples, **refuse** with `depth_discontinuity`.

> This ordering was a real bug found by the tests. MAD outlier rejection centres
> on the *majority* surface and discards the other as "outliers" — on a window
> straddling a leaf and soil 500 mm behind it, 45 % of samples were being thrown
> away and a *confident* depth reported for whichever surface won. That is
> exactly the confidently-wrong failure the system exists to prevent.

4. **Robust centre.** With a confidence map, use the **weighted median** so
   outlier rejection is anchored where the model believes the crown is, not on
   whichever surface occupies the most pixels.
5. **MAD filtering:** keep `|v − centre| ≤ max(3·MAD, 5 mm)`.
6. **Spread check:** refuse if the inlier spread > 40 mm.
7. **Back-project** through the stored rectified intrinsics:

```
X = z·(u − cx)/fx      Y = z·(v − cy)/fy      Z = z
```

8. **Propagate uncertainty** through the pinhole model:

```
var_X = (z/fx · σ_u)² + ((u−cx)/fx · σ_z)²
var_Y = (z/fy · σ_v)² + ((v−cy)/fy · σ_z)²
var_Z = σ_z²
```

so an uncertain depth **widens** the 3D uncertainty rather than being hidden
behind a confident-looking single number.

**Bed-plane fallback** exists for when depth fails entirely: intersect the pixel
ray with an assumed bed plane. It is **explicitly marked** `is_fallback=True`
and returns a deliberately large σ, because it assumes the crown lies on the bed
surface — wrong by the plant's own height.

> **No metric 3D accuracy is claimed anywhere.** The covariance describes the
> *spread of the depth samples* — internal consistency, not agreement with a
> surveyed point. Establishing metric accuracy needs 3D reference labels that do
> not exist yet.

---

## 6. The safety decision (`perception/safety.py`)

`decide()` **starts from reject.** Every check can veto; **no branch can
promote**. Acceptance is simply the absence of any veto, so a forgotten case
fails **closed** (no target) rather than open. All reasons accumulate — one call
explains every problem with a candidate.

LEP-related rejection reasons:

| Reason | Trigger |
|---|---|
| `no_lep_predicted` | no LEP produced at all |
| `lep_not_visible` | visibility verdict, or below confidence threshold |
| `low_lep_confidence` | heatmap peak < 0.30 |
| `high_heatmap_uncertainty` | `σ_px` > 12.0 |
| `outside_owning_mask` | LEP not on the plant it claims |
| `onion_safety_conflict` | laser spot **+ margin** intersects crop tissue |
| `weed_cluster` | no separable single growth point |
| `insufficient_valid_depth` / `depth_discontinuity` | 3D unreliable |
| `high_3d_uncertainty` | 3D σ > 15 mm |

**Two details that matter:**

- **Crop conflict tests a disc, not a pixel.** The beam has physical extent
  (`laser_spot_radius_px = 6`, `onion_safety_margin_px = 12`), so a spot whose
  *centre* clears the crop but whose *edge* does not would still damage it.
- **Snapping is bounded and recorded.** A LEP up to `allow_snap_to_mask_px = 2`
  outside its mask is nudged in and the correction written into the result.
  Beyond `lep_outside_mask_tolerance_px = 3` it is **rejected**, never silently
  reprojected.

**The module has no I/O and no actuator handle**, and a test asserts it never
acquires one. It produces *candidates*; turning one into a laser command is a
separate, deliberate act by the control layer.

---

## 7. Where LEP ground truth comes from

1. **SAM 3 prelabeling** runs estimator A and writes a proposed LEP per weed
   into COCO for CVAT.
2. **In CVAT**, the annotator drags the `weed_LEP` point onto the true crown and
   **groups it with its weed mask (`G`)**.
3. **The importer binds mask ↔ LEP by `group_id` ONLY.**

> **Ownership is never inferred by proximity.** An ungrouped LEP is *rejected*,
> not matched to the nearest weed — in a dense frame the nearest crown is
> frequently the neighbouring plant, and a wrong owner trains the model to aim at
> the wrong tissue. A nearest-instance hint may appear in the report marked
> *"SUGGESTION ONLY — not applied"*, and it never modifies a target.

**Validated rules:**

| Rule | Severity |
|---|---|
| `visible` + `targetable` weed has exactly one grouped LEP | error |
| `not_visible` weed is **not** required to have one | — |
| `weed_cluster` must not carry a single LEP | error |
| `onion_plant` never carries a weed LEP | error |
| LEP lies inside its owning mask (±4 px tolerance) | error beyond, warning within |
| Duplicate `group_id` in one image | error |

---

## 8. Evaluation

**Metrics** (`evaluation/metrics.py`):

- Pixel error: mean / median / p95
- **% within 2, 5, 10, 15 px**
- **Plant-size-normalised error** — 5 px on a cotyledon is a miss, 5 px on a
  large rosette is a hit; % within 0.1 / 0.25 / 0.5 × plant radius
- **LEP-inside-owning-mask rate** — a prediction outside its own mask is a
  *wrong-instance* error, categorically different from an inaccurate one
- Error broken down by visibility, class, and growth stage
- **Uncertainty calibration** — does predicted `σ_px` track actual error?
  (Spearman + binned table.) An uncertainty that does *not* correlate is worse
  than none: the abstention threshold would reject good targets and pass bad
  ones.

**Required baseline comparison** — the learned model must be compared against
all four:

| Baseline | What it is |
|---|---|
| bbox centre | centre of the bounding box |
| mask centroid | centroid of the mask |
| DT peak | distance-transform maximum |
| **`perception/lep.py`** | the multi-evidence estimator |

Everything is evaluated **by whole session**, never by random frame split —
adjacent video frames are near-identical, so a frame split measures
memorisation.

---

## 9. Configuration reference

```python
# ROI (training/config.py :: RoiConfig)
out_size          = 128        # network input, square
expand_ratio      = 1.35       # box expansion before cropping
min_box_px        = 16

# Heatmap (HeatmapConfig)
stride                    = 4      # → 32×32 heatmap
sigma_px                  = 2.0    # clamped 1.0 … 6.0
sigma_scale_with_plant    = 0.0    # >0 scales σ with plant size
partial_visibility_weight = 0.5

# Model (ModelConfig)
input_mode      = "rgb_mask_geom"   # | "rgb_mask" | "rgb"
width           = 32
heatmap_stride  = 4

# Losses (LossWeights)
heatmap 1.0 | soft_argmax 0.1 | visibility 0.5 | targetability 0.5 | outside_mask 0.2

# Depth (DepthRepresentationConfig)
soil_ring          = 1.10 … 1.60 × instance radius
soil_ring_min_px   = 40
height range       = −50 … +400 mm
depth_dropout_p    = 0.25
hole_dropout_p     = 0.15
noise              = 5 mm/m
quantisation       = 1 mm

# Safety (SafetyConfig)
min_lep_confidence            = 0.30
max_lep_sigma_px              = 12.0
lep_outside_mask_tolerance_px = 3.0
allow_snap_to_mask_px         = 2.0
laser_spot_radius_px          = 6.0
onion_safety_margin_px        = 12.0
max_3d_sigma_mm               = 15.0
min_depth_valid_fraction      = 0.35
max_depth_spread_mm           = 40.0

# Estimator A (perception/lep.py :: LEPEstimator)
top_frac             = 0.25    # dominant-peak region for sub-pixel centroid
agreement_good_frac  = 0.10    # × plant radius
agreement_bad_frac   = 0.30
min_confidence       = 0.35
max_work_px          = 160     # cost cap
```

---

## 10. File map

| File | Role |
|---|---|
| `perception/lep.py` | Estimator A: 5 evidence channels, fusion, uncertainty |
| `training/roi.py` | Invertible ROI transform, local-height geometry channels |
| `training/lep_targets.py` | Gaussian targets, soft-argmax, decode, augmentation, depth degradation |
| `training/lep_roinet.py` | Estimator B network |
| `training/losses.py` | The five multitask losses |
| `training/lep_dataset.py` | Per-sample chain: crop → degrade → augment → geometry → target |
| `training/datumaro_multitask.py` | CVAT ingestion, mask↔LEP grouping, contract validation |
| `perception/depth3d.py` | 2D LEP → 3D point with uncertainty |
| `perception/safety.py` | Abstention decision |
| `perception/pipeline.py` | Orchestration: segment → batch ROIs → LEP → 3D → safety |
| `evaluation/metrics.py` | LEP error, calibration, baseline comparison |

---

## 11. Current status — what is and is not proven

| | Status |
|---|---|
| Estimator A (multi-evidence) | **Working.** Used for prelabeling and as runtime fallback. |
| Estimator A accuracy | Measured on **synthetic** shapes only (1.4 px fused vs 3.0 px DT peak). Not a field claim. |
| Estimator B (LEPRoiNet) | **Implemented and unit-tested** — forward, backward, CPU training step, ONNX export all verified. |
| Estimator B **trained weights** | **Do not exist.** Requires verified CVAT annotations. |
| Any field LEP accuracy number | **Not measured.** |
| 3D metric accuracy | **Not measurable** — no 3D reference labels exist. The reported covariance is depth-sample spread, not accuracy. |
| Jetson latency | **Not measured.** Desktop numbers do not transfer. |

### Known limitations

1. **`weed_cluster` is always rejected.** Correct for safety, but clusters are
   common in dense imagery, so overall recall is limited by how often the
   annotator must use that class.
2. **Single-frame only.** No temporal tracking yet — a weed seen over several
   frames gives several LEP estimates, and fusing them should cut variance and
   allow a decision to be *deferred* rather than abstained.
3. **Occlusion is a hard limit.** If the crown is genuinely hidden, no method
   recovers it; the correct behaviour is `not_visible` + abstain.
4. **Estimator A's YoungTissue channel assumes reasonable colour balance.** A
   severe white-balance cast degrades it (white balance is applied upstream, and
   badly cast frames are flagged out).
