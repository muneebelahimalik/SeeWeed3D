# Leaf Emergence Point (LEP) estimation — method and justification

`seeweed3d/perception/lep.py`

The LEP — the apical meristem / growth point — is the tissue the laser must
destroy to cause irreversible growth failure. This document states the method
and, more importantly, **why each part of it is biologically defensible**, so the
estimate can be justified in writing rather than asserted.

## The problem with a single geometric cue

A mask centroid, bounding-box centre, or distance-transform peak is a property of
the **silhouette**, not of the plant. On a symmetric rosette they land near the
right place by coincidence of shape; on an asymmetric plant — one side shaded,
one side grazed, leaves longer toward the light — they drift off the meristem
with no way to notice. Worse, none of them can be *argued* to be the leaf
emergence point: they are shape summaries that happen to correlate with it.

So the estimator fuses several channels that are each independently motivated by
plant architecture and physiology. Their **mutual agreement** then becomes the
evidence that the point is real, and their **disagreement** becomes a measurable
uncertainty that drives abstention.

## The evidence channels

### 1. Petiole convergence (structural)
In a rosette, every leaf radiates from the shoot apical meristem, so the petiole
axes converge on it. The plant mask is skeletonised (Zhang–Suen thinning,
implemented in-repo so there is no `opencv-contrib` dependency and results are
deterministic), and skeleton pixels with three or more skeleton neighbours are
scored as junctions. Junctions are weighted by their inscribed radius from the
distance transform, because the convergence point of a rosette is *thick* (many
petioles overlapping) while leaf tips are thin — this suppresses spurious
junctions at leaf ends.

*Why it is defensible:* this is the cue a botanist uses by eye. It follows
directly from the definition of a rosette: leaves emerge from one apex.

### 2. Radial isotropy (phyllotactic)
Rosette phyllotaxy places leaves at regular angular intervals about the
meristem. Seen **from** the growth point, plant tissue is nearly uniform in
angle; seen from any off-centre point it is not. For each candidate point the
angular histogram of plant pixels is computed and scored by normalised entropy
(1.0 = perfectly isotropic surroundings).

*Why it is defensible:* it uses the *arrangement* of leaves rather than the
outline, is independent of plant scale and leaf count, and is a direct
measurement of the symmetry that phyllotaxy produces.

### 3. Young-tissue chromatics (physiological)
The LEP is *by definition* the centre of the youngest emerging leaf tissue.
Young expanding leaves have not yet accumulated full chlorophyll, so they
reflect more light and sit closer to yellow-green than mature leaves. The score
combines CIE-Lab lightness `L*` and the blue↔yellow axis `b*` inside the mask,
with the extreme upper tail clipped so a specular glint on a wet leaf cannot
dominate.

*Why it is defensible:* **this is the only channel that keys on plant physiology
rather than geometry, and it is what makes the output a leaf EMERGENCE point
rather than a shape centre.** It is the channel to point at when asked why the
estimate is biological.

### 4. Canopy height (physical, requires depth)
Leaves stack where they emerge, and the growing tip is elevated relative to the
expanded leaves and the soil around it. Where stereo depth is valid, height
above a local soil reference (median depth in a dilated ring just outside the
plant) is an independent physical vote for the crown. The channel **abstains**
unless a configurable fraction of the plant has valid depth.

*Why it is defensible:* it is a direct measurement of 3D structure, entirely
independent of colour and of the silhouette, so agreement with the other
channels is strong corroboration.

### 5. Medial-axis interiority (regulariser)
The meristem is maximally interior to the leaf whorl, so the distance transform
peaks near it. Kept as a stable, cheap regulariser and weighted lowest, because
it is a silhouette property — it is honest evidence, but weak evidence.

## Fusion, uncertainty and abstention

Each channel produces a score map normalised inside the plant mask. Maps are
combined as a weighted sum (weights depend on class: grasses emerge from a basal
point rather than a radial rosette, so radial isotropy is down-weighted for
them). The estimate is the intensity-weighted centroid of the dominant peak
region — a **sub-pixel soft-argmax** — which is robust to single-pixel noise and
yields a 2×2 spatial covariance for free.

Three numbers are reported with every estimate:

| Quantity | Meaning |
|---|---|
| `agreement_px` | median distance of each channel's own argmax from the fused point. Small = independent lines of evidence coincide. |
| `sigma_px` | √(largest covariance eigenvalue) — spatial spread of the fused evidence. |
| `confidence` | combines channel agreement (scaled by plant radius, so it is scale-free), peak sharpness, and how many channels contributed. |

Visibility is then `visible`, `partially_occluded_inferable`, or `not_visible`.
**Abstention is a feature:** a confidently wrong LEP aims a 60 W laser at the
wrong tissue, while an abstained one is simply not treated.

Clusters (`weed cluster`) never receive a single LEP — by definition they
contain several growth points that cannot be individually assigned.

## Measured performance against the baselines

Synthetic rosettes with a known growth point, including asymmetric plants where
the silhouette centre is deliberately displaced (mean absolute error, px):

| Method | Mean error |
|---|---|
| Bounding-box centre | 28.0 |
| Mask centroid | 23.5 |
| Distance-transform peak | 3.0 |
| **Fused multi-evidence** | **1.4** |

The instructive case is the strongly asymmetric plant: the distance-transform
peak degrades to 7.0 px while the fused estimate holds at 2.3 px, because the
physiological and phyllotactic channels do not care that the silhouette is
lopsided. On real data these numbers must be re-measured against **human LEP
annotations** — that is what the CVAT round-trip exists for.

## Reproducibility and ablation

`LEPEstimator.describe()` returns the full method provenance (version, fusion
rule, every channel with its weight and written rationale) and is written to
`lep_method.json` beside each session's results, so any stored estimate can be
traced to the exact model that produced it.

Every channel's own argmax is stored per instance in `instances.csv`
(`lep_<channel>_x/y`). An **ablation table** — which evidence actually carries
the estimate, and how much accuracy each channel contributes — can therefore be
produced from stored results with no re-processing.

The three geometric baselines (`lep_dt`, `centroid`, `bbox_ctr`) are still
recorded per instance, so the project plan's LEP-method comparison (§15/§22) is
directly computable once human LEP labels exist.

## Extending to 3D

The 2D LEP plus its covariance feeds the 3D pipeline: `common/depth_utils.py`
already samples depth over a neighbourhood rather than a single pixel, rejects
outliers, and abstains when the local depth is untrustworthy. `sigma_px` gives a
principled radius for that neighbourhood, and the 2×2 pixel covariance is the
starting point for the 3D covariance the plan requires per target.
