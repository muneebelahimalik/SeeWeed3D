#!/usr/bin/env python3
"""
SeeWeed3D - MIXED-scene instance prelabeling with SAM 3  (masks only)
=====================================================================
Builds one precise instance mask per plant - every weed and every onion
separately - in scenes that contain both, and exports them for class
assignment in CVAT.

    Run extraction/extract_sessions.py first (this reads its output pool).

WHAT THIS IS FOR, AND WHAT IT DELIBERATELY DOES NOT DO
------------------------------------------------------
One job: SEPARATION AND BOUNDARY QUALITY. Every plant its own mask, every mask
on the plant and not on the soil.

It emits a single homogeneous class, `plant`. It proposes no species, no
crop/weed split and no growth point. That is not a limitation being worked
around, it is the correct call for a mixed scene:

  * Shape can separate a blade from a rosette, and an ONION IS A BLADE. The one
    morphology call the weed-only prelabeler makes confidently would label the
    crop as grass_weed - the single worst error this project can make.
  * A wrong prelabel costs more than a neutral one. An annotator CONFIRMS a
    plausible label and RE-EXAMINES a blank one, so a plausible-but-wrong class
    is the one most likely to survive review.
  * Reassigning a class in CVAT is one keystroke. Fixing a boundary is a minute
    of mouse work. Spend the pipeline's effort on the expensive half.

No LEP, no growth stage, no instance crops, no depth. Those belong to the weed
prelabeler and can be run later over corrected masks, which is the better order
anyway - an LEP estimated on a mask that turns out to be two plants is wasted.

THE MASK LOGIC: VEGETATION OWNS THE BOUNDARY, SAM OWNS THE IDENTITY
-------------------------------------------------------------------
These are two different questions and the pipeline is bad at them in opposite
directions.

WHERE a plant ends is nearly free here. Top-down field imagery is green tissue
on brown soil, and a colour index answers that per pixel, at full resolution,
with no model. What it cannot do is say which green pixel belongs to WHICH
plant - a colour index sees one blob where two rosettes touch.

SAM 3 is the reverse. Concept segmentation returns instances, so it answers
identity well. But its boundaries are a learned prior at its own working
resolution: it clips leaf tips, rounds off dissected margins, and bleeds a few
pixels onto soil. Intersecting is not enough either, because a clipped tip is
already gone before the intersection happens.

So each is used only for what it is good at:

  1. Vegetation prior -> `veg`, the set of plant pixels. Boundary, done.
  2. SAM 3 -> proposals. Each is reduced to a SEED: its overlap with `veg`,
     eroded to its confident interior. The seed asserts "a distinct plant is
     here", nothing about where it ends.
  3. MARKER-CONTROLLED WATERSHED over `veg`, seeded by those markers, flooding
     the inverted distance transform. Every vegetation pixel is assigned to
     exactly one seed, and the surfaces meet at the NECK between plants.

That last step is what produces the masks. A seed that undershot grows out to
the true leaf edge; a proposal that bled onto soil is cut back to `veg` by
construction; two plants sharing one green blob are cut apart where the blob is
thinnest, which is where they actually touch.

WHY THE SPLIT IS SAFE HERE WHEN IT WAS NOT IN prelabel_weeds_sam3
------------------------------------------------------------------
That module ships SPLIT_TOUCHING_INSTANCES off, because splitting on
distance-transform PEAKS fragmented single plants: a leaf reaching away from
the crown raises a second peak, and one rosette became two annotations.

Nothing here splits on peaks. The markers come from SAM, so a blob is divided
only where SAM saw two distinct plants - and a leaf reaching away from a crown
is not a second SAM detection. The watershed then decides only WHERE the cut
falls, never WHETHER there is one. The failure mode that closed that door does
not exist on this path.

ONION'S GLAUCOUS LEAF DEFEATS THE VEGETATION PRIOR, AND HAS TO BE BRIDGED
--------------------------------------------------------------------------
Everything above assumes the vegetation prior is a solid, connected footprint
of each plant. On real mixed-scene frames it often is not, and the failure is
specific to onion: its leaf is a glossy, glaucous (waxy blue-green) tube, and
that surface defeats the prior's colour gates in two distinct ways at once.

  1. THE WAX BLOOM SHIFTS COLOUR TOWARD BLUE. `vegetation_mask()` requires
     green to dominate both red AND blue at every pixel. A matte broadleaf
     weed clears that easily; onion's glaucous coating measurably lifts blue
     reflectance, so stretches of a real onion leaf fail g>=b outright.
  2. THE GLOSSY CURVE THROWS SPECULAR HIGHLIGHTS. A round, wet-looking leaf
     catches direct light as a near-white streak along its length. A highlight
     has no leaf colour to detect - it IS the light source's colour - so it
     fails ExG, green-dominance and saturation simultaneously. No amount of
     threshold tuning admits it, because there is no green signal there at all.

Both cut STRAIGHT ACROSS a leaf's width, so the resulting gap touches the
surrounding soil on both sides. `fill_holes()` explicitly does not touch that
kind of gap - it only fills holes fully ENCLOSED by vegetation, by design, so
a genuine gap between two adjacent leaves (which is soil, and must stay soil)
is never filled in. A single onion leaf can come out of `vegetation_mask()`
broken into a dozen disconnected slivers, and every step after that inherits
the damage: SAM's exemplar boxes are built one per fragment instead of one per
leaf, `seed_masks()` intersects a proposal with the fragmented prior and
shreds it right back, the watershed can only flood within `veg` so it cannot
cross the gaps either, and `clean_instance()`'s fragment-dropping then throws
away the smaller slivers as if they were soil-texture speckle - because from
its side of the pipeline, that is exactly what they look like.

`recover_glaucous_pixels()` fixes this without touching the shared
`vegetation.py` (the onion and weed prelabelers are tuned against its current
behaviour, and this project has already reverted one boundary "improvement"
that looked better on paper and was worse in the field - see
docs/mixed_prelabeling.md). It relaxes the green-dominance and saturation
gates, but ONLY within a small halo around pixels the STRICT gate already
accepted - so a genuinely bare patch of pale soil earns nothing, while a
highlight or blue-shifted stretch sitting inside an already-confirmed leaf
does. Measured sufficient on its own on an afflicted synthetic leaf, and safe
on touching plants because the soil around a genuine gap between two of them
fails the colour test. `close_thin_gaps()` is an OFF-by-default escape hatch
for whatever it still misses - it has no colour context at all, so it smooths
a plant-to-plant neck exactly as readily as a gap within one leaf; raise it
only after re-checking a touching-plants preview, not just a single-leaf one.

TWO MORE THINGS THE FRAGMENTATION BREAKS, DOWNSTREAM OF THE MASK ITSELF
-------------------------------------------------------------------------
Reconnecting the mask is necessary and not sufficient. Two more places assume
a clean, unfragmented prior, and a first version of this fix missed both,
which meant real field sessions still came back with empty masks on most
frames after the mask itself was already fixed.

  1. THE SIZE FLOOR HAS TO RUN LAST. `vegetation_mask()` deletes any component
     under `VEG_MIN_COMPONENT_PX` before returning - and on a thin, heavily
     afflicted leaf, every INDIVIDUAL fragment can be smaller than that floor
     even though the leaf as a whole is not. Calling `vegetation_mask()` with
     the real floor already applied prunes every fragment before bridging ever
     sees them, and dilating an already-empty mask stays empty - the whole
     plant vanishes. `strict_vegetation()` calls it with `min_component_px=0`
     instead, so bridging gets every surviving speck to reconnect, and
     `plant_pixels()` applies the real floor once, at the very end, to the
     RECONNECTED result.
  2. CONFIDENCE HAS TO KNOW WHAT WAS BRIDGED. `unseeded_components()` and the
     exemplar-confidence gate both score a component by `vegetation_score()` -
     the exact strict colour rule that fragmented the leaf in the first place.
     A bridged highlight or blue-shifted pixel therefore scores near zero on
     it, and a component substantially made of bridged pixels can score well
     under `RECOVER_MIN_VEG_SCORE` (0.90) or `EXEMPLAR_MIN_VEG_SCORE` (0.85)
     even though it is real, reconnected leaf tissue - rejected by the very
     gate meant to keep noise out, on every frame where bridging did real
     work. `plant_confidence()` scores a bridged pixel by the BEST
     `vegetation_score()` found within `VEG_BRIDGE_PX` of it among pixels that
     passed the strict gate directly - the same context that justified
     admitting the pixel in the first place, reused rather than re-litigated.

RECALL: UNCLAIMED GREEN BECOMES AN INSTANCE
-------------------------------------------
A vegetation component containing no seed is a plant SAM missed entirely. Those
are recovered as instances of their own, gated on mean vegetation score.

The weed prelabeler disables its equivalent backstop, for a reason that does
not carry over. There, a recovered instance also had to be given a CLASS from
colour alone, and a green-tinted mineral fleck entered the training set as a
labelled weed. Here every instance is `plant` and every instance is reviewed by
a human before it becomes training data, so a phantom costs one keypress to
delete, while a missed plant is a plant the annotator is never shown. Trust the
gate, but read the previews for the first session.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation.prelabel_weeds_sam3 import (link_or_copy,  # noqa: E402
                                            load_sam3, mask_iou, mask_polygons,
                                            pool_frames, print_pool_report,
                                            refine_boundary, sam3_instances,
                                            smooth_boundary)
from common.ontology import (CLASSES, PRELABEL_CATEGORY_ID,  # noqa: E402
                             PRELABEL_CLASS, prelabel_categories,
                             prelabel_cvat_labels)
from common.progress import Progress  # noqa: E402
from perception.ground import height_veto  # noqa: E402,F401
from common.vegetation import (component_boxes, distance_peaks,  # noqa: E402
                               excess_green, remove_small, vegetation_mask,
                               vegetation_score, white_balance)
from perception.segmenter import drop_fragments  # noqa: E402

# #############################################################################
# ##   DATASET_ROOT   -  the OUTPUT_ROOT you gave extract_sessions.py        ##
# ##   ONLY_SESSIONS  -  your MIXED (onion + weed) sessions                  ##
# #############################################################################

DATASET_ROOT   = r"E:\Dataset_Vidalia\Vidalia_visit_1_20250221_all_sessions"
SAM_VERSION    = "sam3"
SAM_CHECKPOINT = r"E:\Models\sam3.pt"
# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "DATASET_ROOT":   DATASET_ROOT,
    "SAM_VERSION":    SAM_VERSION,
    "SAM_CHECKPOINT": SAM_CHECKPOINT,
    "OUTPUT_SUBDIR":  "auto_labels_mixed_v1",

    # Mixed sessions. Empty = every session under sessions/.
    "ONLY_SESSIONS": ["Visit1_20250221_142227","Visit1_20250221_142905","Visit1_20250221_143322"],

    "CVAT_READY_SUBDIR":  "cvat_ready",
    "FLAGGED_RGB_SUBDIR": "flagged_rgb",

    # -- Preprocessing ---------------------------------------------------------
    "WHITE_BALANCE": True,
    "WB_CAST_RATIO": 1.15,

    # -- Vegetation prior ------------------------------------------------------
    # This decides every boundary in the output, so it is the setting to tune
    # first and the one to check previews against. Lower EXG_THRESHOLD to catch
    # pale or shadowed tissue at the cost of soil speckle; the speckle is then
    # mostly removed by VEG_MIN_COMPONENT_PX, so the two move together.
    #
    # Onion is BLUER than most broadleaf weeds. If previews show onion blades
    # eroded while weeds look right, that is this threshold, not SAM.
    "EXG_THRESHOLD": 0.05,
    "VEG_MIN_SATURATION": 40,
    "VEG_MORPH_KERNEL": 3,
    # Lower than the weed prelabeler's 150: this floor deletes tissue outright
    # and, unlike there, nothing downstream can recover it. A cotyledon at this
    # mount height is a few hundred px, so 60 keeps real seedlings while still
    # clearing single-pixel colour noise.
    "VEG_MIN_COMPONENT_PX": 60,
    "VEG_SCORE_SOFTNESS": 0.04,
    # Close pinholes inside leaves (specular highlights read as non-green).
    # A hole left in the prior becomes a hole in the exported polygon.
    "VEG_FILL_HOLES_MAX_PX": 400,

    # -- Bridging onion's glaucous, glossy leaf ---------------------------------
    # See the module docstring section "ONION'S GLAUCOUS LEAF DEFEATS THE
    # VEGETATION PRIOR". A single onion leaf can come out of vegetation_mask()
    # broken into many disconnected slivers, because its waxy blue-green
    # cuticle both lifts blue reflectance past green (failing g>=b outright)
    # and throws specular highlights that carry no colour signal at all
    # (failing every gate at once). fill_holes() cannot rescue either case: both
    # cut across the leaf's width and connect to the surrounding soil, so
    # neither reads as an ENCLOSED hole.
    #
    # These knobs only ever ADD pixels within VEG_BRIDGE_PX of vegetation the
    # strict gate already accepted - a bare patch of pale soil sitting on its
    # own earns nothing from either rule, however low its saturation.
    #
    # 0 to disable both and fall back to the strict gate alone.
    "VEG_BRIDGE_PX": 6,
    # How much lower than EXG_THRESHOLD a bridge-halo pixel may score and still
    # be admitted - recovers the blue-shifted stretches of a glaucous leaf.
    "VEG_BRIDGE_EXG_RELAX": 0.02,
    # How far blue may exceed green in the bridge halo. 0 keeps the strict
    # g>=b rule; onion's wax bloom is what this exists to tolerate.
    "VEG_BLUE_TOLERANCE": 15,
    # HSV saturation at or below which a bridge-halo pixel is treated as a
    # specular highlight and admitted regardless of hue - a true highlight is
    # the light source's colour, not the leaf's, so no colour rule can key on
    # it directly; only "very low saturation, sitting against confirmed
    # vegetation" identifies one.
    "VEG_BRIDGE_HIGHLIGHT_SAT": 15,
    # Final, purely geometric safety net for whatever the colour-aware bridge
    # still misses: elliptical morphological closing on the resulting mask.
    # OFF by default - measured against two plants touching at gap=-4px (an
    # existing, deliberately tight scene elsewhere in this test suite), even a
    # small closing kernel smooths over the concave NECK between them exactly
    # the way it smooths over a gap in one leaf, because to a shape-only
    # operator the two look the same. The colour-aware bridge above has no
    # such failure mode (it only adds pixels that pass a colour test, and the
    # soil around a real neck fails it), and measured on an afflicted leaf it
    # is sufficient on its own. Only raise this if bridging alone still leaves
    # gaps after EXG_THRESHOLD/VEG_BLUE_TOLERANCE have been tuned, and check a
    # touching-plants preview afterward, not just a single-leaf one.
    "VEG_CLOSE_KERNEL_PX": 0,

    # -- SAM 3 prompting -------------------------------------------------------
    "SAM_PROMPT_MODE": "auto_exemplar",     # auto_exemplar | text
    "SAM_TEXT_PROMPTS": ["plant", "green plant", "weed"],
    # Exemplar hygiene, carried over from the weed path where it was measured:
    # SAM is asked to find every instance of "the same concept" across the whole
    # frame, so ONE marginal exemplar (gravel, lichen, a shadowed pit) teaches a
    # bad concept and produces phantoms everywhere, not just one bad box.
    "EXEMPLAR_MIN_AREA_PX": 300,
    "EXEMPLAR_MAX_BOXES": 30,
    "EXEMPLAR_PAD_PX": 8,
    "EXEMPLAR_MIN_VEG_SCORE": 0.85,
    "SAM_CONF": 0.25,
    "DEVICE": "cuda",

    # -- Seeds (what SAM's proposals are reduced to) ---------------------------
    # A seed only has to be inside the right plant and distinct from its
    # neighbours. It does NOT have to be the whole plant - the watershed grows
    # it - so these gates can be strict without costing extent.
    "SEED_MIN_AREA_PX": 80,           # after intersecting with vegetation
    "SEED_MAX_FRAC": 0.25,            # one plant covering >25% of frame = failure
    "SEED_VEG_OVERLAP_MIN": 0.30,     # proposal must sit on vegetation
    "SEED_NMS_IOU": 0.60,             # de-duplicate overlapping SAM proposals
    # Erosion pulls the marker off the boundary so the flood decides the edge
    # rather than inheriting SAM's. Skipped automatically for a seed too small
    # to survive it - a cotyledon must not be erased for tidiness.
    "SEED_ERODE_PX": 2,
    "SEED_ERODE_MIN_AREA_PX": 300,

    # -- Splitting an under-seeded component -----------------------------------
    # When several plants have grown into one another their canopies are ONE
    # connected vegetation component. If SAM proposed a mask for only one of
    # them, the watershed hands that single marker the whole group and four
    # rosettes emerge as one instance - the `a seed grew 91x` line in
    # review_first.txt. peak_seeds() adds a marker at each distance-transform
    # crown so the flood can separate them.
    #
    # ONLY components whose flood would exceed SPLIT_GROWTH_TRIGGER are touched,
    # so a component SAM seeded properly is never re-seeded and cannot be
    # fragmented. That trigger is the same ratio review_reasons() prints, so the
    # symptom and the fix are denominated in the same units: if the review file
    # says a seed grew 20x and the trigger is 8, this now acts on it.
    #
    # Raise SPLIT_GROWTH_TRIGGER if single plants are coming apart; lower it if
    # merges survive. Set SPLIT_UNDERSEEDED False to restore the previous
    # behaviour, where only SAM could ever separate two plants.
    "SPLIT_UNDERSEEDED": True,
    "SPLIT_GROWTH_TRIGGER": 8.0,        # matches REVIEW_MAX_GROWTH
    "SPLIT_MIN_COMPONENT_PX": 2000,     # a group of plants, not one seedling
    "SPLIT_PEAK_REL_THRESHOLD": 0.5,    # a peak this fraction of the largest
    "SPLIT_PEAK_MIN_SEPARATION_PX": 25,
    # THE SETTING THAT STOPS THIS SHATTERING A LEAF. A long leaf has a FLAT
    # distance ridge - every point along a ribbon of constant width has the
    # same inscribed radius - so height alone accepts a dozen "crowns" strung
    # along one onion leaf. Measured on a synthetic 240px ribbon: 10 peaks
    # without this test, 1 with it, while two rosettes joined by a thin neck
    # still give 2. Lower it (towards 0) to split more eagerly; 0 disables the
    # test and restores the behaviour that fragments leaves.
    "SPLIT_PEAK_SADDLE_DROP": 0.7,
    # Component area divided by the area of its own largest inscribed disc.
    # ~1 = one compact plant, ~4 = four merged rosettes, >20 = a long leaf.
    # Above this the component is a ribbon and is never re-seeded, which is
    # what keeps this off onion leaves entirely.
    "SPLIT_MAX_SPREAD": 8.0,
    "SPLIT_SEED_RADIUS_FRAC": 0.55,     # marker as a fraction of inscribed radius

    # -- Recall backstop -------------------------------------------------------
    # Vegetation components holding no seed at all. ON here, unlike the weed
    # prelabeler - see the module docstring for why the reasoning differs.
    #
    # RECOVER_MIN_AREA_PX IS THE SPECKLE CONTROL, AND ITS RIGHT VALUE DEPENDS ON
    # YOUR MOUNT HEIGHT. On a pale, pebbly substrate a colour index cannot tell
    # green-tinted mineral from a cotyledon - that is why the weed prelabeler
    # ships its own backstop OFF - so size is the discriminator that is left.
    # Gravel on a 1080p frame runs to a few hundred px^2; the smallest real
    # plant is usually several thousand. Run with LOG_INSTANCE_AREAS to see both
    # populations for YOUR imagery and put the floor in the gap between them.
    #
    # RECOVER_MIN_VEG_SCORE is a much weaker filter than it looks: the score
    # saturates at 1.0 for anything comfortably past the ExG threshold, so an
    # olive pebble and a leaf both score ~1.0. Do not rely on it alone.
    "RECOVER_UNSEEDED": True,
    "RECOVER_MIN_AREA_PX": 600,
    "RECOVER_MIN_VEG_SCORE": 0.90,

    # -- Instance cleanup ------------------------------------------------------
    # Relative, not absolute: a component at 15% of the main body is a speck, a
    # second true lobe is not. An absolute floor deletes cotyledons.
    "FRAGMENT_MIN_FRAC": 0.15,
    "FILL_HOLES_MAX_PX": 400,

    # -- Boundary quality ----------------------------------------------------
    #
    # The weed prelabeler's, imported rather than reimplemented: they were tuned
    # on this ground and a second copy would drift from it.
    #
    # THE ERROR IS SIZE-DEPENDENT, which is why a fixed band is the right shape
    # of fix. SAM decodes on a fixed grid over the whole frame, so a rosette
    # gets many cells across it and a seedling a handful - and the seedling's
    # outline is quantised to those few cells before anything else runs. 2 px is
    # a large fraction of a 30 px seedling and cosmetic on a 200 px one.
    #
    # It matters MORE in a mixed scene than in a weed-only one. Onion leaves are
    # thin tubes a few pixels wide crossing everything, so a boundary that
    # bleeds two pixels does not merely blur an edge - it swallows a leaf that
    # belongs to the crop, and the crop is the thing that must not be hit.
    "BOUNDARY_REFINE_BAND_PX": 2,
    "BOUNDARY_REFINE_VEG_MIN": 0.5,

    # Only refine instances at or below this area; 0 refines everything. 1500 px
    # is the project's own definition of a small weed (eval_seg.py), so the
    # prelabeler and the metric agree on what "small" means.
    #
    # DO NOT RAISE THIS to "improve" the big plants. They are the half that
    # already works, and #29 is the case where a boundary pipeline that improved
    # every number produced worse masks in the field.
    "BOUNDARY_REFINE_MAX_AREA_PX": 1500,

    # Anti-aliasing: blur-and-rethreshold to remove the single-pixel staircase
    # refinement leaves behind. Set ~0.7 to enable.
    "BOUNDARY_SMOOTH_SIGMA_PX": 0.0,
    "BOUNDARY_SMOOTH_MIN_RETAINED_FRAC": 0.85,
    # The last gate before an instance reaches CVAT. Every instance under this
    # is one shape somebody has to select and delete by hand, and on pebbly
    # ground there can be hundreds per frame. 250 is the value the weed
    # prelabeler uses on this same imagery without producing speckle; this path
    # ran at 120 and did. Calibrate with LOG_INSTANCE_AREAS rather than guessing
    # - a cotyledon must survive it.
    "MIN_INSTANCE_AREA_PX": 250,

    # -- Depth: height above the soil (v2 / metric-depth sessions only) --------
    # A colour index cannot tell a green-tinted pebble from a cotyledon. Height
    # can: the pebble is at 0 mm above the ground and the cotyledon is not.
    # This is a physical discriminator where MIN_INSTANCE_AREA_PX is only a
    # proxy, so where depth exists it is the better gate.
    #
    # "auto"  use it when the session's meta/session.json says depth_kind is
    #         "metric" AND a depth PNG exists for the frame. Sessions without
    #         real depth - every v1 AVI capture - behave exactly as before.
    # False   never use depth.
    # True    require it, and fail loudly if the session has none, so a run you
    #         believe is depth-gated cannot quietly not be.
    # OFF BY DEFAULT - FIELD-TESTED AND REVERTED. A trial on a real metric-depth
    # onion session (dense transplanted seedlings, ~885mm boom height) showed
    # the veto removing genuine onion tissue: a thin leaf next to a taller one
    # in the same cluster lost its mask entirely. The likely mechanism is the
    # local soil-surface estimate - dense clusters leave few bare-soil pixels
    # per GROUND_TILE_PX tile, so a nearby tile's estimate (pulled toward a
    # TALLER neighbour) gets borrowed by nearest-neighbour fill and makes the
    # shorter, thinner leaf read as below that borrowed ground level.
    #
    # A missed onion is a worse failure than a phantom speckle survives here:
    # this project already reverted RECOVER_MISSED_PLANTS and the weed
    # boundary-refinement block on exactly this trade-off, and the same
    # judgment applies. The infrastructure (perception/ground.py) is unchanged
    # and worth revisiting for a use that does not delete on this evidence -
    # e.g. restricting the veto to backstop-only "recovered" instances that
    # have no other corroboration, or making it advisory (route to
    # review_first.txt) rather than a silent drop. See
    # docs/depth_assisted_masking.md.
    "USE_DEPTH_HEIGHT": False,

    # An instance whose body sits below this many millimetres above the local
    # soil surface is not a plant. Set it well under your smallest real
    # seedling: this DELETES instances, and a cotyledon deleted here never
    # reaches the annotator to be recovered.
    "HEIGHT_MIN_MM": 6.0,

    # THE SAFETY PROPERTY: an instance whose height cannot be measured is KEPT.
    # Stereo drops out on thin tissue and at every depth discontinuity - which
    # is to say, on exactly the small plants this would otherwise delete - so
    # the veto only fires when enough of the instance carries real depth to
    # support the claim. Below this fraction it abstains and the instance
    # survives on the colour and size gates as before.
    "HEIGHT_MIN_MEASURED_FRAC": 0.25,
    # Percentile of the instance's height used as its height. A mask's edge
    # pixels straddle the plant's own depth discontinuity and read low; the
    # question is how tall the BODY is.
    "HEIGHT_PERCENTILE": 75.0,

    # Soil-surface estimation. See perception/ground.py for the measured
    # trade-off - briefly, small tiles follow beds and furrows, and passing the
    # vegetation mask (which this always does) removes the reason to want large
    # ones.
    "GROUND_TILE_PX": 32,
    "GROUND_PERCENTILE": 80.0,

    # Confidence gating, when the capture wrote a confidence map AND
    # session.json records which direction means "good". An unknown polarity
    # disables gating rather than guessing: guessing backwards keeps precisely
    # the pixels it was meant to drop.
    "DEPTH_MIN_CONFIDENCE": 0.30,

    # -- Metric size floor (needs depth AND calibration) -----------------------
    # The mm^2 counterpart of MIN_INSTANCE_AREA_PX. When depth and fx/fy are
    # both available this is used INSTEAD, which is what stops every size
    # threshold in the pipeline depending on how high the boom happens to be.
    # None = keep using pixels even where depth exists.
    "MIN_INSTANCE_AREA_MM2": 40.0,

    # -- Frame-level safety ----------------------------------------------------
    "MAX_VEG_FRACTION": 0.5,          # more green than this = colour cast/glare

    # -- Review triage ---------------------------------------------------------
    # Frames whose numbers say the masks are probably wrong, listed so they can
    # be corrected first instead of found at random.
    "REVIEW_MIN_COVERAGE": 0.90,      # vegetation left outside every instance
    "REVIEW_MAX_GROWTH": 8.0,         # seed grown this many times = under-seeded
    "REVIEW_MAX_INSTANCE_FRAC": 0.12,  # one instance this big is probably a merge

    # -- Polygon export --------------------------------------------------------
    "POLY_APPROX_EPS_FRAC": 0.010,
    "POLY_APPROX_EPS_MIN": 0.5,
    "POLY_APPROX_EPS_MAX": 1.5,
    "POLY_MIN_PART_AREA_PX": 40,
    # TRUE here, unlike the weed prelabeler. There, largest-part-only kept
    # previews looking tidy. Here precision is the entire deliverable, and
    # dropping a leaf that an occluding blade separated from its crown puts
    # real plant into the training target as background - which is exactly the
    # imprecision this pass exists to remove.
    "POLY_ALL_PARTS": True,

    # -- Run control -----------------------------------------------------------
    "LIMIT_PER_SESSION": None,        # start small; None for the full pool

    # Run over only PART of a session, by video frame index. Empty = all frames.
    #
    # A single drive can pass from one crop zone into another - weeds only for
    # the first stretch, then onions only. That is NOT a mixed scene: it is two
    # single-class scenes end to end, and the specialised prelabelers give
    # correct classes for free on each stretch, which the mixed one cannot.
    # Splitting by index lets each half be prelabelled by the module whose
    # assumption actually holds there.
    #
    # Same token syntax as curate_pool's MANUAL_DROPS - an inclusive index
    # range, an open-ended one, a bare index, or a frame/preview filename:
    #   {"vid1_20250221_131902": ["0-1200"]}     up to and including 1200
    #   {"vid1_20250221_131902": ["1500-"]}      1500 to the end of the session
    #
    # LEAVE A GAP AT THE TRANSITION. The frames where one zone becomes the
    # other are exactly where a single-class assumption is most dangerous, and
    # calling an onion a weed is the worst error this project can make. Give
    # the ambiguous stretch to prelabel_mixed_sam3.py, or leave it out.
    "ONLY_FRAMES": {},
    # Print the distribution of instance areas at the end of a run. The floors
    # above are the only thing separating a cotyledon from a pebble, and their
    # right values depend on your mount height - so read them off your own
    # imagery instead of inheriting a number measured on somebody else's.
    "LOG_INSTANCE_AREAS": True,

    "SAVE_PREVIEWS": True,
    "PREVIEW_SCALE": 0.5,
}

# =============================================================================


# --------------------------------------------------------------------------- #
# Vegetation prior
# --------------------------------------------------------------------------- #
def fill_holes(mask, max_px):
    """Fill enclosed holes up to max_px.

    A specular highlight on a leaf reads as non-green and punches a hole
    through the prior; that hole survives every later step and comes out as a
    hole in the exported polygon. Bounded by area so a genuine gap between two
    leaves - which is soil, and should stay soil - is not filled in."""
    m = np.asarray(mask).astype(np.uint8)
    if max_px <= 0:
        return m.astype(bool)
    inv = 1 - m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = m.astype(bool).copy()
    # Component 0 of the inverse touching the border is the outside world. Any
    # inverse component NOT touching the border is enclosed, i.e. a hole.
    h, w = m.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > max_px:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        out |= labels == i
    return out


def recover_glaucous_pixels(bgr, veg, cfg):
    """Admit blue-tolerant, low-saturation pixels ONLY inside a halo around
    already-confirmed vegetation.

    Onion's waxy glaucous cuticle defeats the strict gate two ways at once: it
    lifts blue reflectance enough to occasionally fail g>=b outright, and its
    curved, glossy surface throws a specular highlight that desaturates to
    near-white and fails every colour gate simultaneously - a highlight has no
    leaf colour to detect, because it IS the light source's colour, not the
    tissue's. Neither failure can be fixed by relaxing a threshold globally
    without also inviting in pale, low-saturation soil and gravel.

    So the relaxation only ever applies within VEG_BRIDGE_PX of a pixel the
    STRICT gate already accepted. A patch of bare soil sitting on its own,
    however pale, never earns that context; a highlight or blue-shifted stretch
    running through an already-confirmed leaf does.
    """
    if cfg["VEG_BRIDGE_PX"] <= 0 or not veg.any():
        return veg
    k = 2 * int(cfg["VEG_BRIDGE_PX"]) + 1
    halo = cv2.dilate(veg.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                      ).astype(bool) & ~veg
    if not halo.any():
        return veg

    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    exg = excess_green(bgr)
    loose = exg > (cfg["EXG_THRESHOLD"] - cfg["VEG_BRIDGE_EXG_RELAX"])
    loose &= g >= (b - cfg["VEG_BLUE_TOLERANCE"])
    loose &= g >= r                     # red is soil, wax bloom is never red

    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    highlight = sat <= cfg["VEG_BRIDGE_HIGHLIGHT_SAT"]

    return veg | (halo & (loose | highlight))


def close_thin_gaps(veg, kernel_px):
    """Final, purely geometric safety net: bridge whatever the colour-aware
    pass still missed by morphological closing.

    Has no colour context at all, so unlike recover_glaucous_pixels it CAN
    bridge two truly separate nearby plants into one blob - identity is SAM's
    job downstream regardless, so a modest kernel here is a shape safety net,
    not the primary fix. Keep it small; the primary fix is the function above."""
    if kernel_px <= 0 or not veg.any():
        return veg
    k = 2 * int(kernel_px) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)


def strict_vegetation(bgr, cfg):
    """Pixels passing the strict colour gate, before ANY size floor.

    This is the trustworthy "confirmed plant" set bridging anchors to, and
    that plant_confidence() falls back on for pixels bridging added. It is
    deliberately NOT vegetation_mask()'s own pruned output: on a thin, heavily
    afflicted leaf every individual fragment can be smaller than
    VEG_MIN_COMPONENT_PX on its own, and pruning at that stage deletes every
    one of them before bridging ever gets a chance to reconnect them into
    something real. min_component_px=0 keeps every surviving speck; the real
    floor is applied once, at the very end, in plant_pixels()."""
    return vegetation_mask(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                           cfg["VEG_MORPH_KERNEL"], min_component_px=0)


def plant_pixels(bgr, cfg):
    """The vegetation prior that will own every boundary in this frame."""
    strict = strict_vegetation(bgr, cfg)
    # Colour-aware bridging first, since it adds real evidence rather than
    # merely reshaping what is already there; pure-shape closing follows as a
    # smaller-radius mop-up for anything the colour rules still missed.
    veg = recover_glaucous_pixels(bgr, strict, cfg)
    veg = close_thin_gaps(veg, cfg["VEG_CLOSE_KERNEL_PX"])
    veg = fill_holes(veg, cfg["VEG_FILL_HOLES_MAX_PX"])
    # THE SIZE FLOOR RUNS LAST, on the fully reconnected mask. Pruning first
    # - which is what a single vegetation_mask() call would do - deletes the
    # very fragments bridging exists to heal before it ever sees them; on a
    # sufficiently thin, sufficiently afflicted leaf that deletes the WHOLE
    # plant, since dilating an already-empty mask stays empty and there is
    # nothing left to bridge from.
    return remove_small(veg, cfg["VEG_MIN_COMPONENT_PX"])


def plant_confidence(bgr, veg, cfg):
    """Continuous plant-likelihood, aware of what bridging already vetted.

    vegetation_score() applies the exact strict colour rule that fragments a
    glaucous onion leaf in the first place, so a highlight or blue-shifted
    pixel scores near zero on it - the SAME defect that got the pixel bridged
    into `veg` also tanks its own confidence score. Used as-is, a component
    that is genuinely real tissue but was substantially recovered by bridging
    scores far below RECOVER_MIN_VEG_SCORE or EXEMPLAR_MIN_VEG_SCORE and gets
    rejected by the very gates meant to keep noise out - discarding exactly
    the material the bridge exists to save, on every frame where bridging did
    real work.

    So a bridged pixel is not scored on its own (deliberately failing)
    colour. It inherits the BEST vegetation_score found within VEG_BRIDGE_PX
    of it among pixels that passed the strict gate directly - the same
    context recover_glaucous_pixels already used to decide the pixel was
    plant in the first place, reused here instead of re-litigated."""
    score = vegetation_score(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                             cfg["VEG_SCORE_SOFTNESS"])
    strict = strict_vegetation(bgr, cfg)
    bridged = veg & ~strict
    if not bridged.any():
        return score
    reach = max(1, int(cfg["VEG_BRIDGE_PX"]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    seed_score = np.where(strict, score, 0.0).astype(np.float32)
    spread = cv2.dilate(seed_score, k)          # max strict score within reach
    out = score.copy()
    out[bridged] = spread[bridged]
    return out


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def seed_masks(sam_masks, veg, cfg):
    """SAM proposals reduced to markers: on vegetation, sensible size, deduped.

    Returned largest-first. Each is the proposal's INTERSECTION with the
    vegetation prior, so a proposal that bled onto soil contributes no soil to
    the marker - and since the watershed can only grow a marker within `veg`,
    soil cannot re-enter later either."""
    if veg.size == 0:
        return []
    h, w = veg.shape
    frame_px = h * w
    cand = []
    for m in sam_masks:
        m = np.asarray(m)
        if m.ndim != 2 or 0 in m.shape:
            continue
        if m.shape != veg.shape:
            m = cv2.resize(m.astype(np.uint8), (int(w), int(h)),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m.astype(bool)
        area = int(m.sum())
        if area == 0 or area > cfg["SEED_MAX_FRAC"] * frame_px:
            continue
        # Judged BEFORE intersecting: a proposal mostly on soil is a bad
        # detection, and its green sliver should not be promoted to a plant.
        if float((m & veg).sum()) / area < cfg["SEED_VEG_OVERLAP_MIN"]:
            continue
        s = m & veg
        if int(s.sum()) < cfg["SEED_MIN_AREA_PX"]:
            continue
        cand.append((int(s.sum()), s))

    cand.sort(key=lambda t: -t[0])
    kept = []
    for _, s in cand:
        if all(mask_iou(s, k) <= cfg["SEED_NMS_IOU"] for k in kept):
            kept.append(s)
    return kept


def erode_seed(seed, cfg):
    """Pull a marker back off its own boundary so the flood decides the edge.

    Skipped for a seed small enough that erosion would take most of it: a
    cotyledon that survives every other gate must not be erased here, and a
    small seed is not near enough to a neighbour for its boundary to matter."""
    if int(seed.sum()) < cfg["SEED_ERODE_MIN_AREA_PX"] or cfg["SEED_ERODE_PX"] <= 0:
        return seed
    k = 2 * int(cfg["SEED_ERODE_PX"]) + 1
    e = cv2.erode(seed.astype(np.uint8),
                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return e.astype(bool) if e.any() else seed


def unseeded_components(veg, seeds, cfg, score):
    """Vegetation components no seed touches - plants SAM returned nothing for.

    Gated on MEAN vegetation score rather than the binary prior, because the
    binary prior is what let them in: a component that only just crosses the
    threshold everywhere is the profile of green-tinted mineral, while real
    tissue clears it comfortably."""
    if not cfg["RECOVER_UNSEEDED"]:
        return []
    claimed = np.zeros(veg.shape, bool)
    for s in seeds:
        claimed |= s
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        veg.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg["RECOVER_MIN_AREA_PX"]:
            continue
        comp = labels == i
        if (comp & claimed).any():
            continue
        if score is not None and float(score[comp].mean()) < cfg["RECOVER_MIN_VEG_SCORE"]:
            continue
        out.append(comp)
    return out


def _crown_markers(comp, area, claimed, cfg):
    """Markers at each distinct crown of one vegetation component, or [].

    ELONGATION IS THE GUARD THAT MAKES THIS SAFE ON ONION. Compare the
    component's area with the area of its own largest inscribed disc: a single
    rosette covers about one such disc, two merged rosettes about two, four
    about four - but a long thin leaf covers dozens, because its inscribed disc
    is only as wide as the leaf. Measured on the synthetic cases: one rosette
    1.0, two merged 2.1, four merged 4.1, one curved onion leaf 27.3. Above the
    ceiling the component is a ribbon, and a ribbon has a FLAT distance ridge
    whose maxima are an artefact of its own irregular edge rather than evidence
    of separate plants - splitting there shattered a synthetic leaf into 11
    instances before this guard existed.

    The cost is that two merged ONIONS are not split either: two ribbons are
    still a ribbon. That is the safe direction. A missed split leaves one shape
    to divide by hand; a false split teaches the model that a fragment of a
    leaf is a whole plant, which is the failure that closed this door in
    prelabel_weeds_sam3."""
    dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    rmax = float(dt.max())
    if rmax <= 0 or area / (np.pi * rmax * rmax) > cfg["SPLIT_MAX_SPREAD"]:
        return []
    peaks = distance_peaks(comp, cfg["SPLIT_PEAK_REL_THRESHOLD"],
                           cfg["SPLIT_PEAK_MIN_SEPARATION_PX"],
                           min_saddle_drop=cfg["SPLIT_PEAK_SADDLE_DROP"])
    if len(peaks) < 2:
        return []                       # one plant, however large - leave it
    out = []
    for (px, py), radius in peaks:
        if claimed[py, px]:
            continue                    # SAM already owns this crown
        disc = np.zeros(comp.shape, np.uint8)
        cv2.circle(disc, (int(px), int(py)),
                   max(1, int(radius * cfg["SPLIT_SEED_RADIUS_FRAC"])), 1, -1)
        marker = disc.astype(bool) & comp
        if marker.any():
            out.append(marker)
            claimed |= marker
    return out


def peak_seeds(veg, seeds, cfg):
    """Extra markers for vegetation components the SAM seeds cannot separate.

    THE MERGE THIS FIXES. The watershed gives every pixel of a connected
    vegetation component to the nearest seed. When several plants have grown
    into one another their canopies are ONE component, so if SAM proposed a
    mask for only one of them, that single marker inherits the whole group -
    four rosettes emerge as one instance, and the QA line reports it as `a seed
    grew 91x`.

    The watershed is not the problem and needs no change; it was simply given
    one marker for a region containing four plants. So the fix is more markers,
    not a different partition - and markers are the safe place to intervene. A
    doubtful extra peak produces two markers that the flood may divide almost
    evenly, whereas cutting a FINISHED mask in half on the same evidence is
    what made splitting untenable in the weed prelabeler: there, a leaf
    reaching away from the crown produced a second peak and severed one plant.

    TRIGGERED BY THE MEASUREMENT, NOT APPLIED EVERYWHERE. Only components whose
    flood would exceed SPLIT_GROWTH_TRIGGER are re-seeded, which is the same
    ratio review_reasons() already prints. A component SAM seeded properly is
    never touched, so this cannot fragment a plant that was already correct.
    """
    if not cfg.get("SPLIT_UNDERSEEDED", True):
        return []
    claimed = np.zeros(veg.shape, bool)
    for s in seeds:
        claimed |= s

    n, labels, stats, _ = cv2.connectedComponentsWithStats(veg.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg["SPLIT_MIN_COMPONENT_PX"]:
            continue
        comp = labels == i
        seeded_px = int((comp & claimed).sum())
        # A component with NO seed is the recall backstop's business, not this
        # one's. Creating instances there from the vegetation prior alone is
        # exactly what RECOVER_UNSEEDED governs, and quietly doing it here
        # would make that switch a lie. Such components are split, if they
        # deserve it, in split_recovered() - which only runs when the backstop
        # is on.
        if not seeded_px or area / seeded_px <= cfg["SPLIT_GROWTH_TRIGGER"]:
            continue
        for marker in _crown_markers(comp, area, claimed, cfg):
            out.append(marker)
    return out


def split_recovered(blobs, cfg):
    """Turn a recovered blob that holds several crowns into one marker each.

    The recall backstop returns whole unseeded vegetation components, so a
    group of merged plants SAM missed entirely arrives as ONE instance. Same
    evidence and same guards as peak_seeds - it is the same question asked one
    step later, about a component that had no seed rather than too few."""
    if not cfg.get("SPLIT_UNDERSEEDED", True):
        return list(blobs)
    out = []
    for comp in blobs:
        area = int(comp.sum())
        markers = _crown_markers(comp, area, np.zeros(comp.shape, bool), cfg)
        out.extend(markers if len(markers) >= 2 else [comp])
    return out


# --------------------------------------------------------------------------- #
# The watershed that actually produces the masks
# --------------------------------------------------------------------------- #
def partition_vegetation(veg, seeds):
    """Assign every vegetation pixel to exactly one seed.

    Marker-controlled watershed flooding the INVERTED distance transform of the
    vegetation mask, which puts the ridge line - where two floods meet - at the
    thinnest part of a shared blob. That is where two plants touch.

    The alternative, flooding the image gradient, sounds more principled and is
    worse here: inside a canopy the strongest gradients are leaf veins, shadow
    edges and specular highlights, so the cut chases texture within one plant
    instead of the join between two.

    Watershed marks its ridge pixels -1, and OpenCV also stamps the image
    border. Left alone that shaves pixels off instances - along the join
    between two plants, where it belongs, but also off any plant running to the
    frame edge, where it does not.

    So ridge pixels are reclaimed, but only where the answer is unambiguous: a
    ridge pixel whose 8-neighbourhood touches exactly ONE instance is an outer
    edge and goes to that instance, while one touching two instances is a real
    join between plants and stays unassigned. Adjacent instances therefore end
    up separated by a one-pixel line rather than sharing pixels, which is the
    right answer for a training target - no pixel belongs to two instances.

    Returns a list of masks, one per seed, in the order the seeds were given.
    A seed whose territory vanishes returns an empty mask, so the caller can
    still line results up with whatever metadata it kept per seed.
    """
    h, w = veg.shape
    out = [np.zeros((h, w), bool) for _ in seeds]
    if not seeds or not veg.any():
        return out

    # NO background marker, deliberately. Seeding the soil looks like the
    # careful thing to do and quietly destroys thin tissue: soil is flat ground
    # at the top of the relief, so a background flood arrives at an arm tip -
    # which is also near-flat, being barely wider than the transform's reach -
    # before the flood coming up the arm from the crown does. Measured on a
    # six-armed synthetic rosette, that soil marker ate four of the six tips.
    #
    # Plant-versus-soil is not this step's question anyway; the vegetation
    # prior already answered it, and intersecting with `veg` at the end
    # enforces it exactly. The watershed is left to decide only which plant.
    markers = np.zeros((h, w), np.int32)
    for i, s in enumerate(seeds):
        markers[s & veg] = i + 2

    # Distance from soil, inverted: deep inside a plant is low ground, the neck
    # between two plants is high ground. uint8 because cv2.watershed wants an
    # 8-bit 3-channel surface.
    dt = cv2.distanceTransform(veg.astype(np.uint8), cv2.DIST_L2, 5)
    peak = float(dt.max()) or 1.0
    relief = (255 - np.clip(dt / peak, 0, 1) * 255).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)

    # Reclaim the ridge where it is unambiguous. lab holds instance labels
    # only; a 3x3 dilate gives the largest neighbouring label and a 3x3 erode
    # over the same array with non-instance pixels raised to a sentinel gives
    # the smallest. Equal means the neighbourhood saw exactly one instance.
    ridge = veg & (markers == -1)
    if ridge.any():
        lab = np.where(markers > 1, markers, 0).astype(np.float32)
        k = np.ones((3, 3), np.uint8)
        hi = cv2.dilate(lab, k)
        sentinel = float(len(seeds) + 3)
        lo = cv2.erode(np.where(lab > 0, lab, sentinel).astype(np.float32), k)
        unique = ridge & (hi == lo) & (hi > 1)
        markers[unique] = hi[unique].astype(np.int32)

    for i, s in enumerate(seeds):
        out[i] = _own_components((markers == i + 2) & veg, s)
    return out


def _own_components(claimed, seed):
    """Keep only the parts of a territory that contain the seed itself.

    With no background marker the flood runs across soil too, so an instance
    can be handed a vegetation component on the far side of a gap that its
    seed never touched. Intersecting with `veg` does not catch that - the
    stolen component is real vegetation. Requiring a part to contain its own
    seed does.

    Tissue a leaf genuinely severed from its crown is dropped here rather than
    absorbed by whichever plant happened to win it. That is not a loss in the
    default configuration: an unclaimed vegetation component is exactly what
    RECOVER_UNSEEDED turns into an instance of its own, so the tissue still
    reaches the annotator - as a separate shape to merge rather than as a limb
    silently attached to the wrong plant."""
    if not claimed.any():
        return claimed
    n, labels, _, _ = cv2.connectedComponentsWithStats(claimed.astype(np.uint8), 8)
    if n <= 2:
        return claimed
    keep = set(np.unique(labels[seed & claimed])) - {0}
    return np.isin(labels, list(keep)) if keep else np.zeros_like(claimed)


def clean_instance(mask, cfg, veg_score=None):
    """Fill pinholes, snap the edge, anti-alias, drop specks.

    Runs AFTER the watershed, because every operation here is about the shape of
    a FINISHED instance: filling before it would let a hole get assigned, and
    dropping specks before it would delete markers.

    The boundary steps are the weed prelabeler's, imported rather than copied -
    they were tuned on this data and a second implementation would drift. They
    are no-ops until configured, so the mixed pipeline behaves exactly as before
    unless BOUNDARY_REFINE_BAND_PX is set.

    ORDER. Refine decides each edge pixel on the image's own evidence, which
    leaves single-pixel staircase noise a real leaf margin does not have; smooth
    removes it. Both before drop_fragments, so a speck the refinement created is
    still caught, and both before the area floor, so an instance is measured at
    the size it will actually be exported at."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    m = fill_holes(m, cfg["FILL_HOLES_MAX_PX"])
    if veg_score is not None:
        m = refine_boundary(m, veg_score, cfg)
    m = smooth_boundary(m, cfg)
    m = drop_fragments(m, min_frac=cfg["FRAGMENT_MIN_FRAC"])
    return m if int(m.sum()) >= cfg["MIN_INSTANCE_AREA_PX"] else None


def instance_bbox(mask):
    ys, xs = np.nonzero(mask)
    x, y = int(xs.min()), int(ys.min())
    return [x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1]


# --------------------------------------------------------------------------- #
# One frame
# --------------------------------------------------------------------------- #
def analyze_frame(bgr, sam_masks, cfg, depth_mm=None, conf=None,
                  polarity=None, fx=None, fy=None):
    """Vegetation prior -> seeds -> watershed -> cleaned instances.

    Returns (instances, veg, qa). Pure CPU; the SAM call happens outside, which
    is what makes every decision in this module testable without a GPU.

    depth_mm, when given, is metric millimetres with NaN for invalid - see
    perception/ground.py. It is a VETO and a scale reference only: colour still
    owns every boundary, because stereo is least reliable exactly at the leaf
    margins a mask is deciding. Omitting it reproduces the previous behaviour
    exactly, which is what every v1 session relies on."""
    veg = plant_pixels(bgr, cfg)
    score = plant_confidence(bgr, veg, cfg)
    seeds = seed_masks(sam_masks, veg, cfg)
    # Extra markers BEFORE the recall backstop, so a group of merged plants is
    # split into one instance each rather than recovered as a single blob.
    peaks = peak_seeds(veg, seeds, cfg)
    recovered = split_recovered(
        unseeded_components(veg, seeds + peaks, cfg, score), cfg)
    sources = (["sam"] * len(seeds) + ["peak"] * len(peaks)
               + ["vegetation"] * len(recovered))

    allseeds = ([erode_seed(s, cfg) for s in seeds] + list(peaks)
                + list(recovered))
    seed_px = [int(s.sum()) for s in allseeds]
    grown = partition_vegetation(veg, allseeds)

    instances = []
    for m, src, sp in zip(grown, sources, seed_px):
        m = clean_instance(m, cfg, score)
        if m is None:
            continue
        area = int(m.sum())
        instances.append({
            "mask": m, "cls": PRELABEL_CLASS, "source": src,
            "area_px": area, "seed_px": sp,
            # How far the watershed had to carry this seed. Near 1 means SAM
            # already had the plant; very large means one marker inherited a
            # blob far bigger than itself, which is the signature of two plants
            # merged into one instance.
            "growth": round(area / max(1, sp), 2),
            "bbox": instance_bbox(m)})

    depth_qa = {}
    if depth_mm is not None:
        instances, depth_qa = height_veto(instances, veg, depth_mm, cfg,
                                          conf=conf, polarity=polarity,
                                          fx=fx, fy=fy)

    union = np.zeros(veg.shape, bool)
    for inst in instances:
        union |= inst["mask"]
    veg_px = int(veg.sum())
    frame_px = veg.size
    qa = {
        "instances": len(instances),
        "from_sam": sum(1 for i in instances if i["source"] == "sam"),
        "recovered": sum(1 for i in instances if i["source"] == "vegetation"),
        "split": sum(1 for i in instances if i["source"] == "peak"),
        "veg_px": veg_px,
        # The number to watch. Vegetation outside every instance is plant an
        # annotator will never be shown, and in a prelabelled task nobody draws
        # what is not already there.
        "veg_coverage": round(int((veg & union).sum()) / veg_px, 4) if veg_px else 0.0,
        "max_growth": max((i["growth"] for i in instances), default=0.0),
        "max_instance_frac": round(
            max((i["area_px"] for i in instances), default=0) / frame_px, 4),
        **depth_qa,
    }
    return instances, veg, qa


def area_histogram(areas, cfg):
    """Where the size floors sit relative to the instances actually produced.

    On a pale, pebbly substrate a colour index cannot separate green-tinted
    mineral from a cotyledon, so SIZE is the only discriminator left - and the
    right floor depends on mount height, which is a property of your rig and
    not of this code. Real plants and gravel are usually an order of magnitude
    apart, so the two populations are obvious once printed. Put the floor in
    the gap between them rather than inheriting a number measured elsewhere."""
    a = sorted(int(v) for v in areas)
    if not a:
        return "no instances"
    def pct(p):
        return a[min(len(a) - 1, int(p * (len(a) - 1)))]
    floor = int(cfg.get("MIN_INSTANCE_AREA_PX", 0))
    tiny = sum(1 for v in a if v < floor * 4)
    return (f"instance area px: p10={pct(0.10)} p50={pct(0.50)} "
            f"p90={pct(0.90)} max={a[-1]} | {tiny} of {len(a)} "
            f"({tiny / len(a):.0%}) under 4x the {floor}px floor"
            + (" - suspect speckle, raise MIN_INSTANCE_AREA_PX and "
               "RECOVER_MIN_AREA_PX" if tiny > len(a) * 0.5 else ""))


def review_reasons(qa, cfg):
    """Why this frame should be corrected before the others. Empty = looks fine.

    Every one of these is a measurable symptom of a mask problem, not a guess
    at annotation difficulty."""
    why = []
    if qa["instances"] == 0:
        why.append("no instances")
    if qa["veg_px"] and qa["veg_coverage"] < cfg["REVIEW_MIN_COVERAGE"]:
        why.append(f"only {qa['veg_coverage']:.0%} of vegetation covered")
    if qa["max_growth"] > cfg["REVIEW_MAX_GROWTH"]:
        why.append(f"a seed grew {qa['max_growth']:.0f}x - possible merge")
    if qa["max_instance_frac"] > cfg["REVIEW_MAX_INSTANCE_FRAC"]:
        why.append(f"one instance covers {qa['max_instance_frac']:.0%} of the frame")
    return why


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
class PrelabelCoco:
    """COCO instance segmentation with the single `plant` category.

    Its category id sits far above the ontology's so that a prelabel file and a
    corrected file can be held side by side without either shadowing the
    other's ids, and so a stray `plant` surviving into a merged dataset reads
    as an unknown category instead of silently becoming class 1."""

    def __init__(self):
        self.images, self.anns = [], []
        self.categories = prelabel_categories()
        self._img = self._ann = 0

    def add_image(self, file_name, h, w):
        self._img += 1
        self.images.append({"id": self._img, "file_name": file_name,
                            "height": h, "width": w})
        return self._img

    def add_instance(self, image_id, polygons, bbox, area_px):
        self._ann += 1
        self.anns.append({"id": self._ann, "image_id": image_id,
                          "category_id": PRELABEL_CATEGORY_ID,
                          "segmentation": polygons, "area": float(area_px),
                          "bbox": [float(v) for v in bbox], "iscrowd": 0})
        return self._ann

    def dump(self, path):
        Path(path).write_text(json.dumps({
            "info": {"description": "SeeWeed3D SAM 3 mixed-scene prelabels "
                                    "(masks only, class assigned in CVAT)",
                     "date_created": datetime.now(timezone.utc).isoformat()},
            "licenses": [], "images": self.images, "annotations": self.anns,
            "categories": self.categories}, indent=2), encoding="utf-8")


def overlay(bgr, instances, scale):
    """Preview built to answer one question: is each plant its own mask?

    Instances are coloured by INDEX, not by class - every instance here has the
    same class, so a per-class palette would paint the frame one colour and
    show nothing. Adjacent instances in different colours is the whole signal.
    A filled tint shows extent; a bright outline shows the boundary; a white
    halo marks an instance SAM never proposed."""
    vis = bgr.copy()
    tint = np.zeros_like(bgr)
    for k, inst in enumerate(instances):
        # Golden-angle hue stepping keeps neighbouring indices far apart in
        # colour, so two adjacent instances are never near-identical shades.
        hue = int((k * 137) % 180)
        col = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]),
                           cv2.COLOR_HSV2BGR)[0, 0].tolist()
        tint[inst["mask"]] = col
    vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
    for k, inst in enumerate(instances):
        hue = int((k * 137) % 180)
        col = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]),
                           cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cnts, _ = cv2.findContours(inst["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if inst["source"] != "sam":
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 4)
        cv2.drawContours(vis, cnts, -1, col, 2)
    return cv2.resize(vis, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale != 1.0 else vis


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def prelabel_session(sid, session_dir, out_root, cfg, predictor, sam_fn):
    frames = pool_frames(session_dir)
    # Frame range BEFORE the limit, so LIMIT_PER_SESSION trials the selected
    # stretch rather than the first N frames of the whole session - which on
    # a split-zone drive would be entirely the wrong zone.
    spec = (cfg.get("ONLY_FRAMES") or {}).get(sid)
    if spec:
        from common.frame_spec import select_filenames
        before = len(frames)
        frames = select_filenames(frames, spec)
        print(f"  [{sid}] ONLY_FRAMES kept {len(frames)} of {before} pool frames")
        if not frames:
            print(f"  [{sid}] ONLY_FRAMES {spec} matched no pool frame - "
                  f"check the indices against meta/pool.csv")
            return None
    if cfg["LIMIT_PER_SESSION"]:
        frames = frames[:cfg["LIMIT_PER_SESSION"]]
    print_pool_report(sid, session_dir, len(frames))

    # Decided ONCE per session, not per frame: whether depth may be used at
    # all, and with what calibration and confidence polarity. Deciding it here
    # means a session that cannot support the veto says so before the GPU
    # starts rather than silently skipping it 800 times.
    from perception import ground as gr
    want = cfg.get("USE_DEPTH_HEIGHT", False)
    depth_kind = gr.session_depth_kind(session_dir)
    use_depth = bool(want) and gr.has_metric_depth(session_dir)
    if want is True and not use_depth:
        sys.exit(
            f"ERROR: [{sid}] USE_DEPTH_HEIGHT is True but this session's "
            f"depth_kind is {depth_kind!r}, not 'metric'. Only v2 (MKV/FFV1) "
            f"captures carry real millimetres; a v1 preview cannot be used as "
            f"height. Set it to \"auto\" to skip depth on sessions that lack "
            f"it, or drop this session from ONLY_SESSIONS.")
    fx = fy = polarity = None
    if use_depth:
        fx, fy = gr.calibration(session_dir)
        polarity = gr.confidence_polarity(session_dir)
        print(f"  [{sid}] depth: metric | height veto on "
              f"(>= {cfg['HEIGHT_MIN_MM']:.0f} mm above local soil)")
        if fx is None:
            print(f"  [!] no calibration.json - MIN_INSTANCE_AREA_MM2 cannot "
                  f"be applied, falling back to the pixel floor.")
        if polarity is None and (session_dir / "conf").is_dir():
            print(f"  [!] a confidence map exists but session.json does not "
                  f"record its polarity, so it is NOT used. Gating the wrong "
                  f"way round keeps exactly the pixels it should drop.")
    elif want:
        print(f"  [{sid}] depth: {depth_kind} - height veto off, colour and "
              f"pixel-size gates only")
    if not frames:
        print(f"  [{sid}] no pool frames - run extract_sessions.py first")
        return None

    out = out_root / sid
    cvat_dir = out / cfg["CVAT_READY_SUBDIR"]
    flag_dir = out / cfg["FLAGGED_RGB_SUBDIR"]
    for d in (cvat_dir, flag_dir, out / "masks"):
        d.mkdir(parents=True, exist_ok=True)
    if cfg["SAVE_PREVIEWS"]:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    coco, rows, flagged, review = PrelabelCoco(), [], [], []
    stats = {"frames": 0, "instances": 0, "flagged": 0, "empty": 0,
             "recovered": 0, "split": 0, "veg_px": 0, "veg_covered_px": 0}
    areas, relief, depth_frac = [], [], []

    prog = Progress(len(frames), f"[{sid}]", unit="frames")
    for fn in frames:
        rgb_path = session_dir / "rgb" / fn
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            prog.update(note="missing frame")
            continue
        proc = white_balance(bgr, cfg["WB_CAST_RATIO"]) if cfg["WHITE_BALANCE"] else bgr

        veg_pre = plant_pixels(proc, cfg)
        if float(veg_pre.mean()) > cfg["MAX_VEG_FRACTION"]:
            # Colour cast or glare: the prior owns every boundary here, so if
            # the prior has failed there is nothing trustworthy to export.
            flagged.append(fn)
            link_or_copy(rgb_path, flag_dir / fn)
            stats["frames"] += 1
            stats["flagged"] += 1
            prog.update(note=f"{stats['instances']} inst, {stats['flagged']} flagged")
            continue

        if cfg["SAM_PROMPT_MODE"] == "auto_exemplar":
            score_pre = plant_confidence(proc, veg_pre, cfg)
            exemplars = component_boxes(veg_pre, cfg["EXEMPLAR_MIN_AREA_PX"],
                                        cfg["EXEMPLAR_PAD_PX"],
                                        cfg["EXEMPLAR_MAX_BOXES"],
                                        confidence=score_pre,
                                        min_confidence=cfg["EXEMPLAR_MIN_VEG_SCORE"])
        else:
            exemplars = None

        sam_masks = sam_fn(predictor, proc, cfg, exemplars) if predictor else []
        depth_mm = conf_img = None
        if use_depth:
            dpath = session_dir / "depth" / fn
            if dpath.exists():
                depth_mm = gr.load_depth_mm(dpath)
            cpath = session_dir / "conf" / fn
            if depth_mm is not None and polarity and cpath.exists():
                conf_img = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)
        instances, veg, qa = analyze_frame(proc, sam_masks, cfg,
                                           depth_mm=depth_mm, conf=conf_img,
                                           polarity=polarity, fx=fx, fy=fy)

        link_or_copy(rgb_path, cvat_dir / fn)
        img_id = coco.add_image(fn, bgr.shape[0], bgr.shape[1])
        union = np.zeros(bgr.shape[:2], bool)
        for k, inst in enumerate(instances):
            polys = mask_polygons(inst["mask"], cfg)
            if not polys:
                continue
            coco.add_instance(img_id, polys, inst["bbox"], inst["area_px"])
            union |= inst["mask"]
            rows.append({"session_id": sid, "filename": fn, "instance_idx": k,
                         "source": inst["source"], "area_px": inst["area_px"],
                         "seed_px": inst["seed_px"], "growth": inst["growth"],
                         "bbox_x": inst["bbox"][0], "bbox_y": inst["bbox"][1],
                         "bbox_w": inst["bbox"][2], "bbox_h": inst["bbox"][3],
                         # Empty rather than 0 when unmeasured: a height of
                         # zero is a claim that the thing is flat, and "we
                         # could not tell" is a different statement.
                         "height_mm": inst.get("height_mm", ""),
                         "height_measured_frac":
                             inst.get("height_measured_frac", ""),
                         "area_mm2": inst.get("area_mm2", "")})

        why = review_reasons(qa, cfg)
        if why:
            review.append((fn, "; ".join(why)))

        cv2.imwrite(str(out / "masks" / fn), (union.astype(np.uint8) * 255))
        if cfg["SAVE_PREVIEWS"]:
            cv2.imwrite(str(out / "preview" / Path(fn).with_suffix(".jpg").name),
                        overlay(proc, instances, cfg["PREVIEW_SCALE"]))
        stats["frames"] += 1
        stats["instances"] += len(instances)
        stats["empty"] += int(not instances)
        stats["recovered"] += qa["recovered"]
        stats["split"] += qa.get("split", 0)
        for k2 in ("height_dropped_flat", "height_dropped_small",
                   "height_abstained"):
            stats[k2] = stats.get(k2, 0) + qa.get(k2, 0)
        if "ground_relief_mm" in qa:
            relief.append(qa["ground_relief_mm"])
            depth_frac.append(qa["depth_measured_frac"])
        areas.extend(i["area_px"] for i in instances)
        stats["veg_px"] += qa["veg_px"]
        stats["veg_covered_px"] += int((veg & union).sum())
        prog.update(note=f"{stats['instances']} inst, {stats['flagged']} flagged")

    prog.close(note=f"{stats['instances']} inst, {stats['flagged']} flagged")
    coco.dump(out / "instances_default.json")
    (out / "mixed_cvat_labels.json").write_text(
        json.dumps(prelabel_cvat_labels(), indent=2), encoding="utf-8")
    if flagged:
        (out / "flagged_for_manual.txt").write_text("\n".join(flagged),
                                                    encoding="utf-8")
    if review:
        (out / "review_first.txt").write_text(
            "\n".join(f"{f}\t{r}" for f, r in review), encoding="utf-8")
    if rows:
        with open(out / "instances.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"  [{sid}] {stats['frames']} frames | {stats['instances']} instances "
          f"| {stats['flagged']} flagged | {stats['empty']} with none")
    if stats["veg_px"]:
        cov = 100.0 * stats["veg_covered_px"] / stats["veg_px"]
        sam_n = stats["instances"] - stats["recovered"]
        # Coverage, not instance count, is the honest recall readout: the
        # pipeline can look productive while quietly leaving plants out.
        print(f"      coverage: {cov:.1f}% of vegetation inside an instance "
              f"| {sam_n - stats['split']} from SAM "
              f"+ {stats['split']} split from a merge "
              f"+ {stats['recovered']} recovered")
    if cfg.get("LOG_INSTANCE_AREAS", True) and areas:
        print(f"      {area_histogram(areas, cfg)}")
    if relief:
        flat = stats.get("height_dropped_flat", 0)
        small = stats.get("height_dropped_small", 0)
        abst = stats.get("height_abstained", 0)
        print(f"      depth: {sum(depth_frac) / len(depth_frac):.0%} of pixels "
              f"measured | ground relief {sum(relief) / len(relief):.0f} mm "
              f"| dropped {flat} flat + {small} undersized, abstained on {abst}")
        if abst > stats["instances"]:
            print(f"  [!] the height veto abstained on more instances than it "
                  f"kept. Stereo drops out on thin tissue, so that is expected "
                  f"on seedlings - but it means the gate is mostly not acting. "
                  f"Lower HEIGHT_MIN_MEASURED_FRAC only if the previews show "
                  f"it is missing real ground clutter.")
    if review:
        print(f"      {len(review)} frame(s) to review first -> review_first.txt")
    print(f"      -> {out}")
    return stats


def main(predictor_factory=load_sam3, sam_fn=sam3_instances):
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"])
    from common.dataset_paths import require_sessions_root
    sessions_root = require_sessions_root(root)
    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    if cfg["ONLY_SESSIONS"]:
        sids = [s for s in sids if s in cfg["ONLY_SESSIONS"]]
    if not sids:
        sys.exit("No sessions selected. Set ONLY_SESSIONS to your mixed sessions.")

    print(f"Mixed-scene instance prelabeling on {len(sids)} session(s) with SAM 3.")
    print(f"  One class ({PRELABEL_CLASS!r}) - masks only. Assign classes in CVAT.\n")
    predictor = predictor_factory(cfg)
    out_root = root / cfg["OUTPUT_SUBDIR"]
    for sid in sids:
        prelabel_session(sid, sessions_root / sid, out_root, cfg, predictor, sam_fn)

    print(f"\nDone -> {out_root}")
    print(f"Next: CVAT task from <sid>/{cfg['CVAT_READY_SUBDIR']}/, paste "
          f"<sid>/mixed_cvat_labels.json into the Raw label editor, import "
          f"<sid>/instances_default.json (COCO 1.0), then reassign each shape's "
          f"class. Anything still labelled {PRELABEL_CLASS!r} at the end is "
          f"unreviewed. Start with the frames in review_first.txt.")
    print(f"Shortcut order is the label order: "
          f"{', '.join(f'{i + 1}={c}' for i, c in enumerate(CLASSES))}.")


if __name__ == "__main__":
    main()
