"""Mixed-scene prelabeler: the mask logic, which is the whole deliverable.

The claim this module makes is that vegetation owns the boundary and SAM owns
the identity. Every test here is one half of that: SAM's proposals must not be
able to put soil into a mask or to keep a leaf out of one, and the split between
two plants must happen only where SAM saw two plants - never on a peak, which is
the failure that closed that door in the weed prelabeler.

SAM 3 is stubbed throughout (GPU + gated weights); every decision under test is
pure OpenCV/NumPy by design.
"""
import json

import cv2
import numpy as np
import pytest

from conftest import load_script

mx = load_script("annotation/prelabel_mixed_sam3.py")
ont = load_script("common/ontology.py")

CFG = mx.CONFIG


# --------------------------------------------------------------------------- #
# Synthetic scenes. Green tissue on brown soil, which is what the prior sees.
# --------------------------------------------------------------------------- #
SOIL = (60, 85, 120)          # BGR, a plausible brown
LEAF = (60, 190, 70)          # BGR, clearly green


def _scene(h=240, w=320):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = SOIL
    return img


def _blob(img, cx, cy, r, colour=LEAF):
    cv2.circle(img, (cx, cy), r, colour, -1)
    return img


def _disc(shape, cx, cy, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def _two_touching(h=240, w=320, gap=-4):
    """Two plants whose canopies meet: one vegetation component, a visible neck
    where they join. This is the case a colour index alone cannot separate."""
    img = _scene(h, w)
    r = 34
    cx1, cx2 = 130, 130 + 2 * r + gap
    _blob(img, cx1, 120, r)
    _blob(img, cx2, 120, r)
    return img, (cx1, 120), (cx2, 120), r


# --------------------------------------------------------------------------- #
# Bridging onion's glaucous, glossy leaf
#
# A real onion leaf is a glossy, waxy blue-green tube. That surface defeats
# the vegetation prior two ways at once: the wax bloom lifts blue reflectance
# past green (failing g>=b outright), and the glossy curve throws specular
# highlights carrying no leaf colour at all (failing every gate at once). Both
# cut across the leaf's WIDTH, so the gap connects to the surrounding soil and
# fill_holes() - which only fills gaps fully ENCLOSED by vegetation - cannot
# touch either one. Left unfixed, a single leaf comes out of vegetation_mask()
# broken into a handful of disconnected slivers, and every step downstream
# (SAM's exemplar boxes, seed intersection, the watershed's connectivity,
# fragment-dropping) inherits the damage and produces a scatter of speckle
# instances instead of one clean leaf.
# --------------------------------------------------------------------------- #
def _curved_leaf(h=200, w=300, thickness=13):
    """A tubular onion-leaf-like curve: thin, elongated, gently bent - not a
    disc, because that is the shape the real failure occurs on."""
    img = _scene(h, w)
    pts = np.array([[15, 170], [70, 55], [150, 35], [225, 85], [280, 175]],
                   np.int32)
    curve = np.zeros((h, w), np.uint8)
    cv2.polylines(curve, [pts], False, 1, thickness)
    img[curve.astype(bool)] = LEAF
    return img, curve.astype(bool)


def _afflict(img, leaf, n=14, seed=0):
    """Cut alternating specular-highlight and blue-shifted (glaucous) patches
    through the leaf's full width at even intervals along its length - a
    glossy tubular leaf under directional field lighting, not one stray spot."""
    out = img.copy()
    ys, xs = np.nonzero(leaf)
    order = np.argsort(xs)
    ys, xs = ys[order], xs[order]
    for j, i in enumerate(np.linspace(0, len(xs) - 1, n).astype(int)):
        colour = (235, 235, 235) if j % 2 == 0 else (150, 110, 40)
        cv2.circle(out, (int(xs[i]), int(ys[i])), 7, colour, -1)
    return out


def test_the_strict_gate_loses_a_quarter_of_an_afflicted_leaf():
    """Pins the failure this section exists to fix. Without it, this assertion
    would be the bug report.

    Measured as TISSUE LOST, not as a component count. Whether the survivors
    also come apart into separate blobs depends on the morphological open in
    vegetation_mask() and therefore on the OpenCV build: measured across
    kernels 0/1/3/5 the same afflicted leaf yields 1, 1, 4 and 8 components,
    while the fraction of leaf retained barely moves (0.737, 0.737, 0.724,
    0.705). An earlier version of this test asserted `components > 1` and so
    passed on one machine and failed on another, saying nothing about the code
    either time.

    Lost tissue is the precondition that actually matters: it is what the
    bridging recovers, and it is what makes every downstream stage - exemplar
    boxes, seed intersection, watershed connectivity, fragment dropping - see a
    leaf that is not there."""
    img, leaf = _curved_leaf()
    healthy = mx.vegetation_mask(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                                 CFG["VEG_MORPH_KERNEL"], CFG["VEG_MIN_COMPONENT_PX"])
    strict = mx.vegetation_mask(_afflict(img, leaf), CFG["EXG_THRESHOLD"],
                                CFG["VEG_MIN_SATURATION"], CFG["VEG_MORPH_KERNEL"],
                                CFG["VEG_MIN_COMPONENT_PX"])

    assert float((healthy & leaf).sum()) / leaf.sum() > 0.99, \
        "the undamaged leaf should pass the strict gate whole"
    assert float((strict & leaf).sum()) / leaf.sum() < 0.85, \
        "the affliction is not severe enough for this section to be about it"


def test_plant_pixels_reconnects_the_afflicted_leaf():
    img, leaf = _curved_leaf()
    sick = _afflict(img, leaf)
    fixed = mx.plant_pixels(sick, CFG)
    n, _, _, _ = cv2.connectedComponentsWithStats(fixed.astype(np.uint8), 8)
    assert n - 1 == 1
    assert float((fixed & leaf).sum()) / leaf.sum() > 0.8


def test_recovery_only_reaches_within_the_halo_of_confirmed_vegetation():
    """A bare patch of pale, low-saturation soil sitting on its own must earn
    nothing from either rule - only context next to already-confirmed
    vegetation does."""
    img = _scene()
    cv2.circle(img, (60, 120), 8, (235, 235, 235), -1)   # isolated highlight-
    cv2.circle(img, (240, 120), 8, (150, 110, 40), -1)   # -coloured soil, alone
    veg = mx.vegetation_mask(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                             CFG["VEG_MORPH_KERNEL"], 0)
    recovered = mx.recover_glaucous_pixels(img, veg, CFG)
    assert not recovered[112:128, 52:68].any()
    assert not recovered[112:128, 232:248].any()


def test_the_highlight_rule_needs_low_saturation_not_just_proximity():
    """Sitting in the halo is necessary but not sufficient - a halo pixel with
    ordinary saturation and no green must still be rejected."""
    img, leaf = _curved_leaf()
    veg = mx.vegetation_mask(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                             CFG["VEG_MORPH_KERNEL"], CFG["VEG_MIN_COMPONENT_PX"])
    saturated_non_green = img.copy()
    saturated_non_green[80:84, 140:144] = (30, 30, 220)   # saturated red, in the halo
    recovered = mx.recover_glaucous_pixels(saturated_non_green, veg, CFG)
    assert not recovered[80:84, 140:144].all()


def test_bridging_is_off_at_zero():
    img, leaf = _curved_leaf()
    sick = _afflict(img, leaf)
    veg = mx.vegetation_mask(sick, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                             CFG["VEG_MORPH_KERNEL"], CFG["VEG_MIN_COMPONENT_PX"])
    off = dict(CFG, VEG_BRIDGE_PX=0)
    assert (mx.recover_glaucous_pixels(sick, veg, off) == veg).all()


def test_closing_is_off_at_zero():
    img, leaf = _curved_leaf()
    veg = mx.vegetation_mask(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                             CFG["VEG_MORPH_KERNEL"], CFG["VEG_MIN_COMPONENT_PX"])
    assert (mx.close_thin_gaps(veg, 0) == veg).all()


def test_two_genuinely_separate_plants_do_not_merge_at_realistic_spacing():
    """The bridging exists to reconnect ONE afflicted leaf, not to erase the
    gap between two different plants. Bounded by construction - VEG_BRIDGE_PX
    plus VEG_CLOSE_KERNEL_PX - and this pins that the bound holds well below a
    typical inter-plant gap."""
    img = _scene(h=160, w=300)
    r = 30
    gap = 20
    cx1, cx2 = 150 - r - gap // 2, 150 + r + gap // 2
    _blob(img, cx1, 80, r)
    _blob(img, cx2, 80, r)
    fixed = mx.plant_pixels(img, CFG)
    n, _, _, _ = cv2.connectedComponentsWithStats(fixed.astype(np.uint8), 8)
    assert n - 1 == 2


def test_a_leaf_with_no_defects_is_unaffected():
    """The fix must be inert on tissue that never needed it."""
    img, leaf = _curved_leaf()
    plain = mx.vegetation_mask(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                               CFG["VEG_MORPH_KERNEL"], CFG["VEG_MIN_COMPONENT_PX"])
    fixed = mx.plant_pixels(img, CFG)
    assert float((fixed ^ plain).sum()) / plain.sum() < 0.05


def test_the_afflicted_leaf_now_seeds_one_instance_not_a_scatter():
    """End to end: the symptom that was actually reported - a real leaf coming
    out as a scatter of speckle instances rather than one clean mask."""
    img, leaf = _curved_leaf()
    sick = _afflict(img, leaf)
    ys, xs = np.nonzero(leaf)
    mid = len(ys) // 2
    cy, cx = int(ys[mid]), int(xs[mid])   # a point ON the curve, not its mean
    sam = [_disc(leaf.shape, cx, cy, 6)]
    instances, veg, qa = mx.analyze_frame(sick, sam, CFG)
    assert qa["instances"] <= 2          # not a dozen slivers
    assert qa["veg_coverage"] > 0.75


# --------------------------------------------------------------------------- #
# Pruning must run LAST, or bridging never gets a chance
#
# A first version of the bridge called vegetation_mask() with the real size
# floor already applied, so on a thin, heavily afflicted leaf every individual
# fragment could be smaller than VEG_MIN_COMPONENT_PX on its own and got
# deleted before bridging ever saw it. Reported as near-total recall
# collapse: real sessions came back with empty masks and empty previews on
# most frames, sparse AND dense alike.
# --------------------------------------------------------------------------- #
def _sparse_shoot(h=300, w=500, thickness=3, n_afflict=16, r=5, seed=0):
    """A thin, heavily afflicted shoot on pale soil - the sparser, harsher
    scene the collapse was reported on, distinct from _curved_leaf's thicker,
    denser one."""
    img = _scene(h, w); img[:] = (140, 150, 160)
    pts = [[30, 220], [100, 150], [180, 110], [260, 140], [340, 210]]
    curve = np.zeros((h, w), np.uint8)
    cv2.polylines(curve, [np.array(pts, np.int32)], False, 1, thickness)
    leaf = curve.astype(bool)
    img[leaf] = LEAF
    ys, xs = np.nonzero(leaf)
    order = np.argsort(xs)
    ys, xs = ys[order], xs[order]
    for j, i in enumerate(np.linspace(0, len(xs) - 1, n_afflict).astype(int)):
        colour = (235, 235, 235) if j % 2 == 0 else (150, 110, 40)
        cv2.circle(img, (int(xs[i]), int(ys[i])), r, colour, -1)
    return img, leaf


def test_pruning_before_bridging_would_have_deleted_the_whole_leaf():
    """Pins the mechanism, not just the symptom: with the floor applied before
    bridging ever runs, every individual fragment of this leaf is too small to
    survive on its own."""
    img, leaf = _sparse_shoot()
    pruned_first = mx.vegetation_mask(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                                      CFG["VEG_MORPH_KERNEL"], CFG["VEG_MIN_COMPONENT_PX"])
    n, _, stats, _ = cv2.connectedComponentsWithStats(pruned_first.astype(np.uint8), 8)
    assert n <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < CFG["VEG_MIN_COMPONENT_PX"] * 3


def test_plant_pixels_reconnects_the_sparse_shoot_pruning_second():
    img, leaf = _sparse_shoot()
    veg = mx.plant_pixels(img, CFG)
    assert veg.any()
    assert float((veg & leaf).sum()) / leaf.sum() > 0.7


def test_scattered_soil_noise_still_gets_nothing_from_pruning_last():
    """Deferring the size floor could in principle let sparse noise
    self-assemble into a survivor. It must not: spaced further apart than the
    bridge reach, isolated noise stays isolated and still gets pruned."""
    rng = np.random.default_rng(0)
    img = _scene(300, 400)
    for _ in range(40):
        x, y = int(rng.integers(10, 390)), int(rng.integers(10, 290))
        colour = (int(rng.integers(30, 90)), int(rng.integers(150, 220)),
                 int(rng.integers(30, 90)))
        cv2.circle(img, (x, y), 2, colour, -1)
    veg = mx.plant_pixels(img, CFG)
    assert not veg.any()


# --------------------------------------------------------------------------- #
# plant_confidence(): the recall/exemplar gate must not re-reject what
# bridging just admitted
#
# unseeded_components() and the exemplar confidence check both used to read
# vegetation_score() directly - the exact strict colour rule that fragmented
# the leaf in the first place. A pixel bridged in for failing THAT rule then
# scored near zero on it and got rejected all over again by the gate meant to
# keep noise out, discarding the material the bridge existed to save. This
# was not a rare edge case: it fired on every frame where bridging did real
# work, sparse or dense.
# --------------------------------------------------------------------------- #
def test_a_bridged_pixel_no_longer_scores_on_its_own_failing_colour():
    img, leaf = _curved_leaf()
    sick = _afflict(img, leaf)
    veg = mx.plant_pixels(sick, CFG)
    old_score = mx.vegetation_score(sick, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                                    CFG["VEG_SCORE_SOFTNESS"])
    fixed_score = mx.plant_confidence(sick, veg, CFG)
    strict = mx.strict_vegetation(sick, CFG)
    bridged_only = veg & ~strict
    assert bridged_only.any()
    assert float(fixed_score[bridged_only].mean()) > float(old_score[bridged_only].mean())


def test_reconnected_components_now_clear_the_recall_gate():
    """The concrete regression: real, reconnected leaf material - not
    noise - was scoring 0.6-0.7 against a 0.9 floor and being thrown away."""
    img, leaf = _sparse_shoot()
    veg = mx.plant_pixels(img, CFG)
    score = mx.plant_confidence(img, veg, CFG)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(veg.astype(np.uint8), 8)
    assert n > 1
    means = [float(score[lbl == i].mean()) for i in range(1, n)]
    assert max(means) >= CFG["RECOVER_MIN_VEG_SCORE"]


def test_exemplar_generation_now_actually_fires_on_an_afflicted_leaf():
    """The path PR #74's own tests never exercised: component_boxes() gated by
    the SAME confidence score prelabel_session actually uses to prompt SAM.
    Feeding hand-built SAM stubs into analyze_frame, as the earlier tests did,
    proved the mask logic worked but never proved SAM would ever be asked."""
    img, leaf = _curved_leaf()
    sick = _afflict(img, leaf)
    veg_pre = mx.plant_pixels(sick, CFG)
    score_pre = mx.plant_confidence(sick, veg_pre, CFG)
    boxes = mx.component_boxes(veg_pre, CFG["EXEMPLAR_MIN_AREA_PX"], CFG["EXEMPLAR_PAD_PX"],
                               CFG["EXEMPLAR_MAX_BOXES"], confidence=score_pre,
                               min_confidence=CFG["EXEMPLAR_MIN_VEG_SCORE"])
    assert len(boxes) >= 1


def test_the_old_score_would_have_produced_no_exemplars_at_all():
    """Regression pin for the mechanism, not just the outcome."""
    img, leaf = _curved_leaf()
    sick = _afflict(img, leaf)
    veg_pre = mx.plant_pixels(sick, CFG)
    old_score = mx.vegetation_score(sick, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                                    CFG["VEG_SCORE_SOFTNESS"])
    old_boxes = mx.component_boxes(veg_pre, CFG["EXEMPLAR_MIN_AREA_PX"], CFG["EXEMPLAR_PAD_PX"],
                                   CFG["EXEMPLAR_MAX_BOXES"], confidence=old_score,
                                   min_confidence=CFG["EXEMPLAR_MIN_VEG_SCORE"])
    assert old_boxes == []


def test_plant_confidence_is_identical_to_the_strict_score_when_nothing_bridged():
    """Must be inert on tissue that never needed bridging - a leaf with no
    defects gets the same confidence it always did."""
    img, leaf = _curved_leaf()
    veg = mx.plant_pixels(img, CFG)
    fixed_score = mx.plant_confidence(img, veg, CFG)
    old_score = mx.vegetation_score(img, CFG["EXG_THRESHOLD"], CFG["VEG_MIN_SATURATION"],
                                    CFG["VEG_SCORE_SOFTNESS"])
    assert np.allclose(fixed_score, old_score)


def test_isolated_noise_still_scores_low_with_no_confirmed_neighbour():
    """A bridged pixel borrows confidence from a nearby STRICT pixel. Noise
    with nothing strict-passing nearby has nothing to borrow from and must
    stay low."""
    img = _scene(200, 200)
    score = mx.plant_confidence(img, np.zeros((200, 200), bool), CFG)
    assert float(score.max()) < 0.5


# --------------------------------------------------------------------------- #
# The vegetation prior owns the boundary
# --------------------------------------------------------------------------- #
def test_a_proposal_that_bleeds_onto_soil_contributes_no_soil():
    """SAM boundaries are a learned prior at its own working resolution and do
    bleed. Nothing downstream can put a soil pixel back, so an over-reaching
    proposal must lose the excess here rather than be trusted."""
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    sloppy = _disc(veg.shape, 160, 120, 40)          # reaches well past the plant
    seeds = mx.seed_masks([sloppy], veg, CFG)
    assert len(seeds) == 1
    assert not (seeds[0] & ~veg).any()


def test_a_proposal_that_clips_the_leaf_tips_is_grown_back_out():
    """Intersecting alone cannot fix an undershoot - the tissue is already gone.
    The watershed is what recovers it, by carrying the seed out to the real
    vegetation edge."""
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    timid = _disc(veg.shape, 160, 120, 12)           # a fraction of the plant
    grown = mx.partition_vegetation(veg, [timid])[0]
    assert grown.sum() > timid.sum() * 3
    # Grown to the plant, not past it.
    assert not (grown & ~veg).any()
    assert grown.sum() == pytest.approx(veg.sum(), rel=0.02)


def test_no_instance_pixel_is_ever_off_the_vegetation_prior():
    img, c1, c2, r = _two_touching()
    veg = mx.plant_pixels(img, CFG)
    seeds = [_disc(veg.shape, *c1, 10), _disc(veg.shape, *c2, 10)]
    for m in mx.partition_vegetation(veg, seeds):
        assert not (m & ~veg).any()


# --------------------------------------------------------------------------- #
# SAM owns the identity: splits happen where SAM saw two plants, and only there
# --------------------------------------------------------------------------- #
def test_two_seeds_in_one_blob_are_cut_apart_at_the_neck():
    img, c1, c2, r = _two_touching()
    veg = mx.plant_pixels(img, CFG)
    # One connected component - a colour index alone would return one plant.
    n, _, _, _ = cv2.connectedComponentsWithStats(veg.astype(np.uint8), 8)
    assert n - 1 == 1

    parts = mx.partition_vegetation(veg, [_disc(veg.shape, *c1, 10),
                                          _disc(veg.shape, *c2, 10)])
    assert len(parts) == 2 and all(p.any() for p in parts)
    # Each part keeps its own seed's plant and not the other's.
    assert parts[0][c1[1], c1[0]] and not parts[0][c2[1], c2[0]]
    assert parts[1][c2[1], c2[0]] and not parts[1][c1[1], c1[0]]
    # The cut lands between the two centres, not off in one plant's interior.
    xs = np.nonzero(parts[0].any(axis=0))[0]
    assert c1[0] < xs.max() < c2[0]


def test_the_two_parts_do_not_share_a_single_pixel():
    """A pixel belonging to two instances is not a training target. Watershed
    ridge pixels are left unassigned, which is the right answer."""
    img, c1, c2, r = _two_touching()
    veg = mx.plant_pixels(img, CFG)
    a, b = mx.partition_vegetation(veg, [_disc(veg.shape, *c1, 10),
                                         _disc(veg.shape, *c2, 10)])
    assert not (a & b).any()


def test_one_seed_on_a_lobed_plant_is_never_split():
    """THE regression this design exists to avoid. prelabel_weeds_sam3 ships
    SPLIT_TOUCHING_INSTANCES off because splitting on distance-transform peaks
    fragmented single plants - a leaf reaching away from the crown raises a
    second peak. Here the markers come from SAM, so one proposal can only ever
    produce one instance no matter how many peaks its shape has."""
    img = _scene()
    for k in range(6):                                # a crown with long arms
        a = 2 * np.pi * k / 6
        cv2.line(img, (160, 120),
                 (int(160 + 60 * np.cos(a)), int(120 + 60 * np.sin(a))),
                 LEAF, 11)
    cv2.circle(img, (160, 120), 16, LEAF, -1)
    veg = mx.plant_pixels(img, CFG)

    dt = cv2.distanceTransform(veg.astype(np.uint8), cv2.DIST_L2, 5)
    assert dt.max() > 0                               # shape does have relief
    parts = mx.partition_vegetation(veg, [_disc(veg.shape, 160, 120, 10)])
    assert len(parts) == 1
    assert parts[0].sum() == pytest.approx(veg.sum(), rel=0.02)


def test_thin_arm_tips_survive_the_watershed():
    """Regression. Seeding the soil as background - the obvious way to give the
    flood somewhere to stop - ate four of this rosette's six arm tips: soil is
    flat ground at the top of the relief, so the background flood reaches a
    near-flat arm tip before the flood coming up the arm from the crown does.
    Thin tissue is precisely what this pass must not lose."""
    img = _scene()
    for k in range(6):
        a = 2 * np.pi * k / 6
        cv2.line(img, (160, 120),
                 (int(160 + 60 * np.cos(a)), int(120 + 60 * np.sin(a))),
                 LEAF, 11)
    veg = mx.plant_pixels(img, CFG)
    part = mx.partition_vegetation(veg, [_disc(veg.shape, 160, 120, 10)])[0]
    for k in range(6):                                # every tip, individually
        a = 2 * np.pi * k / 6
        tip = (int(120 + 55 * np.sin(a)), int(160 + 55 * np.cos(a)))
        assert veg[tip] and part[tip], f"arm {k} lost its tip"


def test_a_plant_at_the_frame_edge_keeps_its_edge_pixels():
    """OpenCV stamps the image border as watershed ridge, so a plant running
    off the top of the frame - which is most of them, on a moving robot -
    would otherwise be trimmed."""
    img = _scene()
    cv2.circle(img, (160, 0), 40, LEAF, -1)
    veg = mx.plant_pixels(img, CFG)
    assert veg[0].any()
    part = mx.partition_vegetation(veg, [_disc(veg.shape, 160, 10, 8)])[0]
    assert (part[0] == veg[0]).all()


def test_an_instance_cannot_claim_vegetation_across_bare_soil():
    """With no background marker the flood runs over soil too, so a territory
    can include a component the seed never touched. Intersecting with the
    prior does not catch it - the stolen component is real vegetation."""
    img = _blob(_blob(_scene(), 80, 120, 26), 250, 120, 26)
    veg = mx.plant_pixels(img, CFG)
    part = mx.partition_vegetation(veg, [_disc(veg.shape, 80, 120, 10)])[0]
    assert part[120, 80]                              # its own plant
    assert not part[120, 250]                         # not the far one


def test_results_line_up_with_the_seeds_they_came_from():
    """The caller keeps per-seed metadata (source, seed area) positionally, so
    a seed whose territory vanishes must still occupy its slot."""
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    empty = np.zeros(veg.shape, bool)
    parts = mx.partition_vegetation(veg, [_disc(veg.shape, 160, 120, 10), empty])
    assert len(parts) == 2
    assert parts[0].any() and not parts[1].any()


# --------------------------------------------------------------------------- #
# Seed gating
# --------------------------------------------------------------------------- #
def test_a_proposal_sitting_mostly_on_soil_is_rejected():
    """Judged before intersecting: a detection mostly on bare ground is a bad
    detection, and its green sliver must not be promoted to a plant."""
    img = _blob(_scene(), 60, 60, 12)
    veg = mx.plant_pixels(img, CFG)
    mostly_soil = _disc(veg.shape, 60, 60, 60)
    assert mx.seed_masks([mostly_soil], veg, CFG) == []


def test_duplicate_proposals_of_one_plant_become_one_seed():
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    d = _disc(veg.shape, 160, 120, 30)
    assert len(mx.seed_masks([d, d.copy(), d.copy()], veg, CFG)) == 1


def test_two_genuinely_different_plants_survive_nms():
    img, c1, c2, _ = _two_touching(gap=40)
    veg = mx.plant_pixels(img, CFG)
    seeds = mx.seed_masks([_disc(veg.shape, *c1, 30),
                           _disc(veg.shape, *c2, 30)], veg, CFG)
    assert len(seeds) == 2


def test_erosion_never_erases_a_seedling():
    """A cotyledon that passed every other gate must not be deleted for
    tidiness - small weeds are the recall this project is already short of."""
    tiny = _disc((240, 320), 100, 100, 4)
    assert mx.erode_seed(tiny, CFG).any()


def test_erosion_does_pull_a_large_seed_off_its_boundary():
    big = _disc((240, 320), 160, 120, 40)
    assert mx.erode_seed(big, CFG).sum() < big.sum()


# --------------------------------------------------------------------------- #
# The recall backstop
# --------------------------------------------------------------------------- #
def test_green_that_sam_returned_nothing_for_becomes_an_instance():
    """In a prelabelled task nobody draws what is not already there, so a plant
    left out of the export is a plant the annotator never sees."""
    img = _blob(_blob(_scene(), 100, 120, 28), 240, 120, 28)
    veg = mx.plant_pixels(img, CFG)
    score = mx.vegetation_score(img, CFG["EXG_THRESHOLD"],
                                CFG["VEG_MIN_SATURATION"],
                                CFG["VEG_SCORE_SOFTNESS"])
    seeds = mx.seed_masks([_disc(veg.shape, 100, 120, 25)], veg, CFG)
    extra = mx.unseeded_components(veg, seeds, CFG, score)
    assert len(extra) == 1
    assert extra[0][120, 240]                         # the one SAM missed


def test_a_plant_sam_already_found_is_not_recovered_twice():
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    score = mx.vegetation_score(img, CFG["EXG_THRESHOLD"],
                                CFG["VEG_MIN_SATURATION"],
                                CFG["VEG_SCORE_SOFTNESS"])
    seeds = mx.seed_masks([_disc(veg.shape, 160, 120, 25)], veg, CFG)
    assert mx.unseeded_components(veg, seeds, CFG, score) == []


def test_the_backstop_can_be_turned_off():
    """It is on here and off in the weed prelabeler, and the difference is a
    judgement call - so it has to be reachable from config."""
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    off = dict(CFG, RECOVER_UNSEEDED=False)
    assert mx.unseeded_components(veg, [], off, None) == []
    assert mx.unseeded_components(veg, [], CFG, None) != []


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def test_a_specular_pinhole_does_not_become_a_hole_in_the_polygon():
    m = _disc((240, 320), 160, 120, 40)
    m[118:123, 158:163] = False                       # highlight punched through
    assert mx.fill_holes(m, CFG["FILL_HOLES_MAX_PX"])[120, 160]


def test_soil_between_two_leaves_is_not_filled_in():
    """Hole filling is bounded by area precisely so a real gap stays soil."""
    m = np.zeros((240, 320), bool)
    cv2.circle(m.view(np.uint8), (160, 120), 90, 1, 12)   # a ring, hollow centre
    assert not mx.fill_holes(m, CFG["FILL_HOLES_MAX_PX"])[120, 160]


def test_specks_are_dropped_but_a_second_lobe_survives():
    m = _disc((240, 320), 100, 120, 30) | _disc((240, 320), 170, 120, 22)
    m[10, 10] = True
    cleaned = mx.clean_instance(m, CFG)
    assert cleaned[120, 100] and cleaned[120, 170]
    assert not cleaned[10, 10]


def test_an_instance_reduced_to_nothing_is_dropped():
    assert mx.clean_instance(np.zeros((50, 50), bool), CFG) is None


# --------------------------------------------------------------------------- #
# One class, and why
# --------------------------------------------------------------------------- #
def test_every_instance_carries_the_neutral_class():
    """Shape separates a blade from a rosette, and an onion IS a blade - so the
    one confident morphology call would label the crop as grass_weed."""
    img, c1, c2, _ = _two_touching(gap=40)
    veg = mx.plant_pixels(img, CFG)
    sam = [_disc(veg.shape, *c1, 28), _disc(veg.shape, *c2, 28)]
    instances, _, _ = mx.analyze_frame(img, sam, CFG)
    assert instances
    assert {i["cls"] for i in instances} == {ont.PRELABEL_CLASS}


def test_the_sentinel_is_not_a_trainable_class():
    """A model trained on 'plant' has learned nothing this project needs, and
    the crop-safety metrics would have no crop to measure."""
    assert ont.PRELABEL_CLASS not in ont.CLASSES
    assert ont.PRELABEL_CATEGORY_ID not in ont.CATEGORY_ID.values()


def test_the_cvat_schema_puts_real_classes_on_the_low_shortcuts():
    """Label order is shortcut order in CVAT, and reassigning is the entire
    job - the keystrokes have to land on classes actually used."""
    labels = ont.prelabel_cvat_labels()
    names = [lbl["name"] for lbl in labels]
    assert names[:len(ont.CLASSES)] == ont.CLASSES
    assert names.index(ont.PRELABEL_CLASS) > names.index(ont.CLASSES[-1])


def test_the_schema_offers_every_class_an_annotator_could_need():
    names = {lbl["name"] for lbl in ont.prelabel_cvat_labels()}
    assert set(ont.CLASSES) <= names
    assert ont.CROP_CLASS in names                    # the crop above all
    assert ont.IGNORE_LABEL in names


def test_no_lep_label_is_offered():
    """LEP is explicitly out of scope for this pass, and a shape type nobody
    draws still costs a slot in the shortcut list."""
    names = {lbl["name"] for lbl in ont.prelabel_cvat_labels()}
    assert ont.LEP_LABEL not in names


def test_the_schema_is_json_serialisable_for_the_raw_editor():
    json.loads(json.dumps(ont.prelabel_cvat_labels()))


def test_every_attribute_has_a_non_empty_values_array():
    """CVAT rejects the whole schema otherwise - including for text
    attributes, where the array holds the default."""
    for lbl in ont.prelabel_cvat_labels():
        for a in lbl["attributes"]:
            assert isinstance(a["values"], list) and a["values"]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_the_coco_export_uses_the_sentinel_category(tmp_path):
    c = mx.PrelabelCoco()
    img = c.add_image("f.png", 240, 320)
    c.add_instance(img, [[0, 0, 10, 0, 10, 10]], [0, 0, 10, 10], 50)
    c.dump(tmp_path / "i.json")
    d = json.loads((tmp_path / "i.json").read_text())
    assert [x["name"] for x in d["categories"]] == [ont.PRELABEL_CLASS]
    assert d["annotations"][0]["category_id"] == ont.PRELABEL_CATEGORY_ID


def test_the_sentinel_id_cannot_shadow_an_ontology_id(tmp_path):
    """A prelabel file and a corrected file get held side by side. A stray
    'plant' surviving into a merged dataset must read as an unknown category,
    not silently become class 1."""
    assert ont.PRELABEL_CATEGORY_ID > max(ont.CATEGORY_ID.values())


def test_area_is_the_mask_area_not_the_bbox_area(tmp_path):
    """COCO area drives the small/medium/large split this project measures, and
    a bbox badly overstates a thin or lobed plant."""
    c = mx.PrelabelCoco()
    img = c.add_image("f.png", 240, 320)
    c.add_instance(img, [[0, 0, 100, 0, 100, 100]], [0, 0, 100, 100], 137)
    assert c.anns[0]["area"] == 137.0


def test_detached_tissue_is_exported_rather_than_dropped():
    """POLY_ALL_PARTS is True here and False in the weed prelabeler. Dropping a
    leaf an occluding blade separated from its crown puts real plant into the
    training target as background - the exact imprecision this pass removes."""
    assert CFG["POLY_ALL_PARTS"] is True
    m = _disc((240, 320), 100, 120, 30) | _disc((240, 320), 200, 120, 26)
    assert len(mx.mask_polygons(m, CFG)) == 2


# --------------------------------------------------------------------------- #
# Frame-level QA, which is how mask quality gets checked at all
# --------------------------------------------------------------------------- #
def test_coverage_reports_vegetation_left_outside_every_instance():
    img = _blob(_blob(_scene(), 100, 120, 28), 240, 120, 28)
    veg = mx.plant_pixels(img, CFG)
    off = dict(CFG, RECOVER_UNSEEDED=False)           # so the miss stays missed
    sam = [_disc(veg.shape, 100, 120, 25)]
    _, _, qa = mx.analyze_frame(img, sam, off)
    assert 0.3 < qa["veg_coverage"] < 0.7             # one of two plants
    assert "only" in " ".join(mx.review_reasons(qa, off))


def test_full_coverage_raises_no_review_flag():
    img, c1, c2, _ = _two_touching(gap=40)
    veg = mx.plant_pixels(img, CFG)
    sam = [_disc(veg.shape, *c1, 28), _disc(veg.shape, *c2, 28)]
    _, _, qa = mx.analyze_frame(img, sam, CFG)
    assert qa["veg_coverage"] > 0.99
    assert mx.review_reasons(qa, CFG) == []


def test_a_frame_with_nothing_found_is_flagged_for_review():
    qa = {"instances": 0, "veg_px": 5000, "veg_coverage": 0.0,
          "max_growth": 0.0, "max_instance_frac": 0.0}
    assert "no instances" in mx.review_reasons(qa, CFG)


def test_a_seed_that_inherited_a_huge_blob_is_flagged_as_a_possible_merge():
    """One marker growing enormously is the signature of two plants that ended
    up as one instance - the under-segmentation this design cannot fix by
    itself, so it says so instead."""
    qa = {"instances": 1, "veg_px": 5000, "veg_coverage": 1.0,
          "max_growth": 40.0, "max_instance_frac": 0.01}
    assert any("merge" in r for r in mx.review_reasons(qa, CFG))


def test_growth_is_recorded_per_instance():
    img = _blob(_scene(), 160, 120, 30)
    veg = mx.plant_pixels(img, CFG)
    inst, _, _ = mx.analyze_frame(img, [_disc(veg.shape, 160, 120, 8)], CFG)
    assert inst and inst[0]["growth"] > 1.0
    assert inst[0]["seed_px"] < inst[0]["area_px"]


def test_recovered_instances_stay_distinguishable_from_sam_ones():
    """The two populations have different reliability, so an audit has to be
    able to separate them."""
    img = _blob(_blob(_scene(), 100, 120, 28), 240, 120, 28)
    veg = mx.plant_pixels(img, CFG)
    inst, _, qa = mx.analyze_frame(img, [_disc(veg.shape, 100, 120, 25)], CFG)
    assert qa["from_sam"] == 1 and qa["recovered"] == 1
    assert {i["source"] for i in inst} == {"sam", "vegetation"}


# --------------------------------------------------------------------------- #
# End to end on the stub
# --------------------------------------------------------------------------- #
def test_a_frame_of_touching_plants_comes_out_as_separate_instances():
    img, c1, c2, _ = _two_touching()
    veg = mx.plant_pixels(img, CFG)
    sam = [_disc(veg.shape, *c1, 26), _disc(veg.shape, *c2, 26)]
    instances, _, qa = mx.analyze_frame(img, sam, CFG)
    assert qa["instances"] == 2
    a, b = (i["mask"] for i in instances)
    assert not (a & b).any()
    assert qa["veg_coverage"] > 0.95


def test_an_empty_frame_produces_nothing_rather_than_failing():
    instances, veg, qa = mx.analyze_frame(_scene(), [], CFG)
    assert instances == [] and not veg.any()
    assert qa["instances"] == 0 and qa["veg_coverage"] == 0.0


def _session(tmp_path, n=2):
    sess = tmp_path / "sess"
    (sess / "rgb").mkdir(parents=True)
    (sess / "meta").mkdir(parents=True)
    names = []
    for i in range(n):
        img, c1, c2, _ = _two_touching()
        cv2.imwrite(str(sess / "rgb" / f"f{i}.png"), img)
        names.append(f"f{i}.png")
    (sess / "meta" / "pool.csv").write_text(
        "filename\n" + "\n".join(names) + "\n", encoding="utf-8")
    return sess, c1, c2


def _stub(centres, radius=26):
    def sam(pred, image, cfg, exemplars=None):
        shape = image.shape[:2]
        return [_disc(shape, *c, radius) for c in centres]
    return sam


def test_a_session_exports_everything_cvat_needs(tmp_path):
    sess, c1, c2 = _session(tmp_path)
    out = tmp_path / "out"
    st = mx.prelabel_session("sess", sess, out, CFG, "STUB", _stub([c1, c2]))
    assert st["frames"] == 2 and st["instances"] == 4

    d = out / "sess"
    assert (d / "instances_default.json").exists()
    assert (d / "mixed_cvat_labels.json").exists()
    # The images CVAT needs, beside the annotations that reference them.
    coco = json.loads((d / "instances_default.json").read_text())
    for im in coco["images"]:
        assert (d / CFG["CVAT_READY_SUBDIR"] / im["file_name"]).exists()
    assert len(coco["annotations"]) == 4
    assert {a["category_id"] for a in coco["annotations"]} == \
        {ont.PRELABEL_CATEGORY_ID}


def test_the_exported_schema_matches_the_exported_categories(tmp_path):
    """A category name the task has no label for fails the CVAT import."""
    sess, c1, c2 = _session(tmp_path, n=1)
    out = tmp_path / "out"
    mx.prelabel_session("sess", sess, out, CFG, "STUB", _stub([c1, c2]))
    d = out / "sess"
    coco = json.loads((d / "instances_default.json").read_text())
    labels = json.loads((d / "mixed_cvat_labels.json").read_text())
    assert {c["name"] for c in coco["categories"]} <= {lbl["name"] for lbl in labels}


def test_a_frame_with_no_pool_row_is_never_processed(tmp_path):
    sess, c1, c2 = _session(tmp_path, n=1)
    cv2.imwrite(str(sess / "rgb" / "not_pooled.png"), _scene())
    out = tmp_path / "out"
    st = mx.prelabel_session("sess", sess, out, CFG, "STUB", _stub([c1, c2]))
    assert st["frames"] == 1


def test_review_triage_is_written_when_a_frame_looks_wrong(tmp_path):
    """The point of the list: correct the frames whose numbers say the masks
    are bad, instead of finding them at random halfway through the task."""
    sess, c1, c2 = _session(tmp_path, n=1)
    out = tmp_path / "out"
    # No SAM proposals and no backstop, so the vegetation goes uncovered.
    cfg = dict(CFG, RECOVER_UNSEEDED=False)
    mx.prelabel_session("sess", sess, out, cfg, "STUB", _stub([]))
    txt = (out / "sess" / "review_first.txt").read_text()
    assert "f0.png" in txt and "vegetation covered" in txt


def test_a_clean_session_writes_no_review_list(tmp_path):
    sess, c1, c2 = _session(tmp_path, n=1)
    out = tmp_path / "out"
    mx.prelabel_session("sess", sess, out, CFG, "STUB", _stub([c1, c2]))
    assert not (out / "sess" / "review_first.txt").exists()


def test_a_glare_frame_is_flagged_rather_than_exported(tmp_path):
    """The prior owns every boundary here, so if the prior has failed there is
    nothing trustworthy to export."""
    sess = tmp_path / "sess"
    (sess / "rgb").mkdir(parents=True)
    (sess / "meta").mkdir(parents=True)
    green = np.zeros((240, 320, 3), np.uint8)
    green[:] = LEAF                                   # the whole frame is green
    cv2.imwrite(str(sess / "rgb" / "g.png"), green)
    (sess / "meta" / "pool.csv").write_text("filename\ng.png\n", encoding="utf-8")
    out = tmp_path / "out"
    # White balance off, so the guard itself is what is under test: gray-world
    # already cancels a perfectly uniform cast, which would leave nothing for
    # the guard to catch and prove nothing about it.
    cfg = dict(CFG, WHITE_BALANCE=False)
    st = mx.prelabel_session("sess", sess, out, cfg, "STUB", _stub([(160, 120)]))
    assert st["flagged"] == 1 and st["instances"] == 0
    assert (out / "sess" / CFG["FLAGGED_RGB_SUBDIR"] / "g.png").exists()
    assert "g.png" in (out / "sess" / "flagged_for_manual.txt").read_text()


def test_the_preview_colours_by_instance_not_by_class():
    """Every instance shares one class here, so a per-class palette would paint
    the frame a single colour and show nothing about separation."""
    img, c1, c2, _ = _two_touching()
    veg = mx.plant_pixels(img, CFG)
    sam = [_disc(veg.shape, *c1, 26), _disc(veg.shape, *c2, 26)]
    instances, _, _ = mx.analyze_frame(img, sam, CFG)
    vis = mx.overlay(img, instances, 1.0)
    left = vis[instances[0]["mask"]].mean(axis=0)
    right = vis[instances[1]["mask"]].mean(axis=0)
    assert np.abs(left - right).max() > 25


# --------------------------------------------------------------------------- #
# A drive that changes crop zone partway through
#
# Weeds only for the first stretch, then onions only. Not a mixed scene - two
# single-class scenes end to end, where each half wants the prelabeler whose
# assumption actually holds there.
# --------------------------------------------------------------------------- #
def _split_zone_session(tmp_path, n=8):
    """A session whose pool frames span two zones by frame index."""
    sess = tmp_path / "sess"
    (sess / "rgb").mkdir(parents=True)
    (sess / "meta").mkdir(parents=True)
    names = []
    for i in range(n):
        img, c1, c2, _ = _two_touching()
        fn = f"sess_{i * 10:06d}.png"
        cv2.imwrite(str(sess / "rgb" / fn), img)
        names.append(fn)
    (sess / "meta" / "pool.csv").write_text(
        "filename\n" + "\n".join(names) + "\n", encoding="utf-8")
    return sess, c1, c2


def test_only_frames_restricts_a_run_to_one_zone(tmp_path):
    sess, c1, c2 = _split_zone_session(tmp_path)
    cfg = dict(CFG, ONLY_FRAMES={"sess": ["0-30"]})   # first 4 frames: 0,10,20,30
    out = tmp_path / "out"
    st = mx.prelabel_session("sess", sess, out, cfg, "STUB", _stub([c1, c2]))
    assert st["frames"] == 4
    coco = json.loads((out / "sess" / "instances_default.json").read_text())
    assert {im["file_name"] for im in coco["images"]} == {
        "sess_000000.png", "sess_000010.png", "sess_000020.png",
        "sess_000030.png"}


def test_the_other_zone_is_a_separate_disjoint_run(tmp_path):
    """The two halves must not overlap - a frame prelabelled under both
    assumptions would reach CVAT twice with contradictory classes."""
    sess, c1, c2 = _split_zone_session(tmp_path)
    a = mx.prelabel_session("sess", sess, tmp_path / "a",
                            dict(CFG, ONLY_FRAMES={"sess": ["0-30"]}),
                            "STUB", _stub([c1, c2]))
    b = mx.prelabel_session("sess", sess, tmp_path / "b",
                            dict(CFG, ONLY_FRAMES={"sess": ["40-70"]}),
                            "STUB", _stub([c1, c2]))
    assert a["frames"] == 4 and b["frames"] == 4
    ca = json.loads((tmp_path / "a" / "sess" / "instances_default.json").read_text())
    cb = json.loads((tmp_path / "b" / "sess" / "instances_default.json").read_text())
    na = {im["file_name"] for im in ca["images"]}
    nb = {im["file_name"] for im in cb["images"]}
    assert not (na & nb)


def test_a_gap_can_be_left_at_the_transition(tmp_path):
    """The frames where one zone becomes the other are where a single-class
    assumption is most dangerous, so it must be possible to leave them out of
    BOTH specialised runs."""
    sess, c1, c2 = _split_zone_session(tmp_path)
    a = mx.prelabel_session("sess", sess, tmp_path / "a",
                            dict(CFG, ONLY_FRAMES={"sess": ["0-20"]}),
                            "STUB", _stub([c1, c2]))
    b = mx.prelabel_session("sess", sess, tmp_path / "b",
                            dict(CFG, ONLY_FRAMES={"sess": ["50-70"]}),
                            "STUB", _stub([c1, c2]))
    assert a["frames"] == 3 and b["frames"] == 3      # 30 and 40 left out


# --------------------------------------------------------------------------- #
# Merged plants: the watershed was given one marker for a region holding four
# --------------------------------------------------------------------------- #
def _rosettes(centres, r=45, size=400, neck=10):
    """Several compact plants joined into ONE vegetation component."""
    m = np.zeros((size, size), np.uint8)
    for cx, cy in centres:
        cv2.circle(m, (cx, cy), r, 1, -1)
    for a, b in zip(centres, centres[1:]):
        cv2.line(m, a, b, 1, neck)
    return m.astype(bool)


def test_a_group_of_merged_plants_is_split_into_one_each():
    """The reported symptom: `a seed grew 91x`. Four rosettes grown together
    are one connected component, so a single SAM proposal inherits the lot."""
    veg = _rosettes([(90, 120), (230, 120), (90, 280), (230, 280)])
    seed = _disc(veg.shape, 90, 120, 12)
    extra = mx.peak_seeds(veg, [seed], CFG)
    assert len(extra) >= 3, "the other three crowns got no marker"


def test_a_component_sam_seeded_properly_is_never_re_seeded():
    """The guard that stops this fragmenting correct work: only components
    whose flood would exceed the growth trigger are touched."""
    veg = _rosettes([(90, 150), (230, 150)])
    seeds = [_disc(veg.shape, 90, 150, 40), _disc(veg.shape, 230, 150, 40)]
    assert mx.peak_seeds(veg, seeds, CFG) == []


def test_a_long_leaf_is_never_split_however_under_seeded():
    """A ribbon has a FLAT distance ridge, so every point along it is a local
    maximum. Without the elongation guard this shattered a synthetic onion leaf
    into 11 instances - the exact failure that closed this door in the weed
    prelabeler."""
    leaf = np.zeros((300, 300), np.uint8)
    cv2.line(leaf, (30, 150), (270, 150), 1, 24)
    leaf = leaf.astype(bool)
    tiny = _disc(leaf.shape, 40, 150, 5)
    assert mx.peak_seeds(leaf, [tiny], CFG) == []


def test_the_elongation_ceiling_is_what_rejects_it():
    """Stated as a measurement so the threshold can be re-derived rather than
    trusted: area over the area of the component's own largest inscribed disc."""
    def spread(m):
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        return int(m.sum()) / (np.pi * float(dt.max()) ** 2)

    leaf = np.zeros((300, 300), np.uint8)
    cv2.line(leaf, (30, 150), (270, 150), 1, 24)
    one = np.zeros((300, 300), np.uint8)
    cv2.circle(one, (150, 150), 50, 1, -1)

    four = _rosettes([(90, 120), (230, 120), (90, 280), (230, 280)])
    # The ceiling has to sit between the largest merge worth splitting and the
    # most compact ribbon. Measured: 1.0 one rosette, 2.1 two merged, 4.1 four
    # merged, 12.1 a straight leaf, 27.3 a curved one - so 8 has margin on both
    # sides rather than being a value that happens to work.
    assert spread(one.astype(bool)) < 1.5
    assert spread(_rosettes([(90, 150), (230, 150)])) < 3
    assert spread(four) < CFG["SPLIT_MAX_SPREAD"]
    assert spread(leaf.astype(bool)) > CFG["SPLIT_MAX_SPREAD"]


def test_splitting_can_be_turned_off():
    veg = _rosettes([(90, 120), (230, 120), (90, 280), (230, 280)])
    seed = _disc(veg.shape, 90, 120, 12)
    assert mx.peak_seeds(veg, [seed], dict(CFG, SPLIT_UNDERSEEDED=False)) == []


def test_the_split_never_creates_a_marker_where_sam_already_has_one():
    """Two markers inside one plant would cut it in half."""
    veg = _rosettes([(90, 150), (230, 150)])
    seed = _disc(veg.shape, 90, 150, 12)
    for extra in mx.peak_seeds(veg, [seed], CFG):
        assert not (extra & seed).any()


def test_an_unseeded_component_is_left_to_the_recall_backstop():
    """Creating instances from the vegetation prior alone is what
    RECOVER_UNSEEDED governs. Doing it here would make that switch a lie."""
    veg = _rosettes([(90, 120), (230, 120), (90, 280), (230, 280)])
    assert mx.peak_seeds(veg, [], CFG) == []


def test_a_recovered_blob_of_several_plants_is_still_split():
    """Same question one step later - and only when the backstop is on."""
    veg = _rosettes([(90, 120), (230, 120), (90, 280), (230, 280)])
    assert len(mx.split_recovered([veg], CFG)) >= 3
    assert mx.split_recovered([veg], dict(CFG, SPLIT_UNDERSEEDED=False)) == [veg]


def test_a_recovered_single_plant_survives_whole():
    one = np.zeros((300, 300), np.uint8)
    cv2.circle(one, (150, 150), 50, 1, -1)
    out = mx.split_recovered([one.astype(bool)], CFG)
    assert len(out) == 1 and int(out[0].sum()) == int(one.sum())


def test_split_instances_are_reported_apart_from_sam_and_recovery():
    """Three recall paths now; conflating them hides which one is working."""
    veg = _rosettes([(90, 120), (230, 120), (90, 280), (230, 280)])
    bgr = np.full(veg.shape + (3,), (70, 45, 60), np.uint8)
    bgr[veg] = (35, 110, 40)
    _, _, qa = mx.analyze_frame(bgr, [_disc(veg.shape, 90, 120, 12)], CFG)
    assert qa["split"] >= 1


# --------------------------------------------------------------------------- #
# Speckle on pale, pebbly ground
# --------------------------------------------------------------------------- #
def test_the_instance_floor_matches_the_weed_prelabeler():
    """On the same imagery the weed prelabeler produces no speckle at 250 and
    this path produced hundreds at 120."""
    wd = load_script("annotation/prelabel_weeds_sam3.py")
    assert mx.CONFIG["MIN_INSTANCE_AREA_PX"] >= wd.CONFIG["MIN_INSTANCE_AREA_PX"]


def test_the_recall_backstop_floor_is_above_gravel():
    """A colour index cannot tell green-tinted mineral from a cotyledon, so
    size is the discriminator that is left."""
    assert mx.CONFIG["RECOVER_MIN_AREA_PX"] >= 400


def test_the_area_histogram_reports_both_populations():
    areas = [180, 200, 220, 240, 5200, 6100, 7000]
    line = mx.area_histogram(areas, mx.CONFIG)
    assert "p50=" in line and "max=7000" in line


def test_the_histogram_calls_out_a_speckled_run():
    speckle = [150] * 90 + [6000] * 10
    assert "suspect speckle" in mx.area_histogram(speckle, mx.CONFIG)


def test_the_histogram_stays_quiet_on_a_clean_run():
    clean = [4000, 5200, 6100, 7000, 9000]
    assert "suspect speckle" not in mx.area_histogram(clean, mx.CONFIG)


# --------------------------------------------------------------------------- #
# The height veto. Depth is a VETO and a scale reference, never a boundary
# source - so the tests are about what it deletes, and about what it refuses to
# delete because it could not measure it.
# --------------------------------------------------------------------------- #
def _depth_scene(h=300, w=400, camera_mm=1200.0):
    """A frame with a raised plant and a flat, green-tinted stone. Colour sees
    one class; only height separates them."""
    bgr = np.full((h, w, 3), SOIL, np.uint8)
    plant = _disc((h, w), 110, 150, 26)
    stone = _disc((h, w), 290, 150, 24)
    bgr[plant] = LEAF
    bgr[stone] = LEAF                       # indistinguishable to the prior
    depth = np.full((h, w), camera_mm, np.float32)
    depth[plant] -= 18.0                    # 18 mm proud of the ground
    return bgr, depth, plant, stone


def test_a_flat_green_stone_is_vetoed_and_the_plant_is_not():
    bgr, depth, plant, stone = _depth_scene()
    inst, _, qa = mx.analyze_frame(bgr, [_disc(bgr.shape[:2], 110, 150, 8),
                                         _disc(bgr.shape[:2], 290, 150, 8)],
                                   dict(CFG, MIN_INSTANCE_AREA_MM2=None),
                                   depth_mm=depth)
    kept = np.zeros(bgr.shape[:2], bool)
    for i in inst:
        kept |= i["mask"]
    assert (kept & plant).sum() > plant.sum() * 0.5
    assert (kept & stone).sum() < stone.sum() * 0.2
    assert qa["height_dropped_flat"] >= 1


def test_without_depth_nothing_changes():
    """Every v1 session depends on this: no depth means the previous behaviour
    exactly, not a degraded version of the new one."""
    bgr, depth, _, _ = _depth_scene()
    sam = [_disc(bgr.shape[:2], 110, 150, 8), _disc(bgr.shape[:2], 290, 150, 8)]
    a, _, qa_a = mx.analyze_frame(bgr, sam, CFG)
    assert "height_dropped_flat" not in qa_a
    assert len(a) == 2                       # the stone survives, as before


def test_an_instance_whose_height_cannot_be_measured_is_kept():
    """THE safety property. Stereo drops out on thin tissue - which is exactly
    the small plants a height gate would otherwise delete - so absence of
    evidence must never act as evidence of flatness."""
    bgr, depth, plant, stone = _depth_scene()
    depth[plant] = np.nan                    # no depth on the plant at all
    inst, _, qa = mx.analyze_frame(bgr, [_disc(bgr.shape[:2], 110, 150, 8)],
                                   dict(CFG, MIN_INSTANCE_AREA_MM2=None),
                                   depth_mm=depth)
    assert qa["height_abstained"] >= 1
    kept = np.zeros(bgr.shape[:2], bool)
    for i in inst:
        kept |= i["mask"]
    assert (kept & plant).any(), "a plant with no depth was deleted"


def test_the_measured_fraction_gates_the_abstention():
    bgr, depth, plant, _ = _depth_scene()
    ys, xs = np.nonzero(plant)
    depth[plant] = np.nan
    depth[ys[:15], xs[:15]] = 1182.0         # a sliver of valid depth
    _, _, qa = mx.analyze_frame(bgr, [_disc(bgr.shape[:2], 110, 150, 8)],
                                dict(CFG, MIN_INSTANCE_AREA_MM2=None),
                                depth_mm=depth)
    assert qa["height_abstained"] >= 1


def test_the_threshold_is_what_decides():
    """Raising it past the plant's real height deletes the plant - which is
    what makes HEIGHT_MIN_MM the setting to be careful with."""
    bgr, depth, _, _ = _depth_scene()
    sam = [_disc(bgr.shape[:2], 110, 150, 8)]
    low = dict(CFG, HEIGHT_MIN_MM=6.0, MIN_INSTANCE_AREA_MM2=None)
    high = dict(CFG, HEIGHT_MIN_MM=40.0, MIN_INSTANCE_AREA_MM2=None)
    assert len(mx.analyze_frame(bgr, sam, low, depth_mm=depth)[0]) == 1
    assert len(mx.analyze_frame(bgr, sam, high, depth_mm=depth)[0]) == 0


def test_the_numbers_behind_a_decision_travel_with_the_instance():
    """height_mm and area_mm2 reach instances.csv, so a veto is auditable
    rather than a count in a console line."""
    bgr, depth, _, _ = _depth_scene()
    inst, _, _ = mx.analyze_frame(bgr, [_disc(bgr.shape[:2], 110, 150, 8)],
                                  dict(CFG, MIN_INSTANCE_AREA_MM2=None),
                                  depth_mm=depth, fx=1000.0, fy=1000.0)
    assert inst[0]["height_mm"] == pytest.approx(18, abs=4)
    assert inst[0]["area_mm2"] > 0
    assert 0 <= inst[0]["height_measured_frac"] <= 1


def test_the_metric_floor_replaces_the_pixel_one_where_depth_allows():
    """A session recorded at a different boom height must not need its own
    thresholds - that is the whole point of measuring in mm^2."""
    bgr, depth, _, _ = _depth_scene()
    sam = [_disc(bgr.shape[:2], 110, 150, 8)]
    generous = dict(CFG, MIN_INSTANCE_AREA_MM2=1.0)
    absurd = dict(CFG, MIN_INSTANCE_AREA_MM2=1e6)
    assert len(mx.analyze_frame(bgr, sam, generous, depth_mm=depth,
                                fx=1000.0, fy=1000.0)[0]) == 1
    out, _, qa = mx.analyze_frame(bgr, sam, absurd, depth_mm=depth,
                                  fx=1000.0, fy=1000.0)
    assert out == [] and qa["height_dropped_small"] >= 1


def test_the_metric_floor_is_inert_without_calibration():
    """No fx/fy means no mm^2, and a floor that cannot be computed must not
    silently delete everything."""
    bgr, depth, _, _ = _depth_scene()
    out, _, _ = mx.analyze_frame(bgr, [_disc(bgr.shape[:2], 110, 150, 8)],
                                 dict(CFG, MIN_INSTANCE_AREA_MM2=1e6),
                                 depth_mm=depth)
    assert len(out) == 1


def test_the_frame_reports_its_ground_relief():
    """So a bed-and-furrow field is visible as one, rather than the surface
    estimate quietly doing something unverifiable."""
    bgr, depth, _, _ = _depth_scene()
    _, _, qa = mx.analyze_frame(bgr, [], dict(CFG), depth_mm=depth)
    assert "ground_relief_mm" in qa and "depth_measured_frac" in qa
