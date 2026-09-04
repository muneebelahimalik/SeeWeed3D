"""Compositing weed cut-outs into real onion frames.

Two things here can make the module actively harmful rather than merely
useless, and most of these tests are about those two:

  * the BACKGROUND SCREEN. A background with unlabelled plants produces a
    composite holding a labelled weed and an unlabelled one side by side -
    teaching that the same plant is both a target and soil.
  * Z-ORDER. Getting it backwards paints onion pixels as weed, which is the
    exact error the whole system exists to prevent, arriving through the
    training data instead of the model.
"""
import json
import random

import numpy as np
import pytest

from conftest import load_script

cm = load_script("annotation/compose_mixed.py")

from common.ontology import CROP_CLASS, LEP_LABEL  # noqa: E402

H = W = 120


def _disc(cx, cy, r, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _box(y, x, hh, ww, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[y:y + hh, x:x + ww] = True
    return m


# --------------------------------------------------------------------------
# the background screen - the gate that decides whether this module helps


def test_a_background_with_unlabelled_vegetation_is_refused():
    """The failure this exists to prevent: a composite where one plant is a
    target and an identical unlabelled one is background."""
    veg = _disc(30, 30, 12) | _disc(80, 80, 12) | _disc(30, 90, 12)
    claimed = _disc(30, 30, 12)                     # only one of three
    ok, n, why = cm.background_ok(veg, claimed)
    assert not ok
    assert n == 2, "two plant-shaped patches nobody claimed"
    assert "nobody labelled" in why


def test_a_fully_annotated_background_is_accepted():
    veg = _disc(30, 30, 12)
    ok, n, why = cm.background_ok(veg, _disc(30, 30, 12))
    assert ok and n == 0 and why == ""


def test_mask_boundary_slop_does_not_reject_a_good_background():
    """A mask a couple of pixels inside the leaf is normal annotation, not an
    unlabelled plant. The first real run rejected two onion backgrounds at 19%
    and 21% unclaimed - which is a rim, not a missing plant, and is exactly why
    the gate counts patches instead of a fraction now."""
    veg = _disc(60, 60, 24)
    claimed = _disc(60, 60, 21)
    ok, n, _ = cm.background_ok(veg, claimed)
    frac = cm.unclaimed_vegetation(veg, claimed)
    assert frac > 0.15, "the rim really is a large fraction"
    assert ok and n == 0, "and still no missing plant"


def test_a_background_with_no_vegetation_is_refused():
    """Bare soil has no onions to be near, so every band collapses to
    'isolated' and the composite teaches nothing about contact."""
    ok, _, why = cm.background_ok(np.zeros((H, W), bool), np.zeros((H, W), bool))
    assert not ok and "no vegetation" in why


def test_a_single_patch_is_tolerated_but_two_are_not():
    """The prior calls moss and green debris vegetation, so one patch is as
    likely a fleck as a plant. Two is a pattern."""
    claimed = _disc(30, 30, 12)
    one = claimed | _disc(80, 80, 12)
    two = one | _disc(30, 90, 12)
    assert cm.background_ok(one, claimed)[0]
    assert not cm.background_ok(two, claimed)[0]


def test_unclaimed_vegetation_is_a_fraction_of_vegetation_not_of_the_frame():
    """Of the frame, two small plants in a big image is a rounding error; of
    the vegetation, it is half of everything growing."""
    veg = _box(0, 0, 4, 4) | _box(50, 50, 4, 4)
    assert cm.unclaimed_vegetation(veg, _box(0, 0, 4, 4)) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# contact, which is the whole point


def test_contact_distance_is_negative_when_they_overlap():
    """One signed number orders every band, so no caller needs a special
    case for 'touching' versus 'overlapping'."""
    onion = _box(40, 40, 30, 30)
    assert cm.contact_distance(_box(45, 45, 10, 10), onion) < 0
    assert cm.contact_distance(_box(0, 0, 5, 5), onion) > 0


def test_contact_distance_measures_the_gap_in_pixels():
    onion = _box(50, 50, 10, 10)
    weed = _box(50, 70, 10, 10)          # 10 px to the right of the onion
    assert cm.contact_distance(weed, onion) == pytest.approx(10, abs=2)


def test_a_deeper_overlap_is_further_into_the_overlap_band():
    """A graze and a burial must not be the same band - one is the decision
    boundary, the other is a weed that is mostly not visible."""
    onion = _box(30, 30, 60, 60)
    graze = cm.contact_distance(_box(25, 25, 12, 12), onion)
    buried = cm.contact_distance(_box(45, 45, 12, 12), onion)
    assert buried < graze < 0


def test_band_of_covers_every_distance_the_generator_can_produce():
    for d in (-40.0, -0.5, 0.0, 0.5, 15.0, 60.0, 500.0):
        assert cm.band_of(d) is not None, f"no band for {d}"


def test_bands_do_not_overlap():
    """Two bands claiming one distance would make the achieved mix
    unreproducible."""
    for d in (-5.0, 0.0, 0.5, 20.0, 50.0, 200.0):
        hits = [n for n, (lo, hi) in cm.CONTACT_BANDS.items() if lo <= d < hi]
        assert len(hits) == 1, f"{d} matched {hits}"


def test_the_plan_delivers_the_requested_mix_not_a_draw_from_it():
    """With a few hundred instances an independently-sampled 10% band comes
    out anywhere between 5% and 15%, and the ablation is then about the
    sampler."""
    plan = cm.band_plan(200, {"touching": 0.5, "isolated": 0.5})
    assert len(plan) == 200
    assert plan.count("touching") == pytest.approx(100, abs=1)


def test_a_band_with_no_definition_is_refused_loudly():
    with pytest.raises(SystemExit, match="no definition"):
        cm.band_plan(10, {"snuggling": 1.0})


def test_an_all_zero_mix_says_which_bands_matter():
    with pytest.raises(SystemExit, match="touching"):
        cm.band_plan(10, {"isolated": 0.0})


# --------------------------------------------------------------------------
# z-order: getting this backwards paints onion pixels as weed


def test_a_weed_in_front_takes_the_overlap_from_the_onion():
    onion, weed = _box(40, 40, 20, 20), _box(45, 45, 20, 20)
    vis_onions, vis_weed = cm.visible_masks([onion], weed, weed_in_front=True)
    assert (vis_weed == weed).all(), "the front instance keeps all its pixels"
    assert not (vis_onions[0] & weed).any(), "the onion loses the overlap"


def test_an_onion_in_front_takes_the_overlap_from_the_weed():
    onion, weed = _box(40, 40, 20, 20), _box(45, 45, 20, 20)
    vis_onions, vis_weed = cm.visible_masks([onion], weed, weed_in_front=False)
    assert (vis_onions[0] == onion).all()
    assert not (vis_weed & onion).any()


def test_no_pixel_is_ever_claimed_by_both():
    """A pixel labelled onion AND weed is the one annotation that could teach
    the model to fire on crop."""
    onion, weed = _box(30, 30, 40, 40), _box(50, 50, 40, 40)
    for front in (True, False):
        vis_onions, vis_weed = cm.visible_masks([onion], weed, front)
        assert not (vis_onions[0] & vis_weed).any()


def test_hidden_fraction_measures_what_the_composite_took_away():
    before = _box(0, 0, 10, 10)
    assert cm.hidden_fraction(before, before) == 0.0
    assert cm.hidden_fraction(before, _box(0, 0, 5, 10)) == pytest.approx(0.5)
    assert cm.hidden_fraction(np.zeros((H, W), bool), np.zeros((H, W), bool)) == 0.0


# --------------------------------------------------------------------------
# the paste itself


def test_paste_puts_the_cutout_where_it_was_asked_to():
    bg = np.full((H, W, 3), 40, np.uint8)
    cut = np.full((10, 10, 3), 200, np.uint8)
    img, placed = cm.paste(bg, cut, np.ones((10, 10), bool), (30, 40),
                           feather=0, illumination=0)
    assert placed[30:40, 40:50].all()
    assert placed.sum() == 100
    assert img[35, 45].mean() > 150


def test_a_cutout_hanging_off_the_edge_is_clipped_not_wrapped():
    bg = np.full((H, W, 3), 40, np.uint8)
    cut = np.full((20, 20, 3), 200, np.uint8)
    img, placed = cm.paste(bg, cut, np.ones((20, 20), bool), (-10, -10),
                           feather=0, illumination=0)
    assert placed[:10, :10].all() and placed.sum() == 100
    assert not placed[-1, -1], "a wrap would light up the opposite corner"


def test_a_cutout_entirely_outside_the_frame_places_nothing():
    bg = np.full((H, W, 3), 40, np.uint8)
    cut = np.full((10, 10, 3), 200, np.uint8)
    _, placed = cm.paste(bg, cut, np.ones((10, 10), bool), (500, 500))
    assert not placed.any()


def test_feathering_leaves_the_plant_interior_untouched():
    """A blend that softened the whole instance would hand the model a cue
    that pasted plants are blurry."""
    a = cm.feathered_alpha(_box(40, 40, 30, 30), px=2)
    assert a[55, 55] == pytest.approx(1.0)
    assert 0.0 < a[40, 55] < 1.0
    assert a[20, 20] == 0.0


def test_no_feathering_is_a_hard_edge():
    a = cm.feathered_alpha(_box(40, 40, 10, 10), px=0)
    assert set(np.unique(a)) <= {0.0, 1.0}


def test_illumination_match_moves_toward_the_target_without_recolouring():
    mask = np.ones((10, 10), bool)
    dark = np.full((10, 10, 3), 60, np.uint8)
    bright = np.full((10, 10, 3), 180, np.uint8)
    out = cm.match_illumination(dark, bright, mask, strength=0.6)
    assert out.mean() > dark.mean()
    assert out.mean() < bright.mean(), "mild, not a full transfer"
    # Hue is preserved: a uniform gain keeps the channels in proportion.
    tinted = np.dstack([np.full((10, 10), 30, np.uint8),
                        np.full((10, 10), 90, np.uint8),
                        np.full((10, 10), 60, np.uint8)])
    got = cm.match_illumination(tinted, bright, mask, strength=0.6)
    r = got[..., 1].mean() / max(1e-6, got[..., 0].mean())
    assert r == pytest.approx(90 / 30, rel=0.05)


def test_illumination_match_is_a_noop_at_zero_strength():
    mask = np.ones((10, 10), bool)
    dark = np.full((10, 10, 3), 60, np.uint8)
    out = cm.match_illumination(dark, np.full((10, 10, 3), 180, np.uint8),
                                mask, strength=0.0)
    assert (out == dark).all()


# --------------------------------------------------------------------------
# the LEP rides along - the annotation compositing gets for free


def test_the_lep_follows_its_mask_through_rotation():
    mask = _box(10, 10, 20, 20, h=40, w=40)
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[mask] = 255
    _, m2, lep2 = cm.transform_cutout(rgb, mask, 1.0, 90.0, lep=(12.0, 12.0))
    assert lep2 is not None
    ys, xs = np.nonzero(m2)
    assert xs.min() - 2 <= lep2[0] <= xs.max() + 2
    assert ys.min() - 2 <= lep2[1] <= ys.max() + 2


def test_a_cutout_without_a_lep_stays_without_one():
    mask = _box(5, 5, 10, 10, h=30, w=30)
    _, _, lep = cm.transform_cutout(np.zeros((30, 30, 3), np.uint8), mask,
                                    1.0, 30.0, lep=None)
    assert lep is None


def test_scaling_changes_the_mask_area_about_as_asked():
    mask = _box(10, 10, 20, 20, h=60, w=60)
    _, big, _ = cm.transform_cutout(np.zeros((60, 60, 3), np.uint8), mask,
                                    2.0, 0.0)
    assert big.sum() == pytest.approx(4 * mask.sum(), rel=0.15)


# --------------------------------------------------------------------------
# the Datumaro the dataset build actually reads


def _instances():
    return [{"class_name": CROP_CLASS, "mask": _box(10, 10, 30, 30),
             "attributes": {}},
            {"class_name": "grass_weed", "mask": _box(60, 60, 20, 20),
             "attributes": {"lep_visibility": "visible", "targetable": "yes"},
             "lep": (70.0, 70.0)}]


def test_the_document_is_datumaro_not_coco():
    """prepare_dataset refuses COCO, and is right to: COCO cannot carry shape
    groups, so every mask-to-LEP link would be silently discarded."""
    doc = cm.datumaro_doc([cm._item("f0", "f0.png", H, W, _instances())])
    assert isinstance(doc.get("items"), list)
    assert "images" not in doc
    assert doc["categories"]["label"]["labels"][-1]["name"] == LEP_LABEL


def test_a_pasted_lep_is_grouped_with_its_own_mask():
    """Ownership is by group id everywhere in this project; an ungrouped LEP
    is rejected by the contract rather than guessed at."""
    item = cm._item("f0", "f0.png", H, W, _instances())
    polys = [a for a in item["annotations"] if a["type"] == "polygon"]
    points = [a for a in item["annotations"] if a["type"] == "points"]
    assert len(points) == 1
    weed_groups = {a["group"] for a in polys
                   if a["label_id"] != cm.CLASSES.index(CROP_CLASS)}
    assert points[0]["group"] in weed_groups


def test_annotator_attributes_travel_with_the_cutout():
    """A composite must satisfy the same contract its source did, so the
    attributes a person set ride along rather than being invented here."""
    item = cm._item("f0", "f0.png", H, W, _instances())
    weed = [a for a in item["annotations"] if a["type"] == "polygon"
            and a["attributes"].get("targetable") == "yes"]
    assert weed and weed[0]["attributes"]["lep_visibility"] == "visible"


def test_the_document_declares_itself_synthetic():
    """Six months on, an annotation file with no provenance is
    indistinguishable from ground truth - and this project has been bitten by
    exactly that once already."""
    doc = cm.datumaro_doc([])
    assert doc["info"]["label_provenance"] == "synthetic"
    assert "never validate" in doc["info"]["description"].lower()


def test_an_empty_mask_produces_no_annotation(tmp_path):
    item = cm._item("f0", "f0.png", H, W, [
        {"class_name": "grass_weed", "mask": np.zeros((H, W), bool),
         "attributes": {}}])
    assert item["annotations"] == []
    json.dumps(item)          # still serialisable


# --------------------------------------------------------------------------
# one whole composite


def _bank():
    rgb = np.full((16, 16, 3), 200, np.uint8)
    return [{"rgb": rgb, "mask": np.ones((16, 16), bool),
             "class_name": "grass_weed", "attributes": {}, "lep": (8.0, 8.0),
             "source": "vid2/f0"}]


def test_compose_one_places_a_weed_and_keeps_the_onion():
    # "near", not "isolated": a 120px frame has no pixel more than ~100px from
    # a central onion, so isolated is unsatisfiable here - and the generator
    # says so rather than placing something and mislabelling its band.
    bgr = np.full((H, W, 3), 50, np.uint8)
    onion = _box(50, 50, 20, 20)
    img, instances, records = cm.compose_one(
        bgr, [onion], _bank(), ["near"], random.Random(0))
    assert img is not None, records
    kinds = [i["class_name"] for i in instances]
    assert CROP_CLASS in kinds and "grass_weed" in kinds


def test_a_band_the_frame_cannot_satisfy_is_reported_not_faked():
    """A 120px frame has no pixel 100px clear of a central onion. Placing
    something anyway and calling it 'isolated' would corrupt the very
    stratification the module exists to provide."""
    img, _, records = cm.compose_one(
        np.full((H, W, 3), 50, np.uint8), [_box(50, 50, 20, 20)], _bank(),
        ["isolated"], random.Random(0))
    assert img is None
    assert any("isolated" in (r.get("reason") or "") for r in records)


def test_compose_one_reports_why_nothing_was_placed():
    """A generator that silently drops most of its attempts produces a dataset
    whose composition nobody can account for."""
    bgr = np.full((H, W, 3), 50, np.uint8)
    onion = np.ones((H, W), bool)          # onion everywhere: no isolated spot
    img, instances, records = cm.compose_one(
        bgr, [onion], _bank(), ["isolated"], random.Random(0))
    assert img is None and instances is None
    assert records and all(r.get("reason") for r in records)


def test_compose_one_records_the_achieved_distance_not_the_requested_band():
    bgr = np.full((H, W, 3), 50, np.uint8)
    img, _, records = cm.compose_one(
        bgr, [_box(50, 50, 20, 20)], _bank(), ["near"], random.Random(1))
    placed = [r for r in records if r.get("band")]
    assert placed and "distance_px" in placed[0]
    assert cm.band_of(placed[0]["distance_px"]) == placed[0]["band"]


def test_composites_are_deterministic_for_a_seed():
    bgr = np.full((H, W, 3), 50, np.uint8)
    onion = [_box(50, 50, 20, 20)]
    a, _, _ = cm.compose_one(bgr, onion, _bank(), ["near"], random.Random(7))
    b, _, _ = cm.compose_one(bgr, onion, _bank(), ["near"], random.Random(7))
    assert (a is None) == (b is None)
    if a is not None:
        assert (a == b).all()


# --------------------------------------------------------------------------
# the report


def test_the_report_states_that_composites_are_not_for_validation():
    txt = cm.format_report(10, 25, {"touching": 20, "near": 5}, {}, 40, 10)
    assert "never validate" in txt
    assert "SYNTHETIC" in txt
    assert "prelabels" in txt, "the onion masks are machine labels; say so"


def test_the_report_shows_the_achieved_bands():
    txt = cm.format_report(10, 25, {"touching": 20, "near": 5}, {}, 40, 10)
    assert "touching" in txt and "measured, not requested" in txt


def test_the_report_names_rejections():
    txt = cm.format_report(1, 1, {"near": 1}, {"nobody labelled it": 7}, 9, 1)
    assert "nobody labelled it" in txt and "7" in txt


def test_touching_is_wide_enough_to_actually_hit():
    """The first real run asked for 30% touching and achieved 4%, rejecting
    145 of 159 attempts, because the band was ONE PIXEL wide. Its neighbours
    came out over target by the amount that went missing - that is where the
    failures landed."""
    lo, hi = cm.CONTACT_BANDS["touching"]
    assert hi - lo >= 3, "a 1px window is not a physical distinction at 2208px"


def test_the_bands_still_tile_the_line_after_widening():
    """Widening one band must not leave a distance no band claims, or an
    achieved placement would be silently unclassifiable."""
    edges = sorted(cm.CONTACT_BANDS.values())
    for (lo1, hi1), (lo2, _) in zip(edges, edges[1:]):
        assert hi1 == lo2, f"gap or overlap between {hi1} and {lo2}"


def test_a_placement_a_few_pixels_from_an_onion_is_touching():
    onion = _box(40, 40, 30, 30)
    weed = _box(40, 72, 10, 10)          # 2 px clear of it
    d = cm.contact_distance(weed, onion)
    assert 0 <= d < 4
    assert cm.band_of(d) == "touching"


# --------------------------------------------------------------------------
# The hit rate is the point: a crown anchored at a band pixel does not put the
# PLANT in the band, and rejecting on that alone threw away most attempts.
# --------------------------------------------------------------------------
def _dist_and_grad(union):
    import cv2
    d = cv2.distanceTransform((~np.asarray(union, bool)).astype(np.uint8),
                              cv2.DIST_L2, 3)
    return d, (cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3),
               cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))


def test_refine_pulls_a_far_placement_into_touching():
    union = _box(40, 40, 40, 40)
    d, g = _dist_and_grad(union)
    weed = np.ones((10, 10), bool)
    far = (45, 95)                              # ~15 px clear of the onion
    assert cm.band_of(cm.contact_distance(
        cm.placed_mask(weed, far, union.shape), union)) != "touching"
    spot = cm.refine_offset(union, d, g, weed, far, "touching")
    assert spot is not None, "the nudge should reach a 4px-wide band"
    got = cm.contact_distance(cm.placed_mask(weed, spot, union.shape), union)
    assert cm.band_of(got) == "touching"


def test_refine_pushes_an_overlapping_placement_out_to_touching():
    union = _box(40, 40, 40, 40)
    d, g = _dist_and_grad(union)
    weed = np.ones((10, 10), bool)
    inside = (50, 50)
    assert cm.contact_distance(
        cm.placed_mask(weed, inside, union.shape), union) < 0
    spot = cm.refine_offset(union, d, g, weed, inside, "touching")
    if spot is not None:
        got = cm.contact_distance(
            cm.placed_mask(weed, spot, union.shape), union)
        assert cm.band_of(got) == "touching"


def test_refine_returns_none_rather_than_a_wrong_band():
    """A placement that cannot reach its band must be rejected, not silently
    recorded under a band it does not occupy - that would corrupt the strata
    the module exists to control."""
    union = _box(40, 40, 40, 40)
    d, g = _dist_and_grad(union)
    weed = np.ones((10, 10), bool)
    spot = cm.refine_offset(union, d, g, weed, (45, 95), "isolated", steps=1)
    if spot is not None:
        got = cm.contact_distance(
            cm.placed_mask(weed, spot, union.shape), union)
        assert cm.band_of(got) == "isolated"


def test_refine_keeps_a_placement_that_is_already_right():
    union = _box(40, 40, 40, 40)
    d, g = _dist_and_grad(union)
    weed = np.ones((8, 8), bool)
    good = (45, 82)                             # 2 px clear
    assert cm.band_of(cm.contact_distance(
        cm.placed_mask(weed, good, union.shape), union)) == "touching"
    assert cm.refine_offset(union, d, g, weed, good, "touching") == good


def test_band_target_aims_inside_a_band_not_at_its_edge():
    """Aiming at an edge and landing a pixel the wrong side is a rejection."""
    for name, (lo, hi) in cm.CONTACT_BANDS.items():
        t = cm.band_target(name)
        assert lo <= t < hi, f"{name}: {t} outside [{lo}, {hi})"


def test_placed_mask_matches_what_paste_places():
    """The two must agree, or a placement is judged on one geometry and
    rendered with another."""
    bg = np.zeros((H, W, 3), np.uint8)
    cut = np.full((12, 12, 3), 200, np.uint8)
    m = np.ones((12, 12), bool)
    for spot in ((30, 40), (-5, -5), (H - 4, W - 4)):
        _, pasted = cm.paste(bg, cut, m, spot, feather=0, illumination=0)
        assert (cm.placed_mask(m, spot, (H, W)) == pasted).all(), spot


# --------------------------------------------------------------------------
# Cut-outs are drawn WITH REPLACEMENT, so a paste count is not a plant count.
# A reused cut-out is the same pixels in several frames, and the frame-block
# split cannot see it: it separates by frame index, and a composite set has no
# video order for an index to mean anything.
# --------------------------------------------------------------------------
def _recs(sources):
    return [{"band": "touching", "source": s} for s in sources]


def test_the_report_says_how_many_distinct_plants_the_pastes_are():
    txt = "\n".join(cm.reuse_note(_recs(["a", "b", "a", "c"]), bank_size=9))
    assert "4 paste(s) from 3 distinct cut-out(s)" in txt
    assert "bank held 9" in txt


def test_a_reused_cutout_is_called_out_as_measuring_memorisation():
    txt = "\n".join(cm.reuse_note(_recs(["a", "a", "b"])))
    assert "1 cut-out(s) pasted more than once" in txt
    assert "appears 2 time(s)" in txt
    assert "memorisation" in txt and "source_instance" in txt


def test_no_reuse_says_nothing_alarming():
    """The warning has to be absent when it does not apply, or it becomes the
    line everybody scrolls past."""
    txt = "\n".join(cm.reuse_note(_recs(["a", "b", "c"])))
    assert "3 paste(s) from 3 distinct" in txt
    assert "memorisation" not in txt


def test_reuse_is_silent_when_nothing_records_a_source():
    assert cm.reuse_note([]) == []
    assert cm.reuse_note([{"band": "near"}]) == []


def test_a_pasted_instance_carries_the_plant_it_came_from():
    """Without this the dataset holds no way to tell two copies of one plant
    from two different plants, and no split can honour the difference."""
    bgr = np.full((H, W, 3), 50, np.uint8)
    bank = _bank()
    bank[0]["attributes"] = {"targetable": "yes"}
    _, instances, _ = cm.compose_one(bgr, [_box(50, 50, 20, 20)], bank,
                                     ["near"], random.Random(0))
    weeds = [i for i in instances if i["class_name"] == "grass_weed"]
    assert weeds, "nothing was pasted, so nothing is being checked"
    for w in weeds:
        assert w["attributes"]["source_instance"] == "vid2/f0"
        assert w["attributes"]["targetable"] == "yes", (
            "the annotator's own attributes have to survive alongside it")


def test_the_provenance_survives_into_the_written_annotations():
    """It is only useful if a split can read it back out of default.json."""
    bgr = np.full((H, W, 3), 50, np.uint8)
    _, instances, _ = cm.compose_one(bgr, [_box(50, 50, 20, 20)], _bank(),
                                     ["near"], random.Random(0))
    item = cm._item("f1", "f1.png", H, W, instances)
    names = list(cm.CLASSES) + [cm.LEP_LABEL]
    weed_anns = [a for a in item["annotations"]
                 if names[a["label_id"]] == "grass_weed"]
    assert weed_anns
    assert all(a["attributes"].get("source_instance") == "vid2/f0"
               for a in weed_anns)
    json.dumps(item)


# --------------------------------------------------------------------------
# Drawing without replacement. A bigger bank makes reuse less likely; drawing
# each plant once before any of them twice removes it while the bank lasts.
# --------------------------------------------------------------------------
def test_every_plant_is_drawn_before_any_is_drawn_twice():
    bank = [{"id": i} for i in range(10)]
    d = cm.CutoutDrawer(bank, random.Random(0))
    got = [d.draw()["id"] for _ in range(10)]
    assert sorted(got) == list(range(10)), f"a plant repeated early: {got}"
    assert d.cycles == 1


def test_reuse_begins_only_once_the_pastes_outnumber_the_bank():
    d = cm.CutoutDrawer([{"id": i} for i in range(4)], random.Random(0))
    for _ in range(4):
        d.draw()
    assert d.cycles == 1, "the bank was not exhausted yet"
    d.draw()
    assert d.cycles == 2, "a second pass should be counted, not hidden"


def test_the_draw_order_follows_the_seed():
    """Reproducible, and not simply bank order - otherwise every run pastes the
    same plants into the same early frames."""
    bank = [{"id": i} for i in range(8)]

    def order(seed):
        d = cm.CutoutDrawer(bank, random.Random(seed))
        return [d.draw()["id"] for _ in range(8)]

    assert order(1) == order(1), "same seed, different order"
    assert order(1) != order(2), "the order does not depend on the seed"
    assert order(1) != list(range(8)), "the bank is being read in order"


def test_an_empty_bank_says_so_rather_than_looping():
    with pytest.raises(IndexError):
        cm.CutoutDrawer([], random.Random(0)).draw()


def test_one_run_shares_a_drawer_so_frames_cannot_repeat_a_plant():
    """The point of the shared drawer: per-frame drawers each start a fresh
    pool, which promises nothing ACROSS composites - and across composites is
    where a split gets crossed."""
    bank = [dict(_bank()[0], source=f"vid2/f{i}") for i in range(6)]
    drawer = cm.CutoutDrawer(bank, random.Random(0))
    bgr = np.full((H, W, 3), 50, np.uint8)
    seen = []
    for _ in range(3):
        _, _, recs = cm.compose_one(bgr, [_box(50, 50, 20, 20)], drawer,
                                    ["near", "near"], random.Random(0))
        seen += [r["source"] for r in recs if r.get("band")]
    assert len(seen) == len(set(seen)), f"a plant crossed frames: {seen}"


def test_a_plain_list_still_works_for_callers_that_pass_one():
    bgr = np.full((H, W, 3), 50, np.uint8)
    img, instances, _ = cm.compose_one(bgr, [_box(50, 50, 20, 20)], _bank(),
                                       ["near"], random.Random(0))
    assert img is not None and instances


def test_no_reuse_is_stated_positively():
    txt = "\n".join(cm.reuse_note(_recs(["a", "b", "c"])))
    assert "every paste is a different cut-out" in txt
    assert "Plants recurring across source frames still can" in txt, (
        "the positive line must not overstate: distinct cut-outs is not "
        "distinct plants when the source is video")


def test_reuse_advice_points_at_the_two_knobs_that_cause_it():
    txt = "\n".join(cm.reuse_note(_recs(["a", "a"])))
    assert "BANK_MAX" in txt and "WEEDS_PER_IMAGE" in txt


def test_the_bank_cap_of_zero_keeps_every_cutout():
    """It was a function default of 600 against ~3,300 hand-drawn instances,
    discarding five sixths of the project's weeds where nobody could see it."""
    assert cm.BANK_MAX == 0 or cm.BANK_MAX >= 3000


def test_a_cutout_id_names_the_instance_not_just_its_frame():
    """Every weed in one source frame shared an id, so the reuse report counted
    source FRAMES: it read '794 pastes from 131 plants' on a run where the
    drawer had in fact handed out 794 different cut-outs."""
    ids = {f"vid2/f{f}#{i}" for f in range(3) for i in range(5)}
    assert len(ids) == 15


def test_the_report_bounds_distinct_plants_by_the_source_frames():
    """Cut-out count is an upper bound on distinct plants and a loose one: the
    weed drives are video, so one weed recurs in every frame it was driven
    past. The honest smaller number belongs next to it."""
    recs = [{"band": "near", "source": f"vid2/f{i // 6}#{i}",
             "source_frame": f"vid2/f{i // 6}"} for i in range(30)]
    txt = "\n".join(cm.reuse_note(recs, 3842))
    assert "30 paste(s) from 30 distinct cut-out(s)" in txt
    assert "5 source frame(s)" in txt
    assert "DISTINCT" in txt and "below that" in txt


def test_the_frame_bound_is_omitted_when_nothing_records_one():
    txt = "\n".join(cm.reuse_note([{"band": "near", "source": "a"}]))
    assert "source frame(s)" not in txt


# --------------------------------------------------------------------------
# Looking at what was generated. compose_mixed writes RGB and Datumaro and no
# pictures, so until this there was no way to check a composite by eye except
# the audit's worst-15 in audit colours.
# --------------------------------------------------------------------------
def _run(tmp_path, bands=("touching",), with_item=True):
    import cv2
    run = tmp_path / "synth_mixed_20260101_0000"
    (run / "rgb").mkdir(parents=True)
    (run / "annotations").mkdir()
    cv2.imwrite(str(run / "rgb" / "synth_1.png"),
                np.full((H, W, 3), 60, np.uint8))

    def box(y, x, h, w):
        m = np.zeros((H, W), bool)
        m[y:y + h, x:x + w] = True
        return m

    item = cm._item("synth_1", "synth_1.png", H, W, [
        {"class_name": CROP_CLASS, "mask": box(20, 20, 30, 30),
         "attributes": {}},
        {"class_name": "grass_weed", "mask": box(60, 60, 20, 20),
         "attributes": {}}])
    (run / "annotations" / "default.json").write_text(
        json.dumps(cm.datumaro_doc([item])), encoding="utf-8")
    recs = [{"band": b, "source": "vid2/f0#1",
             **({"item": "synth_1"} if with_item else {})} for b in bands]
    (run / "compose_report.json").write_text(json.dumps({"records": recs}),
                                             encoding="utf-8")
    return run


def test_a_finished_run_can_be_drawn_without_regenerating_it(tmp_path):
    """Looking must never be a reason to change what was generated."""
    run = _run(tmp_path)
    cm.render_overlays(run, scale=1.0)
    assert (run / "overlays" / "synth_1.png").is_file()


def test_a_split_mask_draws_as_one_instance(tmp_path):
    """An onion in front can cut a pasted weed into two polygons. Drawing them
    as separate instances would show four weeds where the annotation says
    two."""
    groups = cm.instance_groups({"annotations": [
        {"type": "polygon", "label_id": 0, "group": 1, "points": [0, 0, 1, 1]},
        {"type": "polygon", "label_id": 0, "group": 1, "points": [5, 5, 6, 6]},
        {"type": "polygon", "label_id": 1, "group": 2, "points": [2, 2, 3, 3]},
    ]})
    assert len(groups) == 2
    assert len(groups[0][1]) == 2, "one instance, two polygons"


def test_lep_points_are_not_drawn_as_instances(tmp_path):
    groups = cm.instance_groups({"annotations": [
        {"type": "points", "label_id": 4, "group": 1, "points": [3, 3]},
        {"type": "polygon", "label_id": 0, "group": 1, "points": [0, 0, 1, 1]},
    ]})
    assert len(groups) == 1 and groups[0][1] == [[0, 0, 1, 1]]


def test_weeds_are_coloured_by_the_band_they_achieved(tmp_path):
    """The thing to check in a composite is not what the plant is - the cut-out
    carries a hand-drawn class - but whether the placement the report claims is
    the one you can see."""
    bgr = np.full((H, W, 3), 60, np.uint8)
    groups = [("grass_weed", [[10, 10, 40, 10, 40, 40, 10, 40]])]
    touching = cm.draw_composite(bgr, groups, ["touching"], scale=1.0)
    overlap = cm.draw_composite(bgr, groups, ["overlap"], scale=1.0)
    assert not np.array_equal(touching, overlap)


def test_a_run_with_no_band_records_still_draws(tmp_path):
    """An older run predates per-frame records. It is still worth looking at."""
    run = _run(tmp_path, with_item=False)
    cm.render_overlays(run, scale=1.0)
    assert (run / "overlays" / "synth_1.png").is_file()


def test_pointing_it_at_the_wrong_folder_says_what_it_wanted(tmp_path):
    with pytest.raises(SystemExit) as e:
        cm.render_overlays(tmp_path)
    assert "synth_mixed" in str(e.value)


def test_overlays_go_somewhere_else_if_asked(tmp_path):
    run = _run(tmp_path)
    cm.render_overlays(run, out_dir=tmp_path / "look", scale=1.0)
    assert (tmp_path / "look" / "synth_1.png").is_file()


def test_no_band_colour_looks_like_the_crop():
    """The first palette drew `touching` in (0,140,255) against a crop of
    (0,165,255) - the same orange to within a shade, and `touching` is 30% of
    the weeds. Half the pasted weeds were invisible and the composites read as
    one-weed scenes when they hold four."""
    for band, colour in cm.OVERLAY_BAND.items():
        d = sum(abs(a - b) for a, b in zip(colour, cm.OVERLAY_CROP))
        assert d > 120, f"{band} {colour} is the crop's colour {cm.OVERLAY_CROP}"


def test_no_band_colour_is_grey():
    """Grey outlines on grey-green soil are the other way to hide a weed."""
    for band, (b, g, r) in cm.OVERLAY_BAND.items():
        assert max(b, g, r) - min(b, g, r) > 60, f"{band} is grey"


def test_every_band_has_its_own_colour():
    assert len(set(cm.OVERLAY_BAND.values())) == len(cm.OVERLAY_BAND)
    assert set(cm.OVERLAY_BAND) == set(cm.CONTACT_BANDS)


def test_the_run_says_how_many_weeds_it_drew():
    """'It looks like one or two weeds a frame' has two very different
    answers - the generator pasted two, or it pasted four and you cannot see
    them. Only counting separates those."""
    txt = "\n".join(cm.count_note([4, 3, 5, 2], [200.0, 5000.0, 40000.0]))
    assert "14 across 4 frame(s)" in txt
    assert "mean 3.50" in txt and "min 2" in txt and "max 5" in txt


def test_tiny_weeds_are_called_out_before_a_frame_is_called_empty():
    """A pasted cut-out is not scaled: a real cotyledon is a few hundred px in
    a 2.7 megapixel frame, and at overlay scale 0.5 that is a mark you miss."""
    txt = "\n".join(cm.count_note([2], [200.0, 300.0, 90000.0]))
    assert "2 of 3 are under 1500 px" in txt
    assert "--scale 1.0" in txt


def test_no_tiny_weeds_means_no_warning_about_them():
    txt = "\n".join(cm.count_note([2], [40000.0, 90000.0]))
    assert "under 1500 px" not in txt


def test_polygon_area_is_the_enclosed_pixels():
    square = [0, 0, 10, 0, 10, 10, 0, 10]
    assert cm.polygon_area(square) == pytest.approx(100.0)
    assert cm.polygon_area([0, 0, 1, 1]) == 0.0, "a line encloses nothing"


def test_counting_survives_a_run_with_nothing_in_it():
    assert cm.count_note([], []) == []
