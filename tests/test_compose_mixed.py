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
    veg = _disc(30, 30, 12) | _disc(90, 90, 12)     # two plants
    claimed = _disc(30, 30, 12)                     # only one annotated
    ok, frac, why = cm.background_ok(veg, claimed)
    assert not ok
    assert frac > 0.4
    assert "nobody labelled" in why


def test_a_fully_annotated_background_is_accepted():
    veg = _disc(30, 30, 12)
    ok, frac, why = cm.background_ok(veg, _disc(30, 30, 12))
    assert ok and frac == pytest.approx(0.0) and why == ""


def test_mask_boundary_slop_does_not_reject_a_good_background():
    """A mask a couple of pixels inside the leaf is normal annotation, not an
    unlabelled plant. A screen that fired on it would reject everything."""
    veg = _disc(60, 60, 20)
    claimed = _disc(60, 60, 19)
    ok, frac, _ = cm.background_ok(veg, claimed)
    assert ok, f"rejected a 1px-tight mask at {frac:.0%} unclaimed"


def test_a_background_with_no_vegetation_is_refused():
    """Bare soil has no onions to be near, so every band collapses to
    'isolated' and the composite teaches nothing about contact."""
    ok, _, why = cm.background_ok(np.zeros((H, W), bool), np.zeros((H, W), bool))
    assert not ok and "no vegetation" in why


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
