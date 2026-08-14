"""Complement prelabeling: onions from a model, weeds from the remainder.

The arithmetic - "not onion, therefore weed" - is right. The ERROR DIRECTION is
what makes it dangerous: a missed onion becomes a weed, and a weed is something
the machine fires at. So every test here is about the third outcome. What the
module must do is refuse to call something a weed whenever the reason it is not
an onion might be that the model failed.
"""
import json

import cv2
import numpy as np
import pytest

from conftest import load_script

cp = load_script("annotation/prelabel_complement_sam3.py")
ont = load_script("common/ontology.py")

CFG = cp.CONFIG
SOIL = (60, 85, 120)
LEAF = (60, 190, 70)


def _disc(shape, cx, cy, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def _scene(h=300, w=400):
    """One onion and one weed, well apart, both plainly vegetation."""
    img = np.full((h, w, 3), SOIL, np.uint8)
    onion = _disc((h, w), 100, 150, 32)
    weed = _disc((h, w), 300, 150, 30)
    img[onion] = LEAF
    img[weed] = LEAF
    return img, onion, weed


# --------------------------------------------------------------------------- #
# The three outcomes
# --------------------------------------------------------------------------- #
def test_a_detected_onion_and_a_distant_weed_are_separated():
    img, onion, weed = _scene()
    veg = cp.vegetation(img, CFG)
    on, wd, ig = cp.partition(veg, onion, CFG, halo=10)
    assert (on & onion).sum() > onion.sum() * 0.8
    assert (wd & weed).sum() > weed.sum() * 0.8
    assert not (wd & onion).any()


def test_vegetation_beside_a_detected_onion_is_never_called_weed():
    """The most common failure the complement faces: a leaf of a DETECTED onion
    that the model's mask did not cover. It is adjacent to what WAS covered, so
    widening the onion region before taking the complement catches it directly
    rather than hoping a confidence threshold does."""
    h, w = 300, 400
    img = np.full((h, w, 3), SOIL, np.uint8)
    whole = _disc((h, w), 150, 150, 40)
    img[whole] = LEAF
    detected = _disc((h, w), 150, 150, 22)      # the model saw only the middle

    veg = cp.vegetation(img, CFG)
    on, wd, ig = cp.partition(veg, detected, CFG, halo=30)
    missed = whole & ~detected
    assert not (wd & missed).any(), "an onion's own missed leaf became a weed"
    assert (ig & missed).any()


def test_the_uncertain_band_is_ignore_not_weed_and_not_onion():
    img, onion, weed = _scene()
    veg = cp.vegetation(img, CFG)
    on, wd, ig = cp.partition(veg, onion, CFG, halo=25)
    assert not (ig & on).any()
    assert not (ig & wd).any()


def test_a_small_uncertain_speck_is_ignore_rather_than_weed():
    """Below the weed floor but above the ignore floor: not confident enough to
    call a weed, too big to drop silently."""
    h, w = 300, 400
    img = np.full((h, w, 3), SOIL, np.uint8)
    speck = _disc((h, w), 300, 150, 8)          # ~200 px, under WEED_MIN_AREA_PX
    img[speck] = LEAF
    veg = cp.vegetation(img, CFG)
    cfg = dict(CFG, VEG_MIN_COMPONENT_PX=50, WEED_MIN_AREA_PX=400,
               IGNORE_MIN_AREA_PX=100)
    on, wd, ig = cp.partition(veg, None, cfg, halo=0)
    assert not wd.any() and ig.any()


def test_soil_speckle_below_both_floors_is_dropped_entirely():
    h, w = 300, 400
    img = np.full((h, w, 3), SOIL, np.uint8)
    img[_disc((h, w), 300, 150, 3)] = LEAF
    veg = cp.vegetation(img, dict(CFG, VEG_MIN_COMPONENT_PX=1))
    on, wd, ig = cp.partition(veg, None, CFG, halo=0)
    assert not wd.any() and not ig.any()


# --------------------------------------------------------------------------- #
# When the model fails outright
# --------------------------------------------------------------------------- #
def test_no_onion_detected_does_not_turn_a_frame_into_all_weed():
    """In a MIXED scene, zero onions detected is far more likely a model
    failure than a frame without onions. Calling every plant a weed there is
    the complement's worst case, and it is handled in the session loop - so
    this pins that partition() alone would have done it, and the caller must
    not just take its answer."""
    img, onion, weed = _scene()
    veg = cp.vegetation(img, CFG)
    on, wd, ig = cp.partition(veg, None, CFG, halo=CFG["ONION_HALO_PX"])
    assert not on.any()
    assert (wd & onion).any(), (
        "partition() by itself calls an undetected onion a weed - which is "
        "why prelabel_session escalates a no-onion frame to ignore_region")


def test_the_crop_class_is_matched_by_name_not_by_index():
    """A model trained on a reduced class set has its own ordering. An index
    assumed here would point crop safety at whatever class sat in that slot."""
    src = cp.__file__ if hasattr(cp, "__file__") else None
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "seeweed3d" / "annotation" /
            "prelabel_complement_sam3.py").read_text(encoding="utf-8")
    assert "det.class_name(i) != CROP_CLASS" in text


# --------------------------------------------------------------------------- #
# The halo, in millimetres where depth allows
# --------------------------------------------------------------------------- #
def test_the_halo_falls_back_to_pixels_without_depth():
    assert cp.halo_px(CFG) == CFG["ONION_HALO_PX"]
    assert cp.halo_px(CFG, None, 1000.0, None) == CFG["ONION_HALO_PX"]


def test_the_halo_is_a_fixed_distance_on_the_ground_when_depth_exists():
    """A band fixed in pixels means different distances at different boom
    heights, which is the same defect the metric area floor exists to remove."""
    m = _disc((200, 200), 100, 100, 30)
    near = np.full((200, 200), 1000.0, np.float32)
    far = np.full((200, 200), 2000.0, np.float32)
    cfg = dict(CFG, ONION_HALO_MM=30.0)
    # twice as far away, so a millimetre is half as many pixels
    assert cp.halo_px(cfg, near, 1000.0, m) == pytest.approx(30, abs=1)
    assert cp.halo_px(cfg, far, 1000.0, m) == pytest.approx(15, abs=1)


def test_too_little_depth_falls_back_rather_than_inventing_a_band():
    m = _disc((200, 200), 100, 100, 30)
    d = np.full((200, 200), np.nan, np.float32)
    assert cp.halo_px(CFG, d, 1000.0, m) == CFG["ONION_HALO_PX"]


# --------------------------------------------------------------------------- #
# What reaches CVAT
# --------------------------------------------------------------------------- #
def test_ignore_region_cannot_collide_with_an_ontology_class():
    """It is an annotation-only label, not a CLASS - the dataset builder
    excludes it from training, and a colliding id would train on it."""
    coco = cp.ComplementCoco("other_weed")
    ids = [c["id"] for c in coco.categories]
    assert len(ids) == len(set(ids))
    ignore = [c for c in coco.categories if c["name"] == ont.IGNORE_LABEL][0]
    assert ignore["id"] not in [ont.CATEGORY_ID[n] for n in ont.CLASSES]


def test_the_export_uses_real_ontology_ids_for_the_real_classes():
    """So a complement export merges with a hand-annotated one without
    remapping a single annotation."""
    coco = cp.ComplementCoco("other_weed")
    by_name = {c["name"]: c["id"] for c in coco.categories}
    assert by_name[ont.CROP_CLASS] == ont.CATEGORY_ID[ont.CROP_CLASS]
    assert by_name["other_weed"] == ont.CATEGORY_ID["other_weed"]


def test_all_three_buckets_reach_the_coco(tmp_path):
    coco = cp.ComplementCoco("other_weed")
    img = coco.add_image("f.png", 300, 400)
    for cls in (ont.CROP_CLASS, "other_weed", ont.IGNORE_LABEL):
        coco.add(img, cls, [[0, 0, 10, 0, 10, 10]], [0, 0, 10, 10], 50)
    coco.dump(tmp_path / "i.json")
    doc = json.loads((tmp_path / "i.json").read_text())
    names = {c["id"]: c["name"] for c in doc["categories"]}
    assert {names[a["category_id"]] for a in doc["annotations"]} == \
        {ont.CROP_CLASS, "other_weed", ont.IGNORE_LABEL}


def test_the_preview_colours_by_bucket_not_by_class():
    """The question a complement preview answers is which of the three a region
    landed in; a class palette would answer a different one."""
    img, onion, weed = _scene()
    vis = cp.overlay(img, {"onion": [{"mask": onion}], "weed": [{"mask": weed}],
                           "ignore": []}, 1.0, CFG)
    cols = {tuple(int(v) for v in c) for c in vis.reshape(-1, 3)}
    assert cp.BUCKET_BGR["onion"] in cols and cp.BUCKET_BGR["weed"] in cols
