"""The visual results report.

Its job is to answer "WHICH plants did the model get wrong", so the tests are
about outcome bookkeeping being correct - a report that quietly mislabels a
miss as a match is worse than no report.
"""
import json

import numpy as np
import pytest

from conftest import load_script

rp = load_script("evaluation/report.py")


def _blob(h=200, w=200, y0=20, y1=60, x0=20, x1=60):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


# --------------------------------------------------------------------------- #
# per-frame outcome
# --------------------------------------------------------------------------- #
def test_a_perfect_prediction_has_no_misses_and_no_false_positives():
    gt = [_blob()]
    matches, missed, fp = rp.analyse_frame(gt, ["grass_weed"], [0.9],
                                           gt, ["grass_weed"])
    assert len(matches) == 1 and missed == [] and fp == []


def test_an_undetected_plant_is_a_miss():
    gt = [_blob()]
    matches, missed, fp = rp.analyse_frame([], [], [], gt, ["grass_weed"])
    assert matches == [] and missed == [0] and fp == []


def test_a_prediction_with_no_ground_truth_is_a_false_positive():
    pred = [_blob()]
    matches, missed, fp = rp.analyse_frame(pred, ["grass_weed"], [0.9], [], [])
    assert matches == [] and missed == [] and fp == [0]


def test_the_right_shape_with_the_wrong_class_is_both_a_miss_and_an_fp():
    """Class-aware matching. Calling an onion a weed must not read as a hit."""
    m = _blob()
    matches, missed, fp = rp.analyse_frame([m], ["grass_weed"], [0.9],
                                           [m], ["onion_plant"])
    assert matches == []
    assert missed == [0] and fp == [0]


def test_a_barely_overlapping_prediction_does_not_count_as_found():
    gt = [_blob(y0=0, y1=40, x0=0, x1=40)]
    pred = [_blob(y0=100, y1=140, x0=100, x1=140)]
    matches, missed, fp = rp.analyse_frame(pred, ["grass_weed"], [0.9],
                                           gt, ["grass_weed"])
    assert missed == [0] and fp == [0]


# --------------------------------------------------------------------------- #
# recall by size - the headline table
# --------------------------------------------------------------------------- #
def _rec(gt):
    return {"gt": gt}


def test_recall_by_size_buckets_by_ground_truth_area():
    recs = [_rec([
        {"area_px": 100, "is_crop": False, "matched": False},
        {"area_px": 120, "is_crop": False, "matched": True},
        {"area_px": 5000, "is_crop": False, "matched": True},
    ])]
    rows = {r["range_px"]: r for r in rp.recall_by_size(recs)}
    assert rows["0-250"]["n_gt"] == 2
    assert rows["0-250"]["recall"] == pytest.approx(0.5)
    assert rows["4000-8000"]["recall"] == pytest.approx(1.0)


def test_the_crop_is_excluded_from_recall_by_size():
    """Crop recall is a safety question with its own section; averaging it in
    would let good weed recall mask a missed onion."""
    recs = [_rec([
        {"area_px": 100, "is_crop": True, "matched": False},
        {"area_px": 110, "is_crop": False, "matched": True},
    ])]
    rows = {r["range_px"]: r for r in rp.recall_by_size(recs)}
    assert rows["0-250"]["n_gt"] == 1
    assert rows["0-250"]["recall"] == pytest.approx(1.0)


def test_an_empty_bucket_reports_none_not_zero():
    """Zero recall and 'nothing of this size existed' are different facts."""
    rows = {r["range_px"]: r for r in rp.recall_by_size([_rec([])])}
    assert rows["0-250"]["recall"] is None


def test_every_bucket_boundary_is_half_open():
    recs = [_rec([{"area_px": 250, "is_crop": False, "matched": True}])]
    rows = {r["range_px"]: r for r in rp.recall_by_size(recs)}
    assert rows["0-250"]["n_gt"] == 0
    assert rows["250-500"]["n_gt"] == 1


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def test_a_miss_is_drawn_differently_from_a_match():
    bgr = np.zeros((80, 80, 3), np.uint8)
    m = _blob(80, 80)
    hit = rp.draw_outcomes(bgr, [m], ["grass_weed"], ["tp"])
    miss = rp.draw_outcomes(bgr, [m], ["grass_weed"], ["fn"])
    assert not np.array_equal(hit, miss)


def test_the_crop_is_coloured_as_the_crop_whatever_its_outcome():
    bgr = np.zeros((80, 80, 3), np.uint8)
    m = _blob(80, 80)
    crop = rp.draw_outcomes(bgr, [m], ["onion_plant"], ["tp"])
    weed = rp.draw_outcomes(bgr, [m], ["grass_weed"], ["tp"])
    assert not np.array_equal(crop, weed)


def test_pixels_outside_every_mask_are_untouched():
    bgr = np.full((80, 80, 3), 90, np.uint8)
    m = _blob(80, 80, 5, 15, 5, 15)
    out = rp.draw_outcomes(bgr, [m], ["grass_weed"], ["fn"])
    assert (out[60:, 60:] == 90).all()


def test_an_empty_mask_does_not_crash_the_overlay():
    bgr = np.zeros((40, 40, 3), np.uint8)
    out = rp.draw_outcomes(bgr, [np.zeros((40, 40), bool)], ["grass_weed"],
                           ["fn"])
    assert out.shape == bgr.shape


def test_the_panel_places_ground_truth_beside_prediction():
    bgr = np.zeros((60, 80, 3), np.uint8)
    m = _blob(60, 80)
    panel = rp.frame_panel(bgr, [m], ["grass_weed"], [False],
                           [], [], [])
    assert panel.shape[1] > 80 * 2, "two panels, not one blended image"


def test_a_wide_panel_is_downscaled():
    bgr = np.zeros((100, 2000, 3), np.uint8)
    panel = rp.frame_panel(bgr, [], [], [], [], [], [], max_width=900)
    assert panel.shape[1] == 900


def test_crop_around_centres_on_the_instance():
    bgr = np.zeros((300, 300, 3), np.uint8)
    m = _blob(300, 300, 100, 130, 100, 130)
    c = rp.crop_around(bgr, m)
    assert c is not None and c.shape[0] < 300


def test_crop_around_returns_none_for_an_empty_mask():
    assert rp.crop_around(np.zeros((50, 50, 3), np.uint8),
                          np.zeros((50, 50), bool)) is None


def test_data_uri_is_embeddable():
    uri = rp.png_data_uri(np.zeros((10, 10, 3), np.uint8))
    assert uri.startswith("data:image/jpeg;base64,") and len(uri) > 40


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def _record(tmp_path, name="f1", n_missed=1):
    import cv2
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), np.full((120, 120, 3), 60, np.uint8))
    m = _blob(120, 120)
    return {
        "item_id": name, "session_id": "s", "path": str(p),
        "width": 120, "height": 120,
        "n_gt": 1, "n_pred": 0, "n_matched": 0, "n_missed": n_missed,
        "n_false_pos": 0,
        "gt": [{"class_name": "grass_weed", "area_px": 120, "is_crop": False,
                "matched": n_missed == 0, "iou": 0.0}],
        "pred": [],
        "_gt_masks": [m], "_gt_names": ["grass_weed"],
        "_pred_masks": [], "_pred_names": [],
        "_matched_gt": [n_missed == 0], "_matched_pred": [],
    }


def test_html_is_self_contained(tmp_path):
    html = rp.build_html([_record(tmp_path)], ["grass_weed"], None, "val",
                         "best.pt")
    assert "data:image/jpeg;base64," in html
    assert "<img src=\"http" not in html and "<img src='http" not in html


def test_a_missing_crop_class_is_reported_as_unmeasured_not_safe(tmp_path):
    """The distinction that matters: nothing here says the crop is safe."""
    html = rp.build_html([_record(tmp_path)], ["grass_weed"], None, "val",
                         "best.pt")
    assert "UNMEASURED, not passing" in html


def test_missed_weeds_appear_in_the_gallery(tmp_path):
    html = rp.build_html([_record(tmp_path, n_missed=1)], ["grass_weed"],
                         None, "val", "best.pt")
    assert "Missed weeds, smallest first" in html
    assert "1 missed" in html


def test_no_misses_says_so_rather_than_showing_an_empty_gallery(tmp_path):
    html = rp.build_html([_record(tmp_path, n_missed=0)], ["grass_weed"],
                         None, "val", "best.pt")
    assert "No missed weeds in this split" in html


def test_frames_are_ordered_worst_first(tmp_path):
    good = _record(tmp_path, "good", n_missed=0)
    bad = _record(tmp_path, "bad", n_missed=1)
    html = rp.build_html([good, bad], ["grass_weed"], None, "val", "best.pt",
                         max_frames=2)
    assert html.index("<b>bad</b>") < html.index("<b>good</b>")


def test_the_generalisation_caveat_is_always_present(tmp_path):
    html = rp.build_html([_record(tmp_path)], ["grass_weed"], None, "val",
                         "best.pt")
    assert "not evidence of generalisation" in html
