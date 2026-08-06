"""Stage A evaluation: AP maths, and the reduced-class-list crop-safety bug.

The crop tests here are regression tests for a real defect: Detections resolved
the crop index against the FULL ontology while a --drop-classes checkpoint emits
indices into a REDUCED list. Dropping any class below onion_plant made
onion_safety_mask() return an empty mask and weed_indices() hand every onion to
the targeting stage.
"""
import numpy as np
import pytest

from conftest import load_script

ev = load_script("evaluation/eval_seg.py")
seg = load_script("perception/segmenter.py")
from seeweed3d.common.ontology import CLASSES, CROP_CLASS  # noqa: E402

FULL = list(CLASSES)
REDUCED = [c for c in CLASSES if c != "wild_radish"]     # what --drop-classes builds


def _det(classes, names, n=None, hw=(8, 8)):
    n = len(classes) if n is None else n
    masks = np.zeros((n,) + hw, bool)
    for i in range(n):
        masks[i, i % hw[0], :] = True
    return seg.Detections(masks, np.zeros((n, 4)), np.asarray(classes, int),
                          np.ones(n), hw[1], hw[0], names=list(names))


# --------------------------------------------------------------------------- #
# crop index resolution
# --------------------------------------------------------------------------- #
def test_crop_index_follows_the_models_own_class_list():
    assert _det([0], FULL).crop_index() == FULL.index(CROP_CLASS)
    assert _det([0], REDUCED).crop_index() == REDUCED.index(CROP_CLASS)
    # The whole point: dropping a class below the crop moves it.
    assert FULL.index(CROP_CLASS) != REDUCED.index(CROP_CLASS)


def test_onion_is_protected_under_a_reduced_class_list():
    """The regression. Index 4 is onion in REDUCED and other_weed in FULL."""
    idx = REDUCED.index(CROP_CLASS)
    d = _det([idx], REDUCED)
    m = d.onion_safety_mask()
    assert m is not None and m.any(), "crop mask lost under a reduced class list"
    assert d.weed_indices() == [], "an onion was handed over as a weed"


def test_onion_still_protected_under_the_full_ontology():
    idx = FULL.index(CROP_CLASS)
    d = _det([idx], FULL)
    assert d.onion_safety_mask().any()
    assert d.weed_indices() == []


def test_weeds_are_still_weeds_under_a_reduced_class_list():
    weed = REDUCED.index("grass_weed")
    d = _det([weed], REDUCED)
    assert d.weed_indices() == [0]
    assert not d.onion_safety_mask().any()


def test_crop_absent_from_vocabulary_reports_unavailable_not_empty():
    """A weeds-only model cannot predict a crop. That must be distinguishable
    from 'looked and found no crop', because only one of them is safe."""
    weeds_only = [c for c in CLASSES if c != CROP_CLASS]
    d = _det([0], weeds_only)
    assert d.crop_index() is None
    assert d.onion_safety_mask() is None


def test_class_name_uses_the_models_list():
    d = _det([1], REDUCED)
    assert d.class_name(0) == REDUCED[1] == "grass_weed"
    assert d.class_name(0) != CLASSES[1]


# --------------------------------------------------------------------------- #
# average precision
# --------------------------------------------------------------------------- #
def test_ap_is_one_for_perfect_ranked_detections():
    assert ev.average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_ap_is_undefined_not_zero_without_ground_truth():
    assert ev.average_precision([0.9], [False], 0) is None


def test_ap_is_zero_when_nothing_was_detected():
    assert ev.average_precision([], [], 3) == 0.0


def test_ap_penalises_a_false_positive_ranked_above_a_true_one():
    good = ev.average_precision([0.9, 0.5], [True, True], 2)
    bad = ev.average_precision([0.9, 0.5], [False, True], 2)
    assert bad < good


def test_ap_falls_when_ground_truth_is_missed():
    """One of two objects found, perfectly. Recall caps at 0.5, so does AP."""
    assert ev.average_precision([0.9], [True], 2) == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------- #
# score-ranked matching
# --------------------------------------------------------------------------- #
def _blob(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_highest_scoring_duplicate_wins_and_the_rest_are_false_positives():
    gt = [_blob(20, 20, 2, 10, 2, 10)]
    pred = [_blob(20, 20, 2, 10, 2, 10), _blob(20, 20, 2, 10, 2, 10)]
    tp = ev.match_for_ap(pred, [0.4, 0.9], gt, 0.5)
    assert tp.tolist() == [False, True], "the duplicate must not also count"


def test_a_prediction_below_the_iou_threshold_is_not_a_match():
    gt = [_blob(20, 20, 0, 4, 0, 4)]
    pred = [_blob(20, 20, 10, 14, 10, 14)]
    assert ev.match_for_ap(pred, [0.9], gt, 0.5).tolist() == [False]


def test_each_prediction_claims_a_different_ground_truth():
    gt = [_blob(20, 20, 0, 5, 0, 5), _blob(20, 20, 10, 15, 10, 15)]
    pred = [_blob(20, 20, 0, 5, 0, 5), _blob(20, 20, 10, 15, 10, 15)]
    assert ev.match_for_ap(pred, [0.9, 0.8], gt, 0.5).tolist() == [True, True]
