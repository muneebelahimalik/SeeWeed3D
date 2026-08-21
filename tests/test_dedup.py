"""One plant, one detection.

THE BUG THIS EXISTS FOR
-----------------------
RF-DETR is a set-prediction model: every query proposes independently, and
nothing makes two queries that found the same plant agree on what it is. Real
output from a weed session:

    other_weed               0.275   bbox [1861.4, 121.1, 24.4, 19.0]
    cutleaf_evening_primrose 0.496   bbox [1861.4, 121.1, 24.4, 19.0]

Identical box, IoU 1.000, seen in 6 of 16 frames. A laser weeder fires twice at
that plant while a weed elsewhere goes untreated; a prelabel export hands the
annotator two overlapping polygons to delete; every per-instance count is
inflated by an amount nobody measured.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import dedup as dd                              # noqa: E402
from perception.segmenter import Detections, dedup_detections  # noqa: E402


def box(x, y, w, h, score, name="other_weed"):
    return {"bbox": [x, y, w, h], "score": score, "class_name": name}


def blob(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# --------------------------------------------------------------------------- #
# The observed failure
# --------------------------------------------------------------------------- #
def test_the_same_box_under_two_classes_becomes_one_detection():
    """The real case, with the real numbers."""
    items = [box(1861.4, 121.1, 24.4, 19.0, 0.275, "other_weed"),
             box(1861.4, 121.1, 24.4, 19.0, 0.496, "cutleaf_evening_primrose")]
    kept, dropped = dd.suppress_duplicates(items)
    assert len(kept) == 1 and len(dropped) == 1


def test_the_higher_scoring_label_is_the_one_that_survives():
    """It is the model's own answer to the question it was unsure about."""
    items = [box(10, 10, 20, 20, 0.275, "other_weed"),
             box(10, 10, 20, 20, 0.496, "cutleaf_evening_primrose")]
    kept, _ = dd.suppress_duplicates(items)
    assert kept[0]["class_name"] == "cutleaf_evening_primrose"


def test_suppression_is_class_agnostic():
    """Suppressing only WITHIN a class leaves the case that actually happens
    untouched, which is the whole point."""
    items = [box(10, 10, 20, 20, 0.9, "grass_weed"),
             box(10, 10, 20, 20, 0.4, "other_weed")]
    assert len(dd.suppress_duplicates(items)[0]) == 1


# --------------------------------------------------------------------------- #
# What it must NOT merge
# --------------------------------------------------------------------------- #
def test_two_adjacent_plants_are_left_alone():
    """Merging these costs a real weed its own treatment point, which is the
    failure this project cares about most."""
    items = [box(0, 0, 100, 100, 0.9), box(60, 0, 100, 100, 0.8)]   # IoU ~0.25
    assert len(dd.suppress_duplicates(items)[0]) == 2


def test_the_default_threshold_sits_above_the_touching_plant_regime():
    """Observed duplicates are IoU 1.0; two touching plants reach 0.5-0.6."""
    assert 0.7 < dd.DEFAULT_DEDUP_IOU < 1.0


def test_masks_decide_where_they_exist():
    """Two plants can SHARE a bounding box while overlapping barely at all - an
    L-shaped weed beside a compact one. Box IoU would merge them."""
    shape = (40, 40)
    a = blob(shape, 0, 40, 0, 6)         # tall thin bar
    b = blob(shape, 34, 40, 0, 40)       # wide flat bar, same bbox corner
    items = [{"bbox": [0, 0, 40, 40], "score": 0.9, "mask": a,
              "class_name": "grass_weed"},
             {"bbox": [0, 0, 40, 40], "score": 0.8, "mask": b,
              "class_name": "other_weed"}]
    assert dd.box_iou(items[0]["bbox"], items[1]["bbox"]) == 1.0
    assert dd.mask_iou(a, b) < 0.2
    assert len(dd.suppress_duplicates(items)[0]) == 2, (
        "box IoU merged two plants that barely overlap")


def test_an_empty_or_single_detection_list_is_returned_unchanged():
    assert dd.suppress_duplicates([]) == ([], [])
    one = [box(0, 0, 10, 10, 0.5)]
    assert dd.suppress_duplicates(one) == (one, [])


def test_zero_iou_disables_it_entirely():
    items = [box(10, 10, 20, 20, 0.9), box(10, 10, 20, 20, 0.4)]
    assert len(dd.suppress_duplicates(items, iou=0)[0]) == 2


def test_a_degenerate_box_never_matches():
    assert dd.box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert dd.box_iou([0, 0, 10, 10], [0, 0, -5, 10]) == 0.0


def test_masks_of_different_shapes_do_not_match():
    assert dd.mask_iou(np.ones((4, 4), bool), np.ones((5, 5), bool)) == 0.0


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #
def test_survivors_keep_the_callers_ordering():
    """Callers sort for display; re-sorting here would silently change what an
    overlay shows relative to the JSON beside it."""
    items = [box(0, 0, 10, 10, 0.3, "a"), box(500, 500, 10, 10, 0.9, "b"),
             box(0, 0, 10, 10, 0.8, "c")]
    kept, _ = dd.suppress_duplicates(items)
    assert [k["class_name"] for k in kept] == ["b", "c"] or \
           [k["class_name"] for k in kept] == ["c", "b"]
    # index order preserved: 'c' (index 2) comes after 'b' (index 1)
    assert [k["class_name"] for k in kept] == ["b", "c"]


def test_nothing_is_mutated():
    items = [box(10, 10, 20, 20, 0.9), box(10, 10, 20, 20, 0.4)]
    before = [dict(i) for i in items]
    dd.suppress_duplicates(items)
    assert items == before


def test_it_is_deterministic_on_tied_scores():
    items = [box(10, 10, 20, 20, 0.5, "a"), box(10, 10, 20, 20, 0.5, "b")]
    first = dd.suppress_duplicates(items)[0][0]["class_name"]
    assert all(dd.suppress_duplicates(items)[0][0]["class_name"] == first
               for _ in range(5))


def test_a_chain_collapses_to_one():
    """Three near-identical detections must not leave two survivors."""
    items = [box(10, 10, 20, 20, 0.9), box(10, 10, 21, 20, 0.7),
             box(11, 10, 20, 20, 0.5)]
    kept, dropped = dd.suppress_duplicates(items)
    assert len(kept) == 1 and len(dropped) == 2


def test_the_report_names_the_dropped_labels():
    """A class pair recurring here is a labelling question, not a threshold
    one, so the classes have to be in the message."""
    kept = [box(0, 0, 10, 10, 0.9, "cutleaf_evening_primrose")]
    dropped = [box(0, 0, 10, 10, 0.3, "other_weed")]
    note = dd.describe_suppression(kept, dropped)
    assert "other_weed" in note and "50%" in note


def test_the_report_is_silent_when_nothing_was_suppressed():
    assert dd.describe_suppression([box(0, 0, 1, 1, 0.5)], []) is None


# --------------------------------------------------------------------------- #
# Detections carries every array together
# --------------------------------------------------------------------------- #
def _det(n, shape=(20, 20)):
    return Detections(
        masks=np.stack([blob(shape, i, i + 5, 0, 5) for i in range(n)]),
        boxes=np.array([[0, i, 5, 5] for i in range(n)], float),
        classes=np.zeros(n, int), scores=np.linspace(0.9, 0.3, n),
        width=shape[1], height=shape[0], names=["grass_weed"])


def test_select_moves_every_array_together():
    """Filtering the exported list while leaving masks alone is how an overlay
    draws detections the JSON beside it does not contain."""
    det = _det(4)
    sub = det.select([0, 2])
    assert len(sub) == 2
    assert sub.masks.shape[0] == 2 and sub.boxes.shape[0] == 2
    assert sub.classes.shape[0] == 2
    assert np.allclose(sub.scores, [det.scores[0], det.scores[2]])
    assert sub.names == det.names and sub.width == det.width


def test_select_of_nothing_is_an_empty_detections():
    assert len(_det(3).select([])) == 0


def test_dedup_detections_removes_the_duplicate_and_keeps_the_rest():
    shape = (20, 20)
    same = blob(shape, 0, 10, 0, 10)
    det = Detections(
        masks=np.stack([same, same.copy(), blob(shape, 12, 18, 12, 18)]),
        boxes=np.array([[0, 0, 10, 10], [0, 0, 10, 10], [12, 12, 6, 6]], float),
        classes=np.array([0, 1, 0]), scores=np.array([0.4, 0.8, 0.6]),
        width=20, height=20, names=["grass_weed", "other_weed"])
    out, dropped = dedup_detections(det)
    assert len(out) == 2 and len(dropped) == 1
    assert dropped[0]["class_name"] == "grass_weed"      # the lower score went


def test_dedup_detections_is_a_no_op_when_there_is_nothing_to_drop():
    det = _det(3)
    out, dropped = dedup_detections(det)
    assert dropped == [] and out is det


def test_dedup_detections_respects_a_zero_threshold():
    shape = (20, 20)
    same = blob(shape, 0, 10, 0, 10)
    det = Detections(masks=np.stack([same, same.copy()]),
                     boxes=np.array([[0, 0, 10, 10]] * 2, float),
                     classes=np.array([0, 0]), scores=np.array([0.9, 0.5]),
                     width=20, height=20, names=["grass_weed"])
    assert len(dedup_detections(det, iou=0)[0]) == 2
