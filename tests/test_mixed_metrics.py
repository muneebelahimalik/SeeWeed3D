"""The mixed-scene benchmark: the asymmetry, identity, and cluster over-use.

These pin the properties the metric exists for - that the catastrophic
direction is never averaged into the harmless one, that a merge and a fragment
cannot cancel each other, and that "no contact in this frame" is reported as
absent rather than as success.
"""
import numpy as np

from conftest import load_script

met = load_script("evaluation/metrics.py")

from common.ontology import CROP_CLASS  # noqa: E402

H = W = 80


def _rect(x, y, w, h):
    m = np.zeros((H, W), bool)
    m[y:y + h, x:x + w] = True
    return m


# --------------------------------------------------------------------------- #
# The asymmetry
# --------------------------------------------------------------------------- #
def test_the_two_error_directions_are_reported_separately():
    """A mean would let a laser-on-crop error hide behind a missed weed."""
    onion, weed = _rect(10, 10, 20, 20), _rect(50, 10, 20, 20)
    gt_m, gt_c = [onion, weed], [CROP_CLASS, "other_weed"]

    # The whole onion is called weed: catastrophic, and nothing else is wrong.
    r = met.crop_confusion([onion, weed], ["other_weed", "other_weed"],
                           gt_m, gt_c)
    assert r["onion_as_weed_fraction"] == 1.0
    assert r["weed_as_onion_fraction"] == 0.0


def test_calling_a_weed_onion_is_the_other_direction():
    onion, weed = _rect(10, 10, 20, 20), _rect(50, 10, 20, 20)
    r = met.crop_confusion([onion, weed], [CROP_CLASS, CROP_CLASS],
                           [onion, weed], [CROP_CLASS, "other_weed"])
    assert r["weed_as_onion_fraction"] == 1.0
    assert r["onion_as_weed_fraction"] == 0.0


def test_crop_the_model_saw_as_nothing_is_counted_apart():
    """Not fired at, but not protected either - a different failure from
    calling it weed, so it is not folded into that number."""
    onion = _rect(10, 10, 20, 20)
    r = met.crop_confusion([], [], [onion], [CROP_CLASS])
    assert r["onion_unclaimed_px"] == int(onion.sum())
    assert r["onion_as_weed_px"] == 0


def test_a_perfect_prediction_scores_zero_in_both_directions():
    onion, weed = _rect(10, 10, 20, 20), _rect(50, 10, 20, 20)
    r = met.crop_confusion([onion, weed], [CROP_CLASS, "other_weed"],
                           [onion, weed], [CROP_CLASS, "other_weed"])
    assert r["onion_as_weed_px"] == 0 and r["weed_as_onion_px"] == 0


# --------------------------------------------------------------------------- #
# The contact band
# --------------------------------------------------------------------------- #
def test_errors_at_the_contact_are_reported_apart_from_the_frame():
    """A frame of well-separated plants can score perfectly while every
    onion/weed contact in it is wrong."""
    onion = _rect(10, 10, 30, 30)
    weed = _rect(40, 10, 30, 30)            # touching along x=40
    # Predict the onion 6 px too wide, eating into the weed.
    pred_onion = _rect(10, 10, 36, 30)
    r = met.crop_confusion([pred_onion, _rect(46, 10, 24, 30)],
                           [CROP_CLASS, "other_weed"],
                           [onion, weed], [CROP_CLASS, "other_weed"],
                           contact_band_px=10)
    assert r["contact_band_px_count"] > 0
    assert r["contact_weed_as_onion_fraction"] > 0.0


def test_a_frame_with_no_contact_reports_none_not_zero():
    """'Nothing to get wrong here' and 'got the hard part right' are different
    claims, and a 0 would let the first be read as the second."""
    r = met.crop_confusion([_rect(10, 10, 10, 10)], [CROP_CLASS],
                           [_rect(10, 10, 10, 10)], [CROP_CLASS])
    assert r["contact_band_px_count"] == 0
    assert r["contact_onion_as_weed_fraction"] is None


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_one_prediction_over_two_plants_is_a_merge():
    a, b = _rect(10, 10, 20, 20), _rect(32, 10, 20, 20)
    merged = _rect(10, 10, 42, 20)
    r = met.identity_errors([merged], [a, b])
    assert r["n_merged_predictions"] == 1
    assert r["n_merged_gt_instances"] == 2
    assert r["n_fragmented_gt"] == 0


def test_two_predictions_inside_one_plant_are_a_fragment():
    whole = _rect(10, 10, 40, 20)
    r = met.identity_errors([_rect(10, 10, 18, 20), _rect(32, 10, 18, 20)],
                            [whole])
    assert r["n_fragmented_gt"] == 1
    assert r["n_merged_predictions"] == 0


def test_a_merge_and_a_fragment_do_not_cancel():
    """The reason count_error alone is not enough: this frame merges two plants
    and shatters a third, and comes out with the right number of instances."""
    a, b, c = _rect(5, 5, 15, 15), _rect(22, 5, 15, 15), _rect(5, 40, 30, 15)
    pred = [_rect(5, 5, 32, 15),                       # merges a and b
            _rect(5, 40, 13, 15), _rect(21, 40, 14, 15)]   # shatters c
    r = met.identity_errors(pred, [a, b, c])
    assert r["count_error"] == 0, "the fixture must hide behind the count"
    assert r["n_merged_predictions"] == 1 and r["n_fragmented_gt"] == 1


def test_perfect_instances_have_no_merges_or_fragments():
    a, b = _rect(10, 10, 20, 20), _rect(40, 10, 20, 20)
    r = met.identity_errors([a, b], [a, b])
    assert r["n_merged_predictions"] == 0 and r["n_fragmented_gt"] == 0
    assert r["merge_rate"] == 0.0 and r["fragment_rate"] == 0.0


def test_identity_ignores_the_class():
    """A merge of two plants is the same defect whatever they are called."""
    a, b = _rect(10, 10, 20, 20), _rect(32, 10, 20, 20)
    r = met.identity_errors([_rect(10, 10, 42, 20)], [a, b])
    assert r["n_merged_predictions"] == 1


# --------------------------------------------------------------------------- #
# Cluster over-use
# --------------------------------------------------------------------------- #
def test_a_cluster_over_separable_plants_is_flagged():
    """Annotation policy becomes deployed policy - this is the number that
    catches it happening."""
    a, b = _rect(10, 10, 20, 20), _rect(32, 10, 20, 20)
    r = met.cluster_over_prediction([_rect(10, 10, 42, 20)], ["weed_cluster"],
                                    [a, b], ["other_weed", "other_weed"])
    assert r["clusters_over_separable_gt"] == 1
    assert r["cluster_over_prediction_rate"] == 1.0


def test_a_cluster_over_a_true_cluster_is_not_flagged():
    truth = _rect(10, 10, 42, 20)
    r = met.cluster_over_prediction([truth], ["weed_cluster"],
                                    [truth], ["weed_cluster"])
    assert r["clusters_over_separable_gt"] == 0


def test_splitting_a_true_cluster_is_not_counted_here():
    """The opposite direction produces targets that can be checked, not targets
    that silently never exist, so it is not this metric's business."""
    truth = _rect(10, 10, 42, 20)
    r = met.cluster_over_prediction([_rect(10, 10, 20, 20),
                                     _rect(32, 10, 20, 20)],
                                    ["other_weed", "other_weed"],
                                    [truth], ["weed_cluster"])
    assert r["n_predicted_clusters"] == 0
    assert r["cluster_over_prediction_rate"] is None


def test_mixed_scene_metrics_returns_the_three_groups_unmerged():
    """Deliberately not a single score: the groups answer different questions
    and collapsing them reproduces the averaging this module avoids."""
    a, b = _rect(10, 10, 20, 20), _rect(40, 10, 20, 20)
    r = met.mixed_scene_metrics([a, b], [CROP_CLASS, "other_weed"],
                                [a, b], [CROP_CLASS, "other_weed"])
    assert set(r) == {"crop", "identity", "cluster"}
    assert r["crop"]["onion_as_weed_px"] == 0
