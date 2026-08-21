"""Reading a model's raw mask output without guessing at the threshold.

THE BUG THIS EXISTS FOR
-----------------------
Both SAM prelabelers ended their mask extraction with `arr.astype(bool)`, which
is True for every non-zero value. On a binary mask that is exact and free. On a
probability or logit mask it is a threshold nobody chose: the mask inflates
outward through its own soft edge into low-probability background.

The visible symptom is a boundary that sits OUTSIDE the plant in every
direction and never cuts inside it - which is what the weed prelabels looked
like, and which nothing raises an error about. These masks become CVAT
prelabels, prelabels become the training target, and the training target is the
ceiling on every model trained from it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import masks as mk                        # noqa: E402


@pytest.fixture(autouse=True)
def _fresh():
    mk.reset_reporting()
    yield
    mk.reset_reporting()


# --------------------------------------------------------------------------- #
# Classifying what the array holds
# --------------------------------------------------------------------------- #
def test_a_bool_array_is_binary():
    assert mk.mask_encoding(np.zeros((4, 4), bool)) == mk.BINARY


def test_a_zero_one_integer_array_is_binary():
    assert mk.mask_encoding(np.array([[0, 1], [1, 0]], np.uint8)) == mk.BINARY


def test_the_uint8_image_convention_is_binary():
    """0/255 is how a mask is stored as an image, and it is unambiguous."""
    assert mk.mask_encoding(np.array([[0, 255], [255, 0]], np.uint8)) == mk.BINARY


def test_floats_in_the_unit_interval_are_probabilities():
    a = np.array([[0.01, 0.4], [0.6, 0.99]], np.float32)
    assert mk.mask_encoding(a) == mk.PROBABILITY


def test_a_float_array_holding_only_zero_and_one_is_safe_either_way():
    """Stored as float, meant as binary. Read as a probability it thresholds at
    0.5, which gives the identical answer - so it does not matter which name it
    gets, and the test pins that it cannot go wrong."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
    assert np.array_equal(mk.to_bool(a), mk.naive_bool(a))


def test_signed_scores_are_logits():
    a = np.array([[-12.0, -0.5], [0.5, 9.0]], np.float32)
    assert mk.mask_encoding(a) == mk.LOGIT


def test_non_negative_floats_above_one_are_ambiguous():
    """All-positive and out of [0,1]: neither a probability nor clearly a
    logit. Inventing a threshold here would trade a known unknown for an
    unknown one."""
    a = np.array([[0.0, 3.0], [7.5, 0.0]], np.float32)
    assert mk.mask_encoding(a) == mk.AMBIGUOUS


def test_an_empty_array_does_not_crash_the_classifier():
    assert mk.mask_encoding(np.zeros((0, 4), np.float32)) == mk.BINARY


def test_nan_does_not_decide_the_encoding():
    a = np.array([[np.nan, 0.2], [0.8, np.nan]], np.float32)
    assert mk.mask_encoding(a) == mk.PROBABILITY


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #
def test_a_probability_mask_is_cut_at_a_half_not_at_zero():
    """THE BUG. Everything above 0.0 is the soft edge; everything above 0.5 is
    the plant."""
    a = np.array([[0.0, 0.05, 0.2], [0.49, 0.51, 0.99]], np.float32)
    assert mk.to_bool(a).sum() == 2
    assert mk.naive_bool(a).sum() == 5, "the old behaviour kept the whole ramp"


def test_a_logit_mask_is_cut_at_zero():
    a = np.array([[-8.0, -0.1], [0.1, 6.0]], np.float32)
    assert mk.to_bool(a).sum() == 2
    assert mk.naive_bool(a).sum() == 4


def test_a_binary_mask_is_untouched():
    a = np.array([[0, 1], [1, 1]], np.uint8)
    assert np.array_equal(mk.to_bool(a), mk.naive_bool(a))


def test_an_ambiguous_array_keeps_the_old_behaviour():
    """Loud, not clever: it reports rather than picking a number nobody chose."""
    a = np.array([[0.0, 3.0], [7.5, 0.0]], np.float32)
    assert np.array_equal(mk.to_bool(a), mk.naive_bool(a))


def test_thresholding_a_soft_edge_shrinks_the_mask_toward_the_plant():
    """A synthetic plant with a soft rim - the shape of the real failure."""
    a = np.zeros((40, 40), np.float32)
    a[10:30, 10:30] = 0.95                       # the plant
    a[8:32, 8:32] = np.maximum(a[8:32, 8:32], 0.15)   # low-probability halo
    assert mk.naive_bool(a).sum() == 24 * 24     # halo included
    assert mk.to_bool(a).sum() == 20 * 20        # halo excluded


# --------------------------------------------------------------------------- #
# The report - this is what answers "did it ever matter"
# --------------------------------------------------------------------------- #
def test_the_report_names_the_encoding_and_the_range():
    a = np.array([[0.0, 0.3], [0.7, 1.0]], np.float32)
    note = mk.describe_mask_encoding(a, "SAM 3 (weeds)")
    assert "SAM 3 (weeds)" in note
    assert "PROBABILITY" in note
    assert "range=" in note


def test_the_report_gives_the_area_ratio_when_it_would_have_mattered():
    """The only number that answers the question: how much bigger would the
    old behaviour have made every mask."""
    a = np.zeros((40, 40), np.float32)
    a[10:30, 10:30] = 0.95
    a[5:35, 5:35] = np.maximum(a[5:35, 5:35], 0.1)
    note = mk.describe_mask_encoding(a, "x")
    assert "2.25x larger" in note
    assert "inflated every mask" in note


def test_the_report_says_so_when_it_would_have_made_no_difference():
    """Ruling the hypothesis OUT has to be as clear as confirming it, or the
    check gets run and misread."""
    a = np.array([[0.0, 1.0], [1.0, 1.0]], np.float32)
    note = mk.describe_mask_encoding(a, "x")
    assert "never the boundary problem" in note


def test_a_binary_array_reports_that_there_is_nothing_to_decide():
    note = mk.describe_mask_encoding(np.zeros((4, 4), bool), "x")
    assert "nothing to decide" in note


def test_an_ambiguous_array_is_reported_loudly():
    a = np.array([[0.0, 3.0], [7.5, 0.0]], np.float32)
    note = mk.describe_mask_encoding(a, "x")
    assert "[!]" in note and "guess" in note


def test_each_source_is_described_once_per_run():
    """Once per run is a fact worth having in every log; once per frame trains
    people to scroll past it."""
    a = np.array([[0.0, 1.0]], np.float32)
    assert mk.describe_mask_encoding(a, "SAM 3 (weeds)") is not None
    assert mk.describe_mask_encoding(a, "SAM 3 (weeds)") is None
    assert mk.describe_mask_encoding(a, "SAM 3 (onions)") is not None


# --------------------------------------------------------------------------- #
# The prelabelers actually use it
# --------------------------------------------------------------------------- #
def _state_masks_of(module_name):
    import importlib
    return importlib.import_module(module_name)._state_masks


@pytest.mark.parametrize("module", ["annotation.prelabel_weeds_sam3",
                                    "annotation.prelabel_onions_sam3"])
def test_state_masks_thresholds_probabilities(module, capsys):
    """Both prelabelers had their own copy of this function and their own copy
    of the bug."""
    fn = _state_masks_of(module)
    a = np.zeros((1, 20, 20), np.float32)
    a[0, 5:15, 5:15] = 0.9
    a[0, 3:17, 3:17] = np.maximum(a[0, 3:17, 3:17], 0.2)
    out = fn({"masks": a})
    assert len(out) == 1
    assert out[0].dtype == np.bool_
    assert out[0].sum() == 100, "the 0.2 halo must not be foreground"
    assert "PROBABILITY" in capsys.readouterr().out


@pytest.mark.parametrize("module", ["annotation.prelabel_weeds_sam3",
                                    "annotation.prelabel_onions_sam3"])
def test_state_masks_leaves_a_binary_mask_alone(module):
    fn = _state_masks_of(module)
    a = np.zeros((2, 8, 8), bool)
    a[0, 1:4, 1:4] = True
    a[1, 5:7, 5:7] = True
    out = fn({"masks": a})
    assert [int(m.sum()) for m in out] == [9, 4]


@pytest.mark.parametrize("module", ["annotation.prelabel_weeds_sam3",
                                    "annotation.prelabel_onions_sam3"])
def test_state_masks_still_handles_missing_and_odd_shapes(module):
    fn = _state_masks_of(module)
    assert fn({}) == []
    assert fn({"masks": None}) == []
    assert fn(None) == []
    # [N,1,H,W] squeezes to [N,H,W]; a bare [H,W] gains a batch dim.
    assert len(fn({"masks": np.ones((3, 1, 6, 6), bool)})) == 3
    assert len(fn({"masks": np.ones((6, 6), bool)})) == 1


# --------------------------------------------------------------------------- #
# The cluster-rate guard
# --------------------------------------------------------------------------- #
def test_the_cluster_warning_fires_on_a_prelabel_run_and_not_on_corrected_data():
    """`weed_cluster` means "no separable single LEP", so every one is a plant
    that never gets targeted individually - and a prelabel arriving already
    marked as a cluster biases the annotator toward accepting it. The threshold
    has to separate the observed regimes, not read tidily.

    Measured: the hand-corrected session carries 2 clusters in 1,444 instances;
    the two SAM prelabel runs proposed 800 in 11,816 and 380 in 9,592."""
    from annotation.prelabel_weeds_sam3 import CLUSTER_SHARE_WARN
    assert 800 / 11816 >= CLUSTER_SHARE_WARN      # 6.8% - flagged
    assert 380 / 9592 >= CLUSTER_SHARE_WARN       # 4.0% - flagged
    assert 2 / 1444 < CLUSTER_SHARE_WARN          # 0.1% - silent
