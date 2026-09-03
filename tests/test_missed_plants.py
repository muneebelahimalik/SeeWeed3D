"""Plants nobody labelled: the audit that decides whether a drive can be used
as WHOLE FRAMES or only as a source of instance cut-outs.

The distinction the whole module rests on: a frame whose masks all sit a
pixel inside their leaves has a large unclaimed FRACTION and nothing missing,
while a frame with one unlabelled seedling among forty labelled plants has a
tiny fraction and a real hole in it. Counting blobs separates those; a fraction
cannot.
"""
import numpy as np
import pytest

from conftest import load_script

mp = load_script("annotation/missed_plants.py")

from common.vegetation import unclaimed_blobs  # noqa: E402

H = W = 240


def _blob(y, x, s, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[y:y + s, x:x + s] = True
    return m


# --------------------------------------------------------------------------
# blobs, not fractions - the reason this is worth measuring at all


def test_one_unlabelled_plant_is_found():
    veg = _blob(20, 20, 40) | _blob(120, 120, 40)
    n, px, mask = unclaimed_blobs(veg, _blob(20, 20, 40))
    assert n == 1 and px > 1000
    assert mask[130, 130] and not mask[30, 30]


def test_a_thin_rim_of_annotation_slop_is_not_a_missed_plant():
    """A hand-drawn polygon sits a pixel or two inside the leaf it traces. A
    check that fired on that would reject every frame in the project."""
    veg = _blob(50, 50, 60)
    claimed = _blob(52, 52, 56)
    assert unclaimed_blobs(veg, claimed)[0] == 0


def test_a_big_rim_is_still_not_a_plant_but_a_real_gap_is():
    """The dilation absorbs slop; it must not absorb a plant sitting next to a
    mask."""
    veg = _blob(50, 50, 60) | _blob(50, 130, 40)
    assert unclaimed_blobs(veg, _blob(50, 50, 60))[0] == 1


def test_specks_are_discarded():
    """A report full of leaf tips and green debris is one nobody reads."""
    veg = _blob(20, 20, 40) | _blob(150, 150, 6)
    assert unclaimed_blobs(veg, _blob(20, 20, 40))[0] == 0


def test_a_tiny_fraction_can_still_be_a_real_hole():
    """Forty labelled plants and one missed seedling: the fraction is noise and
    the blob count is 1. This is the case a fraction-based check cannot see."""
    veg = np.zeros((H, W), bool)
    claimed = np.zeros((H, W), bool)
    for i in range(5):
        for j in range(5):
            b = _blob(10 + i * 38, 10 + j * 38, 30)   # spans 10..192
            veg |= b
            claimed |= b
    veg |= _blob(205, 205, 20)                 # the missed one, clear of them
    n, px, _ = unclaimed_blobs(veg, claimed)
    assert n == 1
    assert px / veg.sum() < 0.02, "a fraction check would call this clean"


def test_nothing_claimed_means_everything_is_unclaimed():
    veg = _blob(20, 20, 40) | _blob(120, 120, 40)
    assert unclaimed_blobs(veg, np.zeros((H, W), bool))[0] == 2


def test_no_vegetation_means_nothing_missed():
    assert unclaimed_blobs(np.zeros((H, W), bool), _blob(0, 0, 10))[0] == 0


# --------------------------------------------------------------------------
# what the audit says to DO


def _frames(counts):
    return {f"f{i:03d}": {"n_missed": n, "missed_px": n * 900, "veg_px": 90000,
                          "missed_frac": n * 0.01}
            for i, n in enumerate(counts)}


def test_a_clean_drive_is_told_to_train_on_whole_frames():
    """Whole frames keep real weed-beside-weed context and real lighting - all
    things a cut-out loses - so 'clean' must recommend keeping them."""
    v = mp.verdict(_frames([0, 0, 0, 0]))
    assert v.startswith("CLEAN") and "WHOLE" in v
    assert "misses dark" in v, "the prior's blind spot belongs in the verdict"


def test_a_dirty_drive_is_told_to_become_a_cutout_source():
    """The actionable half: a cut-out carries the labelled pixels and leaves
    the missed ones behind."""
    v = mp.verdict(_frames([4, 5, 3, 6, 4]))
    assert "NOT SAFE AS WHOLE FRAMES" in v
    assert "compose_mixed" in v and "CUT-OUT SOURCE" in v


def test_a_few_bad_frames_do_not_condemn_the_drive():
    v = mp.verdict(_frames([0] * 40 + [5]))
    assert v.startswith("MOSTLY CLEAN")
    assert "exclude those few" in v


def test_an_empty_audit_says_so_rather_than_passing():
    assert "No frames" in mp.verdict({})


def test_the_verdict_threshold_is_the_configured_one():
    frames = _frames([2, 2, 2, 2])
    assert mp.verdict(frames, unsafe=2).startswith("NOT SAFE")
    assert not mp.verdict(frames, unsafe=3).startswith("NOT SAFE")


# --------------------------------------------------------------------------
# the frame audit and the report


def test_audit_frame_counts_against_a_supplied_vegetation_mask():
    """The prior is supplied so the test does not depend on ExG thresholds
    against a synthetic image - what is under test is the accounting."""
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(20, 20, 40) | _blob(120, 120, 40)
    rec, mask = mp.audit_frame(bgr, _blob(20, 20, 40), veg=veg)
    assert rec["n_missed"] == 1
    assert rec["veg_px"] == int(veg.sum())
    assert 0 < rec["missed_frac"] < 1
    assert mask.any()


def test_worst_ranks_by_patches_then_area():
    per = {"a": {"n_missed": 1, "missed_px": 5000},
           "b": {"n_missed": 4, "missed_px": 100},
           "c": {"n_missed": 0, "missed_px": 0}}
    assert mp.worst(per, 3) == ["b", "a"], "a clean frame is not 'worst'"


def test_the_report_carries_the_caveat_that_a_patch_is_not_a_proven_weed():
    """The prior calls moss and debris vegetation and misses dark seedlings, so
    the number cannot settle it and the report must not imply it does."""
    txt = mp.format_report({"vid3": _frames([2, 0, 3])})
    assert "PLACE TO LOOK" in txt
    assert "not proof of a clean frame either" in txt


def test_the_report_names_the_worst_frames():
    txt = mp.format_report({"vid3": _frames([0, 7, 0])})
    assert "f001" in txt and "7 patch" in txt


def test_summarise_counts_frames_and_patches():
    s = mp.summarise(_frames([0, 2, 3]))
    assert s["frames"] == 3 and s["missed_blobs"] == 5
    assert s["frames_with_missed"] == 2


def test_summarise_of_nothing_does_not_divide_by_zero():
    assert mp.summarise({})["mean_blobs_per_frame"] == 0.0


def test_draw_marks_missed_vegetation_and_outlines_what_was_claimed():
    bgr = np.zeros((H, W, 3), np.uint8)
    vis = mp.draw(bgr, _blob(20, 20, 40), _blob(120, 120, 40))
    assert vis.shape == bgr.shape
    assert vis[140, 140, 2] > vis[140, 140, 0], "missed vegetation drawn red"
