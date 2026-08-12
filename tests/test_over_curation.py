"""Curation that keeps almost nothing should say so.

Curation is the one step that decides how much data exists downstream, and it
reports success either way. "754 -> 30 usable frames" is a normal-looking line
and the run exits 0, so a MIN_SHIFT_FRAC left at whatever the last campaign
needed silently deletes 96% of a pool - and the number only looks wrong once
somebody adds it up across nine sessions.
"""
import pytest

from conftest import load_script

cp = load_script("extraction/curate_pool.py")

CFG = dict(cp.CONFIG, DROP_REDUNDANT=True)

#: Shaped like the rows curate_session already computes for its table.
SWEEP = [{"threshold": 0.05, "kept": 150, "dropped": 44},
         {"threshold": 0.10, "kept": 95, "dropped": 99},
         {"threshold": 0.15, "kept": 59, "dropped": 135},
         {"threshold": 0.25, "kept": 28, "dropped": 166},
         {"threshold": 0.40, "kept": 14, "dropped": 180},
         {"threshold": 0.60, "kept": 9, "dropped": 185}]


def test_a_healthy_curation_says_nothing():
    assert cp.over_curation_warning(194, 95, CFG, SWEEP) == []


def test_the_real_case_is_flagged():
    """194 -> 9 is what actually shipped, and nothing said a word."""
    out = cp.over_curation_warning(194, 9, CFG, SWEEP)
    assert out
    assert "kept only 9 of 194" in out[0]
    assert "probably the threshold, not the footage" in out[0]


def test_it_names_a_value_that_would_work(text=None):
    """A warning that does not say what to type instead just costs another dry
    run. The suggestion comes from the sweep already computed for this
    session, so it is that session's own numbers rather than a general rule."""
    out = " ".join(cp.over_curation_warning(194, 9, CFG, SWEEP))
    assert "MIN_SHIFT_FRAC 0.15 would keep 59" in out


def test_the_suggestion_is_the_loosest_value_that_still_works():
    """Loosest = least duplicated annotation effort among the usable options.
    Recommending 0.05 would trade one bad extreme for the other."""
    out = " ".join(cp.over_curation_warning(194, 9, CFG, SWEEP))
    assert "0.05" not in out and "0.1 would" not in out


def test_it_says_how_to_undo():
    out = " ".join(cp.over_curation_warning(194, 9, CFG, SWEEP))
    assert "RESTORE_ALL" in out and "no image was touched" in out


def test_a_small_session_is_flagged_on_count_even_at_a_fine_percentage():
    """40% of 20 frames is 8, which is not an annotation batch whatever the
    percentage says. Both the fraction and the floor have to pass."""
    out = cp.over_curation_warning(20, 8, CFG, SWEEP)
    assert out and "kept only 8 of 20" in out[0]


def test_a_genuinely_redundant_session_is_told_so_rather_than_given_a_bad_hint():
    """When even the loosest threshold keeps little, the footage really is that
    redundant and no value will fix it. Suggesting one anyway would be worse
    than saying nothing."""
    thin = [{"threshold": t, "kept": k, "dropped": 100 - k}
            for t, k in ((0.05, 6), (0.1, 5), (0.25, 3), (0.6, 2))]
    out = " ".join(cp.over_curation_warning(100, 2, CFG, thin))
    assert "really is that redundant" in out
    assert "would keep" not in out


def test_no_sweep_still_produces_the_warning():
    """The sweep is optional config. Losing the suggestion must not lose the
    warning that something is wrong."""
    out = cp.over_curation_warning(194, 9, CFG, None)
    assert out and "kept only 9 of 194" in out[0]


def test_redundancy_dropping_turned_off_is_not_second_guessed():
    """With DROP_REDUNDANT off, whatever survived was a deliberate manual
    choice and is none of this function's business."""
    off = dict(CFG, DROP_REDUNDANT=False)
    assert cp.over_curation_warning(194, 3, off, SWEEP) == []


def test_an_empty_pool_does_not_divide_by_zero():
    assert cp.over_curation_warning(0, 0, CFG, SWEEP) == []


def test_the_thresholds_are_deliberately_forgiving():
    """A slowly-driven session genuinely IS highly redundant, so this must flag
    'you deleted nearly everything', not 'you were slightly generous'."""
    assert cp.OVER_CURATION_KEEP_FRAC <= 0.2
    assert cp.over_curation_warning(100, 25, CFG, SWEEP) == []


def test_the_warning_is_wired_into_the_run():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "seeweed3d" / "extraction"
           / "curate_pool.py").read_text(encoding="utf-8")
    assert "over_curation_warning(before, after, cfg, sweep)" in src
