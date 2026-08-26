"""Three things a self-training report showed but never said out loud.

All three came out of one real round: 79 frames scored on an unseen session,
68 accepted, 11 in review. The numbers were all present in the report and every
one of them had to be read off by hand and reasoned about before it meant
anything. The lesson this project keeps relearning is that a silent wrong
answer beats a loud one every time - so make it loud.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from training import pseudo_label as pl                          # noqa: E402
from training.datasets.weeds_selftrain import (                  # noqa: E402
    stride_redundancy_warning)
from training.splits import MIN_SEAM_SEPARATION                  # noqa: E402

REAL = [[0.5, 68], [0.6, 68], [0.7, 68], [0.8, 68], [0.9, 19]]


def _q(score, failed=(), empty=False):
    return {"score": score, "gates_failed": list(failed), "empty": empty}


# --------------------------------------------------------------------------- #
# 1. the threshold that is not a threshold
# --------------------------------------------------------------------------- #
def test_the_real_sweep_is_reported_as_flat():
    """THE CASE. ACCEPT was 0.70 and 86% of frames were accepted, which reads
    as 'the threshold is too low'. It was not: 0.5 through 0.8 all accept the
    same 68, so ACCEPT was not deciding anything and raising it would have
    changed nothing. The gates were deciding."""
    assert pl.flat_sweep(REAL) == (0.5, 0.8, 68)


def test_a_sweep_that_actually_discriminates_is_not_flagged():
    assert pl.flat_sweep([[0.5, 70], [0.6, 60], [0.7, 50],
                          [0.8, 30], [0.9, 10]]) is None


def test_two_equal_cuts_are_not_enough_to_call_it_flat():
    """Any monotone sweep has ties. Flat means a span wide enough that the
    threshold provably has no effect across it."""
    assert pl.flat_sweep([[0.5, 70], [0.6, 70], [0.7, 50],
                          [0.8, 30], [0.9, 10]]) is None


def test_the_widest_flat_span_wins():
    assert pl.flat_sweep([[0.5, 9], [0.6, 9], [0.7, 9],
                          [0.8, 4], [0.9, 4]]) == (0.5, 0.7, 9)


def test_a_sweep_flat_at_zero_is_not_a_flat_threshold():
    """Everything rejected is a different problem with a different fix, and
    calling it 'the threshold is not selecting' would point at the wrong knob."""
    assert pl.flat_sweep([[0.5, 0], [0.6, 0], [0.7, 0],
                          [0.8, 0], [0.9, 0]]) is None


def test_a_sweep_flat_all_the_way_across_is_flagged():
    assert pl.flat_sweep([[0.5, 12], [0.6, 12], [0.7, 12],
                          [0.8, 12], [0.9, 12]]) == (0.5, 0.9, 12)


def test_summarise_carries_the_verdict_into_the_report_json():
    s = pl.summarise([_q(0.88) for _ in range(68)]
                     + [_q(0.40, ["veg_recall"]) for _ in range(11)])
    assert s["flat_sweep"] is not None


def test_the_readout_says_what_to_do_instead():
    """A warning that only names the problem sends someone back to the same
    knob. This one has to redirect at the gates and the spot check."""
    s = pl.summarise([_q(0.88) for _ in range(68)]
                     + [_q(0.40, ["veg_recall"]) for _ in range(11)])
    text = pl.format_report(s, n_hand=48, budget=96)
    assert "NOT SELECTING" in text
    assert "spot_check" in text and "gates" in text


def test_a_discriminating_run_prints_no_such_warning():
    s = pl.summarise([_q(0.95) for _ in range(5)]
                     + [_q(0.75) for _ in range(5)]
                     + [_q(0.55) for _ in range(5)])
    assert "NOT SELECTING" not in pl.format_report(s, n_hand=48, budget=96)


# --------------------------------------------------------------------------- #
# 2. the gate that never fires
# --------------------------------------------------------------------------- #
def test_every_gate_appears_even_when_it_never_fires():
    """veg_precision failed zero times, so it was absent from the dict - which
    reads as 'no problem' and is indistinguishable from 'not being applied'."""
    s = pl.summarise([_q(0.88) for _ in range(3)]
                     + [_q(0.40, ["veg_recall"])])
    assert set(s["gate_failures"]) == set(pl.GATES)
    assert s["gate_failures"]["veg_precision"] == 0


def test_a_gate_that_never_fired_says_so_in_words():
    s = pl.summarise([_q(0.88) for _ in range(3)]
                     + [_q(0.40, ["veg_recall"])])
    text = pl.format_report(s)
    assert "never fired" in text
    assert "would become BACKGROUND" in text, "the firing gate keeps its reason"


def test_gate_counts_are_still_correct():
    s = pl.summarise([_q(0.40, ["veg_recall"]) for _ in range(11)]
                     + [_q(0.30, ["veg_precision", "veg_recall"])])
    assert s["gate_failures"]["veg_recall"] == 12
    assert s["gate_failures"]["veg_precision"] == 1


# --------------------------------------------------------------------------- #
# 3. frames that are not as many frames as they look
# --------------------------------------------------------------------------- #
def test_the_real_stride_is_called_out():
    """THE CASE. 393 raw frames, INFER_STRIDE 5, 68 accepted - which is about
    6 frames of distinct ground, each plant appearing in ~12 of them, all of it
    weighted as 68 against 48 hand-corrected."""
    w = stride_redundancy_warning(5, 68, 48)
    assert w and "INFER_STRIDE is 5" in w
    assert "6 frame(s) of distinct ground" in w
    assert "48 hand-corrected" in w


def test_it_reuses_the_projects_own_separation_floor():
    """A second, independently-chosen number for 'far enough apart' is how two
    parts of a pipeline come to disagree about the same physical fact."""
    assert MIN_SEAM_SEPARATION in (60,)
    assert stride_redundancy_warning(MIN_SEAM_SEPARATION, 68, 48) is None


def test_a_stride_above_the_floor_is_silent():
    assert stride_redundancy_warning(90, 68, 48) is None


def test_a_stride_one_below_the_floor_still_warns():
    assert stride_redundancy_warning(MIN_SEAM_SEPARATION - 1, 68, 48)


def test_nothing_accepted_means_nothing_to_warn_about():
    assert stride_redundancy_warning(5, 0, 48) is None


def test_it_works_without_a_hand_frame_count():
    w = stride_redundancy_warning(5, 68)
    assert w and "hand-corrected" not in w


def test_a_missing_stride_is_treated_as_one():
    assert stride_redundancy_warning(0, 68, 48)
    assert stride_redundancy_warning(None, 68, 48)


def test_the_distinct_count_never_reads_as_zero():
    """Rounding to 0 would say the batch contains no ground at all, which is
    both false and easy to dismiss as a bug."""
    w = stride_redundancy_warning(1, 1, 48)
    assert "1 frame(s) of distinct ground" in w


def test_the_warning_names_the_fix():
    w = stride_redundancy_warning(5, 68, 48)
    assert f"Raise INFER_STRIDE to {MIN_SEAM_SEPARATION}" in w


def test_the_runner_prints_it_after_the_batch_counts():
    """It explains what those counts mean, so it has to be readable next to
    them rather than scrolled off above."""
    import inspect
    from training.datasets import weeds_selftrain as st
    src = inspect.getsource(st.main)
    assert "stride_redundancy_warning" in src
    assert src.index("review/") < src.index("stride_redundancy_warning")
