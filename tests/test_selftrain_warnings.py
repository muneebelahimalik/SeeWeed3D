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
    separation_warning)
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
def test_a_gap_below_the_floor_is_called_out():
    """THE CASE, restated after the fix. 393 raw frames, and what governs the
    training set is not how many were inferred but how far apart the EXPORTED
    ones are allowed to be."""
    w = separation_warning(5, 40, 48)
    assert w and "MIN_FRAME_GAP is 5" in w
    assert "48 hand-corrected" in w


def test_separation_turned_off_is_the_loudest_case():
    w = separation_warning(0, 40, 48)
    assert w and "no separation at all" in w


def test_it_reuses_the_projects_own_separation_floor():
    """A second, independently-chosen number for 'far enough apart' is how two
    parts of a pipeline come to disagree about the same physical fact."""
    assert MIN_SEAM_SEPARATION in (60,)
    assert separation_warning(MIN_SEAM_SEPARATION, 40, 48) is None


def test_a_gap_above_the_floor_is_silent():
    assert separation_warning(90, 40, 48) is None


def test_a_gap_one_below_the_floor_still_warns():
    assert separation_warning(MIN_SEAM_SEPARATION - 1, 40, 48)


def test_nothing_accepted_means_nothing_to_warn_about():
    assert separation_warning(5, 0, 48) is None


def test_it_works_without_a_hand_frame_count():
    w = separation_warning(5, 40)
    assert w and "hand-corrected" not in w


def test_inferring_every_frame_is_not_itself_a_warning():
    """The old version warned on INFER_STRIDE. Once sampling and separation
    became separate settings that fired on every correct run, and a warning you
    have to ignore to get work done stops being read at all."""
    import inspect
    from training.datasets import weeds_selftrain as st
    assert st.INFER_STRIDE == 1, "look at every frame by default"
    # The warning cannot depend on the stride: it has no way to see it.
    assert "stride" not in inspect.signature(separation_warning).parameters
    assert separation_warning(MIN_SEAM_SEPARATION, 400, 48) is None


def test_the_runner_prints_it_after_the_batch_counts():
    """It explains what those counts mean, so it has to be readable next to
    them rather than scrolled off above."""
    import inspect
    from training.datasets import weeds_selftrain as st
    src = inspect.getsource(st._finish)
    assert "separation_warning" in src
    assert src.index("review/") < src.index("separation_warning")


# --------------------------------------------------------------------------- #
# 4. choosing WHICH frames, now that every frame is looked at
# --------------------------------------------------------------------------- #
ORDER = [f"f{i:03d}" for i in range(393)]
POS = {f: i for i, f in enumerate(ORDER)}


def _sel(chosen, gap, scores=None):
    scores = scores or {}
    return pl.select_spread(ORDER, chosen, lambda f: scores.get(f, 0.5), gap)


def test_a_real_drive_yields_what_the_gap_allows():
    """393 frames at a gap of 60 is 7 windows - the same count a stride of 60
    would have given. The difference is WHICH frame comes out of each."""
    kept, _ = _sel(ORDER, 60)
    assert len(kept) == 7


def test_the_kept_frames_really_are_far_enough_apart():
    """The guarantee the training set depends on."""
    kept, _ = _sel(ORDER, 60)
    gaps = [POS[b] - POS[a] for a, b in zip(kept, kept[1:])]
    assert min(gaps) >= 60


def test_the_best_frame_in_a_neighbourhood_is_the_one_kept():
    """THE POINT of separating at selection rather than at sampling. A stride
    of 60 takes frame 0 because it never looked at frame 31."""
    scores = {f: 0.5 for f in ORDER}
    scores["f031"] = 0.99
    kept, _ = _sel(ORDER, 60, scores)
    assert kept[0] == "f031"


def test_it_does_not_trade_a_whole_frame_for_a_hundredth_of_a_point():
    """Plain greedy-by-score took the highest frame first and let it push the
    next pick past the following window, turning four frames into three. On
    real data p10 to p90 spanned 0.83-0.91, so that trade buys almost nothing
    and costs an entire frame of ground."""
    scores = {f: 0.5 for f in ORDER}
    scores["f017"] = 0.9      # early, high, and greedy-first would start here
    kept, _ = _sel(ORDER[:200], 60, scores)
    assert len(kept) == 4, "windows keep the count; greedy-by-score gave 3"


def test_frames_closer_than_the_gap_collapse_to_one():
    kept, dropped = _sel(["f005", "f006", "f007", "f200"], 60)
    assert kept == ["f005", "f200"]
    assert set(dropped) == {"f006", "f007"}, "equal scores keep the earliest"


def test_a_gap_of_zero_keeps_everything():
    """The escape hatch has to actually be an escape hatch."""
    kept, dropped = _sel(ORDER, 0)
    assert len(kept) == len(ORDER) and dropped == []


def test_a_smaller_gap_keeps_more():
    assert len(_sel(ORDER, 30)[0]) > len(_sel(ORDER, 60)[0])


def test_kept_frames_come_back_in_capture_order():
    """They are written to a COCO and uploaded to CVAT; scrambled order makes
    a batch harder to sweep for what is missing."""
    kept, _ = _sel(ORDER, 60)
    assert kept == sorted(kept, key=lambda f: POS[f])


def test_nothing_chosen_is_not_an_error():
    assert _sel([], 60) == ([], [])


def test_a_frame_not_in_the_scored_order_is_ignored():
    """A frame that failed to load has no position, and guessing one would put
    it at an arbitrary distance from everything else."""
    kept, _ = _sel(["f010", "not_scored.png"], 60)
    assert kept == ["f010"]


def test_the_note_is_printed_even_when_nothing_was_dropped():
    """A filter that only speaks when unhappy leaves you unable to tell
    'nothing was redundant' from 'the filter did not run'."""
    note = pl.separation_note(6, 6, 60)
    assert "kept 6 of 6" in note and "0 dropped" in note


def test_the_note_says_when_separation_is_off():
    assert "OFF" in pl.separation_note(393, 393, 0)


def test_review_is_separated_too():
    """Each near-copy in review/ is a person correcting the same plant a second
    time - the most expensive thing in this loop."""
    import inspect
    from training.datasets import weeds_selftrain as st
    src = inspect.getsource(st._emit)
    assert src.count("select_spread") == 2
    assert 'buckets["review"], dropped_rev' in src
