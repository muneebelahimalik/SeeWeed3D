"""Class balance across a frame-block split.

THE BUG THIS EXISTS FOR
-----------------------
Round 0 of the weed loop split 60 frames into train=42 / val=10 and reported:

    cutleaf_evening_primrose   642 train    73 val     AP 0.557
    grass_weed                 327 train    26 val     AP 0.458
    other_weed                 145 train    74 val     AP 0.214

Val is 19% of the frames, but held 34% of every `other_weed` instance in the
dataset. The class was simultaneously starved in training and over-weighted in
the score, and the mean AP carried the difference - none of which is visible in
a per-class AP table, because the table reports the score and not the split
that produced it.

The cause was that the block layout is rotated by a hash of the session key.
With one block there are exactly three layouts and nothing was choosing the
balanced one.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.splits as sp                            # noqa: E402


def frames(n, session="vid2_20260108_122731"):
    return [f"{session}_{i:06d}" for i in range(n)]


def clustered(ids, cls="other_weed", lo=40, hi=56, dense=14, sparse=1):
    """A class confined to one stretch of the drive, as `other_weed` is."""
    out = {}
    for i, f in enumerate(ids):
        c = Counter({"cutleaf_evening_primrose": 12, "grass_weed": 6})
        c[cls] = dense if lo <= i < hi else sparse
        out[f] = c
    return out


def val_share(out, counts, cls):
    bal = sp.class_balance({k: out[k] for k in sp.SPLITS}, counts)
    return bal["classes"][cls]["val_share"], bal["frame_share"]["val"]


# --------------------------------------------------------------------------- #
# The fix
# --------------------------------------------------------------------------- #
def test_class_counts_pull_a_clustered_class_back_toward_its_frame_share():
    """The whole point: knowing what is IN the frames beats hashing the key."""
    ids = frames(60)
    counts = clustered(ids)
    kw = dict(val_fraction=0.20, test_fraction=0.0, gap_frames=8, n_blocks=1,
              seed=1234)

    blind = sp.assign_frame_blocks(ids, **kw)
    aware = sp.assign_frame_blocks(ids, class_counts_by_frame=counts, **kw)

    got_blind, want = val_share(blind, counts, "other_weed")
    got_aware, _ = val_share(aware, counts, "other_weed")
    assert abs(got_aware - want) < abs(got_blind - want), (
        f"balanced layout is no closer to the frame share: "
        f"{got_aware:.0%} vs {got_blind:.0%}, want {want:.0%}")


def test_balancing_changes_only_which_frames_not_how_many():
    """It picks among the SAME three layouts, so the sizes cannot move. A
    'balanced' split that quietly shrank training would be buying the metric
    with data."""
    ids = frames(60)
    counts = clustered(ids)
    kw = dict(val_fraction=0.20, test_fraction=0.15, gap_frames=6, n_blocks=1,
              seed=99)
    blind = sp.assign_frame_blocks(ids, **kw)
    aware = sp.assign_frame_blocks(ids, class_counts_by_frame=counts, **kw)
    for split in list(sp.SPLITS) + ["_dropped_gap"]:
        assert len(blind[split]) == len(aware[split]), split


def test_more_blocks_balance_a_clustered_class_better_than_one():
    """The advice the build prints when it cannot fix the skew itself. A
    recommendation that does not hold is worse than none."""
    ids = frames(60)
    counts = clustered(ids)
    one = sp.assign_frame_blocks(ids, 0.2, 0.0, gap_frames=4, n_blocks=1,
                                 seed=1234, class_counts_by_frame=counts)
    many = sp.assign_frame_blocks(ids, 0.2, 0.0, gap_frames=4, n_blocks=3,
                                  seed=1234, class_counts_by_frame=counts)
    got_one, want = val_share(one, counts, "other_weed")
    got_many, want_many = val_share(many, counts, "other_weed")
    assert abs(got_many - want_many) < abs(got_one - want)


def test_a_uniform_class_is_balanced_either_way():
    """Nothing to fix means nothing changed - the mechanism must not invent
    movement where the drive was already even."""
    ids = frames(60)
    counts = {f: Counter({"grass_weed": 5}) for f in ids}
    out = sp.assign_frame_blocks(ids, 0.2, 0.0, gap_frames=8, n_blocks=1,
                                 seed=1234, class_counts_by_frame=counts)
    got, want = val_share(out, counts, "grass_weed")
    assert got == pytest.approx(want, abs=0.02)


# --------------------------------------------------------------------------- #
# Guarantees the fix must not break
# --------------------------------------------------------------------------- #
def test_no_frame_is_shared_between_splits():
    ids = frames(80)
    counts = clustered(ids, lo=10, hi=30)
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=5, n_blocks=2,
                                 seed=7, class_counts_by_frame=counts)
    seen = Counter(sum((out[k] for k in list(sp.SPLITS) + ["_dropped_gap"]), []))
    assert not [f for f, n in seen.items() if n > 1]
    assert set(seen) == set(ids), "every frame must be accounted for"


def test_each_split_is_still_contiguous():
    """Balance must not become a frame shuffle - that would put a frame and its
    near-duplicate on opposite sides of the boundary, which is the failure
    blocks exist to prevent."""
    ids = frames(60)
    counts = clustered(ids)
    out = sp.assign_frame_blocks(ids, 0.2, 0.0, gap_frames=8, n_blocks=1,
                                 seed=1234, class_counts_by_frame=counts)
    for split in sp.SPLITS:
        pos = sorted(ids.index(f) for f in out[split])
        if pos:
            assert pos == list(range(pos[0], pos[0] + len(pos))), split


def test_it_is_deterministic():
    ids = frames(60)
    counts = clustered(ids)
    kw = dict(val_fraction=0.2, test_fraction=0.1, gap_frames=4, n_blocks=2,
              seed=1234, class_counts_by_frame=counts)
    assert sp.assign_frame_blocks(ids, **kw) == sp.assign_frame_blocks(ids, **kw)


def test_ties_keep_the_layout_the_seed_would_have_picked():
    """A build where balance cannot distinguish the layouts must split exactly
    as it did before, so this change does not silently re-draw every dataset
    that had nothing wrong with it."""
    ids = frames(60)
    # Every frame identical: all three rotations score the same.
    counts = {f: Counter({"grass_weed": 3}) for f in ids}
    kw = dict(val_fraction=0.2, test_fraction=0.2, gap_frames=4, n_blocks=1,
              seed=1234)
    assert (sp.assign_frame_blocks(ids, class_counts_by_frame=counts, **kw)
            == sp.assign_frame_blocks(ids, **kw))


def test_frames_with_no_counts_do_not_crash_it():
    """A frame missing from the counts map is empty, not an error - the build
    drops empty frames elsewhere and the two paths must not disagree."""
    ids = frames(60)
    counts = clustered(ids)
    for f in ids[:5]:
        del counts[f]
    out = sp.assign_frame_blocks(ids, 0.2, 0.0, gap_frames=4, n_blocks=1,
                                 seed=1234, class_counts_by_frame=counts)
    assert len(out["train"]) + len(out["val"]) + len(out["_dropped_gap"]) == 60


def test_rotate_false_still_wins():
    """An explicit 'do not rotate' is an instruction, not a preference."""
    ids = frames(60)
    counts = clustered(ids)
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=1,
                                 seed=1234, rotate=False,
                                 class_counts_by_frame=counts)
    assert out["train"][0] == ids[0], "train must still lead the block"


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def test_class_balance_reports_shares_against_the_frame_share():
    split_frames = {"train": ["a", "b", "c", "d"], "val": ["e"], "test": []}
    counts = {"a": Counter({"w": 10}), "b": Counter({"w": 10}),
              "c": Counter({"w": 10}), "d": Counter({"w": 10}),
              "e": Counter({"w": 10})}
    bal = sp.class_balance(split_frames, counts)
    assert bal["classes"]["w"]["total"] == 50
    assert bal["classes"]["w"]["val_share"] == pytest.approx(0.2)
    assert bal["frame_share"]["val"] == pytest.approx(0.2)
    assert not sp.class_balance_problems(bal)


def test_the_real_round_zero_skew_is_flagged():
    """The numbers that motivated all of this: 34% of a class in a 19% split."""
    ids = frames(52)
    counts = {}
    for i, f in enumerate(ids):
        # 42 train frames then 10 val frames, other_weed piled into the val end.
        counts[f] = Counter({"cutleaf_evening_primrose": 15,
                             "other_weed": 7 if i >= 42 else 3})
    bal = sp.class_balance({"train": ids[:42], "val": ids[42:], "test": []},
                           counts)
    assert bal["classes"]["other_weed"]["val_share"] > 0.33
    bad = sp.class_balance_problems(bal)
    assert bad, ("the skew that motivated this whole change must be caught by "
                 "the default tolerance - it is a ratio of 1.79, so a "
                 "tolerance of 2.0 would say nothing about it")
    # Skew is zero-sum: piling one class into val necessarily thins the others,
    # so several rows can be flagged. The one DRIVING it must lead.
    split, cls, got, want, n = bad[0]
    assert (split, cls) == ("val", "other_weed")
    assert got > want


def test_a_rare_class_is_not_flagged_on_noise():
    """One frame either way swings a tiny class's share completely. Flagging it
    would report the split's arithmetic, not a problem with it."""
    ids = frames(52)
    counts = {f: Counter({"grass_weed": 20,
                          "weed_cluster": 1 if i >= 42 else 0})
              for i, f in enumerate(ids)}
    bal = sp.class_balance({"train": ids[:42], "val": ids[42:], "test": []},
                           counts)
    assert bal["classes"]["weed_cluster"]["val_share"] == 1.0
    assert bal["classes"]["weed_cluster"]["total"] < sp.MIN_INSTANCES_FOR_BALANCE
    assert not sp.class_balance_problems(bal)


def test_absence_is_flagged_as_loudly_as_concentration():
    """A class almost missing from val is measured on nothing, which is just as
    misleading as one that dominates it - and far easier to overlook, because
    it makes the mean AP look better rather than worse."""
    ids = frames(52)
    counts = {f: Counter({"grass_weed": 20,
                          "other_weed": 0 if i >= 42 else 5})
              for i, f in enumerate(ids)}
    bal = sp.class_balance({"train": ids[:42], "val": ids[42:], "test": []},
                           counts)
    bad = sp.class_balance_problems(bal)
    assert [row[1] for row in bad] == ["other_weed"]
    assert bad[0][2] == 0.0


def test_problems_are_ordered_worst_first():
    ids = frames(50)
    counts = {}
    for i, f in enumerate(ids):
        counts[f] = Counter({"mild": 6 if i >= 40 else 4,
                             "severe": 40 if i >= 40 else 1})
    bal = sp.class_balance({"train": ids[:40], "val": ids[40:], "test": []},
                           counts)
    bad = sp.class_balance_problems(bal)
    assert bad and bad[0][1] == "severe"


def test_an_empty_split_is_not_reported_as_imbalanced():
    """TEST_FRACTION is 0 in the weed build. A split that does not exist cannot
    be badly balanced, and saying so every run trains people to ignore it."""
    ids = frames(50)
    counts = {f: Counter({"w": 5}) for f in ids}
    bal = sp.class_balance({"train": ids[:40], "val": ids[40:], "test": []},
                           counts)
    assert bal["frame_share"]["test"] == 0.0
    assert not [r for r in sp.class_balance_problems(bal) if r[0] == "test"]


def test_class_balance_survives_an_entirely_empty_dataset():
    bal = sp.class_balance({"train": [], "val": [], "test": []}, {})
    assert bal["classes"] == {}
    assert not sp.class_balance_problems(bal)


# --------------------------------------------------------------------------- #
# The two directions are not equally bad
# --------------------------------------------------------------------------- #
def test_over_representation_is_penalised_harder_than_under():
    """Concentrated in val costs training data AND weights the score; thin in
    val costs only measurement quality. Scoring them equally would let the
    layout trade the expensive harm for the cheap one at par."""
    ids = frames(50)
    counts = {f: Counter({"w": 10}) for f in ids}
    split = {"train": ids[:40], "val": ids[40:], "test": []}

    over = dict(counts)
    for i, f in enumerate(ids):
        over[f] = Counter({"w": 30 if i >= 40 else 5})
    under = dict(counts)
    for i, f in enumerate(ids):
        under[f] = Counter({"w": 1 if i >= 40 else 25})

    # Both drift from the 20% frame share, in opposite directions.
    over_share = sp.class_balance(split, over)["classes"]["w"]["val_share"]
    under_share = sp.class_balance(split, under)["classes"]["w"]["val_share"]
    assert over_share > 0.2 > under_share

    a = sp._worst_class_skew(split, over)
    b = sp._worst_class_skew(split, under)
    assert a > b, ("an over-represented class must score worse than an "
                   f"equally-drifted thin one: {a:.3f} vs {b:.3f}")


def test_the_layout_prefers_thin_to_concentrated_when_it_must_choose():
    """The real outcome on this session: other_weed moved from 34% of val
    (145 train instances) to 8% (199). A worse estimate of a better-trained
    class is the right trade, and the objective has to actually make it."""
    ids = frames(60)
    counts = clustered(ids)
    out = sp.assign_frame_blocks(ids, 0.2, 0.0, gap_frames=8, n_blocks=1,
                                 seed=1234, class_counts_by_frame=counts)
    got, want = val_share(out, counts, "other_weed")
    assert got < want, "the concentrated layout was available and was rejected"
    bal = sp.class_balance({k: out[k] for k in sp.SPLITS}, counts)
    assert bal["classes"]["other_weed"]["train"] > (
        bal["classes"]["other_weed"]["val"])
