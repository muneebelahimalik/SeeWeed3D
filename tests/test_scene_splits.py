"""Scene-stratified splits.

A split that is blind to what a session CONTAINS can put every mixed drive in
train. The model then looks excellent on a validation set that never once
exercises the crop-vs-weed decision - which is the only decision that decides
whether the laser fires on a weed or on the crop.
"""
import pytest

from conftest import load_script

sp = load_script("training/splits.py")


def _s(sid, scene, date="", field="", cam=""):
    return sp.SessionInfo(session_id=sid, scene=scene, date=date,
                          field_id=field, camera=cam, n_frames=100)


def _scenes(split_map, infos, split):
    by = {i.session_id: i.scene for i in infos}
    return sorted(by[s] for s in split_map[split])


# --------------------------------------------------------------------------- #
# The failure this prevents
# --------------------------------------------------------------------------- #
def test_every_scene_reaches_validation():
    """Three scenes, three sessions each. Unstratified, nothing stops val from
    being drawn entirely from one scene."""
    infos = ([_s(f"onion{i}_20260101_100000", "onion_only") for i in range(3)]
             + [_s(f"weeda{i}_20260102_100000", "weed_only") for i in range(3)]
             + [_s(f"mixed{i}_20260103_100000", "mixed") for i in range(3)])
    out = sp.assign_splits(infos, val_fraction=0.34, test_fraction=0.0,
                           seed=7)
    assert set(_scenes(out, infos, "val")) == {"onion_only", "weed_only",
                                               "mixed"}
    assert set(_scenes(out, infos, "train")) == {"onion_only", "weed_only",
                                                 "mixed"}


def test_an_unstratified_split_really_does_lose_a_scene():
    """The measurement behind the feature, not an assertion that two dicts
    differ. Over 40 seeds with three onion and three mixed sessions, the
    unstratified allocator hands validation a single scene on 10 of them -
    a 25% chance of a val set that cannot see the crop-vs-weed decision.
    Stratified, it is zero."""
    infos = ([_s(f"onion{i}_20260101_10000{i}", "onion_only") for i in range(3)]
             + [_s(f"mixed{i}_20260103_10000{i}", "mixed") for i in range(3)])
    by = {i.session_id: i.scene for i in infos}

    blind = strat = 0
    for seed in range(40):
        u = sp.assign_splits(infos, 0.34, 0.0, seed=seed,
                             stratify_by_scene=False)
        s = sp.assign_splits(infos, 0.34, 0.0, seed=seed,
                             stratify_by_scene=True)
        blind += len({by[x] for x in u["val"]}) < 2
        strat += len({by[x] for x in s["val"]}) < 2

    assert blind > 0, "the unstratified allocator was expected to lose a scene"
    assert strat == 0, f"stratified lost a scene on {strat} seed(s)"


def test_the_scene_report_names_what_is_missing():
    infos = [_s("onion1_20260101_100000", "onion_only"),
             _s("onion2_20260101_110000", "onion_only"),
             _s("mixed1_20260103_100000", "mixed")]
    split_map = {"train": ["onion1_20260101_100000", "mixed1_20260103_100000"],
                 "val": ["onion2_20260101_110000"], "test": []}
    rep = sp.scene_representation(split_map, infos)
    assert rep["counts"]["val"] == {"onion_only": 1}
    assert rep["missing"]["val"] == ["mixed"]
    assert rep["missing"]["test"] == []      # empty split reported elsewhere


# --------------------------------------------------------------------------- #
# The guarantees stratification must not break
# --------------------------------------------------------------------------- #
def test_holdouts_still_win():
    infos = [_s(f"m{i}_2026010{i}_100000", "mixed") for i in range(1, 5)]
    out = sp.assign_splits(infos, 0.25, 0.25, seed=3,
                           holdout_test=["m1_20260101_100000"])
    # In test, and in nothing else. The quota may add more sessions alongside
    # it; what a holdout guarantees is placement, not exclusivity.
    assert "m1_20260101_100000" in out["test"]
    assert "m1_20260101_100000" not in out["train"] + out["val"]


def test_it_is_still_deterministic():
    infos = ([_s(f"o{i}_2026010{i}_100000", "onion_only") for i in range(1, 4)]
             + [_s(f"w{i}_2026020{i}_100000", "weed_only") for i in range(1, 4)])
    a = sp.assign_splits(infos, 0.2, 0.2, seed=99)
    b = sp.assign_splits(infos, 0.2, 0.2, seed=99)
    assert a == b


def test_training_is_never_left_empty():
    """Each stratum waives the per-stratum train guarantee so a thin stratum
    can go wholly to val - so the OVERALL guarantee has to catch the case where
    every stratum did that."""
    # Fractions high enough that each one-session stratum fills its own test
    # quota, so every stratum walks away leaving train untouched.
    infos = [_s("o1_20260101_100000", "onion_only"),
             _s("w1_20260201_100000", "weed_only")]
    out = sp.assign_splits(infos, 0.3, 0.6, seed=1)
    assert out["train"], "training split is empty"
    assert sum(len(v) for v in out.values()) == 2


def test_no_session_lands_in_two_splits():
    infos = ([_s(f"o{i}_2026010{i}_100000", "onion_only") for i in range(1, 5)]
             + [_s(f"m{i}_2026030{i}_100000", "mixed") for i in range(1, 5)])
    out = sp.assign_splits(infos, 0.25, 0.25, seed=11)
    flat = sum(out.values(), [])
    assert len(flat) == len(set(flat)) == 8


def test_same_morning_sessions_are_never_separated():
    """Two drives of the same bed on the same morning are near-duplicates.
    Stratification must not become an excuse to split them apart."""
    infos = [_s("vid1_20260108_090000", "mixed", "2026-01-08", "field_A", "vid1"),
             _s("vid1_20260108_093000", "mixed", "2026-01-08", "field_A", "vid1"),
             _s("vid2_20260210_090000", "mixed", "2026-02-10", "field_B", "vid2"),
             _s("vid3_20260315_090000", "mixed", "2026-03-15", "field_C", "vid3")]
    out = sp.assign_splits(infos, 0.25, 0.25, seed=5)
    where = {s: k for k, v in out.items() for s in v}
    assert where["vid1_20260108_090000"] == where["vid1_20260108_093000"]


def test_unknown_scene_sessions_form_their_own_stratum():
    """They carry no evidence about what they contain, so they are allocated
    as before rather than being assumed to match anything."""
    infos = ([_s(f"o{i}_2026010{i}_100000", "onion_only") for i in range(1, 4)]
             + [_s(f"u{i}_2026040{i}_100000", "unknown") for i in range(1, 4)])
    out = sp.assign_splits(infos, 0.34, 0.0, seed=2)
    assert "unknown" in _scenes(out, infos, "val")


def test_a_group_spanning_scenes_is_labelled_mixed():
    """A group is indivisible, so whichever split receives it receives every
    scene in it - calling it onion_only would overstate what that split has."""
    by = {"a": "onion_only", "b": "weed_only"}
    assert sp.scene_of(["a", "b"], by) == "mixed"
    assert sp.scene_of(["a"], by) == "onion_only"
    assert sp.scene_of(["zzz"], by) == "unknown"


def test_string_sessions_still_work():
    """assign_splits accepts bare ids; they have no scene and must not crash
    the stratifier."""
    out = sp.assign_splits(["a", "b", "c", "d"], 0.25, 0.25, seed=4)
    assert sum(len(v) for v in out.values()) == 4


def test_existing_error_cases_are_unchanged():
    with pytest.raises(sp.SplitError):
        sp.assign_splits([])
    with pytest.raises(sp.SplitError):
        sp.assign_splits([_s("a_20260101_100000", "mixed")], 0.6, 0.6)
