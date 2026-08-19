"""A test set drawn from every session, when no session can be spared.

Strictly weaker than a held-out session and the build says so. What it must do
correctly is be REPRESENTATIVE - every session, every scene, and sampled from
along the whole drive rather than only its end - and be honest about how far
the test frames really are from training frames.
"""
import pytest

from conftest import load_script

sp = load_script("training/splits.py")


def _ids(session, indices):
    return [f"{session}_{i:06d}" for i in indices]


def _positions(chosen, all_ids):
    """Where the chosen frames sit, as fractions along the sequence."""
    idx = {f: i for i, f in enumerate(all_ids)}
    return [idx[f] / max(1, len(all_ids) - 1) for f in chosen]


# --------------------------------------------------------------------------- #
# Representative: sampled from along the drive, not only its tail
# --------------------------------------------------------------------------- #
def test_without_rotation_test_is_only_ever_the_end_of_the_drive():
    """The behaviour rotation fixes. A fixed layout puts test at the tail of
    every block of every session, so the test set is made entirely of
    drive-ends - later in the pass, further along the bed, often the headland
    where the rig turns. A test set of drive-ends measures drive-ends."""
    ids = _ids("vid1_20260108_101500", range(0, 2000, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=1,
                                 rotate=False)
    assert min(_positions(out["test"], ids)) > 0.7


def test_rotation_moves_the_test_block_off_the_tail():
    """Across sessions the test block must not always land in the same place.
    One session proves nothing - the rotation is keyed, so this asks whether
    the BIAS is gone across many of them."""
    where = []
    for k in range(24):
        ids = _ids(f"vid{k}_20260108_101500", range(0, 800, 5))
        out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=1,
                                     _key=f"sess{k}")
        where.append(min(_positions(out["test"], ids)))
    assert min(where) < 0.3, "test never starts near the beginning"
    assert max(where) > 0.6, "test never lands late either"
    assert len(set(round(w, 1) for w in where)) >= 2


def test_rotation_is_deterministic_for_a_seed():
    ids = _ids("vid1_20260108_101500", range(0, 800, 5))
    a = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, _key="s")
    b = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, _key="s")
    assert a["test"] == b["test"] and a["train"] == b["train"]


def test_rotation_keeps_every_block_contiguous():
    """Rotating the ORDER must not become a random frame shuffle - that would
    put a frame and its near-duplicate on opposite sides of the boundary, the
    exact failure blocks exist to prevent."""
    ids = _ids("vid1_20260108_101500", range(0, 800, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, _key="s")
    for split in ("train", "val", "test"):
        idx = sorted(ids.index(f) for f in out[split])
        assert idx == list(range(idx[0], idx[0] + len(idx))), \
            f"{split} is not one contiguous run"


def test_rotation_shares_nothing_between_splits():
    ids = _ids("vid1_20260108_101500", range(0, 800, 5))
    for k in range(6):
        out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, _key=f"s{k}")
        seen = out["train"] + out["val"] + out["test"]
        assert len(seen) == len(set(seen))


def test_two_sessions_do_not_rotate_identically():
    """Keyed by session, so the test set is not the same stretch of each."""
    ids = _ids("vid1_20260108_101500", range(0, 800, 5))
    got = {tuple(sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4,
                                        _key=f"sess_{k}")["test"])
           for k in range(12)}
    assert len(got) >= 2


def test_several_blocks_spread_the_test_set_along_the_drive():
    ids = _ids("vid1_20260108_101500", range(0, 2000, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=4)
    pos = _positions(out["test"], ids)
    assert min(pos) < 0.3 and max(pos) > 0.7      # both ends represented
    # and something from the middle too
    assert any(0.3 <= p <= 0.7 for p in pos)


def test_every_session_contributes_to_every_split():
    """The whole point when no session can be held out."""
    by_session = {f"vid{i}_2026010{i}_100000": _ids(f"vid{i}_2026010{i}_100000",
                                                    range(0, 500, 5))
                  for i in (1, 2, 3)}
    out = sp.assign_frame_blocks_per_session(by_session, 0.2, 0.2,
                                             gap_frames=4, n_blocks=4)
    for split in ("train", "val", "test"):
        sessions = {f.rsplit("_", 1)[0] for f in out[split]}
        assert sessions == set(by_session), f"{split} is missing a session"


def test_blocks_stay_contiguous():
    """Contiguity is what stops a frame sitting next to its own near-duplicate
    across the split boundary. More blocks must not become a random split."""
    ids = _ids("vid1_20260108_101500", range(0, 1000, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=4)
    idx = {f: i for i, f in enumerate(ids)}
    runs = []
    for f in sorted(out["test"], key=lambda f: idx[f]):
        if runs and idx[f] == runs[-1][-1] + 1:
            runs[-1].append(idx[f])
        else:
            runs.append([idx[f]])
    assert len(runs) <= 4                  # one contiguous run per block
    assert all(len(r) > 1 for r in runs)   # and each is a genuine block


def test_nothing_is_shared_between_splits():
    ids = _ids("vid1_20260108_101500", range(0, 1000, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=4)
    tr, va, te = (set(out[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_the_block_count_is_a_ceiling_not_a_demand():
    """A stretch shorter than the block floor goes wholly to train, so asking
    for more blocks than a short session can hold would quietly send EVERY
    frame to train and leave val empty - training then runs blind."""
    ids = _ids("vid1_20260108_101500", range(10))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=2, n_blocks=8)
    assert out["val"], "val is empty - the block count was not clamped"


def test_one_block_is_unchanged_from_before():
    ids = _ids("vid1_20260108_101500", range(0, 500, 5))
    a = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4)
    b = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=1)
    assert a == b


# --------------------------------------------------------------------------- #
# Honest about how far apart the splits really are
# --------------------------------------------------------------------------- #
def test_separation_is_measured_in_video_frames_not_pool_frames():
    """A gap is counted in POOL frames and the pool is usually strided, so the
    same GAP_FRAMES can mean 2 video frames or 50. That difference decides
    whether the split measures anything and is invisible in the config."""
    sess = "vid1_20260108_101500"
    got = sp.seam_separation({
        "train": _ids(sess, [0, 5, 10, 15]),
        "val": [],
        "test": _ids(sess, [65, 70]),          # 50 video frames past the last
    })
    assert got["test"][sess] == 50


def test_a_test_frame_adjacent_to_a_training_frame_is_flagged():
    sess = "vid1_20260108_101500"
    got = sp.seam_separation({"train": _ids(sess, [0, 5, 10]),
                              "val": [], "test": _ids(sess, [12])})
    assert sp.seam_separation_problems(got) == [("test", sess, 2)]


def test_a_genuinely_separated_split_is_not_flagged():
    sess = "vid1_20260108_101500"
    got = sp.seam_separation({"train": _ids(sess, range(0, 500, 5)),
                              "val": [], "test": _ids(sess, [900, 905])})
    assert sp.seam_separation_problems(got) == []


def test_separation_is_per_session():
    a, b = "vid1_20260108_101500", "vid2_20260109_101500"
    got = sp.seam_separation({
        "train": _ids(a, [0, 5]) + _ids(b, range(0, 500, 5)),
        "val": [], "test": _ids(a, [8]) + _ids(b, [900])})
    assert got["test"][a] == 3 and got["test"][b] == 405


def test_a_session_absent_from_training_is_not_reported():
    """A held-out session has no training frames to be near, so there is no
    distance to report - and reporting zero would read as contamination."""
    a, b = "vid1_20260108_101500", "vid9_20260301_090000"
    got = sp.seam_separation({"train": _ids(a, [0, 5]), "val": [],
                              "test": _ids(b, [100, 105])})
    assert b not in got["test"]


def test_val_is_measured_too():
    sess = "vid1_20260108_101500"
    got = sp.seam_separation({"train": _ids(sess, [0, 5, 10]),
                              "val": _ids(sess, [11]), "test": []})
    assert got["val"][sess] == 1


def test_unparseable_ids_are_skipped_rather_than_crashing():
    assert sp.seam_separation({"train": ["nonsense"], "val": [], "test": []}) \
        == {"val": {}, "test": {}}


def test_a_real_block_split_measures_its_own_separation():
    """End to end: the numbers the build prints come from the split it made."""
    sess = "vid1_20260108_101500"
    ids = _ids(sess, range(0, 2000, 5))       # stride 5 in the pool
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=8, n_blocks=4)
    got = sp.seam_separation({k: out[k] for k in ("train", "val", "test")})
    # gap 8 pool frames x stride 5 = 40 video frames at every seam, including
    # the seam between one chunk's test tail and the next chunk's train head.
    assert got["test"][sess] >= 40


# --------------------------------------------------------------------------- #
# Choosing it deliberately
# --------------------------------------------------------------------------- #
def test_make_dataset_exposes_the_mode():
    """The knobs exist and hold usable values. Not the shipped literals - this
    CONFIG block is meant to be edited, and choosing "frame_block" for a real
    build must not fail the suite."""
    md = load_script("training/make_dataset.py")
    assert md.CONFIG["SPLIT_MODE"] in ("auto", "session", "frame_block")
    assert md.CONFIG["BLOCKS_PER_SESSION"] >= 1


def test_the_gap_default_is_not_two_pool_frames():
    """Two pool frames is a third of a second at 30 fps and a stride of 5 -
    the same photograph."""
    md = load_script("training/make_dataset.py")
    assert md.CONFIG["GAP_FRAMES"] >= 8, (
        "raising the gap is expected; lowering it below the shipped default "
        "silently reintroduces near-duplicate frames across the split")


def test_an_unknown_mode_is_refused(tmp_path):
    pdz = load_script("training/prepare_dataset.py")
    with pytest.raises(SystemExit, match="split_mode must be"):
        pdz.build(tmp_path, tmp_path, tmp_path / "out", split_mode="random")


# --------------------------------------------------------------------------- #
# What the gaps cost, and not paying for the ones that buy nothing
# --------------------------------------------------------------------------- #
def test_a_seam_between_two_chunks_of_the_same_split_costs_nothing():
    """With the layout rotated, two chunks often meet at the SAME split.
    Dropping frames to separate train from train buys no separation at all, and
    at several blocks it was throwing away a third of every session."""
    ids = _ids("vid1_20260108_101500", range(0, 3000, 5))
    charged = sp.assign_frame_blocks(ids, 0.15, 0.15, gap_frames=12,
                                     n_blocks=6, _key="s", rotate=True)
    # Every real boundary is still buffered: nothing is shared, and every
    # split remains a set of contiguous runs.
    seen = charged["train"] + charged["val"] + charged["test"]
    assert len(seen) == len(set(seen))
    # And the saving is real - some seam somewhere matched and was not charged.
    worst = 12 * (6 - 1) + 12 * 2 * 6
    assert len(charged["_dropped_gap"]) < worst


def test_every_session_still_reaches_every_split_after_rotation():
    """The property frame blocks exist for: no class can be missing from
    training, because every session contributes to every split."""
    by = {f"vid{k}_20260108_10150{k}": _ids(f"vid{k}_20260108_10150{k}",
                                            range(0, 1500, 5))
          for k in range(4)}
    out = sp.assign_frame_blocks_per_session(by, 0.15, 0.15, gap_frames=12,
                                             n_blocks=3, seed=1234)
    for sess in by:
        for split in ("train", "val", "test"):
            assert any(f.startswith(sess) for f in out[split]), \
                f"{sess} missing from {split}"


def test_the_gap_is_still_charged_where_the_split_really_changes():
    """The optimisation must not become 'skip the buffer'. With rotation off,
    every chunk ends with test and begins with train, so every seam is real."""
    ids = _ids("vid1_20260108_101500", range(0, 3000, 5))
    out = sp.assign_frame_blocks(ids, 0.15, 0.15, gap_frames=12, n_blocks=4,
                                 rotate=False)
    assert len(out["_dropped_gap"]) >= 12 * 3, "chunk seams went unbuffered"


def test_the_result_holds_frames_and_nothing_else():
    """Callers sum the returned dict to account for every frame. A layout
    description living in there as an extra key is silently counted as three
    frames that do not exist."""
    ids = _ids("vid1_20260108_101500", range(0, 800, 5))
    for n_blocks in (1, 3):
        out = sp.assign_frame_blocks(ids, 0.15, 0.15, gap_frames=12,
                                     n_blocks=n_blocks, _key="s")
        assert sum(len(v) for v in out.values()) == len(ids)
        for v in out.values():
            assert all(isinstance(f, str) and f in ids for f in v)


def test_the_fractions_describe_what_is_built_not_what_was_asked_for():
    """Sizing val and test from the RAW block and letting train absorb every
    buffered frame turned a requested 70/15/15 into 56/22/22 on real data - the
    shortfall landing entirely on the split that needed the frames most."""
    ids = _ids("vid1_20260108_101500", range(0, 246 * 5, 5))
    out = sp.assign_frame_blocks(ids, 0.15, 0.15, gap_frames=12, n_blocks=3,
                                 _key="s")
    kept = len(out["train"]) + len(out["val"]) + len(out["test"])
    assert abs(len(out["val"]) / kept - 0.15) < 0.03
    assert abs(len(out["test"]) / kept - 0.15) < 0.03
    assert abs(len(out["train"]) / kept - 0.70) < 0.04


def test_the_ratios_hold_across_block_counts_and_gaps():
    """The gap and the block count change how much is buffered; neither should
    change the SHAPE of what survives."""
    ids = _ids("vid1_20260108_101500", range(0, 300 * 5, 5))
    for n_blocks in (1, 2, 4):
        for gap in (0, 6, 12):
            out = sp.assign_frame_blocks(ids, 0.2, 0.1, gap_frames=gap,
                                         n_blocks=n_blocks, _key="s")
            kept = sum(len(out[k]) for k in ("train", "val", "test"))
            assert abs(len(out["val"]) / kept - 0.2) < 0.05, (n_blocks, gap)
            assert abs(len(out["test"]) / kept - 0.1) < 0.05, (n_blocks, gap)


def test_a_block_too_small_for_its_buffers_still_produces_a_split():
    """Short sessions must not raise: the gap is dropped rather than the
    build."""
    ids = _ids("vid1_20260108_101500", range(0, 30 * 5, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=12, n_blocks=1,
                                 _key="s")
    assert out["train"] and out["val"] and out["test"]
    assert sum(len(v) for v in out.values()) == len(ids)
