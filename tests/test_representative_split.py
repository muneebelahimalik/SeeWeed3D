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
def test_one_block_puts_test_only_at_the_end():
    """The behaviour being fixed. With a single block, test is always the LAST
    stretch of the recording - which on this data is later in the day, further
    along the bed, often the headland where the rig turns."""
    ids = _ids("vid1_20260108_101500", range(0, 2000, 5))
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=4, n_blocks=1)
    assert min(_positions(out["test"], ids)) > 0.7


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
