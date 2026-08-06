"""Frame selection: keep only the frames a human actually verified.

The case this exists for: a CVAT task pre-loaded with SAM proposals has
annotations on EVERY frame. The ones never reached carry machine guesses with
the wrong classes. They are not empty, so keep_empty_frames does not touch
them, and training on them is worse than having no data at all - the mask
geometry is correct and only the label is wrong, so the loss is confident.
"""
import json

import pytest

from conftest import load_script

pd = load_script("training/prepare_dataset.py")


class Rec:
    def __init__(self, item_id, n=1):
        self.item_id = item_id
        self.instances = list(range(n))


def frames(n=60):
    return [Rec(f"frame_{i:04d}") for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# spec parsing
# --------------------------------------------------------------------------- #
def test_ranges_are_inclusive_on_both_ends():
    pos, _ = pd.parse_frame_spec("3-5")
    assert pos == {3, 4, 5}


def test_multiple_ranges_and_singles_combine():
    pos, _ = pd.parse_frame_spec("1-3,7,10-11")
    assert pos == {1, 2, 3, 7, 10, 11}


def test_non_numeric_tokens_are_patterns_not_positions():
    pos, pat = pd.parse_frame_spec("1-2,frame_0009,*_001*")
    assert pos == {1, 2}
    assert pat == ["frame_0009", "*_001*"]


def test_empty_spec_selects_nothing_specific():
    assert pd.parse_frame_spec(None) == (set(), [])
    assert pd.parse_frame_spec("") == (set(), [])


def test_reversed_range_is_rejected():
    with pytest.raises(SystemExit, match="reversed"):
        pd.parse_frame_spec("9-3")


def test_zero_is_rejected_because_positions_are_one_based():
    with pytest.raises(SystemExit, match="1-based"):
        pd.parse_frame_spec("0-3")


def test_spec_can_be_read_from_a_file(tmp_path):
    f = tmp_path / "keep.txt"
    f.write_text("1-3\n# a comment\n\n7   # trailing comment\n")
    pos, _ = pd.parse_frame_spec(f"@{f}")
    assert pos == {1, 2, 3, 7}


def test_missing_spec_file_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        pd.parse_frame_spec(f"@{tmp_path / 'nope.txt'}")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_positions_are_one_based_over_item_id_order():
    kept, dropped = pd.select_frames(frames(10), include="1")
    assert [r.item_id for r in kept] == ["frame_0001"]
    assert len(dropped) == 9


def test_the_users_actual_selection():
    """1-36 skipping 27, plus 50-59. The real case."""
    kept, dropped = pd.select_frames(frames(60), include="1-26,28-36,50-59")
    ids = [r.item_id for r in kept]
    assert len(ids) == 45
    assert "frame_0027" not in ids
    assert "frame_0026" in ids and "frame_0028" in ids
    assert "frame_0050" in ids and "frame_0059" in ids
    assert "frame_0037" not in ids and "frame_0060" not in ids
    assert len(dropped) == 15


def test_selection_is_order_independent_of_input_order():
    a, _ = pd.select_frames(frames(10), include="2-3")
    shuffled = list(reversed(frames(10)))
    b, _ = pd.select_frames(shuffled, include="2-3")
    assert [r.item_id for r in a] == [r.item_id for r in b]


def test_exclude_alone_keeps_everything_else():
    kept, _ = pd.select_frames(frames(10), exclude="5")
    assert len(kept) == 9
    assert "frame_0005" not in [r.item_id for r in kept]


def test_exclude_is_applied_after_include():
    kept, _ = pd.select_frames(frames(10), include="1-5", exclude="3")
    assert [r.item_id for r in kept] == [
        "frame_0001", "frame_0002", "frame_0004", "frame_0005"]


def test_glob_selects_by_item_id():
    kept, _ = pd.select_frames(frames(30), include="frame_001*")
    assert len(kept) == 10           # 0010..0019


def test_literal_item_id_selects_exactly_one():
    kept, _ = pd.select_frames(frames(30), include="frame_0017")
    assert [r.item_id for r in kept] == ["frame_0017"]


def test_out_of_range_position_fails_instead_of_selecting_nothing():
    """Silently ignoring 99 would train on the wrong 45 frames."""
    with pytest.raises(SystemExit, match="exceed"):
        pd.select_frames(frames(10), include="1-3,99")


def test_no_selection_keeps_everything():
    kept, dropped = pd.select_frames(frames(10))
    assert len(kept) == 10 and dropped == []


# --------------------------------------------------------------------------- #
# end to end through build()
# --------------------------------------------------------------------------- #
def _export(tmp_path, n_good=8, n_junk=3):
    """A Datumaro export where the first frames are hand-verified and the rest
    carry SAM guesses - all of them NON-empty."""
    items = []
    for i in range(1, n_good + n_junk + 1):
        good = i <= n_good
        items.append({
            "id": f"sess_a/frame_{i:04d}",
            "media": {"path": f"sess_a/frame_{i:04d}.png"},
            "image": {"size": [200, 200]},
            "annotations": [{
                "id": i, "type": "polygon", "label_id": 0 if good else 4,
                "group": i, "points": [10, 10, 60, 10, 60, 60, 10, 60],
                "attributes": {},
            }],
        })
    doc = {"info": {}, "categories": {"label": {"labels": [
        {"name": c} for c in pd.CLASSES]}}, "items": items}
    root = tmp_path / "export" / "annotations"
    root.mkdir(parents=True)
    (root / "default.json").write_text(json.dumps(doc))
    imgs = tmp_path / "sessions" / "sess_a"
    imgs.mkdir(parents=True)
    import numpy as np, cv2
    for i in range(1, n_good + n_junk + 1):
        cv2.imwrite(str(imgs / f"frame_{i:04d}.png"),
                    np.zeros((200, 200, 3), np.uint8))
    return tmp_path / "export", tmp_path / "sessions"


def test_build_keeps_only_the_selected_frames(tmp_path):
    exp, imgs = _export(tmp_path, n_good=8, n_junk=3)
    out = tmp_path / "ds"
    pd.build(exp, imgs, out, include_frames="1-8", val_fraction=0.0,
             test_fraction=0.0, strict=False)
    man = json.loads((out / "seg_manifest.json").read_text())
    assert len(man["frames"]) == 8
    ids = {f["item_id"] for f in man["frames"]}
    assert ids == {f"sess_a/frame_{i:04d}" for i in range(1, 9)}


def test_junk_frames_are_gone_not_merely_unsplit(tmp_path):
    """They must not reach the manifest at all - an 'unassigned' split would
    still be loadable by anything that reads frames without filtering."""
    exp, imgs = _export(tmp_path, n_good=8, n_junk=3)
    out = tmp_path / "ds"
    pd.build(exp, imgs, out, include_frames="1-8", val_fraction=0.0,
             test_fraction=0.0, strict=False)
    man = json.loads((out / "seg_manifest.json").read_text())
    text = json.dumps(man)
    for i in range(9, 12):
        assert f"frame_{i:04d}" not in text


def test_an_empty_selection_fails_rather_than_building_nothing(tmp_path):
    exp, imgs = _export(tmp_path)
    with pytest.raises(SystemExit, match="kept 0 frames"):
        pd.build(exp, imgs, tmp_path / "ds", include_frames="frame_nope",
                 strict=False)


# --------------------------------------------------------------------------- #
# gap buffers are spent only at real boundaries
# --------------------------------------------------------------------------- #
def test_no_gap_is_wasted_when_there_is_no_test_block():
    """With test_fraction=0 there is only ONE seam, so only one gap is due.
    Charging for the absent train|val|test seam would discard hand-corrected
    frames to separate a block from nothing."""
    from seeweed3d.training.splits import assign_frame_blocks
    ids = [f"f{i:03d}" for i in range(45)]
    r = assign_frame_blocks(ids, 0.2, 0.0, gap_frames=2)
    assert len(r["_dropped_gap"]) == 2
    assert len(r["train"]) == 34


def test_no_gap_at_all_when_everything_is_training():
    from seeweed3d.training.splits import assign_frame_blocks
    ids = [f"f{i:03d}" for i in range(20)]
    r = assign_frame_blocks(ids, 0.0, 0.0, gap_frames=2)
    assert r["_dropped_gap"] == []
    assert len(r["train"]) == 20


def test_both_real_boundaries_are_still_buffered():
    from seeweed3d.training.splits import assign_frame_blocks
    ids = [f"f{i:03d}" for i in range(45)]
    r = assign_frame_blocks(ids, 0.2, 0.2, gap_frames=2)
    assert len(r["_dropped_gap"]) == 4
    assert ids.index(r["val"][0]) - ids.index(r["train"][-1]) == 3
    assert ids.index(r["test"][0]) - ids.index(r["val"][-1]) == 3


def test_every_frame_is_accounted_for():
    from seeweed3d.training.splits import assign_frame_blocks
    ids = [f"f{i:03d}" for i in range(45)]
    for vf, tf in [(0.2, 0.2), (0.2, 0.0), (0.0, 0.0)]:
        r = assign_frame_blocks(ids, vf, tf, gap_frames=2)
        assert sum(len(v) for v in r.values()) == len(ids)
