"""Ranking mixed frames by onion/weed contact - where the dangerous decision is
actually exercised, and where annotation time is therefore worth spending."""
import json

import pytest

from conftest import load_script

rk = load_script("annotation/rank_by_contact.py")

from common.ontology import CROP_CLASS  # noqa: E402

H = W = 60


def _sq(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


def _coco(tmp_path, frames, name="pred"):
    cats = [{"id": 1, "name": CROP_CLASS}, {"id": 2, "name": "other_weed"}]
    images, anns, aid = [], [], 1
    for k, (fn, insts) in enumerate(frames, start=1):
        images.append({"id": k, "file_name": fn, "height": H, "width": W})
        for c, p in insts:
            anns.append({"id": aid, "image_id": k,
                         "category_id": 1 if c == CROP_CLASS else 2,
                         "segmentation": [p], "bbox": [0, 0, 1, 1],
                         "area": 1.0, "iscrowd": 0})
            aid += 1
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "instances_default.json").write_text(
        json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return d


# --------------------------------------------------------------------------- #
# The signal
# --------------------------------------------------------------------------- #
def test_touching_plants_score_above_separated_ones():
    touching = rk.contact_score(
        *_masks([(CROP_CLASS, _sq(5, 5, 20)), ("other_weed", _sq(25, 5, 20))]))
    apart = rk.contact_score(
        *_masks([(CROP_CLASS, _sq(0, 0, 12)), ("other_weed", _sq(45, 45, 12))]))
    assert touching["contact_px"] > apart["contact_px"]


def _masks(insts):
    """(masks, classes) from (class, polygon) pairs."""
    import cv2
    import numpy as np
    masks, classes = [], []
    for c, p in insts:
        m = np.zeros((H, W), np.uint8)
        a = np.asarray(p, float).reshape(-1, 2)
        cv2.fillPoly(m, [a.round().astype(np.int32)], 1)
        masks.append(m.astype(bool))
        classes.append(c)
    return masks, classes


def test_a_single_class_frame_has_no_contact_and_says_why():
    """Zero is a real answer here, not a missing measurement."""
    r = rk.contact_score(*_masks([(CROP_CLASS, _sq(5, 5, 20)),
                                  (CROP_CLASS, _sq(30, 5, 20))]))
    assert r["contact_px"] == 0
    assert "only one class" in r["reason"]


def test_an_empty_frame_scores_zero_rather_than_raising():
    assert rk.contact_score([], [])["contact_px"] == 0


# --------------------------------------------------------------------------- #
# The ranking
# --------------------------------------------------------------------------- #
def test_frames_come_back_most_contact_first(tmp_path):
    d = _coco(tmp_path, [
        ("s_000001.png", [(CROP_CLASS, _sq(0, 0, 10)),
                          ("other_weed", _sq(45, 45, 10))]),      # apart
        ("s_000002.png", [(CROP_CLASS, _sq(5, 5, 25)),
                          ("other_weed", _sq(30, 5, 25))]),       # touching
    ])
    kept, _ = rk.rank(rk.load_side(d), per_session=0)
    assert kept[0]["frame"] == "s_000002.png"


def test_frames_with_no_contact_are_not_offered_at_all(tmp_path):
    """They exercise nothing; a human confirming them spends time on what was
    never in doubt."""
    d = _coco(tmp_path, [
        ("s_000001.png", [(CROP_CLASS, _sq(5, 5, 20))]),          # onion only
        ("s_000002.png", [(CROP_CLASS, _sq(5, 5, 25)),
                          ("other_weed", _sq(30, 5, 25))]),
    ])
    kept, _ = rk.rank(rk.load_side(d), per_session=0)
    assert [r["frame"] for r in kept] == ["s_000002.png"]


def test_one_session_cannot_fill_the_whole_list(tmp_path):
    """The classic single-signal trap: forty near-identical frames of the same
    two plants is not forty frames of annotation value."""
    frames = [(f"busy_{i:06d}.png", [(CROP_CLASS, _sq(5, 5, 25)),
                                     ("other_weed", _sq(30, 5, 25))])
              for i in range(10)]
    frames += [("other_000001.png", [(CROP_CLASS, _sq(5, 5, 20)),
                                     ("other_weed", _sq(24, 5, 20))])]
    d = _coco(tmp_path, frames)
    kept, _ = rk.rank(rk.load_side(d), per_session=3)
    from collections import Counter
    per = Counter(r["session_id"] for r in kept)
    assert per["busy"] == 3
    assert "other" in per, "the cap must leave room for another session"


def test_the_cap_can_be_disabled_deliberately(tmp_path):
    frames = [(f"busy_{i:06d}.png", [(CROP_CLASS, _sq(5, 5, 25)),
                                     ("other_weed", _sq(30, 5, 25))])
              for i in range(5)]
    d = _coco(tmp_path, frames)
    kept, _ = rk.rank(rk.load_side(d), per_session=0)
    assert len(kept) == 5


def test_top_limits_the_result(tmp_path):
    frames = [(f"s_{i:06d}.png", [(CROP_CLASS, _sq(5, 5, 25)),
                                  ("other_weed", _sq(30, 5, 25))])
              for i in range(6)]
    d = _coco(tmp_path, frames)
    kept, _ = rk.rank(rk.load_side(d), per_session=0, top=2)
    assert len(kept) == 2


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #
def test_the_frame_list_is_written_one_per_line(tmp_path):
    d = _coco(tmp_path, [("s_000001.png", [(CROP_CLASS, _sq(5, 5, 25)),
                                           ("other_weed", _sq(30, 5, 25))])])
    out = tmp_path / "next.txt"
    rk.main(["--pred", str(d), "--out", str(out)])
    assert out.read_text().split() == ["s_000001.png"]


def test_no_contact_anywhere_warns_that_a_class_may_be_missing(tmp_path,
                                                              capsys):
    """A prelabeler emitting only one class produces an empty ranking, and the
    empty ranking must not read as 'nothing needs annotating'."""
    d = _coco(tmp_path, [("s_000001.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    rk.main(["--pred", str(d)])
    out = capsys.readouterr().out
    assert "not emitting one of the classes" in out
