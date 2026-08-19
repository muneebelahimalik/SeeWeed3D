"""The mixed-scene benchmark runner: loading either input form, pairing frames,
and pooling in the way that keeps the crop number honest."""
import json

import numpy as np
import pytest

from conftest import load_script

bm = load_script("evaluation/bench_mixed.py")

from common.ontology import CROP_CLASS  # noqa: E402

H = W = 60


def _sq(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


def _manifest(tmp_path, name, frames):
    """frames: [(file_name, [(class_name, poly), ...]), ...]"""
    doc = {"classes": [CROP_CLASS, "other_weed"], "frames": [
        {"item_id": fn.split(".")[0], "image_path": fn,
         "width": W, "height": H, "split": "test",
         "instances": [{"class_name": c, "class_index": 0, "polygons": [p]}
                       for c, p in insts]}
        for fn, insts in frames]}
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "seg_manifest.json").write_text(json.dumps(doc))
    return d


def _coco(tmp_path, name, frames):
    cats = [{"id": 1, "name": CROP_CLASS}, {"id": 2, "name": "other_weed"}]
    images, anns = [], []
    aid = 1
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
        json.dumps({"images": images, "annotations": anns,
                    "categories": cats}))
    return d


# --------------------------------------------------------------------------- #
# Both input forms, because truth and prediction rarely come from one place
# --------------------------------------------------------------------------- #
def test_a_dataset_build_loads(tmp_path):
    d = _manifest(tmp_path, "gt", [("f_000001.png",
                                    [(CROP_CLASS, _sq(5, 5, 20))])])
    got = bm.load_side(d)
    assert list(got) == ["f_000001.png"]
    masks, classes = got["f_000001.png"]
    assert classes == [CROP_CLASS] and masks[0].sum() > 0


def test_a_prelabeler_coco_loads(tmp_path):
    d = _coco(tmp_path, "pred", [("f_000001.png",
                                  [("other_weed", _sq(5, 5, 20))])])
    got = bm.load_side(d)
    assert got["f_000001.png"][1] == ["other_weed"]


def test_the_two_forms_can_be_compared_against_each_other(tmp_path):
    """The normal case: hand annotation came through make_dataset, the
    prelabeler emits COCO."""
    gt = _manifest(tmp_path, "gt", [("f_000001.png",
                                     [(CROP_CLASS, _sq(5, 5, 20))])])
    pr = _coco(tmp_path, "pred", [("f_000001.png",
                                   [(CROP_CLASS, _sq(5, 5, 20))])])
    summary, _ = bm.benchmark(bm.load_side(gt), bm.load_side(pr))
    assert summary["n_frames_scored"] == 1
    assert summary["onion_as_weed_px"] == 0


def test_neither_form_present_says_what_to_point_at(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        bm.load_side(tmp_path / "empty")
    assert "make_dataset OUT_DIR" in str(e.value)


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def test_frames_are_paired_by_name_not_by_order(tmp_path):
    gt = _manifest(tmp_path, "gt", [
        ("b.png", [(CROP_CLASS, _sq(5, 5, 20))]),
        ("a.png", [("other_weed", _sq(5, 5, 20))])])
    pr = _manifest(tmp_path, "pred", [
        ("a.png", [("other_weed", _sq(5, 5, 20))]),
        ("b.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    summary, _ = bm.benchmark(bm.load_side(gt), bm.load_side(pr))
    assert summary["n_frames_scored"] == 2
    assert summary["onion_as_weed_px"] == 0, "pairing followed order, not name"


def test_only_the_intersection_is_scored_and_the_rest_is_reported(tmp_path):
    gt = _manifest(tmp_path, "gt", [
        ("a.png", [(CROP_CLASS, _sq(5, 5, 20))]),
        ("only_gt.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    pr = _manifest(tmp_path, "pred", [
        ("a.png", [(CROP_CLASS, _sq(5, 5, 20))]),
        ("only_pred.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    summary, _ = bm.benchmark(bm.load_side(gt), bm.load_side(pr))
    assert summary["n_frames_scored"] == 1
    assert summary["n_truth_only"] == 1 and summary["n_pred_only"] == 1


def test_no_overlap_at_all_fails_with_the_reason(tmp_path, capsys):
    gt = _manifest(tmp_path, "gt", [("a.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    pr = _manifest(tmp_path, "pred", [("z.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    with pytest.raises(SystemExit) as e:
        bm.main(["--truth", str(gt), "--pred", str(pr)])
    assert "FILE NAME" in str(e.value)


# --------------------------------------------------------------------------- #
# The pooling decision
# --------------------------------------------------------------------------- #
def test_crop_error_pools_by_pixels_not_by_frame(tmp_path):
    """A frame with one small onion must not outweigh a frame full of them just
    because both are one frame."""
    gt = _manifest(tmp_path, "gt", [
        ("small.png", [(CROP_CLASS, _sq(0, 0, 6))]),        # 36 px
        ("big.png", [(CROP_CLASS, _sq(0, 0, 40))])])        # 1600 px
    # The SMALL frame is wholly wrong; the big one is right.
    pr = _manifest(tmp_path, "pred", [
        ("small.png", [("other_weed", _sq(0, 0, 6))]),
        ("big.png", [(CROP_CLASS, _sq(0, 0, 40))])])
    summary, _ = bm.benchmark(bm.load_side(gt), bm.load_side(pr))
    # Frame-averaged this would be ~50%. Pixel-pooled it is small.
    assert summary["onion_as_weed_fraction"] < 0.05


def test_a_small_ruler_is_flagged_rather_than_refused(tmp_path, capsys):
    gt = _manifest(tmp_path, "gt", [("a.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    pr = _manifest(tmp_path, "pred", [("a.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    bm.main(["--truth", str(gt), "--pred", str(pr)])
    out = capsys.readouterr().out
    assert "small ruler" in out
    assert "still the right ruler" in out


def test_the_catastrophic_direction_is_named_in_the_report(tmp_path, capsys):
    """The report has to say which row matters; a table of four percentages
    does not tell a reader that one of them is a laser at the crop."""
    gt = _manifest(tmp_path, "gt", [("a.png", [(CROP_CLASS, _sq(5, 5, 20))])])
    pr = _manifest(tmp_path, "pred", [("a.png", [("other_weed", _sq(5, 5, 20))])])
    bm.main(["--truth", str(gt), "--pred", str(pr)])
    out = capsys.readouterr().out
    assert "laser at the crop" in out
    assert "100.00%" in out


def test_the_json_report_carries_every_frame(tmp_path):
    gt = _manifest(tmp_path, "gt", [
        ("a.png", [(CROP_CLASS, _sq(5, 5, 20))]),
        ("b.png", [("other_weed", _sq(5, 5, 20))])])
    pr = _manifest(tmp_path, "pred", [
        ("a.png", [(CROP_CLASS, _sq(5, 5, 20))]),
        ("b.png", [("other_weed", _sq(5, 5, 20))])])
    out = tmp_path / "bench" / "run.json"
    bm.main(["--truth", str(gt), "--pred", str(pr), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert len(doc["frames"]) == 2
    assert {f["frame"] for f in doc["frames"]} == {"a.png", "b.png"}
    assert "onion_as_weed_px" in doc["summary"]
