"""The config-block runners: make_dataset.py and train_model.py.

These are the entry points actually used day to day, so their guard rails
matter more than their happy path - a wrong path should say which CONFIG key is
wrong, not raise a FileNotFoundError from three frames deep.
"""
import json

import numpy as np
import pytest

from conftest import load_script

md = load_script("training/make_dataset.py")
tm = load_script("training/train_model.py")
pd = load_script("training/prepare_dataset.py")


def _export(tmp_path, n=8):
    items = []
    for i in range(1, n + 1):
        items.append({
            "id": f"sess_a/frame_{i:04d}",
            "media": {"path": f"sess_a/frame_{i:04d}.png"},
            "image": {"size": [200, 200]},
            "annotations": [{
                "id": i, "type": "polygon", "label_id": 0, "group": i,
                "points": [10, 10, 60, 10, 60, 60, 10, 60], "attributes": {},
            }],
        })
    doc = {"info": {}, "categories": {"label": {"labels": [
        {"name": c} for c in pd.CLASSES]}}, "items": items}
    ann = tmp_path / "export" / "annotations"
    ann.mkdir(parents=True)
    (ann / "default.json").write_text(json.dumps(doc))
    imgs = tmp_path / "sessions" / "sess_a"
    imgs.mkdir(parents=True)
    import cv2
    for i in range(1, n + 1):
        cv2.imwrite(str(imgs / f"frame_{i:04d}.png"),
                    np.zeros((200, 200, 3), np.uint8))
    return tmp_path / "export", tmp_path / "sessions"


def _cfg(tmp_path, **over):
    exp, imgs = _export(tmp_path)
    c = dict(md.CONFIG)
    c.update({"DATUMARO_ROOT": str(exp), "IMAGES_ROOT": str(imgs),
              "OUT_DIR": str(tmp_path / "ds"), "LIST_FRAMES": False,
              "INCLUDE_FRAMES": "", "DROP_CLASSES": [],
              "VAL_FRACTION": 0.0, "TEST_FRACTION": 0.0})
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# make_dataset guard rails
# --------------------------------------------------------------------------- #
def test_empty_config_path_names_the_key(tmp_path):
    c = _cfg(tmp_path, DATUMARO_ROOT="  ")
    with pytest.raises(SystemExit, match="DATUMARO_ROOT"):
        md.main(c)


def test_missing_datumaro_root_says_what_to_point_at(tmp_path):
    c = _cfg(tmp_path, DATUMARO_ROOT=str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="does not exist"):
        md.main(c)


def test_missing_images_root_is_reported_separately(tmp_path):
    c = _cfg(tmp_path, IMAGES_ROOT=str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="IMAGES_ROOT"):
        md.main(c)


def test_listing_pass_writes_nothing(tmp_path, capsys):
    c = _cfg(tmp_path, LIST_FRAMES=True)
    md.main(c)
    assert not (tmp_path / "ds").exists()
    out = capsys.readouterr().out
    assert "NOTHING WAS WRITTEN" in out
    assert "frame_0001" in out


def test_build_pass_writes_the_manifest(tmp_path):
    c = _cfg(tmp_path)
    md.main(c)
    man = json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())
    assert len(man["frames"]) == 8


def test_empty_include_frames_warns_that_everything_is_used(tmp_path, capsys):
    """Silence here is how SAM's wrong-class frames reach training."""
    md.main(_cfg(tmp_path, INCLUDE_FRAMES=""))
    assert "EVERY frame" in capsys.readouterr().out


def test_include_frames_is_honoured_through_the_runner(tmp_path):
    md.main(_cfg(tmp_path, INCLUDE_FRAMES="1-5"))
    man = json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())
    assert len(man["frames"]) == 5


# --------------------------------------------------------------------------- #
# COCO / Datumaro sniffing
# --------------------------------------------------------------------------- #
def test_a_coco_export_is_identified_by_name(tmp_path):
    """Pointing at the SAM 3 output folder is the easy mistake - it is where
    the CVAT round trip started, and it holds COCO."""
    d = tmp_path / "auto_labels" / "sess_a"
    d.mkdir(parents=True)
    (d / "instances_default.json").write_text(json.dumps(
        {"images": [{"id": 1}], "annotations": [], "categories": []}))
    with pytest.raises(SystemExit, match="COCO"):
        pd.find_annotation_files(d)


def test_the_coco_error_says_to_export_datumaro_instead(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "instances_default.json").write_text(json.dumps(
        {"images": [], "annotations": [], "categories": []}))
    with pytest.raises(SystemExit, match="Datumaro 1.0"):
        pd.find_annotation_files(d)


def test_unrelated_json_does_not_masquerade_as_an_export(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"some": "setting"}))
    with pytest.raises(SystemExit, match="neither Datumaro nor COCO"):
        pd.find_annotation_files(d)


def test_a_real_datumaro_file_is_found_next_to_junk(tmp_path):
    exp, _ = _export(tmp_path)
    (exp / "stray.json").write_text(json.dumps({"unrelated": True}))
    (exp / "coco.json").write_text(json.dumps(
        {"images": [], "annotations": []}))
    files = pd.find_annotation_files(exp)
    assert [p.name for p in files] == ["default.json"]


def test_unreadable_json_is_skipped_not_fatal(tmp_path):
    exp, _ = _export(tmp_path)
    (exp / "broken.json").write_text("{ this is not json")
    assert len(pd.find_annotation_files(exp)) == 1


# --------------------------------------------------------------------------- #
# train_model guard rails
# --------------------------------------------------------------------------- #
def test_training_without_a_dataset_points_at_make_dataset(tmp_path):
    c = dict(tm.CONFIG)
    c.update({"DATASET_DIR": str(tmp_path / "nothing"),
              "IMAGES_ROOT": str(tmp_path)})
    with pytest.raises(SystemExit, match="make_dataset"):
        tm.main(c)


def test_training_with_a_missing_images_root_is_caught_before_the_loop(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "seg_manifest.json").write_text(json.dumps({"frames": []}))
    c = dict(tm.CONFIG)
    c.update({"DATASET_DIR": str(ds), "IMAGES_ROOT": str(tmp_path / "nope")})
    with pytest.raises(SystemExit, match="IMAGES_ROOT"):
        tm.main(c)
