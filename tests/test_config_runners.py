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


def _export(tmp_path, name="export", session="sess_a", n=8, label_id=0,
           img_dir=None):
    """A minimal Datumaro export plus the session folder its images live
    under. `img_dir` lets a second export share (or not share) a sessions
    root with the first, to exercise the multi-source case.

    Filenames follow the REAL extractor convention `<session>_<index>.png`,
    flat with no directory component in the CVAT media path (CVAT tasks are
    flat uploads) - `_session_of` in datumaro_multitask.py derives the session
    from those trailing digits, not from any path prefix, so a fixture using a
    directory to distinguish sessions instead would silently collapse every
    session to the same wrong id."""
    items = []
    for i in range(1, n + 1):
        fname = f"{session}_{i:06d}.png"
        items.append({
            "id": f"{session}_{i:06d}",
            "media": {"path": fname},
            "image": {"size": [200, 200]},
            "annotations": [{
                "id": i, "type": "polygon", "label_id": label_id, "group": i,
                "points": [10, 10, 60, 10, 60, 60, 10, 60], "attributes": {},
            }],
        })
    doc = {"info": {}, "categories": {"label": {"labels": [
        {"name": c} for c in pd.CLASSES]}}, "items": items}
    ann = tmp_path / name / "annotations"
    ann.mkdir(parents=True)
    (ann / "default.json").write_text(json.dumps(doc))

    imgs_root = img_dir or (tmp_path / f"{name}_sessions")
    sdir = imgs_root / session / "rgb"
    sdir.mkdir(parents=True, exist_ok=True)
    import cv2
    for i in range(1, n + 1):
        cv2.imwrite(str(sdir / f"{session}_{i:06d}.png"),
                    np.zeros((200, 200, 3), np.uint8))
    return tmp_path / name, imgs_root


def _cfg(tmp_path, sources=None, **over):
    if sources is None:
        exp, imgs = _export(tmp_path)
        sources = [{"DATUMARO_ROOT": str(exp), "IMAGES_ROOT": str(imgs)}]
    c = dict(md.CONFIG)
    # Neutralise every selection key, not just the ones that existed when this
    # helper was written. These tests are about the RUNNER's guard rails, so
    # they must not inherit whatever class filter the checked-in CONFIG happens
    # to carry - an onion-only CONFIG silently dropped their synthetic weeds and
    # the failure surfaced as "no annotated frames left", nowhere near the cause.
    c.update({"SOURCES": sources, "OUT_DIR": str(tmp_path / "ds"),
              "LIST_FRAMES": False, "INCLUDE_FRAMES": "", "EXCLUDE_FRAMES": "",
              "DROP_CLASSES": [], "KEEP_CLASSES": None,
              "VAL_FRACTION": 0.0, "TEST_FRACTION": 0.0})
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# make_dataset guard rails - single source
# --------------------------------------------------------------------------- #
def test_empty_sources_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="SOURCES"):
        md.main(_cfg(tmp_path, sources=[]))


def test_empty_datumaro_root_in_a_source_names_it(tmp_path):
    with pytest.raises(SystemExit, match=r"SOURCES\[1\]\['DATUMARO_ROOT'\]"):
        md.main(_cfg(tmp_path, sources=[{"DATUMARO_ROOT": "  ",
                                         "IMAGES_ROOT": str(tmp_path)}]))


def test_missing_datumaro_root_says_what_to_point_at(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        md.main(_cfg(tmp_path, sources=[
            {"DATUMARO_ROOT": str(tmp_path / "nope"),
             "IMAGES_ROOT": str(tmp_path)}]))


def test_missing_images_root_is_reported_separately(tmp_path):
    exp, _ = _export(tmp_path)
    with pytest.raises(SystemExit, match=r"SOURCES\[1\]\['IMAGES_ROOT'\]"):
        md.main(_cfg(tmp_path, sources=[
            {"DATUMARO_ROOT": str(exp), "IMAGES_ROOT": str(tmp_path / "nope")}]))


def test_listing_pass_writes_nothing(tmp_path, capsys):
    c = _cfg(tmp_path, LIST_FRAMES=True)
    md.main(c)
    assert not (tmp_path / "ds").exists()
    out = capsys.readouterr().out
    assert "NOTHING WAS WRITTEN" in out
    assert "sess_a_000001" in out


def test_build_pass_writes_the_manifest(tmp_path):
    md.main(_cfg(tmp_path))
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
# make_dataset: multiple sources, the feature this session added
# --------------------------------------------------------------------------- #
def test_two_sources_under_different_parents_both_resolve(tmp_path):
    """The real case: a weed capture set and a separately-recorded onion set,
    each with its own sessions folder."""
    weed_exp, weed_imgs = _export(tmp_path, name="weed", session="vid2_weed",
                                  n=6, label_id=0)
    onion_exp, onion_imgs = _export(tmp_path, name="onion",
                                    session="onion1", n=4, label_id=5)
    assert weed_imgs != onion_imgs, "the fixture must actually use two roots"

    md.main(_cfg(tmp_path, sources=[
        {"DATUMARO_ROOT": str(weed_exp), "IMAGES_ROOT": str(weed_imgs)},
        {"DATUMARO_ROOT": str(onion_exp), "IMAGES_ROOT": str(onion_imgs)},
    ], INCLUDE_FRAMES="", VAL_FRACTION=0.2, TEST_FRACTION=0.0))

    man = json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())
    # POSIX separators, deliberately: the manifest records them that way so a
    # dataset built on Windows is readable on Linux and back. Comparing against
    # str(Path) instead made this test pass on one OS and fail on the other
    # while the code was right on both.
    assert sorted(man["images_root"]) == sorted(
        [weed_imgs.as_posix(), onion_imgs.as_posix()])
    # Both sources' frames resolved and reached the manifest. Not an exact
    # count: a per-session block split legitimately discards gap-buffer frames
    # at each boundary, so asserting 10 would be asserting the absence of a
    # feature.
    sessions = {f["session_id"] for f in man["frames"]}
    assert sessions == {"vid2_weed", "onion1"}
    assert 0 < len(man["frames"]) <= 10
    assert any(f["split"] == "val" for f in man["frames"])


def test_two_sources_sharing_one_images_root_are_not_duplicated(tmp_path):
    shared = tmp_path / "shared_sessions"
    exp1, _ = _export(tmp_path, name="s1", session="sess_1", n=3,
                      img_dir=shared)
    exp2, _ = _export(tmp_path, name="s2", session="sess_2", n=3,
                      img_dir=shared)
    md.main(_cfg(tmp_path, sources=[
        {"DATUMARO_ROOT": str(exp1), "IMAGES_ROOT": str(shared)},
        {"DATUMARO_ROOT": str(exp2), "IMAGES_ROOT": str(shared)},
    ], INCLUDE_FRAMES=""))
    man = json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())
    assert man["images_root"] == [shared.as_posix()], "one root, not repeated"
    assert len(man["frames"]) == 6


def test_include_frames_scoped_per_session_across_two_sources(tmp_path):
    weed_exp, weed_imgs = _export(tmp_path, name="weed", session="vid2_weed",
                                  n=6)
    onion_exp, onion_imgs = _export(tmp_path, name="onion", session="onion1",
                                    n=4, label_id=5)
    md.main(_cfg(tmp_path, sources=[
        {"DATUMARO_ROOT": str(weed_exp), "IMAGES_ROOT": str(weed_imgs)},
        {"DATUMARO_ROOT": str(onion_exp), "IMAGES_ROOT": str(onion_imgs)},
    ], INCLUDE_FRAMES="vid2_weed:1-3,onion1:*"))
    man = json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())
    ids = {f["item_id"] for f in man["frames"]}
    assert ids == ({f"vid2_weed_{i:06d}" for i in (1, 2, 3)}
                  | {f"onion1_{i:06d}" for i in range(1, 5)})


def test_listing_pass_covers_every_source(tmp_path, capsys):
    weed_exp, weed_imgs = _export(tmp_path, name="weed", session="vid2_weed",
                                  n=3)
    onion_exp, onion_imgs = _export(tmp_path, name="onion", session="onion1",
                                    n=2, label_id=5)
    md.main(_cfg(tmp_path, sources=[
        {"DATUMARO_ROOT": str(weed_exp), "IMAGES_ROOT": str(weed_imgs)},
        {"DATUMARO_ROOT": str(onion_exp), "IMAGES_ROOT": str(onion_imgs)},
    ], LIST_FRAMES=True))
    out = capsys.readouterr().out
    assert "vid2_weed" in out and "onion1" in out


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


def test_empty_images_root_falls_back_to_the_manifests_own_value(tmp_path):
    """The DRY path: make_dataset.py already recorded the roots, so
    train_model.py should not need them typed out a second time."""
    real_root = tmp_path / "sessions"
    real_root.mkdir()
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "seg_manifest.json").write_text(json.dumps(
        {"frames": [], "images_root": [str(real_root)]}))
    resolved = tm._resolve_images_root({"IMAGES_ROOT": ""},
                                       ds / "seg_manifest.json")
    assert resolved == str(real_root)


def test_empty_images_root_with_nothing_recorded_fails_clearly(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "seg_manifest.json").write_text(json.dumps({"frames": []}))
    with pytest.raises(SystemExit, match="IMAGES_ROOT"):
        tm._resolve_images_root({"IMAGES_ROOT": ""}, ds / "seg_manifest.json")


def test_images_root_as_a_list_is_validated_per_entry(tmp_path):
    ok = tmp_path / "ok"; ok.mkdir()
    with pytest.raises(SystemExit, match="do not exist"):
        tm._resolve_images_root(
            {"IMAGES_ROOT": [str(ok), str(tmp_path / "missing")]},
            tmp_path / "unused.json")


def test_the_manifest_records_posix_paths_on_every_platform(tmp_path):
    """A dataset built on Windows must be readable on Linux and back, so
    seg_manifest.json stores separators one way. Pinned because the tests that
    caught this compared against str(Path) and so passed on one OS while
    failing on the other, with the code right on both."""
    exp, imgs = _export(tmp_path, name="s", session="sess_1", n=3)
    md.main(_cfg(tmp_path, sources=[{"DATUMARO_ROOT": str(exp),
                                     "IMAGES_ROOT": str(imgs)}],
                 INCLUDE_FRAMES=""))
    man = json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())
    assert all("\\" not in r for r in man["images_root"])
    assert all("\\" not in f["image_path"] for f in man["frames"])


def test_the_runner_tests_do_not_inherit_the_checked_in_class_filter(tmp_path):
    """CONFIG is edited between runs - it is the file you set up a build in.
    A class filter left there once made every test in this module fail with
    'no annotated frames left', which points at the export rather than at the
    config that caused it. The helper must neutralise the filter regardless of
    what is checked in, so pin that rather than trusting it."""
    c = _cfg(tmp_path)
    assert c["KEEP_CLASSES"] is None
    assert c["DROP_CLASSES"] == []
    assert c["INCLUDE_FRAMES"] == "" and c["EXCLUDE_FRAMES"] == ""


# --------------------------------------------------------------------------- #
# The config blocks are hand-edited, and Python does not complain
# --------------------------------------------------------------------------- #
def test_no_config_block_has_a_duplicate_key():
    """A dict literal with the same key twice is legal Python: the LAST one
    wins, silently. In a config block that is edited by hand every run - and
    patched by scripts that match one line at a time - that is a setting you
    believe you changed and did not.

    It has happened: make_dataset.py carried two "OUT_DIR" lines, so the build
    wrote to whichever came second regardless of which one was edited."""
    import ast
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "seeweed3d"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name) and target.id.isupper()):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            keys = [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            for key, n in Counter(keys).items():
                if n > 1:
                    offenders.append(
                        f"{path.relative_to(root)}: {target.id}[{key!r}] "
                        f"appears {n} times")
    assert not offenders, "duplicate keys silently take the last value:\n  " \
        + "\n  ".join(offenders)
