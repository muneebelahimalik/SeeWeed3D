"""fix_coco_categories.py: repair pre-rename category names in an old SAM 3
COCO export before it is imported into CVAT.

The failure this exists to prevent: CVAT's COCO import matches labels by NAME.
An export with a stale category name does not error on import - it silently
creates a second, duplicate label instead of filling the one already in the
task, so hours of prelabelling never show up where they should."""
import json

import pytest

from conftest import load_script

fx = load_script("annotation/fix_coco_categories.py")


def _coco(*names):
    return {
        "info": {}, "licenses": [], "images": [{"id": 1, "file_name": "a.png"}],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "segmentation": [[0, 0, 1, 0, 1, 1]], "area": 1,
                         "bbox": [0, 0, 1, 1], "iscrowd": 0}],
    }


# --------------------------------------------------------------------------- #
# fix_categories()
# --------------------------------------------------------------------------- #
def test_the_exact_stale_onion_export_is_repaired():
    fixed, renamed, unresolved = fx.fix_categories(_coco("onion plant"))
    assert unresolved == []
    assert renamed == [("onion plant", "onion_plant")]
    assert fixed["categories"][0]["name"] == "onion_plant"


def test_a_current_name_is_left_alone():
    fixed, renamed, unresolved = fx.fix_categories(_coco("onion_plant"))
    assert renamed == [] and unresolved == []
    assert fixed["categories"][0]["name"] == "onion_plant"


def test_ignore_region_alias_is_known():
    _, renamed, unresolved = fx.fix_categories(_coco("ignore region"))
    assert renamed == [("ignore region", "ignore_region")]
    assert unresolved == []


def test_multiple_categories_in_one_file():
    fixed, renamed, unresolved = fx.fix_categories(
        _coco("onion plant", "ignore region", "grass_weed"))
    names = [c["name"] for c in fixed["categories"]]
    assert names == ["onion_plant", "ignore_region", "grass_weed"]
    assert len(renamed) == 2
    assert unresolved == []


def test_an_unrecognised_name_is_reported_not_guessed():
    """A wrong guess would relabel a class to a DIFFERENT one - refuse."""
    _, renamed, unresolved = fx.fix_categories(_coco("purple sow thistle"))
    assert unresolved == ["purple sow thistle"]
    assert renamed == []


def test_the_generic_space_to_underscore_fallback_only_fires_on_a_real_class():
    """A name that happens to have a space but is NOT a real class after the
    substitution must still be reported, not silently accepted."""
    _, renamed, unresolved = fx.fix_categories(_coco("totally made up name"))
    assert unresolved == ["totally made up name"]


def test_annotations_and_geometry_are_never_touched():
    fixed, _, _ = fx.fix_categories(_coco("onion plant"))
    assert fixed["annotations"] == _coco("onion plant")["annotations"]


def test_the_input_dict_is_not_mutated():
    original = _coco("onion plant")
    frozen = json.dumps(original)
    fx.fix_categories(original)
    assert json.dumps(original) == frozen


def test_category_ids_are_preserved():
    fixed, _, _ = fx.fix_categories(_coco("onion plant"))
    assert fixed["categories"][0]["id"] == 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_writes_the_repaired_file(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(_coco("onion plant")))
    out = tmp_path / "out.json"
    fx.main(["--in", str(src), "--out", str(out)])
    fixed = json.loads(out.read_text())
    assert fixed["categories"][0]["name"] == "onion_plant"


def test_cli_can_overwrite_the_same_path(tmp_path):
    src = tmp_path / "instances_default.json"
    src.write_text(json.dumps(_coco("onion plant")))
    fx.main(["--in", str(src), "--out", str(src)])
    assert json.loads(src.read_text())["categories"][0]["name"] == "onion_plant"


def test_cli_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(_coco("onion plant")))
    out = tmp_path / "out.json"
    fx.main(["--in", str(src), "--out", str(out), "--dry-run"])
    assert not out.exists()
    assert "onion plant" in src.read_text()


def test_cli_refuses_to_write_when_a_category_is_unresolved(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(_coco("mystery weed")))
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="mystery weed"):
        fx.main(["--in", str(src), "--out", str(out)])
    assert not out.exists()


def test_cli_reports_no_changes_needed_for_an_already_current_file(tmp_path,
                                                                    capsys):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(_coco("onion_plant")))
    out = tmp_path / "out.json"
    fx.main(["--in", str(src), "--out", str(out)])
    assert "Nothing to do" in capsys.readouterr().out


def test_missing_input_file_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        fx.main(["--in", str(tmp_path / "nope.json"), "--out",
                 str(tmp_path / "out.json")])


def test_not_actually_coco_fails_clearly(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"items": []}))     # a Datumaro file, not COCO
    with pytest.raises(SystemExit, match="categories"):
        fx.main(["--in", str(src), "--out", str(tmp_path / "out.json")])


def test_invalid_json_fails_clearly(tmp_path):
    src = tmp_path / "in.json"
    src.write_text("{ not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        fx.main(["--in", str(src), "--out", str(tmp_path / "out.json")])
