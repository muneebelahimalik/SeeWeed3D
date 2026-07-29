"""regen_cvat_labels.py must refresh the label schema in existing session
folders in seconds, without touching anything SAM 3 produced, and without
creating folders that were never actually processed."""
import json

from conftest import load_script

regen = load_script("annotation/regen_cvat_labels.py")


def _session(dir_, has_instances=True):
    dir_.mkdir(parents=True)
    if has_instances:
        (dir_ / "instances_default.json").write_text("{}")
    (dir_ / "masks").mkdir()
    (dir_ / "masks" / "keep.png").write_text("do not touch")


def test_regenerates_weed_and_onion_labels_in_place(tmp_path):
    root = tmp_path / "dataset"
    _session(root / "auto_labels_weeds" / "weed1_20260108_143022")
    _session(root / "auto_labels_onion" / "vid3_20260108_132749")
    # a folder that was never actually prelabeled must be skipped, not created
    unprocessed = root / "auto_labels_weeds" / "weed1_20260109_090000"
    unprocessed.mkdir(parents=True)

    written = regen.regenerate({"DATASET_ROOT": str(root),
                                "TARGETS": {"auto_labels_weeds": "weed_cvat_labels.json",
                                            "auto_labels_onion": "onion_cvat_labels.json"}})
    assert len(written) == 2
    assert not (unprocessed / "weed_cvat_labels.json").exists()

    weed_json = json.loads((root / "auto_labels_weeds" /
                            "weed1_20260108_143022" / "weed_cvat_labels.json").read_text())
    onion_json = json.loads((root / "auto_labels_onion" /
                             "vid3_20260108_132749" / "onion_cvat_labels.json").read_text())

    from common.ontology import cvat_labels as ontology_labels
    assert weed_json == ontology_labels()
    assert onion_json == regen.onion_labels()

    # Nothing SAM 3 produced was touched.
    assert (root / "auto_labels_weeds" / "weed1_20260108_143022" /
           "masks" / "keep.png").read_text() == "do not touch"


def test_every_attribute_still_has_nonempty_values(tmp_path):
    """The whole point of this script existing: it must never regenerate a
    schema that CVAT will reject."""
    root = tmp_path / "dataset"
    _session(root / "auto_labels_weeds" / "s1")
    regen.regenerate({"DATASET_ROOT": str(root),
                      "TARGETS": {"auto_labels_weeds": "weed_cvat_labels.json"}})
    labels = json.loads((root / "auto_labels_weeds" / "s1" /
                         "weed_cvat_labels.json").read_text())
    for label in labels:
        for attr in label["attributes"]:
            assert attr.get("values"), f"{label['name']}.{attr['name']} would break CVAT"


def test_missing_dataset_root_targets_are_skipped_not_fatal(tmp_path):
    root = tmp_path / "empty_dataset"
    written = regen.regenerate({"DATASET_ROOT": str(root),
                                "TARGETS": {"auto_labels_weeds": "weed_cvat_labels.json"}})
    assert written == []
