"""Pre-flight: the checks that catch a run which finishes, prints a plausible
metric, and means nothing.

None of these raise during training. That is exactly why they need a pass of
their own before hours are committed to a run.
"""
import json

import pytest

from conftest import load_script

pf = load_script("training/preflight.py")


def _coco(dirpath, classes, per_class, n_frames):
    """A minimal COCO split on disk with the requested class counts."""
    dirpath.mkdir(parents=True, exist_ok=True)
    cats = [{"id": i + 1, "name": c, "supercategory": "plant"}
            for i, c in enumerate(classes)]
    images = [{"id": i + 1, "file_name": f"f{i:04d}.png", "height": 100,
               "width": 100} for i in range(n_frames)]
    anns, aid = [], 1
    for i, c in enumerate(classes):
        for _ in range(per_class.get(c, 0)):
            anns.append({"id": aid, "image_id": 1, "category_id": i + 1,
                         "segmentation": [[0, 0, 10, 0, 10, 10]],
                         "bbox": [0, 0, 10, 10], "area": 50, "iscrowd": 0})
            aid += 1
    (dirpath / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": anns,
                    "categories": cats}), encoding="utf-8")


def _tree(tmp_path, train=None, val=None, test=None, classes=("onion_plant",
                                                              "other_weed")):
    root = tmp_path / "coco"
    for name, spec in (("train", train), ("valid", val), ("test", test)):
        if spec is None:
            continue
        counts, frames = spec
        _coco(root / name, list(classes), counts, frames)
    return root


def _codes(findings):
    return {f.code for f in findings}


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
def test_a_class_with_no_validation_instances_is_flagged(tmp_path):
    """Nothing measures it, so early stopping and best-checkpoint selection are
    both blind to it - and when that class is the crop, the checkpoint chosen
    as 'best' may be the one that segments onions worst."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 400}, 40),
                 val=({"onion_plant": 0, "other_weed": 90}, 12))
    _, f = pf.preflight(root)
    assert "class_missing_from_val" in _codes(f)


def test_a_class_missing_from_training_is_an_error(tmp_path):
    """The model can never predict it, and a class it never predicts reports an
    empty mask downstream - indistinguishable from 'found nothing'."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 0, "other_weed": 400}, 40),
                 val=({"onion_plant": 30, "other_weed": 90}, 12))
    _, f = pf.preflight(root)
    errs = [x for x in f if x.code == "class_missing_from_train"]
    assert errs and errs[0].level == "error"


def test_a_class_too_rare_to_learn_is_flagged(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 7}, 40),
                 val=({"onion_plant": 40, "other_weed": 3}, 12))
    _, f = pf.preflight(root)
    rare = [x for x in f if x.code == "class_too_rare"]
    assert rare and "DROP_CLASSES" in rare[0].fix


def test_a_class_absent_from_the_whole_build_is_not_flagged(tmp_path):
    """DROP_CLASSES removed it deliberately; complaining would punish the fix."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 0}, 40),
                 val=({"onion_plant": 40, "other_weed": 0}, 12))
    _, f = pf.preflight(root)
    assert not {"class_too_rare", "class_missing_from_train",
                "class_missing_from_val"} & _codes(f)


def test_a_healthy_dataset_raises_no_class_findings(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 300, "other_weed": 500}, 60),
                 val=({"onion_plant": 60, "other_weed": 90}, 15),
                 test=({"onion_plant": 55, "other_weed": 80}, 14))
    _, f = pf.preflight(root, epochs=60, patience=20)
    assert not [x for x in f if x.level == "error"]
    assert not {"class_too_rare", "class_missing_from_val"} & _codes(f)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_an_empty_val_split_is_an_error(tmp_path):
    root = _tree(tmp_path, train=({"onion_plant": 200, "other_weed": 300}, 40))
    _, f = pf.preflight(root)
    assert "no_val_frames" in _codes(f)


def test_a_tiny_val_split_is_a_warning_not_an_error(tmp_path):
    """Below ~10 frames, epoch-to-epoch noise exceeds the difference between
    checkpoints, so 'best' is chosen substantially by chance."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 300}, 40),
                 val=({"onion_plant": 20, "other_weed": 30}, 4))
    _, f = pf.preflight(root)
    small = [x for x in f if x.code == "val_too_small"]
    assert small and small[0].level == "warn"


def test_no_test_split_is_reported(tmp_path):
    """Every number then comes from the set used to choose the checkpoint."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 300}, 40),
                 val=({"onion_plant": 40, "other_weed": 60}, 12))
    _, f = pf.preflight(root)
    assert "no_test_split" in _codes(f)


# --------------------------------------------------------------------------- #
# The schedule
# --------------------------------------------------------------------------- #
def test_steps_per_epoch_is_a_ceiling():
    assert pf.steps_per_epoch(60, 2, 8) == 4      # 60/16 -> 4
    assert pf.steps_per_epoch(16, 2, 8) == 1
    assert pf.steps_per_epoch(17, 2, 8) == 2


def test_the_step_count_is_reported(tmp_path):
    """`epochs` means ten times the compute at ten times the dataset, for the
    same number in the config."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 300, "other_weed": 500}, 640),
                 val=({"onion_plant": 60, "other_weed": 90}, 20))
    _, f = pf.preflight(root, epochs=60, batch=2, grad_accum=8)
    sched = [x for x in f if x.code == "schedule"][0]
    assert "40 step(s)/epoch" in sched.message and "2400" in sched.message


def test_an_effective_batch_larger_than_the_dataset_is_flagged(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 60, "other_weed": 80}, 12),
                 val=({"onion_plant": 40, "other_weed": 60}, 12))
    _, f = pf.preflight(root, batch=2, grad_accum=8)
    assert "one_step_per_epoch" in _codes(f)


def test_a_patience_that_can_never_fire_is_flagged(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 300, "other_weed": 500}, 640),
                 val=({"onion_plant": 60, "other_weed": 90}, 20))
    _, f = pf.preflight(root, epochs=20, patience=25)
    assert "patience_exceeds_run" in _codes(f)


def test_patience_is_not_flagged_when_early_stopping_is_off(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 300, "other_weed": 500}, 640),
                 val=({"onion_plant": 60, "other_weed": 90}, 20))
    _, f = pf.preflight(root, epochs=20, patience=25, early_stopping=False)
    assert "patience_exceeds_run" not in _codes(f)


# --------------------------------------------------------------------------- #
# Reading the dataset
# --------------------------------------------------------------------------- #
def test_it_reads_the_tree_when_the_sidecar_is_missing(tmp_path):
    """A COCO folder shared between runs may predate the summary, and refusing
    to check then would make the check skippable exactly when it matters."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 300}, 40),
                 val=({"onion_plant": 40, "other_weed": 60}, 12))
    assert not (root / "seeweed3d_export.json").exists()
    s, _ = pf.preflight(root)
    assert s["splits"]["train"]["per_class"]["other_weed"] == 300


def test_it_prefers_the_sidecar_when_it_has_per_class_counts(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 300}, 40),
                 val=({"onion_plant": 40, "other_weed": 60}, 12))
    (root / "seeweed3d_export.json").write_text(json.dumps({
        "classes": ["onion_plant"],
        "splits": {"train": {"frames": 999, "instances": 1,
                             "per_class": {"onion_plant": 1}}}}),
        encoding="utf-8")
    s, _ = pf.preflight(root)
    assert s["splits"]["train"]["frames"] == 999


def test_an_old_sidecar_without_per_class_falls_back_to_the_tree(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 300}, 40),
                 val=({"onion_plant": 40, "other_weed": 60}, 12))
    (root / "seeweed3d_export.json").write_text(json.dumps({
        "classes": ["onion_plant"],
        "splits": {"train": {"frames": 999, "instances": 1}}}),
        encoding="utf-8")
    s, _ = pf.preflight(root)
    assert s["splits"]["train"]["frames"] == 40


def test_an_empty_directory_says_what_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="no COCO splits"):
        pf.preflight(tmp_path / "nothing")


def test_the_report_renders(tmp_path):
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 7}, 40),
                 val=({"onion_plant": 40, "other_weed": 0}, 12))
    text = pf.format_report(*pf.preflight(root))
    assert "other_weed" in text and "[!]" in text


def test_findings_serialise(tmp_path):
    """They are written to preflight.json beside the run."""
    root = _tree(tmp_path,
                 train=({"onion_plant": 200, "other_weed": 300}, 40),
                 val=({"onion_plant": 40, "other_weed": 60}, 12))
    _, f = pf.preflight(root)
    assert json.loads(json.dumps([x.to_dict() for x in f]))


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_the_rfdetr_runner_exposes_the_override():
    tm = load_script("training/train_model_rfdetr.py")
    assert isinstance(tm.CONFIG["SKIP_PREFLIGHT"], bool)


def test_the_trainer_runs_preflight_before_training():
    """After the COCO export - it needs the tree - and before model.train()."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "seeweed3d" / "training" /
           "train_seg_rfdetr.py").read_text(encoding="utf-8")
    assert src.index("pf.preflight(") < src.index("model.train(")
    assert src.index("from training.coco_export import export") \
        < src.index("pf.preflight(")


def test_coco_export_records_per_class_counts():
    """Pre-flight's whole argument is that the total instance count is the
    number least worth looking at."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "seeweed3d" / "training" /
           "coco_export.py").read_text(encoding="utf-8")
    assert '"per_class"' in src
