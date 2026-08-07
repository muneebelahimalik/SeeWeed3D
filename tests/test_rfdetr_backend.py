"""RF-DETR-Seg backend: COCO export and the resolution guard.

The guard is the point of this module. RF-DETR-Seg defaults to 432x432, which
on a 2208x1242 ZED frame loses small weeds far more severely than the 1333 px
Mask R-CNN default already did - so adopting it without choosing a resolution
would REGRESS the exact metric it looks like an upgrade for.
"""
import json

import numpy as np
import pytest

from conftest import load_script

ce = load_script("training/coco_export.py")
rf = load_script("training/train_seg_rfdetr.py")

rfdetr = pytest.importorskip("rfdetr", reason="optional backend")


# --------------------------------------------------------------------------- #
# COCO export
# --------------------------------------------------------------------------- #
def _dataset(tmp_path, n_train=3, n_val=2, classes=("grass_weed",
                                                    "onion_plant")):
    import cv2
    root = tmp_path / "sessions" / "s1" / "rgb"
    root.mkdir(parents=True)
    frames = []
    for i in range(1, n_train + n_val + 1):
        name = f"s1_{i:06d}.png"
        cv2.imwrite(str(root / name), np.zeros((120, 160, 3), np.uint8))
        frames.append({
            "session_id": "s1", "item_id": f"s1_{i:06d}", "image_path": name,
            "width": 160, "height": 120,
            "split": "train" if i <= n_train else "val",
            "instances": [
                {"class_name": classes[0],
                 "polygons": [[10, 10, 50, 10, 50, 50, 10, 50]]},
                {"class_name": classes[-1],
                 "polygons": [[80, 60, 120, 60, 120, 100, 80, 100]]},
            ]})
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "seg_manifest.json").write_text(json.dumps(
        {"images_root": [str(tmp_path / "sessions")], "classes": list(classes),
         "frames": frames}))
    return ds


def test_export_writes_the_roboflow_layout(tmp_path):
    """rfdetr discovers a dataset by looking for these exact paths - and it is
    'valid', not 'val'."""
    ds = _dataset(tmp_path)
    out = tmp_path / "coco"
    ce.export(ds, out)
    assert (out / "train" / "_annotations.coco.json").exists()
    assert (out / "valid" / "_annotations.coco.json").exists()
    assert not (out / "val").exists()


def test_images_land_beside_their_annotations(tmp_path):
    ds = _dataset(tmp_path)
    out = tmp_path / "coco"
    ce.export(ds, out)
    coco = json.loads((out / "train" / "_annotations.coco.json").read_text())
    for im in coco["images"]:
        assert (out / "train" / im["file_name"]).exists()


def test_category_ids_are_one_based_and_contiguous(tmp_path):
    """COCO reserves 0 for background. Using the ontology's global ids would
    leave a gap wherever --drop-classes removed a class, and a gap becomes an
    off-by-one rather than an error."""
    ds = _dataset(tmp_path, classes=("grass_weed", "onion_plant"))
    out = tmp_path / "coco"
    s = ce.export(ds, out)
    coco = json.loads((out / "train" / "_annotations.coco.json").read_text())
    ids = sorted(c["id"] for c in coco["categories"])
    assert ids == list(range(1, len(s["classes"]) + 1))
    assert min(a["category_id"] for a in coco["annotations"]) >= 1


def test_area_is_rasterised_not_summed_polygon_area(tmp_path):
    """Two overlapping polygons of one instance must not double-count. COCO
    area drives the small/medium/large evaluation split, so an inflated value
    quietly promotes a small weed out of the bucket this project measures."""
    one = [[0, 0, 40, 0, 40, 40, 0, 40]]
    _, area_one = ce.instance_bbox_and_area(one, 120, 160)
    bbox, area_two = ce.instance_bbox_and_area(one * 2, 120, 160)
    # The same polygon twice must not double the area. (fillPoly is inclusive
    # of both endpoints, so a 0..40 square rasterises to 41x41 - the property
    # under test is the absence of doubling, not the exact count.)
    assert area_two == area_one
    assert area_two < 2 * area_one
    assert bbox[2] == pytest.approx(41, abs=1.5)


def test_an_instance_with_no_drawable_polygon_is_skipped(tmp_path):
    bbox, area = ce.instance_bbox_and_area([[1, 2, 3, 4]], 50, 50)
    assert bbox is None and area == 0


def test_a_missing_val_split_is_refused(tmp_path):
    """rfdetr evaluates against valid/ every epoch. Without it there is no
    early stopping, no best checkpoint and no metric."""
    ds = _dataset(tmp_path, n_train=3, n_val=0)
    with pytest.raises(SystemExit, match="no val frames"):
        ce.export(ds, tmp_path / "coco")


def test_a_non_empty_output_dir_is_refused_without_overwrite(tmp_path):
    """rfdetr scans the directory, so a leftover image from a previous build
    would silently join this dataset."""
    ds = _dataset(tmp_path)
    out = tmp_path / "coco"
    ce.export(ds, out)
    with pytest.raises(SystemExit, match="not empty"):
        ce.export(ds, out)
    ce.export(ds, out, overwrite=True)      # explicit is fine


def test_export_counts_match_the_manifest(tmp_path):
    ds = _dataset(tmp_path, n_train=3, n_val=2)
    s = ce.export(ds, tmp_path / "coco")
    assert s["splits"]["train"]["frames"] == 3
    assert s["splits"]["valid"]["frames"] == 2
    assert s["splits"]["train"]["instances"] == 6      # 3 frames x 2 instances


# --------------------------------------------------------------------------- #
# the resolution guard
# --------------------------------------------------------------------------- #
def test_the_step_comes_from_the_package_not_a_constant():
    """Hard-coding 24 would drift the moment a variant changed patch_size."""
    from rfdetr import config as C
    for v in ("nano", "medium"):
        cfg = getattr(C, f"RFDETRSeg{v.capitalize()}Config")
        assert rf.resolution_step(v) == (
            cfg.model_fields["patch_size"].default
            * cfg.model_fields["num_windows"].default)


def test_a_valid_resolution_passes():
    step = rf.resolution_step("medium")
    assert rf.check_resolution("medium", step * 42) == step * 42


def test_an_invalid_resolution_names_the_valid_neighbours():
    """rfdetr does raise, but only after the dataset loads and weights
    download, and it does not say which values would work."""
    step = rf.resolution_step("medium")
    with pytest.raises(SystemExit) as e:
        rf.check_resolution("medium", step * 42 + 1)
    msg = str(e.value)
    assert f"multiple of {step}" in msg
    assert str(step * 42) in msg


def test_training_refuses_the_low_default_resolution(tmp_path):
    """The whole reason this backend needs a guard: silently accepting 432
    would regress small-weed recall while looking like an upgrade."""
    ds = _dataset(tmp_path)
    with pytest.raises(SystemExit, match="resolution"):
        rf.train(ds, tmp_path / "run", resolution=None, device="cpu")


def test_the_refusal_explains_the_consequence(tmp_path):
    ds = _dataset(tmp_path)
    with pytest.raises(SystemExit) as e:
        rf.train(ds, tmp_path / "run", resolution=None, device="cpu")
    msg = str(e.value)
    assert "small weeds" in msg
    assert "allow-default-resolution" in msg


def test_an_unknown_variant_is_rejected(tmp_path):
    ds = _dataset(tmp_path)
    with pytest.raises(SystemExit, match="variant"):
        rf.train(ds, tmp_path / "run", variant="enormous", device="cpu")


def test_only_apache_licensed_variants_are_offered():
    """XLarge/2XLarge may fall under Roboflow's Platform Model License, which
    is not a licence this project can ship under without checking."""
    assert set(rf.VARIANTS) == {"nano", "small", "medium", "large"}


# --------------------------------------------------------------------------- #
# the config runner
# --------------------------------------------------------------------------- #
def test_the_config_runner_points_at_make_dataset_when_the_dataset_is_absent(
        tmp_path):
    tm = load_script("training/train_model_rfdetr.py")
    c = dict(tm.CONFIG)
    c.update({"DATASET_DIR": str(tmp_path / "nothing"),
              "IMAGES_ROOT": str(tmp_path)})
    with pytest.raises(SystemExit, match="make_dataset"):
        tm.main(c)


def test_the_config_default_resolution_is_not_the_models_low_default():
    """432 would regress small-weed recall while looking like an upgrade, so
    the shipped config must not quietly inherit it."""
    tm = load_script("training/train_model_rfdetr.py")
    assert tm.CONFIG["RESOLUTION"] != rf.default_resolution(
        tm.CONFIG["VARIANT"])
    assert tm.CONFIG["RESOLUTION"] % rf.resolution_step(
        tm.CONFIG["VARIANT"]) == 0


def test_the_config_effective_batch_is_sane():
    """RF-DETR expects an effective batch near 16; gradient accumulation is
    what lets resolution stay high on small VRAM."""
    tm = load_script("training/train_model_rfdetr.py")
    eff = tm.CONFIG["BATCH"] * tm.CONFIG["GRAD_ACCUM"]
    assert 8 <= eff <= 32, f"effective batch {eff}"


# --------------------------------------------------------------------------- #
# MLflow: the store rfdetr's lightning logger points at
#
# rfdetr builds pytorch-lightning's MLFlowLogger without a tracking_uri, so
# lightning falls back to the './mlruns' FILE store - which MLflow 3 refuses
# outright, killing the run before the first epoch.
# --------------------------------------------------------------------------- #
def test_the_tracking_uri_is_sqlite_not_the_file_store(tmp_path):
    """A bare directory is what MLflow 3 rejects; nothing else here matters if
    this is wrong."""
    uri = rf._point_lightning_at_our_mlflow_store(tmp_path / "run" / "rfdetr_v1")
    assert uri.startswith("sqlite:///")
    assert not uri.startswith("file:")


def test_rfdetr_and_maskrcnn_resolve_to_the_same_store(tmp_path, monkeypatch):
    """Two run folders under one training dir must share a store, or the
    comparison table the second backend exists for never materialises."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tk = load_script("training/tracking.py")
    mask, _ = tk.mlflow_store_uri(tmp_path / "training1" / "run3")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    detr = rf._point_lightning_at_our_mlflow_store(
        tmp_path / "training1" / "rfdetr_v1")
    assert mask == detr


def test_the_uri_is_exported_before_rfdetr_could_be_imported(tmp_path,
                                                             monkeypatch):
    """Lightning captures os.getenv("MLFLOW_TRACKING_URI") as a DEFAULT
    ARGUMENT, evaluated once at import. Setting it after the fact is too
    late, so this must be an environment variable and not a return value
    alone."""
    import os
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = rf._point_lightning_at_our_mlflow_store(tmp_path / "run")
    assert os.environ["MLFLOW_TRACKING_URI"] == uri


def test_an_explicit_tracking_uri_is_not_overridden(tmp_path, monkeypatch):
    """Someone pointing at a shared MLflow server has made a decision."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
    assert rf._point_lightning_at_our_mlflow_store(tmp_path / "run") == \
        "http://mlflow.internal:5000"


def test_setting_up_the_store_creates_its_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    run = tmp_path / "training1" / "rfdetr_v1"
    uri = rf._point_lightning_at_our_mlflow_store(run)
    assert rf._mlflow_store_is_reachable(run, uri)
    assert (tmp_path / "training1" / "mlruns").is_dir()


def test_a_broken_store_disables_mlflow_instead_of_killing_the_run(
        tmp_path, monkeypatch, capsys):
    """Losing hours of training to a charting library is never the right
    trade - the same rule the Mask R-CNN tracker already follows."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    run = tmp_path / "training1" / "rfdetr_v1"
    uri = rf._point_lightning_at_our_mlflow_store(run)
    import mlflow
    monkeypatch.setattr(mlflow, "set_tracking_uri",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            RuntimeError("locked")))
    assert rf._mlflow_store_is_reachable(run, uri) is False
    assert "mlflow disabled" in capsys.readouterr().out


def test_both_backends_log_into_one_experiment():
    """rfdetr names the MLflow experiment after `project`; a different name
    means two tables that cannot be compared."""
    tk = load_script("training/tracking.py")
    import inspect
    assert (inspect.signature(tk.Tracker.__init__)
            .parameters["experiment"].default) == tk.EXPERIMENT


def test_a_missing_lightning_does_not_disable_mlflow(tmp_path, monkeypatch):
    """The environment variable is the supported mechanism and works on its
    own; the lightning rebind is only a guard against import order. Treating
    its absence as a store failure would silently drop tracking for a reason
    that has nothing to do with the store."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    run = tmp_path / "training1" / "rfdetr_v1"
    uri = rf._point_lightning_at_our_mlflow_store(run)
    monkeypatch.setattr(rf, "_rebind_lightning_tracking_uri",
                        lambda _u: False)
    assert rf._mlflow_store_is_reachable(run, uri) is True
