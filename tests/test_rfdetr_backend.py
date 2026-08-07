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


def test_the_shipped_config_uses_no_dataloader_workers(tmp_path):
    """A lightning worker that fails to start on Windows kills the process with
    no traceback - training stops after the model-summary table and there is
    nothing to read. At 62 frames the parent can do the loading."""
    tm = load_script("training/train_model_rfdetr.py")
    assert tm.CONFIG["WORKERS"] == 0


# --------------------------------------------------------------------------- #
# loading a trained checkpoint back
#
# Three things travel with rfdetr weights and none are inside the .pth: the
# variant, the training resolution and the class list. Every wrong answer is
# silent, and one of them is the crop-safety bug already fixed once for
# Mask R-CNN.
# --------------------------------------------------------------------------- #
import json as _json                                            # noqa: E402

import numpy as _np                                             # noqa: E402

sg = load_script("perception/segmenter.py")
from seeweed3d.common.ontology import CLASSES, CROP_CLASS       # noqa: E402

ACTIVE = ["cutleaf_evening_primrose", "grass_weed", "other_weed", CROP_CLASS]


def _run_dir(tmp_path, coco_only=False, **over):
    d = tmp_path / "rfdetr_v1"
    d.mkdir(exist_ok=True)
    cfg = {"variant": "medium", "resolution": 1008, "classes": list(ACTIVE),
           "category_ids": {str(i + 1): c for i, c in enumerate(ACTIVE)}}
    if coco_only:
        cfg.pop("category_ids")
        a = d / "coco" / "train"
        a.mkdir(parents=True, exist_ok=True)
        (a / "_annotations.coco.json").write_text(_json.dumps(
            {"categories": [{"id": i + 1, "name": c}
                            for i, c in enumerate(ACTIVE)]}))
    cfg.update(over)
    (d / "rfdetr_train_config.json").write_text(_json.dumps(cfg))
    (d / "checkpoint_best_total.pth").write_bytes(b"x")
    return d


def _stub(seg_obj, ids, n=None):
    """Attach a fake rfdetr model returning supervision-like detections.

    Also does what load() would do apart from building the network, so the
    class list and the id mapping come from the run config exactly as they do
    in a real run."""
    cfg = seg_obj.sidecar()
    if seg_obj.classes is None:
        seg_obj.classes = list(cfg.get("classes") or [])
    if seg_obj._id_to_name is None:
        seg_obj._id_to_name = seg_obj._category_ids(cfg) or None
    n = len(ids) if n is None else n

    class _Sv:
        mask = _np.ones((n, 8, 8), bool)
        xyxy = _np.tile(_np.array([[1.0, 2.0, 5.0, 7.0]]), (n, 1))
        class_id = _np.asarray(ids, int)
        confidence = _np.linspace(0.9, 0.5, n)
        def __len__(self): return n

    class _M:
        def predict(self, _rgb, threshold=0.5): return _Sv()

    seg_obj._model = _M()
    return seg_obj


def test_the_variant_and_resolution_come_from_the_run_config(tmp_path):
    """Medium weights loaded into Preview is a different architecture; it warns
    about partial loading and predicts noise rather than raising."""
    d = _run_dir(tmp_path)
    s = sg.RFDETRSegmenter(d / "checkpoint_best_total.pth")
    cfg = s.sidecar()
    assert cfg["variant"] == "medium" and cfg["resolution"] == 1008


def test_a_missing_run_config_is_an_error_not_a_default(tmp_path):
    """Every available default is wrong: Preview is the wrong architecture, 432
    is the wrong resolution, and the COCO 80 is the wrong class list."""
    (tmp_path / "loose.pth").write_bytes(b"x")
    s = sg.RFDETRSegmenter(tmp_path / "loose.pth")
    with pytest.raises(SystemExit) as e:
        s.load()
    msg = str(e.value)
    assert "rfdetr_train_config.json" in msg
    assert "crop safety" in msg


def test_explicit_arguments_override_the_run_config(tmp_path):
    d = _run_dir(tmp_path)
    s = sg.RFDETRSegmenter(d / "checkpoint_best_total.pth", variant="nano",
                           resolution=504, classes=["grass_weed"])
    assert s.classes == ["grass_weed"]
    assert s.variant == "nano" and s.resolution == 504


def test_an_unknown_variant_is_refused(tmp_path):
    d = _run_dir(tmp_path, variant="enormous")
    with pytest.raises(SystemExit, match="variant"):
        sg.RFDETRSegmenter(d / "checkpoint_best_total.pth").load()


def test_predictions_carry_the_models_own_class_list(tmp_path):
    """Without names= the Detections falls back to the FULL ontology while the
    model emits indices into the 4-class active list."""
    d = _run_dir(tmp_path)
    s = _stub(sg.RFDETRSegmenter(d / "checkpoint_best_total.pth"), [0, 3])
    det = s(_np.zeros((8, 8, 3), _np.uint8))
    assert det.names == ACTIVE
    assert det.class_name(1) == CROP_CLASS


def test_the_crop_is_still_the_crop_under_the_reduced_class_list(tmp_path):
    """THE regression. ACTIVE drops wild_radish and weed_cluster, so onion sits
    at 3 here and 5 in the ontology. Resolved against the ontology, the crop
    mask would come back empty and every onion would be handed to the laser."""
    assert ACTIVE.index(CROP_CLASS) != CLASSES.index(CROP_CLASS)
    d = _run_dir(tmp_path)
    s = _stub(sg.RFDETRSegmenter(d / "checkpoint_best_total.pth"), [3, 1])
    det = s(_np.zeros((8, 8, 3), _np.uint8))
    assert det.crop_index() == 3
    assert det.onion_safety_mask().any(), "crop mask lost"
    assert det.weed_indices() == [1], "an onion was handed over as a weed"


def test_a_class_id_outside_the_recorded_list_fails_loudly(tmp_path):
    """Silently mislabelling a plant here is a crop-safety failure, so an
    index the class list cannot explain must stop the run."""
    d = _run_dir(tmp_path)
    s = _stub(sg.RFDETRSegmenter(d / "checkpoint_best_total.pth"), [0, 9])
    with pytest.raises(SystemExit, match="outside the 4 classes"):
        s(_np.zeros((8, 8, 3), _np.uint8))


def test_an_empty_prediction_still_reports_the_right_class_list(tmp_path):
    d = _run_dir(tmp_path)
    s = _stub(sg.RFDETRSegmenter(d / "checkpoint_best_total.pth"), [], n=0)
    s._model.predict = lambda *_a, **_k: type(
        "E", (), {"mask": None, "__len__": lambda s: 0})()
    det = s(_np.zeros((8, 8, 3), _np.uint8))
    assert len(det) == 0 and det.names == ACTIVE


def test_eval_seg_can_read_the_class_list_after_load(tmp_path):
    """eval_seg compares seg.classes against the manifest before scoring; an
    absent attribute would crash --backend rfdetr on the first line."""
    d = _run_dir(tmp_path)
    s = sg.RFDETRSegmenter(d / "checkpoint_best_total.pth")
    assert hasattr(s, "classes")


# --------------------------------------------------------------------------- #
# which checkpoint to score
# --------------------------------------------------------------------------- #
def test_the_overall_winner_is_preferred_over_the_ema_file(tmp_path):
    """rfdetr keeps _regular, _ema and _total, copying _total from whichever
    won. Naming the EMA file scores the loser whenever the live weights were
    better - which is the run this project has already seen (regular 0.4498,
    ema 0.4475)."""
    tm = load_script("training/train_model_rfdetr.py")
    d = tmp_path / "run"
    d.mkdir()
    for n in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
              "checkpoint_best_total.pth"):
        (d / n).write_bytes(b"x")
    assert tm._best_checkpoint(d).name == "checkpoint_best_total.pth"


def test_it_falls_back_when_the_total_checkpoint_is_absent(tmp_path):
    tm = load_script("training/train_model_rfdetr.py")
    d = tmp_path / "run"
    d.mkdir()
    (d / "checkpoint_best_ema.pth").write_bytes(b"x")
    assert tm._best_checkpoint(d).name == "checkpoint_best_ema.pth"


# --------------------------------------------------------------------------- #
# the id -> class mapping
#
# rfdetr's docstring promises "a 0-based index into class_names" for a
# fine-tuned model. It is not: its COCO loader remaps sparse category ids to
# contiguous labels for TRAINING and maps them BACK to the original category_id
# at predict time. A model trained on ids 1..4 predicts 1..4, and the first
# real run raised on exactly that.
# --------------------------------------------------------------------------- #
def _mapped(tmp_path, ids, **kw):
    d = _run_dir(tmp_path, **kw)
    s = sg.RFDETRSegmenter(d / "checkpoint_best_total.pth")
    cfg = s.sidecar()
    s.classes = list(cfg["classes"])
    s._id_to_name = s._category_ids(cfg)
    idx, keep = s.map_class_ids(_np.asarray(ids), s.classes)
    return idx, keep


def test_labels_are_zero_based_raw_model_outputs(tmp_path):
    """PostProcess computes `labels = topk_indexes % out_logits.shape[2]`, and
    label2cat is used only by rfdetr's COCO evaluator - never by predict(). So
    the dataset's category ids never appear in a prediction."""
    idx, _ = _mapped(tmp_path, [0, 1, 2, 3])
    assert idx.tolist() == [0, 1, 2, 3]


def test_the_crop_label_maps_to_the_crop(tmp_path):
    idx, _ = _mapped(tmp_path, [3])
    assert ACTIVE[idx[0]] == CROP_CLASS


def test_the_unused_extra_logit_is_dropped_not_treated_as_an_error(tmp_path):
    """num_classes=4 builds a FIVE-output classifier - LW-DETR allocates
    num_classes + 1 - and slot 4 has no training target. It can still win a
    top-k slot at a low threshold, so erroring on it makes the AP sweep, which
    scores down to 0.05, unrunnable."""
    idx, keep = _mapped(tmp_path, [0, 4, 3])
    assert keep.tolist() == [True, False, True]
    assert idx.tolist() == [0, 3]


def test_a_label_beyond_the_extra_slot_still_stops_the_run(tmp_path):
    with pytest.raises(SystemExit, match="outside the 4 classes"):
        _mapped(tmp_path, [9])


def test_labels_follow_ascending_category_id_order(tmp_path):
    """cat2label is {cat_id: i for i, cat_id in enumerate(sorted(cats))}, so
    label i is the i-th category BY ID - not the i-th entry of the class list.
    With ids out of class order the two differ, and only one is right."""
    d = _run_dir(tmp_path, category_ids={"9": "grass_weed", "3": CROP_CLASS})
    s = sg.RFDETRSegmenter(d / "checkpoint_best_total.pth")
    cfg = s.sidecar()
    s.classes, s._id_to_name = list(cfg["classes"]), s._category_ids(cfg)
    assert s.label_order() == [CROP_CLASS, "grass_weed"]     # id 3 then id 9
    idx, _ = s.map_class_ids(_np.asarray([0, 1]), s.classes)
    assert [s.classes[i] for i in idx] == [CROP_CLASS, "grass_weed"]


def test_an_older_run_falls_back_to_its_coco_annotations(tmp_path):
    """A checkpoint trained before category_ids was recorded stays usable: the
    COCO tree in the run directory carries the same information."""
    idx, _ = _mapped(tmp_path, [0, 3], coco_only=True)
    assert idx.tolist() == [0, 3]


def test_no_class_list_anywhere_refuses_to_guess(tmp_path):
    s = sg.RFDETRSegmenter(tmp_path / "loose.pth")
    with pytest.raises(SystemExit, match="no recorded class list"):
        s.map_class_ids(_np.asarray([0]), [])


def test_a_label_naming_a_class_this_model_lacks_is_refused(tmp_path):
    d = _run_dir(tmp_path, category_ids={"1": "bindweed"})
    s = sg.RFDETRSegmenter(d / "checkpoint_best_total.pth")
    cfg = s.sidecar()
    s.classes, s._id_to_name = list(cfg["classes"]), s._category_ids(cfg)
    with pytest.raises(SystemExit, match="not one of this model's classes"):
        s.map_class_ids(_np.asarray([0]), s.classes)


def test_dropping_the_extra_slot_keeps_masks_boxes_and_scores_aligned(tmp_path):
    """The unused slot is removed from the ids AND from everything indexed
    alongside them; a length mismatch would silently pair a mask with another
    instance's class."""
    d = _run_dir(tmp_path)
    s = _stub(sg.RFDETRSegmenter(d / "checkpoint_best_total.pth"), [0, 4, 3])
    det = s(_np.zeros((8, 8, 3), _np.uint8))
    assert len(det) == 2
    assert len(det.masks) == len(det.boxes) == len(det.classes) == 2
    assert {det.class_name(i) for i in range(len(det))} == \
        {ACTIVE[0], CROP_CLASS}


def test_the_export_records_the_ids_it_assigned(tmp_path):
    """coco_export is the only place that knows which id each class got, so it
    is the only honest source for the label order."""
    ds = _dataset(tmp_path, classes=("grass_weed", "onion_plant"))
    s = ce.export(ds, tmp_path / "coco")
    assert s["category_ids"] == {"1": "grass_weed", "2": "onion_plant"}
    coco = json.loads(
        (tmp_path / "coco" / "train" / "_annotations.coco.json").read_text())
    assert {str(c["id"]): c["name"] for c in coco["categories"]} == \
        s["category_ids"]


# --------------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------------- #
def test_the_learning_rate_actually_decays():
    """rfdetr defaults to lr_scheduler='step' with lr_drop=100, so on a
    60-epoch run the step never fires and the LR is constant start to finish -
    which is what the first run did."""
    from rfdetr import config as C
    tc = C.TrainConfig.model_fields
    assert tc["lr_scheduler"].default == "step"
    assert tc["lr_drop"].default > 60, "the default drop is past a normal run"
    tm = load_script("training/train_model_rfdetr.py")
    assert tm.CONFIG["LR_SCHEDULER"] == "cosine"
    assert tm.CONFIG["EPOCHS"] < tc["lr_drop"].default


def test_the_schedule_reaches_the_trainer(tmp_path, monkeypatch):
    seen = {}
    tm = load_script("training/train_model_rfdetr.py")
    import training.train_seg_rfdetr as _t
    monkeypatch.setattr(_t, "train", lambda *a, **k: seen.update(k))
    ds = _dataset(tmp_path)
    c = dict(tm.CONFIG)
    c.update({"DATASET_DIR": str(ds), "IMAGES_ROOT": str(tmp_path),
              "RUN_DIR": str(tmp_path / "run")})
    tm.main(c)
    assert seen["lr_scheduler"] == "cosine"
    assert seen["warmup_epochs"] == 1.0
    assert seen["patience"] == tm.CONFIG["PATIENCE"]


def test_patience_outlasts_the_dead_head_epochs():
    """The re-initialised detection head leaves whole classes at AP 0.000 for
    the first several epochs; patience 10 stopped a 60-epoch run at 23 while a
    class was still improving."""
    tm = load_script("training/train_model_rfdetr.py")
    assert tm.CONFIG["PATIENCE"] >= 20


def test_the_classifier_really_has_one_more_output_than_num_classes():
    """The premise of dropping slot len(classes). If rfdetr ever stops
    allocating num_classes + 1, this test fails and the drop becomes wrong."""
    from rfdetr import RFDETRSegNano
    m = RFDETRSegNano(num_classes=3, resolution=504)
    w = dict(m.model.model.named_parameters())["class_embed.weight"]
    assert w.shape[0] == 4
