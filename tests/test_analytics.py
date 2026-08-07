"""Figures and the run analyser.

The point of this module is that NO tracker draws these. W&B, MLflow and
TensorBoard all store scalars and images; none of them knows what a missed
onion is, or that recall at conf 0.5 and 0.25 are different questions. So the
figures are computed here and the tracker only files them - which is also why
swapping the tracker changes one call and none of this.
"""
import csv
import json

import numpy as np
import pytest

from conftest import load_script

pl = load_script("evaluation/plots.py")
ar = load_script("evaluation/analyze_run.py")
from seeweed3d.common.ontology import CROP_CLASS  # noqa: E402


def _history(n=12, eval_every=2):
    """Loss every epoch, val metrics only on evaluated epochs - the shape both
    trainers actually produce."""
    h = []
    for e in range(n):
        row = {"epoch": e, "train_loss": 2.0 - e * 0.1,
               "val_loss": 1.8 - e * 0.08}
        if e % eval_every == 1:
            row.update({"val_map50": 0.4 + e * 0.02,
                        "val_map50_95": 0.2 + e * 0.015,
                        "missed_onion_fraction": 0.3 - e * 0.01,
                        "weed_on_crop_fraction": 0.001})
        h.append(row)
    return h


def _metrics():
    return {
        "summary": {"split": "val", "n_frames": 16,
                    "classes": ["grass_weed", CROP_CLASS],
                    "map50": 0.7, "map50_95": 0.45,
                    "classes_without_ground_truth": []},
        "detection": {"grass_weed": {"n_gt": 19, "ap50": 0.74,
                                     "ap50_95": 0.55},
                      CROP_CLASS: {"n_gt": 125, "ap50": 0.88,
                                   "ap50_95": 0.48}},
        "operating_point": {"conf": 0.25, "small_weed_recall": 0.73,
                            "small_weed_n": 112},
        "crop_safety": {"missed_onion_fraction": 0.17,
                        "weed_on_crop_fraction": 0.0001,
                        "frames_with_onion": 7},
        "conf_sweep": [
            {"conf": c, "small_weed_recall": r, "weed_recall": r,
             "weed_precision": 1 - r, "crop_recall": 0.92,
             "missed_onion_fraction": 0.17, "weed_on_crop_fraction": b}
            for c, r, b in ((0.15, 0.82, 0.00044), (0.25, 0.73, 0.00010),
                            (0.5, 0.28, 0.00005), (0.7, 0.15, 0.0))],
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def test_every_figure_is_written(tmp_path):
    figs = pl.figures_for_run(_history(), _metrics(),
                              [{"range_px": "0-500", "n_gt": 40, "n_found": 12,
                                "recall": 0.3}], tmp_path)
    assert set(figs) == {"training_curves", "per_class_ap", "confidence_sweep",
                         "crop_safety", "recall_by_size"}
    for p in figs.values():
        assert p.exists() and p.stat().st_size > 1000


def test_a_figure_with_no_data_is_omitted_not_drawn_empty(tmp_path):
    """A blank axis in a report reads as a measured zero."""
    figs = pl.figures_for_run([], {"detection": {}, "conf_sweep": []}, [],
                              tmp_path)
    assert figs == {}
    assert not list(tmp_path.glob("*.png"))


def test_a_single_point_sweep_draws_nothing(tmp_path):
    """One threshold is not a sweep, and a one-point line implies a trend that
    was never measured."""
    m = _metrics()
    m["conf_sweep"] = [{"conf": 0.5}]
    assert pl.confidence_sweep(m, tmp_path / "x.png") is None


def test_val_metrics_are_plotted_against_their_real_epochs(tmp_path):
    """Evaluation runs every N epochs, so those series are shorter than the
    loss curve. Plotted against range() they would land on the wrong epochs."""
    h = _history(n=12, eval_every=4)
    xs, ys = pl._series(h, "val_map50")
    assert xs == [1.0, 5.0, 9.0]
    assert len(ys) == 3
    assert pl._series(h, "train_loss")[0][:3] == [0.0, 1.0, 2.0]


def test_missing_and_nan_values_are_skipped(tmp_path):
    h = [{"epoch": 0, "val_map50": 0.5}, {"epoch": 1},
         {"epoch": 2, "val_map50": float("nan")},
         {"epoch": 3, "val_map50": None}, {"epoch": 4, "val_map50": 0.6}]
    xs, ys = pl._series(h, "val_map50")
    assert xs == [0.0, 4.0] and ys == [0.5, 0.6]


def test_one_broken_figure_does_not_lose_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "per_class_ap",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    figs = pl.figures_for_run(_history(), _metrics(), [], tmp_path)
    assert "per_class_ap" not in figs
    assert "training_curves" in figs


# --------------------------------------------------------------------------- #
# backend detection
# --------------------------------------------------------------------------- #
def test_a_maskrcnn_run_is_detected_from_its_checkpoint(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"x")
    backend, ckpt = ar.detect(tmp_path)
    assert backend == "maskrcnn" and ckpt.name == "best.pt"


def test_an_rfdetr_run_is_detected_and_prefers_the_overall_winner(tmp_path):
    (tmp_path / "rfdetr_train_config.json").write_text("{}")
    for n in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth"):
        (tmp_path / n).write_bytes(b"x")
    backend, ckpt = ar.detect(tmp_path)
    assert backend == "rfdetr" and ckpt.name == "checkpoint_best_total.pth"


def test_an_rfdetr_run_with_no_checkpoint_says_training_may_not_have_finished(
        tmp_path):
    (tmp_path / "rfdetr_train_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="Did training finish"):
        ar.detect(tmp_path)


def test_an_empty_directory_names_both_layouts(tmp_path):
    with pytest.raises(SystemExit, match="best.pt"):
        ar.detect(tmp_path)


def test_a_missing_directory_says_so(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        ar.detect(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# history, from either backend
# --------------------------------------------------------------------------- #
def test_maskrcnn_history_loads_verbatim(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(_history(4)))
    h = ar.load_history(tmp_path, "maskrcnn")
    assert len(h) == 4 and h[0]["train_loss"] == 2.0


def test_rfdetr_train_and_val_rows_are_merged_by_epoch(tmp_path):
    """Lightning writes one row per logging event with blanks elsewhere, so
    train and val land on SEPARATE rows for the same epoch. Left unmerged
    every curve is half gaps."""
    cols = ["epoch", "step", "train/loss", "val/loss", "val/segm_mAP_50",
            "val/segm_mAP_50_95", "val/AP/grass_weed"]
    rows = [
        {"epoch": 0, "step": 4, "train/loss": "", "val/loss": "1.5",
         "val/segm_mAP_50": "0.4", "val/segm_mAP_50_95": "0.2",
         "val/AP/grass_weed": "0.35"},
        {"epoch": 0, "step": 4, "train/loss": "2.1", "val/loss": "",
         "val/segm_mAP_50": "", "val/segm_mAP_50_95": "",
         "val/AP/grass_weed": ""},
        {"epoch": 1, "step": 9, "train/loss": "", "val/loss": "1.2",
         "val/segm_mAP_50": "0.5", "val/segm_mAP_50_95": "0.3",
         "val/AP/grass_weed": "0.45"},
        {"epoch": 1, "step": 9, "train/loss": "1.8", "val/loss": "",
         "val/segm_mAP_50": "", "val/segm_mAP_50_95": "",
         "val/AP/grass_weed": ""},
    ]
    with (tmp_path / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)

    h = ar.load_history(tmp_path, "rfdetr")
    assert len(h) == 2, "one row per epoch, not one per logging event"
    assert h[0]["train_loss"] == 2.1 and h[0]["val_loss"] == 1.5
    assert h[0]["val_map50"] == 0.4
    assert h[0]["ap_grass_weed"] == 0.35
    assert [r["epoch"] for r in h] == [0, 1]


def test_rfdetr_history_uses_mask_ap_not_box_ap(tmp_path):
    """eval_seg reports MASK AP. Pairing a box curve with a mask number would
    make the two backends look different for the wrong reason."""
    assert ar.RFDETR_COLUMNS["val/segm_mAP_50_95"] == "val_map50_95"
    assert "val/mAP_50_95" not in ar.RFDETR_COLUMNS


def test_a_run_with_no_history_is_not_an_error(tmp_path):
    assert ar.load_history(tmp_path, "maskrcnn") == []
    assert ar.load_history(tmp_path, "rfdetr") == []


def test_the_figures_survive_a_real_rfdetr_history(tmp_path):
    """The merge and the plotting have to agree about the row shape."""
    cols = ["epoch", "train/loss", "val/loss", "val/segm_mAP_50",
            "val/segm_mAP_50_95"]
    with (tmp_path / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for e in range(6):
            w.writerow({"epoch": e, "train/loss": 2 - e * 0.2, "val/loss": "",
                        "val/segm_mAP_50": "", "val/segm_mAP_50_95": ""})
            w.writerow({"epoch": e, "train/loss": "", "val/loss": 1.9 - e * 0.1,
                        "val/segm_mAP_50": 0.4 + e * 0.03,
                        "val/segm_mAP_50_95": 0.2 + e * 0.02})
    h = ar.load_history(tmp_path, "rfdetr")
    figs = pl.figures_for_run(h, _metrics(), [], tmp_path / "figs")
    assert "training_curves" in figs


# --------------------------------------------------------------------------- #
# the trainer hook
# --------------------------------------------------------------------------- #
def test_a_failed_analysis_never_kills_a_finished_run(tmp_path, capsys):
    """The weights and the metrics are already on disk by the time this runs.
    analyze() raises SystemExit for a missing run directory, and SystemExit is
    a BaseException - `except Exception` would sail straight past it and lose
    a run that had finished training."""
    tm = load_script("training/train_model.py")
    tm._analyze(tmp_path / "nothing", tmp_path, {"EVAL_SPLIT": "val",
                                                 "DEVICE": "cpu",
                                                 "EVAL_CONF": 0.5})
    out = capsys.readouterr().out
    assert "analysis skipped" in out
    assert "analyze_run.py" in out, "it must say how to retry"


def test_the_rfdetr_runner_survives_a_failed_analysis(tmp_path, monkeypatch):
    tm = load_script("training/train_model_rfdetr.py")
    import training.train_seg_rfdetr as _t
    monkeypatch.setattr(_t, "train", lambda *a, **k: None)
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "seg_manifest.json").write_text(json.dumps(
        {"images_root": [str(tmp_path)], "classes": ["grass_weed"],
         "frames": []}))
    c = dict(tm.CONFIG)
    c.update({"DATASET_DIR": str(ds), "IMAGES_ROOT": str(tmp_path),
              "RUN_DIR": str(tmp_path / "run")})
    tm.main(c)          # must return normally despite no checkpoint to analyse
