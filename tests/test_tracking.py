"""Experiment tracking: no-op safety, explicit-backend failure, overlays.

The contract that matters: tracking must never be able to kill a training run,
EXCEPT when the user explicitly named a backend that is not installed - in that
case failing loudly beats finishing a run that was silently never logged.
"""
import json

import numpy as np
import pytest

from conftest import load_script

tk = load_script("training/tracking.py")


def test_none_backend_is_a_working_no_op(tmp_path):
    t = tk.Tracker("none", out_dir=tmp_path)
    t.log_params({"lr": 0.01, "classes": ["a", "b"]})
    t.log_metrics({"loss": 1.0}, step=0)
    t.log_image("x", np.zeros((4, 4, 3), np.uint8), 0)
    t.log_artifact(tmp_path / "missing.pt")
    t.close()
    assert t.active == []


def test_close_is_idempotent(tmp_path):
    t = tk.Tracker("none", out_dir=tmp_path)
    t.close(); t.close()


def test_auto_never_raises_even_with_nothing_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(tk, "_available", lambda m: False)
    t = tk.Tracker("auto", out_dir=tmp_path)
    assert t.active == []
    t.close()


def test_explicitly_requested_missing_backend_fails_loudly(tmp_path,
                                                           monkeypatch):
    """The whole point: a silent downgrade would let a run finish unlogged."""
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "mlflow":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(SystemExit, match="mlflow"):
        tk.Tracker("mlflow", out_dir=tmp_path)


def test_params_are_written_to_disk_regardless_of_backend(tmp_path):
    t = tk.Tracker("none", out_dir=tmp_path)
    t.log_params({"lr": 0.01, "nested": {"a": 1}, "classes": ["x", "y"]})
    t.close()
    d = json.loads((tmp_path / "params.json").read_text())
    assert d["lr"] == 0.01
    assert d["nested.a"] == 1, "nested config must flatten to dotted keys"
    assert json.loads(d["classes"]) == ["x", "y"]


def test_none_and_nan_metrics_are_dropped_not_logged_as_zero(tmp_path):
    """A missing metric plotted as 0.0 reads as a catastrophic result."""
    t = tk.Tracker("none", out_dir=tmp_path)
    seen = {}
    t._tb = type("W", (), {"add_scalar": lambda s, k, v, st: seen.update({k: v})})()
    t.log_metrics({"a": 1.0, "b": None, "c": float("nan")}, step=0)
    assert seen == {"a": 1.0}


def test_invalid_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        tk.Tracker("wandb", out_dir=tmp_path)


def test_hint_names_a_command_when_nothing_is_active(tmp_path):
    t = tk.Tracker("none", out_dir=tmp_path)
    assert "pip install" in t.hint()


# --------------------------------------------------------------------------- #
# overlays
# --------------------------------------------------------------------------- #
def test_overlay_distinguishes_crop_from_weed_by_colour():
    bgr = np.zeros((20, 20, 3), np.uint8)
    m = np.zeros((20, 20), bool); m[5:15, 5:15] = True
    crop = tk.overlay_masks(bgr, [m], ["onion_plant"], "onion_plant")
    weed = tk.overlay_masks(bgr, [m], ["grass_weed"], "onion_plant")
    assert not np.array_equal(crop, weed)


def test_overlay_leaves_unmasked_pixels_untouched():
    bgr = np.full((20, 20, 3), 77, np.uint8)
    m = np.zeros((20, 20), bool); m[5:10, 5:10] = True
    out = tk.overlay_masks(bgr, [m], ["grass_weed"], "onion_plant")
    assert (out[15:, 15:] == 77).all()


def test_overlay_handles_an_empty_mask():
    bgr = np.zeros((20, 20, 3), np.uint8)
    out = tk.overlay_masks(bgr, [np.zeros((20, 20), bool)], ["grass_weed"],
                           "onion_plant")
    assert out.shape == bgr.shape


def test_side_by_side_puts_the_two_panels_next_to_each_other():
    a = np.zeros((30, 40, 3), np.uint8)
    b = np.zeros((30, 40, 3), np.uint8)
    p = tk.side_by_side(a, b)
    assert p.shape[1] > 40 * 2, "panels must be side by side, not blended"
    assert p.shape[0] == 30


def test_side_by_side_downscales_a_wide_pair():
    a = np.zeros((100, 2000, 3), np.uint8)
    p = tk.side_by_side(a, a, max_width=800)
    assert p.shape[1] == 800
