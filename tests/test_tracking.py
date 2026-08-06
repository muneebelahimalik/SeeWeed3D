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


# --------------------------------------------------------------------------- #
# provenance: what makes a run explainable months later
# --------------------------------------------------------------------------- #
def test_git_commit_flags_a_dirty_tree(tmp_path, monkeypatch):
    """A bare hash claims reproducibility. If the tree differed from that
    commit, it is a false claim."""
    import subprocess
    calls = {"n": 0}

    class R:
        def __init__(self, out, rc=0):
            self.stdout, self.returncode = out, rc

    def fake_run(args, **kw):
        calls["n"] += 1
        if "rev-parse" in args:
            return R("abc123\n")
        return R(" M some_file.py\n")     # porcelain: uncommitted changes

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tk.git_commit(tmp_path) == "abc123-dirty"


def test_git_commit_is_clean_when_nothing_is_modified(tmp_path, monkeypatch):
    import subprocess

    class R:
        def __init__(self, out, rc=0):
            self.stdout, self.returncode = out, rc

    monkeypatch.setattr(subprocess, "run",
                        lambda args, **kw: R("abc123\n" if "rev-parse" in args
                                             else ""))
    assert tk.git_commit(tmp_path) == "abc123"


def test_git_commit_returns_none_outside_a_repo(tmp_path, monkeypatch):
    import subprocess

    class R:
        def __init__(self):
            self.stdout, self.returncode = "", 128

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: R())
    assert tk.git_commit(tmp_path) is None


def test_git_commit_survives_git_being_absent(tmp_path, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert tk.git_commit(tmp_path) is None


def test_file_digest_changes_when_the_file_changes(tmp_path):
    """OUT_DIR is reused between rebuilds, so a path is not an identity."""
    f = tmp_path / "seg_manifest.json"
    f.write_text('{"frames": []}')
    a = tk.file_digest(f)
    f.write_text('{"frames": [1]}')
    assert a != tk.file_digest(f)


def test_file_digest_is_stable_for_identical_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("same"); b.write_text("same")
    assert tk.file_digest(a) == tk.file_digest(b)


def test_file_digest_of_a_missing_file_is_none(tmp_path):
    assert tk.file_digest(tmp_path / "nope.json") is None


def test_environment_params_records_code_and_runtime():
    p = tk.environment_params()
    assert "python_version" in p and "platform" in p


def test_environment_params_hashes_the_dataset_manifest(tmp_path):
    (tmp_path / "seg_manifest.json").write_text('{"frames": []}')
    p = tk.environment_params(tmp_path)
    assert p["seg_manifest_sha256"]
    assert p["dataset_dir"] == str(tmp_path)


def test_environment_params_omits_what_it_cannot_determine(tmp_path):
    """None values are dropped rather than logged - a param reading 'None'
    is indistinguishable from one genuinely set to the string."""
    p = tk.environment_params(tmp_path)
    assert None not in p.values()


# --------------------------------------------------------------------------- #
# mlflow backend: it must never take the training run with it
# --------------------------------------------------------------------------- #
def test_a_broken_mlflow_backend_disables_itself_instead_of_raising(
        tmp_path, monkeypatch):
    """MLflow 3 refuses the old './mlruns' file store outright. Whatever the
    cause, a backend that cannot start must not end a 3-hour training run."""
    import types
    fake = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("store in maintenance mode")),
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    t = tk.Tracker("mlflow", out_dir=tmp_path)   # explicitly requested
    assert "mlflow" not in t.active              # disabled, not fatal
    t.log_metrics({"loss": 1.0}, step=0)         # still a working no-op
    t.close()


def test_a_missing_mlflow_is_still_fatal_when_explicitly_requested(
        tmp_path, monkeypatch):
    """The one case that must stay loud: silently no-op'ing a backend you
    named would let a run finish while you believed it was logged."""
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "mlflow":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    # Asserts the INTENT (it is fatal, and names the package), not the
    # exact wording - the message now also names the interpreter.
    with pytest.raises(SystemExit, match="mlflow is not installed"):
        tk.Tracker("mlflow", out_dir=tmp_path)


def test_the_hint_prints_the_uri_the_run_was_written_to(tmp_path):
    """Printing the DIRECTORY would send you to a store mlflow 3 refuses."""
    t = tk.Tracker("none", out_dir=tmp_path)
    t.active.append("mlflow")
    t._uri = "sqlite:////x/mlruns/mlflow.db"
    assert "sqlite:////x/mlruns/mlflow.db" in t.hint()
    t.close()


# --------------------------------------------------------------------------- #
# "not installed" must say WHICH interpreter
# --------------------------------------------------------------------------- #
def test_the_missing_package_error_names_the_interpreter(tmp_path, monkeypatch):
    """`pip install X` is not actionable with several interpreters around: a
    prompt reading (sw-train) says nothing about which python an explicit path
    or an IDE's configured interpreter actually launched."""
    import builtins
    import sys as _sys
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "mlflow":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(SystemExit) as e:
        tk.Tracker("mlflow", out_dir=tmp_path)
    msg = str(e.value)
    assert _sys.executable in msg, "must name the interpreter that ran"
    assert "requirements-training.txt" in msg


def test_the_error_warns_against_installing_into_the_sam3_environment(
        tmp_path, monkeypatch):
    """Installing the training stack into `dl` is how SAM 3's numpy pin breaks,
    so the message must not simply say 'pip install' into whatever is current."""
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "mlflow":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(SystemExit) as e:
        tk.Tracker("mlflow", out_dir=tmp_path)
    msg = str(e.value)
    assert "numpy<2" in msg and "sw-train" in msg


def test_tensorboard_gets_the_same_treatment(tmp_path, monkeypatch):
    import builtins
    import sys as _sys
    real = builtins.__import__

    def blocked(name, *a, **k):
        if "tensorboard" in name:
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(SystemExit) as e:
        tk.Tracker("tensorboard", out_dir=tmp_path)
    assert _sys.executable in str(e.value)
