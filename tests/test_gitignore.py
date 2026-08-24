"""What must never be committed.

THE FAILURE THIS PREVENTS
-------------------------
Nothing in this repo stops a `git add -A` from committing whatever a training
run left in the working tree. It has already happened twice:

  * `mlruns/` reached a commit carrying a 3.8 MB report.html - the largest
    tracked file in the repo - because tracking.py writes its store beside the
    run and .gitignore said nothing about it.
  * a mistyped shell redirect created a file named after the text that followed
    it, and one of those reached a commit too.

Every one of these is REGENERABLE from a run directory in one command, and none
of it is reviewable in a diff. A checkpoint is worse: at 150-600 MB it is
committed once and carried in the history forever.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def ignored(path):
    """Does git consider `path` ignored? Asks git, so the answer matches what
    would actually happen rather than what a pattern looks like it does."""
    r = subprocess.run(["git", "check-ignore", "-q", path],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


@pytest.mark.parametrize("path", [
    "mlruns/1/abc/artifacts/report.html",
    "mlflow.db",
    "runs/weeds_r0/checkpoint_best_total.pth",
    "tb/events.out.tfevents.123",
    "weeds_r0/events.out.tfevents.1787245391.PP-15PXG04.157636.0",
])
def test_experiment_output_is_ignored(path):
    """Regenerable from a run directory, and unreviewable in a diff."""
    assert ignored(path), f"{path} would be committed by `git add -A`"


@pytest.mark.parametrize("path", [
    "checkpoint_best_total.pth",
    "last.ckpt",
    "model.onnx",
    "model.engine",
    "best.pt",
])
def test_model_weights_are_ignored(path):
    """150-600 MB each. Committed once, carried in the history forever."""
    assert ignored(path)


@pytest.mark.parametrize("path", ["tatus", "status --short", "0.85", "0.5"])
def test_shell_redirect_mishaps_are_ignored(path):
    """A mistyped redirect names the file after whatever followed it: a
    truncated subcommand, or a bare numeric argument value."""
    assert ignored(path)


def test_source_and_docs_are_not_ignored():
    """A broad pattern that swallowed real files would be worse than the
    problem - `[0-9]*` would have matched nothing here today and plenty later."""
    for path in ("seeweed3d/training/pseudo_label.py", "docs/training.md",
                 "README.md", "tests/test_gitignore.py",
                 "seeweed3d/training/datasets/weeds.py"):
        assert not ignored(path), f"{path} is ignored and must not be"


def test_nothing_regenerable_is_currently_tracked():
    """The check that would have caught the 3.8 MB report.html."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    bad = [f for f in out
           if f.startswith(("mlruns/", "runs/", "tb/"))
           or f.endswith((".pth", ".ckpt", ".onnx", ".engine", ".tfevents"))
           or f == "mlflow.db"]
    assert not bad, f"regenerable artifacts are tracked: {bad[:5]}"
