"""The error you get when DATASET_DIR has no dataset in it.

THE FAILURE THIS PREVENTS
-------------------------
The message used to say:

    Build the dataset first: edit and run seeweed3d/training/make_dataset.py

That is the SHARED runner. Editing it to build one dataset overwrites the
config of every other one - which is the reason training/datasets/ exists at
all. Naming the wrong script in an error is worse than naming none, because the
error is read at exactly the moment someone is willing to do what it says.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_model import (_dataset_runners, _norm,   # noqa: E402
                                  missing_dataset_hint)
from training.datasets.weeds import OUT_DIR as WEEDS_OUT     # noqa: E402


def test_the_weed_runner_is_discovered_with_its_out_dir():
    assert _dataset_runners().get(WEEDS_OUT) == "weeds"


def test_a_known_dataset_names_the_runner_that_builds_it():
    hint = missing_dataset_hint(WEEDS_OUT)
    assert "python -m seeweed3d.training.datasets.weeds" in hint


def test_it_never_points_at_the_shared_runner():
    """The whole point of the change."""
    for path in (WEEDS_OUT, r"E:\somewhere\else"):
        assert "make_dataset" not in missing_dataset_hint(path)


def test_an_unknown_path_says_nothing_will_ever_build_it():
    """The more interesting failure: a DATASET_DIR matching no OUT_DIR is a typo
    or a stale edit, and it will not fix itself by running anything."""
    hint = missing_dataset_hint(r"E:\Dataset_Vidalia\datasets\weeds_v99")
    assert "nothing will ever build it" in hint
    assert "drifted" in hint


def test_an_unknown_path_lists_the_builds_that_do_exist():
    """Turns a dead end into a one-line fix."""
    hint = missing_dataset_hint(r"E:\nope")
    assert "weeds" in hint and WEEDS_OUT in hint


def test_windows_and_posix_separators_compare_equal():
    """The configs hold Windows paths; the tests and CI run on posix. A hint
    that only matched on one would be silently useless on the other."""
    assert _norm(r"E:\a\b") == _norm("E:/a/b") == _norm("E:/a/b/")


def test_case_does_not_decide_the_match():
    assert _norm(r"E:\Dataset\Weeds") == _norm(r"e:\dataset\weeds")


def test_train_and_mine_runners_are_not_offered_as_builders():
    """weeds_train and weeds_mine READ an OUT_DIR from the build; they do not
    define one. Offering them here would send someone to run a trainer to fix a
    missing dataset."""
    assert set(_dataset_runners().values()) == {"weeds"}


def test_discovery_does_not_import_the_runners(monkeypatch):
    """Read with the AST on purpose. Importing would execute config modules to
    fetch one constant, and weeds_train imports the very module that raises this
    error - a circular import triggered only on the error path."""
    import builtins
    real = builtins.__import__

    def boom(name, *a, **kw):
        if "datasets.weeds" in name:
            raise AssertionError(f"discovery imported {name}")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert _dataset_runners()


def test_both_trainers_use_the_hint():
    """The old literal is gone from BOTH raise sites. Matched on the exact
    former message rather than on the words "edit and run", which also appear
    in the docstring explaining why they were removed."""
    for mod in ("train_model.py", "train_model_rfdetr.py"):
        src = (ROOT / "training" / mod).read_text(encoding="utf-8")
        assert "missing_dataset_hint" in src, mod
        assert "Build the dataset first: edit and run " not in src, mod
