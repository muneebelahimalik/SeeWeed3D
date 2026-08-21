"""The message you get when `sam3` is missing.

THE FAILURE THIS PREVENTS
-------------------------
This project runs in two conda environments - `dl` for SAM 3 prelabeling and
`sw-train` for training and evaluation - and running the prelabeler from the
training one raises a bare:

    ModuleNotFoundError: No module named 'sam3'

which is true and useless. It reads as "SAM is not installed", so the obvious
response is to install it into whichever environment is active. That is usually
the environment that should not have it, and the second copy then hides the
question permanently.

The fact that resolves it in one second is which environment is ACTIVE, and
Python knows that.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import sam_env                                   # noqa: E402


@pytest.fixture
def env(monkeypatch):
    def _set(name):
        monkeypatch.setenv("CONDA_DEFAULT_ENV", name)
    return _set


def message(exc=None):
    return str(sam_env.sam_import_error(exc or ModuleNotFoundError("sam3")))


def test_it_names_the_active_environment(env):
    env("sw-train")
    assert "'sw-train'" in message()


def test_the_training_environment_is_told_to_switch_not_install(env):
    """The whole point. In sw-train the answer is `conda activate dl`, and
    installing sam3 there is the thing that must NOT be suggested."""
    env("sw-train")
    m = message()
    assert "conda activate dl" in m
    assert "pip install" not in m


def test_the_training_environment_is_told_not_to_install_a_second_copy(env):
    env("sw-train")
    assert "Do NOT install" in message()


def test_another_environment_is_told_how_to_install(env):
    env("some-other-env")
    m = message()
    assert "pip install" in m
    assert "facebookresearch/sam3" in m


def test_the_sam_environment_is_not_told_to_activate_itself(env):
    """Telling someone already in `dl` to `conda activate dl` reads as a broken
    message and gets the whole thing ignored."""
    env("dl")
    m = message()
    assert "pip install" in m
    assert "conda activate" not in m


def test_no_environment_at_all_still_gets_a_useful_message(monkeypatch):
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    m = message()
    assert "no conda/venv environment is active" in m
    assert sys.executable in m


def test_a_virtualenv_is_recognised_when_conda_is_absent(monkeypatch):
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", "/home/x/.venvs/weeds")
    assert sam_env.active_env() == "weeds"


def test_both_environments_are_always_named(env):
    """Whatever branch it takes, the reader should end up knowing which
    environment each half of the pipeline needs."""
    for name in ("sw-train", "dl", "random", ""):
        env(name)
        m = message()
        assert "'dl'" in m and "'sw-train'" in m


def test_the_cause_is_preserved(env):
    """`raise X from exc` is a statement, so a function that RETURNS the
    exception has to attach the cause itself - it was silently dropped once."""
    env("sw-train")
    original = ModuleNotFoundError("No module named 'sam3'")
    err = sam_env.sam_import_error(original)
    assert err.__cause__ is original
    assert err.name == "sam3"


# --------------------------------------------------------------------------- #
# Both prelabelers route through it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["annotation.prelabel_weeds_sam3",
                                    "annotation.prelabel_onions_sam3"])
def test_load_sam3_raises_the_helpful_error(module, env, monkeypatch):
    import importlib
    env("sw-train")
    mod = importlib.import_module(module)
    # No sam3 in this environment either, so the real import path is exercised.
    with pytest.raises(ModuleNotFoundError) as e:
        mod.load_sam3(dict(mod.CONFIG, DEVICE="cpu"))
    assert "conda activate dl" in str(e.value)


@pytest.mark.parametrize("module", ["annotation.prelabel_weeds_sam3",
                                    "annotation.prelabel_onions_sam3"])
def test_an_unrelated_missing_module_is_not_reblamed_on_sam3(module, env,
                                                             monkeypatch):
    """If sam3 imports fine but one of ITS dependencies is missing, saying
    'sam3 is not importable, switch environments' sends you the wrong way."""
    import importlib
    env("sw-train")
    mod = importlib.import_module(module)
    real = __import__

    def fake(name, *a, **kw):
        if name.startswith("sam3"):
            raise ModuleNotFoundError("No module named 'flash_attn'",
                                      name="flash_attn")
        return real(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fake)
    with pytest.raises(ModuleNotFoundError) as e:
        mod.load_sam3(dict(mod.CONFIG, DEVICE="cpu"))
    assert "flash_attn" in str(e.value)
    assert "conda activate" not in str(e.value)
