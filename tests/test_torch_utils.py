"""Device checks that fail early and say what to do.

`Torch not compiled with CUDA enabled` is an AssertionError from fifteen frames
inside Module.to(), raised only after the dataset loaded, the tracker opened a
run and pretrained weights downloaded. The check is trivial; doing it first and
turning it into an instruction is the whole value.
"""
import sys
import types

import pytest

from conftest import load_script

tu = load_script("common/torch_utils.py")


def _fake_torch(monkeypatch, available, version="2.5.1+cpu", cuda_build=None):
    fake = types.SimpleNamespace(
        __version__=version,
        cuda=types.SimpleNamespace(is_available=lambda: available),
        version=types.SimpleNamespace(cuda=cuda_build),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_cpu_is_always_allowed(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    assert tu.require_device("cpu") == "cpu"


def test_none_defaults_to_cpu_and_is_allowed(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    assert tu.require_device(None) == "cpu"


def test_cuda_passes_through_when_available(monkeypatch):
    _fake_torch(monkeypatch, available=True, version="2.5.1+cu121",
                cuda_build="12.1")
    assert tu.require_device("cuda") == "cuda"


def test_an_indexed_device_is_preserved(monkeypatch):
    _fake_torch(monkeypatch, available=True, version="2.5.1+cu121",
                cuda_build="12.1")
    assert tu.require_device("cuda:1") == "cuda:1"


def test_a_cpu_only_build_is_named_as_such(monkeypatch):
    """'Wrong build installed' and 'right build, no GPU visible' have
    different fixes, and the traceback shows neither."""
    _fake_torch(monkeypatch, available=False, version="2.5.1+cpu")
    with pytest.raises(SystemExit) as e:
        tu.require_device("cuda")
    msg = str(e.value)
    assert "CPU-ONLY torch build" in msg
    assert "download.pytorch.org/whl/cu121" in msg


def test_a_cuda_build_with_no_gpu_gives_the_other_diagnosis(monkeypatch):
    _fake_torch(monkeypatch, available=False, version="2.5.1+cu121",
                cuda_build="12.1")
    with pytest.raises(SystemExit) as e:
        tu.require_device("cuda")
    msg = str(e.value)
    assert "no usable GPU" in msg
    assert "nvidia-smi" in msg
    assert "CPU-ONLY" not in msg


def test_the_error_names_the_interpreter(monkeypatch):
    """With two conda environments in play, the fix is often to switch rather
    than to install."""
    _fake_torch(monkeypatch, available=False)
    with pytest.raises(SystemExit) as e:
        tu.require_device("cuda")
    assert sys.executable in str(e.value)


def test_the_error_offers_the_cpu_fallback(monkeypatch):
    """On a few dozen frames, CPU is slow but not impossible - worth saying
    rather than leaving someone blocked on a driver install."""
    _fake_torch(monkeypatch, available=False)
    with pytest.raises(SystemExit) as e:
        tu.require_device("cuda")
    assert 'DEVICE = "cpu"' in str(e.value)


def test_missing_torch_is_reported_before_anything_else(monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "torch":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(SystemExit, match="torch is not installed"):
        tu.require_device("cuda")


def test_describe_interpreter_includes_the_environment_name():
    d = tu.describe_interpreter()
    assert sys.executable in d and "environment:" in d
