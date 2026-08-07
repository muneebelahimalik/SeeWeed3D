#!/usr/bin/env python3
"""
SeeWeed3D - torch device checks that fail early and say what to do.

`Torch not compiled with CUDA enabled` is an AssertionError raised from deep
inside `Module.to()`, which means it surfaces AFTER the dataset has loaded, the
experiment tracker has opened a run, and pretrained weights have downloaded -
half a minute of work, a stack trace fifteen frames deep, and a stranded
tracking run. Nothing about it names the actual problem, which is that this
environment holds a CPU-only torch build.

The check itself is one line. Doing it BEFORE anything expensive, and turning
it into an instruction, is the point.
"""
from __future__ import annotations

import sys
from pathlib import Path


def describe_interpreter():
    """`<path>  (environment: <name>)`, so a wrong-environment mistake is
    visible. A shell prompt naming a conda env says nothing about which python
    an explicit path or an IDE's configured interpreter actually launched."""
    exe = Path(sys.executable)
    env = exe.parent.name if exe.parent.name != "bin" else exe.parents[1].name
    return f"{exe}\n    (environment: {env})"


def require_device(device):
    """Return `device` if usable, else raise SystemExit explaining the fix.

    Called before model construction, dataset loading or tracker start-up, so
    the failure costs seconds rather than the whole setup."""
    dev = str(device or "cpu")
    # Reject a device string nothing understands BEFORE it reaches a backend.
    # A truncated paste ('--device c' instead of 'cuda' - a real occurrence)
    # otherwise surfaces as a pydantic ValidationError from deep inside rfdetr,
    # minutes later and pointing at a config class rather than the command.
    head = dev.split(":")[0]
    if head not in ("cpu", "cuda", "mps", "xpu", "meta"):
        raise SystemExit(
            f"ERROR: device={dev!r} is not a device. Expected 'cuda', 'cpu', "
            f"or an indexed form such as 'cuda:1'.\n"
            f"    (a truncated '--device cuda' looks exactly like this)")
    if not dev.startswith("cuda"):
        return dev
    try:
        import torch
    except ImportError:
        raise SystemExit(
            f"ERROR: torch is not installed in the interpreter that is "
            f"running:\n    {describe_interpreter()}\n"
            f'    "{sys.executable}" -m pip install -r '
            f"requirements-training.txt")

    if torch.cuda.is_available():
        return dev

    # A CPU-only wheel advertises itself in its version string ('2.5.1+cpu'),
    # which distinguishes "wrong build installed" from "right build, no driver
    # or no GPU visible" - different fixes, and the traceback shows neither.
    version = getattr(torch, "__version__", "unknown")
    cpu_only = "+cpu" in str(version) or getattr(torch.version, "cuda", None) is None

    lines = [
        f"ERROR: device='{dev}' was requested but this torch cannot use CUDA.",
        f"    interpreter: {describe_interpreter()}",
        f"    torch:       {version}",
        "",
    ]
    if cpu_only:
        lines += [
            "This is a CPU-ONLY torch build. Install the CUDA build into THIS "
            "environment",
            "(check your driver's CUDA version with `nvidia-smi` and match the "
            "index URL):",
            "",
            f'    "{sys.executable}" -m pip uninstall -y torch torchvision',
            f'    "{sys.executable}" -m pip install torch torchvision'
            f" --index-url https://download.pytorch.org/whl/cu121",
            "",
        ]
    else:
        lines += [
            "torch has CUDA support but no usable GPU was found. Check "
            "`nvidia-smi` runs,",
            "that the driver is new enough for this torch build, and that no "
            "CUDA_VISIBLE_DEVICES",
            "setting is hiding the card.",
            "",
        ]
    lines += [
        "Or train on the CPU by setting DEVICE = \"cpu\" (train_model.py) or "
        "--device cpu.",
        "Expect roughly an order of magnitude more time per epoch; on a few "
        "dozen frames",
        "that is slow but not impossible.",
    ]
    raise SystemExit("\n".join(lines))
