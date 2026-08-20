"""Where the data lives. One place, so a moved drive is one edit.

Paths are resolved at RUN time rather than import time: this file is read on
both machines, and the one that is not holding the data should not fail to
import because of it.
"""
from pathlib import Path

#: The parent of every capture campaign. Override with SEEWEED3D_DATA_ROOT when
#: the drive differs - which it does between the annotation machine (E:) and the
#: training machine (D:\LaserWeeding Research...).
import os

DATA_ROOT = Path(os.environ.get(
    "SEEWEED3D_DATA_ROOT",
    r"D:\LaserWeeding Research (Muneeb. E Malik)\Dataset_Vidalia"))


def campaign(name):
    """The `sessions` folder of one capture campaign.

    Campaign-level rather than per-session on purpose: pointing at `sessions`
    discovers every session under it, so a seventh recording or a renamed
    campaign folder is not six paths to edit. A per-session list broke exactly
    that way once already.
    """
    return DATA_ROOT / name / "sessions"


def out(name):
    """Where a built dataset is written."""
    return DATA_ROOT / "datasets" / name


def runs(name):
    """Where a training run writes its checkpoints and figures."""
    return DATA_ROOT / "runs" / name
