#!/usr/bin/env python3
"""SeeWeed3D - saying which environment you are in when `sam3` is missing.

THE FAILURE THIS PREVENTS
-------------------------
This project runs in TWO conda environments, and they are not interchangeable:

    dl        SAM 3 prelabeling  - needs the `sam3` package
    sw-train  training and eval  - needs rfdetr, lightning, mlflow

Running the prelabeler from the training environment raises a bare
`ModuleNotFoundError: No module named 'sam3'`, which is true and useless. It
reads as "SAM is not installed", so the obvious response is to install it - into
the wrong environment, where it will then shadow the question forever.

The information that resolves it in one second is which environment is ACTIVE,
and Python knows that. So say it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Which environment each capability lives in. Names are the ones actually used
#: on the annotation/training machine; the message degrades gracefully if a
#: different set is in use, because it always reports what IS active.
SAM_ENV = "dl"
TRAIN_ENV = "sw-train"


def active_env():
    """Best-effort name of the active conda/venv environment, or ""."""
    name = os.environ.get("CONDA_DEFAULT_ENV") or ""
    if name:
        return name
    prefix = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX") or ""
    return Path(prefix).name if prefix else ""


def sam_import_error(exc, module="sam3"):
    """A ModuleNotFoundError worth reading. Returns the replacement exception.

    Names the active environment first, because that is the fact that decides
    what to do next: in the TRAINING environment the answer is to switch, and
    everywhere else it is to install."""
    env = active_env()
    where = f"the active environment is {env!r}" if env else \
        f"no conda/venv environment is active (python at {sys.executable})"

    if env == TRAIN_ENV:
        fix = (f"That is the TRAINING environment. SAM 3 lives in {SAM_ENV!r}:\n"
               f"        conda activate {SAM_ENV}\n"
               f"    Do NOT install sam3 into {TRAIN_ENV!r} to make this go "
               f"away - the two environments are separate on purpose, and a "
               f"second copy is one more thing to keep in step.")
    else:
        fix = (f"Install it into this environment:\n"
               f'        pip install "git+https://github.com/facebookresearch/'
               f'sam3.git"')
        if env != SAM_ENV:
            # Only worth suggesting when it is somewhere else. Telling someone
            # already in the SAM environment to activate the SAM environment
            # reads as a broken message and gets the whole thing ignored.
            fix += (f"\n    Or switch to the one that is meant to have it:\n"
                    f"        conda activate {SAM_ENV}")

    err = ModuleNotFoundError(
        f"{module} is not importable, and {where}.\n"
        f"    {fix}\n"
        f"    Prelabeling needs {SAM_ENV!r}; training and evaluation need "
        f"{TRAIN_ENV!r}.")
    # `raise X from exc` is a statement, so a function that RETURNS the
    # exception has to attach the cause itself.
    err.__cause__ = exc
    err.name = module
    return err
