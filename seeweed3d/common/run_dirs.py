"""One output folder per run, named for when it ran.

Every runner here derived its output folder from ROUND and the session name, so
two runs of the same script wrote to the same place and the second silently
overwrote the first. That is cheap when the output is overlays and expensive
when it is a CVAT batch you have half-corrected, or the predictions a set of
pseudo-labels was scored from.

A timestamp rather than a counter, because the question anyone actually asks of
an old folder is "when did this run", and `_2` does not answer it. Minute
resolution: two runs of a multi-minute GPU pass cannot collide, and a name a
person has to read aloud on a call should not carry seconds.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

#: Sorts lexicographically as well as chronologically, which is what makes
#: `newest` a max() over names rather than a stat() of every candidate.
STAMP = "%Y%m%d_%H%M"

_SUFFIX = re.compile(r"_\d{8}_\d{4}(_\d+)?$")


def stamped(parent, stem, when=None):
    """`parent/stem_YYYYmmdd_HHMM`, guaranteed not to already exist.

    The collision tail is for the case a minute cannot separate - a re-run
    started immediately after a crash, or a test. It is deliberately ugly so it
    reads as an exception rather than as part of the scheme."""
    when = when or datetime.now()
    base = f"{stem}_{when.strftime(STAMP)}"
    parent = Path(parent)
    cand = parent / base
    n = 2
    while cand.exists():
        cand = parent / f"{base}_{n}"
        n += 1
    return str(cand)


def newest(parent, stem):
    """The most recent `stem_<stamp>` under `parent`, or None.

    Used for the one folder that is meant to be REUSED - predictions, so that
    re-scoring at a different threshold does not re-run the GPU. Everything
    else gets a fresh `stamped` folder."""
    parent = Path(parent)
    if not parent.is_dir():
        return None
    hits = [d for d in parent.iterdir()
            if d.is_dir() and d.name.startswith(stem + "_")
            and _SUFFIX.search(d.name[len(stem):])]
    return str(max(hits, key=lambda d: d.name)) if hits else None


def is_stamped(path):
    """Whether a path already carries a run stamp."""
    return bool(_SUFFIX.search(Path(path).name))


def stale_predictions_warning(pred_dir, checkpoint):
    """A warning when predictions being reused predate the model.

    Reuse is the point of `newest`, and it is also how an old model's output
    gets scored and written back into the training set as pseudo-labels - which
    is worse than useless, because the round then teaches the model to agree
    with a version of itself it has already improved on. Retraining a round
    leaves the predictions untouched and correct-looking, so nothing about the
    folder says it is stale except its mtime."""
    pred = Path(pred_dir) / "instances_default.json"
    ckpt = Path(checkpoint)
    if not pred.exists() or not ckpt.exists():
        return None
    if pred.stat().st_mtime >= ckpt.stat().st_mtime:
        return None
    fmt = "%Y-%m-%d %H:%M"
    return (f"  [!] these predictions are OLDER than the checkpoint.\n"
            f"      predictions {datetime.fromtimestamp(pred.stat().st_mtime):{fmt}}"
            f"  {pred_dir}\n"
            f"      checkpoint  {datetime.fromtimestamp(ckpt.stat().st_mtime):{fmt}}"
            f"  {checkpoint}\n"
            f"      They came from an earlier model. Delete or rename that "
            f"folder to force a fresh inference pass - scoring them would feed "
            f"the old model's mistakes back in as pseudo-labels.")
