#!/usr/bin/env python3
"""SeeWeed3D - reading a model's raw mask output without guessing at it.

THE FAILURE THIS PREVENTS
-------------------------
`arr.astype(bool)` is True for every non-zero value. On a genuinely binary mask
that is correct and free. On anything else it is a threshold nobody chose:

  * probabilities in [0, 1]  -> every pixel above 0.0 becomes foreground, so a
    mask inflates outward through its own soft edge into low-probability
    background. The result still looks like a plant, just consistently too big
    in every direction - which is exactly what a bloated boundary looks like,
    and exactly what nothing raises an error about.
  * logits (roughly -20..20) -> the same, at a threshold of 0.0 that happens to
    be right for logits by luck rather than by decision.

A silent wrong answer beats a loud one, so this module makes the encoding an
explicit, reported decision. It never guesses in silence: an array it cannot
classify keeps the old behaviour AND says so.

WHY IT MATTERS HERE
-------------------
These masks become CVAT prelabels, prelabels become the training target, and
the training target is the ceiling on what any model trained from it can
produce. A boundary that is systematically a few pixels too generous teaches
every later model to be systematically a few pixels too generous.
"""

from __future__ import annotations

import numpy as np

#: The array is already 0/1 (or bool). Nothing to decide.
BINARY = "binary"
#: Values in [0, 1]. The correct cut is 0.5, NOT "anything non-zero".
PROBABILITY = "probability"
#: Signed scores; foreground is > 0. The cut is 0.0 - by decision, not luck.
LOGIT = "logit"
#: Non-negative and above 1, so neither a probability nor obviously a logit.
#: Old behaviour is kept and reported rather than a threshold being invented.
AMBIGUOUS = "ambiguous"


def mask_encoding(arr):
    """What the numbers in a raw mask array mean.

    Classification is on the VALUES, not on the dtype: a float array holding
    only {0.0, 1.0} is a binary mask that happens to be stored as float, and
    treating it as a probability gives the identical answer, so both are safe.
    The case that must not be mistaken is a float array with values BETWEEN 0
    and 1 - there, 0.0 and 0.5 give very different masks."""
    a = np.asarray(arr)
    if a.dtype == np.bool_ or a.size == 0:
        return BINARY

    finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    if finite.size == 0:
        return BINARY
    lo, hi = float(np.min(finite)), float(np.max(finite))

    if np.issubdtype(a.dtype, np.integer):
        # 0/1 masks are binary; 0/255 is the usual uint8 image convention and
        # is equally unambiguous. Anything else is not a mask this understands.
        return BINARY if (lo >= 0 and hi <= 1) or set(np.unique(finite)) <= {0, 255} \
            else AMBIGUOUS

    if lo < 0.0:
        return LOGIT              # signed scores: foreground is > 0
    if hi <= 1.0:
        return PROBABILITY        # includes float-stored {0.0, 1.0}
    return AMBIGUOUS


def to_bool(arr):
    """Raw mask values -> foreground pixels, thresholded for their encoding."""
    a = np.asarray(arr)
    enc = mask_encoding(a)
    if enc == PROBABILITY:
        return a >= 0.5
    if enc == LOGIT:
        return a > 0.0
    # BINARY is exact either way. AMBIGUOUS deliberately keeps the historical
    # behaviour: inventing a threshold for an array nobody understands would
    # trade a known unknown for an unknown one.
    return a.astype(bool)


def naive_bool(arr):
    """What `astype(bool)` alone would have produced - the comparison baseline."""
    return np.asarray(arr).astype(bool)


_REPORTED = set()


def describe_mask_encoding(arr, source="SAM"):
    """One line saying what the raw masks are and how they will be read.

    Returns the text, or None when this source has already been described -
    once per run is a fact worth having in every log, and once per frame is
    noise that trains people to scroll past it.

    The AREA RATIO is the number to look at. It compares the pixels a plain
    `astype(bool)` would have kept against the pixels the correct threshold
    keeps, so it answers the only question that matters: did this ever make a
    difference, and how much."""
    key = str(source)
    if key in _REPORTED:
        return None
    _REPORTED.add(key)

    a = np.asarray(arr)
    enc = mask_encoding(a)
    finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 0.0

    line = (f"  [i] {source} raw masks: shape={tuple(a.shape)} dtype={a.dtype} "
            f"range=[{lo:.4g}, {hi:.4g}] -> read as {enc.upper()}")

    if enc == BINARY:
        return line + " (thresholding is exact; nothing to decide)."

    naive = int(naive_bool(a).sum())
    fixed = int(to_bool(a).sum())
    if enc == AMBIGUOUS:
        return (line + ".\n"
                f"  [!] Values are non-negative but exceed 1, so this is "
                f"neither a probability nor clearly a logit. Keeping the old "
                f"`astype(bool)` behaviour ({naive} px) because inventing a "
                f"threshold here would be a guess. Check what your SAM build "
                f"returns before trusting these boundaries.")

    ratio = (naive / fixed) if fixed else float("inf")
    line += f".\n      thresholded at {0.5 if enc == PROBABILITY else 0.0}: "
    line += f"{fixed} px foreground, vs {naive} px for a plain astype(bool)"
    if ratio > 1.05:
        line += (f" - {ratio:.2f}x larger.\n"
                 f"  [!] A plain astype(bool) would have inflated every mask "
                 f"outward through its own soft edge, which is a boundary that "
                 f"sits outside the plant in every direction. That is now "
                 f"thresholded properly.")
    else:
        line += " - effectively the same, so this was never the boundary problem."
    return line


def reset_reporting():
    """Forget which sources have been described. For tests, and for a process
    that prelabels several sessions and wants the line once per session."""
    _REPORTED.clear()
