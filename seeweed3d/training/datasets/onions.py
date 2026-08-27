#!/usr/bin/env python3
"""
SeeWeed3D - the ONION-ONLY dataset (EDIT THE CONFIG BELOW, then run it)

    python -m seeweed3d.training.datasets.onions

Every key not named here comes from make_dataset.py's CONFIG, so the split
logic, the gap accounting and the image verification are the shared ones.

WHY THIS EXISTS AS A RUNNER
---------------------------
This build used to live IN make_dataset.py's CONFIG - the shared base that
weeds.py overrides. One CONFIG edited back and forth between an onion build and
a weed build is exactly how a stale DATASET_DIR reaches a training run, which is
the mistake the per-dataset runners were split out to prevent. It was only the
weed build that got a runner; this is the other half.

WHAT MAKES AN ONION BUILD DIFFERENT FROM A WEED BUILD
-----------------------------------------------------
The labels are NOT hand corrected. They are SAM 3 prelabels, and the class is
free the same way it is in a weed-only drive - an onion row contains onions -
but the MASK GEOMETRY is unreviewed. So this build's val and test scores measure
agreement with a prelabeler, not performance, and LABEL_PROVENANCE says so. Read
every number computed on it through that.

That is also why it is a legitimate build to have: the crop class has to come
from somewhere before anyone has hand-corrected a crop frame, and a model that
cannot predict onion_plant cannot be deployed at all - perception/preflight.py
rejects it outright, because a model that cannot see onions must never have its
silence read as "no onions here".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.ontology import CROP_CLASS  # noqa: E402
from training.make_dataset import CONFIG as BASE, main  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: WHERE THE ANNOTATED ONION FRAMES ARE. Either a session folder holding
#: annotations/ + rgb/, or a folder whose CHILDREN are session folders.
ONION_SESSIONS = [
    r"E:\Dataset_Vidalia\onions_20260108_1\sessions",
    r"E:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions",
]

#: WHERE THE BUILT DATASET IS WRITTEN. Safe to delete and rebuild.
OUT_DIR = r"E:\Dataset_Vidalia\datasets\onions_v1"

#: Sessions that must never enter training, so they can measure it.
#:
#: Unlike the weed build, there are several onion sessions - so there is no
#: reason for this to be empty. A holdout drive is what makes one round
#: comparable with the next; without one every score is an upper bound computed
#: on ground the model has already seen.
HOLDOUT_TEST = [
]

CONFIG = dict(
    BASE,
    SOURCES=[{"DATUMARO_ROOT": p, "IMAGES_ROOT": p} for p in ONION_SESSIONS],
    OUT_DIR=OUT_DIR,

    # The crop, and nothing else. A weed instance in an onion-only drive is a
    # mislabel worth seeing rather than training on - the mirror of the weed
    # build dropping onion_plant.
    KEEP_CLASSES=[CROP_CLASS],
    DROP_CLASSES=[],

    # NOT hand corrected. These are SAM 3 masks nobody opened, so every score
    # computed on this build measures agreement with a prelabeler. Change it the
    # day a person corrects them, and not before.
    LABEL_PROVENANCE="prelabel_unreviewed",

    SPLIT_MODE="auto",
    SPLIT_GRANULARITY="auto",
    HOLDOUT_TEST_SESSIONS=HOLDOUT_TEST,

    # With several sessions the split can be BY SESSION, which is the honest
    # granularity - blocks within one drive share its light, soil and plants.
    VAL_FRACTION=0.15,
    TEST_FRACTION=0.15,

    # Same seed as every other build. A new seed re-draws every split, which
    # makes one round incomparable with the last for a reason that has nothing
    # to do with the model.
    SEED=1234,
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    main(CONFIG)
