#!/usr/bin/env python3
"""
SeeWeed3D - the WEED-ONLY dataset (EDIT THE CONFIG BELOW, then run it)

    python -m seeweed3d.training.datasets.weeds

Every key not named here comes from make_dataset.py's CONFIG, so the split
logic, the gap accounting and the image verification are the shared ones and a
fix to them reaches this build too.

WHAT MAKES A WEED-ONLY BUILD DIFFERENT
--------------------------------------
The class label is FREE and CERTAIN. Every plant in a weed-only drive is a
weed - by how the recording was made, not by anyone's judgment. That is why
these frames are worth more per frame than a mixed scene: the expensive part of
annotation there is deciding WHICH plant something is, and here that question
does not arise.

What is NOT free is instance identity - which pixels belong to which plant.
That is the whole cost, and it is what the active-learning loop is buying.

LABEL PROVENANCE IS THE ONE FIELD TO GET RIGHT
----------------------------------------------
These frames are HAND CORRECTED, unlike the onion build, whose masks are
unreviewed SAM output. So this dataset's val and test scores measure real
performance rather than agreement with a prelabeler, and that distinction is
recorded in the manifest and restated by preflight at train time.

Set it back to "mixed" the moment a round merges unreviewed frames in - it
stops being true the first time it stops being true.

THE HOLDOUT IS THE POINT
------------------------
HOLDOUT_TEST_SESSIONS names sessions that active learning must never touch.
Mining picks the frames the model finds HARDEST, which are precisely the frames
it would benefit most from having seen - so mining a test session does not
merely leak it, it leaks it in the most flattering possible direction. Pin it
here AND in mine_pool's HOLDOUT_SESSIONS; the two are checked independently.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.datasets import common as loc  # noqa: E402
from training.make_dataset import CONFIG as BASE, main  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: Campaign folders holding the weed recordings. One entry per campaign,
#: pointed at its `sessions` folder - every session under it is discovered.
WEED_CAMPAIGNS = [
    "Weeds_3_good",
]

#: Sessions active learning must NEVER mine, and that never enter training.
#: Leave [] only if you accept that no number this dataset produces can be
#: compared across rounds. See the module docstring.
HOLDOUT_TEST = [
    # "vid2_20260108_122731",
]

CONFIG = dict(
    BASE,
    SOURCES=[{"DATUMARO_ROOT": str(loc.campaign(c)),
              "IMAGES_ROOT": str(loc.campaign(c))}
             for c in WEED_CAMPAIGNS],

    OUT_DIR=str(loc.out("weeds_v1")),

    # Every weed class, and NOT onion_plant. A weed-only drive has no crop in
    # it, so an onion_plant instance here is a mislabel worth seeing rather
    # than training on.
    KEEP_CLASSES=["cutleaf_evening_primrose", "wild_radish", "grass_weed",
                  "weed_cluster", "other_weed"],
    DROP_CLASSES=[],

    # HAND CORRECTED - unlike the onion build. Change to "mixed" as soon as a
    # round merges frames that were accepted rather than corrected.
    LABEL_PROVENANCE="hand_corrected",

    # These frames WERE all corrected, so there is no unverified subset to
    # exclude. Set this only if that stops being true.
    INCLUDE_FRAMES="",

    # Whole sessions where there are enough of them, frame blocks otherwise -
    # and the build says which it used.
    SPLIT_MODE="auto",
    SPLIT_GRANULARITY="auto",
    HOLDOUT_TEST_SESSIONS=HOLDOUT_TEST,

    VAL_FRACTION=0.15,
    TEST_FRACTION=0.15,
    BLOCKS_PER_SESSION=3,
    GAP_FRAMES=12,

    # NEVER change this between rounds. The split is deterministic for a seed,
    # so a session that was in test stays in test as the dataset grows - and a
    # new seed re-draws every split, which makes round 3 incomparable with
    # round 1 for a reason that has nothing to do with the model.
    SEED=1234,
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    main(CONFIG)
