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
from training.make_dataset import CONFIG as BASE, main  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: WHERE YOUR ANNOTATED WEED FRAMES ARE. Full paths, one per entry.
#:
#: Either form works, and both are found the same way:
#:   ...\sessions\<session>   ONE session folder holding annotations/ + rgb/
#:   ...\sessions             a folder whose CHILDREN are session folders
#:
#: With one session, name it directly - the `sessions` form only earns its keep
#: once there are several under one parent.
WEED_SESSIONS = [
    r"E:\Dataset_Vidalia\Weeds_20260108_3_good\sessions\vid2_20260108_122731",
]

#: WHERE THE BUILT DATASET IS WRITTEN. Safe to delete and rebuild.
#: Keep it on the same drive as the images unless you have a reason not to -
#: the manifest records absolute paths, so a dataset and its images that live
#: on different drives are two things to keep in step instead of one.
OUT_DIR = r"E:\Dataset_Vidalia\datasets\weeds_v2"

#: THE UNLABELLED POOL - the `sessions` folder holding weed recordings that are
#: not yet corrected. Mining reads it, and weeds_look.py runs the model over a
#: session from it to see whether the masks still land on plants.
#:
#: This is a DIFFERENT campaign from WEED_SESSIONS above, and that is the point:
#: a pool containing only the session you trained on has nothing new to find.
WEED_POOL_ROOT = r"E:\Dataset_Vidalia\Weeds_20260108_1\sessions"

#: Sessions active learning must NEVER mine, and that never enter training.
#:
#: WITH ONE ANNOTATED SESSION THIS MUST STAY EMPTY - holding out your only
#: session leaves nothing to train on. The build then splits within the session
#: by contiguous frame blocks and says so. That is the honest fallback, and its
#: scores are an upper bound: val and test share the session's light, soil and
#: growth stage.
#:
#: THE MOMENT A SECOND SESSION IS CORRECTED, name it here. Pick the one you will
#: NOT correct for training - see docs/weed_active_learning.md. Mining reads the
#: same list and a test asserts they agree, because mining selects the frames
#: the model finds hardest and those are exactly the frames it would most
#: benefit from having seen.
HOLDOUT_TEST = [
]

CONFIG = dict(
    BASE,
    SOURCES=[{"DATUMARO_ROOT": p, "IMAGES_ROOT": p} for p in WEED_SESSIONS],

    OUT_DIR=OUT_DIR,

    # Every weed class, and NOT onion_plant. A weed-only drive has no crop in
    # it, so an onion_plant instance here is a mislabel worth seeing rather
    # than training on.
    KEEP_CLASSES=["cutleaf_evening_primrose", "wild_radish", "grass_weed",
                  "weed_cluster", "other_weed"],

    # weed_cluster has TWO instances in this build. A class in single digits
    # reports an AP near zero and drags the mean down for a reason that has
    # nothing to do with the model, so it is dropped from THIS build - the
    # ontology is untouched and it returns the moment there are examples.
    #
    # The cost is that those two clusters become background, which is the wrong
    # lesson in principle; at 2 instances out of 1446 it is noise either way,
    # and a class the metrics cannot measure is worse. Remove this the first
    # round that brings real cluster examples in.
    #
    # wild_radish has ZERO instances here. A class with no examples still gets
    # a head, still costs capacity, and can never be predicted - and preflight
    # lists it as 0/0 rather than flagging it, because "not in this build at
    # all" is a legitimate state it declines to complain about. Dropping it
    # makes the model 3 classes instead of 4.
    DROP_CLASSES=["weed_cluster", "wild_radish"],

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

    # SIZED FOR 60 FRAMES, NOT FOR THE ONION BUILD'S 1647.
    #
    # Carrying those settings over (3 blocks, 12-frame gap, 15/15) spent 24 of
    # 60 frames on buffers and produced val=5 test=5 - two splits too small for
    # any number computed on them to mean anything, and a training set of 26.
    # One block has two seams instead of six, which is the whole difference.
    #
    # TEST_FRACTION IS 0 ON PURPOSE. A 9-frame test set has error bars wider
    # than the number it reports, so it is not a weaker measurement - it is a
    # misleading one. Val does double duty for now: it selects the checkpoint
    # AND tracks round-to-round change, which is what the loop needs. It is
    # optimistic as an absolute score and consistent as a relative one.
    #
    # Cut a real test set from a SECOND weed session the moment you have one.
    VAL_FRACTION=0.20,
    TEST_FRACTION=0.0,
    BLOCKS_PER_SESSION=3,
    GAP_FRAMES=3,

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
