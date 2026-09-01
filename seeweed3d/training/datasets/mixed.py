#!/usr/bin/env python3
"""
SeeWeed3D - the MIXED dataset: weeds AND the crop. THE DEPLOYABLE BUILD.

    python -m seeweed3d.training.datasets.mixed

WHY THIS IS THE ONE THAT MATTERS
--------------------------------
A weed-only model is a Stage A model, not a system. The safety decision cannot
approve a single target without a crop mask, and it will not accept the absence
of one as evidence of absence - `SafetyConfig.allow_missing_crop_mask` defaults
to False precisely so that "this model cannot see onions" is never read as
"there are no onions here". So every candidate is rejected, the run completes,
and the output is indistinguishable from a clean field.

perception/preflight.py reports that as BLOCKING. This build is the fix: one
model that predicts the crop and the weeds, which is what the laser needs before
it can be pointed at anything.

THE HONEST PART: THE LABELS ARE NOT ALL THE SAME KIND
------------------------------------------------------
The weed sessions are hand corrected. The onion sessions are SAM 3 prelabels
nobody opened. Merging them does not average those into something in between -
it produces a dataset where the weed classes are measured and the crop class is
agreement with a prelabeler, and LABEL_PROVENANCE = "mixed" is the flag that
keeps that recoverable six months from now.

It matters asymmetrically. An unreviewed weed mask costs a slightly wrong
boundary. An unreviewed CROP mask decides whether the laser fires, so the crop
class is the one whose labels most deserve a human, and it is currently the one
that has had the least. Correct crop frames before trusting a crop-safety
number, not after.

CLASS COUNTS DECIDE WHAT TO DROP, AND ONLY THE BUILD KNOWS THEM
----------------------------------------------------------------
weeds.py drops weed_cluster (2 instances) and wild_radish (0). Those numbers are
properties of that build, not of the ontology, and merging sessions changes
them. So DROP_CLASSES starts EMPTY here and the build's own class report is what
should decide it - a class in single digits reports an AP near zero and drags
the mean down for a reason that has nothing to do with the model, and a class
with none at all still costs a head that can never fire.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.ontology import CLASSES  # noqa: E402
from training.datasets.onions import ONION_SESSIONS  # noqa: E402
from training.datasets.weeds import WEED_SESSIONS  # noqa: E402
from training.make_dataset import CONFIG as BASE, main  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: Imported from the two single-purpose builds rather than repeated, so a
#: session added there reaches this build without a second edit - and the three
#: builds cannot silently come to disagree about which drives exist.
#:
#: Add genuinely MIXED sessions here, the ones holding both onions and weeds in
#: the same frame. Those are the most valuable frames in the project: they are
#: the only ones that teach the model to tell the two apart at a boundary, which
#: is the exact decision the laser depends on.
#:
#: A hand-curated batch folder (annotations/ + rgb/, no sessions/ parent) works
#: here as-is. If its filenames no longer name a drive, the build reads the
#: EXPORT FOLDER NAME as the session id and says so - check that line if the
#: batch came from several drives, because one id over frames metres apart lets
#: a split put near-copies on both sides of it.
MIXED_SESSIONS = [
    r"E:\Dataset_Vidalia\Mix_raj_Batch 01",
]

SOURCES_ROOTS = list(WEED_SESSIONS) + list(ONION_SESSIONS) + MIXED_SESSIONS

#: WHERE THE BUILT DATASET IS WRITTEN. Safe to delete and rebuild.
OUT_DIR = r"E:\Dataset_Vidalia\datasets\mixed_v1"

#: Sessions that must never enter training.
#:
#: THIS SHOULD NOT BE EMPTY IN A MIXED BUILD. With several sessions across two
#: campaigns there is no excuse for splitting within a drive, and a held-out
#: drive is the only thing that makes round N comparable with round N-1. Pin the
#: same list in mine_pool's HOLDOUT_SESSIONS - a test asserts they agree.
#:
#: THE CONTACT BATCH IS THE ONE REAL CANDIDATE, and it is a genuine trade.
#:
#: It is hand corrected, it holds both classes in one frame, and it is small.
#: As TRAINING data it is the only thing that teaches the onion/weed boundary,
#: and a handful of frames of that is still the most valuable data in the
#: project. As a TEST set it is the only honest crop-safety measurement that
#: exists: every other number here is computed on weed-only or onion-only
#: frames, or against unreviewed prelabels, and neither can see a crop mistake
#: at a boundary at all.
#:
#: Hold it out until there is a second batch. An untested crop-safety claim is
#: the failure this project keeps designing against, and a model that trained
#: on its only contact frames cannot be asked whether it is safe. Move it into
#: MIXED_SESSIONS the day Batch 02 exists to take its place here.
#:
#: The batch's own frame count decides whether that is even possible: below
#: roughly 20 frames it is too small to measure with AND too small to train on,
#: and the answer is more annotation rather than a choice between the two.
HOLDOUT_TEST = [
    "Mix_raj_Batch_01",
]

#: Sessions pinned to VAL. Not a redundant knob - it is the one that decides
#: which checkpoint you keep.
#:
#: Left to the allocator, val came out as two onion-only drives: 572 frames with
#: essentially no weeds in them. `checkpoint_best_total.pth` is selected on val,
#: so that run would have kept whichever epoch was best at ONIONS and never once
#: looked at weed recall - the exact way a mixed model becomes a crop detector
#: that ignores weeds, while every number on the page goes up.
#:
#: A mixed drive was the obvious pin and it did NOT work: Visit2_20260210_164614
#: is scene "mixed" and carries zero weed instances, because only the crop was
#: ever annotated there. The scene said the weeds are in the picture; the export
#: said nobody labelled them.
#:
#: So this pins the one session whose weed labels a PERSON checked. 60 frames is
#: small for a validation set, and a small honest one beats 572 frames that
#: cannot score the class the machine exists to find.
#:
#: It costs training the only fully hand-corrected weed drive. That is the right
#: way round: val decides which checkpoint survives, so it should hold the
#: labels you trust most, not the ones you can spare.
HOLDOUT_VAL = [
    "vid2_20260108_122731",
]

CONFIG = dict(
    BASE,
    SOURCES=[{"DATUMARO_ROOT": p, "IMAGES_ROOT": p} for p in SOURCES_ROOTS],
    OUT_DIR=OUT_DIR,

    # EVERYTHING the ontology defines. This is the build that has to cover the
    # whole decision, so a class left out here is a class the deployed model
    # scores as background.
    KEEP_CLASSES=list(CLASSES),

    # MERGED, not dropped, and the build's own counts are why.
    #
    #   weed_cluster       5 instances, 0 in val, 0 in test
    #   wild_radish       97 instances, 0 in val, 0 in test
    #
    # Both are unmeasurable: a class with no val instances has an AP that moves
    # between rounds on noise, and a class with 5 costs a detection head that
    # can never fire. But DROPPING them would turn 102 real weeds into
    # background - teaching that those plants are soil, which on a weeder is an
    # untreated weed. Merging keeps every one of them a TARGET, which is the
    # decision the laser actually needs; only the species label is lost, and
    # neither class had enough instances to support one.
    #
    # weed_cluster merges too even though it means "intermingled weeds, no
    # separable LEP". This build carries 0 LEPs, so the distinction it exists to
    # make is not being trained anyway. Revisit when Stage B has labels.
    MERGE_CLASSES={"wild_radish": "other_weed", "weed_cluster": "other_weed"},
    DROP_CLASSES=[],

    # Hand-corrected weeds beside unreviewed crop masks. Neither label answers
    # for the other, and every score computed on this build is read through
    # this field.
    LABEL_PROVENANCE="mixed",

    # Whole sessions where there are enough of them - and with two campaigns
    # merged there are. The build says which granularity it used.
    SPLIT_MODE="auto",
    SPLIT_GRANULARITY="auto",
    HOLDOUT_TEST_SESSIONS=HOLDOUT_TEST,
    HOLDOUT_VAL_SESSIONS=HOLDOUT_VAL,

    # Scene stratification earns its keep here and nowhere else: a split that
    # put every onion frame in train and every weed frame in val would report a
    # confident number about a model that had never been asked the question.
    STRATIFY_BY_SCENE=True,

    # WITHOUT THESE, STRATIFICATION CANNOT RUN. The first mixed build reported
    #
    #     [!] val contains no mixed session - the crop-vs-weed decision is
    #         never exercised there.
    #
    # and the cause was not the split: four sessions had no meta/session.json,
    # so their scene was "unknown" and they were invisible to the allocator.
    # The mixed batch was one of them - the single session whose scene matters
    # most, since a mixed frame is the only place the crop-vs-weed decision is
    # exercised at all.
    #
    # Ids come from the SESSIONS IN THIS BUILD table, which is NOT always the
    # export folder name: vid3_20260108_132749's frames live in a folder called
    # Visit1_20260108_132749.
    SCENE_HINTS={
        "Mix_raj_Batch_01": "mixed",
        "vid2_20260108_122731": "weed_only",
        "vid3_20260108_110444": "weed_only",
        "vid3_20260108_132749": "onion_only",
    },

    VAL_FRACTION=0.15,
    TEST_FRACTION=0.15,

    # NEVER change between rounds.
    SEED=1234,
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    main(CONFIG)
