#!/usr/bin/env python3
"""
SeeWeed3D - choose the next WEED frames to annotate (round N of the loop)

    python -m seeweed3d.training.datasets.weeds_mine

Runs the current weed checkpoint over the unlabelled weed pool, ranks every
frame by how much annotating it would teach, picks a spread-out batch, and
writes it as a CVAT-ready folder with the model's own predictions already in it.

WHY THE FRAMES IT GETS WRONG, NOT THE ONES IT GETS RIGHT
--------------------------------------------------------
The intuitive loop - run the model, keep the frames it did WELL on, add those
to training - teaches almost nothing. Where the model is already right the loss
is already near zero, so there is no gradient and nothing to learn; all that
changes is that the model becomes more confident about what it already knew,
while its blind spots stay exactly as blind.

Worse, that loop feeds the model's own predictions back as ground truth, so its
confident errors become training truth and compound. That is why nothing here
writes a prediction into a manifest: every exported frame goes to a human
first. The export is CORRECTION, not annotation - the classes are usually right
because this is your model and it knows this ontology, and the work is fixing
boundaries and adding what was missed.

ROUND 0 IS DIFFERENT
--------------------
Uncertainty ranking needs a model good enough for its uncertainty to mean
something. Trained on a few dozen frames it is not, so the first batch should
lean on COVERAGE - different sessions, densities and light - which the
diversity pass already provides. Switch to trusting the ranking from round 2.

THE ONE THING THAT INVALIDATES EVERYTHING
------------------------------------------
Mining a session you intend to test on. Mining selects the frames the model
finds HARDEST, which are exactly the frames it would benefit most from having
seen, so mining a test session does not merely leak it - it leaks it in the
most flattering possible direction. HOLDOUT_SESSIONS below and
HOLDOUT_TEST_SESSIONS in weeds.py must name the same sessions.
"""
import ntpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from annotation.mine_pool import CONFIG as BASE, mine  # noqa: E402
from training.datasets.weeds import (HOLDOUT_TEST,  # noqa: E402
                                     OUT_DIR as WEEDS_OUT_DIR,
                                     WEED_SESSIONS)
from training.datasets.weeds_train import RUNS_ROOT  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: Which round this is. Only used to name the output folder, so an unfinished
#: batch is never overwritten by the next one.
ROUND = 1

#: WHERE A MINED BATCH IS WRITTEN, one folder per round so an unfinished batch
#: is never overwritten by the next one. Upload this folder to CVAT.
BATCHES_ROOT = r"E:\Dataset_Vidalia\batches"

#: THE POOL TO MINE - the `sessions` folder holding unlabelled weed sessions
#: alongside the annotated one. With only the annotated session under it there
#: is nothing new to find, so this wants a second recording before round 1.
POOL_ROOT = r"E:\Dataset_Vidalia\Weeds_20260108_3_good\sessions"

#: The checkpoint doing the ranking - the model trained on what you have so far.
#: rfdetr writes checkpoint_best_total.pth; use _total, never _ema (rfdetr keeps
#: three files and _total is copied from whichever actually won).
CHECKPOINT = ntpath.join(RUNS_ROOT, "weeds_r0",
                         "checkpoint_best_total.pth")

CONFIG = dict(
    BASE,
    CHECKPOINT=CHECKPOINT,
    BACKEND="rfdetr",
    DEVICE="cuda",

    # What is already labelled: skips those frames, and counts class
    # frequencies so a scarce class scores higher.
    DATASET_DIR=WEEDS_OUT_DIR,

    # The pool to mine: the SESSIONS folder, so unlabelled sessions beside the
    # annotated one are in scope. Its own labelled frames are skipped, but a
    # pool containing only them has nothing new to find.
    SESSIONS_ROOT=POOL_ROOT,
    ONLY_SESSIONS=[],

    # Must match weeds.py. Checked independently on purpose - one list being
    # right does not make the other right.
    HOLDOUT_SESSIONS=list(HOLDOUT_TEST),

    OUT_DIR=ntpath.join(BATCHES_ROOT, f"weeds_round{ROUND}"),

    # Lower than you would deploy at: a spurious mask costs one delete, a
    # MISSING one costs the annotator noticing an absence, which is far harder.
    CONF=0.20,

    # Consecutive frames are near-identical, so scanning every one wastes
    # inference on duplicates the diversity pass then discards.
    STRIDE=5,

    # Size it to what you will actually annotate this round. An over-large
    # batch goes stale, and a stale batch blocks its frames from being
    # re-selected until it is merged or abandoned.
    BATCH_SIZE=60,

    SKIP_FRAMES_IN_FLIGHT=True,
    RECORD_ROUND=True,
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    raise SystemExit(mine(CONFIG))
