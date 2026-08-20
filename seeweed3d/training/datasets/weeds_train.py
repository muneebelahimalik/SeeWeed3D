#!/usr/bin/env python3
"""
SeeWeed3D - train the WEED model for round N

    python -m seeweed3d.training.datasets.weeds_train

Every key not named here comes from train_model_rfdetr.py, so the loss shaping,
the preflight checks and the resolution guard are the shared ones.

ONE RUN DIRECTORY PER ROUND
---------------------------
ROUND names the run folder. Reusing a directory overwrites the checkpoint you
would have compared against, and the whole point of the loop is the comparison:
without round N-1 still on disk, "it got better" is a memory rather than a
measurement.

WHAT TO CHANGE BETWEEN ROUNDS
-----------------------------
Only ROUND. Changing the architecture, the resolution or the seed in the same
step that adds data means the improvement cannot be attributed - and attributing
it is the only reason to run the loop rather than annotate everything at random.
Settle the settings once, on round 0, then leave them alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.datasets import common as loc  # noqa: E402
from training.train_model_rfdetr import CONFIG as BASE, main  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: Bump this each round. Round 0 is the model trained on what you have today.
ROUND = 0

CONFIG = dict(
    BASE,
    DATASET_DIR=str(loc.out("weeds_v1")),
    RUN_DIR=str(loc.runs(f"weeds_r{ROUND}")),

    DEVICE="cuda:1",          # GPU 0 drives the display

    # 48 GB is a great deal more than these defaults assume. Resolution first:
    # it has bought more than capacity on this data every time. Must be a
    # multiple of 24 for the medium/large variants.
    RESOLUTION=1344,
    VARIANT="medium",
    BATCH=8,
    GRAD_ACCUM=2,             # BATCH x GRAD_ACCUM stays near 16

    # 0 for the first run on Windows: a lightning dataloader worker that fails
    # to start kills the process after the model-summary table with no
    # traceback. Raise to 2-4 once a run is known to finish.
    WORKERS=0,

    # Overnight is far more than 60 epochs at this size. Early stopping decides
    # when to stop; PATIENCE is deliberately higher than rfdetr's default of 10
    # because a re-initialised head leaves whole classes at AP 0.000 for the
    # first several epochs.
    EPOCHS=200,
    EARLY_STOPPING=True,
    PATIENCE=25,

    # Recall-leaning. A missed weed sets seed and is invisible; a spurious one
    # costs one laser pulse. WATCH PRECISION in the confidence sweep - if
    # weed_precision falls further than small-weed recall rises, the answer is
    # a higher operating confidence, not a higher BETA.
    TVERSKY_ALPHA=0.3,
    TVERSKY_BETA=0.7,
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    main(CONFIG)
