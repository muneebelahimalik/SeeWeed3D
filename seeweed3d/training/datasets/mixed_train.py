#!/usr/bin/env python3
"""
SeeWeed3D - train the MIXED model (weeds + crop) for round N

    python -m seeweed3d.training.datasets.mixed_train

The deployable model. weeds_train.py produces a Stage A model that cannot be
pointed at anything - the safety decision rejects every candidate without a crop
mask, on purpose - and this one produces the model the pipeline can actually
run. See perception/preflight.py, which reports the difference as BLOCKING.

Every key not named here comes from train_model_rfdetr.py, and the training
settings are deliberately IDENTICAL to weeds_train.py's. Changing the recipe in
the same step that adds the crop class would make the comparison between the two
uninterpretable, and that comparison is the thing worth having: it is how you
learn whether adding onions cost you weed recall.

WHAT THE EXTRA CLASS COSTS
--------------------------
Watch weed recall, not the mean AP. Adding the crop adds a class that is large,
regular and easy, so mAP goes UP almost regardless of what happens to the weeds -
and a model that got better at onions while getting worse at small weeds is a
worse weeder with a better number. eval_seg's recall-by-size is the column that
answers this.
"""
import ntpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.datasets.mixed import OUT_DIR as MIXED_OUT_DIR  # noqa: E402
from training.datasets.weeds_train import CONFIG as WEED_CONFIG  # noqa: E402
from training.train_model_rfdetr import CONFIG as BASE  # noqa: E402
from training.train_model_rfdetr import main  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: Bump this each round. Numbered independently of the weed rounds, because
#: they are different models trained on different data - sharing the counter
#: would make "round 2" ambiguous in every later conversation.
ROUND = 0

#: WHERE RUNS ARE WRITTEN. Separate from the weed runs root so a mixed round
#: never overwrites the weed checkpoint it should be compared against.
RUNS_ROOT = r"E:\Dataset_Vidalia\runs_2_mixed"

CONFIG = dict(
    BASE,
    # Read from the build's own setting, so a DATASET_DIR cannot drift from the
    # OUT_DIR that produced it - the mismatch that has killed a run twice.
    DATASET_DIR=MIXED_OUT_DIR,
    RUN_DIR=ntpath.join(RUNS_ROOT, f"mixed_r{ROUND}"),

    # THE SAME RECIPE AS THE WEED MODEL, imported rather than copied. Two
    # config blocks that are meant to be identical and are maintained by hand
    # stop being identical, and the day they do is the day the comparison
    # between the two models stops meaning anything.
    **{k: WEED_CONFIG[k] for k in (
        "DEVICE", "RESOLUTION", "VARIANT", "BATCH", "GRAD_ACCUM", "WORKERS",
        "EPOCHS", "EARLY_STOPPING", "PATIENCE",
        "TVERSKY_ALPHA", "TVERSKY_BETA")},
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    main(CONFIG)
