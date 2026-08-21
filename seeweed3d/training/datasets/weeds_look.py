#!/usr/bin/env python3
"""
SeeWeed3D - LOOK at what the weed model does on a session it never saw

    python -m seeweed3d.training.datasets.weeds_look

Runs the current round's checkpoint over a weed session and writes an overlay
per frame. No ground truth needed, so it works the moment a recording exists -
which is the whole point: correcting frames is expensive and this costs nothing.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
It answers "do the masks still sit on plants at all on ground from a different
drive". That question is not available from the val score: with one annotated
session, val is contiguous blocks WITHIN it, so it shares the light, the soil,
the growth stage and often the individual plants.

It CANNOT tell you recall. A frame with nothing drawn means "the model found
nothing", which is indistinguishable from "there was nothing to find". Missed
weeds are this project's failure mode and they are exactly what an overlay
cannot show you. For that you need corrected frames and eval_seg.

SO THIS IS A GO/NO-GO, NOT A MEASUREMENT
----------------------------------------
Look for: masks on soil, one plant split into several, several plants merged
into one, whole plants absent where you can see green. If those look sane, the
model transfers well enough for its rankings to be worth something and mining
is worth running. If they do not, more annotation of the same kind is the
answer and mining would just rank noise.

RUN IT ON THE HOLDOUT, TOO
--------------------------
Looking is not scoring, so it does not burn a test session - no threshold is
tuned and no frame is selected from what you see. That is the one thing you may
do with a holdout before it becomes a test set.
"""
import ntpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perception.predict_images import CONFIG as BASE, predict  # noqa: E402
from training.datasets.weeds import WEED_POOL_ROOT  # noqa: E402
from training.datasets.weeds_train import ROUND, RUNS_ROOT  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: Which session to look at. A session folder (its rgb/ is used), a plain
#: folder of images, or a single image.
#:
#: Point this at a session the model has NEVER trained on - a session it has
#: seen tells you it can memorise, which was never in doubt.
SESSION = "vid3_20260108_110444"

#: Which round's model. Defaults to the round weeds_train.py is set to, so the
#: two cannot drift; override with a number to look at an older one.
LOOK_AT_ROUND = ROUND

CONFIG = dict(
    BASE,
    IMAGES=ntpath.join(WEED_POOL_ROOT, SESSION),

    # rfdetr writes three checkpoints and _total is copied from whichever
    # actually won. Scoring _ema silently reports the loser.
    CHECKPOINT=ntpath.join(RUNS_ROOT, f"weeds_r{LOOK_AT_ROUND}",
                           "checkpoint_best_total.pth"),
    BACKEND="rfdetr",
    DEVICE="cuda",

    OUT_DIR=ntpath.join(RUNS_ROOT, f"weeds_r{LOOK_AT_ROUND}",
                        f"look_{SESSION}"),

    # 40 frames spread across the drive, not 40 pictures of one plant.
    # Consecutive ZED frames are near-identical, so the stride matters more
    # than the count.
    LIMIT=40,
    STRIDE=25,

    # LOWER than a deployment threshold on purpose. The question here is how the
    # model fails, and a mask it nearly drew is evidence; a mask it did not draw
    # is silence. Raise it to 0.5 once you are asking "would I ship this".
    CONF=0.25,

    # RGB only - no depth, no LEP, no safety decision. Stage A is the thing
    # being judged and "full" would fold three more failure modes into the
    # picture.
    MODE="segmentation",

    # ZED frames are 2208x1242; half size stays readable and keeps the folder
    # small enough to flick through.
    OVERLAY_SCALE=0.5,
    LABELS="class_score",
    LEGEND=True,
)

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

if __name__ == "__main__":
    raise SystemExit(predict(CONFIG))
