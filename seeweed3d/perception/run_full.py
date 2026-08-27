#!/usr/bin/env python3
"""
SeeWeed3D - the WHOLE system on real frames: class, LEP, 3D, safety.

    python -m seeweed3d.perception.run_full

Stage A finds and classifies each plant, Stage B places the growth point inside
it, depth turns that pixel into a 3D point in the camera frame, and the safety
decision says whether it may be fired at. One overlay and one JSON record per
frame.

IT CHECKS ITSELF FIRST
----------------------
Every failure in this pipeline produces output that looks fine. A weed-only
model abstains on every target and reports zero candidates, which reads exactly
like a clean frame. A session without calibration returns every target with no
xyz_mm, which reads exactly like nothing found. So this runs
perception/preflight.py before the GPU and prints what it found; STOP_ON_BLOCKING
decides whether a blocking finding ends the run or is merely stated.

WHICH MODEL
-----------
Defaults to the MIXED model, not the weed model. A weed-only checkpoint cannot
produce a candidate at all: the safety decision will not accept a missing crop
mask as evidence that there is no crop. Point CHECKPOINT at a weed run if you
want to see Stage A behaviour, and expect every target to abstain.

WHAT THE RECORD HOLDS, AND WHAT IT MEANS
-----------------------------------------
    class_name / score      Stage A
    lep_uv                  the growth point, full-frame pixels
    xyz_mm                  that point in the camera frame, millimetres
    xyz_sigma_mm            how sure - the delta robot's tolerance lives here
    safety_status           "candidate" or "abstain"
    rejection_reasons       WHY it abstained, one string per failed threshold

`rejection_reasons` is the field to read first on a disappointing run. Zero
candidates with reasons dominated by one code is a threshold problem; zero
candidates with no targets at all is a Stage A problem, and they are fixed in
completely different places.
"""
import ntpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.run_dirs import stamped  # noqa: E402
from perception import preflight as pf  # noqa: E402
from perception.predict_images import CONFIG as BASE, predict  # noqa: E402
from training.config import PipelineConfig  # noqa: E402
from training.datasets.mixed import HOLDOUT_TEST  # noqa: E402
from training.datasets.mixed_train import ROUND, RUNS_ROOT  # noqa: E402
from training.datasets.weeds import WEED_POOL_ROOT  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: A session folder (its rgb/ is used, and its depth/ and meta/ are what make
#: the 3D half work), a plain folder of images, or a single image.
#:
#: A PLAIN FOLDER OF IMAGES HAS NO DEPTH. The run still works and every target
#: comes back 2D - useful for looking at Stage A and Stage B, useless for
#: aiming anything.
SESSION = ""

#: The MIXED model by default - see the module docstring for why a weed-only
#: checkpoint cannot produce a candidate.
CHECKPOINT = ntpath.join(RUNS_ROOT, f"mixed_r{ROUND}",
                         "checkpoint_best_total.pth")

#: Stage B. Empty falls back to the hand-engineered estimator in
#: perception/lep.py, which is the baseline a learned model has to beat rather
#: than the deployed stage.
LEP_CHECKPOINT = ""

#: A DEPLOYMENT threshold, unlike the 0.25 the scorers use. There the question
#: is how the model fails and a mask it nearly drew is evidence; here the
#: question is whether to fire.
CONF = 0.50

#: 0 = every frame found. Full-pipeline frames are expensive and near-identical
#: consecutive ones tell you nothing new, so a stride is the right default here
#: even though the scorers set it to 1.
LIMIT = 40
STRIDE = 25

#: Stop before the GPU if preflight finds something blocking. False prints the
#: findings and runs anyway, which is right when you are deliberately looking at
#: a model you already know is incomplete.
STOP_ON_BLOCKING = True

#: Where overlays and records are written. Stamped, so two runs over one session
#: both survive - comparing this round against the last is most of why you run it.
OUT_DIR = stamped(ntpath.join(RUNS_ROOT, f"mixed_r{ROUND}"), "full")

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

CONFIG = dict(
    BASE,
    IMAGES=SESSION or ntpath.join(WEED_POOL_ROOT, ""),
    CHECKPOINT=CHECKPOINT,
    LEP_CHECKPOINT=LEP_CHECKPOINT,
    BACKEND="rfdetr",
    DEVICE="cuda",
    MODE="full",
    OUT_DIR=OUT_DIR,
    CONF=CONF,
    LIMIT=LIMIT,
    STRIDE=STRIDE,
    OVERLAY_SCALE=0.5,
    LABELS="class_score",
    LEGEND=True,
    # The COCO export is a Stage A artefact and this is not a Stage A run; the
    # per-frame JSON here carries the LEP and the 3D point, which is the part
    # that cannot be recovered from a COCO file.
    WRITE_COCO=False,
)


def main():
    cfg = PipelineConfig()
    findings, classes = pf.inspect(
        CONFIG["CHECKPOINT"], session_dir=SESSION or None,
        lep_checkpoint=LEP_CHECKPOINT,
        allow_missing_crop_mask=cfg.safety.allow_missing_crop_mask,
        holdout_sessions=HOLDOUT_TEST, dedup_iou=cfg.dedup_iou)
    print(pf.format_report(findings, checkpoint=CONFIG["CHECKPOINT"],
                           model_classes=classes))

    blocking = [f for f in findings if f.level == pf.ERROR]
    if blocking and STOP_ON_BLOCKING:
        raise SystemExit(
            f"Stopping: {len(blocking)} blocking finding(s) above.\n"
            f"Set STOP_ON_BLOCKING = False to run anyway - the output will "
            f"look ordinary, which is the reason this check exists.")

    predict(CONFIG)
    print(f"\n  Read rejection_reasons in predictions.json first.\n"
          f"  Zero candidates with reasons dominated by one code is a "
          f"threshold problem;\n"
          f"  zero candidates with no targets at all is a Stage A problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
