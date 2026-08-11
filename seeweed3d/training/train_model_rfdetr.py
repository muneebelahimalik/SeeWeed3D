#!/usr/bin/env python3
"""
SeeWeed3D - train Stage A on RF-DETR-Seg (EDIT THE CONFIG BELOW)
================================================================
    python seeweed3d/training/train_model_rfdetr.py

Build the dataset first with make_dataset.py - this reads the SAME
seg_manifest.json the Mask R-CNN path uses, so both backends train on exactly
the frames, classes and splits that pipeline verified.

WHAT THIS BUYS OVER train_model.py (Mask R-CNN)
-----------------------------------------------
Verified against rfdetr 1.9.1:

  MASK_DICE_COEF / MASK_CE_COEF   Dice + cross-entropy mask loss, weighted.
                                  torchvision computes Mask R-CNN's losses
                                  internally; changing them there means
                                  forking its ROI heads.
  CLS_COEF, IA-BCE                classification loss designed for DETR-style
                                  set prediction
  USE_EMA                         exponential moving average weights
  EARLY_STOPPING / PATIENCE       built in
  MULTI_SCALE                     multi-scale training
  GRAD_ACCUM                      an effective batch of 16 on small VRAM,
                                  instead of actually training at batch 2
  no anchors                      DETR set prediction, so the anchor-size trap
                                  that cost the Mask R-CNN path its small weeds
                                  cannot occur in the same form
  ONNX / TensorRT export          in the package, for the Jetson Orin

Apache-2.0. No source-release obligation, unlike the ultralytics backend.

THE ONE SETTING THAT DECIDES WHETHER THIS IS AN UPGRADE
-------------------------------------------------------
RESOLUTION. RF-DETR-Seg's own default is 432x432. A 2208x1242 ZED frame at
432 px loses small weeds far more severely than the 1333 px Mask R-CNN default
already did - so leaving it alone would REGRESS small-weed recall while looking
like a move to a better model. It must be a multiple of patch_size*num_windows
for the variant (24 for medium/large, 12 for nano/small); the runner checks and
names valid values rather than failing deep inside training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

CONFIG = {
    # -- Where things are ------------------------------------------------------
    # OUT_DIR from make_dataset.py (the folder holding seg_manifest.json).
    "DATASET_DIR": r"E:\Dataset_Vidalia\training1",

    # Sessions root(s). "" reuses whatever make_dataset.py recorded.
    "IMAGES_ROOT": "",

    # One folder per run - do not reuse, or you lose the comparison.
    "RUN_DIR": r"E:\Dataset_Vidalia\training1\rfdetr_v4",

    # RF-DETR trains from a Roboflow-style COCO tree, which this builds from
    # seg_manifest.json. "" puts it inside RUN_DIR. Point several runs at ONE
    # shared path to convert once instead of per run.
    "COCO_DIR": "",

    # Hardlink images into the COCO tree instead of copying. Same volume only;
    # falls back to copying automatically.
    "LINK_IMAGES": True,

    # -- Model -----------------------------------------------------------------
    # "nano" | "small" | "medium" | "large" - all Apache-2.0.
    # XLarge/2XLarge are omitted: they may fall under Roboflow's Platform Model
    # License, which is not one this project can ship under unchecked.
    "VARIANT": "medium",

    # THE setting. See the module docstring. 432 is the model's own default and
    # is NOT a good choice here. Must be a multiple of 24 for medium/large.
    #   1008 = 24 x 42   a sensible first try
    #   1248 = 24 x 52   close to the 1333 the Mask R-CNN path used
    #   1344 = 24 x 56   above it
    # VRAM grows with the square: raise GRAD_ACCUM, do not lower this.
    "RESOLUTION": 1008,

    # -- Training --------------------------------------------------------------
    "DEVICE": "cuda",
    "EPOCHS": 60,

    # BATCH x GRAD_ACCUM is the EFFECTIVE batch; keep it near 16. Gradient
    # accumulation is why resolution need not be traded for batch size.
    "BATCH": 2,
    "GRAD_ACCUM": 8,

    "LR": 1e-4,          # RF-DETR's own default; far lower than Mask R-CNN's

    # 0 ON WINDOWS. rfdetr trains through pytorch-lightning, and a lightning
    # dataloader worker that fails to start on Windows kills the process with
    # NO Python traceback - training simply stops after the model-summary table
    # and you are back at the prompt. Workers load in the parent instead, which
    # at 62 frames costs almost nothing.
    #
    # On Linux, or once a run is known to work, raise it to 2-4.
    "WORKERS": 0,

    # -- Advanced features -----------------------------------------------------
    "USE_EMA": True,             # averaged weights; usually a small free gain
    "MULTI_SCALE": True,         # trains across scales, helps small objects

    # "cosine" | "step". RF-DETR's own default is "step" with lr_drop=100, so
    # on any run shorter than 100 epochs the step NEVER FIRES and the learning
    # rate is constant start to finish - the first run here ended at the same
    # 1e-4 it began with. "cosine" sizes itself to EPOCHS and actually decays.
    "LR_SCHEDULER": "cosine",

    # The detection head is re-initialised for our class count, so step 0 is a
    # random classifier at full LR next to a pretrained backbone.
    "WARMUP_EPOCHS": 1.0,

    "EARLY_STOPPING": True,

    # EVALUATED epochs without improvement. Higher than rfdetr's 10 by design:
    # the re-initialised head leaves whole classes at AP 0.000 for the first
    # several epochs (cutleaf was dead until epoch 6 here), and patience 10
    # stopped a 60-epoch run at 23 while other_weed was still improving.
    "PATIENCE": 25,

    # -- Loss weights ----------------------------------------------------------
    # None = the model's defaults (mask_ce 5.0, mask_dice 5.0, cls 1.0).
    #
    # Raise MASK_DICE_COEF relative to MASK_CE_COEF when masks are roughly the
    # right shape but boundaries are poor: Dice is computed over the whole mask
    # and is insensitive to how many background pixels surround it, so it does
    # not get swamped by a large empty frame the way per-pixel CE can.
    "MASK_CE_COEF": None,
    "MASK_DICE_COEF": None,
    "CLS_COEF": None,

    # -- Tversky mask loss -----------------------------------------------------
    # Dice weights a false positive and a false negative EQUALLY. This system
    # does not: a missed weed survives to set seed, a spurious one costs one
    # laser pulse, and onion the model fails to mark is onion nothing protects.
    #
    #     TI = TP / (TP + ALPHA*FP + BETA*FN)
    #
    # BETA > ALPHA penalises MISSED pixels harder. 0.5/0.5 is exactly Dice and
    # patches nothing at all, so the default run is unchanged.
    #
    #   0.5 / 0.5   Dice. The baseline.
    #   0.3 / 0.7   recall-leaning. The natural first try here.
    #   0.2 / 0.8   aggressive; watch precision and the burn fraction.
    "TVERSKY_ALPHA": 0.5,
    "TVERSKY_BETA": 0.5,

    # Focal exponent on (1 - TI). >1 concentrates gradient on the masks the
    # model gets WRONG, which here means small weeds. Noisier on 62 frames,
    # so it is off by default. 1.33-2.0 is the usual range.
    "FOCAL_GAMMA": 1.0,

    # -- Monitoring ------------------------------------------------------------
    # RF-DETR's own tensorboard/mlflow flags, so its runs land in the SAME
    # MLflow store as the Mask R-CNN runs and the two are comparable in one
    # table - which is the entire point of having a tracker.
    "TRACK": "auto",
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################


def main(cfg=None):
    c = dict(CONFIG if cfg is None else cfg)

    ds = Path(c["DATASET_DIR"])
    man = ds / "seg_manifest.json"
    if not man.exists():
        raise SystemExit(
            f"ERROR: {man} not found.\n"
            f"Build the dataset first: edit and run "
            f"seeweed3d/training/make_dataset.py")

    from training.train_model import _resolve_images_root
    images = _resolve_images_root(c, man)

    run = Path(c["RUN_DIR"])
    if run.exists() and any(run.glob("*.pt")):
        print(f"\n[!] {run} already holds checkpoints and WILL BE ADDED TO.\n"
              f"    Point RUN_DIR at a new folder to keep runs comparable.\n")

    from training.train_seg_rfdetr import train
    train(ds, run,
          variant=c["VARIANT"], resolution=c["RESOLUTION"],
          epochs=c["EPOCHS"], batch=c["BATCH"], grad_accum=c["GRAD_ACCUM"],
          lr=c["LR"], device=c["DEVICE"], workers=c["WORKERS"],
          coco_dir=c["COCO_DIR"] or None, images_root=images,
          link=c["LINK_IMAGES"], overwrite=True,
          early_stopping=c["EARLY_STOPPING"], patience=c["PATIENCE"],
          use_ema=c["USE_EMA"], multi_scale=c["MULTI_SCALE"],
          lr_scheduler=c["LR_SCHEDULER"], warmup_epochs=c["WARMUP_EPOCHS"],
          mask_ce_coef=c["MASK_CE_COEF"], mask_dice_coef=c["MASK_DICE_COEF"],
          cls_coef=c["CLS_COEF"], track=c["TRACK"],
          tversky_alpha=c["TVERSKY_ALPHA"], tversky_beta=c["TVERSKY_BETA"],
          focal_gamma=c["FOCAL_GAMMA"])

    # checkpoint_best_total.pth, NOT checkpoint_best_ema.pth. rfdetr keeps
    # three: _regular (best live weights), _ema (best averaged weights) and
    # _total, which it copies from whichever of the two actually won. Naming
    # the EMA file would silently score the loser whenever the regular weights
    # were better - which is the run this project has already seen.
    best = _best_checkpoint(run)
    try:
        from evaluation.analyze_run import analyze
        analyze({"RUN_DIR": str(run), "DATASET_DIR": str(ds), "SPLIT": "val",
                 "DEVICE": c["DEVICE"], "CONF": 0.25, "CHECKPOINT": str(best),
                 "BACKEND": "rfdetr", "TRACK": c.get("TRACK", "auto")})
    # SystemExit too - it is a BaseException, so `except Exception` would let
    # it kill a run that has already finished training.
    except (Exception, SystemExit) as e:                # noqa: BLE001
        print(f"\n[warn] analysis skipped: {type(e).__name__}: {e}")

    print(f"\nScored with the SAME table the Mask R-CNN runs use:\n"
          f"  python -m seeweed3d.evaluation.eval_seg --backend rfdetr "
          f"--checkpoint {best} --dataset {ds} "
          f"--split val --device {c['DEVICE']} --sweep\n"
          f"  python -m seeweed3d.evaluation.report --backend rfdetr "
          f"--checkpoint {best} --dataset {ds} "
          f"--split val --device {c['DEVICE']}\n")


def _best_checkpoint(run):
    """The checkpoint to evaluate, preferring rfdetr's own overall winner."""
    for name in ("checkpoint_best_total.pth", "checkpoint_best_regular.pth",
                 "checkpoint_best_ema.pth", "checkpoint_best.pth"):
        if (Path(run) / name).exists():
            return Path(run) / name
    return Path(run) / "checkpoint_best_total.pth"


if __name__ == "__main__":
    main()
