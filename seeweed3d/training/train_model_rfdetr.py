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
    "RUN_DIR": r"E:\Dataset_Vidalia\training1\rfdetr_v1",

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
    "EARLY_STOPPING": True,
    "PATIENCE": 10,

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
          mask_ce_coef=c["MASK_CE_COEF"], mask_dice_coef=c["MASK_DICE_COEF"],
          cls_coef=c["CLS_COEF"], track=c["TRACK"])

    print(f"\nNext, score it with the SAME table the Mask R-CNN runs use:\n"
          f"  python -m seeweed3d.evaluation.eval_seg --backend rfdetr "
          f"--checkpoint {run}/checkpoint_best_ema.pth --dataset {ds} "
          f"--split val --device {c['DEVICE']}\n"
          f"  python -m seeweed3d.evaluation.report --backend rfdetr "
          f"--checkpoint {run}/checkpoint_best_ema.pth --dataset {ds} "
          f"--split val --device {c['DEVICE']}\n")


if __name__ == "__main__":
    main()
