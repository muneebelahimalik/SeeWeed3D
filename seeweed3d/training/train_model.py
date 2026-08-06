#!/usr/bin/env python3
"""
SeeWeed3D - train Stage A and evaluate it (EDIT THE CONFIG BELOW)
================================================================
Runs `train_seg_torchvision` then `eval_seg`, driven by the config block in this
file. Edit, save, run:

    python seeweed3d/training/train_model.py

Build the dataset first with `make_dataset.py`.

WHAT TO WATCH
-------------
Loss is not a quality number. Two things tell you whether this is working:

  * The PREVIEW PANELS (TensorBoard -> Images tab). Ground truth left,
    prediction right, on a fixed sample of val frames. At 45 annotated frames
    this beats every scalar - a loss curve cannot show you that every mask is
    one plant too large, or that the model has learned to call every onion a
    weed.
  * The EVAL TABLE printed at the end: mAP, per-class precision/recall, and
    missed onion pixels kept out of every averaged score.
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
    "DATASET_DIR": r"E:\Dataset_Vidalia\Weeds_3_good\training\subset45",

    # The sessions root(s) - a single path, or a LIST of them if make_dataset.py
    # merged sources under more than one parent folder (e.g. a weed sessions
    # folder plus a separately-recorded onion sessions folder). Leave "" to
    # reuse exactly what make_dataset.py already recorded in seg_manifest.json,
    # which is the normal choice - there is then nothing to keep in sync
    # between the two files.
    "IMAGES_ROOT": "",

    # Where checkpoints, curves and metrics go. One folder per run - do not
    # reuse it, or you lose the comparison.
    "RUN_DIR": r"E:\Dataset_Vidalia\Weeds_3_good\runs\seg_v1",

    # -- Training --------------------------------------------------------------
    "DEVICE": "cuda",          # "cuda", or "cpu" if you have no GPU (very slow)

    # ~34 training frames is ~17 steps per epoch at BATCH 2. The default of 20
    # epochs is only ~340 steps, nowhere near enough to fit a fresh head.
    "EPOCHS": 60,

    # Drop to 1 on CUDA out-of-memory - ZED frames are large and Mask R-CNN v2
    # at full resolution is heavy. If you do, halve LR too.
    "BATCH": 2,

    # Mask R-CNN's reference schedule is 0.02 at batch 16; linear scaling to
    # batch 2 gives 2.5e-3. The module default of 5e-3 is twice that and can
    # diverge on a set this small.
    "LR": 2.5e-3,

    # Dataloader processes. Full-resolution decode is the bottleneck; 0 makes
    # the GPU wait on the CPU. Set 0 if you hit Windows multiprocessing errors.
    "WORKERS": 4,

    # COCO-pretrained weights. Downloaded once, needs network access. NOT
    # optional at this data volume - 34 frames cannot train a backbone from
    # scratch.
    "PRETRAINED": True,

    "SEED": 1234,

    # -- Monitoring ------------------------------------------------------------
    # "auto" uses whatever of tensorboard/mlflow is installed and never fails.
    # "all" requires both and errors if either is missing. "none" disables it.
    # Everything is written locally; nothing is uploaded anywhere.
    #     python -m pip install tensorboard mlflow
    "TRACK": "auto",

    # GT-vs-prediction overlay panels every N epochs. 0 disables.
    "PREVIEW_EVERY": 5,

    # Val mAP every N epochs. 0 disables. A full pass over val, so it costs
    # real time on a large val split; the final evaluation below runs
    # regardless. Forced to 1 if SELECT_BY needs mAP and this is 0.
    "EVAL_EVERY": 0,

    # What best.pt is chosen on: "val_loss" | "map50" | "map50_95".
    #
    # val_loss is the default only because it is free. It is a poor proxy for a
    # detector - it sums classification, box and mask terms whose scales have
    # nothing to do with whether a plant was found - and on a small set it
    # often bottoms out long before detection quality peaks. If your run
    # reports a best epoch early in the schedule, switch to "map50_95" and
    # compare: it selects on the number you actually report.
    "SELECT_BY": "val_loss",

    # -- Evaluation ------------------------------------------------------------
    "EVALUATE_AFTER": True,
    "EVAL_SPLIT": "val",       # "val", or "test" once you have a real one

    # The confidence you would actually deploy at, for the precision/recall
    # table. mAP is threshold-free; a robot is not.
    "EVAL_CONF": 0.5,
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################


def _resolve_images_root(c, manifest_path):
    """CONFIG['IMAGES_ROOT'] if set, else whatever make_dataset.py already
    recorded in seg_manifest.json - so a multi-source build does not need its
    root list typed out a second time in a second file."""
    given = c.get("IMAGES_ROOT")
    if isinstance(given, (list, tuple)):
        roots = [str(r) for r in given if str(r).strip()]
    elif str(given or "").strip():
        roots = [str(given)]
    else:
        import json as _json
        doc = _json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = doc.get("images_root")
        roots = stored if isinstance(stored, list) else ([stored] if stored else [])
        if not roots:
            raise SystemExit(
                f"ERROR: CONFIG['IMAGES_ROOT'] is empty and {manifest_path} "
                f"has no 'images_root' recorded either. Set IMAGES_ROOT.")

    missing = [r for r in roots if not Path(r).exists()]
    if missing:
        raise SystemExit(f"ERROR: IMAGES_ROOT path(s) do not exist: {missing}")
    return roots[0] if len(roots) == 1 else roots


def main(cfg=None):
    c = dict(CONFIG if cfg is None else cfg)

    ds = Path(c["DATASET_DIR"])
    man_path = ds / "seg_manifest.json"
    if not man_path.exists():
        raise SystemExit(
            f"ERROR: {man_path} not found.\n"
            f"Build the dataset first: edit and run "
            f"seeweed3d/training/make_dataset.py")
    images = _resolve_images_root(c, man_path)

    run = Path(c["RUN_DIR"])
    if (run / "best.pt").exists():
        print(f"\n[!] {run / 'best.pt'} already exists and WILL BE "
              f"OVERWRITTEN.\n    Point RUN_DIR at a new folder to keep the "
              f"old run comparable.\n")

    from training.train_seg_torchvision import train
    train(ds, images, run,
          epochs=c["EPOCHS"], batch=c["BATCH"], lr=c["LR"],
          device=c["DEVICE"], workers=c["WORKERS"], seed=c["SEED"],
          pretrained=c["PRETRAINED"], track=c["TRACK"],
          preview_every=c["PREVIEW_EVERY"], eval_every=c["EVAL_EVERY"],
          select_by=c.get("SELECT_BY", "val_loss"))

    if not c["EVALUATE_AFTER"]:
        return
    ckpt = run / "best.pt"
    if not ckpt.exists():
        print("\n[!] no best.pt was written, so there is nothing to evaluate. "
              "That happens when the val split is empty - check "
              "splits/splits_summary.json.\n")
        return

    import json
    from evaluation.eval_seg import evaluate, format_report
    print("\n" + "=" * 74)
    res = evaluate(ckpt, ds, images, c["EVAL_SPLIT"], c["DEVICE"],
                   conf=c["EVAL_CONF"])
    print(format_report(res))
    out = run / f"metrics_{c['EVAL_SPLIT']}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    print("\nSingle-session splits: val comes from the same drive as train - "
          "same light,\nsame soil, often the same plants. These numbers show "
          "training WORKS. They are\nnot evidence it generalises; only a "
          "held-out session can show that.\n")


if __name__ == "__main__":
    main()
