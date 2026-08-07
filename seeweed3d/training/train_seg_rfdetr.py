#!/usr/bin/env python3
"""
SeeWeed3D - Stage A training on RF-DETR-Seg (Apache-2.0).

    python -m seeweed3d.training.train_seg_rfdetr \
        --dataset D:/training/subset45 --out D:/runs/rfdetr_v1 \
        --variant medium --resolution 1008 --epochs 60 --device cuda

WHY THIS EXISTS ALONGSIDE THE MASK R-CNN BACKEND
------------------------------------------------
Things RF-DETR provides natively that torchvision's Mask R-CNN does not expose
without forking its ROI heads - verified against rfdetr 1.9.1, not assumed:

    mask_ce_loss_coef / mask_dice_loss_coef   Dice + CE mask loss, weighted
    cls_loss_coef, ia_bce_loss                IA-BCE classification loss
    use_ema / ema_decay                       exponential moving average weights
    early_stopping{,_patience,_min_delta}     built in
    multi_scale / expanded_scales             multi-scale training
    grad_accum_steps + auto_batch_*           small-VRAM training at an
                                              effective batch of 16
    tensorboard / mlflow                      native logging flags

It is also DETR-style set prediction: no anchors, so the anchor/resolution trap
that cost the Mask R-CNN path its small weeds cannot occur in the same form.
And it is the real-time architecture for the Jetson Orin, with ONNX/TensorRT
export in the package.

RESOLUTION IS THE WHOLE GAME HERE - READ THIS
---------------------------------------------
RF-DETR-Seg's DEFAULT is 432x432. A 2208x1242 ZED frame at 432 px loses small
weeds far more severely than the 1333 px Mask R-CNN default already did. Left
alone, switching to this model would REGRESS small-weed recall while looking
like an upgrade.

So --resolution is required to be chosen deliberately. It must be a multiple of
`patch_size * num_windows` for the variant (24 for the Seg variants, checked
here rather than left to a ValueError deep inside training), and this script
refuses to run at the default without --allow-default-resolution.

VRAM: cost grows with the square of resolution. RF-DETR handles this properly
through gradient accumulation rather than by shrinking the effective batch -
--batch 2 with --grad-accum 8 trains at an effective batch of 16.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Seg variant -> the rfdetr class name. Nano..Large are Apache-2.0; XLarge and
#: 2XLarge may fall under Roboflow's Platform Model License, so they are not
#: offered here - see docs/edge_model_research.md.
VARIANTS = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def resolution_step(variant):
    """The multiple a resolution must be for this variant, from the package's
    own config rather than a hard-coded 24 that could drift."""
    from rfdetr import config as C
    cfg = getattr(C, f"RFDETRSeg{variant.capitalize()}Config")
    return (cfg.model_fields["patch_size"].default
            * cfg.model_fields["num_windows"].default)


def default_resolution(variant):
    from rfdetr import config as C
    cfg = getattr(C, f"RFDETRSeg{variant.capitalize()}Config")
    return cfg.model_fields["resolution"].default


def check_resolution(variant, resolution):
    """Validate here, with the valid neighbours named.

    rfdetr does raise on a bad value, but only after the dataset has loaded and
    weights have downloaded, and the message does not say which values would
    work."""
    step = resolution_step(variant)
    if resolution % step:
        near = [k * step for k in range(max(1, resolution // step - 2),
                                        resolution // step + 3)]
        raise SystemExit(
            f"ERROR: --resolution {resolution} is not a multiple of {step}, "
            f"which RF-DETR-Seg-{variant} requires "
            f"(patch_size x num_windows).\n"
            f"Nearest valid values: {', '.join(str(n) for n in near)}")
    return resolution


def train(dataset_dir, out_dir, variant="medium", resolution=None, epochs=60,
          batch=2, grad_accum=8, lr=1e-4, device="cuda", workers=2,
          coco_dir=None, images_root=None, link=False, overwrite=False,
          early_stopping=True, patience=10, use_ema=True,
          mask_ce_coef=None, mask_dice_coef=None, cls_coef=None,
          multi_scale=True, track="auto", allow_default_resolution=False,
          extra=None):
    if variant not in VARIANTS:
        raise SystemExit(f"ERROR: --variant must be one of "
                         f"{sorted(VARIANTS)}; got {variant!r}")
    from common.torch_utils import require_device
    device = require_device(device)

    # Decide on tracking, and point lightning at our store, BEFORE importing
    # rfdetr - see _point_lightning_at_our_mlflow_store for why the order is
    # load-bearing. Only find_spec and an environment variable here; nothing
    # touches the disk until every guard below has passed.
    want_tb = track in ("auto", "all", "tensorboard")
    want_mf = track in ("auto", "all", "mlflow")
    if track in ("mlflow", "all") and not _installed("mlflow"):
        raise SystemExit(f"ERROR: --track {track} needs mlflow:\n"
                         f'    "{sys.executable}" -m pip install mlflow')
    if track == "auto":
        want_tb, want_mf = _installed("tensorboard"), _installed("mlflow")
    mlflow_uri = (_point_lightning_at_our_mlflow_store(out_dir)
                  if want_mf else None)

    try:
        import rfdetr  # noqa: F401
    except ImportError:
        raise SystemExit(
            f"ERROR: rfdetr is not installed in the interpreter that is "
            f"running:\n    {sys.executable}\n"
            f'    "{sys.executable}" -m pip install rfdetr\n'
            f"RF-DETR is Apache-2.0, so unlike the ultralytics backend it "
            f"carries no source-release obligation.")

    dflt = default_resolution(variant)
    if resolution is None:
        if not allow_default_resolution:
            step = resolution_step(variant)
            raise SystemExit(
                f"ERROR: --resolution not given, and RF-DETR-Seg-{variant}'s "
                f"default is {dflt}x{dflt}.\n"
                f"A 2208x1242 ZED frame at {dflt} px loses small weeds far "
                f"more severely than the Mask R-CNN path already did, so "
                f"defaulting here would REGRESS small-weed recall while "
                f"looking like an upgrade.\n"
                f"Choose deliberately - any multiple of {step}, e.g. "
                f"{step * 42} ({step * 42} px) or {step * 56}. Higher costs "
                f"VRAM quadratically; raise --grad-accum rather than dropping "
                f"resolution.\n"
                f"Pass --allow-default-resolution to use {dflt} anyway.")
        resolution = dflt
    resolution = check_resolution(variant, resolution)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coco = Path(coco_dir) if coco_dir else out / "coco"

    from training.coco_export import export
    summary = export(dataset_dir, coco, images_root, link=link,
                     overwrite=overwrite or not coco_dir)
    n_classes = len(summary["classes"])
    print(f"  COCO export -> {coco}")
    for k, v in summary["splits"].items():
        print(f"    {k:<6} {v['frames']:>4} frames  {v['instances']:>5} inst")

    # Now that every guard has passed, create the store and confirm lightning
    # will use it. rfdetr's own tensorboard/mlflow flags do the logging -
    # reusing them rather than wrapping its Lightning loop keeps this backend
    # thin.
    if want_mf:
        want_mf = _mlflow_store_is_reachable(out, mlflow_uri)

    from training.tracking import EXPERIMENT
    cfg = {
        "dataset_dir": str(coco),
        "output_dir": str(out),
        "epochs": epochs,
        "batch_size": batch,
        "grad_accum_steps": grad_accum,
        "lr": lr,
        "device": device,
        "num_workers": workers,
        "resolution": resolution,
        "use_ema": use_ema,
        "early_stopping": early_stopping,
        "early_stopping_patience": patience,
        "multi_scale": multi_scale,
        "tensorboard": want_tb,
        "mlflow": want_mf,
        # rfdetr names the MLflow experiment after `project` and the run after
        # `run`. Matching the Mask R-CNN tracker's experiment is what puts both
        # backends in one comparison table instead of two.
        "project": EXPERIMENT,
        "run": out.name,
    }
    for key, val in (("mask_ce_loss_coef", mask_ce_coef),
                     ("mask_dice_loss_coef", mask_dice_coef),
                     ("cls_loss_coef", cls_coef)):
        if val is not None:
            cfg[key] = val
    cfg.update(extra or {})

    if workers and sys.platform == "win32":
        print(f"\n  [!] workers={workers} on Windows. If training stops right "
              f"after the\n      model-summary table with no traceback, that "
              f"is a lightning\n      dataloader worker failing to start - "
              f"set WORKERS to 0 and retry.\n")

    print(f"\n  RF-DETR-Seg-{variant} | {n_classes} classes | "
          f"resolution {resolution} (default {dflt}) | "
          f"effective batch {batch * grad_accum}")
    print(f"  losses: mask_ce={cfg.get('mask_ce_loss_coef', 'default')} "
          f"mask_dice={cfg.get('mask_dice_loss_coef', 'default')} "
          f"cls={cfg.get('cls_loss_coef', 'default')}")
    (out / "rfdetr_train_config.json").write_text(
        json.dumps({**cfg, "variant": variant, "classes": summary["classes"]},
                   indent=2), encoding="utf-8")

    import rfdetr as R
    model = getattr(R, VARIANTS[variant])(num_classes=n_classes,
                                          resolution=resolution)
    model.train(**cfg)
    print(f"\n-> {out}")
    return cfg


def _installed(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def _point_lightning_at_our_mlflow_store(out_dir):
    """Return the tracking URI, and make pytorch-lightning use it.

    rfdetr does not build its MLFlowLogger with a tracking_uri, so lightning
    falls back to `save_dir`, i.e. the './mlruns' FILE store - which MLflow 3
    refuses outright, killing the run before the first epoch. Setting the
    variable is also what makes the claim "RF-DETR runs land in the same store
    as the Mask R-CNN runs" true rather than aspirational; without it they
    would go to whatever directory python happened to be launched from.

    MUST be called before rfdetr (hence lightning) is imported: lightning
    captures os.getenv("MLFLOW_TRACKING_URI") as a DEFAULT ARGUMENT, evaluated
    once when its module is imported. No I/O here for that reason - the store
    is only created once every other guard has passed.
    """
    from training.tracking import mlflow_store_uri
    uri, _ = mlflow_store_uri(out_dir)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    return uri


def _mlflow_store_is_reachable(out_dir, uri):
    """Create the store, register the shared experiment, and confirm lightning
    will actually use the URI. Returns False to mean "log without MLflow",
    never an exception: losing a training run to a charting library is never
    the right trade."""
    try:
        import mlflow
        from training.tracking import EXPERIMENT, mlflow_store_uri
        _, artifacts = mlflow_store_uri(out_dir, create=True)
        mlflow.set_tracking_uri(uri)
        # With a database backend the artifact root is not implied by the
        # tracking URI, so preview images would otherwise land next to the
        # working directory.
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(
                EXPERIMENT,
                artifact_location=artifacts.as_uri() if artifacts else None)
    except Exception as e:                      # noqa: BLE001
        print(f"  [warn] mlflow disabled for this run: {type(e).__name__}: {e}")
        return False

    _rebind_lightning_tracking_uri(uri)
    return True


def _rebind_lightning_tracking_uri(uri):
    """Belt-and-braces for the import-order hazard above.

    The environment variable is the supported mechanism and works whenever
    rfdetr is imported after us, which is the normal path. But lightning froze
    its default at import time, so if something pulled it in first the variable
    is ignored and rfdetr would reach the file store. Detect that and bind the
    URI at the call site rfdetr actually uses.

    Failure here is not a reason to give up MLflow: in the ordinary case the
    variable alone is enough, and a missing lightning means rfdetr cannot train
    at all - a problem this function is not the right place to report.
    """
    try:
        import inspect

        from pytorch_lightning.loggers import MLFlowLogger
        baked = inspect.signature(
            MLFlowLogger.__init__).parameters["tracking_uri"].default
        if str(baked or "") == uri:
            return True
        import rfdetr.training.trainer as T
        original = T.MLFlowLogger
        T.MLFlowLogger = (lambda *a, _o=original, **k:
                          _o(*a, **{"tracking_uri": uri, **k}))
        return True
    except Exception:                           # noqa: BLE001
        return False


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="make_dataset.py OUT_DIR")
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="medium", choices=sorted(VARIANTS))
    p.add_argument("--resolution", type=int, default=None,
                   help="MUST be chosen deliberately; a multiple of 24 for the "
                        "Seg variants. The 432 default loses small weeds.")
    p.add_argument("--allow-default-resolution", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="batch x grad-accum is the effective batch; keep ~16")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--images-root", nargs="*", default=None)
    p.add_argument("--coco-dir", default=None,
                   help="reuse an existing COCO export instead of rebuilding")
    p.add_argument("--link", action="store_true",
                   help="hardlink images instead of copying")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--mask-ce-coef", type=float, default=None)
    p.add_argument("--mask-dice-coef", type=float, default=None)
    p.add_argument("--cls-coef", type=float, default=None)
    p.add_argument("--no-multi-scale", action="store_true")
    p.add_argument("--track", default="auto",
                   choices=["auto", "none", "tensorboard", "mlflow", "all"])
    a = p.parse_args(argv)
    roots = a.images_root if a.images_root else None
    if roots and len(roots) == 1:
        roots = roots[0]
    train(a.dataset, a.out, variant=a.variant, resolution=a.resolution,
          epochs=a.epochs, batch=a.batch, grad_accum=a.grad_accum, lr=a.lr,
          device=a.device, workers=a.workers, coco_dir=a.coco_dir,
          images_root=roots, link=a.link, overwrite=a.overwrite,
          early_stopping=not a.no_early_stopping, patience=a.patience,
          use_ema=not a.no_ema, mask_ce_coef=a.mask_ce_coef,
          mask_dice_coef=a.mask_dice_coef, cls_coef=a.cls_coef,
          multi_scale=not a.no_multi_scale, track=a.track,
          allow_default_resolution=a.allow_default_resolution)


if __name__ == "__main__":
    main()
