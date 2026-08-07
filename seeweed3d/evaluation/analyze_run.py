#!/usr/bin/env python3
"""
SeeWeed3D - everything a finished run should tell you (EDIT THE CONFIG BELOW)
============================================================================
    python seeweed3d/evaluation/analyze_run.py

Point it at a run directory. It detects the backend, reads whatever training
history that backend wrote, scores the checkpoint with the confidence sweep,
draws the figures, builds the visual report, and files the lot in MLflow beside
every other run.

WHY NOT JUST USE A HOSTED TRACKER
---------------------------------
Because the tracker is not the missing piece. W&B, MLflow, TensorBoard and
Comet all store scalars, images and tables; none of them knows what a missed
onion is, that recall at conf 0.5 and 0.25 are different questions, or how to
draw a mask on a plant. Every figure here has to be COMPUTED either way - the
tracker only decides where the PNG is filed. Switching to a hosted service
would move that one call and save none of this.

What it does buy is a shared URL and a comparison UI. That is a real
convenience and a real decision about where field imagery and a customer's row
geometry live, which is why it is not made by default. `TRACK` sends these
artifacts wherever Tracker is pointed.

WHAT IT PRODUCES
----------------
    analysis/training_curves.png    is it still learning, or was it cut off?
    analysis/per_class_ap.png       which class holds the headline number down
    analysis/confidence_sweep.png   THE deployment-threshold decision
    analysis/crop_safety.png        did it get safer, or just better at weeds?
    analysis/recall_by_size.png     the failure this system is limited by
    analysis/report.html            GT-vs-prediction panels, missed-weed gallery
    metrics_<split>.json            the numbers behind all of it
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

CONFIG = {
    # The RUN_DIR from a training run - either backend.
    "RUN_DIR": r"E:\Dataset_Vidalia\training1\run4",

    # OUT_DIR from make_dataset.py (the folder holding seg_manifest.json).
    "DATASET_DIR": r"E:\Dataset_Vidalia\training1",

    "SPLIT": "val",
    "DEVICE": "cuda",

    # The confidence the P/R table is reported at. The sweep covers the rest.
    "CONF": 0.25,

    # "" auto-detects from the run directory. Override to score a specific
    # checkpoint, e.g. an EMA file you want to compare against the regular one.
    "CHECKPOINT": "",
    "BACKEND": "",

    # File the figures and the report in MLflow too, so both backends' runs sit
    # in one comparison table. "none" writes to disk only.
    "TRACK": "auto",
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

#: rfdetr's metrics.csv column -> the name used by the Mask R-CNN history, so
#: one set of figures serves both backends. segm_* is chosen over the plain
#: mAP columns because eval_seg reports MASK AP; pairing a box curve with a
#: mask number would make the two backends look different for the wrong reason.
RFDETR_COLUMNS = {
    "train/loss": "train_loss",
    "val/loss": "val_loss",
    "val/segm_mAP_50": "val_map50",
    "val/segm_mAP_50_95": "val_map50_95",
}


def detect(run_dir):
    """(backend, checkpoint) for a run directory.

    Detected from what training actually wrote rather than asked for, so a
    directory holding both is unambiguous instead of silently preferring one."""
    run = Path(run_dir)
    if not run.is_dir():
        raise SystemExit(f"ERROR: RUN_DIR does not exist: {run}")
    if (run / "rfdetr_train_config.json").exists():
        for name in ("checkpoint_best_total.pth", "checkpoint_best_regular.pth",
                     "checkpoint_best_ema.pth"):
            if (run / name).exists():
                return "rfdetr", run / name
        raise SystemExit(
            f"ERROR: {run} looks like an RF-DETR run but holds no "
            f"checkpoint_best_*.pth. Did training finish?")
    if (run / "best.pt").exists():
        return "maskrcnn", run / "best.pt"
    raise SystemExit(
        f"ERROR: no checkpoint found in {run}.\n"
        f"Expected best.pt (Mask R-CNN) or checkpoint_best_total.pth with "
        f"rfdetr_train_config.json (RF-DETR).")


def load_history(run_dir, backend):
    """Training history as [{epoch, train_loss, val_loss, val_map50, ...}].

    Mask R-CNN writes history.json directly. RF-DETR writes a lightning
    metrics.csv with one row per logging event and blanks everywhere a metric
    was not produced, so train and val land on SEPARATE rows for the same
    epoch - they are merged here, or every curve would be half gaps."""
    run = Path(run_dir)
    if backend == "maskrcnn":
        p = run / "history.json"
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return []

    csvs = sorted(run.rglob("metrics.csv"))
    if not csvs:
        return []
    merged = {}
    with csvs[0].open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                ep = int(float(row.get("epoch") or 0))
            except ValueError:
                continue
            out = merged.setdefault(ep, {"epoch": ep})
            for src, dst in RFDETR_COLUMNS.items():
                v = row.get(src, "")
                if v not in ("", None):
                    out[dst] = float(v)
            for k, v in row.items():
                if k.startswith("val/AP/") and v not in ("", None):
                    out[f"ap_{k.split('/')[-1]}"] = float(v)
    return [merged[k] for k in sorted(merged)]


def analyze(cfg=None):
    c = dict(CONFIG if cfg is None else cfg)
    run = Path(c["RUN_DIR"])
    backend, ckpt = detect(run)
    if c.get("BACKEND"):
        backend = c["BACKEND"]
    if c.get("CHECKPOINT"):
        ckpt = Path(c["CHECKPOINT"])

    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    print(f"  {backend} | {ckpt.name} | split={c['SPLIT']}")

    from evaluation.eval_seg import DEFAULT_SWEEP, evaluate, format_report
    res = evaluate(ckpt, c["DATASET_DIR"], None, c["SPLIT"], c["DEVICE"],
                   conf=c["CONF"], backend=backend, sweep=DEFAULT_SWEEP)
    (run / f"metrics_{c['SPLIT']}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(format_report(res))

    history = load_history(run, backend)
    if not history:
        print(f"  [note] no training history found in {run}; the loss and "
              f"crop-safety curves need it")

    # The visual report doubles as the source of the size table, so the two
    # can never disagree about which instances were missed. Driven through its
    # CLI rather than a copy of its internals - one assembly path, so the page
    # and the figure cannot drift apart.
    size_rows, html = [], None
    try:
        from evaluation import report as _report
        html = out / "report.html"
        _report.main(["--checkpoint", str(ckpt), "--dataset",
                      str(c["DATASET_DIR"]), "--split", c["SPLIT"],
                      "--device", c["DEVICE"], "--conf", str(c["CONF"]),
                      "--backend", backend, "--out", str(html)])
        side = html.with_suffix(".json")
        if side.exists():
            size_rows = json.loads(
                side.read_text(encoding="utf-8")).get("recall_by_size") or []
    except Exception as e:                              # noqa: BLE001
        print(f"  [warn] visual report skipped: {type(e).__name__}: {e}")
        html = None

    from evaluation.plots import figures_for_run
    figs = figures_for_run(history, res, size_rows, out)
    for name, p in figs.items():
        print(f"    {name:<18} -> {p}")

    _track(c, run, backend, res, history, figs, html)
    print(f"\n-> {out}")
    return {"backend": backend, "checkpoint": str(ckpt), "metrics": res,
            "figures": {k: str(v) for k, v in figs.items()}}


def _track(c, run, backend, res, history, figs, html):
    """File the artifacts alongside the training runs.

    Never fatal. The figures are already on disk by this point, and losing them
    to a charting service would be the tail wagging the dog."""
    if c.get("TRACK", "auto") == "none":
        return
    try:
        from training.tracking import Tracker
        t = Tracker(backend=c.get("TRACK", "auto"), out_dir=run,
                    run_name=f"{run.name}-analysis")
        t.log_params({"backend": backend, "split": c["SPLIT"],
                      "conf": c["CONF"], "run_dir": str(run)})
        op, cs = res["operating_point"], res["crop_safety"]
        flat = {"map50": res["summary"]["map50"],
                "map50_95": res["summary"]["map50_95"],
                "small_weed_recall": op.get("small_weed_recall"),
                "missed_onion_fraction": cs.get("missed_onion_fraction"),
                "weed_on_crop_fraction": cs.get("weed_on_crop_fraction")}
        for cls, d in res["detection"].items():
            flat[f"ap50/{cls}"] = d.get("ap50")
            flat[f"ap50_95/{cls}"] = d.get("ap50_95")
        t.log_metrics({k: v for k, v in flat.items() if v is not None})
        for p in list(figs.values()) + ([html] if html else []):
            t.log_artifact(p)
        t.close()
    except Exception as e:                              # noqa: BLE001
        print(f"  [warn] tracking skipped: {type(e).__name__}: {e}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run")
    p.add_argument("--dataset")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--device")
    p.add_argument("--conf", type=float)
    p.add_argument("--backend", choices=["maskrcnn", "rfdetr"])
    p.add_argument("--checkpoint")
    p.add_argument("--track", choices=["auto", "none", "mlflow", "tensorboard",
                                       "all"])
    a = p.parse_args(argv)
    c = dict(CONFIG)
    for flag, key in (("run", "RUN_DIR"), ("dataset", "DATASET_DIR"),
                      ("split", "SPLIT"), ("device", "DEVICE"),
                      ("conf", "CONF"), ("backend", "BACKEND"),
                      ("checkpoint", "CHECKPOINT"), ("track", "TRACK")):
        v = getattr(a, flag)
        if v is not None:
            c[key] = v
    return analyze(c)


if __name__ == "__main__":
    main()
