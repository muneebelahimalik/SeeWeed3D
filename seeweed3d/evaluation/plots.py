#!/usr/bin/env python3
"""
SeeWeed3D - the figures a training run should leave behind.

WHY THIS EXISTS RATHER THAN A HOSTED DASHBOARD
----------------------------------------------
No experiment tracker draws these for you. W&B, MLflow, TensorBoard and Comet
all store scalars, images and tables; not one of them knows what a missed onion
is, or that recall at conf 0.5 and conf 0.25 are different questions. The mask
overlays, the crop-safety curve and the confidence sweep have to be COMPUTED
here either way - the tracker only decides where the resulting PNG is filed.

So these are plain matplotlib figures written to disk. Tracker then files them
wherever it is pointed, which today is a local MLflow store; swapping that for a
hosted service later changes one call and none of this module.

WHAT EACH FIGURE IS FOR
-----------------------
    training_curves     is it still learning, or is the schedule over?
    per_class_ap        which class is holding the headline number down
    confidence_sweep    THE deployment decision - see docs/RUNBOOK.md §8.1
    crop_safety_curve   did the model get safer or just better at weeds?
    recall_by_size      the failure this project is actually limited by
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CROP_CLASS  # noqa: E402

#: Crop safety is drawn in the same orange the overlays use; weeds in green.
C_CROP = "#ff8c00"
C_BURN = "#d62728"
C_WEED = "#2ca02c"
C_SMALL = "#1f77b4"
C_ALLWEED = "#9467bd"
C_PREC = "#7f7f7f"


def _plt():
    """matplotlib with a headless backend, imported late.

    Late because plotting must never be a reason a training run cannot start:
    the figures are a by-product, and a missing matplotlib should cost you the
    pictures and nothing else."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    _plt().close(fig)
    return out


def _series(history, key):
    """(epochs, values) for the rows where `key` is present.

    Mid-training evaluation is usually every N epochs, so these series are
    SHORTER than the loss curve and their x values are not 0..n. Plotting them
    against a range() would silently compress them onto the wrong epochs."""
    xs, ys = [], []
    for row in history:
        v = row.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(f):
            continue
        xs.append(float(row.get("epoch", len(xs))))
        ys.append(f)
    return xs, ys


def training_curves(history, out):
    """Loss and val mAP against epoch.

    The question it answers is whether the run ENDED or was CUT OFF. A best
    epoch in the last third means the schedule was too short - which is exactly
    what the first RF-DETR run looked like."""
    plt = _plt()
    # Nothing to draw is not a figure. An empty pair of axes filed next to the
    # real ones reads as a run that flatlined rather than one with no history.
    if not any(_series(history, k)[0] for k in
               ("train_loss", "val_loss", "val_map50", "val_map50_95")):
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for key, label in (("train_loss", "train"), ("val_loss", "val")):
        xs, ys = _series(history, key)
        if xs:
            ax1.plot(xs, ys, label=label, lw=1.6)
    ax1.set_xlabel("epoch"), ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend()

    for key, label in (("val_map50", "mAP@50"), ("val_map50_95", "mAP@50:95")):
        xs, ys = _series(history, key)
        if xs:
            ax2.plot(xs, ys, marker="o", ms=3, label=label)
            best = int(np.argmax(ys))
            ax2.annotate(f"best e{int(xs[best])}", (xs[best], ys[best]),
                         textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax2.set_xlabel("epoch"), ax2.set_ylabel("mAP")
    ax2.margins(y=0.12)                     # headroom for the best-epoch tags
    ax2.set_title("Validation mAP")
    ax2.grid(alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend()
    fig.tight_layout()
    return _save(fig, out)


def per_class_ap(metrics, out):
    """AP per class, with the ground-truth count on each bar.

    The count matters: an AP computed over 19 instances is not a measurement,
    and a bar chart without it invites reading noise as a result."""
    plt = _plt()
    det = metrics.get("detection") or {}
    names = [c for c in det if det[c].get("ap50") is not None]
    if not names:
        return None
    ap50 = [det[c]["ap50"] for c in names]
    ap = [det[c].get("ap50_95") or 0.0 for c in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 3, 4))
    ax.bar(x - 0.2, ap50, 0.4, label="AP@50",
           color=[C_CROP if c == CROP_CLASS else C_WEED for c in names])
    ax.bar(x + 0.2, ap, 0.4, label="AP@50:95", alpha=0.55,
           color=[C_CROP if c == CROP_CLASS else C_WEED for c in names])
    for k, c in enumerate(names):
        ax.text(x[k], -0.06, f"n={det[c]['n_gt']}", ha="center", va="top",
                fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP")
    ax.set_title("Per-class average precision")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def confidence_sweep(metrics, out):
    """Recall, precision and CROP DAMAGE against the deployment threshold.

    The plot the operating point is chosen from. Burn is on its own axis
    because it lives three orders of magnitude below the others - drawn on a
    shared axis it would be a flat line along zero, which is exactly the
    impression that must not be given."""
    plt = _plt()
    rows = metrics.get("conf_sweep") or []
    if len(rows) < 2:
        return None
    conf = [r["conf"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Explicit colours: matplotlib's default cycle puts orange second, which
    # would collide with the orange reserved for the crop everywhere else.
    for key, label, style in (
            ("small_weed_recall", "small-weed recall",
             dict(lw=2.4, color=C_SMALL)),
            ("weed_recall", "weed recall", dict(lw=1.6, color=C_ALLWEED)),
            ("weed_precision", "weed precision",
             dict(lw=1.6, ls="--", color=C_PREC)),
            ("crop_recall", "onion recall", dict(lw=1.6, ls=":",
                                                 color=C_CROP))):
        ys = [r.get(key) for r in rows]
        if any(v is not None for v in ys):
            ax.plot(conf, [np.nan if v is None else v for v in ys],
                    marker="o", ms=3, label=label, **style)
    ax.set_xlabel("deployment confidence")
    ax.set_ylabel("instance rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    burn = [r.get("weed_on_crop_fraction") for r in rows]
    if any(v for v in burn):
        ax2 = ax.twinx()
        ax2.plot(conf, [0.0 if v is None else v for v in burn],
                 color=C_BURN, lw=2, marker="s", ms=4,
                 label="onion burned (right axis)")
        ax2.set_ylabel("fraction of onion pixels fired at", color=C_BURN)
        ax2.tick_params(axis="y", labelcolor=C_BURN)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8)
    ax.set_title("Choosing the deployment confidence")
    fig.tight_layout()
    return _save(fig, out)


def crop_safety_curve(history, out):
    """Missed onion against epoch.

    Separate from mAP on purpose. A run can improve its headline number while
    getting worse at the only failure that destroys a customer's crop, and a
    combined chart would average that away."""
    plt = _plt()
    xs, ys = _series(history, "missed_onion_fraction")
    if len(xs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", ms=3, color=C_CROP, lw=2,
            label="missed onion")
    bx, by = _series(history, "weed_on_crop_fraction")
    if len(bx) >= 2:
        ax.plot(bx, by, marker="s", ms=3, color=C_BURN, lw=2,
                label="onion called weed")
    ax.set_xlabel("epoch"), ax.set_ylabel("fraction of onion pixels")
    ax.set_title("Crop safety over training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def recall_by_size(rows, out):
    """Recall against instance area - the metric this system is limited by.

    Overall mAP is dominated by the large easy plants; this is where a
    cotyledon-stage weed shows up as missed."""
    plt = _plt()
    rows = [r for r in (rows or []) if r.get("n_gt")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rows))
    vals = [r["recall"] if r.get("recall") is not None else 0.0 for r in rows]
    ax.bar(x, vals, 0.6, color=C_WEED)
    for k, r in enumerate(rows):
        ax.text(k, vals[k] + 0.02, f"{r['n_found']}/{r['n_gt']}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["range_px"] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("instance area (px)"), ax.set_ylabel("recall")
    ax.set_title("Recall by object size")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def figures_for_run(history, metrics, size_rows, out_dir):
    """Every figure this run supports. Returns {name: path}.

    A figure whose inputs are absent is OMITTED rather than drawn empty: a
    blank axis in a report reads as a measured zero."""
    out_dir = Path(out_dir)
    made = {}
    for name, fn, arg in (
            ("training_curves", training_curves, history),
            ("per_class_ap", per_class_ap, metrics),
            ("confidence_sweep", confidence_sweep, metrics),
            ("crop_safety", crop_safety_curve, history),
            ("recall_by_size", recall_by_size, size_rows)):
        try:
            p = fn(arg, out_dir / f"{name}.png")
        except Exception as e:                          # noqa: BLE001
            print(f"  [warn] figure {name} skipped: {type(e).__name__}: {e}")
            continue
        if p is not None:
            made[name] = p
    return made
