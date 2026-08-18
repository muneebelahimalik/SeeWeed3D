#!/usr/bin/env python3
"""
SeeWeed3D - what to check before committing hours to a training run.

Every finding here is something that makes a run finish successfully, print a
plausible metric, and mean nothing. None of them raise an exception during
training, which is precisely why they need a pass of their own before it
starts.

    python -m seeweed3d.training.preflight --coco E:\\...\\run5\\coco \\
        --epochs 60 --batch 2 --grad-accum 8 --patience 25

WHAT IT LOOKS FOR
-----------------
1. A CLASS TOO RARE TO LEARN. Under roughly 20 instances a class cannot be
   learned; it contributes an AP near zero that drags the mean down for a
   reason that has nothing to do with the model. The fix is DROP_CLASSES in
   make_dataset.py - which is per-build and reversible - not a smaller learning
   rate.

2. A CLASS THAT CANNOT BE MEASURED. A class present in train but with zero
   instances in val has no validation signal at all: early stopping and
   best-checkpoint selection are both blind to it. When that class is the crop,
   the checkpoint chosen as "best" may be the one that segments onions worst.

3. A CLASS MISSING FROM TRAINING. The mirror image, and worse - the model can
   never predict it, and a class it never predicts reports an empty mask
   downstream, which is indistinguishable from "looked and found nothing".

4. EPOCHS THAT MEAN DIFFERENT THINGS AT DIFFERENT SIZES. `epochs` is a count of
   passes, so 60 epochs over 60 frames is 1,800 optimiser steps at an effective
   batch of 16, and 60 epochs over 600 frames is 18,000 - ten times the compute
   for the same number in the config. Steps, not epochs, is what a schedule and
   a patience are really denominated in.

5. A PATIENCE LONGER THAN THE RUN. Early stopping with patience 25 on a 20
   epoch run cannot fire, so the run has no early stopping regardless of what
   the config says.

6. A VALIDATION SET TOO SMALL TO CHOOSE A CHECKPOINT WITH. Below ~10 frames the
   epoch-to-epoch noise in val mAP exceeds the differences between checkpoints,
   so "best" is chosen by chance and the number it reports is optimistic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Instances below which a class is not learnable in practice. Not a hard rule
#: - it depends on how distinctive the class is - but at this dataset's scale a
#: class in single digits has never once produced a usable AP.
MIN_LEARNABLE_INSTANCES = 20

#: Below this many validation frames, val mAP moves more between epochs from
#: sampling noise than from actual learning, so best-checkpoint selection is
#: effectively random.
MIN_VAL_FRAMES = 10

#: Roboflow's spelling, which is what the COCO tree on disk uses.
TRAIN, VAL, TEST = "train", "valid", "test"


class Finding:
    """One thing worth knowing before the run, with its consequence attached.

    `level` is "error" for something that makes the run meaningless and
    "warn" for something that bounds what it can tell you. Nothing here stops
    a run by itself - the caller decides - because a deliberately small
    smoke-test run trips several of these on purpose."""

    def __init__(self, level, code, message, fix=""):
        self.level, self.code, self.message, self.fix = level, code, message, fix

    def __repr__(self):
        return f"Finding({self.level}, {self.code})"

    def to_dict(self):
        return {"level": self.level, "code": self.code,
                "message": self.message, "fix": self.fix}


def load_export_summary(coco_dir):
    """The summary coco_export wrote, or one rebuilt by reading the tree.

    Reading the tree is the fallback because a COCO directory shared between
    runs may predate the summary, and refusing to check a dataset because its
    sidecar is missing would make the check skippable exactly when it is most
    likely to matter."""
    coco_dir = Path(coco_dir)
    side = coco_dir / "seeweed3d_export.json"
    if side.exists():
        try:
            doc = json.loads(side.read_text(encoding="utf-8"))
            if doc.get("splits") and all("per_class" in v
                                         for v in doc["splits"].values()):
                return doc
        except (OSError, ValueError):
            pass

    splits, classes = {}, []
    for name in (TRAIN, VAL, TEST):
        ann = coco_dir / name / "_annotations.coco.json"
        if not ann.exists():
            continue
        doc = json.loads(ann.read_text(encoding="utf-8"))
        by_id = {c["id"]: c["name"] for c in doc.get("categories", [])}
        classes = classes or [by_id[k] for k in sorted(by_id)]
        counts = {c: 0 for c in classes}
        for a in doc.get("annotations", []):
            n = by_id.get(a.get("category_id"))
            if n:
                counts[n] = counts.get(n, 0) + 1
        splits[name] = {"frames": len(doc.get("images", [])),
                        "instances": len(doc.get("annotations", [])),
                        "per_class": counts}
    if not splits:
        raise SystemExit(
            f"ERROR: no COCO splits found under {coco_dir}. Expected "
            f"train/_annotations.coco.json - build it with "
            f"training/coco_export.py, or point --coco at the right folder.")
    return {"classes": classes, "splits": splits}


def steps_per_epoch(n_frames, batch, grad_accum):
    """Optimiser steps in one epoch at this effective batch."""
    eff = max(1, int(batch) * max(1, int(grad_accum)))
    return max(1, -(-int(n_frames) // eff))       # ceil


def check_classes(summary):
    out = []
    classes = summary.get("classes") or []
    splits = summary.get("splits") or {}
    train = (splits.get(TRAIN) or {}).get("per_class") or {}
    val = (splits.get(VAL) or {}).get("per_class") or {}

    for c in classes:
        n_tr, n_va = train.get(c, 0), val.get(c, 0)
        if n_tr == 0 and n_va == 0:
            continue                      # not in this build at all; fine
        if n_tr == 0:
            out.append(Finding(
                "error", "class_missing_from_train",
                f"{c!r} has {n_va} instance(s) in val and NONE in train. The "
                f"model can never predict it, and a class it never predicts "
                f"reports an empty mask downstream - indistinguishable from "
                f"'looked and found nothing'.",
                "Rebuild with a split that puts this class in train, or drop "
                "it via DROP_CLASSES until you have examples on both sides."))
            continue
        if n_va == 0:
            out.append(Finding(
                "warn", "class_missing_from_val",
                f"{c!r} has {n_tr} training instance(s) but NONE in val, so "
                f"nothing measures it. Early stopping and best-checkpoint "
                f"selection are both blind to this class.",
                "Rebuild the split, or read this class's score from the test "
                "split only and do not trust the chosen checkpoint for it."))
        if n_tr < MIN_LEARNABLE_INSTANCES:
            out.append(Finding(
                "warn", "class_too_rare",
                f"{c!r} has only {n_tr} training instance(s) (floor "
                f"{MIN_LEARNABLE_INSTANCES}). It will report an AP near zero "
                f"and drag the mean down for a reason unrelated to the model.",
                f'Add "{c}" to DROP_CLASSES in make_dataset.py until you have '
                f"more, then rebuild. Dropping is per-build and reversible."))
    return out


def check_splits(summary):
    out = []
    splits = summary.get("splits") or {}
    n_val = (splits.get(VAL) or {}).get("frames", 0)
    if not splits.get(TRAIN, {}).get("frames"):
        out.append(Finding("error", "no_train_frames",
                           "the train split is empty.",
                           "Rebuild the dataset."))
    if not n_val:
        out.append(Finding(
            "error", "no_val_frames",
            "the val split is empty, so there is no early stopping, no best "
            "checkpoint and no metric.",
            "Rebuild with VAL_FRACTION > 0 or a pinned HOLDOUT_VAL_SESSIONS."))
    elif n_val < MIN_VAL_FRAMES:
        out.append(Finding(
            "warn", "val_too_small",
            f"val has {n_val} frame(s). Below ~{MIN_VAL_FRAMES}, epoch-to-"
            f"epoch noise in val mAP exceeds the difference between "
            f"checkpoints, so 'best' is chosen substantially by chance and "
            f"the score it reports is optimistic.",
            "Annotate more, or accept the val score as a sanity check rather "
            "than a measurement."))
    if not splits.get(TEST, {}).get("frames"):
        out.append(Finding(
            "warn", "no_test_split",
            "there is no test split, so every number this run produces comes "
            "from the set used to choose the checkpoint.",
            "Pin a session with HOLDOUT_TEST_SESSIONS in make_dataset.py - see "
            "docs/dataset_growth.md."))
    return out


def check_schedule(summary, epochs, batch, grad_accum, patience,
                   early_stopping=True):
    out = []
    n_train = (summary.get("splits", {}).get(TRAIN) or {}).get("frames", 0)
    if not n_train:
        return out
    spe = steps_per_epoch(n_train, batch, grad_accum)
    total = spe * max(1, int(epochs))
    out.append(Finding(
        "info", "schedule",
        f"{n_train} train frames | effective batch {batch * grad_accum} | "
        f"{spe} step(s)/epoch | {epochs} epochs = {total} optimiser steps.",
        ""))
    if spe < 2:
        out.append(Finding(
            "warn", "one_step_per_epoch",
            f"the effective batch ({batch * grad_accum}) is at or above the "
            f"whole training set ({n_train} frames), so an epoch is a single "
            f"optimiser step. Every schedule denominated in epochs - warmup, "
            f"cosine decay, patience - is really counting single steps.",
            "Lower GRAD_ACCUM, or raise EPOCHS to keep the step count "
            "sensible."))
    if early_stopping and patience and int(patience) >= int(epochs):
        out.append(Finding(
            "warn", "patience_exceeds_run",
            f"early-stopping patience ({patience}) is not shorter than the run "
            f"({epochs} epochs), so it can never fire. The run has no early "
            f"stopping regardless of the config.",
            f"Set PATIENCE below EPOCHS (a third is a common choice), or "
            f"accept that the run will always go the full length."))
    return out


def check_provenance(summary):
    """What the labels ARE decides what every metric below MEANS.

    Restated here, at train time, because the build that recorded it may have
    been months and several rounds ago - and this is the finding that changes
    how the resulting number should be read, not whether the run will finish."""
    prov = summary.get("label_provenance")
    if prov in (None, "hand_corrected"):
        return []
    if prov == "prelabel_unreviewed":
        return [Finding(
            "warn", "labels_unreviewed",
            "these labels are UNREVIEWED prelabels, so this run is "
            "DISTILLATION: the model learns the prelabeler's misses as if they "
            "were correct and cannot exceed it. Every AP below measures "
            "agreement WITH THE PRELABELER, not with reality - a high score "
            "says 'faithfully reproduces the teacher', which is not the same "
            "as 'finds the crop'.",
            "Hand-correct a small held-out test split (20-30 frames is "
            "enough) and read the score from that alone. Without one, a good "
            "run and a bad run are indistinguishable.")]
    return [Finding(
        "warn", "labels_partly_unreviewed",
        "labels are a MIX of hand-corrected and unreviewed prelabels. Any "
        "split containing unreviewed frames scores agreement with the "
        "prelabeler on those frames.",
        "Confirm the test split is hand-corrected throughout, and read the "
        "score from it alone.")]


def preflight(coco_dir, *, epochs=60, batch=2, grad_accum=8, patience=25,
              early_stopping=True):
    """Every finding for this dataset and this schedule."""
    summary = load_export_summary(coco_dir)
    findings = (check_splits(summary) + check_classes(summary)
                + check_provenance(summary)
                + check_schedule(summary, epochs, batch, grad_accum, patience,
                                 early_stopping))
    return summary, findings


def format_report(summary, findings):
    lines = ["", "  Pre-flight", "  " + "-" * 62]
    classes = summary.get("classes") or []
    splits = summary.get("splits") or {}
    order = [s for s in (TRAIN, VAL, TEST) if s in splits]
    if classes and order:
        head = "  {:<26}".format("class") + "".join(f"{s:>9}" for s in order)
        lines += [head, "  " + "-" * (26 + 9 * len(order))]
        for c in classes:
            row = "  {:<26}".format(c[:26])
            row += "".join(f"{(splits[s].get('per_class') or {}).get(c, 0):>9}"
                           for s in order)
            lines.append(row)
        lines.append("  {:<26}".format("FRAMES")
                     + "".join(f"{splits[s].get('frames', 0):>9}"
                               for s in order))
    for f in findings:
        tag = {"error": "[ERROR]", "warn": "[!]", "info": "  "}.get(f.level, "")
        lines.append(f"  {tag} {f.message}")
        if f.fix:
            lines.append(f"        -> {f.fix}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", required=True, help="the COCO tree to check")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any finding is an error")
    a = p.parse_args(argv)

    summary, findings = preflight(a.coco, epochs=a.epochs, batch=a.batch,
                                  grad_accum=a.grad_accum,
                                  patience=a.patience,
                                  early_stopping=not a.no_early_stopping)
    print(format_report(summary, findings))
    errors = [f for f in findings if f.level == "error"]
    return 1 if (a.strict and errors) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
