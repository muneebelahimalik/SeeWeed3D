#!/usr/bin/env python3
"""
SeeWeed3D - is the DEPLOYED system ready? Check before trusting an inference run.

    python -m seeweed3d.perception.preflight

training/preflight.py checks a dataset before a training run. This checks the
other end: the model, its class list, the depth it needs, the LEP stage and the
safety policy - together, as one system, because that is the only level at which
most of these failures exist.

WHY A SEPARATE CHECK, AND NOT JUST READING THE OUTPUT
-----------------------------------------------------
Every failure here produces output that looks fine. A weed-only model run
through the full pipeline abstains on every single target and reports zero
candidates - which is indistinguishable, in the JSON, from a clean frame with no
weeds in it. A checkpoint whose class list is missing maps labels through the
full ontology and mislabels every plant with complete confidence. A session
without calibration silently drops to 2D and every target comes back with no
xyz_mm. None of those raise; all of them are wrong.

THE ONE THAT WILL SURPRISE YOU
-------------------------------
`SafetyConfig.allow_missing_crop_mask` defaults to False, so a model with no
onion_plant class rejects EVERY candidate - deliberately, because "this model
cannot see onions" must never be read as "there are no onions here". A weed-only
model is therefore not a deployable system, however good its masks are. It is a
Stage A model, and this says so rather than letting a run of all-abstentions be
read as a quiet field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402
from training.preflight import Finding  # noqa: E402

ERROR, WARN, OK = "error", "warn", "ok"


def checkpoint_classes(checkpoint):
    """The class list a checkpoint records, or None if it records none.

    rfdetr writes it beside the weights, and the segmenter refuses to map labels
    without it - correctly, because the fallback is the full ontology and a
    3-class model read through a 6-class list mislabels every plant while
    looking completely normal."""
    d = Path(checkpoint).parent
    cfg = d / "rfdetr_train_config.json"
    if cfg.is_file():
        try:
            blob = json.loads(cfg.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            blob = {}
        for key in ("classes", "class_names", "category_names"):
            if blob.get(key):
                return list(blob[key])
    for name in ("annotations/instances_train.json", "train/_annotations.coco.json"):
        p = d / "coco" / name
        if p.is_file():
            try:
                cats = json.loads(p.read_text(encoding="utf-8")).get("categories")
            except (ValueError, OSError):
                cats = None
            if cats:
                return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    return None


def session_depth_ready(session_dir):
    """(has_depth, has_calibration) for one session folder.

    Both are needed for 3D. Without them the pipeline still runs and every
    target comes back with xyz_mm None, which reads as 'nothing found' rather
    than 'nothing measured'."""
    d = Path(session_dir)
    depth = d / "depth"
    has_depth = depth.is_dir() and any(depth.iterdir())
    return has_depth, (d / "meta" / "calibration.json").is_file()


def system_findings(*, checkpoint=None, model_classes=None, lep_checkpoint="",
                    allow_missing_crop_mask=False, has_depth=None,
                    has_calibration=None, holdout_sessions=(),
                    label_provenance="", dedup_iou=None):
    """Every readiness problem, as Findings. Pure - takes facts, not paths.

    Split from the gathering so the rules can be tested without a filesystem,
    and so the same rules apply whether the facts came from disk, from a
    manifest, or from a caller that already knows them."""
    out = []

    if checkpoint is None:
        out.append(Finding(ERROR, "no_checkpoint",
                           "No Stage A checkpoint.",
                           "Train a round: python -m "
                           "seeweed3d.training.datasets.weeds_train"))
        return out

    # -- the class list -------------------------------------------------- #
    if not model_classes:
        out.append(Finding(
            ERROR, "no_class_list",
            f"{checkpoint} records no class list.",
            "The segmenter falls back to the FULL ontology, so a model with "
            "fewer classes has every prediction mapped to the wrong name - "
            "confidently, and with nothing in the output to show it. Re-export "
            "the dataset and retrain so rfdetr_train_config.json travels with "
            "the weights."))
    else:
        unknown = [c for c in model_classes if c not in CLASSES]
        if unknown:
            out.append(Finding(
                ERROR, "class_not_in_ontology",
                f"The model predicts {unknown}, which the ontology does not "
                f"define.",
                "common/ontology.py is what every downstream stage indexes by "
                "name. Add the class there or retrain without it."))
        missing = [c for c in CLASSES if c not in model_classes]
        if missing:
            out.append(Finding(
                WARN, "ontology_classes_absent",
                f"The model cannot predict {missing}.",
                "Instances of those classes are scored as background. Fine "
                "while they have too few examples to train on; it stops being "
                "fine the round they have enough."))

    # -- the one that makes a weed-only model undeployable ---------------- #
    has_crop = bool(model_classes) and CROP_CLASS in model_classes
    if not has_crop and not allow_missing_crop_mask:
        out.append(Finding(
            ERROR, "no_crop_class",
            f"The model has no {CROP_CLASS!r} class and "
            f"allow_missing_crop_mask is False, so EVERY candidate will be "
            f"rejected.",
            "That is the safe default, not a bug: a model that cannot see "
            "onions must never have its silence read as 'no onions here'. "
            "Either train a build that includes the crop, or - only if you are "
            "standing in the field and can say there is no crop in front of "
            "the camera - set SafetyConfig.allow_missing_crop_mask = True. It "
            "is recorded in every decision either way."))
    elif not has_crop:
        out.append(Finding(
            WARN, "crop_blind_allowed",
            f"No {CROP_CLASS!r} class, and allow_missing_crop_mask is True.",
            "Every candidate is approved without any crop check. This is a "
            "claim about the field, and it is only true while the camera is "
            "pointed somewhere with no crop in it."))

    # -- Stage B ---------------------------------------------------------- #
    if not str(lep_checkpoint or "").strip():
        out.append(Finding(
            WARN, "no_lep_model",
            "No LEP checkpoint - the hand-engineered estimator will be used.",
            "That fallback exists so the system runs before Stage B is "
            "trained, and it is the baseline the learned model must beat. It "
            "is not the deployed stage: train one with "
            "seeweed3d/training/train_lep.py."))

    # -- 3D --------------------------------------------------------------- #
    if has_depth is False:
        out.append(Finding(
            ERROR, "no_depth",
            "No depth frames for the session being inferred.",
            "The pipeline falls back to 2D and every target returns xyz_mm "
            "None, which reads as 'nothing found' rather than 'nothing "
            "measured'. A delta robot cannot be aimed from that."))
    if has_calibration is False:
        out.append(Finding(
            ERROR, "no_calibration",
            "No meta/calibration.json for the session being inferred.",
            "Intrinsics are what turn a pixel into a ray. Without them depth "
            "is present but unusable and every target is 2D."))

    # -- what the numbers will and will not mean -------------------------- #
    if not holdout_sessions:
        out.append(Finding(
            WARN, "no_holdout",
            "No holdout test session.",
            "val is contiguous blocks inside the training sessions, so it "
            "shares their light, soil, growth stage and often the individual "
            "plants. Every round's score is an upper bound and no round can be "
            "compared with the last. Cut one drive out and correct 20-30 "
            "frames from it."))
    if label_provenance and label_provenance != "hand_corrected":
        out.append(Finding(
            WARN, "machine_labels",
            f"The dataset's label provenance is {label_provenance!r}.",
            "Scores measure agreement with a prelabeler or with the model's "
            "own earlier output, not correctness. Say so wherever the number "
            "is quoted."))

    if dedup_iou == 0:
        out.append(Finding(
            WARN, "dedup_off",
            "Duplicate suppression is disabled (dedup_iou = 0).",
            "A set-prediction model returns one plant twice under two labels - "
            "14% of detections on a real weed session. Each copy becomes its "
            "own target with its own LEP and 3D point, so the laser is aimed "
            "twice at one plant, and a copy labelled onion puts those pixels "
            "in the safety mask while its twin is on the target list."))

    return out


def format_report(findings, *, checkpoint=None, model_classes=None):
    """The readout. Says what is ready as plainly as what is not."""
    L = ["", "  SeeWeed3D - deployed system preflight", "  " + "-" * 44]
    if checkpoint:
        L.append(f"  model    {checkpoint}")
    if model_classes:
        L.append(f"  classes  {len(model_classes)}: {', '.join(model_classes)}")

    errors = [f for f in findings if f.level == ERROR]
    warns = [f for f in findings if f.level == WARN]
    if not findings:
        L += ["", "  READY. Nothing to flag.", ""]
        return "\n".join(L)

    for label, group in (("BLOCKING", errors), ("WORTH KNOWING", warns)):
        if not group:
            continue
        L += ["", f"  {label}"]
        for f in group:
            L.append(f"    [{f.code}] {f.message}")
            for line in (f.fix or "").split(". "):
                if line.strip():
                    L.append(f"        {line.strip().rstrip('.')}.")
    L += ["",
          f"  {len(errors)} blocking, {len(warns)} worth knowing.",
          ""]
    if errors:
        L += ["  A blocking finding does not raise an exception anywhere. The "
              "run will",
              "  complete and its output will look ordinary - that is exactly "
              "why this",
              "  check exists.", ""]
    return "\n".join(L)


def inspect(checkpoint, *, session_dir=None, lep_checkpoint="",
            allow_missing_crop_mask=False, holdout_sessions=(),
            label_provenance="", dedup_iou=None):
    """Gather the facts from disk, then apply `system_findings`."""
    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is None or not ckpt.is_file():
        return system_findings(checkpoint=None), None
    classes = checkpoint_classes(ckpt)
    has_depth = has_calib = None
    if session_dir:
        has_depth, has_calib = session_depth_ready(session_dir)
    return system_findings(
        checkpoint=str(ckpt), model_classes=classes,
        lep_checkpoint=lep_checkpoint,
        allow_missing_crop_mask=allow_missing_crop_mask,
        has_depth=has_depth, has_calibration=has_calib,
        holdout_sessions=holdout_sessions,
        label_provenance=label_provenance, dedup_iou=dedup_iou), classes


def main(argv=None):
    import ntpath
    from training.config import PipelineConfig
    from training.datasets.weeds import HOLDOUT_TEST, WEED_POOL_ROOT
    from training.datasets.weeds_train import ROUND, RUNS_ROOT

    cfg = PipelineConfig()
    ckpt = ntpath.join(RUNS_ROOT, f"weeds_r{ROUND}",
                       "checkpoint_best_total.pth")
    session = None
    if len(sys.argv) > 1:
        session = sys.argv[1]
    elif Path(WEED_POOL_ROOT).is_dir():
        subs = [d for d in sorted(Path(WEED_POOL_ROOT).iterdir()) if d.is_dir()]
        session = str(subs[0]) if subs else None

    findings, classes = inspect(
        ckpt, session_dir=session,
        allow_missing_crop_mask=cfg.safety.allow_missing_crop_mask,
        holdout_sessions=HOLDOUT_TEST, dedup_iou=cfg.dedup_iou)
    print(format_report(findings, checkpoint=ckpt, model_classes=classes))
    if session:
        print(f"  session checked for depth/calibration: {session}\n")
    return 1 if any(f.level == ERROR for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
