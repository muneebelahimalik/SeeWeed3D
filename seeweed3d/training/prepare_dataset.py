#!/usr/bin/env python3
"""
SeeWeed3D - Stage 0: verified CVAT/Datumaro export -> trainable dataset.

Produces, from one verified export:
  1. Ultralytics segmentation labels (labels/<split>/*.txt) + data.yaml
  2. A per-instance LEP manifest (lep_manifest.json)
  3. A dataset integrity report (dataset_report.json)
  4. Per-session and per-class statistics
  5. annotations_needing_correction.json

Images are NOT copied. Manifests reference the existing files, so a large
dataset is not duplicated and Windows paths keep working (paths are stored
posix-style and resolved against --images-root).

    python -m seeweed3d.training.prepare_dataset \\
        --datumaro-root  D:/exports/verified_mixed \\
        --images-root    D:/Dataset_Vidalia/sessions \\
        --out            D:/Dataset_Vidalia/training/mixed_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402
from training import datumaro_multitask as dmm  # noqa: E402
from training import splits as sp  # noqa: E402
from training.config import AnnotationContract  # noqa: E402


def find_annotation_files(roots):
    """Every Datumaro JSON under one or several export roots.

    MERGING SEPARATE CVAT TASKS IS THE NORMAL CASE. Annotating one session per
    CVAT task is good practice - tasks stay small, and a session is the unit
    that splits must respect anyway - so this accepts either a parent folder
    holding many unzipped exports, or several explicit roots.

    Merging across tasks is safe because each file is resolved through its OWN
    `categories` block: `label_id` 2 can legitimately mean different classes in
    two tasks, and only the label NAME is carried forward. A merge keyed on
    label_id would silently relabel half the dataset."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    files = []
    for root in roots:
        root = Path(root)
        found = sorted(root.rglob("annotations/*.json"))
        if not found:
            found = sorted(root.glob("*.json"))
        files.extend(found)
    if not files:
        shown = ", ".join(str(Path(r)) for r in roots)
        raise SystemExit(
            f"ERROR: no Datumaro JSON found under: {shown}\n"
            f"Expected <root>/annotations/*.json. In CVAT use "
            f"Export -> 'Datumaro 1.0', unzip each export, then point "
            f"--datumaro-root at the unzipped folder(s) - or at one parent "
            f"folder containing all of them.")
    return sorted(set(files))


def build(datumaro_root, images_root, out_root, *, contract=None,
          val_fraction=0.2, test_fraction=0.2, seed=1234,
          holdout_val=(), holdout_test=(), strict=True):
    """Everything after out_root is keyword-only on purpose: a positional
    fraction silently landing in the `contract` slot produced a confusing
    AttributeError deep inside validation rather than an error at the call."""
    contract = contract or AnnotationContract()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    ann_files = find_annotation_files(datumaro_root)
    frames, report = [], dmm.MultitaskDatasetReport()
    origin = {}
    for f in ann_files:
        got, report = dmm.load_datumaro(f, contract, report=report)
        for rec in got:
            origin.setdefault(rec.item_id, []).append(str(f))
        frames.extend(got)

    # The same frame annotated in two CVAT tasks would be counted twice and
    # could land in two splits, which is exactly the leakage this pipeline
    # exists to prevent. Detected here rather than discovered as an
    # inexplicably good validation score.
    dupes = {k: v for k, v in origin.items() if len(v) > 1}
    if dupes:
        for item_id, sources in sorted(dupes.items())[:20]:
            report.add_error(item_id, "duplicate_frame_across_exports",
                             f"annotated in {len(sources)} exports: "
                             f"{', '.join(sources)}. Keep exactly one, or the "
                             f"frame is trained on twice and may span splits.")

    report = dmm.validate_frames(frames, contract, report)
    print(f"  merged {len(ann_files)} annotation file(s) -> {len(frames)} frames")

    # -- splits, by whole session ------------------------------------------
    per_session = Counter(f.session_id for f in frames)
    infos = [sp.SessionInfo(session_id=s, n_frames=n,
                            class_counts=report.per_session.get(s, {}))
             for s, n in sorted(per_session.items()) if s]
    split_map = sp.assign_splits(infos, val_fraction, test_fraction, seed,
                                 holdout_val=holdout_val,
                                 holdout_test=holdout_test)
    frames_by_session = {}
    for f in frames:
        frames_by_session.setdefault(f.session_id, []).append(f.image_path)
    sp.check_no_leakage(split_map, {f.item_id: f.session_id for f in frames
                                    if f.session_id})
    summary = sp.write_splits(out / "splits", split_map, frames_by_session, infos)

    where = {s: k for k, v in split_map.items() for s in v}

    # -- Ultralytics labels -------------------------------------------------
    n_labels = 0
    for f in frames:
        split = where.get(f.session_id)
        if split is None:
            continue
        d = out / "labels" / split
        d.mkdir(parents=True, exist_ok=True)
        body = dmm.to_yolo_segmentation(f)
        (d / f"{Path(f.image_path).stem}.txt").write_text(body + "\n",
                                                          encoding="utf-8")
        n_labels += 1

    data_yaml = (
        "# Ultralytics dataset config, generated by prepare_dataset.py.\n"
        "# Class order comes from seeweed3d/common/ontology.py and MUST NOT be\n"
        "# reordered - the indices are baked into every label file.\n"
        f"path: {Path(images_root).as_posix()}\n"
        "train: ../training/splits/train_images.txt\n"
        "val: ../training/splits/val_images.txt\n"
        "test: ../training/splits/test_images.txt\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # -- Segmentation manifest (permissive backends) ------------------------
    # The BSD-3 Mask R-CNN and Apache-2.0 RF-DETR paths train from this, so the
    # permissive default needs no YOLO-format label tree. Derived from the same
    # FrameRecords as the LEP manifest, so the two stages cannot disagree about
    # what was annotated.
    seg_frames = []
    for f in frames:
        split = where.get(f.session_id)
        if split is None or not f.instances:
            continue
        seg_frames.append({
            "session_id": f.session_id, "item_id": f.item_id,
            "image_path": Path(f.image_path).as_posix(),
            "width": f.width, "height": f.height, "split": split,
            "instances": [{"class_name": i.class_name,
                           "class_index": CLASSES.index(i.class_name),
                           "polygons": [[round(float(v), 2) for v in p]
                                        for p in i.polygons]}
                          for i in f.instances],
            "ignore_regions": [[round(float(v), 2) for v in p]
                               for p in f.ignore_regions]})
    (out / "seg_manifest.json").write_text(
        json.dumps({"images_root": Path(images_root).as_posix(),
                    "classes": list(CLASSES),
                    "n_frames": len(seg_frames), "frames": seg_frames}, indent=2),
        encoding="utf-8")

    # -- LEP manifest, split-aware -----------------------------------------
    rows = dmm.to_lep_manifest(frames)
    for r in rows:
        r["split"] = where.get(r["session_id"], "unassigned")
    (out / "lep_manifest.json").write_text(
        json.dumps({"images_root": Path(images_root).as_posix(),
                    "n_rows": len(rows), "rows": rows}, indent=2),
        encoding="utf-8")

    # -- reports ------------------------------------------------------------
    rep = report.to_dict()
    rep["splits"] = summary
    rep["n_yolo_label_files"] = n_labels
    rep["n_lep_rows"] = len(rows)
    rep["lep_rows_per_split"] = dict(Counter(r["split"] for r in rows))
    (out / "dataset_report.json").write_text(json.dumps(rep, indent=2),
                                             encoding="utf-8")
    (out / "annotations_needing_correction.json").write_text(
        json.dumps(report.needs_correction, indent=2), encoding="utf-8")

    xcheck = dmm.cross_check_with_datumaro(ann_files[0], frames)
    if xcheck:
        (out / "datumaro_cross_check.json").write_text(json.dumps(xcheck, indent=2),
                                                       encoding="utf-8")

    print(f"\n{report.summary()}")
    print(f"  classes : {report.per_class}")
    print(f"  splits  : " + " ".join(
        f"{k}={len(v)}" for k, v in split_map.items()))
    print(f"  LEP rows: {len(rows)}  ({rep['lep_rows_per_split']})")
    print(f"  -> {out}")

    if report.errors:
        print(f"\n  {len(report.errors)} ERROR(S). See "
              f"annotations_needing_correction.json. Fix them in CVAT, re-export, "
              f"and re-run - training on a broken contract silently corrupts the "
              f"target.")
        if strict:
            raise SystemExit(1)
    return report, split_map, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datumaro-root", required=True, nargs="+",
                   help="one or more unzipped CVAT 'Datumaro 1.0' exports, or "
                        "a single parent folder containing several of them. "
                        "Annotating one session per CVAT task and merging here "
                        "is the normal workflow.")
    p.add_argument("--images-root", required=True,
                   help="dataset sessions root holding the RGB frames")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-val", nargs="*", default=[])
    p.add_argument("--holdout-test", nargs="*", default=[])
    p.add_argument("--allow-errors", action="store_true",
                   help="write outputs even when the contract is violated "
                        "(for triage only - do NOT train on the result)")
    a = p.parse_args(argv)
    build(a.datumaro_root, a.images_root, a.out,
          val_fraction=a.val_fraction, test_fraction=a.test_fraction,
          seed=a.seed, holdout_val=a.holdout_val, holdout_test=a.holdout_test,
          strict=not a.allow_errors)


if __name__ == "__main__":
    main()
