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
          holdout_val=(), holdout_test=(), strict=True, gap_frames=2,
          drop_classes=(), keep_empty_frames=False):
    """Everything after out_root is keyword-only on purpose: a positional
    fraction silently landing in the `contract` slot produced a confusing
    AttributeError deep inside validation rather than an error at the call.

    drop_classes: exclude these ontology classes from THIS dataset build.

        Use this instead of editing common/ontology.py. The ontology is the
        stable, project-wide source of truth: its order fixes the COCO category
        ids, the CVAT label schema, and every file already exported. Deleting a
        class from it would renumber everything and make a future dataset that
        DOES contain that class impossible to merge with this one. Dropping per
        build is reversible and local - re-run without the flag once you have
        examples.

        A `class_mapping.json` records ontology name -> training index, so a
        model trained on the reduced set can still be interpreted against the
        full ontology.

    keep_empty_frames: by default a frame with ZERO annotations is EXCLUDED.
        In an export from a task you annotated by hand, an empty frame is
        almost always one you did not get to - and it is indistinguishable from
        genuinely bare ground. Training on it teaches the model that a frame
        full of weeds is background, which is far more damaging than the frame
        being missing. Pass True only if your empty frames are deliberately
        empty ground."""
    contract = contract or AnnotationContract()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    drop = {c for c in (drop_classes or ())}
    unknown_drop = sorted(drop - set(CLASSES))
    if unknown_drop:
        raise SystemExit(
            f"ERROR: --drop-classes names classes not in the ontology: "
            f"{unknown_drop}\nKnown: {CLASSES}")
    active_classes = [c for c in CLASSES if c not in drop]
    if not active_classes:
        raise SystemExit("ERROR: every class was dropped; nothing to train on.")

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

    # -- drop classes for THIS build (ontology untouched) --------------------
    if drop:
        removed = 0
        for f in frames:
            before = len(f.instances)
            f.instances = [i for i in f.instances if i.class_name not in drop]
            removed += before - len(f.instances)
        print(f"  dropped {removed} instance(s) of {sorted(drop)} from this "
              f"build. common/ontology.py is UNCHANGED - re-run without "
              f"--drop-classes once you have examples.")
        # Recount, so the report describes the dataset that was actually built
        # rather than the export it was read from. A per_class still listing a
        # dropped class would send you looking for it in the trained model.
        per_class, per_session = Counter(), {}
        for f in frames:
            for i in f.instances:
                per_class[i.class_name] += 1
                per_session.setdefault(f.session_id, Counter())[i.class_name] += 1
        report.per_class = dict(per_class)
        report.per_session = {k: dict(v) for k, v in per_session.items()}
        report.n_instances = int(sum(per_class.values()))

    # -- exclude un-annotated frames ----------------------------------------
    # An empty frame in a hand-annotated export is almost always one you did
    # not reach, and it is indistinguishable from genuinely bare ground.
    # Training on it teaches the model that a frame full of weeds is
    # background, which is far worse than the frame simply being absent.
    empty = [f for f in frames if not f.instances]
    if empty and not keep_empty_frames:
        frames = [f for f in frames if f.instances]
        print(f"  EXCLUDED {len(empty)} frame(s) with no annotations "
              f"({', '.join(f.item_id for f in empty[:5])}"
              f"{' ...' if len(empty) > 5 else ''}).")
        print(f"      If any of those are genuinely bare ground you WANT to "
              f"train on, pass --keep-empty-frames.")
    elif empty:
        print(f"  [!] KEEPING {len(empty)} frame(s) with no annotations as "
              f"negative examples, because --keep-empty-frames was given.")

    if not frames:
        raise SystemExit(
            "ERROR: no annotated frames left after filtering. Check that the "
            "CVAT task actually contains saved annotations.")

    # -- splits -------------------------------------------------------------
    per_session = Counter(f.session_id for f in frames)
    infos = [sp.SessionInfo(session_id=s, n_frames=n,
                            class_counts=report.per_session.get(s, {}))
             for s, n in sorted(per_session.items()) if s]

    split_mode = "session"
    frame_split = None
    if len(infos) < 2:
        # A single recording cannot be split by session. Rather than silently
        # producing empty val/test (which trains blind), fall back to
        # contiguous frame blocks and say plainly what that costs.
        split_mode = "frame_block"
        ordered = sorted(frames, key=lambda f: f.item_id)
        frame_split = sp.assign_frame_blocks(
            [f.item_id for f in ordered], val_fraction, test_fraction,
            gap_frames=gap_frames)
        split_map = {"train": [i.session_id for i in infos], "val": [], "test": []}
        where = {}
        for split in ("train", "val", "test"):
            for item in frame_split[split]:
                where[item] = split
        print(f"\n  [!] ONLY ONE SESSION ({infos[0].session_id if infos else '?'}).")
        print(f"      A session-level split is impossible, so the frames were "
              f"split into CONTIGUOUS BLOCKS with a {gap_frames}-frame gap:")
        print(f"        train={len(frame_split['train'])} "
              f"val={len(frame_split['val'])} test={len(frame_split['test'])} "
              f"(dropped as buffer: {len(frame_split['_dropped_gap'])})")
        print(f"      These val/test frames share the session's lighting, soil, "
              f"growth stage and often the same individual plants.")
        print(f"      Treat the scores as a SANITY CHECK that training works, "
              f"NOT as evidence of generalisation.")
        print(f"      Annotate a SECOND session to get a real held-out test.\n")
    else:
        split_map = sp.assign_splits(infos, val_fraction, test_fraction, seed,
                                     holdout_val=holdout_val,
                                     holdout_test=holdout_test)
        sp.check_no_leakage(split_map, {f.item_id: f.session_id for f in frames
                                        if f.session_id})
        session_where = {s: k for k, v in split_map.items() for s in v}
        where = {f.item_id: session_where.get(f.session_id) for f in frames}

    frames_by_session = {}
    for f in frames:
        frames_by_session.setdefault(f.session_id, []).append(f.image_path)
    summary = sp.write_splits(out / "splits", split_map, frames_by_session, infos)
    summary["split_mode"] = split_mode
    if frame_split:
        summary["frame_blocks"] = {k: v for k, v in frame_split.items()}
        summary["warning"] = (
            "frame_block split: val/test come from the SAME recording as train. "
            "Scores are a sanity check, not evidence of generalisation.")
        (out / "splits" / "splits_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    # -- Ultralytics labels -------------------------------------------------
    n_labels = 0
    for f in frames:
        split = where.get(f.item_id)
        if split is None:
            continue
        d = out / "labels" / split
        d.mkdir(parents=True, exist_ok=True)
        body = dmm.to_yolo_segmentation(f, active_classes)
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
        f"nc: {len(active_classes)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n"
                             for i, n in enumerate(active_classes)))
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # -- Segmentation manifest (permissive backends) ------------------------
    # The BSD-3 Mask R-CNN and Apache-2.0 RF-DETR paths train from this, so the
    # permissive default needs no YOLO-format label tree. Derived from the same
    # FrameRecords as the LEP manifest, so the two stages cannot disagree about
    # what was annotated.
    seg_frames = []
    for f in frames:
        split = where.get(f.item_id)
        if split is None or not f.instances:
            continue
        seg_frames.append({
            "session_id": f.session_id, "item_id": f.item_id,
            "image_path": Path(f.image_path).as_posix(),
            "width": f.width, "height": f.height, "split": split,
            "instances": [{"class_name": i.class_name,
                           "class_index": active_classes.index(i.class_name),
                           "polygons": [[round(float(v), 2) for v in p]
                                        for p in i.polygons]}
                          for i in f.instances],
            "ignore_regions": [[round(float(v), 2) for v in p]
                               for p in f.ignore_regions]})
    (out / "seg_manifest.json").write_text(
        json.dumps({"images_root": Path(images_root).as_posix(),
                    "classes": list(active_classes),
                    "n_frames": len(seg_frames), "frames": seg_frames}, indent=2),
        encoding="utf-8")

    # -- LEP manifest, split-aware -----------------------------------------
    rows = dmm.to_lep_manifest(frames)
    for r in rows:
        r["split"] = where.get(r["item_id"], "unassigned")
    (out / "lep_manifest.json").write_text(
        json.dumps({"images_root": Path(images_root).as_posix(),
                    "n_rows": len(rows), "rows": rows}, indent=2),
        encoding="utf-8")

    (out / "class_mapping.json").write_text(json.dumps({
        "ontology": list(CLASSES),
        "active_classes": list(active_classes),
        "dropped": sorted(drop),
        "train_index_to_ontology_name": {i: n for i, n in
                                         enumerate(active_classes)},
        "note": ("Training indices are into active_classes, which is contiguous. "
                 "common/ontology.py is unchanged, so a future dataset "
                 "containing the dropped classes merges with this one.")},
        indent=2), encoding="utf-8")

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
    p.add_argument("--drop-classes", nargs="*", default=[],
                   help="exclude these ontology classes from THIS build "
                        "(e.g. --drop-classes wild_radish weed_cluster). "
                        "common/ontology.py is NOT modified.")
    p.add_argument("--keep-empty-frames", action="store_true",
                   help="keep frames with no annotations as negative examples. "
                        "Off by default: an empty frame is usually one you did "
                        "not annotate, and training on it teaches the model "
                        "that plants are background.")
    p.add_argument("--gap-frames", type=int, default=2,
                   help="single-session only: frames discarded at each block "
                        "boundary to reduce temporal leakage")
    p.add_argument("--holdout-val", nargs="*", default=[])
    p.add_argument("--holdout-test", nargs="*", default=[])
    p.add_argument("--allow-errors", action="store_true",
                   help="write outputs even when the contract is violated "
                        "(for triage only - do NOT train on the result)")
    a = p.parse_args(argv)
    build(a.datumaro_root, a.images_root, a.out,
          val_fraction=a.val_fraction, test_fraction=a.test_fraction,
          seed=a.seed, holdout_val=a.holdout_val, holdout_test=a.holdout_test,
          strict=not a.allow_errors, gap_frames=a.gap_frames,
          drop_classes=a.drop_classes,
          keep_empty_frames=a.keep_empty_frames)


if __name__ == "__main__":
    main()
