#!/usr/bin/env python3
"""
SeeWeed3D - Stage A training: YOLO26n-seg instance segmentation.

A thin, version-pinned entry point around the official Ultralytics package.
Ultralytics is NOT vendored: see perception/segmenter.py for the interface
boundary and the AGPL-3.0 licensing note, which matters for a commercial
weeder.

    python -m seeweed3d.training.train_seg --data D:/.../mixed_v1/data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402


def train(data_yaml, model="yolo26n-seg.pt", epochs=100, imgsz=1024, batch=8,
          device=0, project=None, name="seg", seed=1234, workers=4):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ERROR: ultralytics is not installed.\n"
            "  python -m pip install -r requirements-training.txt\n"
            "It is an OPTIONAL dependency; the data pipeline and unit tests do "
            "not need it.")
    data = Path(data_yaml)
    if not data.exists():
        raise SystemExit(
            f"ERROR: {data} not found.\n"
            f"Build it first from your verified CVAT export:\n"
            f"  python -m seeweed3d.training.prepare_dataset "
            f"--datumaro-root <export> --images-root <sessions> --out <dir>")
    # Mosaic/MixUp are safe for full-frame segmentation (unlike ROI LEP
    # training, where they destroy per-plant ownership), but copy_paste is
    # disabled: pasting an onion into another frame would fabricate crop-safety
    # geometry that never existed.
    return YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz,
                             batch=batch, device=device, seed=seed,
                             workers=workers, project=project, name=name,
                             copy_paste=0.0, verbose=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml from prepare_dataset")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=0)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default="seg")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    print(f"Training {a.model} on {len(CLASSES)} classes: {CLASSES}")
    train(a.data, a.model, a.epochs, a.imgsz, a.batch, a.device, a.project,
          a.name, a.seed, a.workers)


if __name__ == "__main__":
    main()
