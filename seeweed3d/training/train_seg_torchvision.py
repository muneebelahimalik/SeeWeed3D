#!/usr/bin/env python3
"""
SeeWeed3D - Stage A training on the PERMISSIVE (BSD-3) backend.

torchvision Mask R-CNN. No AGPL dependency, no Enterprise Licence, nothing
new to install beyond what Stage B already needs.

This is the default Stage A path. `train_seg.py` (Ultralytics/YOLO26) remains
available for research use, but it is AGPL-3.0 - see
docs/supervised_perception_baseline.md before shipping anything built with it.

    python -m seeweed3d.training.train_seg_torchvision \
        --dataset     D:/Dataset_Vidalia/training/mixed_v1 \
        --images-root D:/Dataset_Vidalia/sessions \
        --out         D:/Dataset_Vidalia/runs/seg_v1
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402


def train(dataset_dir, images_root, out_dir, epochs=20, batch=2, lr=5e-3,
          device="cpu", workers=0, seed=1234, pretrained=True, min_area_px=16):
    try:
        import torch
    except ImportError:
        raise SystemExit("ERROR: torch is required.\n"
                         "  python -m pip install -r requirements-training.txt")
    from torch.utils.data import DataLoader
    from perception.segmenter import MaskRCNNSegmenter
    from training.seg_dataset import SegManifestDataset, collate

    manifest = Path(dataset_dir) / "seg_manifest.json"
    if not manifest.exists():
        raise SystemExit(
            f"ERROR: {manifest} not found.\n"
            f"Build it from your verified CVAT export first:\n"
            f"  python -m seeweed3d.training.prepare_dataset "
            f"--datumaro-root <export> --images-root <sessions> --out "
            f"{dataset_dir}")
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if not doc.get("frames"):
        raise SystemExit(
            f"ERROR: {manifest} contains no annotated frames. Check "
            f"annotations_needing_correction.json next to it.")

    classes = list(doc.get("classes") or CLASSES)
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    tr = SegManifestDataset(doc, images_root, "train", min_area_px, augment=True,
                            seed=seed)
    va = SegManifestDataset(doc, images_root, "val", min_area_px, augment=False,
                            seed=seed)
    if len(tr) == 0:
        raise SystemExit("ERROR: the training split has no annotated frames. "
                         "See splits/splits_summary.json.")
    print(f"train={len(tr)} val={len(va)} frames | {len(classes)} classes "
          f"{classes} | backend=maskrcnn (BSD-3-Clause)")

    dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers,
                    collate_fn=collate)
    vdl = (DataLoader(va, batch_size=batch, num_workers=workers,
                      collate_fn=collate) if len(va) else None)

    model = MaskRCNNSegmenter.build(len(classes), pretrained=pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best, history = float("inf"), []
    for ep in range(epochs):
        model.train(); tot = n = 0; t0 = time.perf_counter()
        for imgs, targets in dl:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if all(len(t["labels"]) == 0 for t in targets):
                continue                    # torchvision needs >=1 box
            losses = model(imgs, targets)
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); n += 1
        sched.step()
        row = {"epoch": ep, "train_loss": tot / max(1, n),
               "seconds": round(time.perf_counter() - t0, 1)}

        # torchvision returns losses only in train() mode, so validation loss
        # is computed with the model in train() under no_grad - it changes no
        # weights and no BatchNorm statistics are updated by a forward alone.
        if vdl is not None:
            vt = vn = 0
            with torch.no_grad():
                for imgs, targets in vdl:
                    imgs = [i.to(device) for i in imgs]
                    targets = [{k: v.to(device) for k, v in t.items()}
                               for t in targets]
                    if all(len(t["labels"]) == 0 for t in targets):
                        continue
                    vt += float(sum(model(imgs, targets).values())); vn += 1
            row["val_loss"] = vt / max(1, vn)
            if row["val_loss"] < best:
                best = row["val_loss"]
                torch.save({"model": model.state_dict(), "classes": list(classes),
                            "backend": "maskrcnn"}, out / "best.pt")
        history.append(row)
        print(f"  epoch {ep:3d} " + " ".join(f"{k}={v}" for k, v in row.items()
                                             if k != "epoch"))

    torch.save({"model": model.state_dict(), "classes": list(classes),
                "backend": "maskrcnn"}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"-> {out}")
    return history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="prepare_dataset output dir")
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip COCO-pretrained weights (needs network access)")
    a = p.parse_args(argv)
    train(a.dataset, a.images_root, a.out, a.epochs, a.batch, a.lr, a.device,
          a.workers, a.seed, pretrained=not a.no_pretrained)


if __name__ == "__main__":
    main()
