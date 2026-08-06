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


def _preview_frames(doc, images_root, n=6):
    """A stable sample of val frames for the prediction overlays.

    Evenly spaced rather than random so the same ground is compared across
    epochs and across runs - a preview panel that changes frames every epoch
    cannot show you whether anything improved."""
    va = [f for f in doc["frames"] if f.get("split") == "val"]
    if not va:
        return []
    step = max(1, len(va) // n)
    return va[::step][:n]


def train(dataset_dir, images_root, out_dir, epochs=20, batch=2, lr=5e-3,
          device="cpu", workers=0, seed=1234, pretrained=True, min_area_px=16,
          track="auto", preview_every=5, eval_every=0):
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

    from training.tracking import Tracker
    trk = Tracker(track, out_dir=out, run_name=out.name)
    trk.log_params({
        "backend": "maskrcnn", "dataset": str(dataset_dir),
        "epochs": epochs, "batch": batch, "lr": lr, "device": device,
        "seed": seed, "pretrained": pretrained, "min_area_px": min_area_px,
        "n_train": len(tr), "n_val": len(va), "classes": classes,
        "dataset_kind": doc.get("dataset_kind", "unknown"),
        "split_strategy": doc.get("split_strategy", "unknown"),
    })
    print(trk.hint())
    previews = _preview_frames(doc, images_root) if preview_every else []

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
        row["lr"] = float(opt.param_groups[0]["lr"])

        last_ep = (ep == epochs - 1)
        if eval_every and (last_ep or (ep + 1) % eval_every == 0) \
                and (out / "best.pt").exists():
            # mAP on the current BEST checkpoint, not the live weights: the
            # curve should track the model you would actually ship.
            try:
                from evaluation.eval_seg import evaluate
                res = evaluate(out / "best.pt", dataset_dir, images_root,
                               "val", device)
                row["val_map50"] = res["summary"]["map50"]
                row["val_map50_95"] = res["summary"]["map50_95"]
                miss = res["crop_safety"].get("missed_onion_fraction")
                if miss is not None:
                    row["missed_onion_fraction"] = miss
            except Exception as e:                      # never kill a run for a metric
                print(f"  [warn] mid-training eval failed: {e}")

        if previews and (last_ep or (ep + 1) % preview_every == 0):
            _log_previews(trk, model, previews, images_root, classes, device,
                          min_area_px, ep)

        history.append(row)
        trk.log_metrics(row, step=ep)
        print(f"  epoch {ep:3d} " + " ".join(f"{k}={v}" for k, v in row.items()
                                             if k != "epoch"))

    torch.save({"model": model.state_dict(), "classes": list(classes),
                "backend": "maskrcnn"}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    for f in ("best.pt", "history.json", "params.json"):
        trk.log_artifact(out / f)
    trk.close()
    print(f"-> {out}")
    print(trk.hint())
    return history


def _log_previews(trk, model, records, images_root, classes, device,
                  min_area_px, step):
    """GT-vs-prediction overlays on fixed val frames."""
    import cv2
    import numpy as np
    import torch
    from common.ontology import CROP_CLASS
    from training.seg_dataset import polygons_to_mask, resolve_image
    from training.tracking import overlay_masks, side_by_side

    was_training = model.training
    model.eval()
    try:
        for k, rec in enumerate(records):
            try:
                path = resolve_image(rec["image_path"], images_root,
                                     rec.get("session_id"))
            except FileNotFoundError:
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]

            gt_m, gt_n = [], []
            for inst in rec["instances"]:
                m = polygons_to_mask(inst["polygons"], h, w).astype(bool)
                if int(m.sum()) >= min_area_px:
                    gt_m.append(m); gt_n.append(inst["class_name"])

            t = torch.from_numpy(bgr[:, :, ::-1].copy().astype(np.float32)
                                 / 255.0).permute(2, 0, 1).to(device)
            with torch.no_grad():
                o = model([t])[0]
            keep = o["scores"] >= 0.5
            pm = (o["masks"][keep][:, 0] >= 0.5).cpu().numpy()
            pn = [classes[int(i) - 1] for i in o["labels"][keep].cpu().numpy()]

            panel = side_by_side(overlay_masks(bgr, gt_m, gt_n, CROP_CLASS),
                                 overlay_masks(bgr, pm, pn, CROP_CLASS))
            trk.log_image(f"preview/{k:02d}", panel[:, :, ::-1], step)
    except Exception as e:
        print(f"  [warn] preview rendering failed: {e}")
    finally:
        if was_training:
            model.train()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="prepare_dataset output dir")
    p.add_argument("--images-root", required=True, nargs="+",
                   help="sessions root(s); more than one if the datasets were "
                        "not captured under a common parent")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip COCO-pretrained weights (needs network access)")
    p.add_argument("--track", default="auto",
                   choices=["auto", "none", "tensorboard", "mlflow", "all"],
                   help="local experiment tracking; nothing is uploaded")
    p.add_argument("--preview-every", type=int, default=5,
                   help="log GT-vs-prediction overlays every N epochs (0=off)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="compute val mAP every N epochs (0=off; it is slow)")
    a = p.parse_args(argv)
    train(a.dataset, a.images_root, a.out, a.epochs, a.batch, a.lr, a.device,
          a.workers, a.seed, pretrained=not a.no_pretrained, track=a.track,
          preview_every=a.preview_every, eval_every=a.eval_every)


if __name__ == "__main__":
    main()
