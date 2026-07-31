#!/usr/bin/env python3
"""
SeeWeed3D - Stage B training: LEPRoiNet.

    python -m seeweed3d.training.train_lep \
        --manifest D:/.../mixed_v1/lep_manifest.json \
        --images-root D:/Dataset_Vidalia/sessions \
        --out D:/.../runs/lep_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import (ExperimentConfig, LossWeights,  # noqa: E402
                             ModelConfig, PipelineConfig, TrainConfig)


def train(manifest, images_root, out_dir, cfg=None, epochs=None, batch=None,
          device=None, input_mode=None, seed=None):
    import torch
    from torch.utils.data import DataLoader
    from training.lep_dataset import LEPRoiDataset
    from training.lep_roinet import build_model
    from training.losses import LEPLoss

    cfg = cfg or ExperimentConfig()
    if epochs: cfg.train.epochs = epochs
    if batch: cfg.train.batch_size = batch
    if device: cfg.train.device = device
    if seed: cfg.train.seed = seed
    if input_mode: cfg.pipeline.model.input_mode = input_mode

    mpath = Path(manifest)
    if not mpath.exists():
        raise SystemExit(
            f"ERROR: {mpath} not found.\n"
            f"Build it from your verified CVAT export first:\n"
            f"  python -m seeweed3d.training.prepare_dataset "
            f"--datumaro-root <export> --images-root <sessions> --out <dir>")
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    if not doc.get("rows"):
        raise SystemExit(
            f"ERROR: {mpath} contains no LEP rows. Every trainable row needs a "
            f"weed mask GROUPED with its weed_LEP point in CVAT. Check "
            f"annotations_needing_correction.json next to the manifest.")

    torch.manual_seed(cfg.train.seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tr = LEPRoiDataset(doc, images_root, "train", cfg.pipeline, augment=True,
                       seed=cfg.train.seed)
    va = LEPRoiDataset(doc, images_root, "val", cfg.pipeline, augment=False,
                       seed=cfg.train.seed)
    if len(tr) == 0:
        raise SystemExit(
            "ERROR: the training split has no LEP rows. Check the split "
            "assignment in splits/splits_summary.json.")
    print(f"train={len(tr)} val={len(va)} rows | mode={cfg.pipeline.model.input_mode}")

    dl = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=cfg.train.num_workers, drop_last=False)
    vdl = (DataLoader(va, batch_size=cfg.train.batch_size,
                      num_workers=cfg.train.num_workers) if len(va) else None)

    model = build_model(cfg.pipeline.model).to(cfg.train.device)
    crit = LEPLoss(cfg.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.epochs)

    best, history = float("inf"), []
    for ep in range(cfg.train.epochs):
        model.train(); tot = n = 0
        for b in dl:
            opt.zero_grad()
            geom = b["geom"].to(cfg.train.device) if b["geom"].shape[1] else None
            loss, parts = crit(model(b["rgb"].to(cfg.train.device), geom),
                               {k: v.to(cfg.train.device) for k, v in b.items()
                                if k not in ("rgb", "geom")})
            loss.backward(); opt.step()
            tot += float(loss.detach()); n += 1
        sched.step()
        row = {"epoch": ep, "train_loss": tot / max(1, n)}

        if vdl is not None:
            model.eval(); vt = vn = 0
            with torch.no_grad():
                for b in vdl:
                    geom = (b["geom"].to(cfg.train.device)
                            if b["geom"].shape[1] else None)
                    l, _ = crit(model(b["rgb"].to(cfg.train.device), geom),
                                {k: v.to(cfg.train.device) for k, v in b.items()
                                 if k not in ("rgb", "geom")})
                    vt += float(l); vn += 1
            row["val_loss"] = vt / max(1, vn)
            if row["val_loss"] < best:
                best = row["val_loss"]
                torch.save({"model": model.state_dict(),
                            "config": cfg.to_dict()}, out / "best.pt")
        history.append(row)
        print(f"  epoch {ep:3d} " + " ".join(f"{k}={v:.4f}"
                                             for k, v in row.items() if k != "epoch"))

    torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
               out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"-> {out}")
    return history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--input-mode", default=None,
                   choices=["rgb", "rgb_mask", "rgb_mask_geom"],
                   help="ablation: rgb | rgb_mask | rgb_mask_geom")
    a = p.parse_args(argv)
    train(a.manifest, a.images_root, a.out, epochs=a.epochs, batch=a.batch,
          device=a.device, input_mode=a.input_mode, seed=a.seed)


if __name__ == "__main__":
    main()
