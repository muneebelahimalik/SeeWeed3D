#!/usr/bin/env python3
"""
SeeWeed3D - latency / memory benchmarking.

MEASURE ON THE TARGET. Numbers from a development GPU do not transfer to a
Jetson Orin: different memory bandwidth, different power/clock behaviour, and a
shared CPU-GPU memory system. This script records the device it ran on in every
result so a Jetson number can never be confused with a desktop one.
"""
from __future__ import annotations

import argparse, json, platform, statistics, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def device_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "is_jetson": Path("/etc/nv_tegra_release").exists()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    if not info["is_jetson"]:
        info["WARNING"] = ("NOT a Jetson. These numbers must not be reported as "
                           "Jetson Orin performance.")
    return info


def bench_lep(model, batch_sizes=(1, 8, 32), size=128, n_geom=3, iters=50,
              warmup=10, device="cpu"):
    """Latency vs number of weed ROIs - the axis that actually varies in the
    field, since a dense frame holds far more plants than a sparse one."""
    import torch
    model = model.to(device).eval()
    rows = []
    for b in batch_sizes:
        rgb = torch.rand(b, 3, size, size, device=device)
        geom = torch.rand(b, n_geom, size, size, device=device) if n_geom else None
        with torch.no_grad():
            for _ in range(warmup):
                model(rgb, geom)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                model(rgb, geom)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
        rows.append({"batch": b,
                     "p50_ms": float(np.percentile(ts, 50)),
                     "p95_ms": float(np.percentile(ts, 95)),
                     "mean_ms": float(statistics.mean(ts)),
                     "ms_per_roi": float(np.percentile(ts, 50) / b)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="LEPRoiNet checkpoint; omit to benchmark a fresh model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--iters", type=int, default=50)
    a = p.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("ERROR: torch is required to benchmark. "
                         "python -m pip install -r requirements-training.txt")
    import torch
    from training.config import ModelConfig
    from training.lep_roinet import build_model, geometry_channels

    cfg = ModelConfig()
    if a.checkpoint:
        blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        mcfg = blob.get("config", {}).get("pipeline", {}).get("model")
        if mcfg:
            cfg = ModelConfig(**mcfg)
        model = build_model(cfg)
        model.load_state_dict(blob["model"])
    else:
        model = build_model(cfg)

    report = {"device": device_info(), "model": cfg.__dict__,
              "lep_latency": bench_lep(model, n_geom=geometry_channels(cfg.input_mode),
                                       iters=a.iters, device=a.device),
              "note": ("Stage A (YOLO) and end-to-end latency require real "
                       "weights and a real frame; run scripts/run_inference.py "
                       "on the target device for those.")}
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
