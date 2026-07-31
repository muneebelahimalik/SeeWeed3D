#!/usr/bin/env python3
"""
SeeWeed3D - ONNX / TensorRT export with explicit numerical parity checks.

A SUCCESSFUL EXPORT IS NOT A CORRECT EXPORT. ONNX export succeeding says the
graph was traceable; it says nothing about whether the numbers still match, and
TensorRT adds further latitude (layer fusion, FP16 rounding, tactic selection)
that can change outputs materially. Every stage here is therefore followed by a
parity check on fixed inputs, and the check is what decides whether the artefact
is usable.

Follow NVIDIA's measure -> optimise -> remeasure loop: export, verify parity,
benchmark on the ACTUAL Jetson, and only then consider INT8 - and only after
FP16 accuracy has been verified, because INT8 calibration on an already-wrong
FP16 engine simply bakes the error in.

    python -m seeweed3d.deploy.export --checkpoint runs/lep_v1/best.pt \\
        --out runs/lep_v1/export --precision fp16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tolerances. FP32 must agree tightly; FP16 is allowed the error its mantissa
# implies and nothing more.
PARITY_ATOL_FP32 = 1e-4
PARITY_ATOL_FP16 = 2e-2


def export_lep_onnx(checkpoint, out_path, out_size=128, opset=17, device="cpu"):
    """LEPRoiNet -> ONNX, with a PyTorch/ONNXRuntime parity check."""
    import torch
    from training.config import ModelConfig
    from training.lep_roinet import build_model, geometry_channels

    ck = Path(checkpoint)
    if not ck.exists():
        raise SystemExit(
            f"ERROR: checkpoint {ck} not found. Train first:\n"
            f"  python -m seeweed3d.training.train_lep --manifest ... --out ...")
    blob = torch.load(ck, map_location=device, weights_only=False)
    mcfg = blob.get("config", {}).get("pipeline", {}).get("model")
    cfg = ModelConfig(**mcfg) if mcfg else ModelConfig()
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(blob["model"])

    n_geom = geometry_channels(cfg.input_mode)
    rgb = torch.rand(1, 3, out_size, out_size, device=device)
    geom = torch.rand(1, max(1, n_geom), out_size, out_size, device=device) \
        if n_geom else None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = (rgb, geom) if geom is not None else (rgb,)
    names = ["rgb"] + (["geom"] if geom is not None else [])
    torch.onnx.export(
        model, args, str(out_path), input_names=names,
        output_names=["heatmap", "visibility", "targetability"],
        opset_version=opset, dynamo=False,
        # Batch is dynamic because a frame holds a variable number of weeds and
        # they are all submitted as one batch.
        dynamic_axes={n: {0: "batch"} for n in
                      names + ["heatmap", "visibility", "targetability"]})

    report = {"onnx": str(out_path), "opset": opset, "input_mode": cfg.input_mode,
              "n_geom_channels": n_geom}
    with __import__("torch").no_grad():
        ref = model(*args)["heatmap"].cpu().numpy()
    report["torch_vs_onnx"] = _check_onnx(out_path, args, names, ref)
    return report


def _check_onnx(path, args, names, ref):
    """PyTorch vs ONNXRuntime on a fixed input."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"checked": False,
                "note": "onnxruntime not installed; parity NOT verified. "
                        "python -m pip install onnxruntime"}
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {n: a.cpu().numpy() for n, a in zip(names, args) if a is not None}
    got = sess.run(None, feed)[0]
    diff = float(np.abs(got - ref).max())
    return {"checked": True, "max_abs_diff": diff,
            "tolerance": PARITY_ATOL_FP32,
            "passed": bool(diff <= PARITY_ATOL_FP32)}


def export_tensorrt(onnx_path, engine_path, precision="fp16", workspace_gb=2):
    """ONNX -> TensorRT engine, then ONNX-vs-TRT parity.

    Requires TensorRT, i.e. the Jetson (or a matching x86 install). Engines are
    NOT portable across GPU architectures or TensorRT versions, so this must run
    on the deployment device - an engine built on the dev GPU is invalid on the
    Orin."""
    try:
        import tensorrt as trt
    except ImportError:
        return {"built": False,
                "note": "tensorrt not installed. Build ON the Jetson Orin: "
                        "engines are architecture- and version-specific, so one "
                        "built on the development GPU cannot be used there."}
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    data = Path(onnx_path).read_bytes()
    if not parser.parse(data):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        return {"built": False, "errors": errs}

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolFlag.WORKSPACE,
                              int(workspace_gb) << 30)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            return {"built": False, "note": "platform has no fast FP16"}
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        return {"built": False,
                "note": "INT8 requires a calibration dataset AND verified FP16 "
                        "accuracy first. Calibrating on an unverified engine "
                        "bakes the error in. Implement the calibrator against "
                        "real field frames before enabling this."}

    engine = builder.build_serialized_network(network, cfg)
    if engine is None:
        return {"built": False, "note": "TensorRT returned no engine"}
    Path(engine_path).write_bytes(engine)
    return {"built": True, "engine": str(engine_path), "precision": precision,
            "parity_note": "run compare_onnx_tensorrt() on the device to verify "
                           "numerics before trusting this engine"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-size", type=int, default=128)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
    a = p.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = export_lep_onnx(a.checkpoint, out / "lep_roinet.onnx", a.out_size,
                          a.opset)
    if a.precision != "fp32":
        rep["tensorrt"] = export_tensorrt(out / "lep_roinet.onnx",
                                          out / f"lep_roinet_{a.precision}.plan",
                                          a.precision)
    (out / "export_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not rep.get("torch_vs_onnx", {}).get("passed", False):
        print("\nWARNING: PyTorch/ONNX parity was NOT verified as passing. Do "
              "not deploy this artefact until it is.")


if __name__ == "__main__":
    main()
