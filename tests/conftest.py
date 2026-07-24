"""Shared test fixtures: build synthetic v1/v2 ZED sessions with real FFV1 MKVs
and run the extraction pipeline once, so tests exercise the real code paths.

ffmpeg/ffprobe must be on PATH. SAM 3 is never invoked here (GPU + gated
weights); the prelabel test stubs it.
"""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "seeweed3d"
W, H, FPS = 320, 240, 15
_rng = np.random.default_rng(0)


def load_script(rel_path):
    """Import a pipeline script (clean module name) by file path."""
    path = PKG / rel_path
    name = "sw_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ffv1(path, pix_fmt):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", pix_fmt, "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", pix_fmt, str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _frame(i):
    bgr = _rng.integers(60, 110, (H, W, 3), dtype=np.uint8)
    bgr[:, :, 1] //= 2
    depth = np.full((H, W), 800 + (i % 7) * 30, np.uint16)
    depth += _rng.integers(0, 20, (H, W), dtype=np.uint16)
    veg = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for k in range(3):
        cy = 60 + (i * 31 + k * 70) % (H - 120)
        cx = 60 + (i * 47 + k * 90) % (W - 120)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2
        bgr[blob] = (40, 200, 40)
        veg |= blob
    holes = (_rng.random((H, W)) < 0.05) & ~veg
    depth[holes] = 0
    conf = _rng.integers(1, 101, (H, W)).astype(np.uint8)
    conf[:15, :15] = 255                     # no-measure sentinel
    return np.ascontiguousarray(bgr), depth, conf


def _write_session(folder, fmt, n=24, drop=None):
    folder.mkdir(parents=True, exist_ok=True)
    wr = {"rgb": _ffv1(folder / "RGB_video.mkv", "bgr24"),
          "depth": _ffv1(folder / "Depth_video.mkv", "gray16le")}
    if fmt == "v2":
        wr["right"] = _ffv1(folder / "RGB_right_video.mkv", "bgr24")
        wr["conf"] = _ffv1(folder / "Confidence_video.mkv", "gray")
    f = open(folder / "frames.csv", "w", newline="")
    w = csv.writer(f)
    if fmt == "v2":
        w.writerow(["video_frame_idx", "capture_frame_idx", "image_timestamp_ns",
                    "host_monotonic_ns", "host_realtime_ns", "system_time_iso",
                    "width", "height", "fps", "exposure", "gain", "wb_temp",
                    "aec_agc", "pose_state", "tx_mm", "ty_mm", "tz_mm",
                    "qx", "qy", "qz", "qw", "ax", "ay", "az", "gx", "gy", "gz"])
    else:
        w.writerow(["frame_idx", "timestamp_ns", "system_time_iso",
                    "width", "height", "fps"])
    truth, cap = {}, 0
    for v in range(n):
        if drop is not None and cap == drop:
            cap += 1
        bgr, depth, conf = _frame(v)
        wr["rgb"].stdin.write(bgr.tobytes())
        wr["depth"].stdin.write(depth.tobytes())
        if fmt == "v2":
            wr["right"].stdin.write(bgr.tobytes())
            wr["conf"].stdin.write(conf.tobytes())
        ts = 1_700_000_000_000_000_000 + cap * int(1e9 / FPS)
        if fmt == "v2":
            w.writerow([v, cap, ts, ts, ts, "t", W, H, FPS, 20, 20, 4600, 0,
                        "OK", 0, 0, 0, 0, 0, 0, 1, 0, 0, 9.8, 0, 0, 0])
        else:
            w.writerow([cap, ts, "t", W, H, FPS])
        truth[v] = {"depth_sum": int(depth.sum()), "capture_idx": cap,
                    "depth_at": int(depth[H // 2, W // 2])}
        cap += 1
    f.close()
    for p in wr.values():
        p.stdin.close(); p.wait()

    if fmt == "v2":
        with open(folder / "dropped_frames.csv", "w", newline="") as df:
            dw = csv.writer(df); dw.writerow(["capture_frame_idx"])
            if drop is not None:
                dw.writerow([drop])
        json.dump({"schema_version": "seeweed3d/capture/2.0",
                   "confidence_encoding": {"range": [0, 100], "no_measure_sentinel": 255,
                       "polarity": {"determined": True}}},
                  open(folder / "session.json", "w"))
        json.dump({"camera_model": "ZED 2i", "serial_number": 1,
                   "rectified": {"left": {"fx": 350.0, "fy": 350.0, "cx": 160.0,
                       "cy": 120.0, "disto": [0] * 5, "h_fov": 80.0},
                       "right": {"fx": 350.0, "fy": 350.0, "cx": 160.0, "cy": 120.0,
                       "disto": [0] * 5}, "translation_mm": [-120.0, 0, 0],
                       "baseline_mm": 120.0, "rotation_vector_rad": [0, 0, 0]}},
                  open(folder / "calibration.json", "w"))
    else:
        (folder / "calibration_params.txt").write_text(
            "Left Camera Intrinsic Parameters:\nfx: 350.0, fy: 350.0\n"
            "cx: 160.0, cy: 120.0\nRight Camera Intrinsic Parameters:\n"
            "fx: 350.0, fy: 350.0\ncx: 160.0, cy: 120.0\n"
            "Stereo Translation:\nTx: -120.0, Ty: 0.0, Tz: 0.0\n"
            "Horizontal FOV (Left Cam): 80.0\n")
        (folder / "session_meta.txt").write_text("model: ZED 2i\nfps: 15\n"
                                                 "depth_vis_max_mm: 3000\n")
    return truth


@pytest.fixture(scope="session")
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    truth = {
        "vid_20250304_142804": _write_session(root / "A" / "Session_20250304_142804", "v1"),
        "vid_20250305_090000": _write_session(root / "A" / "Session_20250305_090000", "v1"),
        "vid_20260108_103135": _write_session(root / "B" / "Session_20260108_103135", "v2", drop=7),
    }
    return {"root": root, "truth": truth}


@pytest.fixture(scope="session")
def extracted_root(raw_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    s1 = load_script("extraction/extract_sessions.py")
    s1.CONFIG.update({
        "INPUT_ROOTS": [
            {"path": str(raw_root["root"] / "A"), "trip": "vid", "site": "s",
             "field": "f", "scene_hint": "onion_only", "notes": ""},
            {"path": str(raw_root["root"] / "B"), "trip": "vid", "site": "s",
             "field": "f", "scene_hint": "mixed", "notes": ""},
        ],
        "OUTPUT_ROOT": str(out), "FFMPEG": "ffmpeg", "FFPROBE": "ffprobe",
        "TARGET_POOL_FPS": 15.0, "MAX_POOL_PER_SESSION": None,
        "OVERWRITE": True, "DRY_RUN": False,
    })
    s1.main()
    return out
