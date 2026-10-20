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


V3_UNITS_PER_MM = 10          # v3 stores tenths of a millimetre
V3_CONF_SCALE = 100           # ... and confidence at 100 counts per percent


def _write_session(folder, fmt, n=24, drop=None):
    folder.mkdir(parents=True, exist_ok=True)
    if fmt == "v3":
        return _write_session_v3(folder, n=n, drop=drop)
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


def _write_session_v3(folder, n=24, drop=None):
    """A WeedScan v2.0.0 recording, as DATA_SPEC.md describes it.

    Three things here are the point, and each is a silent corruption if the
    reader treats it as v2: depth is in TENTHS of a millimetre, 65534/65535 are
    sentinels that decode to a plausible 6.5 m, and confidence is gray16le
    rather than 8-bit. `truth` records the MILLIMETRE values an extractor is
    supposed to land on, so a test can tell "scaled correctly" from "happened
    to run"."""
    wr = {"rgb": _ffv1(folder / "RGB_left.mkv", "bgr24"),
          "depth": _ffv1(folder / "Depth.mkv", "gray16le"),
          "conf": _ffv1(folder / "Confidence.mkv", "gray16le")}
    f = open(folder / "frames.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["rec_frame_idx", "grab_idx", "mkv_written", "mkv_frame_idx",
                "svo_frame_ok", "image_timestamp_ns", "system_time_ns",
                "dt_ms", "est_missing_before", "exposure_pct",
                "exposure_ms_approx", "gain", "aec_agc", "wb_temp_k",
                "pose_state", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz",
                "qw", "speed_m_s", "distance_m", "tilt_from_nadir_deg"])
    truth, rec, mkv_i = {}, 0, 0
    for v in range(n):
        # A DROPPED FRAME GETS A ROW WITH NO mkv INDEX. Reading v3 in file
        # order instead of by mkv_frame_idx shifts every later frame's
        # timestamp and pose by one, undetectably.
        if drop is not None and rec == drop:
            w.writerow([rec, rec, 0, "", 1,
                        1_700_000_000_000_000_000 + rec * int(1e9 / FPS),
                        0, 66.7, 0, 20, 1.3, 20, 0, 4600, "OK",
                        0, 0, 0, 0, 0, 0, 1, 0.25, rec * 0.017, 2.0])
            rec += 1
        bgr, depth_mm, conf_pct = _frame(v)
        d16 = (depth_mm.astype(np.int64) * V3_UNITS_PER_MM).astype(np.uint16)
        d16[depth_mm == 0] = 0
        d16[0, 0] = 65534                         # nearer than the minimum
        d16[0, 1] = 65535                         # past the maximum
        wr["rgb"].stdin.write(bgr.tobytes())
        wr["depth"].stdin.write(d16.tobytes())
        c16 = (conf_pct.astype(np.int64) * V3_CONF_SCALE).astype(np.uint16)
        c16[conf_pct == 255] = 65535              # undefined
        wr["conf"].stdin.write(c16.tobytes())
        ts = 1_700_000_000_000_000_000 + rec * int(1e9 / FPS)
        w.writerow([rec, rec, 1, mkv_i, 1, ts, ts, 66.7, 0, 20, 1.3, 20, 0,
                    4600, "OK", rec * 17.0, 0, 0, 0, 0, 0, 1,
                    0.25, round(rec * 0.017, 4), 2.0])
        # The MILLIMETRES an extractor must land on, not the counts written.
        # Same keys as v1/v2 so the shared lossless/index tests cover v3 too -
        # which makes them a check that the rescale is right, rather than a
        # check that it ran. The two sentinel pixels become invalid (0).
        expect_mm = depth_mm.astype(np.int64).copy()
        expect_mm[0, 0] = 0
        expect_mm[0, 1] = 0
        truth[mkv_i] = {"capture_idx": rec, "timestamp_ns": ts,
                        "depth_sum": int(expect_mm.sum()),
                        "depth_at": int(depth_mm[H // 2, W // 2]),
                        "distance_m": round(rec * 0.017, 4)}
        rec += 1
        mkv_i += 1
    f.close()
    for pr in wr.values():
        pr.stdin.close(); pr.wait()

    with open(folder / "markers.csv", "w", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow(["rec_frame_idx", "image_timestamp_ns", "system_time_iso",
                     "text"])
        mw.writerow([2, 1_700_000_000_000_000_000, "t", "start row 3"])
    json.dump({
        "status": "COMPLETE",
        "field": {"site": "s", "plot_id": "plotA", "crop_stage": "3-leaf",
                  "camera_height_m": 1.0, "travel_direction": "top_to_bottom"},
        "encoding": {"depth_units_per_mm": V3_UNITS_PER_MM,
                     "depth_codes": {"invalid": 0, "too_close": 65534,
                                     "too_far": 65535},
                     "confidence_scale": V3_CONF_SCALE},
        "capture": {"fps": FPS, "depth_mode": "NEURAL"},
        "verification": {"all_ok": True},
    }, open(folder / "session_metadata.json", "w"))
    json.dump({
        "serial_number": 1, "camera_model": "ZED 2i",
        "camera_firmware_version": "1", "resolution": [W, H], "fps": FPS,
        "calibration_rectified": {
            "left_cam": {"fx": 350.0, "fy": 350.0, "cx": 160.0, "cy": 120.0,
                         "disto": [0] * 5, "h_fov_deg": 80.0,
                         "image_size": [W, H]},
            "right_cam": {"fx": 350.0, "fy": 350.0, "cx": 160.0, "cy": 120.0,
                          "disto": [0] * 5},
            "baseline": 120.0,
            "stereo_transform": {"translation": [-120.0, 0, 0],
                                 "rotation_vector_rad": [0, 0, 0]}},
        "camera_imu_transform": {"translation": [0, 0, 0]},
    }, open(folder / "calibration.json", "w"))
    return truth


@pytest.fixture(scope="session")
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    truth = {
        "vid_20250304_142804": _write_session(root / "A" / "Session_20250304_142804", "v1"),
        "vid_20250305_090000": _write_session(root / "A" / "Session_20250305_090000", "v1"),
        "vid_20260108_103135": _write_session(root / "B" / "Session_20260108_103135", "v2", drop=7),
        "vid_20261008_101500": _write_session(
            root / "C" / "Session_20261008_101500_plotA", "v3", drop=5),
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
            {"path": str(raw_root["root"] / "C"), "trip": "vid", "site": "s",
             "field": "f", "scene_hint": "mixed", "notes": ""},
        ],
        "OUTPUT_ROOT": str(out), "FFMPEG": "ffmpeg", "FFPROBE": "ffprobe",
        "TARGET_POOL_FPS": 15.0, "MAX_POOL_PER_SESSION": None,
        "OVERWRITE": True, "DRY_RUN": False,
    })
    s1.main()
    return out
