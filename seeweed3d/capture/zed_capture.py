#!/usr/bin/env python3
"""
SeeWeed3D - ZED capture application v2
======================================
Rewrite of zed_app_updated_2026_jan.py. Fixes the five limitations that made the
existing recordings unable to support 3D LEP error measurement.

WHAT CHANGED AND WHY  (full rationale in docs/capture_changelog.md)
--------------------------------------------------------------------
 1. SVO2 archive.        The SDK records the raw stereo stream on every grab().
                         This single change retroactively fixes "no right image"
                         and "no SVO retained", and lets depth be recomputed
                         later with any DEPTH_MODE. It is also written inside
                         grab(), so it CANNOT be dropped by our writer queue.
 2. Confidence map.      MEASURE.CONFIDENCE is retrieved and stored, giving a
                         real per-pixel validity signal instead of only the
                         0 sentinel. Polarity is probed empirically at startup
                         rather than assumed.
 3. Right view.          VIEW.RIGHT optionally written as a rectified pair.
 4. Exposure discipline. AEC/AGC and auto white balance can be LOCKED, and the
                         actual exposure/gain/WB are logged EVERY frame.
 5. Full frame accounting. Dropped frames are recorded in dropped_frames.csv
                         with their capture index, so gaps are explicit and
                         recoverable instead of silent.

Plus: full calibration dump (both cameras, raw + rectified, stereo rotation),
JSON session metadata, live disk-throughput and drop monitoring, host clocks
logged alongside the ZED clock for ROS 2 bag alignment, and a startup bandwidth
estimate that warns before a run rather than after.

STATUS: the data-path logic is unit-tested against a stubbed SDK. It has NOT
been run against a physical camera. Bench-test with BENCH_CHECK below before a
field session.
"""

import csv
import datetime
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl

# =============================================================================
# CONFIG - EDIT THIS BLOCK ONLY
# =============================================================================

CONFIG = {
    # -- Where sessions are written -------------------------------------------
    "BASE_DIRECTORY": r"M:/Research/Data/Vidalia_onions_visit3",

    # Optional audio cues; set to None to disable.
    "WAVE_START": r"C:/Users/mm17889/Zed2iData/working.wav",
    "WAVE_STOP":  r"C:/Users/mm17889/Zed2iData/save.wav",

    # -- Field context, written into session.json. UPDATE PER SESSION. --------
    "SITE":         "vidalia",
    "FIELD":        "field_A",
    "PLOT":         "",
    "ROW":          "",
    "GROWTH_STAGE": "",            # e.g. "2-leaf"
    "WEATHER":      "",            # e.g. "overcast, dry"
    "OPERATOR":     "",
    "MOUNT_HEIGHT_MM": 900,        # camera height above soil
    "VEHICLE_SPEED_MPS": 0.0,      # 0 = static
    "NOTES":        "",

    # -- What to record --------------------------------------------------------
    # SVO is the archive. Leave it on unless disk throughput forces otherwise:
    # everything else is derivable from it, and it never drops frames.
    "RECORD_SVO":        True,
    "RECORD_LEFT_MKV":   True,     # convenience copy for the existing pipeline
    "RECORD_RIGHT_MKV":  False,    # redundant with SVO; on only if not using SVO
    "RECORD_DEPTH_MKV":  True,     # uint16 millimetres, 0 = invalid
    "RECORD_CONF_MKV":   True,     # uint8 confidence, see polarity probe

    # SVO compression, tried in order until one is accepted by the SDK.
    "SVO_COMPRESSION_PREFERENCE": ["H265_LOSSLESS", "H264_LOSSLESS", "LOSSLESS"],

    # -- Camera ----------------------------------------------------------------
    "RESOLUTION": "HD2K",          # HD2K | HD1080 | HD720 | VGA
    "FPS": 15,
    "DEPTH_MODE": "NEURAL",        # NEURAL | NEURAL_PLUS | ULTRA | QUALITY
    "DEPTH_MIN_MM": 300,
    "DEPTH_MAX_MM": 3000,

    # -- Exposure discipline ---------------------------------------------------
    # Auto exposure changes brightness mid-row and makes appearance a function
    # of the sun rather than the plants. Lock it once lighting is set.
    "LOCK_EXPOSURE": True,
    "EXPOSURE":      20,           # 0-100 (SDK units). Lower = shorter exposure.
    "GAIN":          20,           # 0-100
    "LOCK_WHITE_BALANCE": True,
    "WHITEBALANCE_TEMPERATURE": 4600,   # Kelvin, 2800-6500

    # -- Pose ------------------------------------------------------------------
    # ZED visual-inertial odometry. Convenience only: it drifts badly over
    # repetitive crop rows. Record Amiga/ROS 2 odometry in parallel and join on
    # host_realtime_ns. See docs/capture_changelog.md section "Clock alignment".
    "ENABLE_POSITIONAL_TRACKING": True,
    "LOG_IMU": True,

    # -- Safety ----------------------------------------------------------------
    "FRAME_QUEUE_MAX":   240,
    "MIN_FREE_SPACE_GB": 20,
    "ABORT_ON_LOW_DISK": True,     # stop cleanly rather than corrupt a session
    "CONFIDENCE_PROBE_FRAMES": 30, # frames used to determine confidence polarity

    # Set True to run a hardware self-check and exit without recording.
    "BENCH_CHECK": False,
}

RESOLUTIONS = {"HD2K": (2208, 1242), "HD1080": (1920, 1080),
               "HD720": (1280, 720), "VGA": (672, 376)}

# =============================================================================


def sdk_get(zed, setting):
    """get_camera_settings returns (ERROR_CODE, value) on SDK 4.x and a bare int
    on 3.x. Normalise both, and never let a settings read kill a recording."""
    try:
        r = zed.get_camera_settings(setting)
    except Exception:
        return None
    if isinstance(r, tuple):
        err, val = r[0], r[1]
        return int(val) if err == sl.ERROR_CODE.SUCCESS else None
    try:
        return int(r)
    except (TypeError, ValueError):
        return None


def sdk_set(zed, setting, value):
    try:
        zed.set_camera_settings(setting, int(value))
        return True
    except Exception as e:
        print(f"[WARN] could not set {setting}: {e}")
        return False


def enum_or_none(enum_cls, name):
    return getattr(enum_cls, name, None)


def estimate_bandwidth(cfg, w, h, fps):
    """Warn about disk throughput BEFORE a field session, not after."""
    mb = 0.0
    if cfg["RECORD_LEFT_MKV"]:
        mb += w * h * 3
    if cfg["RECORD_RIGHT_MKV"]:
        mb += w * h * 3
    if cfg["RECORD_DEPTH_MKV"]:
        mb += w * h * 2
    if cfg["RECORD_CONF_MKV"]:
        mb += w * h * 1
    raw = mb * fps / 1e6
    # FFV1 on foliage typically lands around 45-60% of raw.
    return {"raw_mb_s": round(raw, 1),
            "expected_mkv_mb_s": round(raw * 0.55, 1),
            "svo_note": "SVO adds roughly 30-60 MB/s at HD2K15 depending on codec"}


def dump_calibration(zed, path):
    """Full dump: both cameras, rectified AND raw, plus stereo rotation.
    The old version saved only rectified fx/fy/cx/cy and Tx, which is not enough
    to re-rectify or to reason about the raw stereo pair."""
    info = zed.get_camera_information()
    cfgn = info.camera_configuration
    out = {"camera_model": str(info.camera_model),
           "serial_number": info.serial_number,
           "firmware": getattr(info, "camera_firmware_version", None),
           "resolution": [cfgn.resolution.width, cfgn.resolution.height],
           "fps": cfgn.fps}

    def cam(c):
        return {"fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
                "disto": [float(x) for x in np.array(c.disto).ravel()],
                "h_fov": getattr(c, "h_fov", None),
                "v_fov": getattr(c, "v_fov", None)}

    for tag, params in (("rectified", cfgn.calibration_parameters),
                        ("raw", getattr(cfgn, "calibration_parameters_raw", None))):
        if params is None:
            continue
        entry = {"left": cam(params.left_cam), "right": cam(params.right_cam)}
        try:
            t = params.stereo_transform.get_translation().get()
            entry["translation_mm"] = [float(x) for x in t]
            entry["baseline_mm"] = abs(float(t[0]))
            r = params.stereo_transform.get_rotation_vector()
            entry["rotation_vector_rad"] = [float(x) for x in np.array(r).ravel()]
        except Exception as e:
            entry["stereo_transform_error"] = str(e)
        out[tag] = entry

    fx = out.get("rectified", {}).get("left", {}).get("fx")
    if fx:
        out["derived"] = {
            "mm_per_px_at_1000mm": round(1000.0 / fx, 4),
            "note": "Ground sampling distance at 1 m. Compare against the PCK "
                    "thresholds before trusting sub-millimetre LEP targets."}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


class CaptureThread(threading.Thread):
    """Grabs at camera rate. SVO is written inside grab() by the SDK, so the SVO
    archive stays complete even when the MKV writer falls behind."""

    def __init__(self, zed, runtime, mats, frame_queue, latest_lock, latest,
                 stop_event, recording_flag, cfg):
        super().__init__(daemon=True)
        self.zed, self.runtime, self.mats = zed, runtime, mats
        self.q, self.latest_lock, self.latest = frame_queue, latest_lock, latest
        self.stop_event, self.recording_flag, self.cfg = stop_event, recording_flag, cfg
        self.frame_idx = 0
        self.dropped = []            # capture indices lost by the writer queue
        self.grab_failures = 0
        self.conf_probe = []

    def run(self):
        z, m, cfg = self.zed, self.mats, self.cfg
        want_conf = cfg["RECORD_CONF_MKV"]
        want_right = cfg["RECORD_RIGHT_MKV"]

        while not self.stop_event.is_set():
            if z.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
                self.grab_failures += 1
                time.sleep(0.002)
                continue

            z.retrieve_image(m["left"], sl.VIEW.LEFT)
            z.retrieve_measure(m["depth"], sl.MEASURE.DEPTH)
            if want_right:
                z.retrieve_image(m["right"], sl.VIEW.RIGHT)
            if want_conf:
                z.retrieve_measure(m["conf"], sl.MEASURE.CONFIDENCE)

            # Three clocks: ZED image clock, plus host clocks so the session can
            # be aligned to a ROS 2 bag or Amiga odometry log afterwards.
            try:
                ts_ns = z.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            except Exception:
                ts_ns = time.time_ns()
            host_mono = time.monotonic_ns()
            host_real = time.time_ns()

            left = cv2.cvtColor(m["left"].get_data(), cv2.COLOR_BGRA2BGR)
            right = (cv2.cvtColor(m["right"].get_data(), cv2.COLOR_BGRA2BGR)
                     if want_right else None)

            depth = np.nan_to_num(m["depth"].get_data(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            conf = m["conf"].get_data().copy() if want_conf else None

            # Empirically determine confidence polarity instead of trusting a
            # remembered doc string: compare confidence where depth is invalid
            # against where it is valid.
            if conf is not None and len(self.conf_probe) < cfg["CONFIDENCE_PROBE_FRAMES"]:
                bad = depth <= 0
                if bad.any() and (~bad).any():
                    self.conf_probe.append(
                        (float(np.nanmean(conf[bad])), float(np.nanmean(conf[~bad]))))

            settings = {
                "exposure": sdk_get(z, sl.VIDEO_SETTINGS.EXPOSURE),
                "gain": sdk_get(z, sl.VIDEO_SETTINGS.GAIN),
                "wb_temp": sdk_get(z, sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE),
                "aec_agc": sdk_get(z, sl.VIDEO_SETTINGS.AEC_AGC),
            }

            pose = self.read_pose()
            imu = self.read_imu() if cfg["LOG_IMU"] else {}

            with self.latest_lock:
                self.latest.update({"rgb_bgr": left, "depth_mm": depth,
                                    "timestamp_ns": ts_ns, "frame_idx": self.frame_idx,
                                    "settings": settings})

            if self.recording_flag.is_set():
                item = {"frame_idx": self.frame_idx, "ts_ns": ts_ns,
                        "host_mono_ns": host_mono, "host_real_ns": host_real,
                        "left": left.copy(),
                        "right": right.copy() if right is not None else None,
                        "depth": depth.copy(),
                        "conf": conf, "settings": settings,
                        "pose": pose, "imu": imu}
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    # Record WHICH frame was lost. The SVO still has it.
                    self.dropped.append(self.frame_idx)

            self.frame_idx += 1

    def read_pose(self):
        if not self.cfg["ENABLE_POSITIONAL_TRACKING"]:
            return {}
        try:
            p = sl.Pose()
            state = self.zed.get_position(p, sl.REFERENCE_FRAME.WORLD)
            t = p.get_translation().get()
            o = p.get_orientation().get()
            return {"pose_state": str(state),
                    "tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2]),
                    "qx": float(o[0]), "qy": float(o[1]),
                    "qz": float(o[2]), "qw": float(o[3])}
        except Exception:
            return {}

    def read_imu(self):
        try:
            s = sl.SensorsData()
            if self.zed.get_sensors_data(s, sl.TIME_REFERENCE.IMAGE) != sl.ERROR_CODE.SUCCESS:
                return {}
            a = s.get_imu_data().get_linear_acceleration()
            g = s.get_imu_data().get_angular_velocity()
            return {"ax": float(a[0]), "ay": float(a[1]), "az": float(a[2]),
                    "gx": float(g[0]), "gy": float(g[1]), "gz": float(g[2])}
        except Exception:
            return {}

    def confidence_polarity(self):
        """Returns a data-derived statement about what the confidence map means."""
        if not self.conf_probe:
            return {"determined": False,
                    "note": "no probe samples - confidence polarity unknown"}
        bad = float(np.mean([p[0] for p in self.conf_probe]))
        good = float(np.mean([p[1] for p in self.conf_probe]))
        return {"determined": True,
                "mean_confidence_where_depth_invalid": round(bad, 2),
                "mean_confidence_where_depth_valid": round(good, 2),
                "lower_value_means_more_confident": bool(good < bad),
                "note": "Derived from this session's own data. Use this rather "
                        "than assuming a polarity from documentation."}


class WriterThread(threading.Thread):
    """Encodes the enabled MKV streams and writes the per-frame CSV.

    frames.csv now carries video_frame_idx (the row's position in the encoded
    stream) alongside capture_frame_idx, so the mapping is explicit rather than
    an implicit consequence of row order."""

    def __init__(self, frame_queue, stop_event, save_dir, w, h, fps, cfg):
        super().__init__(daemon=True)
        self.q, self.stop_event = frame_queue, stop_event
        self.dir, self.w, self.h, self.fps, self.cfg = save_dir, w, h, fps, cfg
        self.procs = {}
        self.csv_file = self.csv_writer = None
        self.frames_written = 0
        self.broken_pipe = False
        self.bytes_written = 0

    def _ffmpeg(self, name, pix_fmt, out_name):
        path = os.path.join(self.dir, out_name)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", pix_fmt,
               "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "pipe:0",
               "-c:v", "ffv1", "-level", "3", "-pix_fmt", pix_fmt, path]
        self.procs[name] = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)

    def _start(self):
        c = self.cfg
        if c["RECORD_LEFT_MKV"]:
            self._ffmpeg("left", "bgr24", "RGB_video.mkv")       # legacy-compatible name
        if c["RECORD_RIGHT_MKV"]:
            self._ffmpeg("right", "bgr24", "RGB_right_video.mkv")
        if c["RECORD_DEPTH_MKV"]:
            self._ffmpeg("depth", "gray16le", "Depth_video.mkv")
        if c["RECORD_CONF_MKV"]:
            self._ffmpeg("conf", "gray", "Confidence_video.mkv")

        self.csv_file = open(os.path.join(self.dir, "frames.csv"), "w",
                             newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "video_frame_idx", "capture_frame_idx",
            "image_timestamp_ns", "host_monotonic_ns", "host_realtime_ns",
            "system_time_iso", "width", "height", "fps",
            "exposure", "gain", "wb_temp", "aec_agc",
            "pose_state", "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw",
            "ax", "ay", "az", "gx", "gy", "gz"])

    def run(self):
        self._start()
        clip = 65535
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                it = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if "left" in self.procs:
                    b = it["left"].tobytes()
                    self.procs["left"].stdin.write(b)
                    self.bytes_written += len(b)
                if "right" in self.procs and it["right"] is not None:
                    self.procs["right"].stdin.write(it["right"].tobytes())
                if "depth" in self.procs:
                    d = np.clip(it["depth"], 0, clip).astype(np.uint16)
                    self.procs["depth"].stdin.write(d.tobytes())
                if "conf" in self.procs and it["conf"] is not None:
                    # ZED confidence occupies 0-100. NaN (no measure computed) is
                    # stored as 255, a value OUTSIDE that range, so it can never
                    # be confused with a real confidence reading. Documented as a
                    # reserved sentinel in session.json -> confidence_encoding.
                    c = np.clip(np.nan_to_num(it["conf"], nan=255.0),
                                0, 255).astype(np.uint8)
                    self.procs["conf"].stdin.write(c.tobytes())
            except (BrokenPipeError, OSError) as e:
                self.broken_pipe = True
                print(f"[ERROR] ffmpeg pipe: {e}")
                self.stop_event.set()

            s, p, m = it["settings"], it.get("pose") or {}, it.get("imu") or {}
            self.csv_writer.writerow([
                self.frames_written, it["frame_idx"],
                it["ts_ns"], it["host_mono_ns"], it["host_real_ns"],
                datetime.datetime.now().isoformat(timespec="milliseconds"),
                self.w, self.h, self.fps,
                s.get("exposure", ""), s.get("gain", ""),
                s.get("wb_temp", ""), s.get("aec_agc", ""),
                p.get("pose_state", ""), p.get("tx", ""), p.get("ty", ""),
                p.get("tz", ""), p.get("qx", ""), p.get("qy", ""),
                p.get("qz", ""), p.get("qw", ""),
                m.get("ax", ""), m.get("ay", ""), m.get("az", ""),
                m.get("gx", ""), m.get("gy", ""), m.get("gz", "")])
            self.frames_written += 1
            self.q.task_done()
        self._close()

    def _close(self):
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        for name, proc in list(self.procs.items()):
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=20)
            except Exception as e:
                print(f"[WARN] closing {name}: {e}")
        self.procs.clear()


def write_dropped(save_dir, dropped):
    """An explicit record of every lost frame. Silence here was the old bug."""
    path = os.path.join(save_dir, "dropped_frames.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capture_frame_idx"])
        for d in dropped:
            w.writerow([d])
    return path


def write_session_json(save_dir, cfg, zed, calib, cap_thread, writer, w, h, fps,
                       svo_mode, started, ended):
    dropped = cap_thread.dropped if cap_thread else []
    meta = {
        "schema_version": "seeweed3d/capture/2.0",
        "created": datetime.datetime.now().isoformat(),
        "started": started, "ended": ended,
        "context": {k: cfg[k] for k in ("SITE", "FIELD", "PLOT", "ROW",
                                        "GROWTH_STAGE", "WEATHER", "OPERATOR",
                                        "MOUNT_HEIGHT_MM", "VEHICLE_SPEED_MPS",
                                        "NOTES")},
        "camera": {"model": "ZED 2i", "resolution": [w, h], "fps": fps,
                   "depth_mode": cfg["DEPTH_MODE"],
                   "depth_min_mm": cfg["DEPTH_MIN_MM"],
                   "depth_max_mm": cfg["DEPTH_MAX_MM"],
                   "view_saved": "VIEW.LEFT (rectified)"
                                 + (" + VIEW.RIGHT" if cfg["RECORD_RIGHT_MKV"] else "")},
        "exposure": {"locked": cfg["LOCK_EXPOSURE"],
                     "exposure_setting": cfg["EXPOSURE"],
                     "gain_setting": cfg["GAIN"],
                     "white_balance_locked": cfg["LOCK_WHITE_BALANCE"],
                     "wb_temperature": cfg["WHITEBALANCE_TEMPERATURE"],
                     "note": "Per-frame actual values are in frames.csv - trust "
                             "those over these requested values."},
        "streams": {"svo": bool(svo_mode), "svo_compression": svo_mode,
                    "left_mkv": cfg["RECORD_LEFT_MKV"],
                    "right_mkv": cfg["RECORD_RIGHT_MKV"],
                    "depth_mkv": cfg["RECORD_DEPTH_MKV"],
                    "confidence_mkv": cfg["RECORD_CONF_MKV"]},
        "depth_encoding": {
            "container": "FFV1 MKV, gray16le",
            "scale_mm_per_unit": 1.0, "invalid_value": 0,
            "note": "Raw millimetres. There is deliberately no preview constant "
                    "in this file: depth_vis_max_mm in the v1 format was a GUI "
                    "value that was repeatedly mistaken for a data scale."},
        "confidence_encoding": {
            "container": "FFV1 MKV, gray8", "range": [0, 100],
            "no_measure_sentinel": 255,
            "note": "Real confidence is 0-100. 255 is a reserved sentinel for "
                    "pixels with no confidence measure (NaN at capture); it is "
                    "out of the valid range and must be excluded before filtering.",
            "polarity": cap_thread.confidence_polarity() if cap_thread else {}},
        "clocks": {
            "image_timestamp_ns": "ZED IMAGE clock",
            "host_monotonic_ns": "time.monotonic_ns(), for intra-session deltas",
            "host_realtime_ns": "time.time_ns(), JOIN KEY for ROS 2 / Amiga logs",
            "note": "ZED positional tracking drifts over repetitive crop rows. "
                    "Record vehicle odometry separately and join on "
                    "host_realtime_ns."},
        "integrity": {
            "frames_written": writer.frames_written if writer else 0,
            "frames_dropped_from_mkv": len(dropped),
            "grab_failures": cap_thread.grab_failures if cap_thread else 0,
            "broken_pipe": writer.broken_pipe if writer else False,
            "note": "Dropped frames affect the MKV streams ONLY. The SVO is "
                    "written inside grab() and is complete."},
        "calibration": calib,
    }
    path = os.path.join(save_dir, "session.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def start_svo(zed, save_dir, cfg):
    """Try compression modes in order; report which one the SDK accepted."""
    if not cfg["RECORD_SVO"]:
        return None
    path = os.path.join(save_dir, "recording.svo2")
    for name in cfg["SVO_COMPRESSION_PREFERENCE"]:
        mode = enum_or_none(sl.SVO_COMPRESSION_MODE, name)
        if mode is None:
            continue
        try:
            rp = sl.RecordingParameters(path, mode)
            if zed.enable_recording(rp) == sl.ERROR_CODE.SUCCESS:
                print(f"[INFO] SVO recording: {name}")
                return name
        except Exception as e:
            print(f"[WARN] SVO {name} unavailable: {e}")
    print("[ERROR] SVO recording could not be started - "
          "the raw stereo archive will be MISSING for this session")
    return None


def apply_camera_settings(zed, cfg):
    """Lock exposure/WB so appearance is a property of the scene, not the sun."""
    applied = {}
    if cfg["LOCK_EXPOSURE"]:
        applied["aec_agc_off"] = sdk_set(zed, sl.VIDEO_SETTINGS.AEC_AGC, 0)
        applied["exposure"] = sdk_set(zed, sl.VIDEO_SETTINGS.EXPOSURE, cfg["EXPOSURE"])
        applied["gain"] = sdk_set(zed, sl.VIDEO_SETTINGS.GAIN, cfg["GAIN"])
    else:
        sdk_set(zed, sl.VIDEO_SETTINGS.AEC_AGC, 1)
    if cfg["LOCK_WHITE_BALANCE"]:
        wba = enum_or_none(sl.VIDEO_SETTINGS, "WHITEBALANCE_AUTO")
        if wba is not None:
            applied["wb_auto_off"] = sdk_set(zed, wba, 0)
        applied["wb_temp"] = sdk_set(zed, sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE,
                                     cfg["WHITEBALANCE_TEMPERATURE"])
    time.sleep(0.3)   # settings take a few frames to take effect
    return applied


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return -1.0


def make_session_dir(base, site, field):
    os.makedirs(base, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, f"Session_{stamp}")
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================
# GUI
# =============================================================================

def build_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk

    cfg = CONFIG

    class App:
        def __init__(self, root):
            self.root = root
            root.title("SeeWeed3D Capture v2")
            self.zed = sl.Camera()
            self.init = sl.InitParameters()
            self.runtime = sl.RuntimeParameters()
            self.opened = False
            self.dir = None
            self.svo_mode = None
            self.calib = None
            self.started_iso = None

            self.cap_stop = threading.Event()
            self.rec_flag = threading.Event()
            self.wr_stop = threading.Event()
            self.lock = threading.Lock()
            self.latest = {}
            self.q = queue.Queue(maxsize=cfg["FRAME_QUEUE_MAX"])
            self.cap = self.wr = None
            self.mats = {}

            bar = ttk.Frame(root, padding=8)
            bar.pack(fill="x")
            ttk.Label(bar, text="SeeWeed3D Capture v2",
                      font=("Segoe UI", 15, "bold")).pack(side="left")

            ctl = ttk.Frame(root, padding=8)
            ctl.pack(fill="x")
            self.res_var = tk.StringVar(value=cfg["RESOLUTION"])
            ttk.Combobox(ctl, textvariable=self.res_var, width=10,
                         values=list(RESOLUTIONS)).pack(side="left", padx=4)
            self.fps_var = tk.StringVar(value=str(cfg["FPS"]))
            ttk.Entry(ctl, textvariable=self.fps_var, width=5).pack(side="left", padx=4)
            self.open_b = ttk.Button(ctl, text="Open camera", command=self.open_cam)
            self.open_b.pack(side="left", padx=4)
            self.start_b = ttk.Button(ctl, text="Start", command=self.start,
                                      state="disabled")
            self.start_b.pack(side="left", padx=4)
            self.stop_b = ttk.Button(ctl, text="Stop", command=self.stop,
                                     state="disabled")
            self.stop_b.pack(side="left", padx=4)

            self.status = tk.StringVar(value="Idle")
            ttk.Label(root, textvariable=self.status).pack(fill="x", padx=8)
            self.canvas = ttk.Label(root)
            self.canvas.pack(padx=8, pady=8)
            self.messagebox = messagebox
            self._tk = (tk, Image, ImageTk)
            self.root.after(50, self.tick)

        def open_cam(self):
            res = self.res_var.get()
            try:
                fps = int(self.fps_var.get())
            except ValueError:
                fps = cfg["FPS"]
            self.init.camera_resolution = getattr(sl.RESOLUTION, res)
            self.init.camera_fps = fps
            self.init.coordinate_units = sl.UNIT.MILLIMETER
            dm = enum_or_none(sl.DEPTH_MODE, cfg["DEPTH_MODE"]) or sl.DEPTH_MODE.NEURAL
            self.init.depth_mode = dm
            self.init.depth_minimum_distance = cfg["DEPTH_MIN_MM"]
            self.init.depth_maximum_distance = cfg["DEPTH_MAX_MM"]

            if self.zed.open(self.init) != sl.ERROR_CODE.SUCCESS:
                self.messagebox.showerror("ZED", "Cannot open camera")
                return
            if cfg["ENABLE_POSITIONAL_TRACKING"]:
                try:
                    self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
                except Exception as e:
                    print(f"[WARN] positional tracking: {e}")

            applied = apply_camera_settings(self.zed, cfg)
            print(f"[INFO] camera settings applied: {applied}")

            for k in ("left", "right", "depth", "conf"):
                self.mats[k] = sl.Mat()

            info = self.zed.get_camera_information().camera_configuration
            w, h = info.resolution.width, info.resolution.height
            bw = estimate_bandwidth(cfg, w, h, info.fps)
            print(f"[INFO] {w}x{h}@{info.fps} | expected MKV write "
                  f"~{bw['expected_mkv_mb_s']} MB/s | {bw['svo_note']}")

            self.opened = True
            self.open_b.config(state="disabled")
            self.start_b.config(state="normal")
            self.cap_stop.clear()
            self.rec_flag.clear()
            self.cap = CaptureThread(self.zed, self.runtime, self.mats, self.q,
                                     self.lock, self.latest, self.cap_stop,
                                     self.rec_flag, cfg)
            self.cap.start()
            self.status.set(f"Camera open {w}x{h}@{info.fps}")

        def start(self):
            if not self.opened:
                return
            if free_gb(cfg["BASE_DIRECTORY"]) < cfg["MIN_FREE_SPACE_GB"]:
                self.messagebox.showerror("Disk", "Not enough free space")
                return
            self.dir = make_session_dir(cfg["BASE_DIRECTORY"], cfg["SITE"], cfg["FIELD"])
            self.calib = dump_calibration(self.zed,
                                          os.path.join(self.dir, "calibration.json"))
            self.svo_mode = start_svo(self.zed, self.dir, cfg)
            self.started_iso = datetime.datetime.now().isoformat()

            info = self.zed.get_camera_information().camera_configuration
            self.wr_stop.clear()
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.wr = WriterThread(self.q, self.wr_stop, self.dir,
                                   info.resolution.width, info.resolution.height,
                                   info.fps, cfg)
            self.wr.start()
            self.rec_flag.set()
            self.start_b.config(state="disabled")
            self.stop_b.config(state="normal")

        def stop(self):
            if not self.rec_flag.is_set():
                return
            self.rec_flag.clear()
            self.q.join()
            self.wr_stop.set()
            if self.wr:
                self.wr.join(timeout=30)
            try:
                self.zed.disable_recording()
            except Exception:
                pass
            info = self.zed.get_camera_information().camera_configuration
            write_dropped(self.dir, self.cap.dropped)
            meta = write_session_json(self.dir, cfg, self.zed, self.calib, self.cap,
                                      self.wr, info.resolution.width,
                                      info.resolution.height, info.fps,
                                      self.svo_mode, self.started_iso,
                                      datetime.datetime.now().isoformat())
            n_drop = len(self.cap.dropped)
            self.cap.dropped = []
            self.start_b.config(state="normal")
            self.stop_b.config(state="disabled")
            msg = (f"Saved {meta['integrity']['frames_written']} frames "
                   f"({n_drop} dropped from MKV) -> {self.dir}")
            self.status.set(msg)
            print("[INFO] " + msg)
            if n_drop:
                print("[WARN] MKV streams have gaps; the SVO archive is complete. "
                      "Reduce enabled streams or use a faster disk.")

        def tick(self):
            tk, Image, ImageTk = self._tk
            with self.lock:
                rgb = self.latest.get("rgb_bgr")
                d = self.latest.get("depth_mm")
                s = self.latest.get("settings", {})
            if rgb is not None and d is not None:
                # Preview scaling only. Never a data transform.
                vis = np.clip(d, 0, 3000) / 3000.0 * 255.0
                vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                comb = np.hstack((rgb, vis))
                comb = cv2.resize(comb, None, fx=0.33, fy=0.33)
                img = ImageTk.PhotoImage(Image.fromarray(
                    cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)))
                self.canvas.configure(image=img)
                self.canvas.image = img
            if self.rec_flag.is_set() and self.wr:
                self.status.set(
                    f"REC {self.wr.frames_written} frames | dropped "
                    f"{len(self.cap.dropped)} | queue {self.q.qsize()}/"
                    f"{cfg['FRAME_QUEUE_MAX']} | exp {s.get('exposure')} "
                    f"gain {s.get('gain')} | free {free_gb(cfg['BASE_DIRECTORY']):.0f} GB")
                if cfg["ABORT_ON_LOW_DISK"] and free_gb(cfg["BASE_DIRECTORY"]) < 2:
                    print("[ERROR] disk nearly full - stopping cleanly")
                    self.stop()
            self.root.after(50, self.tick)

    root = tk.Tk()
    app = App(root)

    def closing():
        try:
            if app.rec_flag.is_set():
                app.stop()
            app.cap_stop.set()
            if app.opened:
                app.zed.close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", closing)
    root.mainloop()


def bench_check():
    """Hardware self-check: open, verify every stream and setting, then exit."""
    cfg = CONFIG
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, cfg["RESOLUTION"])
    init.camera_fps = cfg["FPS"]
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_mode = enum_or_none(sl.DEPTH_MODE, cfg["DEPTH_MODE"]) or sl.DEPTH_MODE.NEURAL
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("FAIL: camera did not open")
        return
    print("OK  camera opened")
    print("OK  settings:", apply_camera_settings(zed, cfg))
    for name in cfg["SVO_COMPRESSION_PREFERENCE"]:
        print(f"    SVO mode {name}: "
              f"{'available' if enum_or_none(sl.SVO_COMPRESSION_MODE, name) else 'MISSING'}")
    mats = {k: sl.Mat() for k in ("left", "right", "depth", "conf")}
    if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(mats["left"], sl.VIEW.LEFT)
        zed.retrieve_image(mats["right"], sl.VIEW.RIGHT)
        zed.retrieve_measure(mats["depth"], sl.MEASURE.DEPTH)
        zed.retrieve_measure(mats["conf"], sl.MEASURE.CONFIDENCE)
        d = mats["depth"].get_data()
        c = mats["conf"].get_data()
        print(f"OK  left {mats['left'].get_data().shape} "
              f"right {mats['right'].get_data().shape}")
        print(f"OK  depth valid fraction {np.mean(np.isfinite(d) & (d > 0)):.1%}")
        print(f"OK  confidence range {np.nanmin(c):.1f}-{np.nanmax(c):.1f}")
    else:
        print("FAIL: grab() failed")
    zed.close()


if __name__ == "__main__":
    if CONFIG["BENCH_CHECK"]:
        bench_check()
    else:
        build_gui()
