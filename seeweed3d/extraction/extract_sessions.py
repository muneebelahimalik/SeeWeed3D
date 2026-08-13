#!/usr/bin/env python3
"""
SeeWeed3D - Stage 1: Session extraction and indexing  (v2)
===========================================================
Decodes ZED recordings into an annotation-ready, traceable dataset.

Handles BOTH capture formats transparently:

  v1 (legacy, zed_app_updated_2026_jan.py)
      RGB_video.mkv + Depth_video.mkv + calibration_params.txt
      [+ session_meta.txt] [+ frames.csv with column `frame_idx`]

  v2 (capture/zed_capture.py)
      + RGB_right_video.mkv, Confidence_video.mkv, recording.svo2
      + calibration.json, session.json, dropped_frames.csv
      + frames.csv with `video_frame_idx` / `capture_frame_idx`, exposure,
        gain, white balance, pose and IMU per frame

Everything a format offers is used; nothing missing is fatal. Sessions are
tagged `capture_format` so downstream code knows what exists.

CORE INVARIANTS
---------------
* Frames are selected by FRAME INDEX, never by `-vf fps=`. Capture drops frames
  when the writer queue fills while the encoder is told the stream is constant
  fps, so video time does not track real time. All streams are written from the
  same queue item and are aligned by index.
* Depth: 1 PNG count = 1 mm, 0 = invalid. `depth_vis_max_mm` in v1
  session_meta.txt is a GUI preview constant and must never be applied to data.
* Confidence (v2): stored raw. Polarity is recorded per session from the
  capture-time probe - do not assume it.
"""

import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.progress import Progress  # noqa: E402

# #############################################################################
# ##                                                                         ##
# ##   1)  INPUT DIRECTORIES  -  where your recorded sessions live           ##
# ##   2)  OUTPUT DIRECTORY   -  where the extracted dataset is written      ##
# ##                                                                         ##
# ##   These are the ONLY two things you usually need to change. Everything  ##
# ##   below them has working defaults. ffmpeg is found on PATH automatically##
# ##                                                                         ##
# #############################################################################

# ---------------------------------------------------------------------------
# 1) INPUT DIRECTORIES
# ---------------------------------------------------------------------------
# Add one block per field visit / camera / trip. Each `path` is a folder that
# CONTAINS the Session_* recording folders; it is searched RECURSIVELY, so a
# single path covering many sessions is fine. Add as many blocks as you like -
# v1 (March 2025) and v2 recordings can be mixed freely; the format of each
# session is detected automatically.
#
#   path        : folder holding the Session_* folders (searched recursively).
#                 Use a raw string on Windows: r"C:\ZED\DataCollection\Vidalia2"
#                 A plain POSIX path also works:   "/data/zed/vidalia2"
#   trip        : short, STABLE code. Becomes the session_id prefix, so keep it
#                 unique per visit and never rename it once annotation starts.
#   site/field  : provenance, copied into every session's metadata.
#   scene_hint  : mixed | onion_only | weed_only | unknown  (best guess is fine)
#   notes       : free text, stored in the registry for your own reference.

INPUT_ROOTS = [
    {
        "path":       r"E:\Research (Muneeb)\Datasets\Vidalia\2026_Vidalia_Visit_1\Vidalia3_4_just_onions_jan_2026_transplanted",
        "trip":       "Visit1",
        "site":       "vidalia_1",
        "field":      "field_A",
        "scene_hint": "onions",        # mixed | onion_only | weed_only | unknown
        "notes":      "Vidalia Visit One - onions only",
    },
    # Add more visits by uncommenting and editing:
    # {
    #     "path":       r"M:\Research\Data\Vidalia_onions_visit3",
    #     "trip":       "vid3", "site": "vidalia", "field": "field_A",
    #     "scene_hint": "mixed",
    #     "notes":      "v2 capture - SVO + confidence + locked exposure",
    # },
]

# ---------------------------------------------------------------------------
# 2) OUTPUT DIRECTORY  -  the indexed, QC'd dataset is written here
# ---------------------------------------------------------------------------
OUTPUT_ROOT = r"E:\Dataset_Vidalia\onions_20260108_1"

# =============================================================================
# Everything below is advanced tuning. The defaults are sensible.
# =============================================================================

CONFIG = {
    "INPUT_ROOTS": INPUT_ROOTS,
    "OUTPUT_ROOT": OUTPUT_ROOT,

    # -- ffmpeg / ffprobe ------------------------------------------------------
    # Left as bare names, they are found on PATH automatically (Linux/macOS, or
    # Windows once ffmpeg is on PATH). Override with a full path only if ffmpeg
    # is not on PATH, e.g. r"C:\ffmpeg\bin\ffmpeg.exe".
    "FFMPEG":  "ffmpeg",
    "FFPROBE": "ffprobe",

    # -- File discovery (case-insensitive). Covers every visit variant. --------
    "RGB_PATTERNS":   ["rgb_video.mkv", "rgb.mkv", "left.mkv", "*left*.mkv"],
    "RIGHT_PATTERNS": ["rgb_right_video.mkv", "right.mkv", "*right*.mkv"],
    "DEPTH_PATTERNS": ["depth_video.mkv", "depth.mkv", "*depth*.mkv"],
    "CONF_PATTERNS":  ["confidence_video.mkv", "confidence.mkv", "*conf*.mkv"],

    # -- What to write out (ignored when the session has no such stream) ------
    "WRITE_RIGHT": True,
    "WRITE_CONF":  True,

    # -- Sampling ---------------------------------------------------------------
    # POOL_STRIDE=None derives the stride from TARGET_POOL_FPS. The pool is
    # deliberately generous; stage 2 does the real selection. The MKV/SVO files
    # stay the archive, so the pool is always regenerable.
    "POOL_STRIDE":     None,
    "TARGET_POOL_FPS": 3.0,
    "MAX_POOL_PER_SESSION": 400,   # None = unlimited

    # -- Depth semantics (verified against capture source) ----------------------
    "DEPTH_SCALE_MM_PER_UNIT": 1.0,
    "DEPTH_INVALID_VALUE":     0,
    "DEPTH_MIN_VALID_MM":      300,
    "DEPTH_MAX_VALID_MM":      2000,

    "METRIC_WIDTH": 512,
    # PNG is lossless at every level - this trades file size against write
    # speed and nothing else. It cannot cost image quality.
    "PNG_COMPRESSION": 3,

    # -- Decode fidelity --------------------------------------------------------
    # swscale flags for the YUV->RGB conversion. Only bites on the LOSSY v1
    # captures (AVI/mpeg4/yuv420p); an FFV1 stream is bgr24 or gray16le already
    # and decodes bit-exact either way, which is asserted by a test.
    #
    # `accurate_rnd` makes swscale ROUND rather than truncate. Truncation
    # leaves a systematic negative bias - measured at a uniform -2 per channel
    # on flat patches, dropping to about -0.8 with this flag. That bias is not
    # cosmetic here: the vegetation prior thresholds ExG and compares g against
    # b, so a channel-dependent error moves real decisions. Measured on a
    # synthetic field frame, it cut vegetation_mask() disagreement against the
    # uncompressed original from 0.607% of pixels to 0.474%, and mean absolute
    # pixel error from 3.30 to 2.78, at no measurable decode cost.
    #
    # NOT `full_chroma_int`: measured a no-op here, because it governs chroma
    # interpolation during RESIZING and extraction never resizes.
    #
    # Deliberately no colorspace or range override. v1 AVIs carry no colour
    # metadata at all, so both ends fall back to defaults - and those defaults
    # already agree. Measured on flat patches, the default (BT.601, limited
    # range in) leaves a colour-neutral offset, while forcing BT.709 swings
    # green by -21 and red by +7, and forcing full-range input triples the
    # error. Both were tried; both were worse. Leave this alone unless a future
    # capture writes tagged colour metadata.
    "SWS_FLAGS": "accurate_rnd",

    "OVERWRITE": False,
    "DRY_RUN": False,
}

# =============================================================================

SESSION_RE = re.compile(r"(\d{8})[_\-]?(\d{6})")


def run_json(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


def probe(ffprobe, path):
    """Real stream parameters from the file header - never trust sidecar files."""
    d = run_json([ffprobe, "-v", "error", "-select_streams", "v:0",
                  "-show_streams", "-show_format", "-of", "json", str(path)])
    s = d["streams"][0]
    num, den = (s.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    nb = s.get("nb_frames")
    if nb in (None, "N/A"):
        nb = None
    return {"width": int(s["width"]), "height": int(s["height"]),
            "pix_fmt": s.get("pix_fmt"), "codec": s.get("codec_name"),
            "nominal_fps": fps, "nb_frames_header": int(nb) if nb else None,
            "duration_s": float(d["format"].get("duration", 0) or 0),
            "size_bytes": int(d["format"].get("size", 0) or 0)}


def _glob_match(name, pat):
    import fnmatch
    return fnmatch.fnmatch(name, pat)


def find_one(folder, patterns, want_right=False):
    """Case-insensitive first match in priority order. Files with 'right' in the
    name are only matched by right-hand patterns, so RGB_right_video.mkv can
    never be mistaken for the left stream."""
    try:
        files = [f for f in folder.iterdir() if f.is_file()]
    except (PermissionError, OSError):
        return None
    for pat in patterns:
        pat_l = pat.lower()
        for f in files:
            n = f.name.lower()
            if ("right" in n) != want_right:
                continue
            if ("*" in pat_l and _glob_match(n, pat_l)) or n == pat_l:
                return f
    return None


def _derive_calib(out):
    L = out.get("left") or {}
    fx = L.get("fx")
    if fx:
        out["mm_per_px_at_1000mm"] = round(1000.0 / fx, 4)
        b = out.get("baseline_mm")
        if b:
            # dz = z^2 / (f*B) * d_disp, evaluated at 1 m for 0.1 px of disparity
            out["depth_err_mm_at_1000mm_per_0p1px"] = round(
                (1000.0 ** 2) / (fx * b) * 0.1, 4)
    return out


def parse_calibration_txt(path):
    """v1 calibration_params.txt. RECTIFIED parameters (zero distortion) that
    apply directly to the saved VIEW.LEFT images. Tx is the baseline in mm."""
    if path is None or not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    out = {"source": path.name, "format": "v1_txt"}

    def grab(section):
        m = re.search(section + r".*?fx:\s*([\d.\-]+),\s*fy:\s*([\d.\-]+).*?"
                      r"cx:\s*([\d.\-]+),\s*cy:\s*([\d.\-]+)", txt, re.S)
        if not m:
            return None
        return {"fx": float(m.group(1)), "fy": float(m.group(2)),
                "cx": float(m.group(3)), "cy": float(m.group(4)),
                "distortion": [0.0] * 5, "model": "rectified_pinhole"}

    out["left"] = grab("Left Camera Intrinsic")
    out["right"] = grab("Right Camera Intrinsic")
    m = re.search(r"Tx:\s*([\d.\-]+).*?Ty:\s*([\d.\-]+).*?Tz:\s*([\d.\-]+)", txt, re.S)
    if m:
        out["stereo_translation_mm"] = [float(m.group(i)) for i in (1, 2, 3)]
        out["baseline_mm"] = abs(float(m.group(1)))
    m = re.search(r"Horizontal FOV \(Left Cam\):\s*([\d.]+)", txt)
    if m:
        out["hfov_deg"] = float(m.group(1))
    return _derive_calib(out)


def parse_calibration_json(path):
    """v2 calibration.json - raw AND rectified, both cameras, stereo rotation."""
    if path is None or not path.exists():
        return None
    j = json.loads(path.read_text())
    rect = j.get("rectified", {})
    out = {"source": path.name, "format": "v2_json",
           "left": rect.get("left"), "right": rect.get("right"),
           "baseline_mm": rect.get("baseline_mm"),
           "stereo_translation_mm": rect.get("translation_mm"),
           "rotation_vector_rad": rect.get("rotation_vector_rad"),
           "raw": j.get("raw"), "serial_number": j.get("serial_number"),
           "firmware": j.get("firmware")}
    if out["left"] and out["left"].get("h_fov"):
        out["hfov_deg"] = out["left"]["h_fov"]
    return _derive_calib(out)


def parse_session_meta(txt_path, json_path):
    """v2 session.json wins; v1 session_meta.txt is the fallback."""
    if json_path and json_path.exists():
        return json.loads(json_path.read_text()), "v2"
    if txt_path and txt_path.exists():
        meta = {}
        for line in txt_path.read_text(errors="ignore").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, "v1"
    return None, None


def read_frames_csv(path):
    """Rows are one per ENCODED frame in write order, so video frame i <-> row i.
    Recovers true capture indices and timestamps despite dropped frames.
    Accepts both the v1 and v2 column schemas."""
    if path is None or not path.exists():
        return None, None
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rdr = csv.DictReader(f)
        cols = set(rdr.fieldnames or [])
        ver = "v2" if "capture_frame_idx" in cols else "v1"
        for r in rdr:
            try:
                if ver == "v2":
                    rows.append({
                        "capture_frame_idx": int(r["capture_frame_idx"]),
                        "timestamp_ns": int(r["image_timestamp_ns"]),
                        "host_realtime_ns": r.get("host_realtime_ns", ""),
                        "exposure": r.get("exposure", ""),
                        "gain": r.get("gain", ""),
                        "wb_temp": r.get("wb_temp", ""),
                        "pose_state": r.get("pose_state", ""),
                        "tx_mm": r.get("tx_mm", ""),
                        "ty_mm": r.get("ty_mm", ""),
                        "tz_mm": r.get("tz_mm", "")})
                else:
                    rows.append({"capture_frame_idx": int(r["frame_idx"]),
                                 "timestamp_ns": int(r["timestamp_ns"])})
            except (KeyError, ValueError):
                return None, None
    return (rows, ver) if rows else (None, None)


class RawDecoder:
    """Decode FFV1/MKV to raw frames. -vsync 0 guarantees one output frame per
    coded frame (no duplication or dropping), which keeps indices honest.

    sws_flags reaches the YUV->RGB conversion, so it matters only for the lossy
    v1 captures; a bgr24 or gray16le FFV1 stream needs no conversion and comes
    back bit-exact with or without it. See CONFIG["SWS_FLAGS"] for what is set
    and, just as importantly, what was measured and deliberately not set."""

    def __init__(self, ffmpeg, path, pix_fmt, w, h, channels, dtype,
                 sws_flags=None):
        self.shape = (h, w, channels) if channels > 1 else (h, w)
        self.dtype = dtype
        self.bpf = w * h * channels * np.dtype(dtype).itemsize
        cmd = [ffmpeg, "-loglevel", "error", "-i", str(path)]
        if sws_flags:
            cmd += ["-sws_flags", str(sws_flags)]
        cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-vsync", "0", "pipe:1"]
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self.bpf * 2)

    def __iter__(self):
        while True:
            buf = self.proc.stdout.read(self.bpf)
            if len(buf) < self.bpf:
                break
            yield np.frombuffer(buf, dtype=self.dtype).reshape(self.shape)
        self.close()

    def close(self):
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def phash64(gray_small):
    img = cv2.resize(gray_small, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(img)[:8, :8]
    med = np.median(d.flatten()[1:])
    return "".join(f"{b:02x}" for b in np.packbits((d.flatten() > med).astype(np.uint8)))


def frame_metrics(bgr, depth, conf, cfg):
    """Per-frame QC, computed on a downscaled copy."""
    h, w = bgr.shape[:2]
    scale = cfg["METRIC_WIDTH"] / float(w)
    small = cv2.resize(bgr, (cfg["METRIC_WIDTH"], max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    b, g, r = [small[:, :, i].astype(np.float32) for i in range(3)]
    s = b + g + r + 1e-6
    exg = 2 * (g / s) - (r / s) - (b / s)      # excess green index
    veg = exg > 0.05

    m = {"sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
         "mean_luma": float(gray.mean()),
         "clip_frac": float((gray >= 250).mean()),
         "dark_frac": float((gray <= 12).mean()),
         "veg_frac": float(veg.mean()),
         "phash": phash64(gray)}

    if depth is None:
        m.update({k: "" for k in ("depth_valid_frac", "depth_valid_frac_veg",
                                  "depth_median_mm", "depth_inrange_frac")})
    else:
        ds = cv2.resize(depth, (small.shape[1], small.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        valid = ds > cfg["DEPTH_INVALID_VALUE"]
        inr = valid & (ds >= cfg["DEPTH_MIN_VALID_MM"]) & (ds <= cfg["DEPTH_MAX_VALID_MM"])
        m["depth_valid_frac"] = round(float(valid.mean()), 4)
        m["depth_inrange_frac"] = round(float(inr.mean()), 4)
        # The metric that matters for LEP work: is there depth ON the plants?
        m["depth_valid_frac_veg"] = round(float(valid[veg].mean()), 4) if veg.any() else 0.0
        m["depth_median_mm"] = int(np.median(ds[valid])) if valid.any() else ""

    if conf is None:
        m.update({"conf_mean": "", "conf_mean_veg": ""})
    else:
        cs = cv2.resize(conf, (small.shape[1], small.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # 255 is the capture-time "no measure" sentinel (real confidence is
        # 0-100); exclude it so the mean reflects genuine confidence only.
        real = cs <= 100
        vegr = veg & real
        m["conf_mean"] = round(float(cs[real].mean()), 2) if real.any() else ""
        m["conf_mean_veg"] = round(float(cs[vegr].mean()), 2) if vegr.any() else ""
    return m


<<<<<<< Updated upstream
#: Pixel formats that genuinely carry 16 bits per sample. Anything else cannot
#: hold a millimetre range, whatever the file is named.
SIXTEEN_BIT_HINTS = ("16le", "16be", "16", "p010", "p016", "yuv420p10",
                     "gray10", "gray12", "gray14")


def depth_precision_problem(info):
    """Why this depth stream cannot be trusted as millimetres, or None.

    The decoder asks ffmpeg for gray16le, and ffmpeg will produce gray16le from
    ANY input - including an 8-bit lossy one, by scaling up values it invented.
    The result is a valid PNG full of plausible numbers that are fiction, and
    nothing downstream can tell: the QC fractions compute, the depth range
    looks sane, and the LEP's canopy-height channel reads the noise as terrain.

    So the check happens here, where the pixel format is still visible, rather
    than being inferred from the container. AVI's usual codecs (MJPEG, XVID,
    DIVX) have no 16-bit grayscale format at all - but FFV1 in AVI is fine, and
    a badly-remuxed MKV is not, so the container itself decides nothing.
    """
    if not info:
        return None
    pf = (info.get("pix_fmt") or "").lower()
    codec = (info.get("codec") or "?").lower()
    if not pf:
        return (f"depth codec {codec} reports no pixel format, so 16-bit "
                f"millimetres cannot be confirmed")
    if any(h in pf for h in SIXTEEN_BIT_HINTS):
        return None
    return (f"depth is {codec}/{pf}, which is not 16-bit - decoding it as "
            f"millimetres would invent values that look plausible and are not")


def preview_to_mm(bgr, kind, vis_max_mm, colormap=None, tv_range=False):
    """One frame of an 8-bit depth PREVIEW as approximate millimetres, uint16.

    The inverse of what the capture GUI did: undo the colormap if there was
    one, undo TV range if the encoder used it, then scale 0..255 back onto
    0..vis_max_mm.

    0 stays 0 - the invalid sentinel - but only where it actually survived the
    encode. Lossy compression smears it, so the caller is told separately
    whether it is still there rather than being allowed to assume it.

    The result is deliberately uint16 millimetres so it reads with the same
    tooling as real depth, and deliberately written to a different directory so
    nothing can mistake it for real depth.
    """
    if kind == "colormapped_8bit" and colormap:
        from validation.inspect_depth_video import match_colormap
        _, _, idx = match_colormap(bgr, names=(colormap,))
        code = idx.astype(np.float32)
    else:
        code = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    invalid = code <= 0.5
    if tv_range:
        # 16..235 is the whole scale; leaving it stretches every distance by
        # ~14% and pins the near end at a value that is not zero.
        code = (code - 16.0) * (255.0 / (235.0 - 16.0))
    code = np.clip(code, 0.0, 255.0)

    mm = code * (float(vis_max_mm) / 255.0)
    mm[invalid] = 0.0
    return np.clip(mm, 0, 65535).astype(np.uint16)


class PreviewDepthDecoder:
    """Decodes an 8-bit depth preview and yields approximate uint16 millimetres.

    Wraps RawDecoder so it drops into the same per-frame loop as a real depth
    stream, which keeps frame alignment handled in exactly one place."""

    def __init__(self, ffmpeg, path, w, h, kind, vis_max_mm, colormap=None,
                 tv_range=False, sws_flags=None):
        self._dec = RawDecoder(ffmpeg, path, "bgr24", w, h, 3, np.uint8,
                               sws_flags=sws_flags)
        self._args = (kind, vis_max_mm, colormap, tv_range)

    def __iter__(self):
        for bgr in self._dec:
            yield preview_to_mm(bgr, *self._args)

    def close(self):
        self._dec.close()


def inspect_preview(cfg, path, w, h):
    """Classify an 8-bit depth file. Returns the verdict dict, or None if the
    inspection itself could not run - never a guess."""
    try:
        from validation.inspect_depth_video import classify, read_frames
        frames = read_frames(cfg["FFMPEG"], path, 3, w, h)
        return classify({"codec": "", "pix_fmt": ""}, frames)
    except Exception as e:                       # a probe must never kill a run
        print(f"      [!] could not inspect {Path(path).name}: {e}")
        return None


def _plan_preview_recovery(sess, cfg, sid, w, h, warnings):
    """Decide whether this 8-bit depth file can be recovered, and on what
    terms. Returns the plan dict, or None to leave the session without depth.

    Everything it finds is recorded and printed, because an approximation is
    only safe to use when its terms travel with it."""
    v = inspect_preview(cfg, sess["depth"], w, h)
    if not v or not v.get("recoverable"):
        why = (v or {}).get("detail", "inspection failed")
        warnings.append(f"8-bit depth not recoverable: {why}")
        print(f"      not recoverable: {why}")
        return None

    luma = v.get("luma", {})
    plan = {"kind": v["verdict"], "colormap": v.get("colormap"),
            "tv_range": bool(luma.get("looks_tv_range")),
            "vis_max_mm": float(cfg["DEPTH_VIS_MAX_MM"]),
            "sentinel_survived": bool(luma.get("zero_frac", 0) >= 0.001),
            "quantisation_mm": round(float(cfg["DEPTH_VIS_MAX_MM"]) / 255.0, 2)}
    print(f"      recovering as APPROXIMATE depth -> depth_approx/ "
          f"({plan['kind']}"
          f"{', ' + plan['colormap'] if plan['colormap'] else ''}"
          f"{', TV range' if plan['tv_range'] else ''}) | "
          f"{plan['quantisation_mm']} mm per level at "
          f"{plan['vis_max_mm']:.0f} mm")
    warnings.append(
        f"depth is APPROXIMATE: recovered from an 8-bit preview at "
        f"{plan['quantisation_mm']} mm/level. Not metric - see depth_approx/")
    if not plan["sentinel_survived"]:
        warnings.append("the 0 = invalid sentinel did not survive the encode, "
                        "so invalid regions read as near distances")
        print("      [!] invalid sentinel did not survive - invalid regions "
              "are indistinguishable from near distances")
    return plan


=======
>>>>>>> Stashed changes
def make_session_id(folder, trip):
    m = SESSION_RE.search(folder.name)
    if m:
        return f"{trip}_{m.group(1)}_{m.group(2)}"
    ts = datetime.fromtimestamp(folder.stat().st_mtime, tz=timezone.utc)
    return f"{trip}_{ts:%Y%m%d_%H%M%S}"


def discover(cfg):
    """A folder is a session if it directly contains a left/RGB mkv."""
    found = []
    for rc in cfg["INPUT_ROOTS"]:
        root = Path(rc["path"])
        if not root.exists():
            print(f"  [SKIP] input root does not exist: {root}")
            continue
        for folder in [root] + [p for p in root.rglob("*") if p.is_dir()]:
            rgb = find_one(folder, cfg["RGB_PATTERNS"])
            if rgb is None:
                # Flag folders that look like a recording but have no decodable
                # left/RGB video (e.g. SVO-only), so they are not silently lost.
                has_depth = find_one(folder, cfg["DEPTH_PATTERNS"]) is not None
                has_svo = next((p for p in folder.glob("*.svo*")), None) is not None
                if has_depth or has_svo:
                    why = "SVO only - decode it to MKV first" if has_svo else \
                          "depth present but no RGB/left MKV"
                    print(f"  [SKIP] {folder} : no RGB video ({why})")
                continue

            def opt(name):
                p = folder / name
                return p if p.exists() else None

            found.append({
                "folder": folder, "rgb": rgb,
                "right": find_one(folder, cfg["RIGHT_PATTERNS"], want_right=True),
                "depth": find_one(folder, cfg["DEPTH_PATTERNS"]),
                "conf": find_one(folder, cfg["CONF_PATTERNS"]),
                "calib_txt": opt("calibration_params.txt"),
                "calib_json": opt("calibration.json"),
                "meta_txt": opt("session_meta.txt"),
                "meta_json": opt("session.json"),
                "frames_csv": opt("frames.csv"),
                "dropped_csv": opt("dropped_frames.csv"),
                "svo": next((p for p in folder.glob("*.svo*")), None),
                "cfg": rc})
    return found


def extract_session(sess, out_root, cfg):
    rc = sess["cfg"]
    sid = make_session_id(sess["folder"], rc["trip"])
    sdir = out_root / "sessions" / sid
    warnings = []

    if sdir.exists() and not cfg["OVERWRITE"]:
        print(f"  [SKIP] {sid} already extracted (OVERWRITE=False)")
        return None

    fmt = "v2" if (sess["meta_json"] or sess["conf"] or sess["calib_json"]) else "v1"

    rgb_info = probe(cfg["FFPROBE"], sess["rgb"])
    W, H = rgb_info["width"], rgb_info["height"]
    infos = {"rgb": rgb_info}
    for key in ("right", "depth", "conf"):
        if sess[key]:
            infos[key] = probe(cfg["FFPROBE"], sess[key])
            if (infos[key]["width"], infos[key]["height"]) != (W, H):
                warnings.append(f"{key} resolution differs from RGB - stream dropped")
                sess[key] = None

    if sess["depth"] is None:
        warnings.append("no depth video - RGB only session")
    if fmt == "v1":
        warnings.append("v1 capture format: no confidence map, no right image, no "
                        "SVO, no exposure/pose log - see docs/capture_changelog.md")

    calib = (parse_calibration_json(sess["calib_json"])
             or parse_calibration_txt(sess["calib_txt"]))
    if calib is None:
        warnings.append("NO calibration - session unusable for 3D work")

    meta, meta_ver = parse_session_meta(sess["meta_txt"], sess["meta_json"])
    if meta is None:
        warnings.append("no session metadata - parameters taken from video header")

    fcsv, csv_ver = read_frames_csv(sess["frames_csv"])
    ts_source = f"frames_csv_{csv_ver}" if fcsv else "nominal_fps"
    if not fcsv:
        warnings.append("no frames.csv - timestamps synthesised from nominal fps; "
                        "dropped frames make these approximate, not for pose sync")

    n_dropped = 0
    if sess["dropped_csv"]:
        n_dropped = max(0, sum(1 for _ in open(sess["dropped_csv"])) - 1)
        if n_dropped:
            warnings.append(f"{n_dropped} frames dropped from MKV at capture "
                            f"(the SVO archive is complete)")

    fps = rgb_info["nominal_fps"] or 15.0
    stride = cfg["POOL_STRIDE"] or max(1, int(round(fps / cfg["TARGET_POOL_FPS"])))
    have = [k for k in ("right", "depth", "conf") if sess[k]]

    print(f"  [{sid}] {fmt} | {W}x{H} @ {fps:.2f}fps | stride={stride} | "
          f"ts={ts_source} | left+{'+'.join(have) if have else 'none'}"
          f"{' | SVO' if sess['svo'] else ''}")
    if cfg["DRY_RUN"]:
        return {"session_id": sid, "dry_run": True, "warnings": warnings}

    want_right = bool(sess["right"]) and cfg["WRITE_RIGHT"]
    want_conf = bool(sess["conf"]) and cfg["WRITE_CONF"]
    subs = ["rgb", "meta"] + (["depth"] if sess["depth"] else []) \
        + (["right"] if want_right else []) + (["conf"] if want_conf else [])
    for sub in subs:
        (sdir / sub).mkdir(parents=True, exist_ok=True)

    F = cfg["FFMPEG"]
    SWS = cfg.get("SWS_FLAGS")
    decs = {"rgb": RawDecoder(F, sess["rgb"], "bgr24", W, H, 3, np.uint8,
                              sws_flags=SWS)}
    if sess["depth"]:
<<<<<<< Updated upstream
        decs["depth"] = RawDecoder(F, sess["depth"], "gray16le", W, H, 1,
                                   np.uint16, sws_flags=SWS)
    elif preview:
        # Kept under its own key so nothing here can route it into depth/, and
        # so frame_metrics is never handed it - the QC columns mean METRIC
        # depth, and mixing an approximation into them would make the registry
        # incomparable across sessions.
        decs["depth_approx"] = PreviewDepthDecoder(
            F, preview["path"], W, H, preview["kind"], preview["vis_max_mm"],
            preview["colormap"], preview["tv_range"], sws_flags=SWS)
=======
        decs["depth"] = RawDecoder(F, sess["depth"], "gray16le", W, H, 1, np.uint16)
>>>>>>> Stashed changes
    if sess["right"]:
        decs["right"] = RawDecoder(F, sess["right"], "bgr24", W, H, 3,
                                   np.uint8, sws_flags=SWS)
    if sess["conf"]:
        decs["conf"] = RawDecoder(F, sess["conf"], "gray", W, H, 1, np.uint8,
                                  sws_flags=SWS)
    its = {k: iter(v) for k, v in decs.items() if k != "rgb"}

    index_rows, pool_rows = [], []
    n_written = 0
    png_opt = [cv2.IMWRITE_PNG_COMPRESSION, cfg["PNG_COMPRESSION"]]
    cap = cfg["MAX_POOL_PER_SESSION"]

    est_total = rgb_info.get("nb_frames_header") or (
        int(rgb_info["duration_s"] * fps) if rgb_info.get("duration_s") else 0)
    prog = Progress(est_total, f"  [{sid}]", unit="frames")
    for i, bgr in enumerate(decs["rgb"]):
        got = {k: next(v, None) for k, v in its.items()}
        met = frame_metrics(bgr, got.get("depth"), got.get("conf"), cfg)

        fr = fcsv[i] if (fcsv and i < len(fcsv)) else \
            {"capture_frame_idx": i, "timestamp_ns": int(i / fps * 1e9)}

        row = {"video_frame_idx": i,
               "capture_frame_idx": fr["capture_frame_idx"],
               "timestamp_ns": fr["timestamp_ns"],
               "host_realtime_ns": fr.get("host_realtime_ns", ""),
               "exposure": fr.get("exposure", ""), "gain": fr.get("gain", ""),
               "wb_temp": fr.get("wb_temp", ""),
               "pose_state": fr.get("pose_state", ""),
               "tx_mm": fr.get("tx_mm", ""), "ty_mm": fr.get("ty_mm", ""),
               "tz_mm": fr.get("tz_mm", ""), "in_pool": 0, **met}

        if (i % stride == 0) and (cap is None or n_written < cap):
            name = f"{sid}_{i:06d}.png"
            cv2.imwrite(str(sdir / "rgb" / name), bgr, png_opt)
            if got.get("depth") is not None:
                cv2.imwrite(str(sdir / "depth" / name), got["depth"], png_opt)
            if want_right and got.get("right") is not None:
                cv2.imwrite(str(sdir / "right" / name), got["right"], png_opt)
            if want_conf and got.get("conf") is not None:
                cv2.imwrite(str(sdir / "conf" / name), got["conf"], png_opt)
            row["in_pool"], row["filename"] = 1, name
            n_written += 1
            pool_rows.append({**row, "session_id": sid, "trip": rc["trip"],
                              "site": rc["site"], "field": rc["field"],
                              "scene_hint": rc["scene_hint"],
                              "capture_format": fmt})
        index_rows.append(row)
        prog.update(note=f"{n_written} pooled")

    prog.close(note=f"{n_written} pooled")
    for d in decs.values():
        d.close()

    if not index_rows:
        warnings.append("decoded 0 frames - file may be truncated")
    hdr = rgb_info.get("nb_frames_header")
    if hdr and hdr != len(index_rows):
        warnings.append(f"header claims {hdr} frames, decoded {len(index_rows)}")

    def write_csv(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(sdir / "meta" / "frames_index.csv", index_rows)
    write_csv(sdir / "meta" / "pool.csv", pool_rows)
    if calib:
        (sdir / "meta" / "calibration.json").write_text(json.dumps(calib, indent=2))
    for key in ("calib_txt", "calib_json", "meta_txt", "meta_json",
                "frames_csv", "dropped_csv"):
        src = sess[key]
        if src and src.exists():
            shutil.copy2(src, sdir / "meta" / f"orig_{src.name}")

    pooled = [r for r in index_rows if r["in_pool"]]
    dv = [r["depth_valid_frac_veg"] for r in pooled
          if isinstance(r.get("depth_valid_frac_veg"), float)]

    conf_pol = None
    if fmt == "v2" and isinstance(meta, dict):
        conf_pol = (meta.get("confidence_encoding") or {}).get("polarity")

    session_json = {
        "session_id": sid, "capture_format": fmt,
        "trip": rc["trip"], "site": rc["site"], "field": rc["field"],
        "scene_hint": rc["scene_hint"], "notes": rc.get("notes", ""),
        "extracted_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"folder": str(sess["folder"]),
                   "rgb_video": str(sess["rgb"]),
                   "right_video": str(sess["right"]) if sess["right"] else None,
                   "depth_video": str(sess["depth"]) if sess["depth"] else None,
                   "confidence_video": str(sess["conf"]) if sess["conf"] else None,
                   "svo_archive": str(sess["svo"]) if sess["svo"] else None,
                   "probes": infos},
        # How the pixels were produced, not just where they came from. Decode
        # settings change pixel values on a lossy source, so two sessions
        # extracted under different flags are not strictly comparable and the
        # difference is otherwise invisible after the fact.
        "decode": {"sws_flags": cfg.get("SWS_FLAGS"),
                   "png_compression": cfg["PNG_COMPRESSION"],
                   "note": "PNG is lossless; png_compression trades size "
                           "against write speed only. sws_flags affects the "
                           "YUV->RGB conversion and therefore only lossy "
                           "(v1/AVI) sources - FFV1 decodes bit-exact."},
        "capture_metadata": meta, "capture_metadata_version": meta_ver,
        "camera": {"model": "ZED 2i", "view": "VIEW.LEFT (rectified)",
                   "calibration": calib},
        "depth_encoding": {
            "container": "16-bit grayscale PNG",
            "scale_mm_per_unit": cfg["DEPTH_SCALE_MM_PER_UNIT"],
            "invalid_value": cfg["DEPTH_INVALID_VALUE"],
            "note": "Raw millimetres. depth_vis_max_mm in v1 session_meta.txt is "
                    "a GUI preview constant - do not rescale by it. 0 = invalid."},
        "confidence_encoding": ({"container": "8-bit PNG", "polarity": conf_pol}
                                if want_conf else None),
        "frames": {"decoded": len(index_rows), "pool": len(pooled),
                   "stride": stride, "nominal_fps": fps,
                   "timestamp_source": ts_source,
                   "dropped_at_capture": n_dropped,
                   "alignment": "all streams share the video frame index; each is "
                                "written from the same capture queue item"},
        "pool_summary": {
            "median_depth_valid_frac_veg": round(float(np.median(dv)), 4) if dv else None,
            "median_sharpness": round(float(np.median(
                [r["sharpness"] for r in pooled])), 2) if pooled else None,
            "median_veg_frac": round(float(np.median(
                [r["veg_frac"] for r in pooled])), 4) if pooled else None},
        "warnings": warnings}
    (sdir / "meta" / "session.json").write_text(json.dumps(session_json, indent=2))

    for w in warnings:
        print(f"      ! {w}")
    print(f"      -> {len(pooled)} pool frames of {len(index_rows)} decoded")
    if dv:
        print(f"      -> median valid depth on vegetation: {np.median(dv):.1%}")
    return session_json


def main():
    cfg = CONFIG

    # Resolve ffmpeg/ffprobe: accept a full path, otherwise look on PATH.
    for tool in ("FFMPEG", "FFPROBE"):
        given = cfg[tool]
        resolved = given if Path(given).exists() else shutil.which(given)
        if not resolved:
            sys.exit(f"ERROR: {tool} '{given}' not found. Install ffmpeg and put "
                     f"it on PATH, or set CONFIG['{tool}'] to its full path.")
        cfg[tool] = resolved
        print(f"  {tool}: {resolved}")

    if not cfg["INPUT_ROOTS"]:
        sys.exit("ERROR: INPUT_ROOTS is empty. Add at least one input directory "
                 "at the top of this file.")
    out_root = Path(cfg["OUTPUT_ROOT"])

    print("Discovering sessions...")
    sessions = discover(cfg)
    print(f"Found {len(sessions)} session folder(s)\n")
    if not sessions:
        sys.exit("Nothing to do. Check INPUT_ROOTS and RGB_PATTERNS.")

    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for s in sessions:
        try:
            r = extract_session(s, out_root, cfg)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [FAIL] {s['folder']}: {e}")

    rows = [{"session_id": r["session_id"], "capture_format": r["capture_format"],
             "trip": r["trip"], "site": r["site"], "field": r["field"],
             "scene_hint": r["scene_hint"],
             "decoded": r["frames"]["decoded"], "pool": r["frames"]["pool"],
             "timestamp_source": r["frames"]["timestamp_source"],
             "dropped_at_capture": r["frames"]["dropped_at_capture"],
             "has_svo": bool(r["source"]["svo_archive"]),
             "has_confidence": bool(r["source"]["confidence_video"]),
             "has_right": bool(r["source"]["right_video"]),
             "median_depth_valid_frac_veg": r["pool_summary"]["median_depth_valid_frac_veg"],
             "median_sharpness": r["pool_summary"]["median_sharpness"],
             "n_warnings": len(r["warnings"]),
             "source_folder": r["source"]["folder"]}
            for r in results if not r.get("dry_run")]
    if rows and not cfg["DRY_RUN"]:
        with open(out_root / "registry.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nRegistry written: {out_root / 'registry.csv'}")

    print(f"\nDone. {len(results)} session(s) processed.")


if __name__ == "__main__":
    main()
