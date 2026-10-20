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

  v3 (WeedScan Zed2i_video_datacollection.py, app v2.0.0 - see DATA_SPEC.md)
      RGB_left.mkv + Depth.mkv + Confidence.mkv (all FFV1)
      + calibration.json (nested differently), session_metadata.json,
        markers.csv
      + frames.csv with `rec_frame_idx` / `mkv_frame_idx` / `mkv_written`,
        plus speed, distance travelled, tilt from nadir and a full pose

Everything a format offers is used; nothing missing is fatal. Sessions are
tagged `capture_format` so downstream code knows what exists.

WHY v3 NEEDED MORE THAN A FEW NEW FILENAMES
--------------------------------------------
Its video files are found by the same loose patterns v1 and v2 use, so a
v3 session READS - and that is the danger. Four of its differences are silent
if it is treated as v2, every one of them producing values that are in range
and plausible rather than an error:

  depth scale       v3 stores `depth_units_per_mm` counts per millimetre,
                    default 10. Read as v2 a 1 m crop row becomes 10 m, and
                    10000 is a legal uint16 that nothing range-checks.
  depth sentinels   65534 (too close) and 65535 (too far) decode under
                    value/10 to 6553.4 and 6553.5 mm - a believable distance.
  confidence depth  Confidence.mkv is gray16le. Decoded as 8-bit grey every
                    value is divided by ~257, so a confidence of 50 reads 19.
  frames.csv order  v3 writes a row per CAPTURED frame, dropped ones included.
                    Read in file order the first drop shifts every later
                    frame's timestamp and pose by one.

So depth and confidence are normalised HERE - `depth/` stays millimetres with
0 = invalid for every format, `conf/` becomes 0..100 - and frames are joined by
`mkv_frame_idx` rather than by position. What the capture actually used is
recorded under `depth_encoding.source_units_per_mm`, so a factor-of-ten error
stays diagnosable rather than merely avoided.

A v3 session that does not record its own depth scale is REFUSED, not defaulted
to 10: a crash is recoverable and a ten-fold depth field is not.

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
        "scene_hint": "onion_only",   # mixed | onion_only | weed_only | unknown
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
    # Patterns are matched against the file STEM, and the extension is checked
    # separately against VIDEO_SUFFIXES. Earlier visits wrote AVI and later ones
    # MKV; when the patterns carried ".mkv" the AVI sessions were not merely
    # skipped, they were INVISIBLE - discover() returned them nowhere and
    # printed nothing, so a run over a mixed folder looked like a success while
    # silently extracting a fraction of it.
    "RGB_PATTERNS":   ["rgb_video", "rgb", "left", "*left*"],
    "RIGHT_PATTERNS": ["rgb_right_video", "right", "*right*"],
    "DEPTH_PATTERNS": ["depth_video", "depth", "*depth*"],
    "CONF_PATTERNS":  ["confidence_video", "confidence", "*conf*"],
    "VIDEO_SUFFIXES": [".mkv", ".avi", ".mp4", ".mov"],

    # -- Depth integrity -------------------------------------------------------
    # Depth is decoded by asking ffmpeg for gray16le. ffmpeg will produce that
    # from ANY source, including an 8-bit lossy one, by scaling values it made
    # up - so a depth stream that is not genuinely 16-bit yields plausible-
    # looking millimetres that are fiction. Nothing downstream can detect it:
    # the PNGs are valid, the QC fractions are computed, and the LEP's canopy-
    # height channel reads garbage as terrain.
    #
    # AVI is the reason this exists. Its usual codecs (MJPEG, XVID, DIVX) have
    # no 16-bit grayscale format at all, so an AVI depth track is very likely
    # 8-bit - but SOME writers do put FFV1 in AVI, which is fine. So this is
    # checked per session rather than assumed from the container.
    #
    # True  = refuse to write depth for a session whose depth is not 16-bit,
    #         extract RGB, and say so. The session is still usable for
    #         segmentation; only 3D and LEP need real depth.
    # False = write it anyway. Only correct if you have verified the pipeline
    #         that produced the file actually stored millimetres.
    "REQUIRE_16BIT_DEPTH": True,

    # -- Recovering an 8-bit depth PREVIEW (opt-in, and read this first) -------
    # Some v1 captures wrote the GUI's depth preview instead of the depth data:
    # the range scaled to 0..255 against `depth_vis_max_mm`, optionally
    # colourised, then lossily encoded. The geometry survives that, coarsely.
    #
    # Run validation/inspect_depth_video.py FIRST. It reports whether a given
    # file is a grayscale preview, a colourised one, or not a range image at
    # all - and there is nothing to recover in the last case.
    #
    # WHAT THIS IS WORTH. At DEPTH_VIS_MAX_MM=3000 one level is 11.8 mm, before
    # DCT ringing that concentrates at exactly the depth discontinuities that
    # matter. Usable for a ground plane, camera height or row geometry. NOT
    # usable for the LEP's canopy-height channel, where the whole signal is a
    # few centimetres of relief on one plant.
    #
    # Recovered frames go to `depth_approx/`, NEVER to `depth/`. A file under
    # depth/ is metric millimetres and everything downstream is entitled to
    # assume so; putting an approximation there would recreate, by hand, the
    # exact silent corruption REQUIRE_16BIT_DEPTH exists to prevent.
    "RECOVER_8BIT_DEPTH": False,

    # The scale the preview was drawn at. 3000 is the v1 capture GUI's
    # constant. If your sessions have a session_meta.txt, take it from there;
    # if they do not, this is an assumption and every distance inherits it.
    "DEPTH_VIS_MAX_MM": 3000.0,

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


def find_one(folder, patterns, want_right=False, suffixes=None):
    """Case-insensitive first match in priority order.

    Patterns match the file STEM and the extension is checked separately, so
    one pattern list covers every container a visit happened to be recorded in
    - AVI in the early sessions, MKV in the later ones, sometimes both under
    the same parent folder.

    Files with 'right' in the name are only matched by right-hand patterns, so
    RGB_right_video.mkv can never be mistaken for the left stream."""
    suffixes = [s.lower() for s in (suffixes or CONFIG["VIDEO_SUFFIXES"])]
    try:
        files = [f for f in folder.iterdir() if f.is_file()]
    except (PermissionError, OSError):
        return None
    for pat in patterns:
        pat_l = pat.lower()
        for f in files:
            if f.suffix.lower() not in suffixes:
                continue
            n, stem = f.name.lower(), f.stem.lower()
            if ("right" in n) != want_right:
                continue
            if ("*" in pat_l and _glob_match(stem, pat_l)) or stem == pat_l:
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
    if "calibration_rectified" in j:
        return _parse_calibration_json_v3(j, path)
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


def _parse_calibration_json_v3(j, path):
    """v3 calibration.json: `calibration_rectified.left_cam`, baseline in mm.

    Same intrinsics under different keys - `left_cam` rather than `left`,
    `baseline` rather than `baseline_mm`, and the fields of view suffixed
    `_deg`. Falling through to the v2 reader would not raise; it would return a
    calibration with every value None, and the session would be extracted with
    no intrinsics at all."""
    rect = j.get("calibration_rectified", {})
    left, right = rect.get("left_cam"), rect.get("right_cam")
    st = rect.get("stereo_transform") or {}
    out = {"source": path.name, "format": "v3_json",
           "left": left, "right": right,
           "baseline_mm": rect.get("baseline"),
           "stereo_translation_mm": st.get("translation"),
           "rotation_vector_rad": st.get("rotation_vector_rad"),
           "raw": j.get("calibration_raw_factory"),
           "serial_number": j.get("serial_number"),
           "firmware": j.get("camera_firmware_version"),
           "camera_imu_transform": j.get("camera_imu_transform"),
           "resolution": j.get("resolution"), "fps": j.get("fps")}
    if left and left.get("h_fov_deg") is not None:
        out["hfov_deg"] = left["h_fov_deg"]
    return _derive_calib(out)


def parse_session_meta(txt_path, json_path, v3_path=None):
    """Newest format wins: v3 session_metadata.json, then v2, then v1."""
    if v3_path and v3_path.exists():
        return json.loads(v3_path.read_text()), "v3"
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


#: v3 depth sentinels. They are NOT out-of-band: under `value / 10` they decode
#: to 6553.4 and 6553.5 mm, which is a plausible distance rather than an
#: obvious error - so a reader that handles only `0 = invalid` gets a believable
#: wrong answer instead of a crash.
DEPTH_CODE_TOO_CLOSE = 65534
DEPTH_CODE_TOO_FAR = 65535


def depth_encoding_of(meta, meta_ver):
    """(units_per_mm, confidence_scale) for a capture format.

    THE SILENT 10x. v3 stores depth at `depth_units_per_mm` counts per
    millimetre, default 10, while v1 and v2 store millimetres directly. Read a
    v3 stream as v2 and every distance is ten times too large - a 1 m row
    becomes 10 m - and nothing raises, because 10000 is a perfectly legal
    uint16. Normalising here is what keeps `depth/` meaning millimetres for
    every consumer downstream, none of which look at a scale.

    A MISSING KEY IS NOT A DEFAULT. If a v3 session does not record its own
    encoding we refuse rather than assume 10: a crash is recoverable and a
    ten-fold depth field is not. The spec's "default 10" is a statement about
    the recorder, not a licence for a reader to guess."""
    if meta_ver != "v3":
        return 1.0, None
    enc = (meta or {}).get("encoding") or {}
    units = enc.get("depth_units_per_mm")
    if units is None:
        raise ValueError(
            "v3 session_metadata.json has no encoding.depth_units_per_mm. "
            "Refusing to guess: assuming the default of 10 on a session that "
            "used a different scale produces depth that is silently wrong by "
            "that factor, and nothing downstream checks.")
    return float(units), float(enc.get("confidence_scale") or 100.0)


def depth_to_mm_u16(raw_u16, units_per_mm):
    """A capture's native depth counts as uint16 MILLIMETRES, 0 = invalid.

    Both sentinels become 0. They mean "no usable distance here" - nearer than
    the sensor minimum, or past its maximum - which is exactly what 0 already
    means to every consumer. Keeping them would require every caller to learn
    two more magic numbers, and the one that forgot would read 65 metres."""
    raw = np.asarray(raw_u16)
    bad = (raw == 0) | (raw == DEPTH_CODE_TOO_CLOSE) | (raw == DEPTH_CODE_TOO_FAR)
    if float(units_per_mm) == 1.0:
        mm = raw.astype(np.float64)
    else:
        mm = raw.astype(np.float64) / float(units_per_mm)
    mm = np.rint(mm)
    mm[bad] = 0
    return np.clip(mm, 0, 65535).astype(np.uint16)


def confidence_to_pct_u8(raw_u16, scale):
    """v3 confidence counts as 0..100 in a uint8, NaN code -> 100 (worst).

    v3 writes confidence as gray16le at `confidence_scale` counts per percent;
    decoding it as 8-bit grey - which is what the v1/v2 path does - divides
    every value by about 257 and turns a confidence of 50 into 19.

    65535 means UNDEFINED, and it maps to 100 rather than 0 because ZED
    confidence is inverted: higher is LESS reliable, so 100 is "trust nothing"
    and 0 would be "perfect". A pixel the SDK could not judge must not arrive
    as the most trustworthy one in the frame."""
    raw = np.asarray(raw_u16)
    pct = raw.astype(np.float64) / float(scale or 100.0)
    pct[raw == 65535] = 100.0
    return np.clip(np.rint(pct), 0, 255).astype(np.uint8)


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
        if "mkv_frame_idx" in cols:
            return _read_frames_csv_v3(rdr)
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


def _read_frames_csv_v3(rdr):
    """v3 rows, placed at their OWN mkv index rather than in file order.

    THE TRAP THIS CLOSES. v1 and v2 write one row per ENCODED frame, so row i
    describes video frame i and reading in order is correct. v3 writes one row
    per CAPTURED frame - including frames the encoder dropped, flagged
    `mkv_written=0` - and gives every stored frame its own `mkv_frame_idx`.

    Read v3 in file order and the first drop shifts everything after it by one:
    every later frame gets the timestamp, pose and exposure of its neighbour.
    Nothing downstream can detect that, because the values are all plausible.
    So the list is built BY INDEX, and a gap stays a gap.

    Returns a list where position i is the row for video frame i, or None."""
    by_idx = {}
    for r in rdr:
        if str(r.get("mkv_written", "")).strip() != "1":
            continue
        try:
            idx = int(r["mkv_frame_idx"])
            rec = {
                "capture_frame_idx": int(r["rec_frame_idx"]),
                "timestamp_ns": int(r["image_timestamp_ns"]),
                "host_realtime_ns": r.get("system_time_ns", ""),
                "exposure": r.get("exposure_pct", ""),
                "gain": r.get("gain", ""),
                "wb_temp": r.get("wb_temp_k", ""),
                "pose_state": r.get("pose_state", ""),
                "tx_mm": r.get("tx_mm", ""), "ty_mm": r.get("ty_mm", ""),
                "tz_mm": r.get("tz_mm", ""),
                # v3 only, and the reason it is worth carrying: a split can be
                # made in METRES of ground instead of frame counts, and a
                # blurred or tilted frame can be dropped before anyone
                # annotates it.
                "qx": r.get("qx", ""), "qy": r.get("qy", ""),
                "qz": r.get("qz", ""), "qw": r.get("qw", ""),
                "speed_m_s": r.get("speed_m_s", ""),
                "distance_m": r.get("distance_m", ""),
                "tilt_from_nadir_deg": r.get("tilt_from_nadir_deg", ""),
                "exposure_ms_approx": r.get("exposure_ms_approx", ""),
                "est_missing_before": r.get("est_missing_before", ""),
            }
        except (KeyError, ValueError, TypeError):
            return None, None
        by_idx[idx] = rec
    if not by_idx:
        return None, None
    out = [by_idx.get(i) for i in range(max(by_idx) + 1)]
    return out, "v3"


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
            sfx = cfg["VIDEO_SUFFIXES"]
            rgb = find_one(folder, cfg["RGB_PATTERNS"], suffixes=sfx)
            if rgb is None:
                # Flag anything that looks like a recording, so a folder is
                # never lost without a line of output. The old version only
                # spoke up when it found depth or an SVO - and since it looked
                # for those with the same extension list that had just failed,
                # a folder recorded in an unsupported container produced NO
                # message at all.
                has_depth = find_one(folder, cfg["DEPTH_PATTERNS"],
                                     suffixes=sfx) is not None
                has_svo = next((p for p in folder.glob("*.svo*")), None) is not None
                others = sorted({p.suffix.lower() for p in folder.iterdir()
                                 if p.is_file() and p.suffix.lower()
                                 not in [s.lower() for s in sfx]
                                 and p.suffix.lower() not in (".txt", ".json",
                                                              ".csv")})
                if has_svo:
                    why = "SVO only - decode it to MKV first"
                elif has_depth:
                    why = "depth present but no RGB/left video"
                elif others:
                    why = (f"video-looking files in an unsupported container "
                           f"({', '.join(others)}) - add it to VIDEO_SUFFIXES")
                else:
                    continue                      # not a recording folder
                print(f"  [SKIP] {folder} : no usable RGB video ({why})")
                continue

            def opt(name):
                p = folder / name
                return p if p.exists() else None

            found.append({
                "folder": folder, "rgb": rgb,
                "right": find_one(folder, cfg["RIGHT_PATTERNS"],
                                  want_right=True, suffixes=sfx),
                "depth": find_one(folder, cfg["DEPTH_PATTERNS"], suffixes=sfx),
                "conf": find_one(folder, cfg["CONF_PATTERNS"], suffixes=sfx),
                "calib_txt": opt("calibration_params.txt"),
                "calib_json": opt("calibration.json"),
                "meta_txt": opt("session_meta.txt"),
                "meta_json": opt("session.json"),
                # v3 renamed it. A rename with no version marker is why this
                # lookup has to be explicit: opt("session.json") simply misses,
                # and the depth SCALE lives in here.
                "meta_json_v3": opt("session_metadata.json"),
                "frames_csv": opt("frames.csv"),
                "markers_csv": opt("markers.csv"),
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

    fmt = ("v3" if sess.get("meta_json_v3") else
           "v2" if (sess["meta_json"] or sess["conf"] or sess["calib_json"])
           else "v1")

    rgb_info = probe(cfg["FFPROBE"], sess["rgb"])
    W, H = rgb_info["width"], rgb_info["height"]
    infos = {"rgb": rgb_info}
    for key in ("right", "depth", "conf"):
        if sess[key]:
            infos[key] = probe(cfg["FFPROBE"], sess[key])
            if (infos[key]["width"], infos[key]["height"]) != (W, H):
                warnings.append(f"{key} resolution differs from RGB - stream dropped")
                sess[key] = None

    preview = None                       # recovered 8-bit depth, if any
    if sess["depth"] and cfg.get("REQUIRE_16BIT_DEPTH", True):
        bad = depth_precision_problem(infos.get("depth"))
        if bad:
            warnings.append(f"depth stream dropped: {bad}")
            print(f"  [!] {sid}: {bad}\n"
                  f"      RGB is still extracted; this session has no metric "
                  f"depth. Set REQUIRE_16BIT_DEPTH=False only if you have "
                  f"verified the writer stored millimetres.")
            if cfg.get("RECOVER_8BIT_DEPTH", False):
                preview = _plan_preview_recovery(sess, cfg, sid, W, H, warnings)
                if preview:
                    preview["path"] = sess["depth"]
            sess["depth"] = None

    if sess["depth"] is None and not preview:
        warnings.append("no depth video - RGB only session")
    if fmt == "v1":
        warnings.append("v1 capture format: no confidence map, no right image, no "
                        "SVO, no exposure/pose log - see docs/capture_changelog.md")

    calib = (parse_calibration_json(sess["calib_json"])
             or parse_calibration_txt(sess["calib_txt"]))
    if calib is None:
        warnings.append("NO calibration - session unusable for 3D work")

    meta, meta_ver = parse_session_meta(sess["meta_txt"], sess["meta_json"],
                                        sess.get("meta_json_v3"))
    if meta is None:
        warnings.append("no session metadata - parameters taken from video header")
    # Raises on a v3 session that does not record its own depth scale. That is
    # deliberate: see depth_encoding_of.
    units_per_mm, conf_scale = depth_encoding_of(meta, meta_ver)
    if units_per_mm != 1.0:
        warnings.append(
            f"depth rescaled from {units_per_mm:g} counts/mm to millimetres; "
            f"codes {DEPTH_CODE_TOO_CLOSE} and {DEPTH_CODE_TOO_FAR} mapped to "
            f"0 (invalid)")

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
        + (["depth_approx"] if preview else []) \
        + (["right"] if want_right else []) + (["conf"] if want_conf else [])
    for sub in subs:
        (sdir / sub).mkdir(parents=True, exist_ok=True)

    F = cfg["FFMPEG"]
    SWS = cfg.get("SWS_FLAGS")
    decs = {"rgb": RawDecoder(F, sess["rgb"], "bgr24", W, H, 3, np.uint8,
                              sws_flags=SWS)}
    if sess["depth"]:
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
    if sess["right"]:
        decs["right"] = RawDecoder(F, sess["right"], "bgr24", W, H, 3,
                                   np.uint8, sws_flags=SWS)
    if sess["conf"]:
        # v3 writes confidence as gray16le. Asking ffmpeg for 8-bit grey from
        # it divides every value by ~257, turning a confidence of 50 into 19 -
        # a number that is in range, plausible, and wrong.
        if fmt == "v3":
            decs["conf"] = RawDecoder(F, sess["conf"], "gray16le", W, H, 1,
                                      np.uint16, sws_flags=SWS)
        else:
            decs["conf"] = RawDecoder(F, sess["conf"], "gray", W, H, 1,
                                      np.uint8, sws_flags=SWS)
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
        # NORMALISE BEFORE ANYTHING READS IT, including the QC metrics. Those
        # go into the registry and are compared across sessions, so a v3
        # session measured in its native counts would be incomparable with
        # every v1 and v2 session beside it.
        if got.get("depth") is not None and units_per_mm != 1.0:
            got["depth"] = depth_to_mm_u16(got["depth"], units_per_mm)
        if got.get("conf") is not None and fmt == "v3":
            got["conf"] = confidence_to_pct_u8(got["conf"], conf_scale)
        met = frame_metrics(bgr, got.get("depth"), got.get("conf"), cfg)

        # v3 leaves a hole where the encoder dropped a frame, so a missing row
        # is a real state rather than the end of the file.
        fr = (fcsv[i] if (fcsv and i < len(fcsv) and fcsv[i] is not None)
              else {"capture_frame_idx": i, "timestamp_ns": int(i / fps * 1e9)})

        row = {"video_frame_idx": i,
               "capture_frame_idx": fr["capture_frame_idx"],
               "timestamp_ns": fr["timestamp_ns"],
               "host_realtime_ns": fr.get("host_realtime_ns", ""),
               "exposure": fr.get("exposure", ""), "gain": fr.get("gain", ""),
               "wb_temp": fr.get("wb_temp", ""),
               "pose_state": fr.get("pose_state", ""),
               "tx_mm": fr.get("tx_mm", ""), "ty_mm": fr.get("ty_mm", ""),
               "tz_mm": fr.get("tz_mm", ""),
               # v3 only, blank elsewhere. distance_m is the one that changes
               # what is possible: a split can be separated by METRES of
               # ground rather than by a guessed number of frames.
               "qx": fr.get("qx", ""), "qy": fr.get("qy", ""),
               "qz": fr.get("qz", ""), "qw": fr.get("qw", ""),
               "speed_m_s": fr.get("speed_m_s", ""),
               "distance_m": fr.get("distance_m", ""),
               "tilt_from_nadir_deg": fr.get("tilt_from_nadir_deg", ""),
               "exposure_ms_approx": fr.get("exposure_ms_approx", ""),
               "est_missing_before": fr.get("est_missing_before", ""),
               "in_pool": 0, **met}

        if (i % stride == 0) and (cap is None or n_written < cap):
            name = f"{sid}_{i:06d}.png"
            cv2.imwrite(str(sdir / "rgb" / name), bgr, png_opt)
            if got.get("depth") is not None:
                cv2.imwrite(str(sdir / "depth" / name), got["depth"], png_opt)
            if got.get("depth_approx") is not None:
                cv2.imwrite(str(sdir / "depth_approx" / name),
                            got["depth_approx"], png_opt)
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
        # depth_kind is the field to read before using anything under depth*/:
        #   metric      - depth/, real millimetres from a 16-bit stream
        #   approximate - depth_approx/, recovered from an 8-bit preview
        #   none        - the session has no depth at all
        # The three are not interchangeable and the distinction is not
        # recoverable from the PNGs, which are uint16 millimetres in both cases.
        "depth_kind": ("metric" if sess["depth"] else
                       ("approximate" if preview else "none")),
        "depth_encoding": ({
            "container": "16-bit grayscale PNG",
            "directory": "depth",
            "scale_mm_per_unit": cfg["DEPTH_SCALE_MM_PER_UNIT"],
            "invalid_value": cfg["DEPTH_INVALID_VALUE"],
            # What the CAPTURE used, kept so a 10x error is diagnosable after
            # the fact rather than merely avoided now. scale_mm_per_unit above
            # describes what is in depth/; this describes what it came from.
            "source_units_per_mm": units_per_mm,
            "source_sentinels_mapped_to_invalid": (
                [DEPTH_CODE_TOO_CLOSE, DEPTH_CODE_TOO_FAR]
                if units_per_mm != 1.0 else []),
            "note": "Raw millimetres. depth_vis_max_mm in v1 session_meta.txt is "
                    "a GUI preview constant - do not rescale by it. 0 = invalid."}
            if sess["depth"] else ({
                "container": "16-bit grayscale PNG",
                "directory": "depth_approx",
                "approximate": True,
                "recovered_from": preview["kind"],
                "colormap": preview["colormap"],
                "tv_range_corrected": preview["tv_range"],
                "vis_max_mm": preview["vis_max_mm"],
                "quantisation_mm": preview["quantisation_mm"],
                "invalid_sentinel_survived": preview["sentinel_survived"],
                "note": "NOT measurement. Recovered from an 8-bit depth preview "
                        "by inverting the capture GUI's display scaling, so "
                        "every value inherits the vis_max_mm assumption and "
                        "carries quantisation_mm of error before lossy "
                        "compression. Usable for ground plane and row geometry; "
                        "not for the LEP canopy-height channel."}
                if preview else None)),
        # v3 PINS THE POLARITY, which v1 and v2 could only probe for per
        # session: higher is LESS reliable, so a pixel passes a threshold T
        # when confidence <= T.
        "confidence_encoding": (
            {"directory": "conf", "container": "8-bit grayscale PNG",
             "units": "percent, 0..100", "polarity": "higher_is_worse",
             "undefined_mapped_to": 100, "source_scale": conf_scale,
             "note": "ZED MEASURE.CONFIDENCE. Keep pixels with conf <= T; "
                     "30-50 for clean geometry, 70 for general use."}
            if (want_conf and fmt == "v3") else
            {"container": "8-bit PNG", "polarity": conf_pol}
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

    # OPERATOR NOTES SURVIVE THE EXTRACTION. markers.csv is free text typed
    # during recording - "start row 3", "weed patch", "skip: person in view" -
    # and it is the only record of where a row or plot begins and which
    # stretches to throw away. Copied rather than parsed: the conventions are
    # the operator's, and inventing a schema for them here would guess.
    if sess.get("markers_csv") and sess["markers_csv"].exists():
        shutil.copyfile(sess["markers_csv"], sdir / "meta" / "markers.csv")

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
             # metric | approximate | none. Read this before comparing the
             # depth column across sessions, or before using depth at all.
             "depth_kind": r.get("depth_kind", "none"),
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
