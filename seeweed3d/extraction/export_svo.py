#!/usr/bin/env python3
"""
SeeWeed3D - Stage 0.5: turn an SVO-only recording into an extractable session
=============================================================================
    python -m seeweed3d.extraction.export_svo \
        --session "D:/ZED_Data_Jetson/Session_20261021_123954" \
        --out     "E:/Dataset_Vidalia/Jetson_Oct_exported"

    # always do this first - 50 frames, a few seconds, nothing committed:
    python -m seeweed3d.extraction.export_svo --session ... --out ... --limit 50

WHY THIS EXISTS
---------------
Some captures write only `recording.svo2` and the sidecar logs - no
RGB_left.mkv, no Depth.mkv, no Confidence.mkv. extract_sessions.py reports
those folders as "SVO only - decode it to MKV first" and skips them, which is
correct: an SVO is a stereo archive, not an image stream, and nothing
downstream can read one.

The SDK's own ZED_SVO_Export is not a substitute. It writes depth either as an
8-bit preview or as PNGs with its own conventions, and an 8-bit depth preview
is precisely the input REQUIRE_16BIT_DEPTH exists to refuse: ffmpeg will turn
it into plausible-looking gray16le millimetres that are fiction. This writes
the v3 contract exactly - FFV1, depth in fixed-point counts per millimetre
with the sentinel codes, 16-bit confidence - so the result is a session the
existing extractor reads with no special case at all.

DEPTH IS RECOMPUTED, AND THAT IS NOT A DETAIL
---------------------------------------------
An SVO stores the rectified stereo pair, not depth. Depth comes back by
running the matcher again at export time, so it depends on the SDK version and
the depth mode used HERE, not at capture. Two exports of one SVO under
different modes are different measurements of the same field.

So the mode actually used is written into session_metadata.json under
`export`, beside the SDK version, and defaults to whatever the capture
recorded in `capture.depth_mode`. Never compare depth across sessions exported
under different modes without saying so.

THE JOIN IS ON TIMESTAMPS, NOT POSITIONS
----------------------------------------
frames.csv has one row per captured frame. The export writes one MKV frame per
SVO frame. Those two sequences are not guaranteed to line up by position - and
if they do not, pairing them by position gives every frame its neighbour's
timestamp, pose and exposure, which is undetectable afterwards because every
value is plausible.

Each exported frame is therefore matched to its row by `image_timestamp_ns`,
which both sides record from the same clock. A frame with no matching row
stops the export rather than being written with a guess.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.progress import Progress  # noqa: E402

# The v3 depth contract. See DATA_SPEC and extract_sessions.depth_to_mm_u16 -
# these two must agree, and the round-trip test is what holds them together.
DEPTH_INVALID = 0
DEPTH_TOO_CLOSE = 65534
DEPTH_TOO_FAR = 65535
DEPTH_MAX_COUNT = 65533          # the largest value that is a real distance

CONF_UNDEFINED = 65535

SIDECARS = ("calibration.json", "calibration_params.txt", "markers.csv",
            "events.txt", "README.txt", "verify_report.json",
            "checksums.sha256")


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def encode_depth_mm(depth_mm, units_per_mm):
    """Float millimetres from the SDK -> the v3 uint16 encoding.

    The SDK signals the three failure modes in-band and they are NOT
    interchangeable: NaN is an occlusion (no measurement), -inf is nearer than
    the sensor's minimum, +inf is past its maximum. Collapsing them all to one
    "invalid" loses the only evidence of WHY a region has no depth, which is
    the difference between a camera mounted too low and a crop canopy that
    swallowed the projector.

    A real distance that would encode past DEPTH_MAX_COUNT is clipped to it,
    never allowed to land on a sentinel - a far-but-valid reading that
    became 65535 would be read back as "too far" and then as invalid, which
    is a lie in the one direction that silently removes data.
    """
    d = np.asarray(depth_mm, dtype=np.float64)
    out = np.full(d.shape, DEPTH_INVALID, dtype=np.uint16)

    finite = np.isfinite(d) & (d > 0)
    counts = np.rint(d[finite] * float(units_per_mm))
    out[finite] = np.clip(counts, 1, DEPTH_MAX_COUNT).astype(np.uint16)

    out[np.isneginf(d)] = DEPTH_TOO_CLOSE
    out[np.isposinf(d)] = DEPTH_TOO_FAR
    return out


def encode_confidence(conf_pct, scale):
    """SDK confidence (0..100, HIGHER IS WORSE) -> the v3 uint16 encoding.

    Undefined maps to CONF_UNDEFINED, which is read back as the WORST possible
    confidence. The other direction - undefined read as perfect - would send
    every unmeasured pixel into the dataset wearing full marks.
    """
    c = np.asarray(conf_pct, dtype=np.float64)
    out = np.full(c.shape, CONF_UNDEFINED, dtype=np.uint16)
    ok = np.isfinite(c)
    out[ok] = np.clip(np.rint(c[ok] * float(scale)), 0,
                      CONF_UNDEFINED - 1).astype(np.uint16)
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

class FfvWriter:
    """One lossless FFV1 stream, fed raw frames over a pipe."""

    def __init__(self, path, pix_fmt, w, h, fps, ffmpeg="ffmpeg"):
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", f"{fps}",
             "-i", "pipe:0", "-c:v", "ffv1", "-level", "3",
             "-pix_fmt", pix_fmt, str(path)],
            stdin=subprocess.PIPE)
        self.n = 0

    def write(self, arr):
        self.proc.stdin.write(np.ascontiguousarray(arr).tobytes())
        self.n += 1

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        return self.proc.wait()


# --------------------------------------------------------------------------- #
# Reading the SVO (the only part that needs the ZED SDK)
# --------------------------------------------------------------------------- #

def open_svo(svo_path, depth_mode=None):
    """(info, frame_iterator) for an SVO. Yields (ts_ns, bgr, depth_mm, conf).

    pyzed is imported here rather than at module scope so that everything
    above - the encoding, the join, the metadata - is testable, and fails in
    tests, on a machine with no SDK and no camera.
    """
    try:
        import pyzed.sl as sl
    except ImportError:
        raise SystemExit(
            "pyzed is not installed. The ZED SDK and its Python API are "
            "needed to read an SVO: install the SDK, then run "
            "`python -m pip install <SDK>/python/pyzed-*.whl` (or the SDK's "
            "get_python_api.py) in this environment.")

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.MILLIMETER
    if depth_mode:
        try:
            init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode.upper())
        except AttributeError:
            raise SystemExit(f"unknown depth mode {depth_mode!r}; this SDK has "
                             f"{[m.name for m in sl.DEPTH_MODE]}")

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"cannot open {svo_path}: {status}")

    res = cam.get_camera_information().camera_configuration.resolution
    info = {"width": res.width, "height": res.height,
            "fps": cam.get_camera_information().camera_configuration.fps,
            "n_frames": cam.get_svo_number_of_frames(),
            "depth_mode": init.depth_mode.name,
            "sdk_version": ".".join(str(x) for x in sl.Camera.get_sdk_version()
                                    .split(".")[:3]) if hasattr(
                                        sl.Camera, "get_sdk_version") else "?"}

    def frames():
        img, dep, cnf = sl.Mat(), sl.Mat(), sl.Mat()
        rt = sl.RuntimeParameters()
        try:
            while True:
                err = cam.grab(rt)
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if err != sl.ERROR_CODE.SUCCESS:
                    raise SystemExit(f"grab failed at SVO position "
                                     f"{cam.get_svo_position()}: {err}")
                cam.retrieve_image(img, sl.VIEW.LEFT)
                cam.retrieve_measure(dep, sl.MEASURE.DEPTH)
                cam.retrieve_measure(cnf, sl.MEASURE.CONFIDENCE)
                ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                # BGRA -> BGR. A copy, because the Mat is reused every grab.
                yield (int(ts),
                       np.array(img.get_data()[:, :, :3], copy=True),
                       np.array(dep.get_data(), copy=True),
                       np.array(cnf.get_data(), copy=True))
        finally:
            cam.close()

    return info, frames()


# --------------------------------------------------------------------------- #
# frames.csv
# --------------------------------------------------------------------------- #

def read_frames_csv(path):
    """(fieldnames, rows, {image_timestamp_ns: row}) - or (None, [], {})."""
    if not Path(path).exists():
        return None, [], {}
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        names = rdr.fieldnames or []
    by_ts = {}
    for r in rows:
        try:
            by_ts[int(r["image_timestamp_ns"])] = r
        except (KeyError, ValueError, TypeError):
            continue
    return names, rows, by_ts


def write_frames_csv(path, fieldnames, rows):
    for extra in ("mkv_written", "mkv_frame_idx"):
        if extra not in fieldnames:
            fieldnames = list(fieldnames) + [extra]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# The export
# --------------------------------------------------------------------------- #

def export_session(session_dir, out_dir, open_source=open_svo, limit=None,
                   units_per_mm=10, conf_scale=100, depth_mode=None,
                   ffmpeg="ffmpeg", probe_frames=5):
    """Write an extractable v3 session beside the SVO it came from.

    The source folder is never written to. Everything lands in
    `out_dir/<source folder name>/`, so the archive stays the archive.
    """
    session_dir = Path(session_dir)
    svos = sorted(session_dir.glob("*.svo*"))
    if not svos:
        raise SystemExit(f"no .svo/.svo2 file in {session_dir}")
    svo = svos[0]

    meta_path = session_dir / "session_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    depth_mode = depth_mode or (meta.get("capture") or {}).get("depth_mode")

    names, rows, by_ts = read_frames_csv(session_dir / "frames.csv")
    if not by_ts:
        raise SystemExit(
            f"{session_dir / 'frames.csv'} has no usable image_timestamp_ns "
            f"column, so exported frames cannot be matched to their pose and "
            f"exposure. Refusing to write a session whose metadata would be "
            f"attached by position.")

    out_sess = Path(out_dir) / session_dir.name
    out_sess.mkdir(parents=True, exist_ok=True)

    info, frames = open_source(svo, depth_mode)
    w, h, fps = info["width"], info["height"], info["fps"]
    print(f"  {svo.name}: {info.get('n_frames', '?')} frames, {w}x{h} @ {fps}fps"
          f" | depth_mode={info.get('depth_mode')} "
          f"| sdk={info.get('sdk_version')}")

    wr = {"rgb": FfvWriter(out_sess / "RGB_left.mkv", "bgr24", w, h, fps, ffmpeg),
          "depth": FfvWriter(out_sess / "Depth.mkv", "gray16le", w, h, fps, ffmpeg),
          "conf": FfvWriter(out_sess / "Confidence.mkv", "gray16le", w, h, fps,
                            ffmpeg)}

    for r in rows:                      # nothing is in the MKV until it is
        r["mkv_written"], r["mkv_frame_idx"] = 0, ""

    total = limit or info.get("n_frames") or 0
    prog = Progress(total, f"  [{session_dir.name}]", unit="frames")
    matched = unmatched = 0
    try:
        for i, (ts, bgr, depth_mm, conf) in enumerate(frames):
            if limit and i >= limit:
                break
            row = by_ts.get(int(ts))
            if row is None:
                unmatched += 1
                # FAIL EARLY. Finding this out after 40 minutes of export is
                # the same information at a far worse moment, and the cause is
                # always the same: this frames.csv does not belong to this SVO.
                if i + 1 >= probe_frames and matched == 0:
                    raise SystemExit(
                        f"none of the first {i + 1} SVO frames has a row in "
                        f"frames.csv matching its timestamp. The log and the "
                        f"recording are probably from different sessions - "
                        f"check that {session_dir} holds one capture.")
            else:
                row["mkv_written"], row["mkv_frame_idx"] = 1, wr["rgb"].n
                matched += 1

            wr["rgb"].write(bgr[:, :, :3])
            wr["depth"].write(encode_depth_mm(depth_mm, units_per_mm))
            wr["conf"].write(encode_confidence(conf, conf_scale))
            prog.update(1, note=f"{matched} matched")
    finally:
        for v in wr.values():
            v.close()
    prog.close()

    n_written = wr["rgb"].n
    write_frames_csv(out_sess / "frames.csv", names or list(rows[0].keys()),
                     rows)

    for name in SIDECARS:
        src = session_dir / name
        if src.exists():
            shutil.copyfile(src, out_sess / name)
    for conf_file in session_dir.glob("SN*.conf"):
        shutil.copyfile(conf_file, out_sess / conf_file.name)

    # THE EXPORTER IS THE ENCODER, SO THE EXPORTER DECLARES THE ENCODING.
    # extract_sessions refuses a v3 session that does not state its own depth
    # scale rather than assuming one, and it is right to: a session written
    # here in tenths of a millimetre and read as millimetres is a ten-fold
    # depth field that looks entirely reasonable.
    meta.setdefault("encoding", {})
    meta["encoding"].update({
        "depth_units_per_mm": units_per_mm,
        "depth_codes": {"invalid": DEPTH_INVALID, "too_close": DEPTH_TOO_CLOSE,
                        "too_far": DEPTH_TOO_FAR},
        "confidence_scale": conf_scale})
    meta["export"] = {
        "tool": "seeweed3d.extraction.export_svo",
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_svo": str(svo),
        "source_svo_bytes": svo.stat().st_size,
        "sdk_version": info.get("sdk_version"),
        # Depth was RECOMPUTED under this mode. It is a property of the export,
        # not of the recording, and two exports under different modes are not
        # the same measurement.
        "depth_mode": info.get("depth_mode"),
        "frames_written": n_written,
        "frames_matched_to_log": matched,
        "frames_unmatched": unmatched,
        "partial": bool(limit) and n_written < (info.get("n_frames") or 0),
    }
    (out_sess / "session_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"  -> {out_sess}")
    print(f"     {n_written} frames written, {matched} matched to frames.csv"
          + (f", {unmatched} UNMATCHED" if unmatched else ""))
    if meta["export"]["partial"]:
        print("     PARTIAL export (--limit) - not a dataset, just a check")
    if unmatched:
        print(f"     ! {unmatched} exported frame(s) have no row in "
              f"frames.csv: they carry no timestamp, pose or exposure and "
              f"extraction will fall back to nominal fps for them")
    return {"out": out_sess, "frames": n_written, "matched": matched,
            "unmatched": unmatched, "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True,
                    help="a recording folder containing recording.svo2")
    ap.add_argument("--out", required=True,
                    help="where the exported session folder is written; the "
                         "source folder is never modified")
    ap.add_argument("--limit", type=int, default=None,
                    help="export only the first N frames (trial run)")
    ap.add_argument("--depth-mode", default=None,
                    help="SDK depth mode; defaults to the one the capture "
                         "recorded in session_metadata.json")
    ap.add_argument("--depth-units-per-mm", type=int, default=10,
                    help="fixed-point depth scale to write (default 10, the "
                         "v3 contract: tenths of a millimetre)")
    ap.add_argument("--confidence-scale", type=int, default=100)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args(argv)

    export_session(a.session, a.out, limit=a.limit, depth_mode=a.depth_mode,
                   units_per_mm=a.depth_units_per_mm,
                   conf_scale=a.confidence_scale, ffmpeg=a.ffmpeg)


if __name__ == "__main__":
    main()
