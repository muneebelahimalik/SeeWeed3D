#!/usr/bin/env python3
"""Why a recording cannot be read, and whether anything in it is recoverable.

ffprobe's answer to a broken Matroska file is a wall of EBML parser complaints:

    [matroska,webm] Duplicate element
    [matroska,webm] 0x00 at pos 110 (0x6e) invalid as first byte of an EBML number
    [matroska,webm] Element at 0x59 ending at 0xa4dbb8c53abe6 exceeds containing
                    master element ending at 0x1405

That text says the container is malformed. It does NOT say which of two very
different things happened, and the difference decides what you do next:

  A FALSE START. The capture app opened the file, ffmpeg wrote its EBML header
    and reserved space for the seek index, and then no frame ever arrived -
    the operator stopped immediately, or the grab loop failed. The file is a
    few kilobytes of header followed by zeros. There is no image data in it,
    and no amount of repair will produce any. Those zeros are what ffprobe is
    tripping over at "pos 110".

  AN UNFINALISED RECORDING. Frames were written, then the process died before
    ffmpeg could patch the segment length and append the cue index. The frames
    are all still there; the file just has no usable table of contents, and a
    remux rebuilds one.

Telling them apart needs two reads of a few bytes, so it should never be a
guess. This module writes nothing and changes nothing.
"""
import csv
from pathlib import Path

# EBML header magic - every Matroska/WebM file starts with it.
EBML_MAGIC = b"\x1a\x45\xdf\xa3"
# Cluster element ID. Frame data lives in clusters and nowhere else, so the
# presence of one is the test for "did any image data reach the disk".
CLUSTER_ID = b"\x1f\x43\xb6\x75"

# ffmpeg writes the header, then reserves a few KB for the seek index, then
# starts the first cluster. One MiB therefore covers the first cluster of any
# real recording with room to spare - an FFV1 frame at 2208x1242 is several MB,
# so a file holding even ONE frame has its cluster header inside this window.
SCAN_BYTES = 1 << 20

VIDEO_SUFFIXES = (".mkv", ".avi", ".mp4", ".mov")


def diagnose_video_file(path, scan_bytes=SCAN_BYTES):
    """One video file: what it is, and whether it holds frame data."""
    p = Path(path)
    if not p.exists():
        return {"path": p, "size_bytes": 0, "verdict": "missing",
                "detail": "file does not exist"}

    size = p.stat().st_size
    if size == 0:
        return {"path": p, "size_bytes": 0, "verdict": "empty_file",
                "detail": "0 bytes - the file was created and never written"}

    with open(p, "rb") as f:
        head = f.read(scan_bytes)

    if not head.startswith(EBML_MAGIC):
        kind = ("a RIFF/AVI container" if head[:4] == b"RIFF"
                else "not a Matroska file (no EBML header)")
        return {"path": p, "size_bytes": size, "verdict": "not_matroska",
                "detail": kind}

    if CLUSTER_ID in head:
        return {"path": p, "size_bytes": size, "verdict": "has_frame_data",
                "detail": f"EBML header plus frame data ({_mb(size)})"}

    return {"path": p, "size_bytes": size, "verdict": "header_only",
            "detail": (f"EBML header only, no cluster element in the first "
                       f"{_mb(len(head))} - no frame data reached this file")}


def _mb(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.1f} kB"


def _frames_csv_counts(path):
    """(rows, rows claiming an MKV frame). v3 writes `mkv_written`."""
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except Exception:
        return None, None
    if not rows:
        return 0, 0
    key = next((k for k in ("mkv_written", "written_to_mkv") if k in rows[0]),
               None)
    if key is None:
        return len(rows), None
    return len(rows), sum(1 for r in rows if str(r.get(key, "")).strip() in
                          ("1", "true", "True"))


def diagnose_session(folder):
    """A whole recording folder: one verdict, and what to do about it.

    Returns a dict with `verdict` in:

      false_start   - headers only, nothing recorded. Nothing to recover from
                      the MKVs; the SVO, if the app wrote one before stopping,
                      is the only possible source.
      unfinalised   - frames are present but the container was never closed.
                      Remux to rebuild the index; no pixels are lost.
      readable      - the files look structurally fine, so whatever failed was
                      something else (a missing stream, an odd pixel format).
      no_video      - no video file in the folder at all.
    """
    folder = Path(folder)
    vids = sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    files = [diagnose_video_file(p) for p in vids]

    svos = sorted(p for p in folder.glob("*.svo*") if p.is_file())
    svo_bytes = sum(p.stat().st_size for p in svos)

    fcsv = folder / "frames.csv"
    rows, written = _frames_csv_counts(fcsv) if fcsv.exists() else (None, None)

    if not files:
        verdict = "no_video"
    elif any(f["verdict"] == "has_frame_data" for f in files):
        verdict = "unfinalised"
    elif all(f["verdict"] in ("header_only", "empty_file") for f in files):
        verdict = "false_start"
    else:
        verdict = "readable"

    return {"folder": folder, "verdict": verdict, "files": files,
            "svo_files": svos, "svo_bytes": svo_bytes,
            "frames_csv_rows": rows, "frames_csv_written": written,
            "recovery": recovery_advice(verdict, bool(svos), svo_bytes)}


def recovery_advice(verdict, has_svo, svo_bytes=0):
    """What to actually do. Deliberately concrete - a verdict with no next step
    just moves the question."""
    if verdict == "unfinalised":
        return ("remux to rebuild the index, then re-extract: "
                "ffmpeg -err_detect ignore_err -i <in>.mkv -c copy <out>.mkv "
                "(no re-encode, so no pixels change)")
    if verdict == "false_start":
        if has_svo and svo_bytes > 1_000_000:
            return (f"no frames in the MKVs; the SVO is {_mb(svo_bytes)}, so "
                    f"export it with the ZED SDK and extract that instead")
        return ("nothing was recorded - no frames in the MKVs and no usable "
                "SVO. Safe to delete; it is a false start, not lost data")
    if verdict == "no_video":
        return "not a recording folder"
    return ("structurally readable, so the probe failed on something else - "
            "run validation/inspect_depth_video.py on the depth stream")
