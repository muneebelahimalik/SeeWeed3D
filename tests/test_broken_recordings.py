"""Recordings that cannot be extracted, and what the run says about them.

Five of ten session folders failed on the October 2026 visit with nothing but
ffprobe's EBML parser output - which cannot tell a false start from an
unfinalised recording, and those have opposite answers: one is safe to delete
and the other holds every frame. These pin the classification, and pin that a
failed session leaves a record behind instead of only a scrollback line.
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import load_script

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from common.mkv_integrity import (  # noqa: E402
    diagnose_session, diagnose_video_file)


def _real_mkv(path, seconds=1):
    """A genuine FFV1 Matroska file, written by ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size=160x120:rate=15:duration={seconds}",
                    "-c:v", "ffv1", str(path)], check=True)
    return path


def _false_start(path):
    """The shape the field files have: a header, then the reserved seek space
    never filled in. ffmpeg leaves exactly this when no frame ever arrives -
    and those zeros are what ffprobe trips over at 'pos 110'."""
    good = Path(__file__).parent / "_tmp_real.mkv"
    _real_mkv(good)
    path.write_bytes(good.read_bytes()[:110] + b"\x00" * 5000)
    good.unlink()
    return path


# --------------------------------------------------------------------------- #
# File-level classification
# --------------------------------------------------------------------------- #

def test_a_false_start_is_reported_as_holding_no_frame_data(tmp_path):
    d = diagnose_video_file(_false_start(tmp_path / "RGB_left.mkv"))
    assert d["verdict"] == "header_only"
    assert "no frame data" in d["detail"]


def test_a_recording_with_frames_is_not_called_empty(tmp_path):
    """The dangerous mistake. An unfinalised file is byte-for-byte complete up
    to the cut; calling it empty throws away a whole drive."""
    d = diagnose_video_file(_real_mkv(tmp_path / "RGB_left.mkv"))
    assert d["verdict"] == "has_frame_data"


def test_frames_present_but_the_tail_cut_off_still_reads_as_frame_data(tmp_path):
    """A process killed mid-recording leaves clusters and no index. Every frame
    before the cut is recoverable by remux, so this must not be a false start."""
    full = _real_mkv(tmp_path / "full.mkv").read_bytes()
    assert full.find(b"\x1f\x43\xb6\x75") < 20_000, "fixture has no early cluster"
    cut = tmp_path / "RGB_left.mkv"
    cut.write_bytes(full[:len(full) // 2])
    assert diagnose_video_file(cut)["verdict"] == "has_frame_data"


def test_an_empty_file_is_distinguished_from_a_header(tmp_path):
    p = tmp_path / "RGB_left.mkv"
    p.write_bytes(b"")
    assert diagnose_video_file(p)["verdict"] == "empty_file"


def test_a_missing_file_does_not_raise(tmp_path):
    assert diagnose_video_file(tmp_path / "nope.mkv")["verdict"] == "missing"


def test_a_non_matroska_file_is_named_as_such(tmp_path):
    p = tmp_path / "RGB_video.avi"
    p.write_bytes(b"RIFF" + b"\x00" * 100)
    d = diagnose_video_file(p)
    assert d["verdict"] == "not_matroska" and "AVI" in d["detail"]


# --------------------------------------------------------------------------- #
# Session-level verdict and the advice that follows from it
# --------------------------------------------------------------------------- #

def _session(folder, kind, svo_bytes=0, rows=0):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("RGB_left.mkv", "Depth.mkv", "Confidence.mkv"):
        if kind == "false_start":
            _false_start(folder / name)
        else:
            _real_mkv(folder / name)
    with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "mkv_index", "mkv_written"])
        for i in range(rows):
            w.writerow([i, i, 1])
    if svo_bytes:
        (folder / "recording.svo2").write_bytes(b"\x00" * svo_bytes)
    return folder


def test_a_session_whose_videos_are_all_headers_is_a_false_start(tmp_path):
    d = diagnose_session(_session(tmp_path / "Session_20261016_104209",
                                 "false_start"))
    assert d["verdict"] == "false_start"
    assert d["frames_csv_rows"] == 0


def test_a_false_start_with_no_svo_is_declared_safe_to_delete(tmp_path):
    d = diagnose_session(_session(tmp_path / "Session_20261016_104237",
                                 "false_start"))
    assert "delete" in d["recovery"]
    assert "not lost data" in d["recovery"]


def test_a_false_start_with_a_real_svo_sends_you_to_the_svo(tmp_path):
    """The MKVs being empty does not mean the drive is lost: the SVO is written
    by the SDK on a separate path and may well have survived."""
    d = diagnose_session(_session(tmp_path / "Session_20261016_104550",
                                 "false_start", svo_bytes=4_000_000))
    assert "SVO" in d["recovery"] and "delete" not in d["recovery"]


def test_a_tiny_svo_is_not_mistaken_for_a_recoverable_archive(tmp_path):
    """An SVO that is only its own header is the same false start, one file
    along. Sending someone to export it wastes the afternoon."""
    d = diagnose_session(_session(tmp_path / "Session_20261016_112145",
                                 "false_start", svo_bytes=2048))
    assert "delete" in d["recovery"]


def test_a_session_with_frames_is_told_to_remux_not_to_discard(tmp_path):
    d = diagnose_session(_session(tmp_path / "Session_20261016_112306",
                                 "unfinalised", rows=30))
    assert d["verdict"] == "unfinalised"
    assert "remux" in d["recovery"] and "-c copy" in d["recovery"]
    assert d["frames_csv_rows"] == 30 and d["frames_csv_written"] == 30


# --------------------------------------------------------------------------- #
# What the extraction run does with one
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ex():
    return load_script("extraction/extract_sessions.py")


def test_probe_raises_a_probe_error_carrying_the_path(ex, tmp_path):
    """A bare RuntimeError with ffprobe's stderr in it cannot be acted on. The
    path is what lets the caller go and look at the bytes."""
    bad = _false_start(tmp_path / "RGB_left.mkv")
    with pytest.raises(ex.ProbeError) as e:
        ex.probe("ffprobe", bad)
    assert e.value.path == bad


def test_a_failed_session_is_explained_not_dumped(ex, tmp_path, capsys):
    folder = _session(tmp_path / "Session_20261016_104209", "false_start")
    row = ex.report_failure({"folder": folder},
                            ex.ProbeError(folder / "RGB_left.mkv", "Duplicate element"))
    out = capsys.readouterr().out
    assert "false_start" in out
    assert "no frame data" in out
    assert "delete" in out
    assert row["verdict"] == "false_start"
    assert row["recovery"] and row["folder"] == str(folder)


def test_a_failure_that_is_not_a_probe_error_still_gets_a_row(ex, tmp_path):
    """Anything else - a missing calibration, a disk full - must not vanish
    just because there is nothing to diagnose in the container."""
    row = ex.report_failure({"folder": tmp_path}, ValueError("no depth scale"))
    assert row["verdict"] == "error" and "no depth scale" in row["error"]


def test_failed_sessions_survive_the_run_in_a_csv(ex, tmp_path, capsys):
    """registry.csv has no row for a session that failed, so without this the
    record of a ten-folder run reads as a clean five-session success."""
    root = tmp_path / "raw"
    _session(root / "Session_20261016_104209", "false_start")
    _session(root / "Session_20261016_104237", "false_start")
    out = tmp_path / "out"

    ex.CONFIG.update({
        "INPUT_ROOTS": [{"path": str(root), "trip": "vid", "site": "s",
                         "field": "f", "scene_hint": "mixed", "notes": ""}],
        "OUTPUT_ROOT": str(out), "OVERWRITE": True, "DRY_RUN": False,
    })
    ex.main()
    printed = capsys.readouterr().out

    assert "2 of 2 session folder(s) FAILED" in printed
    rows = list(csv.DictReader(open(out / "extraction_failures.csv",
                                    encoding="utf-8")))
    assert len(rows) == 2
    assert {Path(r["folder"]).name for r in rows} == {
        "Session_20261016_104209", "Session_20261016_104237"}
    assert all(r["verdict"] == "false_start" for r in rows)
    assert all("delete" in r["recovery"] for r in rows)
    assert not (out / "registry.csv").exists()
