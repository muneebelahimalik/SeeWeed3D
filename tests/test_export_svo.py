"""Exporting an SVO-only recording into an extractable v3 session.

The export is the one step where this project becomes the ENCODER rather than
the reader, so every silent corruption the v3 reader guards against is a
corruption this code could introduce instead: a depth scale that disagrees
with the metadata beside it, a sentinel written over a real distance,
confidence whose undefined value reads as perfect, a frames.csv attached by
position.

The decisive test is the round trip - fake camera -> export -> the real
extractor -> millimetres - because that is the only one that fails if the
writer and the reader agree with each other and both are wrong.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import load_script

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from extraction import export_svo as ex_svo  # noqa: E402

W, H, FPS = 64, 48, 15
T0 = 1_700_000_000_000_000_000


def _depth_mm(i):
    """Float millimetres as the SDK hands them over, failure codes included."""
    d = np.full((H, W), 900.0 + i, dtype=np.float32)
    d[0, 0] = np.nan            # occluded - no measurement
    d[0, 1] = -np.inf           # nearer than the sensor minimum
    d[0, 2] = np.inf            # past the maximum
    d[1, 0] = 7000.0            # valid, but past DEPTH_MAX_COUNT at 10/mm
    return d


def _conf(i):
    c = np.full((H, W), 12.0, dtype=np.float32)
    c[0, 0] = np.nan            # undefined
    return c


def _bgr(i):
    rng = np.random.default_rng(i)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def _fake_svo(n=6, ts=None):
    """Stands in for pyzed. Same signature as open_svo."""
    def opener(svo_path, depth_mode=None):
        info = {"width": W, "height": H, "fps": FPS, "n_frames": n,
                "depth_mode": depth_mode or "NEURAL", "sdk_version": "4.1.0"}

        def frames():
            for i in range(n):
                t = ts(i) if ts else T0 + i * int(1e9 / FPS)
                yield t, _bgr(i), _depth_mm(i), _conf(i)
        return info, frames()
    return opener


def _session(folder, n=6, ts=None, with_meta=True, columns=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "recording.svo2").write_bytes(b"\x00" * 1024)
    cols = columns or ["rec_frame_idx", "image_timestamp_ns", "tx_mm",
                       "distance_m", "pose_state"]
    with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            t = ts(i) if ts else T0 + i * int(1e9 / FPS)
            w.writerow([i, t, i * 17.0, round(i * 0.017, 4), "OK"][:len(cols)])
    if with_meta:
        (folder / "session_metadata.json").write_text(json.dumps(
            {"status": "COMPLETE", "capture": {"fps": FPS,
                                               "depth_mode": "NEURAL"}}))
    (folder / "calibration.json").write_text(json.dumps(
        {"serial_number": 1, "camera_model": "ZED 2i", "resolution": [W, H],
         "fps": FPS,
         "calibration_rectified": {
             "left_cam": {"fx": 350.0, "fy": 350.0, "cx": 32.0, "cy": 24.0,
                          "disto": [0] * 5, "h_fov_deg": 80.0,
                          "image_size": [W, H]},
             "right_cam": {"fx": 350.0, "fy": 350.0, "cx": 32.0, "cy": 24.0,
                           "disto": [0] * 5},
             "baseline": 120.0,
             "stereo_transform": {"translation": [-120.0, 0, 0],
                                  "rotation_vector_rad": [0, 0, 0]}}}))
    (folder / "markers.csv").write_text("rec_frame_idx,text\n0,start row 1\n")
    (folder / "README.txt").write_text("jetson capture\n")
    return folder


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def test_the_three_sdk_failure_codes_stay_distinguishable():
    """NaN, -inf and +inf mean occluded, too close and too far. Collapsing them
    into one 'invalid' throws away the only evidence of WHY a region has no
    depth - a camera mounted too low looks exactly like a dense canopy."""
    out = ex_svo.encode_depth_mm(
        np.array([[np.nan, -np.inf, np.inf, 1000.0]], np.float32), 10)
    assert list(out[0]) == [0, 65534, 65535, 10000]


def test_a_far_but_valid_reading_is_never_written_as_a_sentinel():
    """At 10 counts/mm anything past 6553.3 mm overflows into the sentinel
    range. Clipping is a known small error; landing on 65535 would be read
    back as 'too far' and then as invalid - data removed, silently."""
    out = ex_svo.encode_depth_mm(np.array([[7000.0, 99999.0]], np.float32), 10)
    assert out[0, 0] == ex_svo.DEPTH_MAX_COUNT
    assert out[0, 1] == ex_svo.DEPTH_MAX_COUNT
    assert ex_svo.DEPTH_MAX_COUNT < ex_svo.DEPTH_TOO_CLOSE


def test_zero_and_negative_depth_are_invalid_not_tiny():
    out = ex_svo.encode_depth_mm(np.array([[0.0, -5.0, 0.04]], np.float32), 10)
    assert out[0, 0] == 0 and out[0, 1] == 0
    assert out[0, 2] == 1, "a real sub-millimetre reading must not become 0"


def test_undefined_confidence_maps_to_the_worst_value_not_the_best():
    """Confidence here is inverted: higher is worse. An undefined pixel read as
    0 would enter the dataset with full marks."""
    out = ex_svo.encode_confidence(np.array([[np.nan, 0.0, 100.0]], np.float32),
                                   100)
    assert out[0, 0] == ex_svo.CONF_UNDEFINED
    assert out[0, 1] == 0 and out[0, 2] == 10000


# --------------------------------------------------------------------------- #
# The join
# --------------------------------------------------------------------------- #

def test_rows_are_matched_by_timestamp_not_by_position(tmp_path):
    """frames.csv given to the exporter out of order, and with an extra row
    that has no frame. Pairing by position would mislabel everything."""
    src = _session(tmp_path / "Session_20261021_123954", n=4)
    rows = list(csv.DictReader(open(src / "frames.csv", encoding="utf-8")))
    rows = [rows[2], rows[0], rows[3], rows[1]]
    with open(src / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(4))
    out = list(csv.DictReader(
        open(tmp_path / "out" / src.name / "frames.csv", encoding="utf-8")))

    by_rec = {int(r["rec_frame_idx"]): r for r in out}
    for rec in range(4):
        assert by_rec[rec]["mkv_written"] == "1"
        assert int(by_rec[rec]["mkv_frame_idx"]) == rec, (
            "frame order in the MKV must follow the SVO, and each row must "
            "point at its own frame")


def test_an_svo_whose_log_belongs_to_another_session_stops_early(tmp_path):
    """The failure that must not take forty minutes to surface."""
    src = _session(tmp_path / "Session_20261021_123954", n=20)
    with pytest.raises(SystemExit) as e:
        ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(20, ts=lambda i: 42 + i))
    assert "different sessions" in str(e.value)


def test_a_log_with_no_timestamp_column_is_refused_before_any_work(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=4,
                   columns=["rec_frame_idx", "tx_mm"])
    with pytest.raises(SystemExit) as e:
        ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(4))
    assert "image_timestamp_ns" in str(e.value)
    assert not (tmp_path / "out" / src.name / "RGB_left.mkv").exists()


def test_the_source_folder_is_never_written_to(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=4)
    before = {p.name: p.stat().st_mtime_ns for p in src.iterdir()}
    ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(4))
    after = {p.name: p.stat().st_mtime_ns for p in src.iterdir()}
    assert before == after, "the archive drive must come back untouched"


# --------------------------------------------------------------------------- #
# What the exported session declares about itself
# --------------------------------------------------------------------------- #

def test_the_export_declares_the_depth_scale_it_actually_wrote(tmp_path):
    """extract_sessions REFUSES a v3 session that does not state its own scale,
    because the alternative is a ten-fold depth field that looks reasonable.
    The exporter is the encoder here, so it must say."""
    src = _session(tmp_path / "Session_20261021_123954", n=4, with_meta=False)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(4),
                              units_per_mm=10)
    meta = json.loads((r["out"] / "session_metadata.json").read_text())
    assert meta["encoding"]["depth_units_per_mm"] == 10
    assert meta["encoding"]["depth_codes"]["too_far"] == 65535
    assert meta["encoding"]["confidence_scale"] == 100


def test_a_non_default_scale_is_written_not_assumed(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=3)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(3),
                              units_per_mm=1)
    meta = json.loads((r["out"] / "session_metadata.json").read_text())
    assert meta["encoding"]["depth_units_per_mm"] == 1


def test_the_depth_mode_and_sdk_version_are_recorded(tmp_path):
    """Depth is RECOMPUTED at export, so it is a property of this run, not of
    the recording. Two exports under different modes are different
    measurements of the same field and must not be silently comparable."""
    src = _session(tmp_path / "Session_20261021_123954", n=3)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(3),
                              depth_mode="ULTRA")
    exp = json.loads((r["out"] / "session_metadata.json").read_text())["export"]
    assert exp["depth_mode"] == "ULTRA"
    assert exp["sdk_version"] == "4.1.0"
    assert exp["source_svo"].endswith("recording.svo2")


def test_a_partial_run_says_so(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=10)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(10),
                              limit=3)
    assert r["frames"] == 3
    assert json.loads((r["out"] / "session_metadata.json").read_text()
                      )["export"]["partial"] is True


def test_the_sidecars_travel_with_the_export(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=3)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(3))
    for name in ("calibration.json", "markers.csv", "README.txt"):
        assert (r["out"] / name).exists(), f"{name} did not travel"


# --------------------------------------------------------------------------- #
# The round trip: export -> the real extractor -> millimetres
# --------------------------------------------------------------------------- #

def test_exported_depth_comes_back_as_the_millimetres_it_started_as(tmp_path):
    """The one test that fails if the writer and the reader agree with each
    other and are both wrong. Nothing else here would notice."""
    src = _session(tmp_path / "Session_20261021_123954", n=6)
    ex_svo.export_session(src, tmp_path / "exported", open_source=_fake_svo(6))

    ex = load_script("extraction/extract_sessions.py")
    cfg = dict(ex.CONFIG, OUTPUT_ROOT=str(tmp_path / "dataset"),
               DRY_RUN=False, OVERWRITE=True, POOL_STRIDE=1,
               MAX_POOL_PER_SESSION=None)
    cfg["INPUT_ROOTS"] = [{"path": str(tmp_path / "exported"), "trip": "jet",
                           "site": "s", "field": "f", "scene_hint": "mixed",
                           "notes": ""}]
    sessions = ex.discover(cfg)
    assert len(sessions) == 1, "the exported folder must look like a session"
    rec = ex.extract_session(sessions[0], tmp_path / "dataset", cfg)

    assert rec["capture_format"] == "v3"
    sdir = tmp_path / "dataset" / "sessions" / rec["session_id"]
    png = sorted((sdir / "depth").glob("*.png"))
    assert len(png) == 6

    got = cv2.imread(str(png[0]), cv2.IMREAD_UNCHANGED)
    assert got.dtype == np.uint16
    want = _depth_mm(0)
    # The ordinary pixels: millimetres in, millimetres out.
    assert got[H // 2, W // 2] == pytest.approx(want[H // 2, W // 2], abs=1)
    # And the codes, which must arrive as invalid rather than as 6.5 metres.
    assert got[0, 0] == 0 and got[0, 1] == 0 and got[0, 2] == 0


def test_the_round_trip_carries_the_timestamps_and_pose(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=5)
    ex_svo.export_session(src, tmp_path / "exported", open_source=_fake_svo(5))

    ex = load_script("extraction/extract_sessions.py")
    cfg = dict(ex.CONFIG, OUTPUT_ROOT=str(tmp_path / "dataset"),
               DRY_RUN=False, OVERWRITE=True, POOL_STRIDE=1,
               MAX_POOL_PER_SESSION=None)
    cfg["INPUT_ROOTS"] = [{"path": str(tmp_path / "exported"), "trip": "jet",
                           "site": "s", "field": "f", "scene_hint": "mixed",
                           "notes": ""}]
    rec = ex.extract_session(ex.discover(cfg)[0], tmp_path / "dataset", cfg)
    assert rec["frames"]["timestamp_source"] == "frames_csv_v3"

    idx = list(csv.DictReader(open(
        tmp_path / "dataset" / "sessions" / rec["session_id"] / "meta"
        / "frames_index.csv", encoding="utf-8")))
    assert [int(r["timestamp_ns"]) for r in idx] == [
        T0 + i * int(1e9 / FPS) for i in range(5)]
    assert [r["distance_m"] for r in idx] == [
        str(round(i * 0.017, 4)) for i in range(5)]
