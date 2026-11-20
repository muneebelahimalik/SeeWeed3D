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
import subprocess
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
    """Stands in for pyzed. Same signature as open_svo, stride included - the
    fake has to skip frames the way the real reader seeks past them, or a
    strided export would look right here and write the wrong frames in the
    field."""
    def opener(svo_path, depth_mode=None, stride=1, positions=None):
        step = max(1, int(stride))
        taken = ([int(x) for x in positions] if positions is not None
                 else list(range(0, n, step)))
        info = {"width": W, "height": H, "fps": FPS, "n_frames": len(taken),
                "n_recorded": n, "stride": step,
                "depth_mode": depth_mode or "NEURAL", "sdk_version": "4.1.0"}

        def frames():
            for i in taken:
                t = ts(i) if ts else T0 + i * int(1e9 / FPS)
                yield t, _bgr(i), _depth_mm(i), _conf(i)
        return info, frames()
    return opener


def _session(folder, n=6, ts=None, with_meta=True, columns=None,
             dist=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "recording.svo2").write_bytes(b"\x00" * 1024)
    cols = columns or ["rec_frame_idx", "image_timestamp_ns", "tx_mm",
                       "distance_m", "pose_state"]
    with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            t = ts(i) if ts else T0 + i * int(1e9 / FPS)
            d = dist(i) if dist else round(i * 0.017, 4)
            w.writerow([i, t, i * 17.0, d, "OK"][:len(cols)])
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


# --------------------------------------------------------------------------- #
# Striding
#
# A 78 GB SVO is tens of thousands of depth computations to build a pool that
# keeps one frame in five. The frames in between are decoded, encoded, written
# and thrown away.
# --------------------------------------------------------------------------- #

def test_a_strided_export_writes_every_nth_frame_and_says_which(tmp_path):
    src = _session(tmp_path / "Session_20261029_180546", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(20), stride=5)
    assert r["frames"] == 4

    rows = list(csv.DictReader(
        open(r["out"] / "frames.csv", encoding="utf-8")))
    written = [r_ for r_ in rows if r_["mkv_written"] == "1"]
    assert [int(w["rec_frame_idx"]) for w in written] == [0, 5, 10, 15]
    assert [int(w["mkv_frame_idx"]) for w in written] == [0, 1, 2, 3], (
        "mkv_frame_idx is the position in the WRITTEN file, not in the SVO")
    # The skipped rows keep their place in the log and claim no frame.
    assert sum(1 for r_ in rows if r_["mkv_written"] == "0") == 16


def test_a_strided_export_records_the_factor_it_sampled_by(tmp_path):
    """Read as the whole recording, a sample is wrong by exactly this factor -
    in pool density, in metres between split boundaries, and in any frame
    count quoted as a result."""
    src = _session(tmp_path / "Session_20261029_180546", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(20), stride=5)
    exp = json.loads((r["out"] / "session_metadata.json").read_text())["export"]
    assert exp["svo_stride"] == 5
    assert exp["source_frames_recorded"] == 20
    assert exp["frames_written"] == 4


def test_the_written_frame_rate_matches_what_the_file_holds(tmp_path):
    """Tagging a 1-in-5 export at the capture rate makes the container claim a
    duration five times shorter than the ground it covers - and extraction
    derives its own pool stride from that number."""
    src = _session(tmp_path / "Session_20261029_180546", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(20), stride=5)
    exp = json.loads((r["out"] / "session_metadata.json").read_text())["export"]
    assert exp["source_fps"] == FPS
    assert exp["written_fps"] == pytest.approx(FPS / 5)

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-of", "json", str(r["out"] / "RGB_left.mkv")],
        capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    num, den = d["streams"][0]["r_frame_rate"].split("/")
    assert float(num) / float(den) == pytest.approx(FPS / 5), (
        "the container must be tagged at the rate it actually holds")
    assert float(d["format"]["duration"]) == pytest.approx(
        20 / FPS, abs=0.5), "duration must still span the real elapsed time"


def test_stride_one_is_unchanged(tmp_path):
    src = _session(tmp_path / "Session_20261029_180546", n=8)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(8), stride=1)
    assert r["frames"] == 8 and r["matched"] == 8
    exp = json.loads((r["out"] / "session_metadata.json").read_text())["export"]
    assert exp["svo_stride"] == 1 and exp["written_fps"] == FPS


def test_a_strided_export_still_round_trips_through_the_extractor(tmp_path):
    """The sampled session must be an ordinary v3 session, with its timestamps
    still on the right frames."""
    src = _session(tmp_path / "Session_20261029_180546", n=20)
    ex_svo.export_session(src, tmp_path / "exported",
                          open_source=_fake_svo(20), stride=5)

    ex = load_script("extraction/extract_sessions.py")
    cfg = dict(ex.CONFIG, OUTPUT_ROOT=str(tmp_path / "dataset"),
               DRY_RUN=False, OVERWRITE=True, POOL_STRIDE=1,
               MAX_POOL_PER_SESSION=None)
    cfg["INPUT_ROOTS"] = [{"path": str(tmp_path / "exported"), "trip": "jet",
                           "site": "s", "field": "f", "scene_hint": "mixed",
                           "notes": ""}]
    rec = ex.extract_session(ex.discover(cfg)[0], tmp_path / "dataset", cfg)
    assert rec["capture_format"] == "v3"
    assert rec["frames"]["decoded"] == 4

    idx = list(csv.DictReader(open(
        tmp_path / "dataset" / "sessions" / rec["session_id"] / "meta"
        / "frames_index.csv", encoding="utf-8")))
    assert [int(r["timestamp_ns"]) for r in idx] == [
        T0 + i * int(1e9 / FPS) for i in (0, 5, 10, 15)]


# --------------------------------------------------------------------------- #
# Frames the SDK cannot decode
#
# The real recordings carry a handful by design: session_metadata reports
# `svo_frames_failed: 5` of 727, and frames.csv flags each with
# svo_frame_ok=0. Aborting on one would stop a 14,000-frame export on the
# fifth bad frame and leave the other 13,995 unexported.
# --------------------------------------------------------------------------- #

def _flaky_svo(n=20, bad_positions=(3, 7, 11)):
    """A reader that reports some frames as unreadable, as pyzed does."""
    def opener(svo_path, depth_mode=None, stride=1, positions=None):
        step = max(1, int(stride))
        info = {"width": W, "height": H, "fps": FPS, "stride": step,
                "n_frames": len(range(0, n, step)), "n_recorded": n,
                "depth_mode": depth_mode or "NEURAL_LIGHT",
                "sdk_version": "5.2.3"}
        bad = info["unreadable_frames"] = []

        def frames():
            for i in range(0, n, step):
                if i in bad_positions:
                    bad.append((i, "ERROR_CODE.CORRUPTED_FRAME"))
                    continue
                yield (T0 + i * int(1e9 / FPS), _bgr(i), _depth_mm(i),
                       _conf(i))
        return info, frames()
    return opener


def test_an_unreadable_frame_is_skipped_not_fatal(tmp_path):
    src = _session(tmp_path / "Session_20261021_123954", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_flaky_svo(20, (3, 7, 11)))
    assert r["frames"] == 17, "the other 17 frames must still be exported"
    assert r["matched"] == 17


def test_the_skipped_frames_are_counted_in_the_metadata(tmp_path):
    """The capture's own metadata reports the same number as
    `svo_frames_failed`, so the two can be compared. A silent skip could
    not be."""
    src = _session(tmp_path / "Session_20261021_123954", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_flaky_svo(20, (3, 7, 11)))
    exp = json.loads((r["out"] / "session_metadata.json").read_text())["export"]
    assert exp["unreadable_frames"] == 3
    assert exp["frames_written"] == 17


def test_a_skipped_frame_does_not_shift_the_rows_after_it(tmp_path):
    """The whole reason the join is on timestamps. A skipped frame must leave
    its own row unwritten, not hand its timestamp to its neighbour."""
    src = _session(tmp_path / "Session_20261021_123954", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_flaky_svo(20, (3, 7, 11)))
    rows = list(csv.DictReader(open(r["out"] / "frames.csv", encoding="utf-8")))
    by_rec = {int(x["rec_frame_idx"]): x for x in rows}

    for skipped in (3, 7, 11):
        assert by_rec[skipped]["mkv_written"] == "0"
        assert by_rec[skipped]["mkv_frame_idx"] == ""
    # And the frame after a skip points at its own place in the written file.
    assert int(by_rec[4]["mkv_frame_idx"]) == 3      # 0,1,2 written, 3 skipped
    assert int(by_rec[8]["mkv_frame_idx"]) == 6


def test_the_written_stream_matches_the_capture_apps_own_ffv1_settings(tmp_path):
    """-g 1, 24 slices, slice CRCs. The slices are what let the decoder use
    more than one core on a 2208x1242 frame; the CRCs make a corrupted read
    detectable rather than plausible."""
    src = _session(tmp_path / "Session_20261021_123954", n=3)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_fake_svo(3))
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-of", "json", str(r["out"] / "RGB_left.mkv")],
        capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    assert s["codec_name"] == "ffv1"
    assert "-slices" in ex_svo.FfvWriter.FFV1_ARGS
    assert ex_svo.FfvWriter.FFV1_ARGS[
        ex_svo.FfvWriter.FFV1_ARGS.index("-slices") + 1] == "24"
    assert "-slicecrc" in ex_svo.FfvWriter.FFV1_ARGS


# --------------------------------------------------------------------------- #
# Sampling by metres of ground
#
# 886 m of row across eight drives, pushed by hand at 0.17-0.26 m/s. A fixed
# stride is a statement about the recording; a dataset needs one about the
# field, and the two part company every time the operator slows down.
# --------------------------------------------------------------------------- #

def test_frames_are_chosen_by_distance_travelled(tmp_path):
    rows = [{"rec_frame_idx": i, "distance_m": i * 0.05} for i in range(100)]
    got = ex_svo.select_positions_by_distance(rows, 0.25)
    assert got[0] == 0, "the first frame of a drive is always taken"
    assert got[:5] == [0, 5, 10, 15, 20]


def test_a_stall_does_not_produce_a_hundred_pictures_of_one_spot(tmp_path):
    """The failure a stride cannot see. Two of these ten sessions travelled
    one centimetre in 48 seconds; within a drive the same thing happens
    whenever the operator stops to move a cable."""
    rows = ([{"rec_frame_idx": i, "distance_m": i * 0.05} for i in range(10)]
            + [{"rec_frame_idx": 10 + i, "distance_m": 0.5}       # stalled
               for i in range(200)]
            + [{"rec_frame_idx": 210 + i, "distance_m": 0.5 + i * 0.05}
               for i in range(10)])
    got = ex_svo.select_positions_by_distance(rows, 0.25)

    stalled = [g for g in got if 10 <= g < 210]
    assert stalled == [10], (
        f"{len(stalled)} frames taken from a 200-frame stall. One is right - "
        f"frame 10 sits a full 0.25 m past its predecessor and is a genuinely "
        f"new place. The other 199 are the same spot, and a stride of 5 would "
        f"have taken 40 of them")
    assert len(got) < 10, f"{len(got)} frames for 1 m of row"


def test_rows_without_odometry_are_skipped_not_guessed(tmp_path):
    rows = [{"rec_frame_idx": 0, "distance_m": 0.0},
            {"rec_frame_idx": 1, "distance_m": ""},
            {"rec_frame_idx": 2, "distance_m": "nonsense"},
            {"rec_frame_idx": 3, "distance_m": 0.4}]
    assert ex_svo.select_positions_by_distance(rows, 0.25) == [0, 3]


def test_an_export_can_be_planned_in_metres(tmp_path):
    """17 mm per frame in the fixture, so 0.25 m is every 15th frame."""
    src = _session(tmp_path / "Session_20261106_134659", n=60)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(60), metres=0.25)
    assert r["frames"] == 4           # frames 0, 15, 30, 45
    assert r["matched"] == 4

    rows = list(csv.DictReader(open(r["out"] / "frames.csv", encoding="utf-8")))
    written = [x for x in rows if x["mkv_written"] == "1"]
    assert [int(x["rec_frame_idx"]) for x in written] == [0, 15, 30, 45]


def test_the_metre_plan_is_recorded_beside_the_frame_count(tmp_path):
    """A dataset built from a 0.25 m plan is a different object from one built
    at a fixed stride, even when both hold the same number of frames."""
    src = _session(tmp_path / "Session_20261106_134659", n=60)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(60), metres=0.25)
    exp = json.loads((r["out"] / "session_metadata.json").read_text())["export"]
    assert exp["metres_between_frames"] == 0.25
    assert exp["effective_stride"] == pytest.approx(15.0, abs=0.1)


def test_a_metre_plan_over_a_drive_that_stalled_takes_almost_nothing(tmp_path):
    """Session_20261021_123954: 727 frames, 1 cm of travel. Exporting it on a
    frame count would put 145 pictures of one spot into the pool."""
    src = _session(tmp_path / "Session_20261021_123954", n=100,
                   dist=lambda i: 0.0001 * i)        # 0.1 mm per frame
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(100), metres=0.25)
    assert r["frames"] == 1, (
        f"{r['frames']} frames from a drive that covered 1 cm")


def test_a_log_with_no_odometry_at_all_refuses_rather_than_striding(tmp_path):
    src = _session(tmp_path / "Session_20261106_134659", n=20,
                   columns=["rec_frame_idx", "image_timestamp_ns", "tx_mm"])
    with pytest.raises(SystemExit) as e:
        ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(20), metres=0.25)
    assert "distance_m" in str(e.value) and "--stride" in str(e.value)


def test_a_metre_planned_export_round_trips_through_the_extractor(tmp_path):
    src = _session(tmp_path / "Session_20261106_134659", n=60)
    ex_svo.export_session(src, tmp_path / "exported",
                          open_source=_fake_svo(60), metres=0.25)

    ex = load_script("extraction/extract_sessions.py")
    cfg = dict(ex.CONFIG, OUTPUT_ROOT=str(tmp_path / "dataset"),
               DRY_RUN=False, OVERWRITE=True, POOL_STRIDE=1,
               MAX_POOL_PER_SESSION=None)
    cfg["INPUT_ROOTS"] = [{"path": str(tmp_path / "exported"), "trip": "jet",
                           "site": "s", "field": "f", "scene_hint": "mixed",
                           "notes": ""}]
    rec = ex.extract_session(ex.discover(cfg)[0], tmp_path / "dataset", cfg)
    assert rec["capture_format"] == "v3"
    assert rec["frames"]["decoded"] == 4

    idx = list(csv.DictReader(open(
        tmp_path / "dataset" / "sessions" / rec["session_id"] / "meta"
        / "frames_index.csv", encoding="utf-8")))
    assert [r["distance_m"] for r in idx] == [
        str(round(i * 0.017, 4)) for i in (0, 15, 30, 45)]


# --------------------------------------------------------------------------- #
# Planning without the SDK
# --------------------------------------------------------------------------- #

def _no_sdk(*a, **k):
    raise SystemExit("pyzed is not installed.")


def test_the_plan_is_answerable_with_no_sdk_and_no_gpu(tmp_path):
    """How many frames a spacing yields comes from frames.csv alone. Making
    that wait for a GB-scale SDK install, to answer a question about a CSV, is
    the wrong way round - and it is the question you want settled BEFORE
    committing hours of matcher time."""
    src = _session(tmp_path / "Session_20261106_142555", n=60)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_no_sdk,
                              metres=0.25, plan_only=True)
    assert r["plan_only"] is True
    assert r["frames"] == 4
    assert r["rows"] == 60


def test_a_plan_writes_nothing_at_all(tmp_path):
    src = _session(tmp_path / "Session_20261106_142555", n=60)
    ex_svo.export_session(src, tmp_path / "out", open_source=_no_sdk,
                          metres=0.25, plan_only=True)
    assert not (tmp_path / "out" / src.name).exists(), (
        "a plan must not leave an empty session folder that later looks "
        "like a failed export")


def test_a_plan_reports_the_distance_the_drive_actually_covered(tmp_path):
    """The number that separates a 47 m drive from 48 seconds of standing
    still, and the two are indistinguishable by frame count."""
    src = _session(tmp_path / "Session_20261021_123954", n=100,
                   dist=lambda i: 0.0001 * i)
    r = ex_svo.export_session(src, tmp_path / "out", open_source=_no_sdk,
                              metres=0.25, plan_only=True)
    assert r["span_m"] == pytest.approx(0.0099, abs=1e-3)
    assert r["frames"] == 1


def test_a_reader_that_cannot_open_leaves_no_husk_behind(tmp_path):
    """Running the export before the SDK is installed used to create the
    session folder and then fail, leaving an empty directory that looks like
    an export which ran and produced nothing - and that a later discover()
    would find and report as a broken session."""
    src = _session(tmp_path / "Session_20261029_182236", n=20)
    with pytest.raises(SystemExit):
        ex_svo.export_session(src, tmp_path / "out", open_source=_no_sdk,
                              metres=0.25)
    assert not (tmp_path / "out" / src.name).exists()


def test_a_successful_export_still_creates_its_folder(tmp_path):
    src = _session(tmp_path / "Session_20261029_182236", n=20)
    r = ex_svo.export_session(src, tmp_path / "out",
                              open_source=_fake_svo(20), metres=0.25)
    assert (r["out"] / "RGB_left.mkv").exists()
