"""End-to-end checks for extract_sessions.py and select_batches.py against
synthetic v1/v2 sessions built with real FFV1 MKVs (see conftest.py)."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from conftest import load_script


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_cvat_labels_every_attribute_has_nonempty_values():
    """CVAT's Raw label editor rejects the whole schema on paste if any
    attribute - including 'text' ones, where CVAT still expects the default
    wrapped in a single-element list - is missing 'values' or has it empty.
    Regression guard for the same bug class that broke weed_cvat_labels.json."""
    s2 = load_script("extraction/select_batches.py")
    for label in s2.CVAT_LABELS:
        for attr in label["attributes"]:
            assert "values" in attr, (
                f"{label['name']}.{attr['name']} has no 'values' key")
            assert isinstance(attr["values"], list) and len(attr["values"]) > 0, (
                f"{label['name']}.{attr['name']} has an empty 'values' array")


def test_registry_detects_both_formats(extracted_root):
    reg = {r["session_id"]: r for r in _rows(extracted_root / "registry.csv")}
    assert len(reg) == 3
    assert reg["vid_20250304_142804"]["capture_format"] == "v1"
    assert reg["vid_20260108_103135"]["capture_format"] == "v2"
    assert reg["vid_20260108_103135"]["has_confidence"] == "True"
    assert reg["vid_20260108_103135"]["has_right"] == "True"
    assert reg["vid_20250304_142804"]["has_confidence"] == "False"


def test_depth_lossless_and_index_aligned(extracted_root, raw_root):
    for sid, truth in raw_root["truth"].items():
        sdir = extracted_root / "sessions" / sid
        for r in _rows(sdir / "meta" / "pool.csv"):
            v = int(r["video_frame_idx"])
            d = cv2.imread(str(sdir / "depth" / r["filename"]), cv2.IMREAD_UNCHANGED)
            assert d is not None and d.dtype == np.uint16
            assert int(d.sum()) == truth[v]["depth_sum"]
            assert int(d[d.shape[0] // 2, d.shape[1] // 2]) == truth[v]["depth_at"]


def test_dropped_frame_index_recovered(extracted_root, raw_root):
    sid = "vid_20260108_103135"
    pool = _rows(extracted_root / "sessions" / sid / "meta" / "pool.csv")
    truth = raw_root["truth"][sid]
    for r in pool:
        assert int(r["capture_frame_idx"]) == truth[int(r["video_frame_idx"])]["capture_idx"]
    assert 7 not in {int(r["capture_frame_idx"]) for r in pool}   # the dropped one


def test_confidence_sentinel_excluded(extracted_root):
    idx = _rows(extracted_root / "sessions" / "vid_20260108_103135" / "meta" / "frames_index.csv")
    means = [float(r["conf_mean"]) for r in idx if r.get("conf_mean") not in ("", None)]
    assert means and max(means) <= 100     # 255 sentinel removed


def test_depth_not_rescaled_and_calibration_parsed(extracted_root):
    sdir = extracted_root / "sessions" / "vid_20250304_142804"
    pool = _rows(sdir / "meta" / "pool.csv")
    d = cv2.imread(str(sdir / "depth" / pool[0]["filename"]), cv2.IMREAD_UNCHANGED)
    assert 700 < int(np.median(d[d > 0])) < 1100      # raw mm, not ×3000
    calib = json.loads((sdir / "meta" / "calibration.json").read_text())
    assert abs(calib["mm_per_px_at_1000mm"] - 1000 / 350) < 1e-3


def _select(extracted_root, holdout):
    s2 = load_script("extraction/select_batches.py")
    s2.CONFIG.update({
        "DATASET_ROOT": str(extracted_root), "HOLDOUT_SESSIONS": holdout,
        "GATES": {"min_sharpness": 5.0, "min_veg_frac": 0.005, "max_clip_frac": 0.95,
                  "max_dark_frac": 0.95, "min_depth_valid_frac_veg": 0.05},
        "MIN_PHASH_DISTANCE": 2, "MAX_SESSION_SHARE": 0.9,
        "BATCHES": [{"name": "b01", "n": 10, "stress_fraction": 0.3, "pool": "train"},
                    {"name": "b_test", "n": 6, "stress_fraction": 0.2, "pool": "holdout"}],
    })
    s2.main()

    def files(name):
        d = extracted_root / "batches" / name / "images"
        return sorted(p.name for p in d.iterdir()) if d.exists() else []
    return files


def test_batches_respect_holdout_and_reproducible(extracted_root):
    holdout = ["vid_20250305_090000"]
    files = _select(extracted_root, holdout)
    train, test = files("b01"), files("b_test")
    assert train and test
    assert all(not f.startswith("vid_20250305_090000") for f in train)  # no leak
    assert all(f.startswith("vid_20250305_090000") for f in test)
    assert not (set(train) & set(test))                                 # disjoint

    import shutil
    shutil.rmtree(extracted_root / "batches")
    files2 = _select(extracted_root, holdout)
    assert files2("b01") == train and files2("b_test") == test          # stable seed


# --------------------------------------------------------------------------- #
# Container discovery
#
# One field campaign wrote AVI in its early sessions and MKV in its later ones,
# under the same parent folder. The patterns carried ".mkv", so the AVI
# sessions were not merely skipped - discover() returned nothing for them AND
# printed nothing, because the "looks like a recording" check used the same
# extension list that had just failed. A run over the mixed folder looked like
# a success while extracting two sessions out of nine.
# --------------------------------------------------------------------------- #
ex = load_script("extraction/extract_sessions.py")


def _fake_session(root, name, ext, files=("RGB_video", "Depth_video")):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f"{f}{ext}").write_bytes(b"stub")
    (d / "calibration_params.txt").write_text("Left Camera Intrinsic\nfx: 700, fy: 700\ncx: 640, cy: 360\n")
    return d


def _cfg(root, **over):
    return dict(ex.CONFIG, INPUT_ROOTS=[{
        "path": str(root), "trip": "vid1", "site": "vidalia",
        "field": "field_A", "scene_hint": "mixed", "notes": ""}], **over)


def test_avi_and_mkv_sessions_are_both_discovered(tmp_path):
    _fake_session(tmp_path, "Session_20250221_130957", ".avi")
    _fake_session(tmp_path, "Session_20250226_202127", ".mkv")
    got = {s["folder"].name: s for s in ex.discover(_cfg(tmp_path))}
    assert set(got) == {"Session_20250221_130957", "Session_20250226_202127"}
    assert got["Session_20250221_130957"]["rgb"].suffix == ".avi"
    assert got["Session_20250221_130957"]["depth"].suffix == ".avi"


def test_the_right_stream_is_still_never_taken_for_the_left(tmp_path):
    """The stem now carries the match, so the 'right' guard has to survive the
    change - RGB_right_video is a different camera, not a fallback."""
    d = _fake_session(tmp_path, "Session_20250221_130957", ".avi",
                      files=("RGB_video", "RGB_right_video"))
    assert ex.find_one(d, ex.CONFIG["RGB_PATTERNS"]).name == "RGB_video.avi"
    assert ex.find_one(d, ex.CONFIG["RIGHT_PATTERNS"],
                       want_right=True).name == "RGB_right_video.avi"


def test_a_recording_in_an_unknown_container_is_reported_not_swallowed(
        tmp_path, capsys):
    _fake_session(tmp_path, "Session_20250221_130957", ".wmv")
    assert ex.discover(_cfg(tmp_path)) == []
    out = capsys.readouterr().out
    assert "SKIP" in out and ".wmv" in out and "VIDEO_SUFFIXES" in out


def test_an_ordinary_folder_stays_quiet(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "readme.txt").write_text("hi")
    assert ex.discover(_cfg(tmp_path)) == []


# --------------------------------------------------------------------------- #
# Depth integrity
#
# The decoder asks ffmpeg for gray16le, and ffmpeg produces gray16le from ANY
# input - including 8-bit lossy, by scaling up values it invented. The output
# is a valid PNG full of plausible millimetres that are fiction, and nothing
# downstream can tell.
# --------------------------------------------------------------------------- #
def test_sixteen_bit_depth_passes():
    assert ex.depth_precision_problem({"pix_fmt": "gray16le",
                                       "codec": "ffv1"}) is None


def test_eight_bit_depth_is_refused_however_it_is_packaged():
    for codec, pf in (("mjpeg", "yuvj420p"), ("mpeg4", "yuv420p"),
                      ("ffv1", "gray"), ("rawvideo", "bgr24")):
        why = ex.depth_precision_problem({"pix_fmt": pf, "codec": codec})
        assert why and "not 16-bit" in why
        assert pf in why and codec in why


def test_the_container_decides_nothing_by_itself():
    """FFV1 in AVI is fine; a badly remuxed MKV is not. Judging by extension
    would reject good sessions and accept bad ones."""
    assert ex.depth_precision_problem({"pix_fmt": "gray16le",
                                       "codec": "ffv1"}) is None
    assert ex.depth_precision_problem({"pix_fmt": "yuv420p",
                                       "codec": "h264"}) is not None


def test_a_stream_with_no_pixel_format_is_not_assumed_good():
    why = ex.depth_precision_problem({"pix_fmt": None, "codec": "mjpeg"})
    assert why and "cannot be confirmed" in why


def test_the_guard_says_what_it_costs_and_how_to_override():
    """A refusal that does not name its escape hatch gets worked around by
    editing the check out, which is worse than the flag."""
    src = (Path(ex.__file__).read_text(encoding="utf-8")
           if hasattr(ex, "__file__") else "")
    assert "REQUIRE_16BIT_DEPTH" in src
    assert ex.CONFIG["REQUIRE_16BIT_DEPTH"] is True
