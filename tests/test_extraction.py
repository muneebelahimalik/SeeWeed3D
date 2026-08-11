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


# --------------------------------------------------------------------------- #
# Recovering an 8-bit depth PREVIEW
#
# Some v1 captures wrote the GUI's depth preview instead of the data: the range
# scaled to 0..255 against depth_vis_max_mm, optionally colourised, then lossily
# encoded. The geometry survives coarsely - but an approximation that is
# indistinguishable from measurement recreates by hand the exact silent
# corruption REQUIRE_16BIT_DEPTH exists to prevent.
# --------------------------------------------------------------------------- #
def _preview(mm, vis_max=3000.0, colormap=None, tv_range=False):
    """Draw a millimetre image the way the capture GUI would have."""
    code = np.clip(mm / vis_max * 255.0, 0, 255)
    code[mm <= 0] = 0
    g = code.astype(np.uint8)
    if tv_range:
        g = (16 + code * (219.0 / 255.0)).astype(np.uint8)
    if colormap:
        return cv2.applyColorMap(g, getattr(cv2, f"COLORMAP_{colormap}"))
    return np.dstack([g, g, g])


def _mm_field(h=90, w=160):
    mm = np.tile(np.linspace(400, 2800, w), (h, 1)).astype(np.float32)
    mm[10:30, 10:40] = 0.0                      # invalid region
    return mm


def test_a_grayscale_preview_round_trips_to_within_one_quantum():
    """The recovery is only worth anything if it actually inverts the display
    scaling. One level at 3000 mm is 11.8 mm, so that is the tolerance."""
    mm = _mm_field()
    got = ex.preview_to_mm(_preview(mm), "grayscale_8bit", 3000.0)
    ok = mm > 0
    assert np.abs(got[ok].astype(float) - mm[ok]).max() <= 3000.0 / 255.0 + 1


def test_a_colourised_preview_round_trips_too():
    mm = _mm_field()
    got = ex.preview_to_mm(_preview(mm, colormap="JET"), "colormapped_8bit",
                           3000.0, colormap="JET")
    ok = mm > 0
    # JET is not injective at its extremes, so judge the bulk, not the worst px.
    err = np.abs(got[ok].astype(float) - mm[ok])
    assert float(np.percentile(err, 95)) <= 3 * (3000.0 / 255.0)


def test_the_invalid_sentinel_stays_invalid():
    """0 means 'no measurement'. If recovery turned it into 0 mm - a distance -
    every 3D point built there would be fiction rather than skipped."""
    mm = _mm_field()
    got = ex.preview_to_mm(_preview(mm), "grayscale_8bit", 3000.0)
    assert (got[mm <= 0] == 0).all()


def test_tv_range_is_undone_rather_than_stretching_every_distance():
    mm = _mm_field()
    naive = ex.preview_to_mm(_preview(mm, tv_range=True), "grayscale_8bit",
                             3000.0, tv_range=False)
    fixed = ex.preview_to_mm(_preview(mm, tv_range=True), "grayscale_8bit",
                             3000.0, tv_range=True)
    ok = mm > 0
    assert np.abs(fixed[ok].astype(float) - mm[ok]).mean() < \
        np.abs(naive[ok].astype(float) - mm[ok]).mean()


def test_the_scale_is_honoured_not_hard_coded():
    mm = _mm_field()
    a = ex.preview_to_mm(_preview(mm, vis_max=3000.0), "grayscale_8bit", 3000.0)
    b = ex.preview_to_mm(_preview(mm, vis_max=3000.0), "grayscale_8bit", 1500.0)
    assert b.max() < a.max() * 0.6          # halving the scale halves distances


def test_recovery_is_off_by_default():
    """It rests on an assumed scale that is not in these session folders, so it
    must never fire because someone did not read the config."""
    assert ex.CONFIG["RECOVER_8BIT_DEPTH"] is False
    assert ex.CONFIG["DEPTH_VIS_MAX_MM"] == 3000.0


def test_the_plan_records_every_term_of_the_approximation(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "inspect_preview", lambda *a, **k: {
        "verdict": "colormapped_8bit", "colormap": "JET", "recoverable": True,
        "luma": {"looks_tv_range": True, "zero_frac": 0.05}})
    warns = []
    plan = ex._plan_preview_recovery({"depth": tmp_path / "d.avi"},
                                     dict(ex.CONFIG, DEPTH_VIS_MAX_MM=3000.0),
                                     "sid", 160, 90, warns)
    assert plan["kind"] == "colormapped_8bit" and plan["colormap"] == "JET"
    assert plan["tv_range"] is True
    assert plan["quantisation_mm"] == 11.76
    assert plan["sentinel_survived"] is True
    assert any("APPROXIMATE" in w for w in warns)


def test_a_lost_sentinel_is_warned_about_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "inspect_preview", lambda *a, **k: {
        "verdict": "grayscale_8bit", "recoverable": True,
        "luma": {"looks_tv_range": False, "zero_frac": 0.0}})
    warns = []
    plan = ex._plan_preview_recovery({"depth": tmp_path / "d.avi"}, ex.CONFIG,
                                     "sid", 160, 90, warns)
    assert plan["sentinel_survived"] is False
    assert any("sentinel" in w for w in warns)


def test_an_unrecoverable_file_yields_no_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "inspect_preview", lambda *a, **k: {
        "verdict": "unknown_8bit", "recoverable": False, "detail": "not depth"})
    warns = []
    assert ex._plan_preview_recovery({"depth": tmp_path / "d.avi"}, ex.CONFIG,
                                     "sid", 160, 90, warns) is None
    assert any("not recoverable" in w for w in warns)


def test_a_failed_inspection_is_not_treated_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "inspect_preview", lambda *a, **k: None)
    warns = []
    assert ex._plan_preview_recovery({"depth": tmp_path / "d.avi"}, ex.CONFIG,
                                     "sid", 160, 90, warns) is None


def _preview_avi(path, mm_frames, colormap=None, fps=15):
    """A real mpeg4/yuv420p depth preview, the way the field files are."""
    import subprocess
    h, w = mm_frames[0].shape
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "mpeg4", "-qscale:v", "2", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    for mm in mm_frames:
        p.stdin.write(_preview(mm, colormap=colormap).tobytes())
    p.stdin.close()
    p.wait()


def _rgb_avi(path, n, h, w, fps=15):
    import subprocess
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "mpeg4", "-qscale:v", "2", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    rng = np.random.default_rng(3)
    for _ in range(n):
        p.stdin.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def _avi_session(tmp_path, colormap=None, n=6, h=90, w=160):
    d = tmp_path / "Session_20250221_130957"
    d.mkdir(parents=True)
    _rgb_avi(d / "RGB_video.avi", n, h, w)
    _preview_avi(d / "Depth_video.avi", [_mm_field(h, w)] * n, colormap)
    (d / "calibration_params.txt").write_text(
        "Left Camera Intrinsic\nfx: 700, fy: 700\ncx: 80, cy: 45\n")
    return d


def _extract(tmp_path, **over):
    cfg = dict(ex.CONFIG, OUTPUT_ROOT=str(tmp_path / "out"),
               DRY_RUN=False, OVERWRITE=True, POOL_STRIDE=1, **over)
    cfg["INPUT_ROOTS"] = [{"path": str(tmp_path), "trip": "T", "site": "s",
                           "field": "f", "scene_hint": "mixed", "notes": ""}]
    sess = ex.discover(cfg)
    assert len(sess) == 1
    return cfg, ex.extract_session(sess[0], tmp_path / "out", cfg)


def test_without_the_flag_an_8_bit_session_gets_no_depth_at_all(tmp_path):
    _avi_session(tmp_path)
    _, rec = _extract(tmp_path)
    out = tmp_path / "out" / "sessions" / rec["session_id"]
    assert rec["depth_kind"] == "none"
    assert not (out / "depth").exists() and not (out / "depth_approx").exists()
    assert (out / "rgb").exists() and any((out / "rgb").iterdir())


def test_recovered_depth_never_lands_in_the_depth_directory(tmp_path):
    """A file under depth/ is metric millimetres and everything downstream is
    entitled to assume so."""
    _avi_session(tmp_path)
    _, rec = _extract(tmp_path, RECOVER_8BIT_DEPTH=True)
    out = tmp_path / "out" / "sessions" / rec["session_id"]
    assert not (out / "depth").exists()
    assert (out / "depth_approx").exists()
    assert len(list((out / "depth_approx").glob("*.png"))) > 0


def test_the_recovered_pngs_are_16_bit_millimetres_in_the_right_range(tmp_path):
    _avi_session(tmp_path)
    _, rec = _extract(tmp_path, RECOVER_8BIT_DEPTH=True)
    out = tmp_path / "out" / "sessions" / rec["session_id"]
    png = sorted((out / "depth_approx").glob("*.png"))[0]
    d = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16
    real = d[d > 0]
    assert 300 < np.percentile(real, 5) and np.percentile(real, 95) < 3100


def test_the_session_records_that_its_depth_is_approximate(tmp_path):
    """The distinction is not recoverable from the PNGs - they are uint16
    millimetres either way - so it has to be written down."""
    _avi_session(tmp_path)
    _, rec = _extract(tmp_path, RECOVER_8BIT_DEPTH=True)
    assert rec["depth_kind"] == "approximate"
    enc = rec["depth_encoding"]
    assert enc["approximate"] is True
    assert enc["directory"] == "depth_approx"
    assert enc["vis_max_mm"] == 3000.0 and enc["quantisation_mm"] == 11.76
    assert "not for the LEP canopy-height channel" in enc["note"]
    assert any("APPROXIMATE" in w for w in rec["warnings"])


def test_approximate_depth_stays_out_of_the_metric_qc_column(tmp_path):
    """median_depth_valid_frac_veg drives session comparison and a select_batches
    gate. Mixing an approximation into it makes the registry incomparable - the
    same trap the sharpness column already has across containers."""
    _avi_session(tmp_path)
    _, rec = _extract(tmp_path, RECOVER_8BIT_DEPTH=True)
    assert rec["pool_summary"]["median_depth_valid_frac_veg"] is None


def test_a_colourised_preview_is_recovered_end_to_end(tmp_path):
    _avi_session(tmp_path, colormap="JET")
    _, rec = _extract(tmp_path, RECOVER_8BIT_DEPTH=True)
    assert rec["depth_kind"] == "approximate"
    assert rec["depth_encoding"]["recovered_from"] == "colormapped_8bit"
    assert rec["depth_encoding"]["colormap"] == "JET"
    out = tmp_path / "out" / "sessions" / rec["session_id"]
    assert len(list((out / "depth_approx").glob("*.png"))) > 0


def test_approx_frames_stay_aligned_with_their_rgb_frames(tmp_path):
    _avi_session(tmp_path)
    _, rec = _extract(tmp_path, RECOVER_8BIT_DEPTH=True)
    out = tmp_path / "out" / "sessions" / rec["session_id"]
    rgb = {p.name for p in (out / "rgb").glob("*.png")}
    approx = {p.name for p in (out / "depth_approx").glob("*.png")}
    assert rgb and approx == rgb


def test_a_depth_approx_folder_is_not_mistaken_for_training_images(tmp_path):
    """depth_approx PNGs carry their RGB frame's filename, so a recursive image
    search would return one of each and feed depth to a model as a picture."""
    pi = load_script("perception/predict_images.py")
    sess = tmp_path / "sess"
    for sub in ("rgb", "depth", "depth_approx"):
        (sess / sub).mkdir(parents=True)
        cv2.imwrite(str(sess / sub / "f_000001.png"),
                    np.zeros((20, 20, 3), np.uint8))
    assert [p.name for p in pi.find_images(str(sess))] == ["f_000001.png"]
    assert all("depth" not in str(p.parent.name)
               for p in pi.find_images(str(sess)))
