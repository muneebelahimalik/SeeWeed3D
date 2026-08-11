"""Measuring the scale an 8-bit depth preview was drawn at.

The recovery in extract_sessions rests on DEPTH_VIS_MAX_MM, and a session with
no session_meta.txt gives no way to confirm it - every recovered millimetre
inherits the assumption. When other sessions from the same rig have metric
depth, the scale can be measured instead.

The tests that matter here are the ones about REJECTING a fit. A single-point
ratio always produces a number; only consistency across the distribution says
whether that number means anything.
"""
import cv2
import numpy as np
import pytest

from conftest import load_script

cal = load_script("validation/calibrate_preview_scale.py")


def _mm_scene(n=6, h=64, w=96, near=500.0, far=2500.0, seed=0):
    """Frames whose depth spans a plausible working range, plus invalid holes."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        mm = np.tile(np.linspace(near, far, w), (h, 1)).astype(np.float32)
        mm += rng.normal(0, 15, mm.shape)
        mm[:5, :5] = 0.0                                    # invalid
        out.append(np.clip(mm, 0, 65535))
    return out


def _codes(mm_frames, vis_max, gamma=1.0):
    """Draw them the way a capture GUI would, optionally non-linearly."""
    out = []
    for mm in mm_frames:
        c = np.clip(mm / vis_max, 0, 1) ** gamma * 255.0
        c[mm <= 0] = 0
        out.append(c.astype(np.uint8))
    return out


def _pcts(mm_frames):
    v = np.concatenate([m[m > 0].ravel() for m in mm_frames]).astype(np.float64)
    return {p: float(np.percentile(v, p)) for p in cal.FIT_PERCENTILES}


# --------------------------------------------------------------------------- #
# The fit recovers a scale that is actually there
# --------------------------------------------------------------------------- #
def test_a_linear_preview_gives_back_the_scale_it_was_drawn_at():
    mm = _mm_scene()
    for vis_max in (2000.0, 3000.0, 5000.0):
        fit = cal.fit_scale(_pcts(mm),
                            cal.preview_code_percentiles(_codes(mm, vis_max)))
        assert fit["linear"]
        assert fit["vis_max_mm"] == pytest.approx(vis_max, rel=0.03)


def test_the_estimates_agree_across_the_distribution_when_linear():
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm), cal.preview_code_percentiles(_codes(mm, 3000.0)))
    est = list(fit["per_percentile"].values())
    assert max(est) / min(est) < 1.05
    assert fit["relative_spread"] <= cal.LINEARITY_TOL


# --------------------------------------------------------------------------- #
# ...and refuses when there is no single scale to find
#
# This is the point of fitting at several percentiles. A one-point ratio always
# returns a number, whether or not code and distance are linearly related.
# --------------------------------------------------------------------------- #
def test_a_gamma_curve_is_rejected_rather_than_averaged_into_a_number():
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm),
                        cal.preview_code_percentiles(_codes(mm, 3000.0, gamma=0.45)))
    assert not fit["linear"]
    assert "linear map" in fit["reason"]


def test_a_rejection_says_not_to_enable_the_recovery():
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm),
                        cal.preview_code_percentiles(_codes(mm, 3000.0, gamma=0.45)))
    text = " ".join(cal.report(fit))
    assert "REJECTED" in text
    assert "Do not set RECOVER_8BIT_DEPTH" in text


def test_no_overlapping_data_is_a_rejection_not_a_crash():
    fit = cal.fit_scale({20: 500.0}, {80: 100.0})
    assert not fit["linear"]


# --------------------------------------------------------------------------- #
# The invalid sentinel must not be read as a near distance
# --------------------------------------------------------------------------- #
def test_zero_codes_are_excluded_from_the_percentiles():
    """0 means 'no measurement'. Counting it as a distance drags every
    percentile down and inflates the fitted scale."""
    mm = _mm_scene()
    codes = _codes(mm, 3000.0)
    holed = [c.copy() for c in codes]
    for c in holed:
        c[:, :40] = 0                                   # a large invalid region
    a = cal.preview_code_percentiles(codes)
    b = cal.preview_code_percentiles(holed)
    # Removing low-value pixels raises the percentiles; it must not be the
    # case that the zeros were counted, which would drag them down instead.
    assert b[50] >= a[50]


def test_frames_with_nothing_valid_do_not_break_the_fit():
    assert cal.preview_code_percentiles([np.zeros((10, 10), np.uint8)]) is None


# --------------------------------------------------------------------------- #
# What the report tells the user
# --------------------------------------------------------------------------- #
def test_agreement_with_the_documented_constant_is_called_corroboration():
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm), cal.preview_code_percentiles(_codes(mm, 3000.0)))
    text = " ".join(cal.report(fit, assumed=3000.0))
    assert "corroboration" in text
    assert "DEPTH_VIS_MAX_MM" in text
    # 8-bit quantisation means the fit lands near 3000, not on it.
    assert fit["vis_max_mm"] == pytest.approx(3000.0, rel=0.03)


def test_a_modest_disagreement_prefers_the_measurement():
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm), cal.preview_code_percentiles(_codes(mm, 3400.0)))
    text = " ".join(cal.report(fit, assumed=3000.0))
    assert "Prefer the MEASURED value" in text


def test_a_wild_disagreement_blames_the_assumption_not_the_constant():
    """The fit rests on the rig being mounted the same way for both sessions.
    A large gap is far more likely to mean that failed than to be a discovery
    about the capture GUI."""
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm), cal.preview_code_percentiles(_codes(mm, 9000.0)))
    text = " ".join(cal.report(fit, assumed=3000.0))
    assert "WARNING" in text and "mounted the same way" in text


def test_the_report_always_restates_what_the_result_is_worth():
    mm = _mm_scene()
    fit = cal.fit_scale(_pcts(mm), cal.preview_code_percentiles(_codes(mm, 3000.0)))
    text = " ".join(cal.report(fit))
    assert "mm per level" in text
    assert "not for the LEP canopy-height channel" in text


# --------------------------------------------------------------------------- #
# Reading the metric reference
# --------------------------------------------------------------------------- #
def test_the_reference_comes_from_real_depth_pngs(tmp_path):
    d = tmp_path / "sess" / "depth"
    d.mkdir(parents=True)
    for i, mm in enumerate(_mm_scene(n=3)):
        cv2.imwrite(str(d / f"f_{i:03d}.png"), mm.astype(np.uint16))
    pct, n = cal.metric_depth_percentiles(tmp_path / "sess")
    assert n == 3
    assert 400 < pct[50] < 2600


def test_a_session_without_metric_depth_is_refused(tmp_path):
    """A session holding only depth_approx/ cannot calibrate anything - that
    is the output being calibrated."""
    (tmp_path / "sess" / "depth_approx").mkdir(parents=True)
    with pytest.raises(SystemExit, match="the output being calibrated"):
        cal.metric_depth_percentiles(tmp_path / "sess")


def test_invalid_pixels_are_excluded_from_the_reference(tmp_path):
    d = tmp_path / "sess" / "depth"
    d.mkdir(parents=True)
    mm = np.full((64, 96), 1500, np.uint16)
    mm[:32] = 0                                  # half the frame invalid
    cv2.imwrite(str(d / "f.png"), mm)
    pct, _ = cal.metric_depth_percentiles(tmp_path / "sess")
    assert all(v == pytest.approx(1500, abs=1) for v in pct.values())


# --------------------------------------------------------------------------- #
# Calibrating with nothing extracted
#
# Requiring an EXTRACTED session as the reference forces two passes: extract to
# get a reference, set the scale, extract again for the recovery. The source
# depth video carries the same millimetres, so the scale can be measured before
# anything is extracted and a visit goes through once.
# --------------------------------------------------------------------------- #
def _mkv16(path, mm_frames, fps=15):
    """A real lossless 16-bit depth video, as the MKV sessions have."""
    import subprocess
    h, w = mm_frames[0].shape
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "gray16le", str(path)],
        stdin=subprocess.PIPE)
    for mm in mm_frames:
        p.stdin.write(mm.astype(np.uint16).tobytes())
    p.stdin.close()
    p.wait()


def _preview_avi_for_cal(path, mm_frames, vis_max=3000.0, fps=15):
    import subprocess
    h, w = mm_frames[0].shape
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "mpeg4", "-qscale:v", "2", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    for mm in mm_frames:
        c = np.clip(mm / vis_max, 0, 1) * 255.0
        c[mm <= 0] = 0
        g = c.astype(np.uint8)
        p.stdin.write(np.dstack([g, g, g]).tobytes())
    p.stdin.close()
    p.wait()


def test_the_scale_can_be_measured_from_source_videos_alone(tmp_path):
    mm = _mm_scene(n=8, h=64, w=96)
    _mkv16(tmp_path / "Depth_video.mkv", mm)
    _preview_avi_for_cal(tmp_path / "Depth_video.avi", mm, vis_max=3000.0)
    fit = cal.main(["--metric-video", str(tmp_path / "Depth_video.mkv"),
                    "--preview", str(tmp_path / "Depth_video.avi")])
    assert fit["linear"]
    assert fit["vis_max_mm"] == pytest.approx(3000.0, rel=0.05)


def test_a_preview_is_refused_as_the_metric_reference(tmp_path):
    """The reference must be real millimetres. Pointing it at another preview
    would calibrate one guess against another and report success."""
    mm = _mm_scene(n=4, h=64, w=96)
    _preview_avi_for_cal(tmp_path / "Depth_video.avi", mm)
    with pytest.raises(SystemExit, match="not 16-bit"):
        cal.metric_video_percentiles("ffmpeg", "ffprobe",
                                     tmp_path / "Depth_video.avi")


def test_the_two_reference_routes_agree(tmp_path):
    """Same geometry, one read from a source video and one from extracted
    PNGs. If they disagreed, one of the two paths would be wrong."""
    mm = _mm_scene(n=6, h=64, w=96)
    _mkv16(tmp_path / "Depth_video.mkv", mm)
    d = tmp_path / "sess" / "depth"
    d.mkdir(parents=True)
    for i, f in enumerate(mm):
        cv2.imwrite(str(d / f"f_{i:03d}.png"), f.astype(np.uint16))
    a, _ = cal.metric_video_percentiles("ffmpeg", "ffprobe",
                                        tmp_path / "Depth_video.mkv")
    b, _ = cal.metric_depth_percentiles(tmp_path / "sess")
    for p in a:
        assert a[p] == pytest.approx(b[p], rel=0.02)


# --------------------------------------------------------------------------- #
# The error a deleted output directory produces
#
# Extracted output gets deleted between improvements to the pipeline, so this
# is the common case, and "must be a session whose depth_kind is metric" sends
# you looking for the wrong problem when the folder simply is not there.
# --------------------------------------------------------------------------- #
def test_a_missing_session_says_it_is_missing_and_names_the_way_round(tmp_path):
    msg = cal.why_not_a_metric_reference(tmp_path / "gone")
    assert "does not exist" in msg
    assert "--metric-video" in msg and "no extraction is needed" in msg


def test_an_approx_only_session_explains_the_circularity(tmp_path):
    (tmp_path / "s" / "depth_approx").mkdir(parents=True)
    msg = cal.why_not_a_metric_reference(tmp_path / "s")
    assert "the output being calibrated" in msg


def test_a_session_extracted_without_depth_says_so_from_its_metadata(tmp_path):
    m = tmp_path / "s" / "meta"
    m.mkdir(parents=True)
    (m / "session.json").write_text('{"depth_kind": "none"}')
    msg = cal.why_not_a_metric_reference(tmp_path / "s")
    assert "extracted without depth" in msg and "depth_kind: none" in msg


def test_unreadable_metadata_still_produces_a_usable_message(tmp_path):
    m = tmp_path / "s" / "meta"
    m.mkdir(parents=True)
    (m / "session.json").write_text("{not json")
    msg = cal.why_not_a_metric_reference(tmp_path / "s")
    assert "not found" in msg and "--metric-video" in msg
