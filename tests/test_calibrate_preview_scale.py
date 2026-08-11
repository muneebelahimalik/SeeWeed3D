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
    with pytest.raises(SystemExit, match="depth_kind"):
        cal.metric_depth_percentiles(tmp_path / "sess")


def test_invalid_pixels_are_excluded_from_the_reference(tmp_path):
    d = tmp_path / "sess" / "depth"
    d.mkdir(parents=True)
    mm = np.full((64, 96), 1500, np.uint16)
    mm[:32] = 0                                  # half the frame invalid
    cv2.imwrite(str(d / "f.png"), mm)
    pct, _ = cal.metric_depth_percentiles(tmp_path / "sess")
    assert all(v == pytest.approx(1500, abs=1) for v in pct.values())
