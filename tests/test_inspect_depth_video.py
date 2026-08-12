"""What is actually inside a depth video?

extract_sessions refuses to decode a depth stream that is not 16-bit, which is
correct and is also the end of the conversation - it says the file is not
metric depth, not what it IS. The difference decides whether a visit's depth is
coarsely recoverable or gone, so the test has to be decisive rather than a
guess. These pin it against synthetic frames of each kind.
"""
import numpy as np
import pytest

from conftest import load_script

iv = load_script("validation/inspect_depth_video.py")


def _ramp(h=90, w=160):
    """A depth-like image: a horizontal distance ramp with an invalid patch."""
    g = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    g[10:30, 10:40] = 0                      # the invalid sentinel
    return g


def _gray_bgr(g):
    return np.dstack([g, g, g])


def _cmapped(g, name):
    import cv2
    return cv2.applyColorMap(g, getattr(cv2, f"COLORMAP_{name}"))


# --------------------------------------------------------------------------- #
# 16-bit is settled by the pixel format alone
# --------------------------------------------------------------------------- #
def test_a_16_bit_stream_needs_no_frames_at_all():
    v = iv.classify({"codec": "ffv1", "pix_fmt": "gray16le"}, [])
    assert v["verdict"] == "true_16bit"
    assert v["recoverable"] and v["exact"]


def test_the_bit_depth_test_accepts_every_16_bit_spelling():
    for pf in ("gray16le", "gray16be", "yuv420p16le", "p010le"):
        assert iv.sixteen_bit(pf)
    for pf in ("yuv420p", "yuvj420p", "bgr24", "gray", None):
        assert not iv.sixteen_bit(pf)


# --------------------------------------------------------------------------- #
# grayscale vs colour: the decisive, colormap-independent test
# --------------------------------------------------------------------------- #
def test_a_grayscale_preview_is_recognised_as_one():
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"},
                    [_gray_bgr(_ramp())])
    assert v["verdict"] == "grayscale_8bit"
    assert v["recoverable"] and not v["exact"]


def test_chroma_jitter_from_lossy_encoding_does_not_read_as_colour():
    """A true gray image stored in a colour format comes back with a level or
    two of chroma noise. Calling that a colormap would send the recovery down
    the wrong path entirely."""
    rng = np.random.default_rng(0)
    g = _ramp()
    noisy = np.clip(_gray_bgr(g).astype(np.int16)
                    + rng.integers(-2, 3, (g.shape[0], g.shape[1], 3)), 0, 255
                    ).astype(np.uint8)
    assert iv.chroma_spread(noisy) <= iv.GRAY_CHROMA_TOL
    assert iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"},
                       [noisy])["verdict"] == "grayscale_8bit"


def test_a_colormapped_preview_is_recognised_and_named():
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"},
                    [_cmapped(_ramp(), "JET")])
    assert v["verdict"] == "colormapped_8bit"
    assert v["colormap"] == "JET"
    assert v["recoverable"] and not v["exact"]


def test_the_named_colormap_is_the_right_one():
    for name in ("JET", "TURBO", "VIRIDIS", "HOT"):
        got, resid, _ = iv.match_colormap(_cmapped(_ramp(), name))
        assert got == name, f"{name} matched as {got}"
        assert resid <= iv.COLORMAP_RESIDUAL_TOL


def test_the_recovered_index_is_the_original_depth_code():
    """The point of naming the colormap: inverting it gives back the 0..255
    code a scale would turn into millimetres. If that round trip does not
    hold, the identification is worthless."""
    g = _ramp()
    _, _, idx = iv.match_colormap(_cmapped(g, "JET"))
    assert idx.shape == g.shape
    # JET is not injective at its extremes, so a handful of codes are
    # genuinely ambiguous; the bulk must come back.
    assert float(np.mean(np.abs(idx.astype(int) - g.astype(int)) <= 2)) > 0.9


def test_an_ordinary_photograph_is_not_called_depth():
    """Every colormap produces a nearest entry for every pixel, so only the
    RESIDUAL can reject a wrong match. Without it, a colour image would be
    confidently 'recovered' into a depth map of noise."""
    rng = np.random.default_rng(1)
    photo = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"}, [photo])
    assert v["verdict"] == "unknown_8bit"
    assert not v["recoverable"]


def test_an_undecodable_file_is_not_guessed_at():
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"}, [])
    assert v["verdict"] == "undecodable" and not v["recoverable"]


# --------------------------------------------------------------------------- #
# The two encoding traps that silently corrupt a conversion
# --------------------------------------------------------------------------- #
def test_tv_range_is_detected_because_it_costs_14_percent_of_the_scale():
    g = np.tile(np.linspace(16, 235, 160, dtype=np.uint8), (90, 1))
    assert iv.luma_profile(_gray_bgr(g))["looks_tv_range"]
    assert not iv.luma_profile(_gray_bgr(_ramp()))["looks_tv_range"]


def test_the_lost_invalid_sentinel_is_reported():
    """0 means 'no measurement'. If the encode smeared it away, invalid
    regions are indistinguishable from near distances and any 3D point built
    from them is fiction."""
    g = np.tile(np.linspace(40, 255, 160, dtype=np.uint8), (90, 1))
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"}, [_gray_bgr(g)])
    assert v["luma"]["zero_frac"] == 0.0
    assert any("sentinel" in line for line in iv.advise(v))


def test_the_invalid_sentinel_surviving_is_not_flagged():
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"},
                    [_gray_bgr(_ramp())])
    assert v["luma"]["zero_frac"] > 0.01
    assert not any("sentinel" in line for line in iv.advise(v))


# --------------------------------------------------------------------------- #
# The advice has to be honest about what a positive result is worth
# --------------------------------------------------------------------------- #
def test_a_recoverable_preview_is_never_described_as_fine():
    text = " ".join(iv.advise(iv.classify(
        {"codec": "mpeg4", "pix_fmt": "yuv420p"}, [_gray_bgr(_ramp())])))
    assert "PREVIEW, not measurement" in text
    assert "11.8 mm" in text                      # 3000/255, stated not implied
    assert "Not worth it for" in text and "canopy-height" in text


def test_the_missing_scale_is_named_as_the_blocker():
    """Without depth_vis_max_mm the code cannot become a distance at all, and
    it is not in these session folders."""
    text = " ".join(iv.advise(iv.classify(
        {"codec": "mpeg4", "pix_fmt": "yuv420p"}, [_cmapped(_ramp(), "JET")])))
    assert "depth_vis_max_mm" in text


def test_an_unrecoverable_file_says_so_without_hedging():
    rng = np.random.default_rng(2)
    text = " ".join(iv.advise(iv.classify(
        {"codec": "mpeg4", "pix_fmt": "yuv420p"},
        [rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)])))
    assert "Nothing to recover" in text


def test_sixteen_bit_advice_is_one_line():
    assert iv.advise({"verdict": "true_16bit"}) == ["Extract normally. "
                                                    "Nothing to do."]


# --------------------------------------------------------------------------- #
# Per-frame normalisation: the one failure no scale can fix
#
# The v1 capture GUI wrote depth as `depth / depth.max() * 255`, with the max
# recomputed EVERY frame. The scale is a property of the individual frame - one
# distant outlier sets it for every other pixel - so no constant relates code to
# millimetres. depth_vis_max_mm never entered into it.
# --------------------------------------------------------------------------- #
def _normalised(mm_frames):
    """Exactly what the capture code did."""
    return [_gray_bgr((mm / (mm.max() or 1) * 255).astype(np.uint8))
            for mm in mm_frames]


def _fixed_scale_clipped(mm_frames, vis_max=3000.0):
    """A fixed scale that happens to clip - reaches 255 too, but saturates a
    whole region rather than a handful of outliers."""
    out = []
    for mm in mm_frames:
        c = np.clip(mm / vis_max, 0, 1) * 255.0
        out.append(_gray_bgr(c.astype(np.uint8)))
    return out


def _scenes(n=5, h=90, w=160, far_region=False):
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        mm = np.tile(np.linspace(600, 2400 + k * 300, w), (h, 1))
        mm = mm + rng.normal(0, 20, mm.shape)
        if far_region:
            mm[:, 140:] = 4000.0            # genuinely beyond a 3000 mm cut
        out.append(np.clip(mm, 0, 1e9))
    return out


def test_per_frame_normalisation_is_detected():
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"},
                    _normalised(_scenes()))
    assert v["verdict"] == "per_frame_normalised_8bit"
    assert v["recoverable"] is False


def test_a_clipped_fixed_scale_is_not_mistaken_for_it():
    """Both reach 255 in every frame. Only the SIZE of the saturated area
    separates them - a fixed scale saturates everything beyond the cut."""
    frames = _fixed_scale_clipped(_scenes(far_region=True))
    norm = iv.per_frame_normalisation(frames)
    assert all(m >= 250 for m in norm["frame_maxima"])      # tops out too
    assert norm["max_saturated_frac"] > iv.NORMALISED_SAT_FRAC
    assert not norm["normalised"]
    assert iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"},
                       frames)["verdict"] == "grayscale_8bit"


def test_a_fixed_scale_that_never_clips_is_unaffected():
    frames = _fixed_scale_clipped(_scenes(), vis_max=6000.0)
    assert not iv.per_frame_normalisation(frames)["normalised"]


def test_one_frame_is_not_enough_to_call_it():
    """A single frame reaching its own maximum says nothing at all."""
    assert iv.per_frame_normalisation(_normalised(_scenes(n=1))) is None
    assert iv.per_frame_normalisation(_normalised(_scenes(n=2))) is None


def test_it_outranks_the_colormap_question():
    """A per-frame normalised preview is unrecoverable whether or not somebody
    then ran a colormap over it, so it is decided first."""
    import cv2 as _cv2
    frames = [_cv2.applyColorMap(
        (mm / mm.max() * 255).astype(np.uint8), _cv2.COLORMAP_JET)
        for mm in _scenes()]
    v = iv.classify({"codec": "mpeg4", "pix_fmt": "yuv420p"}, frames)
    assert v["verdict"] == "per_frame_normalised_8bit"
    assert v["recoverable"] is False


def test_16_bit_is_still_decided_before_anything_else():
    """A real 16-bit stream is never even decoded, so the normalisation test
    must not be able to reach it."""
    v = iv.classify({"codec": "ffv1", "pix_fmt": "gray16le"}, [])
    assert v["verdict"] == "true_16bit"


def test_the_advice_says_no_calibration_will_help():
    """The expensive mistake here is spending a day calibrating something that
    has no constant to find."""
    text = " ".join(iv.advise(iv.classify(
        {"codec": "mpeg4", "pix_fmt": "yuv420p"}, _normalised(_scenes()))))
    assert "no calibration will change that" in text
    assert "own maximum" in text
    assert "There is no constant to find" in text


def test_the_advice_names_the_one_line_fix_for_future_captures():
    text = " ".join(iv.advise(iv.classify(
        {"codec": "mpeg4", "pix_fmt": "yuv420p"}, _normalised(_scenes()))))
    assert "16-bit" in text and "zed_capture.py" in text


def test_it_still_says_the_sessions_are_usable():
    """RGB-only is not a lost session - segmentation never touches depth."""
    text = " ".join(iv.advise(iv.classify(
        {"codec": "mpeg4", "pix_fmt": "yuv420p"}, _normalised(_scenes()))))
    assert "fully usable for segmentation" in text


def test_a_fixed_scales_peak_tracks_the_scene_while_normalisation_pins_it():
    """The property that separates them. Under a fixed scale the peak is
    wherever the farthest thing in that frame happens to land, so it moves as
    the scene changes. Normalisation forces it to the top of the range every
    frame, by construction."""
    scenes = _scenes()                       # each frame reaches further out
    fixed = iv.per_frame_normalisation(
        _fixed_scale_clipped(scenes, vis_max=6000.0))["frame_maxima"]
    pinned = iv.per_frame_normalisation(_normalised(scenes))["frame_maxima"]
    assert max(fixed) - min(fixed) > iv.NORMALISED_MAX_RANGE
    assert max(pinned) - min(pinned) == 0


def test_the_ambiguous_case_is_documented_and_errs_toward_refusing():
    """A fixed-scale preview whose scene runs past the cut in a small patch
    EVERY frame is indistinguishable from outside. Recovering garbage depth
    costs far more than declining a file that turns out recoverable, so the
    tie goes to refusing - and the docstring says so rather than implying the
    test is infallible."""
    doc = iv.per_frame_normalisation.__doc__
    assert "Not infallible" in doc
    assert "errs toward" in doc and "refusing" in doc
