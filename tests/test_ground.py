"""Height above the soil, and metric scale, from stereo depth.

Depth is a VETO and a scale reference here, never a boundary source - stereo is
least reliable at exactly the leaf margins a mask is deciding. So every test is
about one of two things: does height separate what colour cannot, and does the
code abstain when it has not actually measured anything.
"""
import json

import cv2
import numpy as np
import pytest

from conftest import load_script

gr = load_script("perception/ground.py")


def _disc(shape, cx, cy, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def _flat_scene(h=480, w=640, camera_mm=1200.0):
    return np.full((h, w), camera_mm, np.float32)


# --------------------------------------------------------------------------- #
# The thing colour cannot do
# --------------------------------------------------------------------------- #
def test_a_pebble_and_a_cotyledon_are_separated_by_height():
    """Both pass every colour gate on pale, pebbly ground. Only one is raised."""
    d = _flat_scene()
    plant = _disc(d.shape, 200, 240, 25)
    pebble = _disc(d.shape, 450, 240, 20)
    d[plant] -= 15.0                       # 15 mm tall
    veg = plant | pebble                   # colour sees one class

    height, measured = gr.height_map(d, veg=veg)
    assert gr.instance_height(plant, height, measured)[0] == \
        pytest.approx(15.0, abs=1.5)
    assert gr.instance_height(pebble, height, measured)[0] == \
        pytest.approx(0.0, abs=1.5)


def test_height_is_positive_upward():
    """Nearer the camera is a SMALLER depth. Getting this backwards would make
    every plant negative and the veto would delete all of them."""
    d = _flat_scene()
    plant = _disc(d.shape, 320, 240, 30)
    d[plant] -= 20.0
    height, measured = gr.height_map(d, veg=plant)
    assert gr.instance_height(plant, height, measured)[0] > 0


# --------------------------------------------------------------------------- #
# Local surface, not a plane
# --------------------------------------------------------------------------- #
def _bedded_scene(h=480, w=640, bed_px=160, furrow_mm=60.0, tilt_mm=90.0):
    """Raised beds with furrows between them, under a tilted camera - which is
    what a Vidalia field actually looks like from the boom."""
    x = np.arange(w)
    bed = 1200.0 + furrow_mm * ((x // bed_px) % 2)
    return (bed[None, :] + np.linspace(0, tilt_mm, h)[:, None]).astype(np.float32)


def test_a_plant_in_a_furrow_measures_the_same_as_one_on_a_bed():
    """The whole reason the surface is local. A single fitted plane reports the
    furrow floor as below ground and the bed top as plant."""
    d = _bedded_scene()
    on_bed = _disc(d.shape, 100, 240, 22)
    in_furrow = _disc(d.shape, 260, 240, 22)
    d[on_bed] -= 18.0
    d[in_furrow] -= 18.0
    veg = on_bed | in_furrow

    height, measured = gr.height_map(d, veg=veg)
    a = gr.instance_height(on_bed, height, measured)[0]
    b = gr.instance_height(in_furrow, height, measured)[0]
    assert a == pytest.approx(18.0, abs=4.0)
    assert b == pytest.approx(18.0, abs=4.0)


def test_a_global_plane_would_have_got_it_wrong():
    """States the alternative's error rather than asserting ours is good in a
    vacuum: on the same scene a single plane misreads the furrow plant by more
    than the plant's own height."""
    d = _bedded_scene()
    in_furrow = _disc(d.shape, 260, 240, 22)
    d[in_furrow] -= 18.0
    plane = np.percentile(d[~in_furrow], 80)
    assert abs((plane - d[in_furrow].mean()) - 18.0) > 5.0


def test_a_window_too_large_inflates_the_high_ground():
    """The failure documented in the module, pinned so the defaults cannot
    drift back to it. A tile straddling a bed and a furrow takes its soil from
    the FURROW, so the bed plant reads as bed height plus plant height."""
    d = _bedded_scene()
    on_bed = _disc(d.shape, 100, 240, 22)
    d[on_bed] -= 18.0
    coarse, m_c = gr.height_map(d, veg=on_bed, tile_px=96, smooth_tiles=1)
    fine, m_f = gr.height_map(d, veg=on_bed, tile_px=32, smooth_tiles=0)
    assert gr.instance_height(on_bed, coarse, m_c)[0] > 40      # inflated
    assert gr.instance_height(on_bed, fine, m_f)[0] == pytest.approx(18, abs=4)


def test_passing_vegetation_removes_the_large_plant_trap():
    """A big plant would otherwise define its own local soil and measure itself
    as flat. Measured: 140 px plant, 25 mm tall - exact with veg at every tile
    size, badly under-read without it."""
    d = _flat_scene()
    big = _disc(d.shape, 320, 240, 70)
    d[big] -= 25.0
    with_veg, m1 = gr.height_map(d, veg=big, tile_px=32)
    without, m2 = gr.height_map(d, veg=None, tile_px=32)
    assert gr.instance_height(big, with_veg, m1)[0] == pytest.approx(25, abs=2)
    assert gr.instance_height(big, without, m2)[0] < 20


# --------------------------------------------------------------------------- #
# Invalid depth is not zero depth
# --------------------------------------------------------------------------- #
def test_the_invalid_sentinel_becomes_nan_not_zero_millimetres(tmp_path):
    """0 read naively is 'touching the lens', which is nearer than everything
    and would read as the tallest object in the frame."""
    raw = np.full((20, 20), 1200, np.uint16)
    raw[5:10, 5:10] = 0
    path = tmp_path / "d.png"
    cv2.imwrite(str(path), raw)
    d = gr.load_depth_mm(path)
    assert np.isnan(d[7, 7]) and d[0, 0] == 1200


def test_an_8bit_preview_is_refused(tmp_path):
    """Scaling a normalised preview into plausible millimetres is the exact
    fiction REQUIRE_16BIT_DEPTH exists to prevent at extraction time."""
    path = tmp_path / "d.png"
    cv2.imwrite(str(path), np.full((20, 20), 128, np.uint8))
    assert gr.load_depth_mm(path) is None


def test_unmeasured_pixels_are_not_reported_as_ground_level():
    """height==0 where nothing was measured would read as 'lying flat', which
    is a claim, not an absence of one."""
    d = _flat_scene()
    plant = _disc(d.shape, 320, 240, 30)
    d[plant] = np.nan                        # stereo dropped out on the plant
    height, measured = gr.height_map(d, veg=plant)
    assert not measured[240, 320]
    assert gr.instance_height(plant, height, measured)[0] is None


# --------------------------------------------------------------------------- #
# Abstention - the safety property
# --------------------------------------------------------------------------- #
def test_an_instance_with_too_little_depth_abstains():
    d = _flat_scene()
    plant = _disc(d.shape, 320, 240, 30)
    d[plant] = np.nan
    ys, xs = np.nonzero(plant)
    d[ys[:20], xs[:20]] = 1180.0             # a handful of valid pixels only
    height, measured = gr.height_map(d, veg=plant)
    h, frac = gr.instance_height(plant, height, measured,
                                 min_measured_frac=0.25)
    assert h is None and frac < 0.25


def test_it_answers_once_enough_of_the_instance_is_measured():
    d = _flat_scene()
    plant = _disc(d.shape, 320, 240, 30)
    d[plant] -= 14.0
    height, measured = gr.height_map(d, veg=plant)
    h, frac = gr.instance_height(plant, height, measured)
    assert h is not None and frac > 0.9


def test_an_empty_mask_abstains_rather_than_dividing_by_zero():
    d = _flat_scene()
    height, measured = gr.height_map(d)
    assert gr.instance_height(np.zeros(d.shape, bool), height, measured) \
        == (None, 0.0)


# --------------------------------------------------------------------------- #
# Confidence polarity is never guessed
# --------------------------------------------------------------------------- #
def _session(tmp_path, **meta):
    d = tmp_path / "sess" / "meta"
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return d.parent


def test_an_unknown_polarity_disables_gating(tmp_path):
    """Gating backwards keeps precisely the pixels it was meant to drop, and
    the result looks like a CLEANER depth map. Not gating is the safe error."""
    assert gr.confidence_polarity(_session(tmp_path, depth_kind="metric")) is None
    assert gr.confidence_mask(np.full((4, 4), 200, np.uint8), None, 0.3) is None


def test_both_polarities_are_honoured(tmp_path):
    # Values chosen clear of the midpoint: at threshold 0.5 the cut is 127.5,
    # and asserting on 128 would be testing which side of a boundary a tie
    # falls rather than that the polarity is honoured.
    conf = np.array([[0, 100, 255]], np.uint8)
    hi = gr.confidence_mask(conf, "higher_is_better", 0.5)
    lo = gr.confidence_mask(conf, "lower_is_better", 0.5)
    assert list(hi[0]) == [False, False, True]
    assert list(lo[0]) == [True, True, False]
    # and one number means the same strictness under either convention
    assert int(hi.sum()) + int(lo.sum()) == conf.size


def test_the_polarity_is_read_from_the_session(tmp_path):
    s = _session(tmp_path, depth_kind="metric",
                 confidence_encoding={"polarity": "lower_is_better"})
    assert gr.confidence_polarity(s) == "lower_is_better"


# --------------------------------------------------------------------------- #
# Only metric depth is ever used
# --------------------------------------------------------------------------- #
def test_only_metric_depth_qualifies(tmp_path):
    """A v1 preview was normalised per frame at capture, so no constant relates
    it to millimetres. It must never reach a height calculation."""
    assert gr.has_metric_depth(_session(tmp_path, depth_kind="metric"))
    assert not gr.has_metric_depth(_session(tmp_path, depth_kind="preview"))
    assert not gr.has_metric_depth(_session(tmp_path, depth_kind="none"))


def test_a_session_without_metadata_does_not_qualify(tmp_path):
    (tmp_path / "empty").mkdir()
    assert gr.session_depth_kind(tmp_path / "empty") == "unknown"
    assert not gr.has_metric_depth(tmp_path / "empty")


# --------------------------------------------------------------------------- #
# Metric scale
# --------------------------------------------------------------------------- #
def test_area_in_mm2_matches_the_pinhole_model():
    """At depth Z a pixel spans Z/fx. This is what stops every size floor in
    the pipeline depending on how high the boom happens to be."""
    d = _flat_scene(camera_mm=1000.0)
    m = _disc(d.shape, 320, 240, 50)
    got = gr.area_mm2(m, d, 1000.0, 1000.0)
    assert got == pytest.approx(int(m.sum()) * 1.0, rel=0.02)   # 1 mm per px


def test_the_same_plant_measures_the_same_from_a_different_height():
    """The whole point: a session recorded at a different boom height must not
    need its own thresholds."""
    near, far = _flat_scene(camera_mm=1000.0), _flat_scene(camera_mm=2000.0)
    # twice the distance, half the angular size
    a_near = gr.area_mm2(_disc(near.shape, 320, 240, 60), near, 1000., 1000.)
    a_far = gr.area_mm2(_disc(far.shape, 320, 240, 30), far, 1000., 1000.)
    assert a_far == pytest.approx(a_near, rel=0.05)


def test_area_abstains_without_enough_valid_depth():
    d = _flat_scene()
    m = _disc(d.shape, 320, 240, 40)
    d[m] = np.nan
    assert gr.area_mm2(m, d, 1000.0, 1000.0) is None


def test_area_abstains_without_calibration():
    d = _flat_scene()
    assert gr.area_mm2(_disc(d.shape, 320, 240, 20), d, None, None) is None


def test_calibration_is_read_from_the_session(tmp_path):
    d = tmp_path / "sess" / "meta"
    d.mkdir(parents=True)
    (d / "calibration.json").write_text(
        json.dumps({"left": {"fx": 1050.5, "fy": 1050.5}}), encoding="utf-8")
    assert gr.calibration(d.parent) == (1050.5, 1050.5)


def test_missing_calibration_is_not_an_error(tmp_path):
    assert gr.calibration(tmp_path) == (None, None)


# --------------------------------------------------------------------------- #
# The diagnostic
# --------------------------------------------------------------------------- #
def test_the_surface_report_measures_terrain_relief():
    """Read against the height threshold: 60 mm of relief under a 6 mm gate is
    a field where the surface estimate is doing real work."""
    flat = gr.surface_report(_flat_scene())
    bedded = gr.surface_report(_bedded_scene())
    assert flat["relief_mm"] < 5
    assert bedded["relief_mm"] > 40


def test_the_report_says_how_much_depth_was_usable():
    d = _flat_scene()
    d[:, :320] = np.nan
    assert gr.surface_report(d)["measured_frac"] == pytest.approx(0.5, abs=0.02)


def test_a_frame_with_no_valid_depth_does_not_crash():
    d = np.full((64, 64), np.nan, np.float32)
    height, measured = gr.height_map(d)
    assert not measured.any()
    assert gr.surface_report(d)["measured_frac"] == 0.0


# --------------------------------------------------------------------------- #
# The single-mask path, for callers that export one merged mask per frame
# --------------------------------------------------------------------------- #
def _cfg(**kw):
    base = dict(HEIGHT_MIN_MM=6.0, HEIGHT_MIN_MEASURED_FRAC=0.25,
                HEIGHT_PERCENTILE=75.0, GROUND_TILE_PX=32,
                GROUND_PERCENTILE=80.0, DEPTH_MIN_CONFIDENCE=0.30,
                MIN_INSTANCE_AREA_MM2=None, USE_DEPTH_HEIGHT="auto")
    base.update(kw)
    return base


def test_a_merged_mask_keeps_the_raised_component_and_drops_the_flat_one():
    """The onion and weed prelabelers export one boolean mask per frame, not a
    list of instances, so the veto has to work per connected component."""
    d = _flat_scene()
    plant = _disc(d.shape, 200, 240, 25)
    stone = _disc(d.shape, 450, 240, 22)
    d[plant] -= 15.0
    veg = plant | stone

    out, qa = gr.mask_height_filter(veg, veg, d, _cfg())
    assert (out & plant).sum() > plant.sum() * 0.9
    assert not (out & stone).any()
    assert qa["height_dropped_flat"] == 1


def test_an_unmeasurable_component_survives_the_merged_path_too():
    d = _flat_scene()
    plant = _disc(d.shape, 200, 240, 25)
    d[plant] = np.nan
    out, qa = gr.mask_height_filter(plant, plant, d, _cfg())
    assert (out & plant).any() and qa["height_abstained"] == 1


def test_an_empty_mask_is_returned_unchanged():
    d = _flat_scene()
    empty = np.zeros(d.shape, bool)
    out, qa = gr.mask_height_filter(empty, empty, d, _cfg())
    assert not out.any() and qa == {}


# --------------------------------------------------------------------------- #
# The per-session decision, made once
# --------------------------------------------------------------------------- #
def test_metric_depth_turns_the_veto_on(tmp_path, capsys):
    s = _session(tmp_path, depth_kind="metric")
    use, fx, fy, pol = gr.session_depth_setup("sess", s, _cfg())
    assert use is True
    assert "height veto on" in capsys.readouterr().out


def test_a_preview_session_turns_it_off_and_says_so(tmp_path, capsys):
    s = _session(tmp_path, depth_kind="preview")
    use, fx, fy, pol = gr.session_depth_setup("sess", s, _cfg())
    assert use is False
    assert "height veto off" in capsys.readouterr().out


def test_requiring_depth_on_a_session_without_it_is_a_hard_error(tmp_path):
    """So a run you believe is depth-gated cannot quietly not be."""
    s = _session(tmp_path, depth_kind="preview")
    with pytest.raises(SystemExit, match="not 'metric'"):
        gr.session_depth_setup("sess", s, _cfg(USE_DEPTH_HEIGHT=True))


def test_depth_can_be_switched_off_entirely(tmp_path):
    s = _session(tmp_path, depth_kind="metric")
    assert gr.session_depth_setup("sess", s, _cfg(USE_DEPTH_HEIGHT=False))[0] \
        is False


def test_a_frame_without_a_depth_png_yields_none(tmp_path):
    s = _session(tmp_path, depth_kind="metric")
    assert gr.load_frame_depth(s, "missing.png", True, None) == (None, None)


def test_no_depth_is_read_when_the_veto_is_off(tmp_path):
    s = _session(tmp_path, depth_kind="metric")
    assert gr.load_frame_depth(s, "any.png", False, None) == (None, None)


# --------------------------------------------------------------------------- #
# Every prelabeler can use it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", [
    "annotation/prelabel_weeds_sam3.py",
    "annotation/prelabel_onions_sam3.py",
    "annotation/prelabel_mixed_sam3.py",
    "annotation/prelabel_complement_sam3.py",
])
def test_every_prelabeler_offers_the_height_veto(script):
    """A single-class scene has the same pebbly ground as a mixed one, so
    leaving one prelabeler without it just moves the problem."""
    mod = load_script(script)
    assert mod.CONFIG["USE_DEPTH_HEIGHT"] in ("auto", True, False)
    assert mod.CONFIG["HEIGHT_MIN_MM"] > 0


def test_the_weed_default_is_lower_than_the_onion_one():
    """Some broadleaf weeds grow PROSTRATE, pressed flat to the soil - real
    targets with almost no height. Onions stand up; a weed frame is not the
    same problem, and the same threshold on both would delete the flattest
    weeds first."""
    wd = load_script("annotation/prelabel_weeds_sam3.py")
    on = load_script("annotation/prelabel_onions_sam3.py")
    assert wd.CONFIG["HEIGHT_MIN_MM"] < on.CONFIG["HEIGHT_MIN_MM"]
