"""Checks for the multi-evidence LEP estimator.

The decisive test is the ASYMMETRIC rosette: when a plant has more/longer leaves
on one side, the mask centroid is pulled away from the true growth point. A
defensible LEP estimator must stay at the meristem, and beat the centroid.
"""
import numpy as np
import cv2
import pytest

from conftest import load_script

lep = load_script("perception/lep.py")


def make_rosette(size=220, cx=110, cy=110, leaves=((0, 70), (45, 70), (90, 70),
                                                   (135, 70), (180, 70), (225, 70),
                                                   (270, 70), (315, 70)),
                 core_r=13, leaf_w=11):
    """Rosette mask + image. `leaves` is (angle_deg, length_px) per leaf, so an
    asymmetric plant can be built by giving one side longer leaves."""
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (cx, cy), core_r, 1, -1)
    for ang, length in leaves:
        a = np.deg2rad(ang)
        x2, y2 = int(cx + length * np.cos(a)), int(cy + length * np.sin(a))
        cv2.line(m, (cx, cy), (x2, y2), 1, leaf_w)
    mask = m.astype(bool)

    bgr = np.full((size, size, 3), (70, 45, 60), np.uint8)      # soil
    bgr[mask] = (35, 110, 40)                                    # mature leaf
    # Young tissue at the centre: lighter and more yellow-green.
    young = np.zeros((size, size), np.uint8)
    cv2.circle(young, (cx, cy), core_r, 1, -1)
    bgr[young.astype(bool) & mask] = (70, 190, 120)
    return mask, bgr


def make_depth(mask, size, cx, cy, crown_r=16, soil_mm=900.0, crown_mm=22.0):
    """Depth where the crown is nearer the camera (elevated) than soil."""
    d = np.full((size, size), soil_mm, np.float32)
    d[mask] = soil_mm - 6.0                                    # leaves slightly up
    yy, xx = np.mgrid[0:size, 0:size]
    bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (crown_r ** 2)))
    d -= crown_mm * bump                                        # crown nearest
    return d


def test_symmetric_rosette_lep_at_centre():
    mask, bgr = make_rosette()
    r = lep.LEPEstimator().estimate(
        lep.PlantContext(mask=mask, bgr=bgr, class_name="wild_radish"))
    assert r is not None
    x, y = r.uv
    assert abs(x - 110) < 10 and abs(y - 110) < 10
    assert r.visibility in ("visible", "partially_occluded_inferable")
    assert 0.0 <= r.confidence <= 1.0
    assert len(r.channels) >= 3          # several independent channels contributed


def test_asymmetric_rosette_beats_centroid():
    """One side has much longer leaves, so the centroid is dragged off the
    meristem. The fused estimate must stay closer to the true growth point."""
    cx = cy = 110
    leaves = [(0, 95), (30, 95), (330, 95),        # long leaves to the right
              (90, 40), (135, 40), (180, 40), (225, 40), (270, 40)]
    mask, bgr = make_rosette(cx=cx, cy=cy, leaves=leaves)

    ys, xs = np.nonzero(mask)
    centroid = np.array([xs.mean(), ys.mean()])
    truth = np.array([cx, cy])
    centroid_err = float(np.linalg.norm(centroid - truth))
    assert centroid_err > 8, "test setup should actually displace the centroid"

    r = lep.LEPEstimator().estimate(
        lep.PlantContext(mask=mask, bgr=bgr, class_name="wild_radish"))
    lep_err = float(np.linalg.norm(np.array(r.uv) - truth))
    assert lep_err < centroid_err, f"LEP {lep_err:.1f}px worse than centroid {centroid_err:.1f}px"


def test_depth_channel_used_only_when_available():
    mask, bgr = make_rosette()
    ctx_nodepth = lep.PlantContext(mask=mask, bgr=bgr, class_name="wild_radish")
    ctx_depth = lep.PlantContext(mask=mask, bgr=bgr, class_name="wild_radish",
                                 depth_mm=make_depth(mask, 220, 110, 110))
    est = lep.LEPEstimator()
    assert "canopy_height" not in est.estimate(ctx_nodepth).channels
    r = est.estimate(ctx_depth)
    assert "canopy_height" in r.channels
    # The crown bump is at the centre, so the height channel must vote there.
    hx, hy = r.channels["canopy_height"]["uv"]
    assert abs(hx - 110) < 15 and abs(hy - 110) < 15


def test_channels_are_independent_and_agree_on_a_clean_plant():
    """Agreement between independent channels is the core scientific argument:
    on an unambiguous plant they should converge on the same point."""
    mask, bgr = make_rosette()
    r = lep.LEPEstimator().estimate(
        lep.PlantContext(mask=mask, bgr=bgr, class_name="wild_radish",
                         depth_mm=make_depth(mask, 220, 110, 110)))
    radius = float(np.sqrt(mask.sum() / np.pi))
    assert r.agreement_px < 0.35 * radius
    assert r.sigma_px > 0 and len(r.covariance) == 2


def test_young_tissue_channel_points_at_light_centre():
    """The physiological channel must key on the pale young tissue, which is
    what makes this a leaf EMERGENCE point rather than a shape centre."""
    mask, bgr = make_rosette()
    ch = lep.YoungTissueEvidence()
    s = ch.score(lep.PlantContext(mask=mask, bgr=bgr))
    py, px = np.unravel_index(int(np.argmax(s)), s.shape)
    assert abs(px - 110) < 18 and abs(py - 110) < 18


def test_petiole_convergence_points_at_hub():
    mask, bgr = make_rosette()
    ch = lep.PetioleConvergenceEvidence()
    s = ch.score(lep.PlantContext(mask=mask, bgr=bgr))
    py, px = np.unravel_index(int(np.argmax(s)), s.shape)
    assert abs(px - 110) < 20 and abs(py - 110) < 20


def test_thinning_produces_thin_skeleton():
    m = np.zeros((60, 60), bool)
    m[20:40, 10:50] = True
    sk = lep.zhang_suen_thin(m)
    assert sk.any() and sk.sum() < m.sum() / 4


def test_abstains_on_shapeless_blob():
    """A blob with no radial structure and no young-tissue cue should not be
    reported as a confident LEP - abstention is safer than a wrong target."""
    mask = np.zeros((120, 120), bool)
    mask[30:90, 30:90] = True                   # featureless square
    bgr = np.full((120, 120, 3), (60, 100, 60), np.uint8)
    r = lep.LEPEstimator().estimate(
        lep.PlantContext(mask=mask, bgr=bgr, class_name="other_weed"))
    assert r is not None
    assert r.confidence < 0.95


def test_work_resolution_cap_keeps_accuracy_and_bounds_cost():
    """Evidence is computed on a crop capped at max_work_px so per-instance cost
    does not blow up on large rosettes. The mapped-back estimate must still land
    on the growth point, and lengths/covariance must be rescaled with it."""
    size = cx = cy = None
    size, cx, cy = 340, 170, 170
    leaves = tuple((a, 130) for a in range(0, 360, 45))
    mask, bgr = make_rosette(size=size, cx=cx, cy=cy, leaves=leaves,
                             core_r=22, leaf_w=20)
    ctx = lep.PlantContext(mask=mask, bgr=bgr, class_name="wild_radish")

    capped = lep.LEPEstimator(max_work_px=160).estimate(ctx)
    full = lep.LEPEstimator(max_work_px=None).estimate(ctx)

    # Both land on the meristem; capping costs a little precision, not the answer.
    for r in (capped, full):
        assert abs(r.uv[0] - cx) < 12 and abs(r.uv[1] - cy) < 12
    # Scaled quantities come back in input-crop pixels, not working pixels.
    assert capped.sigma_px > 1.0
    assert np.isfinite(capped.covariance[0][0]) and capped.covariance[0][0] > 0
    for c in capped.channels.values():
        assert 0 <= c["uv"][0] <= size and 0 <= c["uv"][1] <= size


def test_crop_context_maps_back_to_full_frame():
    full_mask = np.zeros((300, 300), bool)
    sub, bgr_sub = make_rosette(size=120, cx=60, cy=60,
                                leaves=((0, 35), (90, 35), (180, 35), (270, 35)))
    full_mask[100:220, 140:260] = sub
    full_bgr = np.full((300, 300, 3), (70, 45, 60), np.uint8)
    full_bgr[100:220, 140:260] = bgr_sub
    ctx = lep.crop_context(full_mask, full_bgr, (140, 100, 120, 120), pad=4)
    r = lep.LEPEstimator().estimate(ctx)
    # Growth point is at (140+60, 100+60) = (200, 160) in full-frame coords.
    assert abs(r.uv[0] - 200) < 15 and abs(r.uv[1] - 160) < 15
    assert r.uv_local != r.uv
