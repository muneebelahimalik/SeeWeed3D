"""Safety abstention, robust depth->3D localisation, and the structured
inference output. These are the paths that decide whether a 60 W laser gets a
target, so every rejection route is exercised explicitly."""
import numpy as np
import pytest

from conftest import load_script

sf = load_script("perception/safety.py")
d3 = load_script("perception/depth3d.py")
schema = load_script("perception/schema.py")
cfgm = load_script("training/config.py")

CFG = cfgm.SafetyConfig()


def _mask(size=200, box=(60, 60, 60, 60)):
    m = np.zeros((size, size), bool)
    x, y, w, h = box
    m[y:y + h, x:x + w] = True
    return m


def _lep(u, v, peak=0.9, sigma=3.0):
    return {"uv_full": (u, v), "peak": peak, "sigma_px": sigma,
            "covariance": [[1.0, 0.0], [0.0, 1.0]]}


def _ok_depth():
    return {"ok": True, "reason": "ok", "xyz_mm": [1.0, 2.0, 900.0],
            "sigma_mm": 3.0,
            "depth_stats": {"valid_fraction": 0.9, "spread_mm": 5.0}}


# --------------------------------------------------------------------------- #
# Acceptance, so the rejections below mean something
# --------------------------------------------------------------------------- #
def test_a_clean_weed_becomes_a_candidate():
    d = sf.decide(class_name="wild_radish", class_confidence=0.9,
                  lep=_lep(90, 90), instance_mask=_mask(),
                  onion_mask=None, cfg=CFG, visibility="visible",
                  visibility_conf=0.9, targetable="yes", targetable_conf=0.9,
                  depth_result=_ok_depth())
    assert d.is_candidate and not d.abstained and d.reasons == []


# --------------------------------------------------------------------------- #
# Every rejection route
# --------------------------------------------------------------------------- #
def test_onion_is_never_a_target():
    d = sf.decide(class_name="onion_plant", class_confidence=0.99,
                  lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                  cfg=CFG, visibility="visible", targetable="yes")
    assert not d.is_candidate and sf.R_ONION in d.reasons


def test_weed_cluster_is_rejected():
    """A cluster has no separable single growth point, so there is nothing to
    aim at even though it is unambiguously weed tissue."""
    d = sf.decide(class_name="weed_cluster", class_confidence=0.95,
                  lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                  cfg=CFG, visibility="visible", targetable="yes")
    assert not d.is_candidate and sf.R_CLUSTER in d.reasons


def test_onion_safety_conflict_uses_the_spot_radius_not_the_centre_pixel():
    """The beam has physical extent: a spot whose centre clears the crop but
    whose edge does not would still damage it."""
    onion = _mask(box=(100, 60, 60, 60))
    near = sf.decide(class_name="wild_radish", class_confidence=0.9,
                     lep=_lep(95, 90), instance_mask=_mask(),
                     onion_mask=onion, cfg=CFG, visibility="visible",
                     visibility_conf=0.9, targetable="yes", targetable_conf=0.9)
    assert sf.R_ONION_CONFLICT in near.reasons

    far = sf.decide(class_name="wild_radish", class_confidence=0.9,
                    lep=_lep(65, 90), instance_mask=_mask(),
                    onion_mask=onion, cfg=CFG, visibility="visible",
                    visibility_conf=0.9, targetable="yes", targetable_conf=0.9,
                    depth_result=_ok_depth())
    assert sf.R_ONION_CONFLICT not in far.reasons


def test_lep_outside_its_owning_mask_is_rejected():
    """A LEP on the neighbour's crown is the wrong-instance failure."""
    d = sf.decide(class_name="wild_radish", class_confidence=0.9,
                  lep=_lep(10, 10), instance_mask=_mask(), onion_mask=None,
                  cfg=CFG, visibility="visible", targetable="yes")
    assert sf.R_OUTSIDE_MASK in d.reasons


def test_tiny_outside_offset_is_snapped_and_recorded_never_silent():
    """A sub-pixel correction is permitted, but it must be visible in the
    result - a silent reprojection hides a real disagreement."""
    m = _mask(box=(60, 60, 60, 60))          # spans x=60..119
    d = sf.decide(class_name="wild_radish", class_confidence=0.9,
                  lep=_lep(120.5, 90), instance_mask=m, onion_mask=None,
                  cfg=CFG, visibility="visible", visibility_conf=0.9,
                  targetable="yes", targetable_conf=0.9,
                  depth_result=_ok_depth())
    assert d.snapped_px > 0
    assert d.notes.get("snapped") is True
    assert sf.R_OUTSIDE_MASK not in d.reasons

    # Beyond the snap tolerance it is a rejection, not a bigger nudge.
    far = sf.decide(class_name="wild_radish", class_confidence=0.9,
                    lep=_lep(126.0, 90), instance_mask=m, onion_mask=None,
                    cfg=CFG, visibility="visible", targetable="yes")
    assert sf.R_OUTSIDE_MASK in far.reasons
    assert far.snapped_px == 0.0


def test_low_confidence_and_high_uncertainty_abstain():
    low = sf.decide(class_name="wild_radish", class_confidence=0.9,
                    lep=_lep(90, 90, peak=0.05), instance_mask=_mask(),
                    onion_mask=None, cfg=CFG, visibility="visible",
                    targetable="yes")
    assert sf.R_LOW_CONF in low.reasons

    unc = sf.decide(class_name="wild_radish", class_confidence=0.9,
                    lep=_lep(90, 90, sigma=99.0), instance_mask=_mask(),
                    onion_mask=None, cfg=CFG, visibility="visible",
                    targetable="yes")
    assert sf.R_HIGH_UNC in unc.reasons


def test_not_visible_and_not_targetable_abstain():
    nv = sf.decide(class_name="wild_radish", class_confidence=0.9,
                   lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                   cfg=CFG, visibility="not_visible", targetable="yes")
    assert sf.R_NOT_VISIBLE in nv.reasons

    for verdict in ("no", "uncertain"):
        nt = sf.decide(class_name="wild_radish", class_confidence=0.9,
                       lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                       cfg=CFG, visibility="visible", targetable=verdict)
        assert sf.R_NOT_TARGETABLE in nt.reasons


def test_uncertain_class_abstains():
    d = sf.decide(class_name="wild_radish", class_confidence=0.10,
                  lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                  cfg=CFG, visibility="visible", targetable="yes")
    assert sf.R_CLASS_UNCERTAIN in d.reasons


def test_missing_lep_abstains():
    d = sf.decide(class_name="wild_radish", class_confidence=0.9, lep=None,
                  instance_mask=_mask(), onion_mask=None, cfg=CFG,
                  visibility="visible", targetable="yes")
    assert sf.R_NO_LEP in d.reasons and not d.is_candidate


def test_bad_depth_abstains_with_the_specific_reason():
    disc = sf.decide(class_name="wild_radish", class_confidence=0.9,
                     lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                     cfg=CFG, visibility="visible", targetable="yes",
                     depth_result={"ok": False, "reason": "depth_discontinuity",
                                   "depth_stats": {}})
    assert sf.R_DEPTH_DISC in disc.reasons

    sparse = sf.decide(class_name="wild_radish", class_confidence=0.9,
                       lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                       cfg=CFG, visibility="visible", targetable="yes",
                       depth_result={"ok": False,
                                     "reason": "insufficient_valid_depth",
                                     "depth_stats": {}})
    assert sf.R_NO_DEPTH in sparse.reasons


def test_high_3d_uncertainty_abstains():
    bad = dict(_ok_depth())
    bad["sigma_mm"] = 500.0
    d = sf.decide(class_name="wild_radish", class_confidence=0.9,
                  lep=_lep(90, 90), instance_mask=_mask(), onion_mask=None,
                  cfg=CFG, visibility="visible", visibility_conf=0.9,
                  targetable="yes", targetable_conf=0.9, depth_result=bad)
    assert sf.R_3D_UNCERTAIN in d.reasons


def test_all_failures_are_reported_not_just_the_first():
    """One pass must explain every problem, so a batch can be triaged without
    iterating."""
    d = sf.decide(class_name="onion_plant", class_confidence=0.05,
                  lep=_lep(10, 10, peak=0.01, sigma=99.0),
                  instance_mask=_mask(), onion_mask=None, cfg=CFG,
                  visibility="not_visible", targetable="no")
    assert {sf.R_ONION, sf.R_CLASS_UNCERTAIN, sf.R_NOT_VISIBLE,
            sf.R_NOT_TARGETABLE, sf.R_LOW_CONF, sf.R_HIGH_UNC,
            sf.R_OUTSIDE_MASK}.issubset(set(d.reasons))


def test_every_reason_string_is_registered():
    """The control layer and the safety metrics key off these strings."""
    for r in (sf.R_ONION, sf.R_CLUSTER, sf.R_NOT_VISIBLE, sf.R_LOW_CONF,
              sf.R_HIGH_UNC, sf.R_OUTSIDE_MASK, sf.R_ONION_CONFLICT,
              sf.R_NO_DEPTH, sf.R_DEPTH_DISC, sf.R_NOT_TARGETABLE,
              sf.R_CLASS_UNCERTAIN):
        assert r in sf.ALL_REJECTION_REASONS


def test_safety_module_has_no_actuator_surface():
    """This module produces candidates only. It must not acquire a way to
    command hardware."""
    import inspect
    src = inspect.getsource(sf)
    for forbidden in ("serial", "socket", "requests", "subprocess", "gpio",
                      "fire_laser", "actuate"):
        assert forbidden not in src.lower()


# --------------------------------------------------------------------------- #
# Depth -> 3D
# --------------------------------------------------------------------------- #
K = (700.0, 700.0, 320.0, 240.0)


def test_backprojection_with_known_intrinsics():
    """A point at the principal point must land on the optical axis, and an
    offset pixel must produce the exact similar-triangles offset."""
    depth = np.full((480, 640), 900.0, np.float32)
    valid = np.ones((480, 640), bool)

    at_centre = d3.localize_lep_3d(depth, valid, (320.0, 240.0), K)
    assert at_centre["ok"]
    x, y, z = at_centre["xyz_mm"]
    assert abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z - 900.0) < 1e-6

    off = d3.localize_lep_3d(depth, valid, (390.0, 240.0), K)
    assert abs(off["xyz_mm"][0] - 900.0 * 70.0 / 700.0) < 1e-6


def test_depth_discontinuity_is_refused_not_averaged():
    """Averaging across a leaf/soil edge invents a distance no surface
    occupies."""
    depth = np.full((100, 100), 900.0, np.float32)
    depth[:, 50:] = 1400.0                       # 500mm step through the sample
    valid = np.ones((100, 100), bool)
    out = d3.localize_lep_3d(depth, valid, (50.0, 50.0), K)
    assert not out["ok"]
    assert out["reason"] == "depth_discontinuity"
    assert out["xyz_mm"] is None


def test_insufficient_valid_depth_abstains():
    depth = np.full((100, 100), np.nan, np.float32)
    valid = np.zeros((100, 100), bool)
    out = d3.localize_lep_3d(depth, valid, (50.0, 50.0), K)
    assert not out["ok"] and out["reason"] == "insufficient_valid_depth"


def test_owning_mask_excludes_a_neighbouring_plant_at_another_depth():
    """Restricting the sample to the owning instance is what stops a
    neighbour's leaf from dragging the estimate."""
    depth = np.full((100, 100), 900.0, np.float32)
    depth[:, 52:] = 1500.0                       # neighbour, much further away
    valid = np.ones((100, 100), bool)
    own = np.zeros((100, 100), bool)
    own[:, :52] = True
    out = d3.localize_lep_3d(depth, valid, (48.0, 50.0), K, mask=own)
    assert out["ok"]
    assert abs(out["xyz_mm"][2] - 900.0) < 5.0


def test_weighting_follows_the_models_own_confidence():
    depth = np.full((100, 100), 1000.0, np.float32)
    depth[45:50, 45:50] = 980.0
    valid = np.ones((100, 100), bool)
    w = np.zeros((100, 100), np.float32)
    w[45:50, 45:50] = 1.0                         # confidence on the near patch
    z_w, st = d3.sample_depth_weighted(depth, valid, (47.0, 47.0),
                                       weight_map=w, radius_px=8)
    assert st.get("weighted") is True
    assert abs(z_w - 980.0) < 5.0


def test_3d_uncertainty_grows_with_depth_spread():
    valid = np.ones((100, 100), bool)
    tight = np.full((100, 100), 900.0, np.float32)
    noisy = tight + np.random.default_rng(0).normal(0, 6.0, tight.shape).astype(
        np.float32)
    a = d3.localize_lep_3d(tight, valid, (50.0, 50.0), K)
    b = d3.localize_lep_3d(noisy, valid, (50.0, 50.0), K)
    assert a["ok"] and b["ok"]
    assert b["sigma_mm"] > a["sigma_mm"]


def test_bed_plane_fallback_is_marked_and_less_certain():
    """It assumes the growth point lies on the bed, which is wrong by the
    plant's own height - so it must never masquerade as a measurement."""
    plane = (np.array([0.0, 0.0, 1.0]), -900.0)   # z = 900mm
    out = d3.bed_plane_fallback((320.0, 240.0), K, plane, sigma_mm=40.0)
    assert out["ok"] and out["is_fallback"] is True
    assert out["used_depth"] is False
    assert abs(out["xyz_mm"][2] - 900.0) < 1e-6
    assert out["sigma_mm"] >= 40.0


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #
def test_result_schema_carries_every_required_field():
    t = schema.WeedTarget()
    d = t.to_dict()
    for key in ("session_id", "frame_id", "instance_index", "class_name",
                "class_confidence", "bbox_xywh", "mask_ref", "lep_uv",
                "lep_peak", "lep_sigma_px", "lep_covariance",
                "visibility_probs", "targetability_probs", "used_depth",
                "xyz_mm", "xyz_sigma_mm", "depth_stats", "safety_status",
                "abstained", "rejection_reasons"):
        assert key in d, f"missing required output field: {key}"


def test_frame_result_separates_candidates_from_abstentions():
    fr = schema.FrameResult(session_id="s1", frame_id="f1")
    good = schema.WeedTarget(safety_status=schema.STATUS_CANDIDATE,
                             abstained=False)
    bad = schema.WeedTarget(rejection_reasons=[sf.R_ONION_CONFLICT])
    fr.targets = [good, bad]
    assert len(fr.candidates) == 1 and len(fr.abstentions) == 1
    assert fr.reason_counts()[sf.R_ONION_CONFLICT] == 1
    assert fr.to_dict()["n_candidates"] == 1
