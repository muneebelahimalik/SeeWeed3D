"""Session splits, ROI coordinate round-trip, LEP heatmap targets, joint
augmentation alignment, and depth handling. All numpy - no torch, no GPU."""
import numpy as np
import pytest

from conftest import load_script

sp = load_script("training/splits.py")
roi = load_script("training/roi.py")
lt = load_script("training/lep_targets.py")
cfgm = load_script("training/config.py")


# --------------------------------------------------------------------------- #
# Session-safe splits
# --------------------------------------------------------------------------- #
def test_every_session_lands_in_exactly_one_split():
    sessions = [f"s{i:02d}" for i in range(10)]
    out = sp.assign_splits(sessions, 0.2, 0.2, seed=1)
    allocated = out["train"] + out["val"] + out["test"]
    assert sorted(allocated) == sorted(sessions)
    assert len(allocated) == len(set(allocated))


def test_split_assignment_is_reproducible():
    """hash() is salted per process by PYTHONHASHSEED; a split built on it
    would differ between runs. crc32 keeps it stable."""
    s = [f"sess_{i}" for i in range(12)]
    assert sp.assign_splits(s, seed=7) == sp.assign_splits(s, seed=7)
    assert sp.assign_splits(s, seed=7) != sp.assign_splits(s, seed=8)


def test_explicit_holdouts_override_automatic_assignment():
    s = [f"s{i}" for i in range(8)]
    out = sp.assign_splits(s, seed=3, holdout_test=["s0"], holdout_val=["s1"])
    assert "s0" in out["test"] and "s1" in out["val"]
    assert "s0" not in out["train"] and "s1" not in out["train"]


def test_conflicting_holdouts_fail():
    with pytest.raises(sp.SplitError):
        sp.assign_splits(["a", "b", "c"], holdout_val=["a"], holdout_test=["a"])


def test_unknown_holdout_session_fails_clearly():
    with pytest.raises(sp.SplitError) as e:
        sp.assign_splits(["a", "b"], holdout_test=["nope"])
    assert "nope" in str(e.value)


def test_frame_level_leakage_is_detected():
    """THE test this module exists for. Adjacent frames are near-identical, so
    a frame on both sides of the split measures memorisation."""
    split_map = {"train": ["s1"], "val": ["s2"], "test": []}
    ok = {"s1_000001.png": "s1", "s2_000001.png": "s2"}
    assert sp.check_no_leakage(split_map, ok)

    # The same frame claimed by two sessions in different splits.
    with pytest.raises(sp.SplitError):
        sp.check_no_leakage({"train": ["s1"], "val": ["s1"], "test": []}, ok)


def test_frame_with_no_resolvable_session_fails():
    with pytest.raises(sp.SplitError) as e:
        sp.check_no_leakage({"train": ["s1"], "val": [], "test": []},
                            {"orphan.png": "unknown_session"})
    assert "unknown_session" in str(e.value)


def test_same_day_field_camera_sessions_stay_together():
    """Two recordings of the same bed on the same morning are near-duplicates;
    separating them leaks just as a frame split does."""
    infos = [sp.SessionInfo("a1", date="2026-01-08", field_id="f1", camera="zed"),
             sp.SessionInfo("a2", date="2026-01-08", field_id="f1", camera="zed"),
             sp.SessionInfo("b1", date="2026-02-02", field_id="f2", camera="zed"),
             sp.SessionInfo("b2", date="2026-02-02", field_id="f2", camera="zed"),
             sp.SessionInfo("c1", date="2026-03-11", field_id="f3", camera="zed"),
             sp.SessionInfo("c2", date="2026-03-11", field_id="f3", camera="zed")]
    out = sp.assign_splits(infos, 0.34, 0.34, seed=5)
    where = {s: k for k, v in out.items() for s in v}
    assert where["a1"] == where["a2"]
    assert where["b1"] == where["b2"]
    assert where["c1"] == where["c2"]
    assert out["train"], "training must never be starved by group quotas"


def test_training_split_is_never_left_empty():
    """A whole group is indivisible, so a few large groups could otherwise
    consume the val/test quotas entirely. Empty training is never the intended
    outcome of a fraction."""
    infos = [sp.SessionInfo("x1", date="d1", field_id="f", camera="c"),
             sp.SessionInfo("x2", date="d1", field_id="f", camera="c"),
             sp.SessionInfo("y1", date="d2", field_id="f", camera="c"),
             sp.SessionInfo("y2", date="d2", field_id="f", camera="c")]
    out = sp.assign_splits(infos, 0.25, 0.25, seed=5)
    assert out["train"]
    assert sum(len(v) for v in out.values()) == 4


def test_write_splits_emits_files_and_summary(tmp_path):
    infos = [sp.SessionInfo(f"s{i}", date=f"2026-01-{i+1:02d}", n_frames=10)
             for i in range(6)]
    out = sp.assign_splits(infos, 0.2, 0.2, seed=2)
    frames = {i.session_id: [f"{i.session_id}/rgb/f{j}.png" for j in range(3)]
              for i in infos}
    summary = sp.write_splits(tmp_path, out, frames, infos)
    for split in sp.SPLITS:
        assert (tmp_path / f"{split}_sessions.txt").exists()
        assert (tmp_path / f"{split}_images.txt").exists()
    assert (tmp_path / "splits_summary.json").exists()
    assert sum(v["n_sessions"] for v in summary["splits"].values()) == 6
    # Manifests use posix separators so they cross Windows/Linux.
    body = (tmp_path / "train_images.txt").read_text()
    assert "\\" not in body


# --------------------------------------------------------------------------- #
# ROI transform
# --------------------------------------------------------------------------- #
def test_roi_coordinate_round_trip_is_exact():
    """A systematic half-pixel error here becomes a permanent aiming bias."""
    tf = roi.make_transform((100.0, 50.0, 40.0, 60.0), 128, 1.4, 640, 480)
    for u, v in [(100.0, 50.0), (120.5, 80.25), (139.0, 109.0)]:
        ur, vr = tf.to_roi(u, v)
        ub, vb = tf.to_full(ur, vr)
        assert abs(ub - u) < 1e-9 and abs(vb - v) < 1e-9


def test_roi_scaling_is_isotropic_and_pads_rather_than_stretches():
    """Aspect ratio must survive: stretching changes the apparent angles between
    leaves, which is the phyllotactic structure the LEP head reads.

    The transform carries ONE scalar `scale`, so x and y are scaled identically
    by construction. A non-square crop (which happens when the square ROI is
    clamped at a frame edge) is then letterboxed, never stretched."""
    tf = roi.make_transform((0.0, 0.0, 80.0, 40.0), 128, 1.0, 640, 480)
    assert isinstance(tf.scale, float)          # single factor => isotropic
    dx = tf.to_roi(10.0, 0.0)[0] - tf.to_roi(0.0, 0.0)[0]
    dy = tf.to_roi(0.0, 10.0)[1] - tf.to_roi(0.0, 0.0)[1]
    assert abs(dx - dy) < 1e-9                  # same px/px on both axes

    # Clamped at the frame edge the crop is non-square, so it is padded.
    edge = roi.make_transform((600.0, 10.0, 80.0, 80.0), 128, 1.4, 640, 480)
    assert edge.src_w != edge.src_h
    assert abs(edge.pad_x) + abs(edge.pad_y) > 0
    assert edge.pad_x >= 0 and edge.pad_y >= 0


def test_extract_roi_moves_image_mask_and_point_together():
    bgr = np.zeros((200, 200, 3), np.uint8)
    mask = np.zeros((200, 200), bool)
    mask[80:120, 80:120] = True
    bgr[mask] = (40, 200, 40)
    lep = (100.0, 100.0)

    tf = roi.make_transform((80.0, 80.0, 40.0, 40.0), 64, 1.5, 200, 200)
    out = roi.extract_roi(bgr, mask, tf)
    u, v = tf.to_roi(*lep)
    assert out["mask"][int(round(v)), int(round(u))]      # point still on plant
    assert out["mask"].dtype == bool
    assert out["rgb"].shape == (64, 64, 3)


def test_local_height_is_camera_transferable_not_raw_depth():
    """Height above the LOCAL soil ring is the transferable quantity: raw depth
    encodes mount height, so a model fed it fails when the rig is raised."""
    cfg = cfgm.DepthRepresentationConfig()
    size = 80
    mask = np.zeros((size, size), bool)
    mask[30:50, 30:50] = True

    # Same plant, two mount heights: soil at 1000mm vs 1500mm, plant 50mm above.
    for soil in (1000.0, 1500.0):
        depth = np.full((size, size), soil, np.float32)
        depth[mask] = soil - 50.0
        h, valid = roi.local_height_map(depth, mask, cfg)
        assert valid.any()
        plant_h = float(np.median(h[mask]))
        soil_h = float(np.median(h[~mask & valid]))
        assert plant_h > soil_h                     # plant reads higher
        # The normalised plant height must be ~identical at both mount heights.
        if soil == 1000.0:
            first = plant_h
        else:
            assert abs(plant_h - first) < 1e-3


def test_height_map_abstains_when_soil_ring_is_unreliable():
    """No reference beats an invented one: a guessed reference biases every
    height on the plant."""
    cfg = cfgm.DepthRepresentationConfig()
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    depth = np.full((60, 60), np.nan, np.float32)
    depth[mask] = 900.0                       # plant only, no valid soil ring
    h, valid = roi.local_height_map(depth, mask, cfg)
    assert not valid.any() or float(h.max()) == 0.0


def test_geometry_channels_keep_validity_separate():
    """'flat ground' and 'no measurement' must be distinguishable, which a
    single zero-filled channel cannot express."""
    cfg = cfgm.DepthRepresentationConfig()
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    r = {"mask": mask, "depth_mm": None}
    g = roi.build_geometry_channels(r, cfg)
    assert g.shape == (3, 40, 40)
    assert g[0].max() == 1.0                  # ownership present
    assert g[2].max() == 0.0                  # validity says "no depth"


# --------------------------------------------------------------------------- #
# Heatmap targets and decoding
# --------------------------------------------------------------------------- #
def test_heatmap_peaks_at_the_annotated_point():
    cfg = cfgm.HeatmapConfig(stride=4, sigma_px=2.0)
    hm, uv_hm = lt.make_heatmap((64.0, 32.0), 128, cfg)
    assert hm.shape == (32, 32)
    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
    assert abs(px - uv_hm[0]) <= 1 and abs(py - uv_hm[1]) <= 1
    # The Gaussian is rendered about the exact SUB-PIXEL location, so no grid
    # cell reaches 1.0 unless the point happens to land on a cell centre. That
    # is the sub-pixel fidelity being preserved, not a defect.
    assert 0.9 <= hm.max() <= 1.0

    # A point placed exactly on a heatmap cell centre does reach 1.0.
    on_grid = lt.heatmap_to_roi((7.0, 5.0), cfg.stride)
    hm2, _ = lt.make_heatmap(on_grid, 128, cfg)
    assert abs(float(hm2.max()) - 1.0) < 1e-5
    assert np.unravel_index(int(np.argmax(hm2)), hm2.shape) == (5, 7)


def test_soft_argmax_recovers_a_known_subpixel_point():
    """Sub-pixel recovery is the reason for soft-argmax: an integer argmax is
    floor-limited by the stride (4px here)."""
    cfg = cfgm.HeatmapConfig(stride=4, sigma_px=2.0)
    for target in [(64.0, 32.0), (50.0, 77.0), (13.5, 91.25)]:
        hm, uv_hm = lt.make_heatmap(target, 128, cfg)
        got = lt.soft_argmax(hm)
        assert abs(got[0] - uv_hm[0]) < 0.25
        assert abs(got[1] - uv_hm[1]) < 0.25
        back = lt.heatmap_to_roi(got, cfg.stride)
        assert abs(back[0] - target[0]) < 1.0
        assert abs(back[1] - target[1]) < 1.0


def test_heatmap_roi_mapping_is_invertible():
    cfg = cfgm.HeatmapConfig(stride=4)
    for p in [(0.0, 0.0), (63.5, 12.25), (127.0, 127.0)]:
        _, uv_hm = lt.make_heatmap(p, 128, cfg)
        back = lt.heatmap_to_roi(uv_hm, cfg.stride)
        assert abs(back[0] - p[0]) < 1e-6 and abs(back[1] - p[1]) < 1e-6


def test_uncertainty_grows_with_heatmap_spread():
    """sigma_px is the abstention signal, so it must actually track ambiguity."""
    cfg_tight = cfgm.HeatmapConfig(stride=4, sigma_px=1.0)
    cfg_broad = cfgm.HeatmapConfig(stride=4, sigma_px=5.0, sigma_max_px=8.0)
    tight, _ = lt.make_heatmap((64.0, 64.0), 128, cfg_tight)
    broad, _ = lt.make_heatmap((64.0, 64.0), 128, cfg_broad)
    assert lt.heatmap_uncertainty(tight)["sigma_px"] < \
        lt.heatmap_uncertainty(broad)["sigma_px"]


def test_not_visible_samples_get_zero_supervision_weight():
    """No LEP exists to regress; supervising zeros would teach that the plant
    has no growth point, which is false."""
    cfg = cfgm.HeatmapConfig(partial_visibility_weight=0.5)
    assert lt.target_weight("visible", cfg) == 1.0
    assert lt.target_weight("partially_occluded_inferable", cfg) == 0.5
    assert lt.target_weight("not_visible", cfg) == 0.0


def test_sigma_can_scale_with_plant_size():
    cfg = cfgm.HeatmapConfig(stride=4, sigma_scale_with_plant=0.15,
                             sigma_min_px=0.5, sigma_max_px=10.0)
    small = lt.resolve_sigma(cfg, plant_radius_px=20)
    large = lt.resolve_sigma(cfg, plant_radius_px=120)
    assert large > small


def test_decode_lep_returns_full_frame_coordinates():
    cfg = cfgm.HeatmapConfig(stride=4, sigma_px=2.0)
    tf = roi.make_transform((100.0, 100.0, 40.0, 40.0), 128, 1.4, 640, 480)
    true_full = (118.0, 122.0)
    uv_roi = tf.to_roi(*true_full)
    hm, _ = lt.make_heatmap(uv_roi, 128, cfg)
    got = lt.decode_lep(hm, tf, cfg)
    assert abs(got["uv_full"][0] - true_full[0]) < 1.5
    assert abs(got["uv_full"][1] - true_full[1]) < 1.5
    assert got["sigma_px"] > 0 and got["peak"] > 0.9


# --------------------------------------------------------------------------- #
# Joint augmentation
# --------------------------------------------------------------------------- #
def _plant_scene(size=96):
    rgb = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), bool)
    mask[30:60, 20:50] = True
    rgb[mask] = (40, 180, 60)
    lep = (34.0, 44.0)
    rgb[int(lep[1]), int(lep[0])] = (255, 255, 255)
    return rgb, mask, lep


def test_augmentation_keeps_point_on_its_plant():
    """RGB, mask and LEP must move under ONE transform - a point that drifts off
    its own mask is a wrong training target."""
    rgb, mask, lep = _plant_scene()
    for seed in range(12):
        aug = lt.JointAugment(seed=seed, max_rotate_deg=12.0, scale_jitter=0.05)
        r2, m2, uv2, _, _ = aug(rgb.copy(), mask.copy(), lep)
        x, y = int(round(uv2[0])), int(round(uv2[1]))
        assert 0 <= x < m2.shape[1] and 0 <= y < m2.shape[0]
        # Allow a 2px margin for interpolation at the mask boundary.
        y0, y1 = max(0, y - 2), min(m2.shape[0], y + 3)
        x0, x1 = max(0, x - 2), min(m2.shape[1], x + 3)
        assert m2[y0:y1, x0:x1].any(), f"seed {seed}: LEP left its own mask"


def test_horizontal_flip_moves_the_point_exactly():
    rgb, mask, lep = _plant_scene()
    aug = lt.JointAugment(hflip=1.0, vflip=0.0, rot90=False, max_rotate_deg=0,
                          scale_jitter=0, brightness=0, seed=0)
    _, _, uv2, _, _ = aug(rgb.copy(), mask.copy(), lep)
    assert abs(uv2[0] - ((mask.shape[1] - 1) - lep[0])) < 1e-6
    assert abs(uv2[1] - lep[1]) < 1e-6


def test_compositing_augmentations_are_refused_not_ignored():
    """Mosaic/MixUp blend pixels from different plants into one ROI, destroying
    the ownership the task is built on. A silently ignored request would look
    like it had been applied."""
    for bad in ("mosaic", "mixup", "copy_paste"):
        with pytest.raises(lt.AugmentationError) as e:
            lt.JointAugment(forbid=[bad])
        assert "ownership" in str(e.value).lower()


def test_depth_moves_with_the_image_under_augmentation():
    rgb, mask, lep = _plant_scene()
    depth = np.full(mask.shape, 1000.0, np.float32)
    depth[mask] = 950.0
    aug = lt.JointAugment(hflip=1.0, vflip=0.0, rot90=False, max_rotate_deg=0,
                          scale_jitter=0, brightness=0, seed=0)
    _, m2, uv2, d2, _ = aug(rgb.copy(), mask.copy(), lep, depth_mm=depth.copy())
    assert d2 is not None
    near = d2[m2]
    assert np.nanmedian(near) < 1000.0        # plant depth followed the flip


# --------------------------------------------------------------------------- #
# Depth degradation
# --------------------------------------------------------------------------- #
def test_depth_dropout_produces_the_no_depth_case():
    """The RGB path must survive with no depth at all - that is the field
    reality this simulates."""
    cfg = cfgm.DepthRepresentationConfig(depth_dropout_p=1.0)
    rng = np.random.default_rng(0)
    d, dropped = lt.simulate_depth_degradation(
        np.full((20, 20), 900.0, np.float32), cfg, rng)
    assert d is None and dropped is True


def test_depth_degradation_adds_holes_and_noise_but_keeps_scale():
    cfg = cfgm.DepthRepresentationConfig(depth_dropout_p=0.0,
                                         hole_dropout_p=0.3,
                                         noise_mm_per_m=5.0)
    rng = np.random.default_rng(1)
    clean = np.full((64, 64), 900.0, np.float32)
    d, dropped = lt.simulate_depth_degradation(clean, cfg, rng)
    assert not dropped and d is not None
    assert np.isnan(d).any()                       # holes appeared
    finite = d[np.isfinite(d)]
    assert 850.0 < float(np.median(finite)) < 950.0   # scale preserved
