"""LEPRoiNet forward/backward and the multitask loss, on CPU with tiny tensors.

Skipped entirely when torch is absent, so the core suite still runs in the
lightweight (data-pipeline) environment."""
import numpy as np
import pytest

from conftest import load_script

torch = pytest.importorskip("torch", reason="torch is an optional training dep")

cfgm = load_script("training/config.py")
net_mod = load_script("training/lep_roinet.py")
loss_mod = load_script("training/losses.py")
lt = load_script("training/lep_targets.py")


def _cfg(mode="rgb_mask_geom", width=8, stride=4):
    return cfgm.ModelConfig(input_mode=mode, width=width, heatmap_stride=stride)


def _batch(b=2, size=32, n_geom=3):
    rgb = torch.rand(b, 3, size, size)
    geom = torch.rand(b, n_geom, size, size) if n_geom else None
    return rgb, geom


def test_forward_shapes_for_every_ablation():
    """All three input modes must run - the RGB-only and RGB+mask variants are
    the required ablations AND the fallbacks when depth is unavailable."""
    size = 32
    for mode, n_geom in (("rgb", 0), ("rgb_mask", 1), ("rgb_mask_geom", 3)):
        model = net_mod.build_model(_cfg(mode))
        assert net_mod.geometry_channels(mode) == n_geom
        rgb, geom = _batch(2, size, n_geom)
        out = model(rgb, geom)
        assert out["heatmap"].shape == (2, 1, size // 4, size // 4)
        assert out["visibility"].shape == (2, 3)
        assert out["targetability"].shape == (2, 3)


def test_rgb_only_model_runs_with_no_geometry_at_all():
    """Depth-free operation is a hard requirement: the field stream is often
    invalid, and the RGB path must not depend on it."""
    model = net_mod.build_model(_cfg("rgb"))
    out = model(torch.rand(2, 3, 32, 32), None)
    assert torch.isfinite(out["heatmap"]).all()


def test_geometry_model_survives_missing_depth_at_runtime():
    """A model configured with a geometry branch must still produce a finite
    prediction when the stream vanishes mid-run - the graph stays static
    (TensorRT wants fixed shapes) and depth dropout trained for this case."""
    model = net_mod.build_model(_cfg("rgb_mask_geom"))
    out = model(torch.rand(2, 3, 32, 32), None)      # geom omitted
    assert out["heatmap"].shape == (2, 1, 8, 8)
    assert torch.isfinite(out["heatmap"]).all()


def test_batched_forward_matches_per_sample_forward():
    """Deployment batches every ROI in a frame; that must not change results."""
    model = net_mod.build_model(_cfg("rgb_mask_geom")).eval()
    rgb, geom = _batch(4, 32, 3)
    with torch.no_grad():
        batched = model(rgb, geom)["heatmap"]
        singles = torch.cat([model(rgb[i:i + 1], geom[i:i + 1])["heatmap"]
                             for i in range(4)])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_tiny_cpu_training_step_reduces_the_loss():
    """The acceptance criterion: a real backward pass on CPU that learns."""
    torch.manual_seed(0)
    model = net_mod.build_model(_cfg("rgb_mask_geom", width=8))
    crit = loss_mod.LEPLoss(cfgm.LossWeights())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    size, stride, b = 32, 4, 4
    hm_size = size // stride
    rgb, geom = _batch(b, size, 3)
    hcfg = cfgm.HeatmapConfig(stride=stride, sigma_px=1.5)
    hms, coords = [], []
    for i in range(b):
        hm, uv = lt.make_heatmap((12.0 + i, 16.0), size, hcfg)
        hms.append(hm)
        coords.append(uv)
    targets = {
        "heatmap": torch.from_numpy(np.stack(hms)).unsqueeze(1).float(),
        "coord": torch.tensor(coords, dtype=torch.float32),
        "weight": torch.ones(b),
        "visibility": torch.zeros(b, dtype=torch.long),
        "targetability": torch.zeros(b, dtype=torch.long),
        "mask": torch.ones(b, 1, hm_size, hm_size)}

    first = None
    for step in range(25):
        opt.zero_grad()
        total, parts = crit(model(rgb, geom), targets)
        total.backward()
        # Gradients must actually reach the parameters.
        if step == 0:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
            first = float(total.detach())
        opt.step()
    assert float(total.detach()) < first, "loss did not decrease"
    assert set(parts) == {"heatmap", "soft_argmax", "visibility",
                          "targetability", "outside_mask"}


def test_soft_argmax_recovers_a_known_point_in_torch():
    """Same decode as the numpy path, so training and deployment agree."""
    hm = torch.zeros(1, 1, 16, 16)
    hm[0, 0, 5, 9] = 1.0
    xy = loss_mod.soft_argmax_2d(hm)
    assert abs(float(xy[0, 0]) - 9.0) < 1e-4
    assert abs(float(xy[0, 1]) - 5.0) < 1e-4


def test_torch_and_numpy_soft_argmax_agree():
    cfg = cfgm.HeatmapConfig(stride=4, sigma_px=2.0)
    hm, _ = lt.make_heatmap((50.0, 30.0), 128, cfg)
    np_xy = lt.soft_argmax(hm)
    t_xy = loss_mod.soft_argmax_2d(torch.from_numpy(hm)[None, None].float())
    assert abs(float(t_xy[0, 0]) - np_xy[0]) < 1e-3
    assert abs(float(t_xy[0, 1]) - np_xy[1]) < 1e-3


def test_not_visible_samples_contribute_no_localisation_loss():
    """Weight 0 must genuinely zero the localisation terms - otherwise the
    model is supervised toward an invented growth point."""
    torch.manual_seed(0)
    model = net_mod.build_model(_cfg("rgb", width=8))
    crit = loss_mod.LEPLoss(cfgm.LossWeights())
    b, size, stride = 2, 32, 4
    hm_size = size // stride
    rgb = torch.rand(b, 3, size, size)
    out = model(rgb, None)

    base = {"heatmap": torch.zeros(b, 1, hm_size, hm_size),
            "coord": torch.zeros(b, 2),
            "visibility": torch.zeros(b, dtype=torch.long),
            "targetability": torch.zeros(b, dtype=torch.long),
            "mask": torch.ones(b, 1, hm_size, hm_size)}

    _, zero_w = crit(out, {**base, "weight": torch.zeros(b)})
    assert zero_w["heatmap"] == pytest.approx(0.0, abs=1e-6)
    assert zero_w["soft_argmax"] == pytest.approx(0.0, abs=1e-6)

    _, full_w = crit(out, {**base, "weight": torch.ones(b)})
    assert full_w["heatmap"] > 0.0


def test_outside_mask_loss_penalises_mass_on_the_wrong_plant():
    """This is the term that teaches ownership - the difference between a safe
    target and the neighbouring plant's crown."""
    crit = loss_mod.LEPLoss(cfgm.LossWeights())
    b, s = 1, 8
    logits = torch.full((b, 1, s, s), 4.0)          # confident everywhere
    out = {"heatmap": logits, "visibility": torch.zeros(b, 3),
           "targetability": torch.zeros(b, 3)}
    base = {"heatmap": torch.zeros(b, 1, s, s), "coord": torch.zeros(b, 2),
            "weight": torch.ones(b),
            "visibility": torch.zeros(b, dtype=torch.long),
            "targetability": torch.zeros(b, dtype=torch.long)}

    full = torch.ones(b, 1, s, s)
    half = torch.zeros(b, 1, s, s)
    half[..., : s // 2] = 1.0
    _, all_inside = crit(out, {**base, "mask": full})
    _, half_outside = crit(out, {**base, "mask": half})
    assert half_outside["outside_mask"] > all_inside["outside_mask"]


def test_model_exports_to_onnx_when_the_exporter_is_available(tmp_path):
    """ONNX export is the first step of the TensorRT path. Numerical parity
    against PyTorch is asserted here; TensorRT parity needs real hardware and is
    left to seeweed3d/deploy/."""
    onnx = pytest.importorskip("onnx", reason="onnx not installed")
    model = net_mod.build_model(_cfg("rgb_mask_geom", width=8)).eval()
    rgb, geom = _batch(1, 32, 3)
    path = tmp_path / "lep.onnx"
    torch.onnx.export(model, (rgb, geom), str(path),
                      input_names=["rgb", "geom"],
                      output_names=["heatmap", "visibility", "targetability"],
                      opset_version=17, dynamo=False)
    assert path.exists()
    onnx.checker.check_model(onnx.load(str(path)))

    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"rgb": rgb.numpy(), "geom": geom.numpy()})
    with torch.no_grad():
        ref = model(rgb, geom)["heatmap"].numpy()
    assert np.abs(got[0] - ref).max() < 1e-4
