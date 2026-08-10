"""Tversky mask loss and connected-component cleanup.

Both exist for the same asymmetry: a missed weed survives to set seed, a
spurious one costs a laser pulse, and onion the model fails to mark is onion
nothing protects. Dice weights those the same. Tversky does not.
"""
import numpy as np
import pytest
import torch

from conftest import load_script

L = load_script("training/losses.py")
sg = load_script("perception/segmenter.py")

rfdetr = pytest.importorskip("rfdetr", reason="optional backend")
from rfdetr.models.criterion import dice_loss  # noqa: E402


# --------------------------------------------------------------------------- #
# Tversky
# --------------------------------------------------------------------------- #
def _pair(seed=0, n=6, d=400, p_fg=0.3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g),
            (torch.rand(n, d, generator=g) < p_fg).float())


def test_tversky_at_half_half_is_exactly_rfdetrs_dice():
    """The claim that this GENERALISES Dice rather than replacing it. If the
    equivalence breaks, every comparison against a Dice-trained run is invalid.
    """
    x, y = _pair()
    a = dice_loss(x, y, 6.0)
    b = L.tversky_loss(x, y, 6.0, alpha=0.5, beta=0.5)
    assert torch.allclose(a, b, atol=1e-6), (a.item(), b.item())


def test_raising_beta_penalises_missing_pixels_harder():
    """beta weights FALSE NEGATIVES. A prediction that misses foreground must
    cost more under beta>alpha than under Dice."""
    y = torch.zeros(1, 100)
    y[0, :50] = 1.0
    under = torch.full((1, 100), -4.0)          # predicts almost nothing
    under[0, :10] = 4.0                          # finds 10 of the 50
    dice = L.tversky_loss(under, y, 1.0, 0.5, 0.5)
    recall_leaning = L.tversky_loss(under, y, 1.0, 0.3, 0.7)
    assert recall_leaning > dice


def test_raising_alpha_penalises_invented_pixels_harder():
    y = torch.zeros(1, 100)
    y[0, :20] = 1.0
    over = torch.full((1, 100), 4.0)            # predicts everything
    dice = L.tversky_loss(over, y, 1.0, 0.5, 0.5)
    precision_leaning = L.tversky_loss(over, y, 1.0, 0.7, 0.3)
    assert precision_leaning > dice


def test_a_perfect_mask_costs_almost_nothing_at_any_weighting():
    y = torch.zeros(2, 200)
    y[:, :80] = 1.0
    x = torch.where(y > 0, torch.full_like(y, 12.0), torch.full_like(y, -12.0))
    for a, b in ((0.5, 0.5), (0.3, 0.7), (0.7, 0.3)):
        assert L.tversky_loss(x, y, 2.0, a, b).item() < 0.02


def test_the_focal_exponent_is_off_at_gamma_one():
    x, y = _pair(seed=3)
    assert torch.allclose(L.tversky_loss(x, y, 6.0, 0.3, 0.7, gamma=1.0),
                          L.tversky_loss(x, y, 6.0, 0.3, 0.7))


def test_focal_down_weights_the_masks_already_getting_right():
    """gamma>1 raises (1-TI) to a power, so a nearly-correct mask (small
    residual) shrinks far more than a badly wrong one. That is the point: the
    gradient goes to the hard instances."""
    y = torch.zeros(1, 100)
    y[0, :50] = 1.0
    good = torch.where(y > 0, torch.full_like(y, 6.0), torch.full_like(y, -6.0))
    bad = torch.full((1, 100), -6.0)

    g1 = L.tversky_loss(good, y, 1.0, 0.5, 0.5, gamma=1.0).item()
    g2 = L.tversky_loss(good, y, 1.0, 0.5, 0.5, gamma=2.0).item()
    b1 = L.tversky_loss(bad, y, 1.0, 0.5, 0.5, gamma=1.0).item()
    b2 = L.tversky_loss(bad, y, 1.0, 0.5, 0.5, gamma=2.0).item()
    assert g2 / g1 < b2 / b1, "focal did not concentrate on the hard mask"


def test_the_loss_is_differentiable():
    x, y = _pair(seed=5)
    x.requires_grad_(True)
    L.tversky_loss(x, y, 6.0, 0.3, 0.7, 1.5).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_make_tversky_matches_rfdetrs_call_signature():
    """It stands in for dice_loss_jit(inputs, targets, num_masks)."""
    fn = L.make_tversky(0.3, 0.7, 1.0)
    x, y = _pair(seed=7)
    assert torch.isfinite(fn(x, y, 6.0))
    assert fn.tversky == (0.3, 0.7, 1.0)


# --------------------------------------------------------------------------- #
# installing it into rfdetr
# --------------------------------------------------------------------------- #
def test_the_default_weights_patch_nothing_at_all(monkeypatch):
    """0.5/0.5/1.0 IS Dice, so a default run must be bit-identical to one with
    this feature absent - no patch, no behaviour change to explain."""
    rf = load_script("training/train_seg_rfdetr.py")
    import rfdetr.models.criterion as C
    before = C.dice_loss_jit
    assert rf._install_tversky(0.5, 0.5, 1.0) is False
    assert C.dice_loss_jit is before


def test_non_default_weights_replace_the_dice_loss(monkeypatch):
    rf = load_script("training/train_seg_rfdetr.py")
    import rfdetr.models.criterion as C
    monkeypatch.setattr(C, "dice_loss_jit", C.dice_loss_jit)   # restore after
    assert rf._install_tversky(0.3, 0.7, 1.0) is True
    assert getattr(C.dice_loss_jit, "tversky", None) == (0.3, 0.7, 1.0)


def test_a_missing_dice_symbol_is_named_rather_than_silently_skipped(
        monkeypatch):
    """The patch depends on rfdetr keeping a module-level dice_loss_jit. If a
    future version computes its mask loss differently, training must say so
    instead of quietly using Dice while the config claims Tversky."""
    rf = load_script("training/train_seg_rfdetr.py")
    import rfdetr.models.criterion as C
    monkeypatch.delattr(C, "dice_loss_jit")
    with pytest.raises(SystemExit, match="dice_loss_jit"):
        rf._install_tversky(0.3, 0.7, 1.0)


def test_the_shipped_config_defaults_to_dice():
    tm = load_script("training/train_model_rfdetr.py")
    assert (tm.CONFIG["TVERSKY_ALPHA"], tm.CONFIG["TVERSKY_BETA"],
            tm.CONFIG["FOCAL_GAMMA"]) == (0.5, 0.5, 1.0)


# --------------------------------------------------------------------------- #
# connected components
# --------------------------------------------------------------------------- #
def test_specks_are_dropped_and_the_plant_is_kept():
    m = np.zeros((100, 100), bool)
    m[20:60, 20:60] = True                       # the plant
    m[5, 5] = m[90, 90] = True                   # sigmoid noise on soil
    out = sg.drop_fragments(m)
    assert out[20:60, 20:60].all()
    assert not out[5, 5] and not out[90, 90]


def test_a_second_real_lobe_survives():
    """Two parts of one plant separated by an occluding leaf are both real,
    which is why this is not simply 'take the largest component'."""
    m = np.zeros((100, 100), bool)
    m[10:40, 10:40] = True
    m[60:85, 60:85] = True                       # ~52% of the larger one
    out = sg.drop_fragments(m, min_frac=0.15)
    assert out[10:40, 10:40].all() and out[60:85, 60:85].all()


def test_the_threshold_is_relative_so_a_small_plant_survives_whole():
    """An ABSOLUTE area threshold would delete genuine cotyledons - the small
    instances this project is already losing."""
    m = np.zeros((100, 100), bool)
    m[50:56, 50:56] = True                       # a 36 px weed, all of it
    assert sg.drop_fragments(m).sum() == 36


def test_a_single_component_is_returned_unchanged():
    m = np.zeros((50, 50), bool)
    m[10:20, 10:20] = True
    assert np.array_equal(sg.drop_fragments(m), m)


def test_an_empty_mask_stays_empty():
    m = np.zeros((30, 30), bool)
    assert not sg.drop_fragments(m).any()


def test_an_absolute_floor_can_be_added_on_top():
    m = np.zeros((100, 100), bool)
    m[10:40, 10:40] = True
    m[60:70, 60:70] = True                       # 100 px, 11% - kept by frac 0.1
    assert sg.drop_fragments(m, min_frac=0.1).sum() > 900
    assert sg.drop_fragments(m, min_frac=0.1, min_px=200).sum() == 900
