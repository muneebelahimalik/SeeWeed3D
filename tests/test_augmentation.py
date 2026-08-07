"""Augmentation, input resolution and anchors - the three levers on small-weed
recall, plus the correctness traps each one hides.

The trap that matters most: v2 transforms dispatch on tv_tensor SUBCLASSES, not
on dict keys. Passing plain tensors leaves boxes and masks untransformed while
the image is warped, so the annotations describe a picture that no longer
exists - and nothing raises.
"""
import numpy as np
import pytest
import torch

from conftest import load_script

sd = load_script("training/seg_dataset.py")
seg = load_script("perception/segmenter.py")


def _sample(h=120, w=160, n=2):
    img = torch.rand(3, h, w)
    masks = np.zeros((n, h, w), np.uint8)
    boxes = []
    for k in range(n):
        y0, x0 = 10 + k * 40, 10 + k * 50
        masks[k, y0:y0 + 30, x0:x0 + 30] = 1
        boxes.append([x0, y0, x0 + 29, y0 + 29])
    return img, {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.arange(1, n + 1),
        "masks": torch.from_numpy(masks),
        "image_id": torch.tensor(0),
    }


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #
def test_none_preset_disables_augmentation():
    assert sd.build_augmentation("none") is None
    assert sd.build_augmentation(None) is None


def test_an_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="preset"):
        sd.build_augmentation("mosaic")


@pytest.mark.parametrize("preset", ["flip", "standard", "strong"])
def test_every_preset_builds_and_runs(preset):
    from torchvision import tv_tensors
    aug = sd.build_augmentation(preset)
    img, t = _sample()
    wrapped = {
        "boxes": tv_tensors.BoundingBoxes(t["boxes"], format="XYXY",
                                          canvas_size=(120, 160)),
        "masks": tv_tensors.Mask(t["masks"]),
        "labels": t["labels"],
    }
    oi, ot = aug(tv_tensors.Image(img), wrapped)
    assert oi.shape[0] == 3
    assert len(ot["boxes"]) == len(ot["labels"]) == len(ot["masks"])


def test_no_preset_contains_a_compositing_transform():
    """Mosaic/MixUp/CopyPaste fabricate crop geometry no field produced. For a
    crop-SAFETY model that is not a tuning choice."""
    for preset in ("flip", "standard", "strong"):
        names = [type(s).__name__.lower()
                 for s in sd.build_augmentation(preset).transforms]
        for banned in ("mosaic", "mixup", "copypaste", "cutmix"):
            assert not any(banned in n for n in names), f"{banned} in {preset}"


def test_standard_has_no_vertical_flip():
    """Top-down field imagery has a real light direction; mirroring vertically
    inverts the shading that separates a leaf from its own shadow."""
    names = [type(s).__name__
             for s in sd.build_augmentation("standard").transforms]
    assert "RandomVerticalFlip" not in names
    assert "RandomHorizontalFlip" in names


def test_sanitize_is_always_last():
    """An instance rotated out of frame must lose box, mask and label
    together; anything after it could reintroduce a degenerate box."""
    for preset in ("flip", "standard", "strong"):
        steps = sd.build_augmentation(preset).transforms
        assert type(steps[-1]).__name__ == "SanitizeBoundingBoxes"


# --------------------------------------------------------------------------- #
# the dispatch trap
# --------------------------------------------------------------------------- #
def test_boxes_and_masks_move_with_the_image():
    """A horizontal flip must move the annotations too. If v2 saw plain
    tensors it would flip only the image and nothing would raise."""
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    img, t = _sample(n=1)
    flip = v2.Compose([v2.RandomHorizontalFlip(1.0)])
    wrapped = {
        "boxes": tv_tensors.BoundingBoxes(t["boxes"], format="XYXY",
                                          canvas_size=(120, 160)),
        "masks": tv_tensors.Mask(t["masks"]),
        "labels": t["labels"],
    }
    oi, ot = flip(tv_tensors.Image(img), wrapped)
    assert not torch.equal(ot["boxes"], t["boxes"]), "boxes were not flipped"
    assert not torch.equal(ot["masks"], t["masks"]), "masks were not flipped"
    # and the mask still agrees with its box
    ys, xs = torch.nonzero(ot["masks"][0], as_tuple=True)
    bx0, by0, bx1, by1 = ot["boxes"][0].tolist()
    assert abs(float(xs.min()) - bx0) <= 2 and abs(float(ys.min()) - by0) <= 2


def test_an_instance_removed_by_sanitize_loses_all_three():
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    img, t = _sample(n=2)
    t["boxes"][1] = torch.tensor([5.0, 5.0, 6.0, 6.0])   # degenerate
    wrapped = {
        "boxes": tv_tensors.BoundingBoxes(t["boxes"], format="XYXY",
                                          canvas_size=(120, 160)),
        "masks": tv_tensors.Mask(t["masks"]),
        "labels": t["labels"],
    }
    _, ot = v2.SanitizeBoundingBoxes(min_size=5)(tv_tensors.Image(img), wrapped)
    assert len(ot["boxes"]) == len(ot["masks"]) == len(ot["labels"]) == 1


# --------------------------------------------------------------------------- #
# through the dataset
# --------------------------------------------------------------------------- #
def _manifest(tmp_path, n=4):
    import json
    import cv2
    root = tmp_path / "sessions" / "s1" / "rgb"
    root.mkdir(parents=True)
    frames = []
    for i in range(1, n + 1):
        name = f"s1_{i:06d}.png"
        cv2.imwrite(str(root / name),
                    np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8))
        frames.append({
            "session_id": "s1", "item_id": f"s1_{i:06d}", "image_path": name,
            "width": 160, "height": 120, "split": "train",
            "instances": [{"class_name": "grass_weed",
                           "polygons": [[20, 20, 70, 20, 70, 70, 20, 70]]}]})
    return {"images_root": [str(tmp_path / "sessions")],
            "classes": ["grass_weed"], "frames": frames}


def test_the_dataset_returns_plain_tensors_after_augmentation(tmp_path):
    """torchvision's detection models take tensors; leaking tv_tensor
    subclasses downstream would be a silent type change."""
    doc = _manifest(tmp_path)
    ds = sd.SegManifestDataset(doc, tmp_path / "sessions", "train",
                               augment=True, aug_preset="standard")
    img, t = ds[0]
    assert type(img) is torch.Tensor
    assert type(t["boxes"]) is torch.Tensor
    assert type(t["masks"]) is torch.Tensor
    assert t["masks"].dtype == torch.uint8
    assert t["boxes"].dtype == torch.float32


def test_augmentation_off_is_deterministic(tmp_path):
    doc = _manifest(tmp_path)
    ds = sd.SegManifestDataset(doc, tmp_path / "sessions", "train",
                               augment=False)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_every_sample_keeps_at_least_one_instance(tmp_path):
    """Augmentation can rotate a plant out of frame; the dataset keeps the
    original rather than handing the trainer a sample it must skip."""
    doc = _manifest(tmp_path)
    ds = sd.SegManifestDataset(doc, tmp_path / "sessions", "train",
                               augment=True, aug_preset="strong")
    for _ in range(12):
        _, t = ds[0]
        assert len(t["labels"]) >= 1
        assert len(t["boxes"]) == len(t["labels"]) == len(t["masks"])


# --------------------------------------------------------------------------- #
# input resolution and anchors
# --------------------------------------------------------------------------- #
def test_default_build_keeps_torchvisions_resize():
    """Unchanged by default, so an existing checkpoint still loads as trained."""
    m = seg.MaskRCNNSegmenter.build(2, pretrained=False)
    assert m.transform.min_size == (800,) and m.transform.max_size == 1333


def test_resolution_is_configurable():
    m = seg.MaskRCNNSegmenter.build(2, pretrained=False, min_size=1000,
                                    max_size=1800)
    assert m.transform.min_size == (1000,) and m.transform.max_size == 1800


def test_small_anchors_are_applied():
    from training.train_seg_torchvision import SMALL_ANCHORS
    m = seg.MaskRCNNSegmenter.build(2, pretrained=False,
                                    anchor_sizes=SMALL_ANCHORS)
    assert m.rpn.anchor_generator.sizes == SMALL_ANCHORS


def test_small_anchors_do_not_reshape_the_rpn_head():
    """Anchors-per-location fixes the head's output channels. Keeping it at 3
    is what lets a checkpoint survive this change."""
    a = seg.MaskRCNNSegmenter.build(2, pretrained=False)
    from training.train_seg_torchvision import SMALL_ANCHORS
    b = seg.MaskRCNNSegmenter.build(2, pretrained=False,
                                    anchor_sizes=SMALL_ANCHORS)
    assert (a.rpn.anchor_generator.num_anchors_per_location() ==
            b.rpn.anchor_generator.num_anchors_per_location())
    sa = {k: v.shape for k, v in a.state_dict().items()}
    sb = {k: v.shape for k, v in b.state_dict().items()}
    assert sa == sb, "state dict shapes must be identical"


def test_anchors_that_would_reshape_the_head_are_rejected():
    """Silently accepting them would produce a checkpoint that cannot be
    loaded by anything else in the pipeline."""
    with pytest.raises(ValueError, match="anchors-per-location"):
        seg.MaskRCNNSegmenter.build(
            2, pretrained=False,
            anchor_sizes=((16, 32), (32, 64), (64,), (128,), (256,)))


def test_boxes_are_recomputed_from_masks_not_carried_from_v2():
    """Under rotation v2 transports a box by rotating its corners and taking
    the axis-aligned hull, which is strictly larger than the rotated mask - a
    45-degree turn inflates a square by ~41% per side. The mask is ground
    truth, so the box must be derived from it."""
    import json
    import cv2
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    doc = _manifest(tmp, n=2)
    ds = sd.SegManifestDataset(doc, tmp / "sessions", "train",
                               augment=True, aug_preset="strong")
    for _ in range(15):
        _, t = ds[0]
        for k in range(len(t["labels"])):
            ys, xs = torch.nonzero(t["masks"][k], as_tuple=True)
            if not len(xs):
                continue
            x0, y0, x1, y1 = t["boxes"][k].tolist()
            assert abs(float(xs.min()) - x0) <= 1
            assert abs(float(ys.min()) - y0) <= 1
            assert abs(float(xs.max()) - x1) <= 1
            assert abs(float(ys.max()) - y1) <= 1


def test_the_keep_mask_is_boolean_not_uint8():
    """torch treats a uint8 index as POSITIONAL, so labels[keep] would select
    by position instead of masking - wrong labels, visible only as bad
    accuracy."""
    m = torch.zeros((3, 8, 8), dtype=torch.uint8)
    m[1] = 1
    keep = m.flatten(1).any(1).bool()
    labels = torch.tensor([10, 20, 30])
    assert labels[keep].tolist() == [20]
