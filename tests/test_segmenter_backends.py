"""Pluggable segmentation backends and the licence policy.

The default backend must be permissively licensed: a commercial laser weeder
cannot carry an AGPL obligation, and unlike a bug that cannot be fixed after
shipping."""
import numpy as np
import pytest

from conftest import load_script

seg = load_script("perception/segmenter.py")

from common.ontology import CLASSES, CROP_CLASS  # noqa: E402


def test_default_backend_is_permissively_licensed():
    """The whole point of the exercise: nothing in the normal path may create
    a licensing obligation by accident."""
    assert seg.DEFAULT_BACKEND in seg.PERMISSIVE_BACKENDS
    _, licence = seg.BACKENDS[seg.DEFAULT_BACKEND]
    assert "AGPL" not in licence
    assert licence == "BSD-3-Clause"


def test_every_backend_declares_a_licence():
    for name, (cls, licence) in seg.BACKENDS.items():
        assert licence, f"backend {name} has no declared licence"
        assert cls is not None


def test_choosing_a_copyleft_backend_warns_loudly(capsys):
    """Using Ultralytics is legitimate for research, so this warns rather than
    refuses - but it must never be silent, or an obligation attaches to a
    product with nobody noticing."""
    seg.build_segmenter("ultralytics", weights="yolo26n-seg.pt")
    out = capsys.readouterr().out
    assert "LICENCE WARNING" in out
    assert "AGPL-3.0" in out
    assert "maskrcnn" in out                      # names the alternatives


def test_permissive_backend_does_not_warn(capsys):
    seg.build_segmenter("maskrcnn", weights="x.pt")
    assert "LICENCE WARNING" not in capsys.readouterr().out


def test_unknown_backend_fails_clearly():
    with pytest.raises(ValueError) as e:
        seg.build_segmenter("yolov8")
    assert "maskrcnn" in str(e.value)


def test_detections_interface_is_backend_independent():
    """Everything downstream sees only this structure, which is what makes a
    backend swap a one-file change."""
    h = w = 32
    masks = np.zeros((2, h, w), bool)
    masks[0, 4:12, 4:12] = True
    masks[1, 16:28, 16:28] = True
    det = seg.Detections(masks, np.array([[4., 4., 8., 8.], [16., 16., 12., 12.]]),
                         np.array([CLASSES.index("wild_radish"),
                                   CLASSES.index(CROP_CLASS)]),
                         np.array([0.9, 0.8]), w, h)
    assert len(det) == 2
    assert det.class_name(0) == "wild_radish"
    assert det.weed_indices() == [0]
    union = det.onion_safety_mask()
    assert union.sum() == int(masks[1].sum())


# --------------------------------------------------------------------------- #
# torchvision Mask R-CNN (the default) - really built and run
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="torch is an optional training dep")
tv = pytest.importorskip("torchvision", reason="torchvision optional")


def test_maskrcnn_head_is_sized_for_the_ontology_plus_background():
    """torchvision reserves label 0 for background, so the head needs N+1."""
    model = seg.MaskRCNNSegmenter.build(len(CLASSES), pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == len(CLASSES) + 1
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == \
        len(CLASSES) + 1


def test_maskrcnn_training_step_runs_and_produces_losses():
    """A real forward+backward on the permissive backend, on CPU."""
    torch.manual_seed(0)
    model = seg.MaskRCNNSegmenter.build(len(CLASSES), pretrained=False)
    model.train()
    img = torch.rand(3, 96, 96)
    m = torch.zeros((1, 96, 96), dtype=torch.uint8)
    m[0, 20:60, 20:60] = 1
    target = {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
              "labels": torch.tensor([CLASSES.index("wild_radish") + 1]),
              "masks": m}
    losses = model([img], [target])
    assert "loss_mask" in losses and "loss_classifier" in losses
    total = sum(losses.values())
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


def test_maskrcnn_inference_returns_ontology_indices_not_offset_ones():
    """The +1 background offset must be undone on the way out, or every class
    downstream would be wrong by one - silently mapping weeds onto onions."""
    torch.manual_seed(0)
    model = seg.MaskRCNNSegmenter.build(len(CLASSES), pretrained=False).eval()
    s = seg.MaskRCNNSegmenter("unused.pt", conf=0.0, device="cpu")
    s._model = model
    det = s(np.zeros((64, 64, 3), np.uint8))
    assert det.width == 64 and det.height == 64
    assert det.masks.shape[1:] == (64, 64)
    if len(det):
        assert det.classes.min() >= 0
        assert det.classes.max() < len(CLASSES)


def test_maskrcnn_missing_weights_fails_with_the_training_command():
    s = seg.MaskRCNNSegmenter("/nonexistent/best.pt")
    with pytest.raises(FileNotFoundError) as e:
        s.load()
    assert "train_seg_torchvision" in str(e.value)


def test_seg_dataset_produces_torchvision_format(tmp_path):
    """The permissive path trains straight from the Datumaro import - no YOLO
    label tree, no second copy of the dataset."""
    import cv2
    sd = load_script("training/seg_dataset.py")

    img_dir = tmp_path / "sessions" / "s1" / "rgb"
    img_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "s1_000001.png"), np.zeros((64, 64, 3), np.uint8))

    manifest = {"images_root": str(tmp_path / "sessions"),
                "classes": list(CLASSES),
                "frames": [{"session_id": "s1", "item_id": "s1_000001",
                            "image_path": "s1/rgb/s1_000001.png",
                            "width": 64, "height": 64, "split": "train",
                            "instances": [
                                {"class_name": "wild_radish",
                                 "class_index": CLASSES.index("wild_radish"),
                                 "polygons": [[10, 10, 40, 10, 40, 40, 10, 40]]},
                                {"class_name": CROP_CLASS,
                                 "class_index": CLASSES.index(CROP_CLASS),
                                 "polygons": [[45, 45, 60, 45, 60, 60, 45, 60]]}],
                            "ignore_regions": []}]}

    ds = sd.SegManifestDataset(manifest, tmp_path / "sessions", "train",
                               augment=False)
    assert len(ds) == 1
    img, target = ds[0]
    assert img.shape == (3, 64, 64)
    assert target["boxes"].shape == (2, 4)
    assert target["masks"].shape == (2, 64, 64)
    # Labels carry the background offset, which the segmenter undoes.
    assert target["labels"].tolist() == [
        CLASSES.index("wild_radish") + 1, CLASSES.index(CROP_CLASS) + 1]
    assert target["masks"].dtype == torch.uint8
    x0, y0, x1, y1 = target["boxes"][0].tolist()
    assert x1 > x0 and y1 > y0                    # no degenerate boxes


def test_seg_dataset_flip_moves_boxes_and_masks_together(tmp_path):
    import cv2
    sd = load_script("training/seg_dataset.py")
    img_dir = tmp_path / "s" / "rgb"
    img_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "s_000001.png"), np.zeros((64, 64, 3), np.uint8))
    manifest = {"images_root": str(tmp_path), "classes": list(CLASSES),
                "frames": [{"session_id": "s", "item_id": "s_000001",
                            "image_path": "s/rgb/s_000001.png",
                            "width": 64, "height": 64, "split": "train",
                            "instances": [{"class_name": "wild_radish",
                                           "class_index": 1,
                                           "polygons": [[5, 5, 25, 5, 25, 25,
                                                         5, 25]]}],
                            "ignore_regions": []}]}
    for seed in range(6):
        ds = sd.SegManifestDataset(manifest, tmp_path, "train", augment=True,
                                   seed=seed)
        _, t = ds[0]
        if not len(t["boxes"]):
            continue
        x0, y0, x1, y1 = t["boxes"][0].tolist()
        ys, xs = np.nonzero(t["masks"][0].numpy())
        # The box must still bound its own mask after any flip.
        assert abs(xs.min() - x0) <= 1 and abs(xs.max() - x1) <= 1
        assert abs(ys.min() - y0) <= 1 and abs(ys.max() - y1) <= 1
