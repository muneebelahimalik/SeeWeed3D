"""Running a trained model on UNLABELLED frames.

The trap this module is mostly about: a session's depth/ PNGs share their RGB
frame's filename exactly, so any recursive image search returns one of each and
feeds 16-bit depth to the segmenter as a picture. The same defect already had
to be fixed once in training/seg_dataset.py.
"""
import json

import cv2
import numpy as np
import pytest

from conftest import load_script

pi = load_script("perception/predict_images.py")
seg = load_script("perception/segmenter.py")
from seeweed3d.common.ontology import CROP_CLASS  # noqa: E402

CLASSES = ["grass_weed", CROP_CLASS]


def _session(tmp_path, n=4, with_depth=True):
    s = tmp_path / "sessions" / "vid1_2026"
    (s / "rgb").mkdir(parents=True)
    if with_depth:
        (s / "depth").mkdir()
    for i in range(1, n + 1):
        name = f"vid1_2026_{i:06d}.png"
        cv2.imwrite(str(s / "rgb" / name),
                    np.zeros((60, 80, 3), np.uint8))
        if with_depth:
            # SAME filename, 16-bit - exactly what must not reach the model.
            cv2.imwrite(str(s / "depth" / name),
                        np.full((60, 80), 900, np.uint16))
    return s


# --------------------------------------------------------------------------- #
# frame discovery
# --------------------------------------------------------------------------- #
def test_a_session_folder_resolves_to_its_rgb_frames(tmp_path):
    files = pi.find_images(_session(tmp_path))
    assert len(files) == 4
    assert all(f.parent.name == "rgb" for f in files)


def test_depth_pngs_are_never_returned_as_images(tmp_path):
    """A depth PNG has the same name as its RGB frame. Handing one to the
    segmenter produces a confident prediction on 16-bit garbage."""
    for f in pi.find_images(_session(tmp_path)):
        assert "depth" not in {p.name for p in f.parents}


def test_a_plain_folder_of_images_works(tmp_path):
    d = tmp_path / "loose"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"x{i}.jpg"), np.zeros((60, 80, 3), np.uint8))
    assert len(pi.find_images(d)) == 3


def test_a_depth_subfolder_is_skipped_in_a_plain_folder_too(tmp_path):
    d = tmp_path / "loose"
    (d / "depth").mkdir(parents=True)
    cv2.imwrite(str(d / "a.png"), np.zeros((60, 80, 3), np.uint8))
    cv2.imwrite(str(d / "depth" / "a.png"), np.zeros((60, 80), np.uint16))
    assert len(pi.find_images(d)) == 1


def test_a_single_file_works(tmp_path):
    f = tmp_path / "one.png"
    cv2.imwrite(str(f), np.zeros((60, 80, 3), np.uint8))
    assert pi.find_images(f) == [f]


def test_stride_samples_across_the_session_not_the_first_n(tmp_path):
    """Consecutive ZED frames are near-identical; LIMIT alone gives you N
    pictures of one plant."""
    s = _session(tmp_path, n=10)
    assert len(pi.find_images(s, limit=3, stride=3)) == 3
    spread = pi.find_images(s, limit=3, stride=3)
    assert spread[0] != spread[1]
    assert [f.stem[-1] for f in spread] == ["1", "4", "7"]


def test_a_missing_path_says_so(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        pi.find_images(tmp_path / "nope")


def test_a_folder_with_no_images_explains_the_expected_layout(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SystemExit, match="session folder"):
        pi.find_images(d)


# --------------------------------------------------------------------------- #
# crop proximity
# --------------------------------------------------------------------------- #
def _det(specs, hw=(60, 80)):
    """specs: [(class_name, (y0, y1, x0, x1))]"""
    masks, cls = [], []
    for name, (y0, y1, x0, x1) in specs:
        m = np.zeros(hw, bool)
        m[y0:y1, x0:x1] = True
        masks.append(m)
        cls.append(CLASSES.index(name))
    n = len(specs)
    return seg.Detections(
        np.asarray(masks, bool) if n else np.zeros((0,) + hw, bool),
        np.zeros((n, 4)), np.asarray(cls, int), np.full(n, 0.9),
        hw[1], hw[0], names=list(CLASSES))


def test_a_weed_touching_predicted_onion_is_flagged():
    d = _det([("grass_weed", (0, 20, 0, 40)), (CROP_CLASS, (10, 30, 0, 40))])
    assert pi.mask_overlap_conflicts(d) == {0}


def test_a_weed_clear_of_the_onion_is_not_flagged():
    d = _det([("grass_weed", (0, 10, 0, 20)), (CROP_CLASS, (40, 50, 60, 79))])
    assert pi.mask_overlap_conflicts(d) == set()


def test_no_predicted_onion_flags_nothing():
    """No crop prediction is not the same as a safe frame, and this function
    must not imply otherwise by returning something."""
    assert pi.mask_overlap_conflicts(_det([("grass_weed", (0, 10, 0, 20))])) \
        == set()


def test_the_crop_itself_is_never_flagged_as_a_conflict():
    d = _det([(CROP_CLASS, (0, 20, 0, 40))])
    assert pi.mask_overlap_conflicts(d) == set()


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def test_the_overlay_colours_crop_weed_and_conflict_differently():
    bgr = np.zeros((60, 80, 3), np.uint8)
    d = _det([("grass_weed", (0, 10, 0, 20)), (CROP_CLASS, (40, 50, 60, 79))])
    out = pi.draw(bgr, d, conflict_idx=set(), scale=1.0)
    assert out.shape == bgr.shape
    assert out.any(), "nothing was drawn"
    flagged = pi.draw(bgr, d, conflict_idx={0}, scale=1.0)
    assert not np.array_equal(out, flagged)


def test_the_overlay_can_be_scaled_down():
    d = _det([("grass_weed", (0, 10, 0, 20))])
    out = pi.draw(np.zeros((60, 80, 3), np.uint8), d, set(), scale=0.5)
    assert out.shape[:2] == (30, 40)


# --------------------------------------------------------------------------- #
# end to end, with a stub model
# --------------------------------------------------------------------------- #
def test_predict_writes_overlays_and_a_json_record(tmp_path, monkeypatch):
    s = _session(tmp_path, n=3)
    d = _det([("grass_weed", (0, 10, 0, 20)), (CROP_CLASS, (5, 25, 0, 40))])

    class _Fake:
        classes = list(CLASSES)
        def load(self): return self
        def __call__(self, _bgr): return d

    import perception.segmenter as _ps
    monkeypatch.setattr(_ps, "build_segmenter", lambda *a, **k: _Fake())
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "preds"
    recs = pi.predict({**pi.CONFIG, "IMAGES": str(s), "CHECKPOINT": str(ckpt),
                       "OUT_DIR": str(out), "DEVICE": "cpu", "LIMIT": 0,
                       "STRIDE": 1, "MODE": "segmentation"})

    assert len(recs) == 3
    assert len(list((out / "overlays").glob("*.png"))) == 3
    doc = json.loads((out / "predictions.json").read_text())
    assert len(doc["frames"]) == 3
    inst = doc["frames"][0]["instances"]
    assert {i["class_name"] for i in inst} == set(CLASSES)
    # the weed overlaps the onion here, so the record must say so
    assert [i["touches_crop"] for i in inst if i["class_name"] != CROP_CLASS] \
        == [True]


def test_a_missing_checkpoint_points_at_the_trainer(tmp_path):
    with pytest.raises(SystemExit, match="train_model"):
        pi.predict({**pi.CONFIG, "IMAGES": str(_session(tmp_path)),
                    "CHECKPOINT": str(tmp_path / "nope.pt"),
                    "OUT_DIR": str(tmp_path / "o"), "DEVICE": "cpu"})
