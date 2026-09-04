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
    spread = pi.find_images(s, limit=3, stride=3)
    assert len(spread) == 3
    assert spread[0] != spread[1]
    # frames 1,4,7,10 after the stride; 3 of those spread end to end.
    assert [f.stem[-1] for f in spread] == ["1", "7", "0"]  # 1, 7, 10


def test_a_limit_spreads_across_the_whole_drive_not_its_head():
    """The trap this closes hides at small scale. On a 4000-frame drive,
    stride 20 with limit 40 used to cover the first 800 frames and nothing
    after - so a stretch of bare crop at the start of a session read as a
    whole session with no weeds in it, and the conclusion drawn was about the
    model rather than about the sampling."""
    got = pi.sample_frames(list(range(4000)), limit=40, stride=20)
    assert len(got) == 40
    assert got[0] == 0
    assert got[-1] == 3980, "the end of the drive is never reached"
    assert max(b - a for a, b in zip(got, got[1:])) < 200


def test_a_limit_larger_than_the_drive_returns_everything():
    assert pi.sample_frames(list(range(5)), limit=99) == list(range(5))
    assert pi.sample_frames(list(range(5)), limit=0) == list(range(5))


def test_a_limit_of_one_is_not_an_error():
    assert pi.sample_frames(list(range(9)), limit=1) == [0]


def test_stride_applies_before_the_limit():
    """Otherwise a limit smaller than the drive would defeat the stride, and
    near-identical consecutive frames come back."""
    assert pi.sample_frames(list(range(10)), limit=0, stride=4) == [0, 4, 8]


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
def test_every_class_gets_its_own_colour():
    """Reading an overlay means telling a cutleaf from a grass weed at a
    glance; one colour for 'weed' cannot do that."""
    seen = [pi.class_colour(n) for n in pi.CLASS_COLOURS]
    assert len(set(seen)) == len(seen), "two classes share a colour"


def test_the_crop_is_the_odd_one_out():
    from seeweed3d.common.ontology import CLASSES
    weeds = {pi.class_colour(n) for n in CLASSES if n != CROP_CLASS}
    assert pi.class_colour(CROP_CLASS) not in weeds


def test_an_unknown_class_is_grey_rather_than_reusing_a_colour():
    """A label that is not in the ontology should look wrong, not look like
    some other plant."""
    c = pi.class_colour("bindweed")
    assert c == pi.C_UNKNOWN
    assert c not in set(pi.CLASS_COLOURS.values())


def test_two_classes_are_drawn_in_different_colours():
    bgr = np.zeros((60, 80, 3), np.uint8)
    a = pi.draw(bgr, _det([("grass_weed", (0, 20, 0, 30))]), set(), scale=1.0,
                show_legend=False)
    b = pi.draw(bgr, _det([(CROP_CLASS, (0, 20, 0, 30))]), set(), scale=1.0,
                show_legend=False)
    assert not np.array_equal(a, b)


def test_a_crop_conflict_adds_an_outline_without_recolouring_the_weed():
    """The class and the hazard are separate facts. Recolouring would tell you
    a weed is on the onion while hiding WHICH weed it is."""
    bgr = np.zeros((60, 80, 3), np.uint8)
    d = _det([("grass_weed", (10, 30, 10, 40)), (CROP_CLASS, (40, 50, 60, 79))])
    plain = pi.draw(bgr, d, conflict_idx=set(), scale=1.0, show_legend=False)
    flagged = pi.draw(bgr, d, conflict_idx={0}, scale=1.0, show_legend=False)
    assert not np.array_equal(plain, flagged)
    # the weed interior keeps its class colour in both
    assert tuple(plain[20, 25]) == tuple(flagged[20, 25])


def test_the_legend_never_covers_the_frame():
    """A key painted into a corner hides whatever was under it, and in these
    scenes the corners hold plants."""
    bgr = np.zeros((60, 80, 3), np.uint8)
    d = _det([("grass_weed", (0, 10, 0, 20))])
    bare = pi.draw(bgr, d, set(), scale=1.0, show_legend=False)
    keyed = pi.draw(bgr, d, set(), scale=1.0, show_legend=True)
    assert keyed.shape[0] > bare.shape[0], "legend must add canvas, not cover"
    assert keyed.shape[1] == bare.shape[1]
    assert np.array_equal(keyed[:bare.shape[0]], bare)


def test_labels_can_be_turned_off_for_a_dense_frame():
    bgr = np.zeros((200, 300, 3), np.uint8)
    d = _det([("grass_weed", (20, 60, 20, 80))], hw=(200, 300))
    with_text = pi.draw(bgr, d, set(), scale=1.0, labels="class_score",
                        show_legend=False)
    without = pi.draw(bgr, d, set(), scale=1.0, labels="none",
                      show_legend=False)
    assert not np.array_equal(with_text, without)


def test_the_overlay_can_be_scaled_down():
    d = _det([("grass_weed", (0, 10, 0, 20))])
    out = pi.draw(np.zeros((60, 80, 3), np.uint8), d, set(), scale=0.5,
                  show_legend=False)
    assert out.shape[:2] == (30, 40)


def test_the_legend_is_drawn_after_the_downscale():
    """Drawn before a 0.5 resize the key comes out at half the font size, which
    is the one part of the picture you cannot zoom into."""
    d = _det([("grass_weed", (0, 10, 0, 20))])
    half = pi.draw(np.zeros((60, 80, 3), np.uint8), d, set(), scale=0.5)
    full = pi.draw(np.zeros((60, 80, 3), np.uint8), d, set(), scale=1.0)
    assert half.shape[0] - 30 == full.shape[0] - 60, \
        "the strip should be the same height at either scale"


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


# --------------------------------------------------------------------------
# Running on a SPLIT. A split is not a directory: its frames are scattered
# across every session, and train and test frames sit side by side in the same
# rgb/ folder - so pointing --images at a session and calling the result
# held-out is wrong, and wrong in the flattering direction.
# --------------------------------------------------------------------------
def _built(tmp_path, splits=("train", "train", "val", "test")):
    sess = _session(tmp_path, n=len(splits), with_depth=False)
    ds = tmp_path / "dataset"
    ds.mkdir()
    frames = [{"image_path": f"vid1_2026/rgb/vid1_2026_{i + 1:06d}.png",
               "session_id": "vid1_2026", "split": s}
              for i, s in enumerate(splits)]
    (ds / "seg_manifest.json").write_text(json.dumps(
        {"classes": CLASSES, "images_root": str(sess.parent),
         "frames": frames}), encoding="utf-8")
    return ds


def test_a_split_yields_only_its_own_frames(tmp_path):
    ds = _built(tmp_path)
    got = pi.split_images(ds, "test")
    assert [p.name for p in got] == ["vid1_2026_000004.png"]
    assert [p.name for p in pi.split_images(ds, "train")] == [
        "vid1_2026_000001.png", "vid1_2026_000002.png"]


def test_a_split_resolves_to_paths_that_exist(tmp_path):
    """The manifest stores a relative path against an images_root, and a
    merged build has several roots. A list of strings that do not open is
    worse than an error."""
    for p in pi.split_images(_built(tmp_path), "train"):
        assert p.is_file()


def test_limit_and_stride_apply_to_a_split(tmp_path):
    ds = _built(tmp_path, ("test",) * 6)
    assert len(pi.split_images(ds, "test", limit=2)) == 2
    assert len(pi.split_images(ds, "test", stride=3)) == 2


def test_an_empty_split_names_the_ones_that_exist(tmp_path):
    """Silently running on nothing would produce an empty folder of overlays
    that looks like 'the model found no weeds'."""
    with pytest.raises(SystemExit) as e:
        pi.split_images(_built(tmp_path, ("train", "val")), "test")
    assert "test" in str(e.value) and "train" in str(e.value)


def test_a_dataset_without_a_manifest_says_what_to_point_at(tmp_path):
    with pytest.raises(SystemExit) as e:
        pi.split_images(tmp_path, "test")
    assert "seg_manifest.json" in str(e.value)


def test_dataset_wins_over_images_so_a_stale_folder_cannot_be_scored(tmp_path):
    """Both keys have defaults, and IMAGES ships pointing at a real session.
    If IMAGES won, asking for the test split would quietly run on a whole
    drive the model trained on."""
    import inspect
    src = inspect.getsource(pi.predict)
    assert src.index('c.get("DATASET")') < src.index('find_images(')


# --------------------------------------------------------------------------
# The same drive holds the frames that trained the model and the frames
# nobody touched, in one rgb/ folder. Predictions on the first kind show you
# what was memorised.
# --------------------------------------------------------------------------
def test_frames_a_build_trained_on_are_dropped(tmp_path):
    ds = _built(tmp_path, ("train", "train", "val", "test"))
    all_frames = pi.find_images(tmp_path / "sessions" / "vid1_2026")
    kept, n = pi.exclude_built(all_frames, ds)
    assert n == 2
    assert [p.name for p in kept] == ["vid1_2026_000003.png",
                                      "vid1_2026_000004.png"]


def test_val_and_test_can_be_excluded_too(tmp_path):
    """Legitimate to look at, but they are the frames the score comes from -
    a decision taken after staring at them has used them for tuning."""
    ds = _built(tmp_path, ("train", "train", "val", "test"))
    all_frames = pi.find_images(tmp_path / "sessions" / "vid1_2026")
    kept, n = pi.exclude_built(all_frames, ds, ("train", "val", "test"))
    assert n == 4 and kept == []


def test_exclusion_matches_on_resolved_paths(tmp_path):
    """A relative manifest path against an images_root will not string-match a
    path found by walking a folder, and a comparison that silently excludes
    nothing is worse than no filter at all."""
    ds = _built(tmp_path, ("train", "train", "val", "test"))
    sess = tmp_path / "sessions" / "vid1_2026"
    awkward = [str(sess / "rgb" / "vid1_2026_000001.png"),
               sess / "rgb" / ".." / "rgb" / "vid1_2026_000003.png"]
    kept, n = pi.exclude_built(awkward, ds)
    assert n == 1 and len(kept) == 1


def test_a_frame_whose_image_is_gone_excludes_nothing(tmp_path):
    """An annotation can outlive its image. It cannot be in the folder being
    filtered either, so it must not raise."""
    ds = _built(tmp_path, ("train", "test"))
    (tmp_path / "sessions" / "vid1_2026" / "rgb"
     / "vid1_2026_000001.png").unlink()
    kept, n = pi.exclude_built(
        pi.find_images(tmp_path / "sessions" / "vid1_2026"), ds)
    assert n == 0 and len(kept) == 1


def test_excluding_everything_says_so_rather_than_running_on_nothing(tmp_path):
    ds = _built(tmp_path, ("train", "train", "train", "train"))
    kept, n = pi.exclude_built(
        pi.find_images(tmp_path / "sessions" / "vid1_2026"), ds)
    assert kept == [] and n == 4
    src = __import__("inspect").getsource(pi.predict)
    assert "has not seen" in src, (
        "an empty frame list must raise, not write an empty folder of "
        "overlays that reads as 'the model found no weeds'")


def test_exclusion_happens_before_limit_and_stride(tmp_path):
    """Otherwise 'the first 20' is the first 20 of a list that is mostly
    training frames, and the filter buys nothing."""
    src = __import__("inspect").getsource(pi.predict)
    assert src.index("exclude_built(") < src.index("sample_frames(frames")
