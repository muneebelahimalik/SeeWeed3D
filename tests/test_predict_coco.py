"""The COCO half of predict_images.

This file exists because of a specific failure. `_write_coco` was written,
`WRITE_COCO` was added to CONFIG, and the accumulator list was initialised at
the top of the prediction loop - but nothing ever appended to it, so
`if coco_frames:` was permanently false and the writer never ran. A 79-frame
GPU pass completed, printed its per-class counts, and then the self-training
round died on a file that was never going to exist.

The end-to-end test next door asserted overlays and predictions.json. Both were
written, so it passed. Nothing asserted the artefact the next stage actually
consumes, which is the whole lesson: test the output your caller reads, not the
output that happens to be easiest to check.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
import perception.predict_images as pi          # noqa: E402
import perception.segmenter as seg              # noqa: E402
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402

HW = (60, 80)


def _det(specs, hw=HW):
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


def _session(tmp_path, n=3):
    import cv2
    d = tmp_path / "sess" / "rgb"
    d.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(d / f"f{i:03d}.png"), np.zeros(HW + (3,), np.uint8))
    return tmp_path / "sess"


def _run(tmp_path, monkeypatch, det, n=3, **over):
    class _Fake:
        classes = list(CLASSES)

        def load(self):
            return self

        def __call__(self, _bgr):
            return det

    monkeypatch.setattr(seg, "build_segmenter", lambda *a, **k: _Fake())
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "preds"
    pi.predict({**pi.CONFIG, "IMAGES": str(_session(tmp_path, n)),
                "CHECKPOINT": str(ckpt), "OUT_DIR": str(out), "DEVICE": "cpu",
                "LIMIT": 0, "STRIDE": 1, "MODE": "segmentation", **over})
    return out


# --------------------------------------------------------------------------- #
# the bug itself
# --------------------------------------------------------------------------- #
def test_a_normal_run_actually_writes_instances_default(tmp_path, monkeypatch):
    """THE REGRESSION. This is the file weeds_selftrain opens by name."""
    out = _run(tmp_path, monkeypatch,
               _det([("grass_weed", (0, 20, 0, 40))]))
    assert (out / "instances_default.json").exists(), \
        "the whole point of WRITE_COCO; a GPU pass that skips it is wasted"


def test_every_detection_reaches_the_coco(tmp_path, monkeypatch):
    d = _det([("grass_weed", (0, 20, 0, 40)),
              ("cutleaf_evening_primrose", (30, 50, 40, 75))])
    out = _run(tmp_path, monkeypatch, d, n=3)
    doc = json.loads((out / "instances_default.json").read_text())
    assert len(doc["images"]) == 3
    assert len(doc["annotations"]) == 6, "2 instances x 3 frames"


def test_write_coco_false_writes_nothing(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, _det([("grass_weed", (0, 20, 0, 40))]),
               WRITE_COCO=False)
    assert not (out / "instances_default.json").exists()
    assert (out / "predictions.json").exists(), "the other output still runs"


# --------------------------------------------------------------------------- #
# the memory shape - the fix that was easy to get wrong
# --------------------------------------------------------------------------- #
def test_the_loop_does_not_hoard_full_frame_masks(tmp_path, monkeypatch):
    """Masks are (N, H, W) bool over the WHOLE frame. Accumulating Detections
    across a session is 2.7 MB x ~3000 instances = ~8 GB live, which is the
    exact shape that already killed one self-training run in bench_mixed.
    Polygonising per frame is not a micro-optimisation, it is the difference
    between finishing and being OOM-killed."""
    seen = []

    class _Watch:
        """A Detections whose masks vanish once the frame is done with."""

        def __init__(self, inner):
            self._inner = inner
            self.names = inner.names
            self.scores = inner.scores
            self.boxes = inner.boxes
            self.classes = inner.classes

        def __len__(self):
            return len(self._inner)

        def class_name(self, i):
            return self._inner.class_name(i)

        @property
        def masks(self):
            seen.append(1)
            return self._inner.masks

    inner = _det([("grass_weed", (0, 20, 0, 40))])
    insts = pi._coco_instances(_Watch(inner))
    assert len(insts) == 1
    assert "segmentation" in insts[0] and "mask" not in insts[0], \
        "only polygons may survive the frame"
    assert all(not isinstance(v, np.ndarray) for v in insts[0].values()), \
        "an array here is a full-frame mask kept alive for the whole session"


def test_polygons_not_masks_are_what_is_accumulated(tmp_path, monkeypatch):
    """Structural: the loop must call the per-frame converter. If someone
    'simplifies' this back to `coco_frames.append((..., det))` the memory bug
    returns and no behavioural test would notice on a 3-frame fixture."""
    import inspect
    src = inspect.getsource(pi.predict)
    assert "_coco_instances(det)" in src, \
        "polygonise inside the loop, do not stash the Detections"


# --------------------------------------------------------------------------- #
# what the next stage needs from the file
# --------------------------------------------------------------------------- #
def test_categories_are_the_models_vocabulary_not_the_sample(tmp_path,
                                                             monkeypatch):
    """A COCO whose categories shift with what happened to be detected is not
    comparable with the next run's, and in CVAT it silently removes labels the
    corrector needs in order to fix a misclassification."""
    out = _run(tmp_path, monkeypatch, _det([("grass_weed", (0, 20, 0, 40))]))
    doc = json.loads((out / "instances_default.json").read_text())
    assert {c["name"] for c in doc["categories"]} == set(CLASSES)


def test_the_file_says_it_is_predictions_not_ground_truth(tmp_path,
                                                          monkeypatch):
    out = _run(tmp_path, monkeypatch, _det([("grass_weed", (0, 20, 0, 40))]))
    doc = json.loads((out / "instances_default.json").read_text())
    assert "not ground truth" in doc["info"]["description"].lower()
    assert doc["info"]["checkpoint"], "which model drew these"


def test_every_annotation_keeps_its_score(tmp_path, monkeypatch):
    """pseudo_label ranks on these; a COCO without them scores nothing."""
    out = _run(tmp_path, monkeypatch, _det([("grass_weed", (0, 20, 0, 40))]))
    doc = json.loads((out / "instances_default.json").read_text())
    assert all("score" in a for a in doc["annotations"])


def test_a_frame_the_model_found_nothing_in_is_still_listed(tmp_path,
                                                            monkeypatch):
    """Empty frames MUST reach CVAT. A missed weed is this project's failure
    mode, and it is only findable by a human looking at the frame - dropping
    frames with no predictions hides exactly the ones worth checking."""
    out = _run(tmp_path, monkeypatch, _det([]), n=4)
    doc = json.loads((out / "instances_default.json").read_text())
    assert len(doc["images"]) == 4
    assert doc["annotations"] == []


def test_image_file_names_are_bare_so_cvat_can_match_them(tmp_path,
                                                          monkeypatch):
    """CVAT matches an imported COCO to uploaded images by file_name. A path
    with directories in it matches nothing and the import lands empty."""
    out = _run(tmp_path, monkeypatch, _det([("grass_weed", (0, 20, 0, 40))]))
    doc = json.loads((out / "instances_default.json").read_text())
    for im in doc["images"]:
        assert "/" not in im["file_name"] and "\\" not in im["file_name"]


def test_the_written_sizes_are_the_real_frame_sizes(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, _det([("grass_weed", (0, 20, 0, 40))]))
    doc = json.loads((out / "instances_default.json").read_text())
    assert {(im["height"], im["width"]) for im in doc["images"]} == {HW}


def test_a_speck_below_the_polygon_floor_is_dropped_not_left_empty(tmp_path,
                                                                   monkeypatch):
    """mask_to_polygons has a min_area_px. An instance under it must not be
    written with an empty segmentation: CVAT renders that as an invisible,
    unselectable annotation the corrector cannot even delete."""
    d = _det([("grass_weed", (0, 2, 0, 2))])       # 4 px, under min_area_px=24
    insts = pi._coco_instances(d)
    assert insts == []


def test_bbox_is_derived_from_the_polygon_it_ships_with(tmp_path, monkeypatch):
    """The box and the mask have to describe the same object - bench_mixed
    matches on one and scores the other."""
    d = _det([("grass_weed", (10, 30, 20, 50))])
    inst = pi._coco_instances(d)[0]
    x, y, w, h = inst["bbox"]
    xs = [v for p in inst["segmentation"] for v in p[0::2]]
    ys = [v for p in inst["segmentation"] for v in p[1::2]]
    assert (x, y) == (min(xs), min(ys))
    assert (w, h) == (max(xs) - min(xs), max(ys) - min(ys))


def test_area_is_mask_pixels_not_bbox_area(tmp_path, monkeypatch):
    """Ranking by area is a proxy for plant size, and a bbox over a sprawling
    grass weed is mostly soil."""
    d = _det([("grass_weed", (10, 30, 20, 50))])
    inst = pi._coco_instances(d)[0]
    assert inst["area"] == pytest.approx(20 * 30)


def test_a_class_outside_the_ontology_is_skipped_not_crashed(tmp_path,
                                                             monkeypatch):
    """A checkpoint trained with a stray class name must not take the whole
    export down after the GPU pass has already been paid for."""
    d = _det([("grass_weed", (0, 20, 0, 40))])
    insts = pi._coco_instances(d)
    insts.append({**insts[0], "class_name": "not_in_the_ontology"})
    out = tmp_path / "o"
    out.mkdir()
    n = pi._write_coco([("f.png", 60, 80, insts)], list(CLASSES),
                       out, "ckpt.pth", 0.25)
    assert n == 1, "the known class still exports"


# --------------------------------------------------------------------------- #
# the failure the user actually hit
# --------------------------------------------------------------------------- #
def test_selftrain_reads_the_name_predict_writes(tmp_path, monkeypatch):
    """These two agreeing is the entire contract between the stages, and they
    are in different files with the string hard-coded in each."""
    import inspect
    from training.datasets import weeds_selftrain as st
    # The whole module, not one function: which function opens it is a detail
    # that has already moved once, and the contract is about the NAME.
    assert "instances_default.json" in inspect.getsource(st)
    assert "instances_default.json" in inspect.getsource(pi._write_coco)


def test_the_crop_class_is_exported_like_any_other(tmp_path, monkeypatch):
    """Onion predictions are how weeds-overlapping-crop gets checked later."""
    out = _run(tmp_path, monkeypatch,
               _det([(CROP_CLASS, (0, 20, 0, 40))]))
    doc = json.loads((out / "instances_default.json").read_text())
    ids = {c["id"]: c["name"] for c in doc["categories"]}
    assert {ids[a["category_id"]] for a in doc["annotations"]} == {CROP_CLASS}
