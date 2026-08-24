"""The self-training round, end to end on synthetic predictions.

The scoring is tested in test_pseudo_label.py. This exercises the plumbing:
that a prediction COCO in, produces two batches out, with the right frames in
each and provenance stamped on both.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import pseudo_label as pl                       # noqa: E402


def soil(shape=(80, 80)):
    return np.full(shape + (3,), (70, 45, 60), np.uint8)


def with_plants(bgr, boxes):
    for (y0, y1, x0, x1) in boxes:
        bgr[y0:y1, x0:x1] = (35, 160, 45)
    return bgr


def poly_of(y0, y1, x0, x1):
    return [float(x0), float(y0), float(x1), float(y0),
            float(x1), float(y1), float(x0), float(y1)]


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A pool session, a prediction COCO over it, and the runner pointed there."""
    pool = tmp_path / "pool"
    sess = pool / "vid_test"
    (sess / "rgb").mkdir(parents=True)
    pred = tmp_path / "look"
    (pred / "overlays").mkdir(parents=True)

    boxes = [(20, 40, 20, 40), (20, 40, 50, 70)]
    images, anns, ann_id = [], [], 1

    def frame(name, plant_boxes, pred_boxes, score=0.9):
        nonlocal ann_id
        bgr = with_plants(soil(), plant_boxes)
        cv2.imwrite(str(sess / "rgb" / name), bgr)
        cv2.imwrite(str(pred / "overlays" / name), bgr)
        iid = len(images) + 1
        images.append({"id": iid, "file_name": name, "height": 80, "width": 80})
        for b in pred_boxes:
            anns.append({"id": ann_id, "image_id": iid, "category_id": 35,
                         "segmentation": [poly_of(*b)], "iscrowd": 0,
                         "bbox": [b[2], b[0], b[3] - b[2], b[1] - b[0]],
                         "area": float((b[1] - b[0]) * (b[3] - b[2])),
                         "score": score})
            ann_id += 1

    # GOOD: predictions cover every plant.
    for i in range(4):
        frame(f"good_{i}.png", boxes, boxes)
    # BAD: two plants, one predicted. Every detection correct, half the
    # vegetation would become background.
    for i in range(2):
        frame(f"miss_{i}.png", boxes, boxes[:1])
    # BAD: a mask on bare soil.
    frame("soil_0.png", [boxes[0]], [(55, 75, 55, 75)])

    (pred / "instances_default.json").write_text(json.dumps({
        "info": {"description": "SeeWeed3D MODEL PREDICTIONS - not ground truth"},
        "images": images, "annotations": anns,
        "categories": [{"id": 35, "name": "grass_weed"}]}))

    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    monkeypatch.setattr(mod, "WEED_POOL_ROOT", str(pool))
    monkeypatch.setattr(mod, "SESSION", "vid_test")
    monkeypatch.setattr(mod, "PREDICTIONS", str(pred))
    monkeypatch.setattr(mod, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(mod, "HOLDOUT_TEST", [])
    monkeypatch.setattr(mod, "N_HAND", 40)
    return mod, tmp_path / "out"


def batch(out, which):
    p = out / which / "instances_default.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_it_runs_and_writes_both_batches(run):
    mod, out = run
    assert mod.main() == 0
    assert batch(out, "accept") is not None
    assert batch(out, "review") is not None
    assert (out / "selftrain_report.json").exists()


def test_frames_covering_their_vegetation_are_accepted(run):
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert any(f.startswith("good_") for f in rep["accepted"])


def test_frames_that_miss_plants_go_to_review_not_accept(run):
    """THE case this exists for: every detection is correct, and the missed
    plants would become BACKGROUND in the pseudo-label."""
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert not any(f.startswith("miss_") for f in rep["accepted"])
    assert any(f.startswith("miss_") for f in rep["review"])


def test_a_mask_on_soil_goes_to_review(run):
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert not any(f.startswith("soil_") for f in rep["accepted"])


def test_both_batches_carry_provenance_in_the_coco(run):
    """A batch that cannot say where its labels came from is indistinguishable
    from a hand-corrected export six months later."""
    mod, out = run
    mod.main()
    assert "PSEUDO-LABELS" in batch(out, "accept")["info"]["description"]
    assert "FOR CORRECTION" in batch(out, "review")["info"]["description"]


def test_the_images_are_placed_beside_the_coco_for_cvat(run):
    mod, out = run
    mod.main()
    ready = out / "accept" / "cvat_ready"
    assert ready.is_dir() and list(ready.glob("*.png"))


def test_a_spot_check_sample_is_always_written(run):
    """Minutes of work, and the only thing between a bad threshold and a
    poisoned dataset."""
    mod, out = run
    mod.main()
    assert (out / "spot_check").is_dir()
    assert list((out / "spot_check").glob("*.png"))


def test_it_refuses_to_pseudo_label_a_holdout(run, monkeypatch):
    """The model's own output in its own test set: every later round would
    score against what it already believes."""
    mod, out = run
    monkeypatch.setattr(mod, "HOLDOUT_TEST", ["vid_test"])
    with pytest.raises(SystemExit, match="HOLDOUT_TEST"):
        mod.main()


def test_no_predictions_and_no_images_names_both(run, monkeypatch, tmp_path):
    """It generates predictions itself when it can, so the only unrecoverable
    case is having neither. The error has to name both, or the reader fixes the
    wrong one."""
    mod, out = run
    monkeypatch.setattr(mod, "PREDICTIONS", str(tmp_path / "nothing"))
    monkeypatch.setattr(mod, "IMAGES", str(tmp_path / "also-nothing"))
    with pytest.raises(SystemExit) as e:
        mod.main()
    msg = str(e.value)
    assert "no predictions at" in msg and "IMAGES does not exist" in msg


def test_a_missing_checkpoint_names_the_trainer(run, monkeypatch, tmp_path):
    """Predictions can be generated, but only if a model exists. Sending
    someone to the scorer when the real gap is an untrained round wastes the
    one thing this loop is short of."""
    mod, out = run
    images = tmp_path / "frames"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    monkeypatch.setattr(mod, "PREDICTIONS", str(tmp_path / "nothing"))
    monkeypatch.setattr(mod, "IMAGES", str(images))
    monkeypatch.setattr(mod, "RUNS_ROOT", str(tmp_path / "no-runs"))
    with pytest.raises(SystemExit, match="weeds_train"):
        mod.main()


def test_the_annotation_areas_are_per_instance(run):
    """They were indexed by the running annotation count, which paired an
    instance with another instance's area."""
    mod, out = run
    mod.main()
    doc = batch(out, "accept")
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        assert a["area"] == pytest.approx(w * h, rel=0.01)


def test_the_report_records_the_thresholds_it_used(run):
    """A batch whose threshold is not recorded cannot be compared with the next
    round's."""
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert rep["accept_threshold"] == pl.ACCEPT_SCORE
    assert rep["n_hand_corrected"] == 40
    assert rep["pseudo_budget"] == 80


# --------------------------------------------------------------------------- #
# The crash this cost a GPU pass to find
# --------------------------------------------------------------------------- #
def test_it_does_not_build_a_mask_per_instance(run):
    """bench_mixed._from_coco materialises one FULL-FRAME mask per annotation.
    That is right for a benchmark of a few frames and fatal here: a real weed
    session came back with 79 frames and 2,840 instances, which at 1242x2208
    bool is 7.8 GB held at once - and the process died AFTER the GPU pass had
    already been paid for.

    The scorer only ever needs the union per frame, so importing that helper is
    the bug itself."""
    src = (ROOT / "training" / "datasets" / "weeds_selftrain.py").read_text()
    assert "_from_coco" not in src, (
        "importing _from_coco rebuilds a full-frame mask per instance")
    assert "fillPoly" in src, "the union has to be rasterised in one array"


def test_a_frame_dense_with_instances_still_scores(run, tmp_path, monkeypatch):
    """Peak memory is not directly assertable, but the shape of the failure is:
    many instances on one frame. Under the old path this allocated one array
    per instance."""
    mod, out = run
    pred = Path(mod.PREDICTIONS)
    doc = json.loads((pred / "instances_default.json").read_text())
    base = doc["annotations"][0]
    nxt = max(a["id"] for a in doc["annotations"]) + 1
    for i in range(300):
        a = dict(base)
        a["id"] = nxt + i
        doc["annotations"].append(a)
    (pred / "instances_default.json").write_text(json.dumps(doc))
    assert mod.main() == 0
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert rep["summary"]["n_frames"] >= 7
