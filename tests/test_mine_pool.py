"""Model-in-the-loop mining: which frames to annotate next.

The dataset is the limit, not the architecture, so the thing this module must
get right is not missing information: no already-labelled frame, no holdout
session, no depth PNG, and no silent claim that predictions are truth.
"""
import json

import cv2
import numpy as np
import pytest

from conftest import load_script

mp = load_script("annotation/mine_pool.py")
seg = load_script("perception/segmenter.py")
from seeweed3d.common.ontology import CLASSES, CROP_CLASS  # noqa: E402

ACTIVE = ["cutleaf_evening_primrose", "grass_weed", "other_weed", CROP_CLASS]


def _sessions(tmp_path, names=("vid1", "vid2"), n=6):
    root = tmp_path / "sessions"
    for s in names:
        (root / s / "rgb").mkdir(parents=True)
        (root / s / "depth").mkdir(parents=True)
        for i in range(1, n + 1):
            name = f"{s}_{i:06d}.png"
            cv2.imwrite(str(root / s / "rgb" / name),
                        np.random.randint(0, 255, (60, 80, 3), np.uint8))
            # SAME filename, 16-bit - must never reach the model.
            cv2.imwrite(str(root / s / "depth" / name),
                        np.full((60, 80), 900, np.uint16))
    return root


def _manifest(tmp_path, item_ids=(), classes=ACTIVE):
    d = tmp_path / "ds"
    d.mkdir(exist_ok=True)
    (d / "seg_manifest.json").write_text(json.dumps({
        "images_root": [str(tmp_path / "sessions")], "classes": list(classes),
        "frames": [{"item_id": i, "session_id": "vid1", "image_path": i + ".png",
                    "width": 80, "height": 60, "split": "train",
                    "instances": [{"class_name": "grass_weed",
                                   "polygons": [[0, 0, 9, 0, 9, 9, 0, 9]]}]}
                   for i in item_ids]}))
    return d


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #
def test_depth_pngs_never_enter_the_pool(tmp_path):
    got = mp.pool_frames(_sessions(tmp_path))
    assert got, "nothing was found at all"
    for _, f in got:
        assert "depth" not in {p.name for p in f.parents}


def test_already_labelled_frames_are_skipped(tmp_path):
    """Re-annotating a frame is worse than wasted time: the same frame in two
    CVAT tasks produces two versions of the truth and nothing downstream knows
    which to believe."""
    root = _sessions(tmp_path, n=4)
    done = {"vid1_000001", "vid1_000002"}
    got = mp.pool_frames(root, exclude_ids=done)
    assert not ({f.stem for _, f in got} & done)
    assert len(got) == 6                      # 8 total minus the 2 labelled


def test_a_holdout_session_is_never_offered_for_annotation(tmp_path):
    """A session kept as a test set stops being one the moment it is annotated
    into training. This is the last point where that is preventable."""
    root = _sessions(tmp_path, names=("vid1", "holdout_a"))
    got = mp.pool_frames(root, holdout=["holdout_a"])
    assert {s for s, _ in got} == {"vid1"}


def test_only_sessions_restricts_the_scan(tmp_path):
    root = _sessions(tmp_path, names=("vid1", "vid2"))
    assert {s for s, _ in mp.pool_frames(root, only=["vid2"])} == {"vid2"}


def test_stride_and_cap_limit_the_scan(tmp_path):
    root = _sessions(tmp_path, names=("vid1",), n=10)
    assert len(mp.pool_frames(root, stride=5)) == 2
    assert len(mp.pool_frames(root, limit=3)) == 3


def test_a_missing_sessions_root_says_so(tmp_path):
    with pytest.raises(SystemExit, match="SESSIONS_ROOT"):
        mp.pool_frames(tmp_path / "nope")


def test_labelled_item_ids_reads_the_manifest(tmp_path):
    d = _manifest(tmp_path, ["a", "b"])
    assert mp.labelled_item_ids(d) == {"a", "b"}
    assert mp.labelled_item_ids(tmp_path / "nothing") == set()


# --------------------------------------------------------------------------- #
# turning detections into a score input
# --------------------------------------------------------------------------- #
def _det(specs, hw=(60, 80), names=ACTIVE):
    masks, cls, sc = [], [], []
    for name, (y0, y1, x0, x1), s in specs:
        m = np.zeros(hw, bool)
        m[y0:y1, x0:x1] = True
        masks.append(m)
        cls.append(names.index(name))
        sc.append(s)
    n = len(specs)
    return seg.Detections(
        np.asarray(masks, bool) if n else np.zeros((0,) + hw, bool),
        np.zeros((n, 4)), np.asarray(cls, int), np.asarray(sc, float),
        hw[1], hw[0], names=list(names))


def test_a_weed_beside_an_onion_records_its_distance():
    """The crop-risk term needs it, and those are the frames where a mistake
    costs the customer a plant."""
    d = _det([("grass_weed", (0, 10, 0, 20), 0.9),
              (CROP_CLASS, (0, 10, 22, 40), 0.9)])
    r = mp.frame_result(d, "vid1", "f")
    weed = r["targets"][0]
    assert weed["safety_notes"]["onion_distance_px"] == pytest.approx(2.0,
                                                                     abs=1.5)


def test_a_model_that_cannot_see_the_crop_reports_no_distance():
    """None is not zero. A weeds-only checkpoint has no opinion about where the
    onion is, and scoring its frames as low crop-risk is exactly backwards."""
    weeds_only = [c for c in CLASSES if c != CROP_CLASS]
    d = _det([("grass_weed", (0, 10, 0, 20), 0.9)], names=weeds_only)
    r = mp.frame_result(d, "vid1", "f")
    assert "onion_distance_px" not in r["targets"][0]["safety_notes"]


def test_an_empty_frame_still_produces_a_scoreable_record():
    """'the model found nothing' is a real signal - it is either genuinely
    empty or a total miss, and the scorer says so."""
    r = mp.frame_result(_det([]), "vid1", "f")
    assert r["targets"] == [] and r["n_instances"] == 0

    from training.active_learning import score_frame
    s = score_frame(r, {c: 5 for c in CLASSES})
    assert any("found nothing" in x for x in s.reasons)


def test_the_scores_flow_through_to_a_ranking():
    from training.active_learning import score_frame
    freq = {c: 100 for c in CLASSES}
    freq["wild_radish"] = 0
    rare = mp.frame_result(_det([("grass_weed", (0, 10, 0, 20), 0.9)]),
                           "v", "rare")
    rare["targets"][0]["class_name"] = "wild_radish"
    common = mp.frame_result(_det([("grass_weed", (0, 10, 0, 20), 0.9)]),
                             "v", "common")
    assert score_frame(rare, freq).total > score_frame(common, freq).total


# --------------------------------------------------------------------------- #
# polygons and the CVAT export
# --------------------------------------------------------------------------- #
def test_a_mask_becomes_an_editable_polygon():
    """A per-pixel contour is exact and useless: dragging 3000 vertices in CVAT
    is slower than redrawing the plant."""
    m = np.zeros((100, 100), bool)
    m[20:70, 30:80] = True
    polys = mp.mask_to_polygons(m)
    assert len(polys) == 1
    assert 4 <= len(polys[0]) // 2 <= 12, "too many handles to edit"


def test_a_speck_produces_no_polygon():
    m = np.zeros((50, 50), bool)
    m[10:12, 10:12] = True
    assert mp.mask_to_polygons(m, min_area_px=24) == []


def test_two_disconnected_parts_both_survive():
    """An instance split by an occluding leaf keeps all of its parts."""
    m = np.zeros((100, 100), bool)
    m[10:30, 10:30] = True
    m[60:80, 60:80] = True
    assert len(mp.mask_to_polygons(m)) == 2


def test_the_export_uses_the_ontology_category_ids(tmp_path):
    """Stable ids across weed, onion and mixed batches, so exports merge
    without remapping a single annotation."""
    from seeweed3d.common.ontology import CATEGORY_ID
    preds = {"f1": {"path": str(_write(tmp_path, "f1.png")), "height": 60,
                    "width": 80, "session_id": "v",
                    "instances": [{"class_name": CROP_CLASS, "area_px": 100,
                                   "polygons": [[0, 0, 9, 0, 9, 9]]}]}}
    ex = mp.export_batch([{"frame_id": "f1"}], preds, tmp_path / "out", ACTIVE)
    doc = json.loads((tmp_path / "out" / "instances_default.json").read_text())
    by_name = {c["name"]: c["id"] for c in doc["categories"]}
    assert by_name[CROP_CLASS] == CATEGORY_ID[CROP_CLASS]
    assert doc["annotations"][0]["category_id"] == CATEGORY_ID[CROP_CLASS]
    assert ex["n_images"] == 1 and ex["n_instances"] == 1


def _write(tmp_path, name):
    p = tmp_path / name
    cv2.imwrite(str(p), np.zeros((60, 80, 3), np.uint8))
    return p


def test_the_images_land_beside_the_annotations(tmp_path):
    preds = {"f1": {"path": str(_write(tmp_path, "f1.png")), "height": 60,
                    "width": 80, "session_id": "v", "instances": []}}
    mp.export_batch([{"frame_id": "f1"}], preds, tmp_path / "out", ACTIVE)
    assert (tmp_path / "out" / "cvat_ready" / "f1.png").exists()


def test_a_class_outside_the_ontology_is_dropped_not_guessed(tmp_path):
    preds = {"f1": {"path": str(_write(tmp_path, "f1.png")), "height": 60,
                    "width": 80, "session_id": "v",
                    "instances": [{"class_name": "bindweed", "area_px": 10,
                                   "polygons": [[0, 0, 9, 0, 9, 9]]}]}}
    ex = mp.export_batch([{"frame_id": "f1"}], preds, tmp_path / "out",
                         ACTIVE)
    assert ex["n_instances"] == 0
