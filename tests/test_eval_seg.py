"""Stage A evaluation: AP maths, and the reduced-class-list crop-safety bug.

The crop tests here are regression tests for a real defect: Detections resolved
the crop index against the FULL ontology while a --drop-classes checkpoint emits
indices into a REDUCED list. Dropping any class below onion_plant made
onion_safety_mask() return an empty mask and weed_indices() hand every onion to
the targeting stage.
"""
import json

import numpy as np
import pytest

from conftest import load_script

ev = load_script("evaluation/eval_seg.py")
seg = load_script("perception/segmenter.py")
from seeweed3d.common.ontology import CLASSES, CROP_CLASS  # noqa: E402

FULL = list(CLASSES)
REDUCED = [c for c in CLASSES if c != "wild_radish"]     # what --drop-classes builds


def _det(classes, names, n=None, hw=(8, 8)):
    n = len(classes) if n is None else n
    masks = np.zeros((n,) + hw, bool)
    for i in range(n):
        masks[i, i % hw[0], :] = True
    return seg.Detections(masks, np.zeros((n, 4)), np.asarray(classes, int),
                          np.ones(n), hw[1], hw[0], names=list(names))


# --------------------------------------------------------------------------- #
# crop index resolution
# --------------------------------------------------------------------------- #
def test_crop_index_follows_the_models_own_class_list():
    assert _det([0], FULL).crop_index() == FULL.index(CROP_CLASS)
    assert _det([0], REDUCED).crop_index() == REDUCED.index(CROP_CLASS)
    # The whole point: dropping a class below the crop moves it.
    assert FULL.index(CROP_CLASS) != REDUCED.index(CROP_CLASS)


def test_onion_is_protected_under_a_reduced_class_list():
    """The regression. Index 4 is onion in REDUCED and other_weed in FULL."""
    idx = REDUCED.index(CROP_CLASS)
    d = _det([idx], REDUCED)
    m = d.onion_safety_mask()
    assert m is not None and m.any(), "crop mask lost under a reduced class list"
    assert d.weed_indices() == [], "an onion was handed over as a weed"


def test_onion_still_protected_under_the_full_ontology():
    idx = FULL.index(CROP_CLASS)
    d = _det([idx], FULL)
    assert d.onion_safety_mask().any()
    assert d.weed_indices() == []


def test_weeds_are_still_weeds_under_a_reduced_class_list():
    weed = REDUCED.index("grass_weed")
    d = _det([weed], REDUCED)
    assert d.weed_indices() == [0]
    assert not d.onion_safety_mask().any()


def test_crop_absent_from_vocabulary_reports_unavailable_not_empty():
    """A weeds-only model cannot predict a crop. That must be distinguishable
    from 'looked and found no crop', because only one of them is safe."""
    weeds_only = [c for c in CLASSES if c != CROP_CLASS]
    d = _det([0], weeds_only)
    assert d.crop_index() is None
    assert d.onion_safety_mask() is None


def test_class_name_uses_the_models_list():
    d = _det([1], REDUCED)
    assert d.class_name(0) == REDUCED[1] == "grass_weed"
    assert d.class_name(0) != CLASSES[1]


# --------------------------------------------------------------------------- #
# average precision
# --------------------------------------------------------------------------- #
def test_ap_is_one_for_perfect_ranked_detections():
    assert ev.average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_ap_is_undefined_not_zero_without_ground_truth():
    assert ev.average_precision([0.9], [False], 0) is None


def test_ap_is_zero_when_nothing_was_detected():
    assert ev.average_precision([], [], 3) == 0.0


def test_ap_penalises_a_false_positive_ranked_above_a_true_one():
    good = ev.average_precision([0.9, 0.5], [True, True], 2)
    bad = ev.average_precision([0.9, 0.5], [False, True], 2)
    assert bad < good


def test_ap_falls_when_ground_truth_is_missed():
    """One of two objects found, perfectly. Recall caps at 0.5, so does AP."""
    assert ev.average_precision([0.9], [True], 2) == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------- #
# score-ranked matching
# --------------------------------------------------------------------------- #
def _blob(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_highest_scoring_duplicate_wins_and_the_rest_are_false_positives():
    gt = [_blob(20, 20, 2, 10, 2, 10)]
    pred = [_blob(20, 20, 2, 10, 2, 10), _blob(20, 20, 2, 10, 2, 10)]
    tp = ev.match_for_ap(pred, [0.4, 0.9], gt, 0.5)
    assert tp.tolist() == [False, True], "the duplicate must not also count"


def test_a_prediction_below_the_iou_threshold_is_not_a_match():
    gt = [_blob(20, 20, 0, 4, 0, 4)]
    pred = [_blob(20, 20, 10, 14, 10, 14)]
    assert ev.match_for_ap(pred, [0.9], gt, 0.5).tolist() == [False]


def test_each_prediction_claims_a_different_ground_truth():
    gt = [_blob(20, 20, 0, 5, 0, 5), _blob(20, 20, 10, 15, 10, 15)]
    pred = [_blob(20, 20, 0, 5, 0, 5), _blob(20, 20, 10, 15, 10, 15)]
    assert ev.match_for_ap(pred, [0.9, 0.8], gt, 0.5).tolist() == [True, True]


# --------------------------------------------------------------------------- #
# crop safety: "did not see the onion" vs "aimed at the onion"
#
# missed_onion_fraction alone scores a model that ignores onions the same as one
# that classifies them as weeds. Only the second fires a 60 W laser into the
# crop, so the two must be reported separately or the metric cannot distinguish
# latent risk from damage.
# --------------------------------------------------------------------------- #
EVAL_CLASSES = ["grass_weed", CROP_CLASS]


def _crop_safety_dataset(tmp_path, with_weed_gt=False):
    """One 20x20 frame whose top half (200 px) is ground-truth onion.

    with_weed_gt adds a weed in the bottom half, needed by any test that reads
    a weed recall - without it that figure is undefined, not zero."""
    import cv2
    root = tmp_path / "sessions" / "s1" / "rgb"
    root.mkdir(parents=True)
    cv2.imwrite(str(root / "f.png"), np.zeros((20, 20, 3), np.uint8))
    inst = [{"class_name": CROP_CLASS,
             "polygons": [[0, 0, 19, 0, 19, 9, 0, 9]]}]
    if with_weed_gt:
        inst.append({"class_name": "grass_weed",
                     "polygons": [[0, 10, 19, 10, 19, 19, 0, 19]]})
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "seg_manifest.json").write_text(json.dumps({
        "images_root": [str(tmp_path / "sessions")],
        "classes": EVAL_CLASSES,
        "frames": [{"session_id": "s1", "item_id": "f", "image_path": "f.png",
                    "width": 20, "height": 20, "split": "val",
                    "instances": inst}],
    }))
    return ds


def _run_with_predictions(tmp_path, monkeypatch, specs, full=False,
                          sweep=(), scores=None, weed_gt=False):
    """specs: [(class_name, (y0, y1))] - each becomes a full-width band."""
    masks, cls = [], []
    for name, (y0, y1) in specs:
        m = np.zeros((20, 20), bool)
        m[y0:y1, :] = True
        masks.append(m)
        cls.append(EVAL_CLASSES.index(name))
    n = len(specs)
    det = seg.Detections(
        np.asarray(masks, bool) if n else np.zeros((0, 20, 20), bool),
        np.zeros((n, 4)), np.asarray(cls, int),
        np.asarray(scores if scores is not None else [1.0] * n, float),
        20, 20, names=list(EVAL_CLASSES))

    class _Fake:
        classes = list(EVAL_CLASSES)
        def load(self): return self
        def __call__(self, _bgr): return det

    # evaluate() imports build_segmenter at call time, so the module has to
    # exist before it can be patched.
    import perception.segmenter as _ps
    monkeypatch.setattr(_ps, "build_segmenter", lambda *a, **k: _Fake())
    res = ev.evaluate("unused.pt", _crop_safety_dataset(tmp_path, weed_gt),
                      None, split="val", device="cpu", sweep=sweep)
    return res if full else res["crop_safety"]


def test_onion_the_model_ignores_is_risk_but_not_damage(tmp_path, monkeypatch):
    """No prediction at all over the onion: every crop pixel is 'missed', but
    the laser has no target there, so nothing is fired at."""
    c = _run_with_predictions(tmp_path, monkeypatch, [])
    assert c["missed_onion_px"] == 200
    assert c["weed_on_crop_px"] == 0
    assert c["frames_with_burn"] == 0


def test_onion_the_model_calls_weed_is_counted_as_damage(tmp_path, monkeypatch):
    """A weed mask over half the onion. missed_onion_px is unchanged - it
    cannot tell these two cases apart, which is exactly why the second number
    exists."""
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (0, 5))])
    assert c["missed_onion_px"] == 200          # identical to the case above
    assert c["weed_on_crop_px"] == 100          # 5 rows x 20 - and this is not
    assert c["weed_on_crop_fraction"] == pytest.approx(0.5)
    assert c["frames_with_burn"] == 1


def test_a_weed_overlapping_a_predicted_onion_is_not_a_burn(tmp_path,
                                                            monkeypatch):
    """The pipeline's onion-conflict check suppresses a shot whose weed mask
    overlaps predicted crop, so scoring it as damage would report a burn the
    robot would never fire."""
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (0, 5)), (CROP_CLASS, (0, 10))])
    assert c["weed_on_crop_px"] == 0
    assert c["missed_onion_px"] == 0


def test_the_burn_is_a_subset_of_the_missed_pixels(tmp_path, monkeypatch):
    """Onion the model both fails to see AND aims at is counted in both, and
    can never exceed what it failed to see."""
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (0, 5)), (CROP_CLASS, (5, 10))])
    assert c["missed_onion_px"] == 100
    assert c["weed_on_crop_px"] == 100
    assert c["weed_on_crop_px"] <= c["missed_onion_px"]


def test_the_table_names_the_burn_rather_than_burying_it(tmp_path, monkeypatch):
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (0, 5))])
    txt = ev.format_report({"summary": {"split": "val", "n_frames": 1,
                                        "classes": [],
                                        "map50": None, "map50_95": None,
                                        "classes_without_ground_truth": []},
                            "detection": {},
                            "operating_point": {"conf": 0.5,
                                                "small_weed_recall": None,
                                                "small_weed_n": 0},
                            "crop_safety": c})
    assert "ONION CALLED WEED" in txt
    assert "100" in txt


# --------------------------------------------------------------------------- #
# the CLI actually reaching evaluate()
# --------------------------------------------------------------------------- #
def test_the_backend_flag_reaches_evaluate(tmp_path, monkeypatch):
    """--backend was parsed and then dropped, so an RF-DETR checkpoint was
    loaded through the Mask R-CNN builder - which silently makes the one
    comparison the second backend exists for meaningless."""
    seen = {}
    monkeypatch.setattr(ev, "evaluate",
                        lambda *a, **k: seen.update(k) or _EMPTY)
    monkeypatch.setattr(ev, "format_report", lambda r: "")
    ev.main(["--checkpoint", str(tmp_path / "c.pth"),
             "--dataset", str(tmp_path), "--backend", "rfdetr",
             "--out", str(tmp_path / "m.json")])
    assert seen.get("backend") == "rfdetr"


def test_images_root_may_come_from_the_manifest(tmp_path, monkeypatch):
    """report.py already defaults to the manifest's images_root. eval_seg
    required it, so the two tools needed different command lines for the same
    checkpoint and the shorter one just failed."""
    seen = {}
    monkeypatch.setattr(ev, "evaluate",
                        lambda *a, **k: seen.update(roots=a[2]) or _EMPTY)
    monkeypatch.setattr(ev, "format_report", lambda r: "")
    ev.main(["--checkpoint", str(tmp_path / "c.pth"),
             "--dataset", str(tmp_path), "--out", str(tmp_path / "m.json")])
    assert seen["roots"] is None       # evaluate() falls back to the manifest


_EMPTY = {"summary": {}, "detection": {}, "operating_point": {},
          "crop_safety": {}}


# --------------------------------------------------------------------------- #
# the confidence sweep
#
# run4 moved small-weed recall from 0.28 to 0.73 between conf 0.5 and 0.25 on
# UNCHANGED weights. A single-threshold table hides that the operating point,
# not the model, is often what the number is measuring.
# --------------------------------------------------------------------------- #
def test_the_sweep_reports_every_requested_confidence(tmp_path, monkeypatch):
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (10, 20))], full=True,
                              sweep=[0.2, 0.4, 0.6])
    assert [r["conf"] for r in c["conf_sweep"]] == [0.2, 0.4, 0.5, 0.6]


def test_the_primary_conf_is_always_in_the_sweep(tmp_path, monkeypatch):
    """Otherwise the printed table and the swept table disagree about the
    threshold the run is actually reported at."""
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (10, 20))], full=True,
                              sweep=[0.9])
    assert 0.5 in [r["conf"] for r in c["conf_sweep"]]


def test_raising_the_threshold_cannot_increase_recall(tmp_path, monkeypatch):
    """The monotonicity that makes the table meaningful. A prediction kept at
    0.6 is kept at 0.2, so recall can only fall as the threshold rises."""
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (10, 20)), (CROP_CLASS, (0, 10))],
                              full=True, sweep=[0.1, 0.3, 0.5, 0.7, 0.95],
                              scores=[0.6, 0.9], weed_gt=True)
    rec = [r["weed_recall"] for r in c["conf_sweep"]]
    assert rec == sorted(rec, reverse=True), rec
    assert rec[0] == 1.0, "the weed is found at the lowest threshold"
    assert rec[-1] == 0.0, "and dropped once the threshold passes its score"


def test_the_operating_point_matches_its_row_in_the_sweep(tmp_path,
                                                          monkeypatch):
    """Two code paths computing the same thing must not drift."""
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (10, 20))], full=True,
                              sweep=[0.25, 0.5])
    row = next(r for r in c["conf_sweep"] if r["conf"] == 0.5)
    assert row["small_weed_recall"] == \
        c["operating_point"]["small_weed_recall"]
    assert row["missed_onion_fraction"] == \
        c["crop_safety"]["missed_onion_fraction"]


def test_no_sweep_requested_still_reports_the_single_operating_point(
        tmp_path, monkeypatch):
    c = _run_with_predictions(tmp_path, monkeypatch,
                              [("grass_weed", (10, 20))], full=True)
    assert len(c["conf_sweep"]) == 1
    assert c["conf_sweep"][0]["conf"] == 0.5


def test_the_sweep_table_is_only_printed_when_there_is_something_to_compare():
    base = {"summary": {"split": "val", "n_frames": 1, "classes": [],
                        "map50": None, "map50_95": None,
                        "classes_without_ground_truth": []},
            "detection": {},
            "operating_point": {"conf": 0.5, "small_weed_recall": None,
                                "small_weed_n": 0},
            "crop_safety": {"note": "no onion_plant ground truth"}}
    one = ev.format_report({**base, "conf_sweep": [{"conf": 0.5}]})
    assert "CONFIDENCE SWEEP" not in one

    rows = [{"conf": t, "small_weed_recall": 0.5, "weed_recall": 0.5,
             "weed_precision": 0.5, "crop_recall": 0.5,
             "missed_onion_fraction": 0.1, "weed_on_crop_fraction": 0.001}
            for t in (0.25, 0.5)]
    many = ev.format_report({**base, "conf_sweep": rows})
    assert "CONFIDENCE SWEEP" in many
    assert "ONION BURNED" in many


def test_weed_recall_pools_classes_rather_than_averaging_them():
    """A class with three instances must not swing the number that decides the
    deployment threshold."""
    classes = ["grass_weed", "other_weed", CROP_CLASS]
    o = {"grass_weed": {"tp": 90, "n_gt": 100, "n_pred": 95},
         "other_weed": {"tp": 0, "n_gt": 3, "n_pred": 1},
         CROP_CLASS: {"tp": 50, "n_gt": 50, "n_pred": 50}}
    pooled = ev._weed_recall(o, classes)
    assert pooled == pytest.approx(90 / 103)
    per_class_mean = (0.9 + 0.0) / 2
    assert pooled > per_class_mean          # the rare class no longer dominates
    # and the crop is excluded from a WEED figure
    assert ev._weed_precision(o, classes) == pytest.approx(90 / 96)
