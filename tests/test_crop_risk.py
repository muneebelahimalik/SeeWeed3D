"""Crop risk: does the weed model claim onion tissue?

The two rates are deliberately different measurements, so most of these tests
are about the cases where they DISAGREE - one sprawling detection over six
onions, six small ones on a single onion. A version that reported either alone
would pass a test that only ever exercised the easy case.
"""
import json

import numpy as np
import pytest

from conftest import load_script

cr = load_script("evaluation/crop_risk.py")

from common.ontology import CROP_CLASS  # noqa: E402

H = W = 60


def _sq(x, y, w, h=None):
    h = w if h is None else h
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _onion(poly):
    return (CROP_CLASS, [poly], 1.0)


def _pred(cls, poly, score=0.9):
    return (cls, [poly], score)


# --------------------------------------------------------------------------
# the two directions


def test_detection_fully_on_crop_counts_both_ways():
    onions = [_onion(_sq(10, 10, 20))]
    preds = [_pred("grass_weed", _sq(12, 12, 10))]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_on_crop"] == 1 and r["n_off_crop"] == 0
    assert r["n_endangered"] == 1
    assert r["hits_by_class"] == {"grass_weed": 1}


def test_detection_far_from_any_onion_is_not_an_error():
    """Onion drives contain real, unannotated weeds. Off-crop is unknown."""
    onions = [_onion(_sq(0, 0, 10))]
    preds = [_pred("other_weed", _sq(40, 40, 10))]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_on_crop"] == 0 and r["n_off_crop"] == 1
    assert r["n_endangered"] == 0 and r["hits_by_class"] == {}


def test_one_sprawling_detection_endangers_many_plants():
    """One bad detection, six endangered plants - the rates must disagree."""
    onions = [_onion(_sq(2 + 8 * i, 5, 4)) for i in range(6)]
    preds = [_pred("weed_cluster", _sq(0, 0, 55))]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_on_crop"] == 0, "a huge mask is mostly soil, not crop"
    # ... but with the coverage threshold met from the onion side, every plant
    # is under it. This is why one number cannot stand in for the other.
    r2 = cr.frame_risk(onions, preds, H, W, det_min=0.0)
    assert r2["n_on_crop"] == 1 and r2["n_endangered"] == 6


def test_many_small_detections_on_one_plant():
    """The reverse case: six bad detections, one endangered plant."""
    onions = [_onion(_sq(10, 10, 30))]
    preds = [_pred("grass_weed", _sq(12 + 4 * i, 12, 4)) for i in range(6)]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_on_crop"] == 6
    assert r["n_endangered"] == 1


def test_partial_overlap_below_threshold_is_not_a_crop_hit():
    """A mask clipping a neighbouring leaf is a boundary error, not a mistake."""
    onions = [_onion(_sq(0, 0, 20))]
    preds = [_pred("grass_weed", _sq(18, 0, 20))]      # 2/20 on crop = 10%
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_on_crop"] == 0
    assert cr.frame_risk(onions, preds, H, W, det_min=0.05)["n_on_crop"] == 1


def test_endangered_needs_only_a_little_coverage():
    """A laser touching a tenth of a plant has damaged the plant."""
    onions = [_onion(_sq(0, 0, 20))]
    preds = [_pred("grass_weed", _sq(0, 0, 20, 3))]    # 3/20 of the onion
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_endangered"] == 1
    assert cr.frame_risk(onions, preds, H, W, onion_min=0.5)["n_endangered"] == 0


def test_score_filter_drops_low_confidence_detections():
    onions = [_onion(_sq(10, 10, 20))]
    preds = [_pred("grass_weed", _sq(12, 12, 10), score=0.4)]
    assert cr.frame_risk(onions, preds, H, W, min_score=0.5)["n_detections"] == 0
    assert cr.frame_risk(onions, preds, H, W, min_score=0.5)["n_on_crop"] == 0
    assert cr.frame_risk(onions, preds, H, W, min_score=0.3)["n_on_crop"] == 1


# --------------------------------------------------------------------------
# a model that HAS the crop class


def test_the_models_own_onion_detections_are_not_counted_as_risk():
    """Scoring a correct onion_plant mask as a crop hit would rank the fixed
    model below the broken one - the exact wrong signal."""
    onions = [_onion(_sq(10, 10, 20))]
    preds = [_pred(CROP_CLASS, _sq(10, 10, 20))]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_detections"] == 0, "a crop detection is not a weed detection"
    assert r["n_on_crop"] == 0 and r["n_endangered"] == 0
    assert r["hits_by_class"] == {}
    assert r["n_crop_detections"] == 1


def test_crop_recall_counts_onions_the_model_actually_found():
    """An onion the model never detects is one the safety mask cannot protect
    - allow_missing_crop_mask=False makes that the other half of crop safety."""
    onions = [_onion(_sq(0, 0, 20)), _onion(_sq(30, 30, 20))]
    preds = [_pred(CROP_CLASS, _sq(0, 0, 20))]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_onions"] == 2 and r["n_onions_detected"] == 1
    assert cr.summarise({"f": r})["crop_recall"] == pytest.approx(0.5)


def test_a_weed_detection_still_counts_when_the_crop_class_exists():
    """Excluding onion_plant must not excuse a weed class on the same plant."""
    onions = [_onion(_sq(10, 10, 20))]
    preds = [_pred(CROP_CLASS, _sq(10, 10, 20)),
             _pred("grass_weed", _sq(12, 12, 10))]
    r = cr.frame_risk(onions, preds, H, W)
    assert r["n_on_crop"] == 1 and r["n_endangered"] == 1
    assert r["hits_by_class"] == {"grass_weed": 1}


def test_report_shows_crop_recall_only_for_a_crop_capable_model():
    """0% recall from a model with no onion class is a tautology, not a score
    - printing it would read as a failure the model cannot have."""
    s = cr.summarise(cr.score_frames(_frames(), 0.5))
    assert "CROP RECALL" not in cr.format_report(s, model_classes=WEED_ONLY)
    assert "CROP RECALL" in cr.format_report(s, model_classes=MIXED)


# --------------------------------------------------------------------------
# pooling and the sweep


def _frames():
    hit = ([_onion(_sq(10, 10, 20))], [_pred("grass_weed", _sq(12, 12, 10), 0.95)])
    miss = ([_onion(_sq(0, 0, 10))], [_pred("other_weed", _sq(40, 40, 10), 0.6)])
    return {"a": (hit[0], hit[1], H, W), "b": (miss[0], miss[1], H, W)}


def test_summarise_pools_both_rates():
    s = cr.summarise(cr.score_frames(_frames(), 0.5))
    assert s["frames"] == 2 and s["n_onions"] == 2
    assert s["n_on_crop"] == 1 and s["n_off_crop"] == 1
    assert s["detection_on_crop_rate"] == pytest.approx(0.5)
    assert s["onion_endangered_rate"] == pytest.approx(0.5)
    assert s["hits_by_class"] == {"grass_weed": 1}


def test_summarise_of_nothing_does_not_divide_by_zero():
    s = cr.summarise({})
    assert s["detection_on_crop_rate"] == 0.0
    assert s["onion_endangered_rate"] == 0.0


def test_sweep_only_reports_at_or_above_conf():
    """Inference already discarded everything below CONF; the sweep must not
    pretend it can go under it."""
    rows = cr.sweep(_frames(), [0.25, 0.5, 0.7, 0.9], conf=0.5)
    assert [r["threshold"] for r in rows] == [0.5, 0.7, 0.9]
    # 0.6 detection survives 0.5 and dies at 0.7; the 0.95 one survives all.
    assert [r["n_detections"] for r in rows] == [2, 1, 1]
    assert rows[-1]["detection_on_crop_rate"] == pytest.approx(1.0)


def test_hits_note_calls_a_dominant_class_a_prior_not_a_resemblance():
    """398 of 407 hits were cutleaf_evening_primrose. Reading that as 'onions
    look like primrose' would chase a resemblance that is really the training
    set's majority class leaking through a model with nowhere else to put an
    onion."""
    note = "\n".join(cr.hits_note({"cutleaf_evening_primrose": 398,
                                   "other_weed": 7, "grass_weed": 2},
                                  WEED_ONLY))
    assert "majority class" in note and "class balance" in note
    assert "resemble" in note


def test_hits_note_keeps_the_appearance_reading_for_a_crop_capable_model():
    """A model that could have said onion_plant and chose grass_weed IS making
    a claim about appearance."""
    note = "\n".join(cr.hits_note({"grass_weed": 100}, MIXED))
    assert "long, thin" in note and "majority class" not in note


def test_hits_note_is_silent_when_no_class_dominates():
    assert cr.hits_note({"other_weed": 5, "cutleaf_evening_primrose": 4},
                        WEED_ONLY) == []
    assert cr.hits_note({}, WEED_ONLY) == []


def _row(t, det, on_crop_rate):
    return {"threshold": t, "n_detections": det,
            "detection_on_crop_rate": on_crop_rate}


def test_sweep_note_flags_a_threshold_that_disables_the_machine():
    """0 detections at 0.9 is not '0% risk' - it is a model whose scores never
    reach the gate, which switches the weeder off rather than making it safe."""
    note = "\n".join(cr.sweep_note([_row(0.5, 416, 0.978), _row(0.9, 0, 0.0)]))
    assert "0.90" in note and "switch the machine off" in note


def test_sweep_note_flags_confidence_ranking_the_crop_highest():
    """The on-crop share RISING with the threshold means the model is surer
    about onions than about the real weeds - so a higher bar makes it worse."""
    note = "\n".join(cr.sweep_note([_row(0.5, 416, 0.978), _row(0.7, 227, 0.991)]))
    assert "RISES" in note and "Confidence is not a fix" in note


def test_sweep_note_is_silent_when_a_higher_bar_actually_helps():
    assert not any("RISES" in n for n in
                   cr.sweep_note([_row(0.5, 416, 0.90), _row(0.7, 200, 0.40)]))


def test_sweep_note_says_nothing_about_an_empty_sweep():
    assert cr.sweep_note([_row(0.5, 0, 0.0)]) == []


def test_sweep_does_not_mutate_or_cache_masks():
    """Re-running the sweep must give the same answer - the failure mode of a
    cached-mask optimisation is a silently different second row."""
    f = _frames()
    assert cr.sweep(f, [0.5, 0.9], 0.5) == cr.sweep(f, [0.5, 0.9], 0.5)


# --------------------------------------------------------------------------
# verdict and report


def test_verdict_distinguishes_no_data_from_a_clean_run():
    """Zero onions must never read as 'no crop risk found'."""
    empty = cr.verdict(cr.summarise({}))
    assert "nothing" in empty.lower() and "SESSIONS" in empty
    clean = cr.verdict({"n_onions": 500, "onion_endangered_rate": 0.0})
    assert clean.startswith("LOW")


MIXED = ["grass_weed", CROP_CLASS]
WEED_ONLY = ["grass_weed"]


@pytest.mark.parametrize("rate,head", [(0.0, "LOW"), (0.05, "MODERATE"),
                                       (0.4, "HIGH")])
def test_verdict_bands_for_a_crop_capable_model(rate, head):
    assert cr.verdict({"n_onions": 100, "onion_endangered_rate": rate},
                      MIXED).startswith(head)


def test_a_weed_only_model_is_not_told_onions_are_confusable():
    """The trap this run walked into: a weed-only model has no class for an
    onion, so in an onion drive a high rate is near the ceiling by
    construction. Reading it as 'onions look like weeds' would buy contact
    frames on the strength of a number that could not come out low."""
    v = cr.verdict({"n_onions": 645, "onion_endangered_rate": 0.61}, WEED_ONLY)
    assert "UNSAFE" in v
    assert "no class for an onion" in v
    assert "CANNOT tell you" in v and "crop class" in v
    assert "confusable" not in v


def test_a_weed_only_model_with_a_low_rate_is_informative():
    """The one weed-only result that IS diagnostic: if it leaves crop tissue
    alone with no onion class, its weed appearance genuinely excludes onions."""
    v = cr.verdict({"n_onions": 645, "onion_endangered_rate": 0.0}, WEED_ONLY)
    assert v.startswith("LOW") and "informative" in v


def test_verdict_without_model_classes_assumes_weed_only():
    """The conservative default: an unknown class list must not be read as
    'has a crop class', which is the assumption that over-concludes."""
    s = {"n_onions": 100, "onion_endangered_rate": 0.4}
    assert cr.verdict(s) == cr.verdict(s, WEED_ONLY)


def test_report_states_the_prelabel_provenance():
    """The whole run produces one quotable percentage; it must carry the
    caveat that 'onion tissue' is itself a machine label."""
    warns = cr.provenance_warnings(WEED_ONLY)
    text = "\n".join(warns).lower()
    assert "prelabel" in text and "ground truth" in text


def test_a_weed_only_checkpoint_is_flagged_as_the_ceiling_case():
    warns = "\n".join(cr.provenance_warnings(WEED_ONLY))
    assert "[!]" in warns and "NO crop class" in warns
    assert "does NOT prove" in warns


def test_a_crop_capable_checkpoint_says_its_detections_are_excluded():
    warns = "\n".join(cr.provenance_warnings(MIXED))
    assert "excluded" in warns and "CROP RECALL" in warns
    assert "[!]" not in warns


def test_skipped_sessions_are_named_not_just_counted():
    """5 of 14 drives measured and 5 drives found look identical otherwise."""
    warns = "\n".join(cr.provenance_warnings(
        WEED_ONLY, sessions_skipped={"Visit2_x": "no onion_plant annotations"}))
    assert "Visit2_x" in warns and "no onion_plant annotations" in warns


def test_report_contains_both_rates_and_the_off_crop_caveat():
    s = cr.summarise(cr.score_frames(_frames(), 0.5))
    txt = cr.format_report(s, checkpoint="ck.pth", conf=0.5,
                           sweep_rows=cr.sweep(_frames(), [0.5, 0.9], 0.5),
                           warnings=cr.provenance_warnings(["grass_weed"]))
    assert "DETECTION-SIDE" in txt and "ONION-SIDE" in txt
    assert "NOT counted as errors" in txt
    assert "grass_weed" in txt
    assert "VERDICT" in txt


def test_report_survives_an_empty_summary():
    cr.format_report(cr.summarise({}), checkpoint="ck.pth", conf=0.5)


# --------------------------------------------------------------------------
# loading


def _write_coco(path, frames, scored=True):
    """frames: [(file_name, [(class_name, poly), ...]), ...]"""
    cats = [{"id": 1, "name": CROP_CLASS}, {"id": 2, "name": "grass_weed"}]
    by_name = {c["name"]: c["id"] for c in cats}
    images, anns, aid = [], [], 1
    for k, (fn, insts) in enumerate(frames, start=1):
        images.append({"id": k, "file_name": fn, "height": H, "width": W})
        for c, p in insts:
            a = {"id": aid, "image_id": k, "category_id": by_name[c],
                 "segmentation": [p], "bbox": [0, 0, 1, 1], "area": 1.0,
                 "iscrowd": 0}
            if scored:
                a["score"] = 0.8
            anns.append(a)
            aid += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"images": images, "annotations": anns,
                                "categories": cats}))
    return path


def test_load_polygons_reads_coco_and_filters_by_class(tmp_path):
    p = _write_coco(tmp_path / "instances_default.json",
                    [("f0.png", [(CROP_CLASS, _sq(0, 0, 5)),
                                 ("grass_weed", _sq(20, 20, 5))])])
    both = cr.load_polygons(p)
    assert len(both["f0"]) == 2
    only = cr.load_polygons(p, want_class=CROP_CLASS)
    assert [c for c, _, _ in only["f0"]] == [CROP_CLASS]


def test_load_polygons_defaults_missing_scores_to_one(tmp_path):
    """Ground truth carries no score and must survive every threshold."""
    p = _write_coco(tmp_path / "gt.json", [("f0.png", [(CROP_CLASS, _sq(0, 0, 5))])],
                    scored=False)
    assert cr.load_polygons(p)["f0"][0][2] == 1.0


def test_load_polygons_reads_datumaro(tmp_path):
    doc = {"categories": {"label": {"labels": [{"name": "grass_weed"},
                                               {"name": CROP_CLASS}]}},
           "items": [{"id": "f0", "annotations": [
               {"type": "polygon", "label_id": 1, "points": _sq(0, 0, 5)},
               {"type": "bbox", "label_id": 1, "points": [0, 0, 1, 1]}]}]}
    d = tmp_path / "sess" / "annotations"
    d.mkdir(parents=True)
    (d / "default.json").write_text(json.dumps(doc))
    got = cr.load_polygons(tmp_path / "sess", want_class=CROP_CLASS)
    assert [c for c, _, _ in got["f0"]] == [CROP_CLASS]


def test_load_polygons_ignores_unreadable_files(tmp_path):
    (tmp_path / "instances_default.json").write_text("not json")
    assert cr.load_polygons(tmp_path) == {}


def test_image_sizes_from_predictions(tmp_path):
    p = _write_coco(tmp_path / "instances_default.json", [("f0.png", [])])
    assert cr.image_sizes(p) == {"f0": (H, W)}


def test_discover_sessions_accepts_a_session_or_a_parent(tmp_path):
    for name in ("s1", "s2"):
        d = tmp_path / "pool" / name / "rgb"
        d.mkdir(parents=True)
        (d / "f0.png").write_bytes(b"x")
    (tmp_path / "pool" / "empty").mkdir()
    assert [d.name for d in cr.discover_sessions([tmp_path / "pool"])] == \
        ["s1", "s2"]
    assert [d.name for d in cr.discover_sessions([tmp_path / "pool" / "s1"])] == \
        ["s1"]
    assert cr.discover_sessions([tmp_path / "nope"]) == []


# --------------------------------------------------------------------------
# overlays


def test_worst_ranks_by_endangered_then_hits():
    pf = {"a": {"n_on_crop": 5, "n_endangered": 0},
          "b": {"n_on_crop": 1, "n_endangered": 3},
          "c": {"n_on_crop": 0, "n_endangered": 0}}
    assert cr.worst(pf, 3) == ["b", "a"], "c has no crop hit to look at"


def test_draw_overlay_marks_hits_and_keeps_off_crop_visible():
    bgr = np.zeros((H, W, 3), np.uint8)
    onions = [_onion(_sq(5, 5, 20))]
    preds = [_pred("grass_weed", _sq(6, 6, 10)),
             _pred("other_weed", _sq(40, 40, 10))]
    r = cr.frame_risk(onions, preds, H, W)
    vis = cr.draw_overlay(bgr, onions, preds, r["on_crop_idx"])
    assert vis.shape == bgr.shape
    assert vis[10, 10, 2] > vis[10, 10, 0], "crop hit drawn red"
    assert vis[45, 45].any(), "off-crop detection still drawn"
