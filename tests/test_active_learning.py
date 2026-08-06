"""Active learning: does the ranking actually pick the frames worth annotating?

The failure that matters is subtle - a plausible-looking ranking that returns
30 near-duplicates of one hard patch - so diversity is tested explicitly, not
assumed."""
import numpy as np
import pytest

from conftest import load_script

al = load_script("training/active_learning.py")

from common.ontology import CLASSES, CROP_CLASS  # noqa: E402


def _target(cls="other_weed", conf=0.9, sigma=2.0, abstained=False,
            reasons=(), onion_dist=None):
    return {"class_name": cls, "class_confidence": conf, "lep_sigma_px": sigma,
            "abstained": abstained, "rejection_reasons": list(reasons),
            "safety_notes": ({"onion_distance_px": onion_dist}
                             if onion_dist is not None else {})}


def _frame(fid, targets, sid="s1"):
    return {"frame_id": fid, "session_id": sid, "targets": targets}


def _freq(**over):
    f = {c: 100 for c in CLASSES}
    f.update(over)
    return f


# --------------------------------------------------------------------------- #
# Individual signals
# --------------------------------------------------------------------------- #
def test_ambiguous_detections_score_higher_than_confident_ones():
    """Both extremes teach nothing: a 0.95 detection is already known, a 0.05
    one is background already rejected. The information is in between."""
    confident = al.score_frame(_frame("a", [_target(conf=0.95)] * 4), _freq())
    ambiguous = al.score_frame(_frame("b", [_target(conf=0.45)] * 4), _freq())
    rejected = al.score_frame(_frame("c", [_target(conf=0.03)] * 4), _freq())
    assert ambiguous.uncertainty > confident.uncertainty
    assert ambiguous.uncertainty > rejected.uncertainty


def test_poorly_localised_leps_raise_the_score():
    tight = al.score_frame(_frame("a", [_target(sigma=0.5)] * 3), _freq())
    loose = al.score_frame(_frame("b", [_target(sigma=30.0)] * 3), _freq())
    assert loose.lep_quality > tight.lep_quality


def test_abstentions_count_as_lep_uncertainty():
    """A frame where the model refuses to place LEPs is exactly a frame whose
    LEPs it needs taught."""
    ok = al.score_frame(_frame("a", [_target(abstained=False)] * 4), _freq())
    absts = al.score_frame(_frame("b", [_target(abstained=True)] * 4), _freq())
    assert absts.lep_quality > ok.lep_quality
    assert absts.n_abstained == 4


def test_scarce_classes_dominate_the_ranking():
    """THE case here: no wild_radish, almost no weed_cluster. A class with zero
    examples can never be predicted, so a frame that might contain one is worth
    far more than another primrose frame."""
    freq = _freq(wild_radish=0, weed_cluster=2, other_weed=400)
    common = al.score_frame(_frame("a", [_target("other_weed", conf=0.9)]), freq)
    rare = al.score_frame(_frame("b", [_target("wild_radish", conf=0.9)]), freq)
    scarce = al.score_frame(_frame("c", [_target("weed_cluster", conf=0.9)]), freq)

    assert rare.rarity > scarce.rarity > common.rarity
    assert rare.total > common.total
    assert any("scarce" in r for r in rare.reasons)


def test_frames_near_onion_tissue_are_prioritised():
    """Where a mistake damages the crop, the training target must be right -
    regardless of how confident the model happens to be."""
    far = al.score_frame(_frame("a", [_target(onion_dist=500.0)]), _freq())
    near = al.score_frame(_frame("b", [_target(onion_dist=10.0)]), _freq())
    conflict = al.score_frame(
        _frame("c", [_target(reasons=["onion_safety_conflict"])]), _freq())
    assert near.crop_risk > far.crop_risk
    assert conflict.crop_risk > 0
    assert any("onion" in r for r in near.reasons)


def test_empty_frames_are_flagged_rather_than_silently_ranked_last():
    """No detections is ambiguous: genuinely bare ground, or a total miss. The
    second is very valuable, so it must be surfaced for a human to judge."""
    s = al.score_frame(_frame("a", []), _freq())
    assert s.n_instances == 0
    assert any("found nothing" in r for r in s.reasons)


# --------------------------------------------------------------------------- #
# Diversity - the failure that actually bites
# --------------------------------------------------------------------------- #
def test_greedy_selection_spreads_across_clusters():
    """Pure top-k on uncertainty returns near-duplicates of one hard patch,
    because if a frame confuses the model its neighbours confuse it
    identically. Farthest-point must spread the batch."""
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float)
    desc, cluster = [], []
    for ci, c in enumerate(centres):
        for _ in range(10):
            desc.append(c + rng.normal(0, 0.2, 2))
            cluster.append(ci)

    idx = al.greedy_diverse(desc, k=4)
    assert len(set(cluster[i] for i in idx)) == 4, "batch collapsed onto clusters"


def test_diversity_beats_naive_top_k_on_duplicate_heavy_pools():
    """End to end: a pool where the highest-scoring frames are all duplicates."""
    rng = np.random.default_rng(1)
    results, descriptors = [], {}
    # 10 near-identical HIGH-uncertainty frames...
    for i in range(10):
        fid = f"dup{i}"
        results.append(_frame(fid, [_target(conf=0.5)] * 3))
        descriptors[fid] = np.array([0.0, 0.0]) + rng.normal(0, 0.01, 2)
    # ...and 5 distinct, slightly less uncertain ones.
    for i in range(5):
        fid = f"uniq{i}"
        results.append(_frame(fid, [_target(conf=0.55)] * 3))
        descriptors[fid] = np.array([float(i + 3) * 5.0, 5.0])

    naive = al.select_next_batch(results, _freq(), k=5)
    assert naive["note"] and "near-duplicate" in naive["note"]

    diverse = al.select_next_batch(results, _freq(), k=5,
                                   descriptors=descriptors)
    picked = {s["frame_id"] for s in diverse["selected"]}
    assert sum(1 for p in picked if p.startswith("uniq")) >= 3, picked


def test_appearance_descriptor_separates_different_looking_frames():
    """Must work with no model at all, so cold start has something to use."""
    import cv2
    soil = np.full((64, 64, 3), (70, 60, 55), np.uint8)
    leafy = soil.copy()
    leafy[10:50, 10:50] = (40, 190, 60)
    other = np.full((64, 64, 3), (150, 150, 150), np.uint8)

    a, b, c = (al.appearance_descriptor(x) for x in (soil, leafy, other))
    assert a.shape == b.shape == c.shape
    assert np.linalg.norm(a - b) > 1e-3
    assert np.linalg.norm(a - c) > 1e-3


def test_cold_start_needs_no_model():
    """Before a model exists there is nothing to be uncertain about, and a
    diverse first batch is what makes the first model good enough for
    uncertainty sampling to mean anything."""
    rng = np.random.default_rng(2)
    desc = {f"f{i}": rng.normal(0, 1, 6) for i in range(40)}
    out = al.select_cold_start(desc, k=8)
    assert out["n_selected"] == 8
    assert len({s["frame_id"] for s in out["selected"]}) == 8
    assert "cold start" in out["note"]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_classes_with_zero_examples_are_called_out():
    """The single most important thing the report can say."""
    freq = _freq(wild_radish=0, weed_cluster=0, grass_weed=3)
    out = al.select_next_batch([_frame("a", [_target()])], freq, k=1)
    assert set(out["classes_with_no_examples"]) == {"wild_radish", "weed_cluster"}
    assert "grass_weed" in out["scarce_classes"]
    joined = " ".join(out["guidance"])
    assert "ZERO labelled instances" in joined
    assert "wild_radish" in joined


def test_class_frequencies_include_every_ontology_class(tmp_path):
    """A class missing from the counts entirely would hide the gap; absent
    classes must read as 0, not be omitted."""
    manifest = {"frames": [{"instances": [
        {"class_name": "other_weed"}, {"class_name": "other_weed"},
        {"class_name": CROP_CLASS}]}]}
    freq = al.labelled_class_frequencies(seg_manifest=manifest)
    assert set(freq) == set(CLASSES)
    assert freq["other_weed"] == 2 and freq[CROP_CLASS] == 1
    assert freq["wild_radish"] == 0


def test_frequencies_can_be_read_from_the_lep_manifest():
    manifest = {"rows": [{"class_name": "grass_weed"},
                         {"class_name": "grass_weed"},
                         {"class_name": "cutleaf_evening_primrose"}]}
    freq = al.labelled_class_frequencies(lep_manifest=manifest)
    assert freq["grass_weed"] == 2
    assert freq["cutleaf_evening_primrose"] == 1


def test_selection_never_returns_more_than_the_pool():
    out = al.select_next_batch([_frame(f"f{i}", [_target()]) for i in range(3)],
                               _freq(), k=50)
    assert out["n_selected"] == 3
    assert len({s["frame_id"] for s in out["selected"]}) == 3
