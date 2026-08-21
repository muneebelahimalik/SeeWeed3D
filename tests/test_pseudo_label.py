"""Self-training: which predictions are safe to train on.

THE FAILURE THIS IS DESIGNED AGAINST
------------------------------------
Keeping the frames the model scored highest and training on them is
self-confirmation. Where the model is already right the loss is near zero, so
nothing is learned; its confident errors meanwhile become training truth and
compound. Confidence is the wrong instrument precisely because it is the
model's own opinion.

The vegetation prior is the independent witness: a per-pixel colour
computation that knows nothing about the model and cannot be talked into
agreeing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import pseudo_label as pl                      # noqa: E402


def scene(shape=(60, 60)):
    """Soil, with a saturated green patch that ExG will find."""
    bgr = np.full(shape + (3,), (70, 45, 60), np.uint8)
    return bgr


def plant(bgr, y0, y1, x0, x1):
    bgr[y0:y1, x0:x1] = (35, 160, 45)
    return bgr


def mask(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def veg_of(bgr):
    from common.vegetation import vegetation_mask
    return vegetation_mask(bgr, 0.05, 40, 3, 1)


# --------------------------------------------------------------------------- #
# The independent witness
# --------------------------------------------------------------------------- #
def test_a_mask_on_the_plant_scores_well_on_both_vegetation_terms():
    shape = (60, 60)
    bgr = plant(scene(shape), 20, 40, 20, 40)
    q = pl.frame_quality(bgr, [mask(shape, 20, 40, 20, 40)], [0.9],
                         veg=veg_of(bgr))
    assert q["veg_precision"] > 0.9 and q["veg_recall"] > 0.9


def test_a_mask_on_soil_fails_the_precision_gate_however_confident():
    """Confidence 0.99 must not rescue a mask that is sitting on dirt."""
    shape = (60, 60)
    bgr = plant(scene(shape), 5, 15, 5, 15)
    q = pl.frame_quality(bgr, [mask(shape, 35, 55, 35, 55)], [0.99],
                         veg=veg_of(bgr))
    assert q["veg_precision"] < pl.GATES["veg_precision"]
    assert "veg_precision" in q["gates_failed"]
    assert pl.classify(q) == "review"


def test_missed_plants_fail_the_recall_gate():
    """THE DANGEROUS CASE. Every detection is correct and the pseudo-label is
    catastrophic: the plants the model skipped become BACKGROUND, teaching it
    that plants like those are soil."""
    shape = (60, 60)
    bgr = scene(shape)
    for x in (5, 25, 45):
        plant(bgr, 10, 22, x, x + 12)
    q = pl.frame_quality(bgr, [mask(shape, 10, 22, 5, 17)], [0.95],
                         veg=veg_of(bgr))
    assert q["veg_precision"] > 0.8, "the one detection it made is correct"
    assert q["veg_recall"] < pl.GATES["veg_recall"]
    assert pl.classify(q) == "review"


def test_confidence_alone_cannot_reach_the_accept_threshold():
    """The whole design: a perfect-confidence frame with no corroboration must
    not be accepted, or this is a confidence filter with extra steps."""
    q = {"veg_precision": 0.0, "veg_recall": 0.0, "stability": 1.0,
         "confidence": 1.0, "score": pl.WEIGHTS["stability"] + pl.WEIGHTS["confidence"],
         "gates_failed": ["veg_precision", "veg_recall"], "empty": False,
         "n_instances": 5}
    assert q["score"] < pl.ACCEPT_SCORE
    assert pl.classify(q) == "review"


def test_confidence_is_the_smallest_weight():
    assert pl.WEIGHTS["confidence"] == min(pl.WEIGHTS.values())
    veg = pl.WEIGHTS["veg_precision"] + pl.WEIGHTS["veg_recall"]
    assert veg > 0.5, "the independent terms must dominate the score"


def test_the_weights_sum_to_one():
    assert sum(pl.WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Degenerate frames
# --------------------------------------------------------------------------- #
def test_a_frame_with_no_predictions_is_not_scored_as_perfect():
    """Vacuous truth would rank a frame the model found NOTHING in above every
    real one - and its pseudo-label would declare the whole frame background."""
    shape = (60, 60)
    bgr = plant(scene(shape), 20, 40, 20, 40)
    q = pl.frame_quality(bgr, [], [], veg=veg_of(bgr))
    assert q["veg_precision"] == 0.0
    assert pl.classify(q) == "review"


def test_bare_ground_is_skipped_not_accepted():
    """Nothing to learn and nothing to get wrong."""
    bgr = scene()
    q = pl.frame_quality(bgr, [], [], veg=np.zeros((60, 60), bool))
    assert q["empty"] and pl.classify(q) == "skip"


def test_duplicate_detections_lower_stability():
    shape = (60, 60)
    bgr = plant(scene(shape), 20, 40, 20, 40)
    m = [mask(shape, 20, 40, 20, 40)]
    clean = pl.frame_quality(bgr, m, [0.9], n_suppressed=0, veg=veg_of(bgr))
    noisy = pl.frame_quality(bgr, m, [0.9], n_suppressed=4, veg=veg_of(bgr))
    assert noisy["stability"] < clean["stability"]
    assert noisy["score"] < clean["score"]


# --------------------------------------------------------------------------- #
# A gate failure is a REVIEW frame, never a discarded one
# --------------------------------------------------------------------------- #
def test_a_gate_failure_never_becomes_skip():
    """A frame the model got badly wrong is the most valuable thing a human can
    annotate. Discarding it throws away the loop's best signal."""
    q = {"veg_precision": 0.1, "veg_recall": 0.1, "stability": 1.0,
         "confidence": 1.0, "score": 0.95,
         "gates_failed": ["veg_precision"], "empty": False, "n_instances": 3}
    assert pl.classify(q) == "review", "a high score must not override a gate"


def test_the_review_threshold_is_below_the_accept_threshold():
    assert pl.REVIEW_SCORE < pl.ACCEPT_SCORE


# --------------------------------------------------------------------------- #
# The guardrails
# --------------------------------------------------------------------------- #
def test_the_pseudo_budget_is_capped_against_hand_corrected_frames():
    assert pl.pseudo_budget(40) == 80
    assert pl.pseudo_budget(40, n_pseudo_existing=60) == 20


def test_the_budget_cannot_bootstrap_itself():
    """Counted against HAND-CORRECTED frames, not against the dataset, so a
    mostly-pseudo dataset cannot use its own size to justify more."""
    assert pl.pseudo_budget(40, n_pseudo_existing=80) == 0
    assert pl.pseudo_budget(0, n_pseudo_existing=0) == 0


def test_class_balance_caps_the_dominant_class():
    """Without this the loop concentrates: the dominant class is predicted most
    confidently, those frames score highest, they are added, it grows more
    dominant."""
    chosen = [f"f{i}" for i in range(10)]
    class_of = {f: ("other_weed" if i < 8 else "grass_weed")
                for i, f in enumerate(chosen)}
    kept = pl.balance_by_class(chosen, class_of, cap_frac=0.6)
    n_other = sum(1 for f in kept if class_of[f] == "other_weed")
    assert n_other <= 6
    assert any(class_of[f] == "grass_weed" for f in kept)


def test_class_balance_of_nothing_is_nothing():
    assert pl.balance_by_class([], {}) == []


def test_pseudo_label_is_a_distinct_provenance_from_prelabel():
    """A SAM prelabel is a different model with a different prior - independent
    evidence. A pseudo-label is this model's own output fed back. A val score
    computed against the second measures self-consistency."""
    src = (ROOT / "training" / "prepare_dataset.py").read_text()
    assert '"pseudo_label"' in src
    assert '"prelabel_unreviewed"' in src


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def make_q(score, gates=(), empty=False):
    return {"veg_precision": 0.9, "veg_recall": 0.9, "stability": 1.0,
            "confidence": 0.8, "score": score, "gates_failed": list(gates),
            "empty": empty, "n_instances": 3}


def test_the_summary_counts_each_bucket():
    qs = [make_q(0.9), make_q(0.8), make_q(0.5), make_q(0.2),
          make_q(0.9, gates=["veg_recall"]), make_q(0.0, empty=True)]
    s = pl.summarise(qs)
    assert s["n_frames"] == 6
    assert s["accept"] == 2 and s["skip"] == 2
    assert s["review"] == 2          # the 0.2 and the gate failure
    assert s["gate_failures"]["veg_recall"] == 1


def test_the_report_sweeps_the_threshold_rather_than_asserting_one():
    """A single threshold is a claim about a threshold, not about the data."""
    s = pl.summarise([make_q(x / 10) for x in range(11)])
    cuts = [c for c, _ in s["accept_sweep"]]
    assert cuts == sorted(cuts) and len(cuts) >= 4
    counts = [n for _, n in s["accept_sweep"]]
    assert counts == sorted(counts, reverse=True), "a higher cut cannot accept more"


def test_the_report_says_to_annotate_the_review_frames():
    """The accepted frames are the ones that teach least. If the report does not
    say so, the loop gets run on half of itself."""
    text = pl.format_report(pl.summarise([make_q(0.9)]), n_hand=40, budget=80)
    assert "ANNOTATE THE REVIEW FRAMES" in text
    assert "model's own opinion" in text


def test_the_report_explains_a_recall_gate_failure():
    s = pl.summarise([make_q(0.9, gates=["veg_recall"])])
    assert "BACKGROUND" in pl.format_report(s)


def test_the_report_survives_an_empty_run():
    pl.format_report(pl.summarise([]))
