#!/usr/bin/env python3
"""
SeeWeed3D - active learning: which frames to annotate NEXT.

At 35 annotated frames the bottleneck is not model capacity, it is annotation
hours. Annotating the next 35 frames AT RANDOM mostly re-teaches what the model
already knows. This module ranks the unlabelled pool so the next round buys the
most learning per hour of your time.

THE SCORE, AND WHY EACH PART IS THERE
-------------------------------------
A pure uncertainty ranking is a classic trap: the top of the list is
overwhelmingly near-duplicate frames of the same difficult patch, because if
one frame confuses the model the frames either side of it confuse it
identically. You would annotate 20 pictures of one plant. So the score combines
four terms and then a diversity pass:

  1. UNCERTAINTY   Detections sitting in the ambiguous confidence band, plus
                   abstentions. These are where the decision boundary is.
  2. LEP QUALITY   Mean heatmap sigma and abstention rate. LEP is the hardest
                   task and the one with the fewest labels, so frames where the
                   growth point is uncertain are worth more than frames where
                   only the mask is rough.
  3. RARITY        Frames predicted to contain classes that are scarce in the
                   labelled set. With no wild_radish and almost no
                   weed_cluster, a frame that might contain one is worth far
                   more than another primrose frame - a class with zero
                   examples can never be learned.
  4. CROP RISK     Frames where a weed candidate sits close to onion tissue.
                   These are the frames where a mistake is expensive, so they
                   deserve human attention regardless of model confidence.

Then GREEDY DIVERSE SELECTION (farthest-point on an appearance descriptor)
picks the final batch, so the chosen frames are spread across the pool instead
of clustered on one hard patch.

COLD START
----------
With no trained model there is nothing to be uncertain about, so
`select_cold_start()` falls back to pure appearance diversity. That is the
correct first round and is what the initial batch should have been.

Nothing here needs a GPU: it consumes stored inference results
(`FrameResult.to_dict()`), so a ranking can be recomputed with different
weights without re-running the model.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402


@dataclass
class ScoringWeights:
    """Relative importance of the four signals. Documented, not magic.

    Rarity is weighted highest on purpose at this stage: with several classes
    at or near zero instances, class COVERAGE is worth more than refining a
    boundary the model already roughly knows."""
    uncertainty: float = 1.0
    lep_quality: float = 0.8
    rarity: float = 1.2
    crop_risk: float = 0.6

    # A detection is "ambiguous" inside this confidence band - neither a
    # confident hit nor obvious background.
    conf_low: float = 0.25
    conf_high: float = 0.70
    # sigma_px at or above this counts as a fully uncertain LEP.
    sigma_saturate_px: float = 12.0
    # A weed candidate closer than this to onion tissue is a crop-risk frame.
    crop_risk_px: float = 40.0


@dataclass
class FrameScore:
    frame_id: str
    session_id: str = ""
    total: float = 0.0
    uncertainty: float = 0.0
    lep_quality: float = 0.0
    rarity: float = 0.0
    crop_risk: float = 0.0
    n_instances: int = 0
    n_abstained: int = 0
    predicted_classes: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Per-signal scoring
# --------------------------------------------------------------------------- #
def _uncertainty_score(targets, w):
    """Fraction of detections whose class confidence is genuinely ambiguous.

    Both extremes are uninformative: a 0.95 detection teaches nothing new, and
    a 0.05 one is background the model already rejects. The information is in
    between."""
    if not targets:
        return 0.0
    amb = sum(1 for t in targets
              if w.conf_low <= float(t.get("class_confidence", 0.0)) <= w.conf_high)
    return float(amb / len(targets))


def _lep_quality_score(targets, w):
    """How poorly the growth points are localised in this frame.

    Combines mean normalised sigma with the abstention rate, because a frame
    where the model refuses to place LEPs is exactly a frame whose LEPs it
    needs taught."""
    if not targets:
        return 0.0
    sig = [min(1.0, float(t.get("lep_sigma_px", 0.0)) / max(1e-6, w.sigma_saturate_px))
           for t in targets]
    abstained = sum(1 for t in targets if t.get("abstained")) / len(targets)
    return float(0.5 * np.mean(sig) + 0.5 * abstained)


def _rarity_score(targets, class_frequencies):
    """Value of this frame's predicted classes given what is already labelled.

    Weight per class is 1/sqrt(1 + n_labelled): a class with zero examples
    scores 1.0, a class with 400 scores ~0.05. Square root rather than 1/n so a
    moderately-represented class is not treated as worthless."""
    if not targets:
        return 0.0
    best = 0.0
    for t in targets:
        n = float(class_frequencies.get(t.get("class_name", ""), 0))
        best = max(best, 1.0 / np.sqrt(1.0 + n))
    return float(best)


def _crop_risk_score(targets, w):
    """Frames where a weed candidate sits near onion tissue.

    Worth annotating regardless of confidence: this is where a mistake damages
    the crop, so the training target needs to be right there specifically."""
    if not targets:
        return 0.0
    risky = 0
    for t in targets:
        d = (t.get("safety_notes") or {}).get("onion_distance_px")
        if d is not None and float(d) <= w.crop_risk_px:
            risky += 1
        elif "onion_safety_conflict" in (t.get("rejection_reasons") or []):
            risky += 1
    return float(min(1.0, risky / max(1, len(targets))))


def score_frame(result, class_frequencies, w=None):
    """Score one stored FrameResult dict."""
    w = w or ScoringWeights()
    targets = result.get("targets", []) or []
    s = FrameScore(frame_id=result.get("frame_id", ""),
                   session_id=result.get("session_id", ""),
                   n_instances=len(targets),
                   n_abstained=sum(1 for t in targets if t.get("abstained")))
    s.uncertainty = _uncertainty_score(targets, w)
    s.lep_quality = _lep_quality_score(targets, w)
    s.rarity = _rarity_score(targets, class_frequencies)
    s.crop_risk = _crop_risk_score(targets, w)
    s.predicted_classes = dict(Counter(t.get("class_name", "?") for t in targets))
    s.total = float(w.uncertainty * s.uncertainty + w.lep_quality * s.lep_quality
                    + w.rarity * s.rarity + w.crop_risk * s.crop_risk)

    if s.rarity > 0.5:
        rare = [c for c in s.predicted_classes
                if class_frequencies.get(c, 0) < 10]
        if rare:
            s.reasons.append(f"may contain scarce class(es): {', '.join(rare)}")
    if s.uncertainty > 0.3:
        s.reasons.append(f"{s.uncertainty*100:.0f}% of detections are ambiguous")
    if s.lep_quality > 0.5:
        s.reasons.append("growth points are poorly localised here")
    if s.crop_risk > 0.2:
        s.reasons.append("weed candidates sit close to onion tissue")
    if not targets:
        s.reasons.append("model found nothing - either genuinely empty, or a "
                         "total miss worth checking")
    return s


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #
def appearance_descriptor(bgr, bins=8):
    """Small, cheap, illumination-tolerant frame descriptor.

    A joint Hue-Saturation histogram plus a coarse spatial grid of mean
    Excess-Green. It captures "what does this patch of field look like and
    where is the vegetation", which is what should differ between two frames
    worth annotating - and unlike a deep embedding it needs no model, so
    diversity works at cold start."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = (hist / max(1e-6, hist.sum())).ravel()

    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    tot = b + g + r + 1e-6
    exg = 2 * (g / tot) - (r / tot) - (b / tot)
    h, wd = exg.shape
    gy, gx = 4, 4
    grid = np.array([[exg[y * h // gy:(y + 1) * h // gy,
                          x * wd // gx:(x + 1) * wd // gx].mean()
                      for x in range(gx)] for y in range(gy)]).ravel()
    return np.concatenate([hist, grid]).astype(np.float32)


def greedy_diverse(descriptors, k, scores=None, seed=0):
    """Farthest-point selection, optionally seeded by score.

    Pure top-k on an uncertainty score returns near-duplicates of one hard
    patch - if a frame confuses the model, its neighbours confuse it
    identically. Farthest-point spreads the batch across the pool, so each
    annotation hour buys new information rather than the same information
    again.

    Returns indices, most valuable first."""
    d = np.asarray(descriptors, np.float32)
    n = len(d)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    scores = np.zeros(n) if scores is None else np.asarray(scores, float)

    # Start from the highest-scoring frame so the batch is anchored on the most
    # informative example rather than an arbitrary one.
    first = int(np.argmax(scores)) if scores.any() else 0
    chosen = [first]
    dist = np.linalg.norm(d - d[first], axis=1)

    while len(chosen) < k:
        # Balance "far from what we already picked" against "valuable on its
        # own". Normalising keeps the two comparable across datasets.
        dn = dist / max(1e-9, dist.max())
        sn = scores / max(1e-9, scores.max()) if scores.any() else np.zeros(n)
        merit = 0.5 * dn + 0.5 * sn
        merit[chosen] = -np.inf
        nxt = int(np.argmax(merit))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(d - d[nxt], axis=1))
    return chosen


# --------------------------------------------------------------------------- #
# Selection entry points
# --------------------------------------------------------------------------- #
def labelled_class_frequencies(lep_manifest=None, seg_manifest=None):
    """How many instances of each class are already annotated.

    Every ontology class is present in the result, including those at zero -
    a class that never appears is the single most important thing this
    function can report."""
    freq = {c: 0 for c in CLASSES}
    if seg_manifest:
        doc = (json.loads(Path(seg_manifest).read_text(encoding="utf-8"))
               if isinstance(seg_manifest, (str, Path)) else seg_manifest)
        for f in doc.get("frames", []):
            for i in f.get("instances", []):
                name = i.get("class_name")
                if name in freq:
                    freq[name] += 1
    elif lep_manifest:
        doc = (json.loads(Path(lep_manifest).read_text(encoding="utf-8"))
               if isinstance(lep_manifest, (str, Path)) else lep_manifest)
        for r in doc.get("rows", []):
            name = r.get("class_name")
            if name in freq:
                freq[name] += 1
    return freq


def select_next_batch(results, class_frequencies, k=30, weights=None,
                      descriptors=None):
    """Rank stored inference results and choose the next annotation batch.

    results: [FrameResult.to_dict()] over the UNLABELLED pool.
    descriptors: optional {frame_id: vector} for the diversity pass. Without
    them the selection is top-k by score, which risks near-duplicates - the
    function says so in the returned report rather than hiding it.
    """
    w = weights or ScoringWeights()
    scored = [score_frame(r, class_frequencies, w) for r in results]
    order = np.argsort([-s.total for s in scored])

    note = None
    if descriptors:
        vecs, keep = [], []
        for i, s in enumerate(scored):
            v = descriptors.get(s.frame_id)
            if v is not None:
                vecs.append(np.asarray(v, np.float32))
                keep.append(i)
        if len(vecs) >= 2:
            idx = greedy_diverse(vecs, k, [scored[i].total for i in keep])
            chosen = [scored[keep[i]] for i in idx]
        else:
            chosen = [scored[i] for i in order[:k]]
            note = "too few descriptors for a diversity pass"
    else:
        chosen = [scored[i] for i in order[:k]]
        note = ("no appearance descriptors supplied, so this is top-k by score. "
                "Expect near-duplicate frames; pass descriptors to spread the "
                "batch.")

    missing = [c for c, n in class_frequencies.items() if n == 0]
    scarce = [c for c, n in class_frequencies.items() if 0 < n < 10]
    return {
        "selected": [s.to_dict() for s in chosen],
        "n_pool": len(results),
        "n_selected": len(chosen),
        "class_frequencies": dict(class_frequencies),
        "classes_with_no_examples": missing,
        "scarce_classes": scarce,
        "note": note,
        "guidance": _guidance(missing, scarce, chosen),
    }


def select_cold_start(descriptors, k=30):
    """First round: no model exists, so choose on appearance diversity alone.

    There is nothing to be uncertain about before a model is trained, and a
    diverse first batch is what makes the FIRST model good enough for
    uncertainty sampling to mean anything afterwards."""
    ids = list(descriptors)
    if not ids:
        return {"selected": [], "n_selected": 0,
                "note": "no descriptors supplied"}
    vecs = [np.asarray(descriptors[i], np.float32) for i in ids]
    idx = greedy_diverse(vecs, k)
    return {"selected": [{"frame_id": ids[i]} for i in idx],
            "n_pool": len(ids), "n_selected": len(idx),
            "note": "cold start: appearance diversity only, no model used"}


def _guidance(missing, scarce, chosen):
    g = []
    if missing:
        g.append(f"{len(missing)} class(es) have ZERO labelled instances "
                 f"({', '.join(missing)}). A class with no examples can never be "
                 f"predicted. Either find and annotate frames containing them, "
                 f"or drop them from the ontology for now and re-add later - "
                 f"training on a class that does not exist wastes capacity and "
                 f"makes the metrics misleading.")
    if scarce:
        g.append(f"scarce class(es): {', '.join(scarce)}. Prioritise frames "
                 f"predicted to contain them; a handful of examples is not "
                 f"enough to learn a class but is enough to make the model "
                 f"overconfident about it.")
    empty = sum(1 for s in chosen if s.n_instances == 0)
    if empty:
        g.append(f"{empty} selected frame(s) had NO detections. Check whether "
                 f"they are genuinely empty ground or a total model miss - the "
                 f"second is far more valuable to annotate.")
    return g
