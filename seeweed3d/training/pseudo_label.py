#!/usr/bin/env python3
"""
SeeWeed3D - self-training: which predictions are safe to train on.

Scores every frame the model predicted on, splits them into frames whose
predictions can be trusted as pseudo-labels and frames that need a human, and
writes both. Round after round.

    python -m seeweed3d.training.datasets.weeds_selftrain

THE FAILURE THIS IS DESIGNED AGAINST
------------------------------------
Keeping the frames the model scored HIGHEST and training on them is
self-confirmation. Where the model is already right the loss is near zero, so
there is no gradient and nothing is learned; all that changes is that the model
grows more confident about what it already knew, while its blind spots stay
exactly as blind. Its confident errors, meanwhile, become training truth and
compound.

Confidence is the wrong instrument for this specifically because it is the
MODEL'S OWN OPINION. A model is most confident where the data looks like its
training set, which is precisely where a new frame teaches least - and softmax
confidence is badly calibrated off-distribution, so a confident wrong mask on
unfamiliar ground looks exactly like a confident right one.

WHAT MAKES THIS WORK INSTEAD: AN INDEPENDENT WITNESS
-----------------------------------------------------
This project has one, and it is free. The ExG VEGETATION PRIOR is a per-pixel
colour computation that knows nothing about the model, was not trained, and
cannot be talked into agreeing. So:

    "does this mask sit on something green"        is corroboration
    "is the model confident about this mask"       is not

Two directions, and they catch opposite failures:

  veg_precision   how much of the predicted mask is on vegetation.
                  Low = the model is claiming soil.
  veg_recall      how much of the frame's vegetation the predictions cover.
                  Low = the model is MISSING plants - the failure an overlay
                  cannot show you, because a plant with nothing drawn on it
                  looks the same as bare ground.

`veg_recall` is the reason this is not just a confidence filter with extra
steps. A frame where the model found three obvious weeds and silently skipped
nine is a frame where every detection is correct and the pseudo-label is
catastrophic: the nine become BACKGROUND in the training target, and the model
is taught that plants like those are soil.

TWO OUTPUTS, NOT ONE
--------------------
The same pass produces both halves of the loop, which is what makes the tension
above resolvable rather than a dilemma:

    accept/   high score. Pseudo-labels. Cheap, safe, and they teach little
              individually - their value is volume and coverage.
    review/   low score. The frames a human should spend time on, because these
              are where the model is wrong and where the gradient is.

Annotating `review` is what moves the model. Adding `accept` is what stops it
forgetting, and costs nothing. Running only one of the two is the mistake.

THE GUARDRAILS, AND WHY EACH IS NOT OPTIONAL
--------------------------------------------
  * PSEUDO-LABELS ARE NEVER CALLED HAND-CORRECTED. They travel with
    provenance "pseudo_label", and a build containing them reports "mixed".
    A dataset that cannot say which labels came from a model cannot be audited
    later, and this project has already been bitten by unreviewed labels being
    read as verified.
  * A CAP ON THE PSEUDO FRACTION. Past roughly 2:1 the model is mostly training
    on itself and the hand-corrected signal is outvoted.
  * HOLDOUT SESSIONS ARE NEVER PSEUDO-LABELLED. Pseudo-labelling a test session
    puts the model's own output into its own test set, and every later round
    then scores against what it already believes.
  * REGENERATED EACH ROUND from the newest checkpoint, never accumulated. A
    stale pseudo-label is a mistake the current model would no longer make,
    kept alive as ground truth.
  * A SPOT-CHECK SAMPLE, always. Some accepted frames go to a folder for a human
    to glance at. Ten frames per round is minutes of work and it is the only
    thing standing between a bad threshold and a poisoned dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.vegetation import vegetation_mask, white_balance  # noqa: E402

#: Frames scoring at or above this are eligible to become pseudo-labels.
#: Calibrate it on YOUR data with `--report`, which prints the distribution and
#: what each cut would accept, rather than adopting this number on faith.
ACCEPT_SCORE = 0.70

#: Below this a frame goes to `review/` - the model is wrong here, so this is
#: where a human's hour buys the most. Frames between the two are neither
#: trustworthy enough to train on nor interesting enough to annotate.
REVIEW_SCORE = 0.45

#: HARD GATES. A frame failing any of these is rejected outright regardless of
#: its score: these are not "lower quality", they are "wrong in a way that would
#: teach the model something false".
GATES = {
    # Predictions claiming mostly soil.
    "veg_precision": 0.55,
    # The dangerous one. Vegetation the model did not claim becomes BACKGROUND
    # in the pseudo-label, so a frame full of missed plants actively teaches
    # that plants are soil. Deliberately strict.
    "veg_recall": 0.80,
}

#: At most this many pseudo-label frames per hand-corrected frame. Past roughly
#: 2:1 the model is mostly training on itself.
MAX_PSEUDO_RATIO = 2.0

#: Accepted frames set aside for a human to glance at, every round.
SPOT_CHECK = 10

#: Weights for the ranking score. Confidence is present and DELIBERATELY the
#: smallest term: it is the model's own opinion, and the whole point of the
#: vegetation terms is that they are not.
WEIGHTS = {
    "veg_precision": 0.35,
    "veg_recall": 0.35,
    "stability": 0.20,
    "confidence": 0.10,
}


def frame_quality(bgr, masks, scores, n_suppressed=0, veg=None, cfg=None):
    """Per-frame pseudo-label quality. Returns a dict of components + `score`.

    `n_suppressed` is how many duplicate detections were removed for this
    frame - a set-prediction model emitting many near-identical masks is
    unstable on that frame, which is a model-side signal that costs nothing to
    collect and does not depend on the model being right.
    """
    c = dict(cfg or {})
    if veg is None:
        proc = white_balance(bgr, 1.15) if c.get("WHITE_BALANCE", True) else bgr
        veg = vegetation_mask(proc,
                              c.get("EXG_THRESHOLD", 0.05),
                              c.get("VEG_MIN_SATURATION", 40),
                              c.get("VEG_MORPH_KERNEL", 3),
                              c.get("VEG_MIN_COMPONENT_PX", 150))

    union = np.zeros(veg.shape, bool)
    for m in masks:
        m = np.asarray(m, bool)
        if m.shape == union.shape:
            union |= m

    pred_px = int(union.sum())
    veg_px = int(veg.sum())
    both = int((union & veg).sum())

    # An empty frame is not a good frame. With no predictions veg_precision is
    # undefined and would default to 1.0 on a vacuous truth, which would rank a
    # frame the model found NOTHING in above every real one - and its
    # pseudo-label would declare the whole frame background.
    veg_precision = (both / pred_px) if pred_px else 0.0
    veg_recall = (both / veg_px) if veg_px else (1.0 if not pred_px else 0.0)

    n = len(scores)
    confidence = float(np.mean(scores)) if n else 0.0
    stability = (n / (n + n_suppressed)) if (n + n_suppressed) else 0.0

    parts = {"veg_precision": veg_precision, "veg_recall": veg_recall,
             "stability": stability, "confidence": confidence}
    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)

    failed = sorted(k for k, floor in GATES.items() if parts[k] < floor)
    # A frame with no vegetation at all is bare ground: nothing to learn and
    # nothing to get wrong, so it is neither accepted nor sent for review.
    return {**parts, "score": round(score, 4), "n_instances": n,
            "pred_px": pred_px, "veg_px": veg_px,
            "gates_failed": failed, "empty": veg_px == 0}


def classify(quality, accept=ACCEPT_SCORE, review=REVIEW_SCORE):
    """"accept" | "review" | "skip" for one frame's quality dict.

    A gate failure sends a frame to REVIEW, never to skip: a frame the model got
    badly wrong is the single most valuable thing a human can annotate, and
    silently discarding it would throw away the loop's best signal."""
    if quality.get("empty"):
        return "skip"
    if quality["gates_failed"] or quality["score"] < review:
        return "review"
    if quality["score"] >= accept:
        return "accept"
    return "skip"


def pseudo_budget(n_hand, n_pseudo_existing=0, ratio=MAX_PSEUDO_RATIO):
    """How many NEW pseudo-label frames may be added this round.

    Counted against hand-corrected frames rather than against the dataset, so a
    dataset that is already mostly pseudo-labelled cannot bootstrap itself into
    accepting more."""
    return max(0, int(ratio * int(n_hand)) - int(n_pseudo_existing))


def select_spread(order, chosen, score_of, min_gap):
    """Keep the BEST frame in each neighbourhood, drop its near-copies.

    Separation and sampling are two different decisions, and using one stride
    for both gets the worse half of each. A stride of 60 never looks at 59 out
    of every 60 frames, so it cannot know that frame 31 was the good one; a
    stride of 1 looks at everything and then hands the training set twelve
    copies of the same plant, each carrying the same error and each weighted.

    So: infer over everything - it is cheap and the overlays are worth having -
    and separate HERE, choosing by score rather than by position. Same number of
    distinct frames, better frames.

    `order` is every scored frame in capture order; `chosen` the subset under
    consideration. `min_gap` is in positions within `order`, so a caller working
    at stride N converts before calling. 0 disables it.

    Highest score first, not first-come: the whole point is that the pick within
    a neighbourhood is the good one."""
    if min_gap <= 0:
        return list(chosen), []
    pos = {f: i for i, f in enumerate(order)}
    cand = sorted((f for f in chosen if f in pos), key=lambda f: pos[f])
    if not cand:
        return [], []

    # WINDOWS, not plain greedy-by-score. Taking the highest-scoring frame
    # first and suppressing its neighbours is the obvious version and it costs
    # yield: one early pick sitting a third of the way into its window pushes
    # the next one past the following window's start, and a drive that could
    # give four frames gives three. That trade would be worth making if the
    # score separated frames - it does not. The observed spread on real data was
    # p10 0.83 to p90 0.91, so sacrificing a whole frame buys a hundredth of a
    # point. Walk the drive in order instead, take the best frame in each
    # window that clears the last pick, and the count is as high as the gap
    # allows while the choice within a neighbourhood is still made on score.
    kept, last = [], None
    for start in range(0, pos[cand[-1]] + 1, min_gap):
        window = [f for f in cand
                  if start <= pos[f] < start + min_gap
                  and (last is None or pos[f] - last >= min_gap)]
        if not window:
            continue
        best = max(window, key=lambda f: (float(score_of(f)), -pos[f]))
        kept.append(best)
        last = pos[best]
    keptset = set(kept)
    return kept, [f for f in cand if f not in keptset]


def separation_note(n_before, n_after, min_gap, stride=1):
    """What the separation step did, in one line, always printed.

    A filter that only speaks when it is unhappy leaves you unable to tell
    "nothing was redundant" from "the filter did not run"."""
    if min_gap <= 0:
        return ("  [!] separation is OFF (MIN_FRAME_GAP = 0). Near-identical "
                "frames will each be weighted\n      in the training set, "
                "carrying the same error once per copy.")
    dropped = n_before - n_after
    return (f"  separation: kept {n_after} of {n_before} frame(s), "
            f"{dropped} dropped as near-copies\n"
            f"              (at least {min_gap} video frame(s) apart; "
            f"the best-scoring frame in each\n"
            f"              neighbourhood is the one kept, not the first)")


#: No single class may take more than this share of an accepted batch. Named
#: rather than defaulted inline so the runner can print the number it enforced:
#: this cap has silently removed a third of a small batch.
BALANCE_CAP_FRAC = 0.6


def balance_by_class(chosen, class_of, cap_frac=BALANCE_CAP_FRAC):
    """Trim a selection so no single class takes more than `cap_frac` of it.

    Without this the loop concentrates: the model predicts its dominant class
    most confidently, those frames score highest, they are added, the class
    grows more dominant. Three rounds of that and the rare classes are noise."""
    if not chosen:
        return []
    cap = max(1, int(cap_frac * len(chosen)))
    kept, per = [], {}
    for f in chosen:
        c = class_of.get(f, "")
        if per.get(c, 0) >= cap:
            continue
        per[c] = per.get(c, 0) + 1
        kept.append(f)
    return kept


def summarise(qualities, accept=ACCEPT_SCORE, review=REVIEW_SCORE):
    """Counts, the score distribution, and what other cuts would have accepted.

    The sweep is the point. A single threshold is a claim about a threshold, not
    about the data, and this project has already had one number move from 0.28
    to 0.73 on unchanged weights by changing one."""
    scores = sorted(q["score"] for q in qualities)
    buckets = {"accept": 0, "review": 0, "skip": 0}
    for q in qualities:
        buckets[classify(q, accept, review)] += 1

    # EVERY gate, seeded at zero. A gate that never fires used to be absent
    # from this dict, which reads as "not a problem" and is indistinguishable
    # from "not being applied". A floor nothing has ever failed is not
    # protecting anything, and that is worth seeing.
    gate_fails = {g: 0 for g in GATES}
    for q in qualities:
        for g in q["gates_failed"]:
            gate_fails[g] = gate_fails.get(g, 0) + 1

    def pct(p):
        return scores[min(len(scores) - 1, int(p * len(scores)))] if scores else 0.0

    sweep = []
    for cut in (0.5, 0.6, 0.7, 0.8, 0.9):
        n = sum(1 for q in qualities
                if not q["gates_failed"] and not q.get("empty")
                and q["score"] >= cut)
        sweep.append((cut, n))

    return {"n_frames": len(qualities), **buckets,
            "gate_failures": gate_fails,
            "score_p10": round(pct(0.10), 4), "score_median": round(pct(0.50), 4),
            "score_p90": round(pct(0.90), 4),
            "accept_sweep": sweep, "flat_sweep": flat_sweep(sweep)}


def flat_sweep(sweep, min_cuts=3):
    """The widest run of cuts that all accept the SAME number, or None.

    When 0.5, 0.6, 0.7 and 0.8 accept 68 frames each, ACCEPT is not selecting
    anything - the hard gates are, and the threshold is decoration. That is
    readable from the sweep, but only if you read four numbers and notice they
    are equal, which nobody does at the end of a run. It matters because the
    obvious response to "86% accepted" is to raise the threshold, and inside a
    flat span raising it changes nothing at all.

    Returns (lo, hi, n) for the widest run of at least `min_cuts` equal counts.
    """
    best = None
    i = 0
    while i < len(sweep):
        j = i
        while j + 1 < len(sweep) and sweep[j + 1][1] == sweep[i][1]:
            j += 1
        run = j - i + 1
        if run >= min_cuts and sweep[i][1] > 0:
            if best is None or run > best[3]:
                best = (sweep[i][0], sweep[j][0], sweep[i][1], run)
        i = j + 1
    return best[:3] if best else None


def format_report(summary, n_hand=None, budget=None):
    """The readout. States what the numbers cannot tell you, next to them."""
    L = [f"\n  {summary['n_frames']} frame(s) scored",
         f"    accept {summary['accept']:>5}   review {summary['review']:>5}"
         f"   skip {summary['skip']:>5}"]
    if summary["gate_failures"]:
        L.append("    hard-gate failures (these go to review, never to accept):")
        for g, n in sorted(summary["gate_failures"].items(),
                           key=lambda kv: -kv[1]):
            why = {"veg_recall": "predictions miss vegetation - those plants "
                                 "would become BACKGROUND in the label",
                   "veg_precision": "predictions sit on soil"}.get(g, "")
            if n == 0:
                why = "never fired - this floor is not filtering anything"
            L.append(f"      {g:<16}{n:>5}   {why}")
    L.append(f"    score  p10 {summary['score_p10']:.3f}  "
             f"median {summary['score_median']:.3f}  "
             f"p90 {summary['score_p90']:.3f}")
    L.append("    frames accepted at each cut (choose from this, not from the "
             "default):")
    for cut, n in summary["accept_sweep"]:
        L.append(f"      >= {cut:.2f}  {n:>5}")
    flat = summary.get("flat_sweep")
    if flat:
        lo, hi, n = flat
        L += [
            f"    [!] THE THRESHOLD IS NOT SELECTING. Every cut from {lo:.2f} "
            f"to {hi:.2f} accepts the same {n} frame(s),",
            f"        so what decides accept-vs-review here is the hard gates, "
            f"not ACCEPT. Raising ACCEPT",
            f"        anywhere inside that span changes nothing. Judge the "
            f"batch by the gates and by the",
            f"        spot_check overlays; a number that does not move is not "
            f"a filter you can tune.",
        ]
    if n_hand is not None:
        L.append(f"    budget: {budget} new pseudo-label frame(s) allowed "
                 f"against {n_hand} hand-corrected")
    L += [
        "",
        "  The vegetation terms are the load-bearing ones: ExG knows nothing",
        "  about the model and cannot be talked into agreeing. Confidence is",
        "  weighted lowest on purpose - it is the model's own opinion, and a",
        "  model is most confident where a frame teaches it least.",
        "",
        "  ANNOTATE THE REVIEW FRAMES. Accepted frames stop the model",
        "  forgetting; review frames are the only ones that move it.",
    ]
    return "\n".join(L)
