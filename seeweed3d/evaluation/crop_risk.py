#!/usr/bin/env python3
"""
SeeWeed3D - how often does the weed model call an ONION a weed?

    python -m seeweed3d.evaluation.crop_risk

THE QUESTION, AND WHY IT IS ANSWERABLE TODAY
--------------------------------------------
A weed-only model has never seen an onion and has no class for one. That is not
the same as being safe around onions: nothing stopped it learning that long
green tissue is a grass weed, and an onion leaf is long green tissue.

Deployed, every such detection is a laser aimed at the crop. It is also
currently unmeasurable by any other route - there is no mixed test set, and the
frame-level scores on weed-only data cannot see it at all, because those frames
contain no onions to get wrong.

But the onion-only sessions solve it for free. Every plant in an onion drive is
an onion BY HOW THE RECORDING WAS MADE, not by anyone's judgement, so their
annotations are a large correctly-classed test set that costs nothing.

WHY FULL FRAMES AND NOT INSTANCE CROPS
--------------------------------------
Running a detector on tight crops changes the regime it was trained for: a crop
normalises the plant to fill the frame, while at inference a seedling is 30 px
in a 2208 px one. A model that behaved well on crops would tell you nothing
about how it behaves in a field. So the model sees whole frames, exactly as
deployed, and the onion masks are used only to decide what its output landed on.

THE TWO NUMBERS, AND WHY BOTH
-----------------------------
  detection-side   of what the model claimed, how much sits on crop tissue.
                   Answers "how wrong is its output".
  onion-side       of the onions present, how many have a weed detection on
                   them. Answers "how many of my plants would be shot at",
                   which is the question a grower asks.

They come apart. One detection sprawling across six onions is one bad detection
and six endangered plants; six small false detections on one onion is the
reverse. Reporting either alone hides a real case.

THE SAME RATE MEANS TWO DIFFERENT THINGS
----------------------------------------
Run against a WEED-ONLY model, a high rate is close to the ceiling BY
CONSTRUCTION: with no class for an onion, in a drive where every plant is one,
almost anything the model detects has to land on crop. That proves the
checkpoint is unsafe and it cannot prove anything else - in particular it cannot
separate "onions look like weeds" from "the model detects plants and has nowhere
to put an onion". The verdict says so instead of sending the next month of
annotation after a conclusion the measurement could not support.

Run against a CROP-CAPABLE model the question becomes discriminative, so its own
onion_plant detections are EXCLUDED from the numerator - scoring the model for
correctly calling an onion an onion would rank the fixed model below the broken
one. What is left is a real confusion. That model also gets CROP RECALL, because
an onion it never detects is an onion the safety mask cannot protect, and
`allow_missing_crop_mask = False` makes that the other half of crop safety.

WHAT THIS CANNOT TELL YOU
-------------------------
A detection that lands on NO onion is not automatically wrong. Onion drives
contain weeds too - they were simply never annotated, which is a separate
problem worth its own fix. So off-crop detections are counted and reported but
NOT called false positives: only overlap with annotated crop tissue is
unambiguous evidence of the failure this measures.

And the onion masks are unreviewed SAM prelabels. "Onion tissue" is itself a
machine label here, so a crop hit against a wrong mask is not a crop hit. The
run says so rather than letting the number be read as ground truth.

THE SWEEP GOES UP, NOT DOWN
---------------------------
The report re-scores the same predictions at several thresholds, because "would
this fire" has a different answer at 0.5 than at 0.9 and the fix for a moderate
number is sometimes just a higher bar. It can only sweep UPWARD from the CONF
inference ran at: a detection the model never emitted cannot be counted later.
Lower CONF and re-run to see below it.
"""
from __future__ import annotations

import json
import ntpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from common.ontology import CROP_CLASS  # noqa: E402
from common.run_dirs import newest, stamped  # noqa: E402
from training.datasets.onions import ONION_SESSIONS  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: The onion-only sessions. Imported from the onion build so a session added
#: there reaches this check without a second edit.
SESSIONS = list(ONION_SESSIONS)

#: The model under test. A WEED-ONLY checkpoint is the point: it has no onion
#: class, so every detection it makes on crop tissue is a mistake it cannot
#: even represent, let alone report.
CHECKPOINT = ntpath.join(r"E:\Dataset_Vidalia\runs_1_only_weeds", "weeds_r1",
                         "checkpoint_best_total.pth")

#: A DEPLOYMENT threshold. The question here is "would this fire", not "how
#: does the model fail", so this is 0.5 rather than the 0.25 the scorers use.
#: Lower it to see the near-misses; the report sweeps upward from it either way.
CONF = 0.50

#: Re-score the same predictions at each of these. Only values >= CONF can be
#: reported - inference already discarded everything below it.
SWEEP = [0.50, 0.70, 0.90]

#: 0 = every frame. Onion drives are long and consecutive frames are
#: near-identical, so a stride buys coverage rather than repetition.
LIMIT = 0
STRIDE = 30

#: Fraction of a DETECTION's mask that must sit on annotated onion tissue for
#: it to count as a crop hit.
#:
#: Not 0: a mask clipping one pixel of a neighbouring leaf is a boundary error,
#: not a plant misidentified. Not 0.5 either - a detection that is a THIRD onion
#: is already aiming at the crop, and requiring a majority would report the
#: dangerous partial overlaps as clean.
DETECTION_ON_CROP_MIN = 0.25

#: Fraction of an ONION's mask that must be covered by weed detections before
#: that plant counts as endangered. Lower than the above on purpose: a laser
#: needs to touch only a little of a plant to damage it.
ONION_COVERED_MIN = 0.10

#: Where the report and the worst-case overlays are written.
OUT_DIR = stamped(ntpath.dirname(CHECKPOINT), "crop_risk")

#: How many of the worst frames to save overlays for. Numbers say how often;
#: only a picture says what it is doing.
WORST_FRAMES = 12

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

#: BGR. The crop is orange everywhere in this project (predict_images.py), and
#: a crop hit is red because it is the thing being counted.
C_ONION = (0, 165, 255)
C_HIT = (0, 0, 255)
C_OFF = (170, 170, 170)


def discover_sessions(roots):
    """Every session folder under `roots` that holds frames.

    A root is either a session itself or a folder whose children are sessions -
    the same two shapes the dataset builds accept, so SESSIONS can be pasted
    from onions.py without editing."""
    out = []
    for root in roots:
        p = Path(root)
        if not p.is_dir():
            continue
        cands = [p] if (p / "rgb").is_dir() else [
            d for d in sorted(p.iterdir()) if d.is_dir()]
        for d in cands:
            imgs = d / "rgb" if (d / "rgb").is_dir() else d
            if not imgs.is_dir():
                continue
            if any(f.suffix.lower() in IMAGE_SUFFIXES
                   for f in imgs.iterdir() if f.is_file()):
                out.append(d)
    return out


def load_polygons(path, want_class=None):
    """{file_stem: [(class_name, [polygon, ...], score), ...]} - Datumaro/COCO.

    POLYGONS, not masks. bench_mixed rasterises one full-frame array per
    annotation, which on a real onion campaign - tens of thousands of instances
    at 1242x2208 - is gigabytes held at once; that shape already killed one run.
    Polygons are a few hundred floats and the frame's arrays are built on demand
    and released.

    The score rides along so the report can sweep the threshold without a second
    inference pass. Ground-truth annotations carry none and get 1.0, which is
    what makes them survive every threshold."""
    p = Path(path)
    # Datumaro FIRST and alone if present, rather than both forms pooled. A
    # session often holds the same masks twice - annotations/ plus a cvat_ready
    # COCO of the export - and pooling them counts every plant as two, which
    # halves the reported rate for a reason nothing in the output would show.
    datumaro = sorted(p.is_dir() and p.rglob("annotations/*.json") or [])
    files = ([p] if p.is_file() else
             datumaro or sorted(p.rglob("instances_default.json")))
    out = {}
    for f in files:
        try:
            doc = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc.get("items"), list):
            names = [c.get("name", "?") for c in
                     (doc.get("categories") or {}).get("label", {})
                     .get("labels", [])]
            for item in doc["items"]:
                key = Path(str(item.get("id", ""))).stem
                inst = []
                for a in item.get("annotations", []):
                    if a.get("type") != "polygon":
                        continue
                    pts = a.get("points") or []
                    idx = a.get("label_id", -1)
                    cls = names[idx] if 0 <= idx < len(names) else "?"
                    if len(pts) >= 6 and (want_class is None or cls == want_class):
                        inst.append((cls, [list(pts)],
                                     float(a.get("score", 1.0))))
                if inst:
                    out.setdefault(key, []).extend(inst)
        elif isinstance(doc.get("images"), list):
            cat = {c["id"]: c["name"] for c in doc.get("categories", [])}
            by_id = {im["id"]: Path(im["file_name"]).stem
                     for im in doc.get("images", [])}
            for a in doc.get("annotations", []):
                key = by_id.get(a.get("image_id"))
                cls = cat.get(a.get("category_id"), "?")
                seg = [s for s in (a.get("segmentation") or [])
                       if isinstance(s, list) and len(s) >= 6]
                if key and seg and (want_class is None or cls == want_class):
                    out.setdefault(key, []).append(
                        (cls, seg, float(a.get("score", 1.0))))
    return out


def image_sizes(coco_path):
    """{file_stem: (h, w)} from a COCO file's images[].

    Read from the PREDICTIONS, which always record it, rather than opening the
    frames: the sizes are needed for every frame and the images are 2208 px."""
    try:
        doc = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {Path(im["file_name"]).stem: (int(im["height"]), int(im["width"]))
            for im in doc.get("images", [])
            if im.get("height") and im.get("width")}


def rasterise(polys, h, w):
    """One boolean mask from a list of polygons. Built per frame, then freed."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    for poly in polys:
        pts = np.asarray(poly, np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def frame_risk(onion_insts, pred_insts, h, w, min_score=0.0,
               det_min=DETECTION_ON_CROP_MIN, onion_min=ONION_COVERED_MIN,
               crop_class=CROP_CLASS):
    """One frame's verdict. Pure: takes polygons, returns counts.

    Both directions are computed from the same masks so they cannot disagree
    about a frame, and the per-class tally records WHICH weed class the model
    mistook the crop for - if one dominates, that is an interpretable finding
    rather than a number.

    ONLY WEED-CLASS DETECTIONS COUNT AS CROP HITS. A crop-capable model puts an
    `onion_plant` mask on an onion, which is the model getting it exactly right;
    counting that as risk would score the fixed model as worse than the broken
    one. Those detections are tallied separately as crop RECALL - an onion the
    model never detects is an onion the safety mask cannot protect, which is the
    other half of crop safety and free to measure here."""
    onion_masks = [rasterise(p, h, w) for _, p, _ in onion_insts]
    onion_union = np.zeros((h, w), bool)
    for m in onion_masks:
        onion_union |= m

    kept = [(c, p) for c, p, s in pred_insts if s >= min_score]
    hits_by_class, n_on_crop, n_off_crop, n_weeds, n_crop_det = {}, 0, 0, 0, 0
    covered = np.zeros((h, w), bool)
    crop_covered = np.zeros((h, w), bool)
    on_crop_idx = []
    for i, (cls, polys) in enumerate(kept):
        m = rasterise(polys, h, w)
        area = int(m.sum())
        if not area:
            continue
        if cls == crop_class:
            n_crop_det += 1
            crop_covered |= m
            continue
        n_weeds += 1
        frac = float((m & onion_union).sum()) / area
        if frac >= det_min:
            n_on_crop += 1
            on_crop_idx.append(i)
            hits_by_class[cls] = hits_by_class.get(cls, 0) + 1
            covered |= m
        else:
            n_off_crop += 1

    def _covered_by(mask):
        return sum(1 for m in onion_masks if m.any()
                   and float((m & mask).sum()) / float(m.sum()) >= onion_min)

    return {"n_detections": n_weeds,
            "n_on_crop": n_on_crop, "n_off_crop": n_off_crop,
            "n_onions": len(onion_masks),
            "n_endangered": _covered_by(covered),
            "n_crop_detections": n_crop_det,
            "n_onions_detected": _covered_by(crop_covered),
            "hits_by_class": hits_by_class, "on_crop_idx": on_crop_idx}


def summarise(per_frame):
    """Totals plus the rates, kept apart because they answer differently."""
    keys = ("n_detections", "n_on_crop", "n_off_crop", "n_onions",
            "n_endangered", "n_crop_detections", "n_onions_detected")
    tot = dict({k: 0 for k in keys}, frames=len(per_frame), hits_by_class={})
    for r in per_frame.values():
        for k in keys:
            tot[k] += r.get(k, 0)
        for c, n in r["hits_by_class"].items():
            tot["hits_by_class"][c] = tot["hits_by_class"].get(c, 0) + n
    tot["detection_on_crop_rate"] = (
        tot["n_on_crop"] / tot["n_detections"] if tot["n_detections"] else 0.0)
    tot["onion_endangered_rate"] = (
        tot["n_endangered"] / tot["n_onions"] if tot["n_onions"] else 0.0)
    # Only meaningful for a model that HAS the crop class; the report shows it
    # only then, because 0% from a weed-only model is a tautology, not a score.
    tot["crop_recall"] = (
        tot["n_onions_detected"] / tot["n_onions"] if tot["n_onions"] else 0.0)
    return tot


def score_frames(frames, min_score=0.0):
    """{key: frame_risk(...)} over `frames` = {key: (onions, preds, h, w)}.

    Masks are rebuilt per threshold rather than cached. Caching them would be
    one full-frame boolean array per instance held across the whole sweep,
    which is the exact shape that has already killed a run here."""
    return {k: frame_risk(on, pr, h, w, min_score=min_score)
            for k, (on, pr, h, w) in frames.items()}


def sweep(frames, thresholds=SWEEP, conf=CONF):
    """The two rates at each threshold at or above the inference CONF."""
    return [dict(summarise(score_frames(frames, t)), threshold=t)
            for t in sorted(t for t in thresholds if t >= conf)]


def verdict(s, model_classes=None):
    """What the number means for the next decision, not just what it is.

    THE SAME RATE MEANS TWO DIFFERENT THINGS depending on the model, and reading
    it the wrong way sends the next month of annotation in the wrong direction.

    A WEED-ONLY model has no class for an onion, so in a drive where every plant
    IS an onion, anything its objectness fires on necessarily lands on crop. A
    high rate there is close to the ceiling by construction: it proves the
    checkpoint is unsafe, but it CANNOT separate "onions look like weeds" from
    "the model detects plants and has nowhere to put an onion". Only a
    crop-capable model can be asked the discriminative question, so the honest
    conclusion is to train one and re-run rather than to buy contact frames on
    the strength of a number that could not have come out low.

    A CROP-CAPABLE model's own onion detections are excluded from the numerator,
    so what remains is a genuine confusion and the bands mean what they say."""
    r = s["onion_endangered_rate"]
    if s["n_onions"] == 0:
        return ("No annotated onions were found in these frames, so nothing "
                "was measured. Check SESSIONS points at the onion campaign.")

    if not (model_classes and CROP_CLASS in model_classes):
        if r < 0.02:
            return ("LOW, and informative. Even with no class for an onion this "
                    "model mostly leaves crop tissue alone, so its learnt weed "
                    "appearance does not cover onions. Adding the crop class "
                    "should be enough - build mixed.py and retrain.")
        return (f"UNSAFE, AND EXPECTED. {r:.0%} of onions carry a weed "
                f"detection - but this model has no class for an onion, so in a "
                f"drive where every plant is one, almost anything it detects "
                f"must land on crop. It confirms the weed-only checkpoint can "
                f"never drive a laser; it CANNOT tell you whether a crop class "
                f"fixes it, because a model with no onion class had no way to "
                f"be right here. Train the mixed model and re-run - this number "
                f"only becomes diagnostic then.")

    if r < 0.02:
        return ("LOW. The model rarely puts a WEED class on crop tissue. Check "
                "crop recall below too: an onion it never detects at all is one "
                "the safety mask cannot protect.")
    if r < 0.15:
        return ("MODERATE. Some crop tissue reads as weed even with an onion "
                "class available. The next annotation should buy contact frames "
                "- onions and weeds touching - rather than more onion-only "
                "drives.")
    return ("HIGH. Onion tissue is genuinely confusable with weeds at this "
            "resolution: the model can say onion_plant and chose a weed class "
            "anyway. Spend the next annotation on MIXED frames with real "
            "contact, and do not trust any crop-safety number until it drops.")


def hits_note(hits_by_class, model_classes=None, dominant=0.8):
    """What the class tally does and does not say about appearance.

    The tempting reading is "onions look like X". For a model with no onion
    class that is usually wrong: it has to emit SOME weed label, so a single
    class taking almost every hit is its class PRIOR showing through, not a
    resemblance. Saying so here stops a training-set imbalance being mistaken
    for a fact about how onions look."""
    if not hits_by_class:
        return []
    total = sum(hits_by_class.values())
    top = max(hits_by_class, key=hits_by_class.get)
    share = hits_by_class[top] / total if total else 0.0
    crop_aware = bool(model_classes) and CROP_CLASS in model_classes
    out = []
    if share >= dominant and not crop_aware:
        out += [f"    ({top} takes {share:.0%} of the crop hits, but this model "
                f"has no onion class",
                f"     and must emit SOME weed label - so that is most likely "
                f"its majority class",
                f"     showing through, not a claim that onions resemble "
                f"{top}. Check the training",
                f"     set's class balance before reading it as appearance.)"]
    elif top == "grass_weed":
        out += ["    (grass_weed dominating is the expected shape: an onion "
                "leaf is long, thin",
                "     and linear, and so is a grass weed. It is a finding "
                "about appearance,",
                "     not a random error.)"]
    return out


def sweep_note(rows):
    """What the threshold sweep says that the table alone does not.

    Two findings hide in these rows and both change what to do next, so they
    are derived rather than left for someone to spot in four numbers."""
    out = []
    live = [r for r in rows if r["n_detections"]]
    if not live:
        return out
    dead = [r for r in rows if not r["n_detections"]]
    if dead:
        top = min(r["threshold"] for r in dead)
        out.append(
            f"    NOTE: nothing at all survives {top:.2f}. The model's scores "
            f"never reach it, so a gate\n"
            f"    that high would switch the machine off rather than make it "
            f"safe.")
    lo, hi = live[0], live[-1]
    if hi["detection_on_crop_rate"] > lo["detection_on_crop_rate"] + 1e-9:
        out.append(
            f"    NOTE: the on-crop share RISES with the threshold "
            f"({lo['detection_on_crop_rate']:.1%} -> "
            f"{hi['detection_on_crop_rate']:.1%}).\n"
            f"    The model is MORE confident on crop tissue than on the "
            f"off-crop plants, so raising\n"
            f"    the bar removes real weeds faster than it removes crop hits. "
            f"Confidence is not a fix.")
    return out


def provenance_warnings(model_classes, skipped_no_onion=0, skipped_no_pred=0,
                        sessions_skipped=None):
    """Everything that makes a number here weaker than it looks.

    Printed with the result rather than in a docstring nobody re-reads, because
    this run produces a single percentage that is very easy to quote out of
    context six weeks from now."""
    out = ["  [i] the onion masks are UNREVIEWED SAM prelabels. A crop hit "
           "against a wrong",
           "      mask is not a crop hit - read this rate as an estimate, not "
           "as ground truth."]
    if model_classes and CROP_CLASS in model_classes:
        out.append(
            f"  [i] this checkpoint CAN predict {CROP_CLASS}. Its own crop "
            f"detections are excluded\n"
            f"      from the rates, so what is left is a genuine confusion - "
            f"and CROP RECALL is\n"
            f"      reported, because an onion it never detects is one the "
            f"safety mask cannot protect.")
    else:
        out.append(
            "  [!] this checkpoint has NO crop class, so in a drive where "
            "every plant is an\n"
            "      onion, nearly anything it detects must land on crop. A high "
            "rate here proves\n"
            "      the model is unsafe; it does NOT prove onions and weeds are "
            "confusable. Only a\n"
            "      crop-capable model can be asked that.")
    if sessions_skipped:
        out.append(
            f"  [i] {len(sessions_skipped)} session(s) contributed nothing:")
        out.extend(f"        {name}: {why}"
                   for name, why in sorted(sessions_skipped.items()))
    if skipped_no_onion:
        out.append(
            f"  [i] {skipped_no_onion} predicted frame(s) had no annotated "
            f"onion and were skipped.\n"
            f"      Counting them would dilute the detection-side rate with "
            f"frames that cannot\n"
            f"      contribute a crop hit.")
    if skipped_no_pred:
        out.append(
            f"  [i] {skipped_no_pred} annotated frame(s) were not predicted "
            f"(STRIDE={STRIDE}).")
    return out


def format_report(s, checkpoint=None, conf=None, sweep_rows=(), warnings=(),
                  per_session=None, model_classes=None):
    crop_aware = bool(model_classes) and CROP_CLASS in model_classes
    L = ["", "  Crop risk - does the weed model claim onion tissue?",
         "  " + "-" * 52]
    if checkpoint:
        L.append(f"  model {checkpoint}")
    if conf is not None:
        L.append(f"  conf  {conf}")
    L += ["",
          f"  {s['frames']} frame(s), {s['n_onions']} annotated onion(s)",
          "",
          f"  DETECTION-SIDE  {s['n_on_crop']} of {s['n_detections']} "
          f"weed detection(s) sit on crop tissue "
          f"({s['detection_on_crop_rate']:.1%})",
          f"  ONION-SIDE      {s['n_endangered']} of {s['n_onions']} onion(s) "
          f"have a weed detection on them "
          f"({s['onion_endangered_rate']:.1%})"]
    if crop_aware:
        L += [f"  CROP RECALL     {s['n_onions_detected']} of {s['n_onions']} "
              f"onion(s) were detected AS {CROP_CLASS} "
              f"({s['crop_recall']:.1%})",
              f"                  an onion the model never detects is one the "
              f"safety mask cannot protect.",
              f"                  Its {s['n_crop_detections']} "
              f"{CROP_CLASS} detection(s) are excluded from the rates above - "
              f"those are",
              f"                  the model getting it right, not risk."]
    L += ["",
          f"  {s['n_off_crop']} weed detection(s) landed off-crop. NOT counted "
          f"as errors: onion drives",
          f"  contain real weeds that were never annotated, so those are "
          f"unknown, not wrong."]
    if s["hits_by_class"]:
        L += ["", "  Which class the crop was mistaken for:"]
        for c, n in sorted(s["hits_by_class"].items(), key=lambda kv: -kv[1]):
            L.append(f"    {c:<28}{n:>6}")
        L += hits_note(s["hits_by_class"], model_classes)
    if per_session:
        L += ["", "  Per session (one bad drive can carry the pooled rate):",
              f"    {'session':<40}{'onions':>8}{'endangered':>12}{'rate':>8}"]
        for name, ss in per_session.items():
            L.append(f"    {name[:39]:<40}{ss['n_onions']:>8}"
                     f"{ss['n_endangered']:>12}"
                     f"{ss['onion_endangered_rate']:>7.1%}")
    if len(sweep_rows) > 1:
        L += ["", "  At a higher bar (same predictions, re-scored):",
              f"    {'conf':>6}{'on-crop':>10}{'endangered':>13}"
              f"{'detections':>13}"]
        for row in sweep_rows:
            L.append(f"    {row['threshold']:>6.2f}"
                     f"{row['detection_on_crop_rate']:>9.1%}"
                     f"{row['onion_endangered_rate']:>12.1%}"
                     f"{row['n_detections']:>13}")
        notes = sweep_note(sweep_rows)
        if notes:
            L += [""] + notes
    if warnings:
        L += [""] + list(warnings)
    L += ["", f"  VERDICT: {verdict(s, model_classes)}", ""]
    return "\n".join(L)


def draw_overlay(bgr, onion_insts, pred_insts, on_crop_idx, min_score=0.0):
    """The crop in orange, the detections that claimed it in red.

    The off-crop detections are drawn too, in grey: without them the picture
    would suggest the model found nothing else, and the whole point of the
    off-crop caveat is that those detections exist and are not being judged."""
    import cv2
    out = bgr.copy()
    for _, polys, _ in onion_insts:
        for poly in polys:
            pts = np.round(np.asarray(poly, np.float64).reshape(-1, 2)
                           ).astype(np.int32)
            if len(pts) >= 3:
                cv2.polylines(out, [pts], True, C_ONION, 3)

    kept = [(c, p) for c, p, s in pred_insts if s >= min_score]
    hit = set(on_crop_idx)
    fill = out.copy()
    for i, (cls, polys) in enumerate(kept):
        # A crop-class detection is drawn in the crop's own colour: the model
        # agreeing with the annotation should not look like the failure being
        # counted.
        colour = (C_ONION if cls == CROP_CLASS else
                  C_HIT if i in hit else C_OFF)
        for poly in polys:
            pts = np.round(np.asarray(poly, np.float64).reshape(-1, 2)
                           ).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(fill, [pts], colour)
                cv2.polylines(out, [pts], True, colour, 2)
        if i in hit and polys:
            pts = np.asarray(polys[0], np.float64).reshape(-1, 2)
            org = (int(pts[:, 0].min()), max(18, int(pts[:, 1].min()) - 6))
            cv2.putText(out, cls, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        C_HIT, 2, cv2.LINE_AA)
    return cv2.addWeighted(fill, 0.35, out, 0.65, 0.0)


def worst(per_frame, n=WORST_FRAMES):
    """The frames to look at: most endangered plants, then most crop hits."""
    ranked = sorted((k for k, r in per_frame.items() if r["n_on_crop"]),
                    key=lambda k: (-per_frame[k]["n_endangered"],
                                   -per_frame[k]["n_on_crop"], k))
    return ranked[:n]


def _find_frame(stem, roots):
    """The image file for a frame stem, searched across the plausible roots."""
    for root in roots:
        d = Path(root)
        if not d.is_dir():
            continue
        for suf in (".png", ".jpg", ".jpeg"):
            p = d / f"{stem}{suf}"
            if p.is_file():
                return p
    return None


def _predict(session_dir, pred_root, checkpoint):
    """Predictions for one session, reused if a folder already has them.

    Same reuse rule as the self-training round: the GPU pass is the expensive
    part and re-scoring at a different threshold must not need it."""
    from common.run_dirs import stale_predictions_warning

    stem = f"crop_look_{Path(session_dir).name}"
    pred_dir = Path(newest(pred_root, stem) or stamped(pred_root, stem))
    coco = pred_dir / "instances_default.json"
    if coco.exists():
        print(f"  reusing predictions in {pred_dir}")
        warn = stale_predictions_warning(pred_dir, checkpoint)
        if warn:
            print(warn)
        return pred_dir

    if not Path(checkpoint).exists():
        raise SystemExit(f"ERROR: no checkpoint at {checkpoint}.")
    print(f"  no predictions yet - running inference over {session_dir}")
    from perception.predict_images import CONFIG as PBASE, predict
    predict(dict(PBASE, IMAGES=str(session_dir), CHECKPOINT=checkpoint,
                 BACKEND="rfdetr", DEVICE="cuda", MODE="segmentation",
                 OUT_DIR=str(pred_dir), CONF=CONF, LIMIT=LIMIT, STRIDE=STRIDE,
                 OVERLAY_SCALE=0.5, WRITE_COCO=True))
    if not coco.exists():
        raise SystemExit(f"ERROR: inference wrote no {coco}.")
    return pred_dir


def main():
    import cv2
    from perception.preflight import checkpoint_classes

    sessions = discover_sessions(SESSIONS)
    if not sessions:
        raise SystemExit(
            "ERROR: no session with frames under any of:\n  " +
            "\n  ".join(str(s) for s in SESSIONS))

    model_classes = checkpoint_classes(CHECKPOINT)
    pred_root = ntpath.dirname(CHECKPOINT)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    frames, roots_by_key = {}, {}
    per_session_frames, skipped_no_onion, skipped_no_pred = {}, 0, 0
    # WHY a session contributed nothing, recorded rather than only printed. A
    # run that measures 5 of 14 drives and a run that found only 5 look the
    # same in the JSON otherwise, and they mean opposite things.
    sessions_skipped = {}

    for sess in sessions:
        print(f"\n  {sess.name}")
        onions = load_polygons(sess, want_class=CROP_CLASS)
        if not onions:
            print(f"    no {CROP_CLASS} annotations - skipped")
            sessions_skipped[sess.name] = f"no {CROP_CLASS} annotations"
            continue
        pred_dir = _predict(sess, pred_root, CHECKPOINT)
        preds = load_polygons(pred_dir / "instances_default.json")
        sizes = image_sizes(pred_dir / "instances_default.json")

        keys = []
        for stem, size in sizes.items():
            if stem not in onions:
                skipped_no_onion += 1
                continue
            key = f"{sess.name}/{stem}"
            frames[key] = (onions[stem], preds.get(stem, []), size[0], size[1])
            roots_by_key[key] = (stem, [sess / "rgb", sess, pred_dir,
                                        pred_dir / "cvat_ready"])
            keys.append(key)
        skipped_no_pred += len(set(onions) - set(sizes))
        per_session_frames[sess.name] = keys
        if not keys:
            sessions_skipped[sess.name] = (
                f"{len(onions)} annotated frame(s), but none of them were "
                f"among the {len(sizes)} predicted (STRIDE={STRIDE})")
        print(f"    {len(keys)} frame(s) with both onions and predictions")

    if not frames:
        raise SystemExit(
            "ERROR: no frame had both an onion annotation and a prediction.\n"
            "Either the sessions carry no onion_plant annotations, or STRIDE "
            f"({STRIDE}) skipped every annotated frame.")

    per_frame = score_frames(frames, min_score=CONF)
    pooled = summarise(per_frame)
    rows = sweep(frames, SWEEP, CONF)
    per_session = {name: summarise({k: per_frame[k] for k in keys})
                   for name, keys in per_session_frames.items() if keys}
    warnings = provenance_warnings(model_classes, skipped_no_onion,
                                   skipped_no_pred, sessions_skipped)

    shots = worst(per_frame, WORST_FRAMES)
    if shots:
        (out / "worst").mkdir(exist_ok=True)
    written = []
    for key in shots:
        stem, roots = roots_by_key[key]
        img_path = _find_frame(stem, roots)
        if img_path is None:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        onion_insts, pred_insts, _, _ = frames[key]
        vis = draw_overlay(bgr, onion_insts, pred_insts,
                           per_frame[key]["on_crop_idx"], min_score=CONF)
        name = key.replace("/", "__") + ".jpg"
        cv2.imwrite(str(out / "worst" / name),
                    cv2.resize(vis, None, fx=0.5, fy=0.5))
        written.append(name)

    report = format_report(pooled, checkpoint=CHECKPOINT, conf=CONF,
                           sweep_rows=rows, warnings=warnings,
                           per_session=per_session,
                           model_classes=model_classes)
    print(report)

    for r in per_frame.values():
        r.pop("on_crop_idx", None)
    (out / "crop_risk.json").write_text(json.dumps({
        "checkpoint": CHECKPOINT, "conf": CONF, "stride": STRIDE,
        "detection_on_crop_min": DETECTION_ON_CROP_MIN,
        "onion_covered_min": ONION_COVERED_MIN,
        "model_classes": model_classes,
        "sessions": [str(s) for s in sessions],
        "sessions_skipped": sessions_skipped,
        "pooled": pooled, "per_session": per_session, "sweep": rows,
        "per_frame": per_frame, "worst_frames": written,
        "warnings": warnings,
    }, indent=2), encoding="utf-8")
    (out / "crop_risk_report.txt").write_text(report, encoding="utf-8")
    print(f"  wrote {out / 'crop_risk.json'}")
    if written:
        print(f"  {len(written)} overlay(s) in {out / 'worst'} - look at these "
              f"before quoting the rate.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
