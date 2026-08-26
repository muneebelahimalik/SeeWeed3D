#!/usr/bin/env python3
"""
SeeWeed3D - the self-training round: score predictions, split, write two batches.

    python -m seeweed3d.training.datasets.weeds_selftrain

Point IMAGES at a folder of frames. It runs inference if predictions for them do
not exist yet, scores every frame against the vegetation prior, and writes:

    accept/      pseudo-labels, safe to merge into the next build
    review/      the frames the model got WRONG - annotate these
    spot_check/  a sample of `accept`, for a human to glance at

SCORING NEEDS NO GPU. Masks are re-derived from the prediction polygons, so
once a folder has been predicted it can be re-scored at a different threshold in
seconds without touching the model. Only the first run per (round, folder) puts
anything on the GPU.

READ pseudo_label.py FIRST
--------------------------
The scoring, the guardrails and the reason confidence is NOT the ranking signal
are all documented there. This module is the plumbing around it.

THE ONE THING TO GET RIGHT
--------------------------
Run BOTH halves. `accept` is cheap and teaches little - its value is volume and
coverage, and it stops the model forgetting. `review` is where the gradient is.
A loop that only ever merges `accept` is a model talking to itself, and it will
look like it is working right up until it stops.
"""
import json
import ntpath
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from common.ontology import coco_categories, cvat_labels  # noqa: E402
from common.run_dirs import (newest, stale_predictions_warning,  # noqa: E402
                             stamped)
from training import pseudo_label as pl  # noqa: E402
from training.datasets.weeds import (HOLDOUT_TEST,  # noqa: E402
                                     OUT_DIR as WEEDS_OUT_DIR,
                                     WEED_POOL_ROOT)
from training.datasets.weeds_train import ROUND, RUNS_ROOT  # noqa: E402
from training.splits import MIN_SEAM_SEPARATION  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: The session whose predictions are being scored. Must NOT be a holdout - a
#: holdout that receives pseudo-labels has the model's own output in its own
#: test set, and every later round then scores against what it already believes.
SESSION = "vid3_20260108_110444"

#: THE FRAMES TO SCORE. A session folder (its rgb/ is used), or any plain
#: folder of images. Leave "" to score a session from the weed pool by name.
#:
#: If no predictions exist for these frames yet, this runs the inference pass
#: itself - so pointing at a folder and running once is the whole loop's input
#: step. An existing prediction folder is reused, which makes re-scoring at a
#: different threshold free.
IMAGES = ""

#: Where predictions live, or will be written.
#: Defaults to the newest look folder for this session, or "" when there is
#: none - in which case a fresh stamped one is made and inference runs into it.
#: This is the ONE folder that is reused rather than stamped: re-scoring at a
#: different ACCEPT threshold should not cost another GPU pass. Reuse is
#: checked against the checkpoint's date, so an older model's predictions
#: cannot be scored without saying so.
PREDICTIONS = newest(ntpath.join(RUNS_ROOT, f"weeds_r{ROUND}"),
                     f"look_{SESSION}") or ""

#: Inference settings, used ONLY when predictions have to be generated.
#: 0 = every frame found. Consecutive ZED frames are near-identical, so the
#: stride matters more than the count - and it matters MORE here than anywhere
#: else, because accepted frames get weighted: an error on one plant enters the
#: training set once per near-copy. Below splits.MIN_SEAM_SEPARATION the run
#: says so and gives the number of distinct frames you are really getting.
INFER_LIMIT = 0
INFER_STRIDE = 5

#: Below a deployment threshold on purpose: a mask the model nearly drew is
#: evidence about where it is unsure, and the scorer needs to see it to judge
#: the frame. This is NOT the confidence the pseudo-labels are filtered by -
#: that is the frame score, and it is mostly not made of confidence at all.
INFER_CONF = 0.25

#: Where the two batches are written. One folder per RUN, not per round: a
#: batch is something you take away and spend hours correcting in CVAT, so a
#: second run of the same round must not land on top of a half-finished one.
OUT_DIR = stamped(ntpath.join(RUNS_ROOT, f"weeds_r{ROUND}"),
                  f"selftrain_{SESSION}")

#: Frames scoring at or above this become pseudo-labels. LOOK AT THE SWEEP the
#: first run prints before trusting the default - it was chosen from synthetic
#: data, and yours is the data that matters.
ACCEPT = pl.ACCEPT_SCORE

#: Below this, a frame goes to review/ - the model is wrong here.
REVIEW = pl.REVIEW_SCORE

#: How many hand-corrected frames the dataset holds. The pseudo budget is
#: computed against THIS, not against the dataset size, so a mostly-pseudo
#: dataset cannot use its own size to justify more. 0 = read it from the build.
N_HAND = 0

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

IMG_SUFFIXES = {".png", ".jpg", ".jpeg"}

#: Written into every batch folder. A batch that travels to another machine, or
#: is opened three weeks later, carries its own instructions - the round trip has
#: four steps and getting the order wrong silently creates a DUPLICATE label in
#: CVAT rather than filling the one the prelabels were meant to correct.
CVAT_STEPS = """SeeWeed3D - {provenance}
{n_images} frame(s), {n_inst} instance(s)

CVAT ROUND TRIP
---------------
1. New task in CVAT. Upload the images from  cvat_ready/
2. BEFORE importing anything: open the task's Raw label editor and paste the
   whole contents of  weed_cvat_labels.json
   The schema must exist FIRST. CVAT matches annotations to labels BY NAME, so
   importing into a task with no matching label silently creates a duplicate
   instead of filling the one you meant.
3. Import  instances_default.json  as "COCO 1.0".
4. Correct, then Export the task as "Datumaro 1.0" into the session's
   annotations/ folder.

WHAT TO CORRECT, IN ORDER OF VALUE
----------------------------------
* WHAT IS MISSING. A pre-labelled frame biases you toward accepting what is
  drawn and not noticing what is absent, and a missed weed is this project's
  failure mode. Sweep the frame for green with no outline on it first.
* SPECIES. The model proposes only what it was trained on. cutleaf_evening_
  primrose vs wild_radish is an appearance call it cannot make reliably.
* CLUSTERS. weed_cluster means "no separable single LEP" - every one is a plant
  that never gets targeted individually. Split it if separating is merely
  tedious rather than impossible.
* BOUNDARIES LAST. Big weeds are already accurate; small ones are where the
  error is, and even there the crown matters more than the leaf margin.

The class list here is the FULL ontology, not the three classes the model can
predict. That is deliberate: you must be able to correct an instance into a
class the model has never seen.
"""



def newest_trained_round(runs_root):
    """The highest weeds_rN under `runs_root` that actually holds a checkpoint.

    A directory alone is not a round: an interrupted run leaves weeds_r3/ with
    tensorboard events and no weights, and treating that as "the latest model"
    would send every later step at a checkpoint that does not exist."""
    best = None
    root = Path(runs_root)
    if not root.is_dir():
        return None
    for d in root.glob("weeds_r*"):
        if not (d / "checkpoint_best_total.pth").exists():
            continue
        try:
            n = int(d.name.split("weeds_r", 1)[1])
        except (IndexError, ValueError):
            continue
        best = n if best is None else max(best, n)
    return best


def stale_round_warning(runs_root, round_in_use):
    """A warning when a NEWER trained round exists than the one being used.

    ROUND is edited by hand in weeds_train.py, and every other runner reads it
    from there. Train round 2 and forget to bump it, and this scores round 1's
    predictions while the report, the batch folder and the pseudo-labels are all
    named for a model that is no longer the best one - silently, because a
    checkpoint that exists loads fine."""
    newest = newest_trained_round(runs_root)
    if newest is None or newest <= int(round_in_use):
        return None
    return (f"  [!] ROUND is {round_in_use}, but weeds_r{newest} has a trained "
            f"checkpoint.\n"
            f"      This is about to use the OLDER model. If that is not "
            f"deliberate, set ROUND = {newest} in weeds_train.py - every runner "
            f"reads it from there.\n"
            f"      Using an older model is legitimate when comparing rounds; "
            f"it is a mistake when you meant 'the latest'.")


def stride_redundancy_warning(stride, n_accepted, n_hand=None):
    """A warning when the accepted frames are not as many frames as they look.

    Consecutive ZED frames are near-identical, so a count of pseudo-label frames
    is only a count of distinct ground if they are far enough apart. The project
    already has a number for that - splits.MIN_SEAM_SEPARATION, the floor below
    which two frames are treated as the same photograph - and INFER_STRIDE is
    set independently of it, in a different file, with nothing connecting them.

    This matters more here than in a split. A split that is too close reports an
    optimistic score; pseudo-labels that are too close get WEIGHTED, so a
    systematic error on one plant enters the training set once per near-copy and
    the next round learns it that many times over."""
    stride = int(stride or 1)
    if stride >= MIN_SEAM_SEPARATION or n_accepted <= 0:
        return None
    repeat = MIN_SEAM_SEPARATION / stride
    distinct = max(1, round(n_accepted * stride / MIN_SEAM_SEPARATION))
    tail = (f"\n      They will be weighted as {n_accepted} against {n_hand} "
            f"hand-corrected frame(s)." if n_hand else "")
    return (f"  [!] INFER_STRIDE is {stride}, but {MIN_SEAM_SEPARATION} video "
            f"frames is this project's floor for two\n"
            f"      frames not being the same photograph "
            f"(splits.MIN_SEAM_SEPARATION).\n"
            f"      So these {n_accepted} accepted frame(s) carry roughly "
            f"{distinct} frame(s) of distinct ground - the\n"
            f"      same plant appears in about {repeat:.0f} of them, and every "
            f"error in it repeats {repeat:.0f} times.{tail}\n"
            f"      Raise INFER_STRIDE to {MIN_SEAM_SEPARATION} and re-run if "
            f"you want the count to mean frames.")


def _hand_frame_count(dataset_dir):
    """Hand-corrected frames in the build, from its own manifest."""
    man = Path(dataset_dir) / "seg_manifest.json"
    if not man.exists():
        return 0
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    prov = str(doc.get("label_provenance", ""))
    frames = doc.get("frames") or []
    # Only hand_corrected counts. A "mixed" build already contains pseudo
    # labels and counting them would let the budget compound each round.
    return len(frames) if prov == "hand_corrected" else 0


def _find_image(name, roots):
    for r in roots:
        p = Path(r)
        for cand in (p / name, p / "rgb" / name):
            if cand.exists():
                return cand
        hits = list(p.rglob(name))
        if hits:
            return hits[0]
    return None


def _write_batch(frames, per_frame, out_dir, names, provenance, link=True):
    """A CVAT-ready folder plus its COCO, stamped with where the labels came
    from. `provenance` travels into info.description so a batch can never be
    mistaken for a hand-corrected export months later."""
    out = Path(out_dir)
    ready = out / "cvat_ready"
    ready.mkdir(parents=True, exist_ok=True)
    cats = coco_categories(list(names))
    cat_id = {c["name"]: c["id"] for c in cats}

    images, anns, ann_id = [], [], 1
    for img_id, fn in enumerate(sorted(frames), start=1):
        rec = per_frame[fn]
        src = rec["image_path"]
        dst = ready / Path(src).name
        if not dst.exists():
            try:
                if link:
                    import os
                    os.link(src, dst)
                else:
                    raise OSError
            except OSError:
                shutil.copy2(src, dst)
        h, w = rec["shape"]
        images.append({"id": img_id, "file_name": Path(src).name,
                       "height": int(h), "width": int(w)})
        for poly, cls, area in zip(rec["polys"], rec["classes"], rec["areas"]):
            if cls not in cat_id or not poly:
                continue
            xs, ys = poly[0::2], poly[1::2]
            anns.append({"id": ann_id, "image_id": img_id,
                         "category_id": cat_id[cls], "segmentation": [poly],
                         "iscrowd": 0,
                         "bbox": [min(xs), min(ys),
                                  max(xs) - min(xs), max(ys) - min(ys)],
                         "area": float(area)})
            ann_id += 1

    (out / "instances_default.json").write_text(json.dumps({
        "info": {"description": provenance},
        "licenses": [], "images": images, "annotations": anns,
        "categories": cats}, indent=2), encoding="utf-8")

    # THE FULL ONTOLOGY, not this model's reduced class list. The model can
    # only predict what it was trained on - three classes right now - but the
    # annotator has to be able to correct an instance INTO a class it cannot
    # predict. Without wild_radish in the schema there is no way to fix a
    # misread one, and without onion_plant an annotator who finds crop in a
    # weed-only frame is forced to call it a weed. That is the one error this
    # project cannot afford.
    (out / "weed_cvat_labels.json").write_text(
        json.dumps(cvat_labels(), indent=2), encoding="utf-8")

    (out / "README.txt").write_text(CVAT_STEPS.format(
        provenance=provenance, n_images=len(images), n_inst=len(anns)),
        encoding="utf-8")
    return len(images), len(anns)


def main():
    import cv2

    warn = stale_round_warning(RUNS_ROOT, ROUND)
    if warn:
        print()
        print(warn)

    if SESSION in set(HOLDOUT_TEST):
        raise SystemExit(
            f"ERROR: {SESSION!r} is in HOLDOUT_TEST.\n"
            f"Pseudo-labelling a holdout puts the model's own output into its "
            f"own test set, and every later round then scores against what it "
            f"already believes. Pick a session that is not held out.")

    images_root = IMAGES or ntpath.join(WEED_POOL_ROOT, SESSION)
    round_dir = ntpath.join(RUNS_ROOT, f"weeds_r{ROUND}")
    ckpt_path = ntpath.join(round_dir, "checkpoint_best_total.pth")
    # Empty means no look folder existed at import. Make one now rather than
    # writing predictions into a name that says nothing about when they ran.
    pred_dir = Path(PREDICTIONS or stamped(round_dir, f"look_{SESSION}"))
    coco = pred_dir / "instances_default.json"

    if coco.exists():
        print(f"\n  reusing predictions in {pred_dir}")
        warn = stale_predictions_warning(pred_dir, ckpt_path)
        if warn:
            print(warn)

    if not coco.exists():
        # Generate them rather than sending the user away and back. Reusing an
        # existing prediction folder is what makes re-scoring at a different
        # threshold free, so this only ever runs once per (round, session).
        if not Path(images_root).exists():
            raise SystemExit(
                f"ERROR: no predictions at {coco},\n"
                f"and IMAGES does not exist either: {images_root}\n"
                f"Set IMAGES to a folder of frames, or SESSION to a session "
                f"under {WEED_POOL_ROOT}.")
        ckpt = ckpt_path
        if not Path(ckpt).exists():
            raise SystemExit(
                f"ERROR: no checkpoint at {ckpt}.\n"
                f"Train round {ROUND} first:\n"
                f"    python -m seeweed3d.training.datasets.weeds_train")
        print(f"\n  no predictions yet - running inference over {images_root}")
        from perception.predict_images import CONFIG as PBASE, predict
        predict(dict(PBASE, IMAGES=images_root, CHECKPOINT=ckpt,
                     BACKEND="rfdetr", DEVICE="cuda", MODE="segmentation",
                     OUT_DIR=str(pred_dir), CONF=INFER_CONF,
                     LIMIT=INFER_LIMIT, STRIDE=INFER_STRIDE,
                     OVERLAY_SCALE=0.5, WRITE_COCO=True))
    if not coco.exists():
        raise SystemExit(f"ERROR: inference wrote no {coco}.")

    doc = json.loads(coco.read_text(encoding="utf-8"))
    names = [c["name"] for c in doc.get("categories", [])]
    cat_name = {c["id"]: c["name"] for c in doc.get("categories", [])}

    # Grouped by image, and NOTHING is rasterised yet. A mask per instance at
    # full frame size is 2.7 MB, so a real session - 79 frames, 2,840 instances
    # - is 7.8 GB held at once, and the process dies after the GPU pass has
    # already been paid for. The scorer only ever needs the UNION per frame, so
    # one frame's polygons are rasterised into one array and released.
    anns_by_image = {}
    for a in doc.get("annotations", []):
        anns_by_image.setdefault(a["image_id"], []).append(a)

    roots = [images_root, str(pred_dir), str(pred_dir / "cvat_ready")]
    qualities, per_frame, descriptors, order = [], {}, [], []
    from training.active_learning import appearance_descriptor
    print(f"\n  scoring {len(doc.get('images', []))} frame(s) from {SESSION}")

    for im in sorted(doc.get("images", []), key=lambda i: i["file_name"]):
        fn = Path(im["file_name"]).name
        img = _find_image(fn, roots)
        if img is None:
            print(f"  [skip] image not found for {fn}")
            continue
        bgr = cv2.imread(str(img))
        if bgr is None:
            print(f"  [skip] unreadable: {img}")
            continue

        h, w = bgr.shape[:2]
        union = np.zeros((h, w), np.uint8)
        polys, classes, areas, scores = [], [], [], []
        for a in anns_by_image.get(im["id"], []):
            seg = a.get("segmentation") or []
            poly = seg[0] if seg else []
            if not poly:
                continue
            pts = np.asarray(poly, np.float64).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(union, [np.round(pts).astype(np.int32)], 1)
            polys.append(poly)
            classes.append(cat_name.get(a["category_id"], ""))
            areas.append(float(a.get("area", 0.0)))
            scores.append(float(a.get("score", 1.0)))

        q = pl.frame_quality(bgr, [union.astype(bool)], scores)
        q["frame"] = fn
        qualities.append(q)
        per_frame[fn] = {"image_path": str(img), "shape": (h, w),
                         "polys": polys, "classes": classes, "areas": areas,
                         "quality": q}
        descriptors.append(appearance_descriptor(bgr))
        order.append(fn)

    if not qualities:
        raise SystemExit("ERROR: no frames could be scored.")

    summary = pl.summarise(qualities, ACCEPT, REVIEW)
    n_hand = N_HAND or _hand_frame_count(WEEDS_OUT_DIR)
    budget = pl.pseudo_budget(n_hand)
    print(pl.format_report(summary, n_hand=n_hand, budget=budget))

    buckets = {"accept": [], "review": [], "skip": []}
    for q in qualities:
        buckets[pl.classify(q, ACCEPT, REVIEW)].append(q["frame"])

    # Diversity BEFORE the budget cut, so the frames that survive are spread
    # across the drive rather than clustered on one easy stretch.
    from training.active_learning import greedy_diverse
    idx = {f: i for i, f in enumerate(order)}
    accepted = buckets["accept"]
    if budget and len(accepted) > budget:
        sub = [descriptors[idx[f]] for f in accepted]
        pick = greedy_diverse(sub, budget,
                              [per_frame[f]["quality"]["score"] for f in accepted])
        accepted = [accepted[i] for i in pick]
    dominant = {f: (per_frame[f]["classes"][0] if per_frame[f]["classes"] else "")
                for f in accepted}
    accepted = pl.balance_by_class(accepted, dominant)

    out = Path(OUT_DIR)
    n_img, n_ann = _write_batch(accepted, per_frame, out / "accept", names,
                                "SeeWeed3D PSEUDO-LABELS - model output, "
                                "not reviewed")
    print(f"\n  accept/     {n_img} frame(s), {n_ann} instance(s)"
          f"  -> {out / 'accept' / 'cvat_ready'}")
    r_img, r_ann = _write_batch(buckets["review"], per_frame, out / "review",
                                names,
                                "SeeWeed3D model prelabels FOR CORRECTION - "
                                "the model scored badly on these")
    print(f"  review/     {r_img} frame(s), {r_ann} instance(s)"
          f"  -> {out / 'review' / 'cvat_ready'}")

    # After the counts, not before: the warning is about what those counts
    # actually mean, so it has to sit where they can be read together.
    warn = stride_redundancy_warning(INFER_STRIDE, n_img, n_hand)
    if warn:
        print(warn)

    # The spot check is not optional. Ten frames is minutes of work and it is
    # the only thing between a badly-chosen threshold and a poisoned dataset.
    spot = sorted(accepted)[:: max(1, len(accepted) // max(1, pl.SPOT_CHECK))]
    spot = spot[:pl.SPOT_CHECK]
    sdir = out / "spot_check"
    sdir.mkdir(parents=True, exist_ok=True)
    for fn in spot:
        ov = pred_dir / "overlays" / (Path(fn).stem + ".png")
        if ov.exists():
            shutil.copy2(ov, sdir / ov.name)
    print(f"  spot_check/ {len(spot)} overlay(s)             -> {sdir}")

    (out / "selftrain_report.json").write_text(json.dumps({
        "session": SESSION, "round": ROUND, "accept_threshold": ACCEPT,
        "review_threshold": REVIEW, "n_hand_corrected": n_hand,
        "pseudo_budget": budget, "summary": summary,
        "accepted": sorted(accepted), "review": sorted(buckets["review"]),
        "per_frame": {f: per_frame[f]["quality"] for f in per_frame},
    }, indent=2, default=float), encoding="utf-8")

    print(f"\n  NEXT, and do BOTH halves:")
    print(f"    1. Open spot_check/ and look. If any accepted frame is wrong, "
          f"raise ACCEPT and re-run - this costs seconds, not a GPU pass.")
    print(f"    2. Upload review/cvat_ready/ to CVAT and CORRECT it. These are "
          f"the frames that move the model.")
    print(f"    3. Add accept/ to the build as a session with "
          f"LABEL_PROVENANCE='pseudo_label', or 'mixed' once corrected frames "
          f"sit beside it.")
    print(f"\n  Merging accept/ alone is a model talking to itself. It will "
          f"look like it is working right up until it stops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
