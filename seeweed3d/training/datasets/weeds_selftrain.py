#!/usr/bin/env python3
"""
SeeWeed3D - the self-training round: score every unseen session, write batches.

    python -m seeweed3d.training.datasets.weeds_selftrain

With no configuration it scores EVERY session in the weed pool that is neither
held out nor already in the training build, runs inference where predictions do
not exist yet, and writes one subfolder per session:

    <session>/accept/      pseudo-labels, safe to merge into the next build
    <session>/review/      the frames the model got WRONG - annotate these
    <session>/spot_check/  a sample of `accept`, for a human to glance at

plus a pooled selftrain_report.json and NEXT_STEPS.txt at the top.

MORE SESSIONS, NOT A SMALLER STRIDE
-----------------------------------
One drive is a few hundred near-identical frames. Scoring it at stride 5 gives
79 "frames" carrying about 6 frames of distinct ground - and unlike a split,
where being too close only flatters a score, pseudo-labels get WEIGHTED, so an
error on one plant enters the training set once per near-copy. So the stride
defaults to the project's own separation floor and the way to get more data is
to point this at more drives.

ONE SUBFOLDER PER SESSION, NOT ONE POOLED BATCH
-----------------------------------------------
Pooling would be one less CVAT task and would break the round trip. The build
takes one session folder per source, and the gap accounting and split logic are
computed from session identity - so a corrected frame has to go back to the
drive it came from.

SCORING NEEDS NO GPU. Masks are re-derived from the prediction polygons, so
once a folder has been predicted it can be re-scored at a different threshold in
seconds without touching the model. Only the first run per (round, session) puts
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
import os.path
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

#: THE SESSIONS TO SCORE. Empty = every session in the pool that is neither
#: held out nor already in the training build. That default is the useful one:
#: one drive is a few hundred near-identical frames and cannot supply a round
#: on its own, so "as many frames as possible" means MORE SESSIONS, not a
#: smaller stride within one.
#:
#: A session already in training is excluded automatically, read from the
#: build's own manifest rather than from a list kept in step by hand.
SESSIONS = []

#: A plain folder of images to score instead of the pool. Set this and SESSIONS
#: is ignored - it is the escape hatch for frames that are not laid out as a
#: session at all.
IMAGES = ""

#: Inference settings, used ONLY when predictions have to be generated.
#: 0 = every frame found.
#:
#: STRIDE IS THE SETTING THAT MATTERS, and it defaults to the project's own
#: floor for two frames not being the same photograph. Consecutive ZED frames
#: are near-identical, and unlike a split - where being too close only flatters
#: a score - pseudo-labels get WEIGHTED, so an error on one plant enters the
#: training set once per near-copy. At stride 5 a 393-frame drive yields 79
#: frames carrying about 6 frames of distinct ground.
INFER_LIMIT = 0
INFER_STRIDE = MIN_SEAM_SEPARATION

#: Below a deployment threshold on purpose: a mask the model nearly drew is
#: evidence about where it is unsure, and the scorer needs to see it to judge
#: the frame. This is NOT the confidence the pseudo-labels are filtered by -
#: that is the frame score, and it is mostly not made of confidence at all.
INFER_CONF = 0.25

#: Where the run is written. One folder per RUN, not per round: a batch is
#: something you take away and spend hours correcting in CVAT, so a second run
#: of the same round must not land on top of a half-finished one.
#:
#: Inside it, ONE SUBFOLDER PER SESSION. Pooling every session into a single
#: batch would be one less CVAT task and would break the round trip: the build
#: takes one session folder per source, and session identity is what the gap
#: accounting and the split logic are computed from. A corrected export has to
#: go back to the drive it came from.
OUT_DIR = stamped(ntpath.join(RUNS_ROOT, f"weeds_r{ROUND}"), "selftrain")

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


def trained_sessions(dataset_dir):
    """Sessions already in the training build, from its own manifest.

    Read rather than listed by hand, because a session that is both trained on
    and pseudo-labelled produces frames the model has effectively seen, scored
    at ceiling, and fed back as if they were new - and nothing in the output
    would look wrong. The manifest already records this; a second list would
    just be a thing to forget to update."""
    man = Path(dataset_dir) / "seg_manifest.json"
    if not man.is_file():
        return set()
    try:
        return set(json.loads(man.read_text(encoding="utf-8"))
                   .get("sessions", []))
    except (ValueError, OSError):
        return set()


def discover_sessions(pool_root, exclude=()):
    """Every session folder under the pool that holds frames, minus `exclude`.

    A session needs images to be worth scoring; a folder with an rgb/ that is
    empty is an aborted recording, and including it would produce an empty
    batch folder that looks like a session someone forgot to correct."""
    root = Path(pool_root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in set(exclude):
            continue
        imgs = d / "rgb" if (d / "rgb").is_dir() else d
        if any(p.suffix.lower() in IMG_SUFFIXES for p in imgs.iterdir()
               if p.is_file()):
            out.append(d.name)
    return out


def session_plan(pool_root, dataset_dir, holdout, requested=()):
    """Which sessions will be scored, and why each excluded one was not.

    The reasons are printed. A run that silently scores three of seven sessions
    is indistinguishable from a run that found only three, and the difference
    matters: one is the guardrail working and the other is a wrong path."""
    trained = trained_sessions(dataset_dir)
    held = set(holdout)
    if requested:
        found = list(requested)
    else:
        found = discover_sessions(pool_root)
    chosen, skipped = [], []
    for s in found:
        if s in held:
            skipped.append((s, "held out - a holdout that receives "
                               "pseudo-labels tests the model on its own output"))
        elif s in trained:
            skipped.append((s, "already in the training build - the model has "
                               "seen this ground, so it scores at ceiling and "
                               "teaches nothing"))
        else:
            chosen.append(s)
    return chosen, skipped


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


def _predict(session, images_root, round_dir, ckpt_path):
    """Predictions for one session, reused if a folder already has them.

    Reuse is what makes re-scoring at a different ACCEPT threshold free, and it
    is also how an older model's output gets scored after a retrain - so the
    reuse is checked against the checkpoint's date and says so."""
    pred_dir = Path(newest(round_dir, f"look_{session}")
                    or stamped(round_dir, f"look_{session}"))
    coco = pred_dir / "instances_default.json"

    if coco.exists():
        print(f"  reusing predictions in {pred_dir}")
        warn = stale_predictions_warning(pred_dir, ckpt_path)
        if warn:
            print(warn)
        return pred_dir

    if not Path(images_root).exists():
        raise SystemExit(
            f"ERROR: no predictions at {coco},\n"
            f"and the frames do not exist either: {images_root}")
    if not Path(ckpt_path).exists():
        raise SystemExit(
            f"ERROR: no checkpoint at {ckpt_path}.\n"
            f"Train round {ROUND} first:\n"
            f"    python -m seeweed3d.training.datasets.weeds_train")
    print(f"  no predictions yet - running inference over {images_root}")
    from perception.predict_images import CONFIG as PBASE, predict
    predict(dict(PBASE, IMAGES=images_root, CHECKPOINT=ckpt_path,
                 BACKEND="rfdetr", DEVICE="cuda", MODE="segmentation",
                 OUT_DIR=str(pred_dir), CONF=INFER_CONF,
                 LIMIT=INFER_LIMIT, STRIDE=INFER_STRIDE,
                 OVERLAY_SCALE=0.5, WRITE_COCO=True))
    if not coco.exists():
        raise SystemExit(f"ERROR: inference wrote no {coco}.")
    return pred_dir


def _score(session, images_root, pred_dir):
    """Score every predicted frame of one session against the vegetation prior.

    NOTHING IS RASTERISED PER INSTANCE. A full-frame mask is 2.7 MB, so a real
    session's worth held at once is gigabytes and the process dies after the
    GPU pass has already been paid for. The scorer only needs the union per
    frame, so one frame's polygons go into one array which is then released."""
    import cv2
    from training.active_learning import appearance_descriptor

    doc = json.loads((pred_dir / "instances_default.json")
                     .read_text(encoding="utf-8"))
    names = [c["name"] for c in doc.get("categories", [])]
    cat_name = {c["id"]: c["name"] for c in doc.get("categories", [])}

    anns_by_image = {}
    for a in doc.get("annotations", []):
        anns_by_image.setdefault(a["image_id"], []).append(a)

    roots = [images_root, str(pred_dir), str(pred_dir / "cvat_ready")]
    qualities, per_frame, descriptors, order = [], {}, [], []
    print(f"  scoring {len(doc.get('images', []))} frame(s) from {session}")

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

    return {"names": names, "qualities": qualities, "per_frame": per_frame,
            "descriptors": descriptors, "order": order}


def _emit(session, scored, pred_dir, out, n_hand, budget):
    """Write one session's two batches, its spot check and its report."""
    qualities, per_frame = scored["qualities"], scored["per_frame"]
    summary = pl.summarise(qualities, ACCEPT, REVIEW)
    print(pl.format_report(summary, n_hand=n_hand, budget=budget))

    buckets = {"accept": [], "review": [], "skip": []}
    for q in qualities:
        buckets[pl.classify(q, ACCEPT, REVIEW)].append(q["frame"])

    # Diversity BEFORE the budget cut, so the frames that survive are spread
    # across the drive rather than clustered on one easy stretch.
    from training.active_learning import greedy_diverse
    idx = {f: i for i, f in enumerate(scored["order"])}
    accepted = buckets["accept"]
    if budget and len(accepted) > budget:
        sub = [scored["descriptors"][idx[f]] for f in accepted]
        pick = greedy_diverse(sub, budget,
                              [per_frame[f]["quality"]["score"]
                               for f in accepted])
        accepted = [accepted[i] for i in pick]
    dominant = {f: (per_frame[f]["classes"][0] if per_frame[f]["classes"] else "")
                for f in accepted}
    accepted = pl.balance_by_class(accepted, dominant)

    names = scored["names"]
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

    report = {"session": session, "round": ROUND, "accept_threshold": ACCEPT,
              "review_threshold": REVIEW, "n_hand_corrected": n_hand,
              "infer_stride": INFER_STRIDE, "pseudo_budget": budget,
              "summary": summary, "accepted": sorted(accepted),
              "review": sorted(buckets["review"]),
              "per_frame": {f: per_frame[f]["quality"] for f in per_frame}}
    (out / "selftrain_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def next_steps(run_dir, pool_root, per_session):
    """The round trip, written into the run folder with the real paths in it.

    Four steps, and getting step 2 wrong silently creates a DUPLICATE label in
    CVAT rather than filling the one the prelabels were meant to correct. It is
    written down because a batch outlives the terminal it was produced in."""
    L = [f"SeeWeed3D - self-training round {ROUND}",
         f"{len(per_session)} session(s), "
         f"{sum(r['summary']['accept'] for r in per_session)} accepted, "
         f"{sum(r['summary']['review'] for r in per_session)} for review",
         "",
         "PER SESSION",
         "-----------"]
    for r in per_session:
        s = r["summary"]
        L.append(f"  {r['session']:<28} scored {s['n_frames']:>4}   "
                 f"accept {s['accept']:>4}   review {s['review']:>4}")
    L += [
        "",
        "1. CORRECT review/ FIRST, one CVAT task per session.",
        "   Those are the frames the model got wrong, and they are the only",
        "   ones that move it. accept/ stops it forgetting; it teaches little.",
        "",
        "   For each session folder below:",
        "     a. New CVAT task, upload  <session>/review/cvat_ready/",
        "     b. BEFORE importing: paste  weed_cvat_labels.json  into the Raw",
        "        label editor. CVAT matches BY NAME - importing into a task",
        "        with no matching label silently creates a duplicate.",
        "     c. Import  instances_default.json  as \"COCO 1.0\".",
        "     d. Correct, then export as \"Datumaro 1.0\".",
        "",
        "2. PUT EACH EXPORT BACK IN ITS OWN SESSION.",
        "   The build takes one session folder per source and computes the gap",
        "   accounting from session identity, so a corrected frame has to go",
        "   back to the drive it came from:",
        "",
    ]
    for r in per_session:
        L.append(f"     {ntpath.join(pool_root, r['session'])}"
                 f"\\annotations\\default.json")
    L += [
        "",
        "3. ADD THOSE SESSIONS TO THE BUILD.",
        "   In training/datasets/weeds.py, extend WEED_SESSIONS with the",
        "   session folders you actually corrected, and set",
        "       LABEL_PROVENANCE = \"mixed\"",
        "   the moment any unreviewed frame is included. It stops being",
        "   \"hand_corrected\" the first time it stops being true, and every",
        "   score computed later is read through that field.",
        "",
        "4. REBUILD, BUMP THE ROUND, RETRAIN.",
        "       python -m seeweed3d.training.datasets.weeds",
        f"       ROUND = {ROUND + 1}   in training/datasets/weeds_train.py",
        "       python -m seeweed3d.training.datasets.weeds_train",
        "",
        "   Every other runner reads ROUND from weeds_train.py, so that one",
        "   edit moves inference, mining and the next self-training round too.",
        "",
        "WHAT TO CORRECT, IN ORDER OF VALUE",
        "----------------------------------",
        "* WHAT IS MISSING. A pre-labelled frame biases you toward accepting",
        "  what is drawn and not noticing what is absent, and a missed weed is",
        "  this project's failure mode: it becomes BACKGROUND in the label.",
        "* SPECIES. The model proposes only what it was trained on.",
        "* CLUSTERS. Split one if separating is merely tedious, not impossible.",
        "* BOUNDARIES LAST, and the crown matters more than the leaf margin.",
    ]
    text = "\n".join(L)
    (Path(run_dir) / "NEXT_STEPS.txt").write_text(text, encoding="utf-8")
    return text


def main():
    warn = stale_round_warning(RUNS_ROOT, ROUND)
    if warn:
        print()
        print(warn)

    # os.path, not ntpath: these are opened, not printed. On Windows the
    # two are the same module; anywhere else ntpath builds a path with
    # separators the filesystem does not recognise.
    round_dir = os.path.join(RUNS_ROOT, f"weeds_r{ROUND}")
    ckpt_path = os.path.join(round_dir, "checkpoint_best_total.pth")
    n_hand = N_HAND or _hand_frame_count(WEEDS_OUT_DIR)
    budget = pl.pseudo_budget(n_hand)
    run_dir = Path(OUT_DIR)

    if IMAGES:
        # The escape hatch: frames that are not laid out as a session at all.
        jobs = [("images", IMAGES)]
        skipped = []
    else:
        chosen, skipped = session_plan(WEED_POOL_ROOT, WEEDS_OUT_DIR,
                                       HOLDOUT_TEST, SESSIONS)
        jobs = [(s, os.path.join(WEED_POOL_ROOT, s)) for s in chosen]

    if skipped:
        print("\n  skipping:")
        for s, why in skipped:
            print(f"    {s:<28} {why}")
    if not jobs:
        raise SystemExit(
            f"ERROR: no sessions to score under {WEED_POOL_ROOT}.\n"
            f"Every session there is either held out or already in the "
            f"training build, or the path is wrong.\n"
            f"Name sessions explicitly in SESSIONS, or point IMAGES at a "
            f"folder of frames.")

    print(f"\n  {len(jobs)} session(s) to score, stride {INFER_STRIDE}, "
          f"round {ROUND}")

    reports = []
    for session, images_root in jobs:
        print(f"\n{'=' * 70}\n  {session}\n{'=' * 70}")
        pred_dir = _predict(session, images_root, round_dir, ckpt_path)
        scored = _score(session, images_root, pred_dir)
        if not scored["qualities"]:
            print(f"  [skip] no frames could be scored in {session}")
            continue
        reports.append(_emit(session, scored, pred_dir,
                             run_dir / session, n_hand, budget))

    if not reports:
        raise SystemExit("ERROR: no frames could be scored in any session.")

    n_acc = sum(r["summary"]["accept"] for r in reports)
    n_rev = sum(r["summary"]["review"] for r in reports)
    (run_dir / "selftrain_report.json").write_text(json.dumps({
        "round": ROUND, "infer_stride": INFER_STRIDE,
        "n_hand_corrected": n_hand, "pseudo_budget": budget,
        "accept_threshold": ACCEPT, "review_threshold": REVIEW,
        "skipped": [{"session": s, "reason": w} for s, w in skipped],
        "sessions": reports,
    }, indent=2, default=float), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"  {len(reports)} session(s): {n_acc} accepted, "
          f"{n_rev} for review")
    # The budget is computed against the HAND-corrected count, not the dataset
    # size, so a mostly-pseudo dataset cannot use its own size to justify more.
    # Pooling sessions is exactly how it gets exceeded without anyone noticing.
    if budget and n_acc > budget:
        print(f"  [!] {n_acc} accepted across all sessions exceeds the budget "
              f"of {budget} for {n_hand} hand-corrected frame(s).")
        print(f"      The budget is enforced PER SESSION, so pooling can pass "
              f"it. Correct more frames, or drop whole sessions from the "
              f"merge - do not merge all of them.")
    print(f"  -> {run_dir}")
    print(next_steps(run_dir, WEED_POOL_ROOT, reports))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
