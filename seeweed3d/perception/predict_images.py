#!/usr/bin/env python3
"""
SeeWeed3D - run a trained model on images and LOOK at the result
================================================================
    python seeweed3d/perception/predict_images.py

Edit the CONFIG block below. Unlike evaluation/report.py this needs NO ground
truth, so it works on any folder of frames - a held-out session, a new field, a
different time of day.

WHAT IT CANNOT TELL YOU
-----------------------
Without labels there is no recall. A frame with no weeds drawn on it means
"the model found nothing", which is indistinguishable from "there was nothing
to find". Use this to see HOW the model fails, and evaluation/eval_seg.py to
learn how OFTEN. On a held-out session the interesting question is usually not
the score but whether the masks still sit on plants at all.

MODES
-----
  "segmentation"  RGB only. Masks, classes, scores. Runs on any image folder.
  "full"          The deployed pipeline: depth -> LEP -> 3D -> safety decision,
                  so each weed comes back as a CANDIDATE or an ABSTENTION with
                  reasons. Needs a session laid out as <session>/rgb, /depth
                  and /meta/calibration.json.

COLOURS
-------
One colour per CLASS (see CLASS_COLOURS), with onion_plant orange and drawn
thickest - it is the one thing in the frame that must not be hit. A weed
touching predicted onion gets an extra WHITE outline and a "!CROP" tag rather
than being recoloured, so you can still see WHICH weed is sitting on the crop.

In "full" mode that crop judgement comes from safety.py and the laser spot
geometry; in "segmentation" mode it is a cruder mask-overlap test, because
without a LEP there is no spot to test. Neither is a substitute for the
crop-safety numbers in eval_seg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # noqa: E402
from common.dedup import DEFAULT_DEDUP_IOU                    # noqa: E402
from common.ontology import CLASSES, CROP_CLASS               # noqa: E402
from perception.schema import STATUS_CANDIDATE                # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

CONFIG = {
    # -- What to run on --------------------------------------------------------
    # A session folder (its rgb/ is used), a plain folder of images, or a
    # single image file. A folder is searched recursively; a depth/ subfolder
    # is skipped, because depth PNGs share their RGB frame's filename and would
    # otherwise be fed to the model as pictures.
    "IMAGES": r"E:\Dataset_Vidalia\Weeds_1\auto_labels_weeds_6\vid3_20260108_110444\cvat_ready",

    # OR a SPLIT of a built dataset, which IMAGES cannot express. Set DATASET
    # to make_dataset.py's OUT_DIR and SPLIT to "test" and the frames come from
    # seg_manifest.json instead of from a folder. DATASET wins when both are
    # set.
    #
    # A split is not a directory. Its frames are scattered across every
    # session, and train and test frames sit side by side in the same rgb/
    # folder - so pointing IMAGES at a session and calling the result held-out
    # is wrong, and wrong in the flattering direction.
    "DATASET": "",
    "SPLIT": "test",

    # 0 = every frame found. Otherwise the first N, which is usually what you
    # want the first time you point this at 4000 frames.
    "LIMIT": 20,

    # Take every Nth frame instead of the first N. Consecutive ZED frames are
    # nearly identical, so 1 gives you 20 pictures of the same plant.
    "STRIDE": 10,

    # -- The model -------------------------------------------------------------
    "CHECKPOINT": r"E:\Dataset_Vidalia\training1\run4\best.pt",
    "BACKEND": "maskrcnn",        # "maskrcnn" | "rfdetr" - must match the file
    "DEVICE": "cuda",

    # The deployment confidence. Lower than the 0.5 the metrics table uses:
    # for looking at failures you want to see what the model nearly said.
    "CONF": 0.35,

    # -- Mode ------------------------------------------------------------------
    "MODE": "segmentation",       # "segmentation" | "full"

    # "full" only. "" uses the hand-engineered growth-point estimator, which
    # works before Stage B is trained.
    "LEP_CHECKPOINT": "",

    # -- Output ----------------------------------------------------------------
    "OUT_DIR": r"E:\Dataset_Vidalia\training1\run4\predictions",

    # Shrink the saved overlays. ZED frames are 2208x1242; 0.5 keeps them
    # readable and the folder small. 1.0 saves full size.
    "OVERLAY_SCALE": 0.5,

    # "class_score" | "class" | "none". A dense frame can carry 50 instances,
    # and at that point the text is the noise - set "none" and read the colours.
    "LABELS": "class_score",

    # Colour key in the corner, so an overlay can be read on its own.
    "LEGEND": True,

    # Drop a detection that duplicates a higher-scoring one at this mask IoU.
    # 0 disables it and restores the raw model output.
    #
    # RF-DETR predicts a SET: every query proposes independently and nothing
    # makes two queries that found the same plant agree on what it is, so the
    # same mask comes back under two class labels at two scores. Seen on a real
    # weed session in 6 of 16 frames, at box IoU 1.000. A laser weeder then
    # fires twice at one plant and a weed elsewhere goes untreated.
    #
    # Deliberately HIGH: the observed duplicates are near-identical, while two
    # genuinely adjacent plants in a dense frame reach 0.5-0.6 and merging THOSE
    # costs a real weed its own treatment point. See common/dedup.py.
    "DEDUP_IOU": DEFAULT_DEDUP_IOU,

    # Also write the predictions as COCO 1.0 beside predictions.json.
    # evaluation/bench_mixed.py consumes it, so the model can be compared
    # against the SAM prelabels on the same frames - and CVAT imports it
    # directly, so a promising frame can be corrected with no second pass.
    # Segmentation mode only; "full" mode's records are a different shape.
    "WRITE_COCO": True,
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

#: One colour per class, BGR. Chosen to stay apart both in hue and in
#: brightness, so they remain distinguishable on the grey-green soil these
#: frames are mostly made of - and for anyone who reads red and green alike.
#:
#: onion_plant is orange and drawn thickest: it is the one thing in the frame
#: that must not be hit, and it should be findable without reading a legend.
CLASS_COLOURS = {
    "cutleaf_evening_primrose": (255, 128, 0),     # blue
    "wild_radish":              (0, 220, 255),     # yellow
    "grass_weed":               (80, 200, 80),     # green
    "weed_cluster":             (200, 90, 200),    # purple
    "other_weed":               (60, 60, 235),     # red
    "onion_plant":              (0, 165, 255),     # orange - the crop
}
C_UNKNOWN = (200, 200, 200)   # grey    - a class not in the ontology
C_CONFLICT = (255, 255, 255)  # white   - outline on a weed touching the crop

#: Growth-point markers in "full" mode. The verdict, not the class: whether the
#: laser would fire is the question the whole pipeline exists to answer, and a
#: point it would refuse looks identical to one it would take unless the picture
#: says so. Cyan/magenta rather than green/red - the frame is already green and
#: red is already other_weed, and both stay apart for anyone who reads the two
#: alike.
C_CANDIDATE = (255, 255, 0)    # cyan    - would be treated
C_ABSTAIN = (255, 0, 255)      # magenta - a growth point the safety check refused


def class_colour(name):
    """Colour for a class, stable across runs and frames.

    A class absent from the ontology gets grey rather than an arbitrary colour:
    an unexpected label should look unexpected."""
    return CLASS_COLOURS.get(name, C_UNKNOWN)


def legend(names):
    """[(class_name, colour)] for the classes actually drawn."""
    return [(n, class_colour(n)) for n in names]


def find_images(spec, limit=0, stride=1):
    """Resolve IMAGES to a sorted list of frames.

    A session folder resolves to its rgb/ only. depth/ is excluded everywhere:
    a depth PNG has the SAME filename as its RGB frame, so a naive recursive
    search returns one of each and silently feeds 16-bit depth to the model as
    a picture.
    """
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"ERROR: IMAGES path does not exist: {p}")
    if p.is_file():
        return [p]

    root = p / "rgb" if (p / "rgb").is_dir() else p
    files = [f for f in sorted(root.rglob("*"))
             if f.suffix.lower() in IMAGE_SUFFIXES
             and not any(q.name.lower().startswith("depth")
                         for q in f.relative_to(root).parents)]
    if not files:
        raise SystemExit(
            f"ERROR: no images under {root}.\n"
            f"Expected a session folder (with an rgb/ subfolder), a folder of "
            f"images, or a single image file.")
    files = files[::max(1, int(stride))]
    return files[:limit] if limit else files


def split_images(dataset_dir, split, limit=0, stride=1, images_root=""):
    """The frames of one SPLIT of a built dataset, resolved to real paths.

    Pointing this at a folder cannot answer "what does it do on data it never
    trained on": a split is a set of frames scattered across every session, not
    a directory, and the frames of train and test sit side by side in the same
    rgb/ folder. Reading them from seg_manifest.json is the only way to be sure
    the pictures you are looking at are the held-out ones.

    Uses the same resolver as eval_seg, so a frame found here is a frame that
    was scored there - and a merged build's several sessions roots all work."""
    from training.seg_dataset import resolve_image

    man = Path(dataset_dir) / "seg_manifest.json"
    if not man.exists():
        raise SystemExit(
            f"ERROR: {man} not found. DATASET should be the OUT_DIR that "
            f"make_dataset.py wrote, the folder holding seg_manifest.json.")
    doc = json.loads(man.read_text(encoding="utf-8"))
    recs = [f for f in doc.get("frames", []) if f.get("split") == split]
    if not recs:
        have = sorted({f.get("split") for f in doc.get("frames", [])} - {None})
        raise SystemExit(
            f"ERROR: split {split!r} has no frames in {man}. "
            f"This dataset has: {', '.join(have) or '(none)'}")
    recs = sorted(recs, key=lambda r: str(r.get("image_path")))
    recs = recs[::max(1, int(stride))]
    if limit:
        recs = recs[:limit]
    root = images_root or doc.get("images_root") or "."
    out = []
    for r in recs:
        try:
            out.append(Path(resolve_image(r["image_path"], root,
                                          r.get("session_id"),
                                          r.get("export_dir"))))
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}")
    return out


def _session_of(path):
    """The session directory for a frame under <session>/rgb/, else None."""
    if path.parent.name == "rgb":
        return path.parent.parent
    return None


def _put_label(img, text, org, colour):
    """Text on a filled dark plate.

    Coloured text straight onto the frame is unreadable: these scenes are grey
    soil and green foliage, so a green label on a green plant disappears exactly
    where there is something to read."""
    import cv2
    f, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, tht), base = cv2.getTextSize(text, f, sc, th)
    x, y = int(org[0]), int(max(tht + 3, org[1]))
    x = max(0, min(x, img.shape[1] - tw - 5))
    cv2.rectangle(img, (x, y - tht - 3), (x + tw + 4, y + base - 1),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), f, sc, colour, th, cv2.LINE_AA)


def _legend_strip(img, names, pad=10, row=26, font=0.5):
    """Return img with a key APPENDED BELOW it.

    Drawn on extra canvas rather than over a corner of the frame: a legend
    painted onto the image hides whatever was underneath, and in these scenes
    the corners hold plants. A key that costs you a detection is a bad trade.
    """
    import cv2
    w = img.shape[1]
    entries = [(n + ("  (CROP)" if n == CROP_CLASS else ""), class_colour(n))
               for n in names]
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0]
              + 34 + pad for t, _ in entries]

    rows, cur, cur_w = [], [], pad
    for e, ew in zip(entries, widths):
        if cur and cur_w + ew > w - pad:
            rows.append(cur)
            cur, cur_w = [], pad
        cur.append((e, ew))
        cur_w += ew
    if cur:
        rows.append(cur)

    strip = np.zeros((pad + row * len(rows) + pad // 2, w, 3), np.uint8)
    for r, items in enumerate(rows):
        x, y = pad, pad + row * r + 14
        for (text, colour), ew in items:
            cv2.rectangle(strip, (x, y - 10), (x + 22, y + 3), colour, -1)
            cv2.putText(strip, text, (x + 30, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1,
                        cv2.LINE_AA)
            x += ew
    return np.vstack([img, strip])


def draw(bgr, det, conflict_idx, scale=1.0, alpha=0.35, labels="class_score",
         show_legend=True, targets=None, note=None):
    """Tint and outline every instance, ONE COLOUR PER CLASS.

    Crop proximity is drawn as an extra WHITE outline rather than by recolouring
    the instance. The class and the hazard are two different facts, and
    overwriting one with the other loses information at the moment it matters
    most - you would no longer be able to see WHICH weed is sitting on an onion.

    `targets` are WeedTargets from the full pipeline. Without them a full-mode
    overlay was pixel-identical to a segmentation one: the LEP and the 3D point
    existed only in the JSON, so the one thing a person runs the whole pipeline
    to LOOK at was the one thing the picture did not show.
    """
    import cv2
    out = bgr.copy()
    drawn = []
    for i in range(len(det)):
        m = np.asarray(det.masks[i]).astype(bool)
        if not m.any():
            continue
        name = det.class_name(i)
        colour = class_colour(name)
        is_crop = name == CROP_CLASS
        drawn.append(name)

        tint = np.zeros_like(out)
        tint[m] = colour
        out = cv2.addWeighted(out, 1.0, tint, alpha, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if i in conflict_idx:
            cv2.drawContours(out, cnts, -1, C_CONFLICT, 5)
        cv2.drawContours(out, cnts, -1, colour, 3 if is_crop else 2)

        if labels != "none":
            ys, xs = np.nonzero(m)
            text = (f"{name} {det.scores[i]:.2f}" if labels == "class_score"
                    else name)
            if i in conflict_idx:
                text += "  !CROP"
            _put_label(out, text, (int(xs.min()), int(ys.min()) - 4), colour)

    # THE GROWTH POINTS, before the downscale so they land on true pixels.
    #
    # Drawn as a CROSSHAIR rather than a filled dot: the LEP sits in the middle
    # of a plant, and a solid marker hides the few pixels someone is checking it
    # against. Colour carries the safety verdict, because a point the laser
    # would not fire at is a different fact from one it would, and they are
    # otherwise indistinguishable in a picture.
    for t in (targets or []):
        uv = getattr(t, "lep_uv", None)
        if not uv:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        ok = getattr(t, "safety_status", "") == STATUS_CANDIDATE
        c = C_CANDIDATE if ok else C_ABSTAIN
        cv2.line(out, (u - 11, v), (u - 4, v), c, 2)
        cv2.line(out, (u + 4, v), (u + 11, v), c, 2)
        cv2.line(out, (u, v - 11), (u, v - 4), c, 2)
        cv2.line(out, (u, v + 4), (u, v + 11), c, 2)
        cv2.circle(out, (u, v), 3, c, -1)
        # Depth beside it when there is one. A target with no xyz is not a
        # target the robot can be sent to, and "no depth" looks exactly like
        # "nothing found" unless the picture says which.
        xyz = getattr(t, "xyz_mm", None)
        if labels != "none":
            txt = f"{xyz[2]:.0f}mm" if xyz else "no 3D"
            _put_label(out, txt, (u + 13, v + 5), c)

    # Resize FIRST, then add the key: a legend drawn before a 0.5 downscale
    # comes out at half the font size and is the one thing you cannot zoom into.
    if scale and scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    if show_legend and drawn:
        out = _legend_strip(out, sorted(set(drawn)))
    # AFTER the downscale, like the legend: a banner drawn before a 0.5 resize
    # comes out at half the font size.
    if note:
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(out, note, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    C_ABSTAIN, 1, cv2.LINE_AA)
    return out


def mask_overlap_conflicts(det):
    """Weeds whose mask touches the predicted crop union.

    A blunt instrument compared with safety.check_onion_conflict, which tests
    the laser SPOT against the crop with a margin - but in segmentation mode
    there is no LEP, so there is no spot to test. Shown so the picture is not
    silent about crop proximity, not offered as a safety verdict.
    """
    import numpy as np
    onion = det.onion_safety_mask()
    if onion is None or not onion.any():
        return set()
    return {i for i in det.weed_indices()
            if np.any(np.asarray(det.masks[i]).astype(bool) & onion)}


def predict(cfg=None):
    import cv2
    from common.torch_utils import require_device
    from perception.segmenter import build_segmenter, dedup_detections

    c = dict(CONFIG if cfg is None else cfg)
    device = require_device(c["DEVICE"])
    if c.get("DATASET"):
        frames = split_images(c["DATASET"], c.get("SPLIT", "test"),
                              c.get("LIMIT", 0), c.get("STRIDE", 1),
                              c.get("IMAGES_ROOT", ""))
        print(f"  {len(frames)} frame(s) from split "
              f"{c.get('SPLIT', 'test')!r} of {c['DATASET']}")
    else:
        frames = find_images(c["IMAGES"], c.get("LIMIT", 0),
                             c.get("STRIDE", 1))

    ckpt = Path(c["CHECKPOINT"])
    if not ckpt.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt}\n"
                         f"Train one first: seeweed3d/training/train_model.py")

    out_dir = Path(c["OUT_DIR"])
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    seg = build_segmenter(c["BACKEND"], str(ckpt), conf=c["CONF"],
                          device=device)
    seg.load()
    mode = c.get("MODE", "segmentation")
    pipe = _build_pipeline(c, seg, device) if mode == "full" else None

    # THE CHECKPOINT, NAMED, WITH ITS DATE. "Which model just ran" is the first
    # thing anyone asks of a prediction folder, and every path here is derived
    # from a ROUND edited by hand three files away - so the run has to say it
    # rather than leave it to be reconstructed. The mtime distinguishes two
    # checkpoints that share a filename, which every round's does.
    import datetime as _dt
    st = ckpt.stat()
    print(f"  model : {ckpt}")
    print(f"          {st.st_size / 1e6:.0f} MB, trained "
          f"{_dt.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
    print(f"  {len(frames)} frames | {c['BACKEND']} | conf {c['CONF']} | "
          f"mode {mode}")

    records, counts, n_conflict = [], {}, 0
    n_dup, dup_labels = 0, {}
    want_coco = bool(c.get("WRITE_COCO", True))
    coco_frames, coco_names = [], None
    for path in frames:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  [skip] unreadable: {path}")
            continue

        targets, note = None, None
        if pipe is not None:
            rec, det, conflicts, targets = _run_full(pipe, path, bgr)
            note = frame_note(rec)
        else:
            det = seg(bgr)
            # BEFORE the conflict check and before the overlay, so the picture,
            # the JSON and the counts all describe the same set of instances.
            det, dup = dedup_detections(det, c.get("DEDUP_IOU"))
            n_dup += len(dup)
            for d in dup:
                dup_labels[d["class_name"]] = dup_labels.get(d["class_name"], 0) + 1
            conflicts = mask_overlap_conflicts(det)
            rec = {"instances": [
                {"class_name": det.class_name(i),
                 "score": float(det.scores[i]),
                 "area_px": int(np.count_nonzero(det.masks[i])),
                 "bbox": [float(v) for v in det.boxes[i]],
                 "touches_crop": i in conflicts}
                for i in range(len(det))]}

        for inst in rec["instances"]:
            counts[inst["class_name"]] = counts.get(inst["class_name"], 0) + 1
        n_conflict += len(conflicts)

        vis = draw(bgr, det, conflicts, c.get("OVERLAY_SCALE", 1.0),
                   labels=c.get("LABELS", "class_score"),
                   show_legend=c.get("LEGEND", True), targets=targets,
                   note=note)
        dst = out_dir / "overlays" / f"{path.stem}.png"
        cv2.imwrite(str(dst), vis)
        rec.update({"image": str(path), "overlay": str(dst)})
        records.append(rec)

        # POLYGONISE HERE, NOT AT THE END. The obvious version accumulates the
        # Detections and converts once - but masks are full-frame (N, H, W)
        # bool, so 79 ZED frames at ~40 instances each is 2.7 MB x 3000 = 8 GB
        # held live. That exact shape already killed the self-training run once
        # (bench_mixed rasterising one full mask per annotation). Polygons are
        # a few hundred floats and the frame's masks are freed with the frame.
        if want_coco:
            if coco_names is None:
                coco_names = list(det.names)
            coco_frames.append((path.name, bgr.shape[0], bgr.shape[1],
                                _coco_instances(det)))

    (out_dir / "predictions.json").write_text(
        json.dumps({"checkpoint": str(ckpt), "backend": c["BACKEND"],
                    "conf": c["CONF"], "mode": mode, "frames": records},
                   indent=2), encoding="utf-8")

    if want_coco:
        n_poly = _write_coco(coco_frames, coco_names or list(CLASSES),
                             out_dir, str(ckpt), c["CONF"])
        print(f"\n-> {out_dir / 'instances_default.json'}  "
              f"({n_poly} instance(s), COCO 1.0)")

    print(f"\n  detections per class over {len(records)} frames:")
    for k in sorted(counts):
        print(f"    {k:<28}{counts[k]:>6}")
    if not counts:
        print("    (nothing above the confidence threshold)")
    print(f"  weeds overlapping predicted onion: {n_conflict}")
    if n_dup:
        # RF-DETR is a set-prediction model: two queries can find the same plant
        # and disagree about what it is, so the same mask comes back twice under
        # two labels. Named because a class pair recurring here is a labelling
        # question, not a threshold one.
        total = n_dup + sum(counts.values())
        body = ", ".join(f"{k}={v}" for k, v in
                         sorted(dup_labels.items(), key=lambda kv: -kv[1])[:5])
        print(f"  [i] suppressed {n_dup} duplicate detection(s) of {total} "
              f"({n_dup / total:.0%}) at IoU >= "
              f"{c.get('DEDUP_IOU') if c.get('DEDUP_IOU') is not None else DEFAULT_DEDUP_IOU}"
              f" - dropped labels: {body}")
    print(f"\n-> {out_dir / 'overlays'}\n-> {out_dir / 'predictions.json'}")
    print("\nNo ground truth here, so an empty frame means 'found nothing', "
          "NOT 'nothing was there'.\nFor recall, annotate a held-out session "
          "and run evaluation/eval_seg.py.")
    return records


def _coco_instances(det):
    """One frame's Detections -> plain COCO-shaped dicts, masks dropped.

    Called inside the prediction loop so the full-frame masks die with the
    frame. Everything it returns is small enough to hold for a whole session."""
    from annotation.mine_pool import mask_to_polygons

    out = []
    for i in range(len(det)):
        polys = mask_to_polygons(det.masks[i])
        if not polys:
            # Below mask_to_polygons' min_area_px. Dropping it here rather than
            # writing an empty segmentation is deliberate: CVAT renders a
            # zero-vertex polygon as an invisible, unselectable annotation.
            continue
        xs = [v for p in polys for v in p[0::2]]
        ys = [v for p in polys for v in p[1::2]]
        out.append({"class_name": det.class_name(i), "segmentation": polys,
                    "bbox": [min(xs), min(ys),
                             max(xs) - min(xs), max(ys) - min(ys)],
                    "area": float(np.count_nonzero(det.masks[i])),
                    "score": float(det.scores[i])})
    return out


def _write_coco(frames, names, out_dir, checkpoint, conf):
    """The model's own predictions as COCO, beside predictions.json.

    TWO USES, AND THEY PULL THE SAME WAY. It is what evaluation/bench_mixed.py
    consumes, so the model can be compared against the SAM prelabels on the same
    frames; and it imports straight into CVAT, so a promising frame can be
    corrected without a second inference pass.

    `info.description` says these are MODEL PREDICTIONS. Every prelabeler here
    stamps its own provenance for the same reason: six months on, a COCO file
    with no provenance is indistinguishable from ground truth, and this project
    has already had one silently treated as a different thing than it was."""
    from common.ontology import coco_categories
    from datetime import datetime, timezone

    # The MODEL's class list, not the classes it happened to predict on these
    # frames. A category absent because nothing triggered it still exists in the
    # model's vocabulary, and a COCO whose categories change with the sample is
    # not comparable with the next run's.
    cats = coco_categories(list(names))
    cat_id = {c["name"]: c["id"] for c in cats}

    images, anns, ann_id, n_inst = [], [], 1, 0
    for img_id, (name, h, w, insts) in enumerate(frames, start=1):
        images.append({"id": img_id, "file_name": name, "height": int(h),
                       "width": int(w)})
        for inst in insts:
            if inst["class_name"] not in cat_id:
                continue
            anns.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_id[inst["class_name"]],
                "segmentation": inst["segmentation"], "iscrowd": 0,
                "bbox": inst["bbox"], "area": inst["area"],
                "score": inst["score"]})
            ann_id += 1
            n_inst += 1
    (Path(out_dir) / "instances_default.json").write_text(json.dumps({
        "info": {"description": "SeeWeed3D MODEL PREDICTIONS - not ground truth",
                 "checkpoint": str(checkpoint), "conf": float(conf),
                 "date_created": datetime.now(timezone.utc).isoformat()},
        "licenses": [], "images": images, "annotations": anns,
        "categories": cats}, indent=2), encoding="utf-8")
    return n_inst


def _build_pipeline(c, seg, device):
    from training.config import PipelineConfig
    lep_model = None
    if str(c.get("LEP_CHECKPOINT") or "").strip():
        import torch
        from training.lep_roinet import build_model
        cfg = PipelineConfig()
        blob = torch.load(c["LEP_CHECKPOINT"], weights_only=False)
        lep_model = build_model(cfg.model)
        lep_model.load_state_dict(blob["model"])
        lep_model.to(device)
    else:
        cfg = PipelineConfig()
        print("  [note] no LEP checkpoint - using the hand-engineered growth "
              "point estimator")
    from perception.pipeline import InferencePipeline
    return InferencePipeline(seg, cfg, lep_model=lep_model,
                             torch_device=device)


def frame_note(rec):
    """A whole-frame warning for the overlay, or None.

    Some abstention reasons are not facts about any plant. `crop_protection_
    unavailable` says the MODEL has no onion class, so it is the same statement
    on every instance in every frame - and repeating it per weed both buries it
    and reads as a hazard. Said once, at the top, it is what it is: a reason
    nothing in this run can be approved, no matter how good the masks are."""
    from perception.safety import R_CROP_UNVERIFIABLE
    n = (rec.get("reason_counts") or {}).get(R_CROP_UNVERIFIABLE, 0)
    if n and n == len(rec.get("instances") or []):
        return "NO CROP MASK - model cannot predict onion_plant; all abstained"
    return None


def _run_full(pipe, path, bgr):
    """One frame through the deployed pipeline. Falls back to segmentation for
    this frame if its depth or calibration is missing, rather than aborting a
    whole run over one bad frame."""
    from common.depth_utils import load_depth_mm, load_intrinsics
    session = _session_of(path)
    depth = valid = K = None
    if session is not None:
        dpath = session / "depth" / path.name
        calib = session / "meta" / "calibration.json"
        if dpath.exists() and calib.exists():
            depth, valid = load_depth_mm(dpath)
            K = load_intrinsics(session)
    if depth is None:
        print(f"  [note] no depth/calibration for {path.name} - "
              f"segmentation only for this frame")

    # ONE segmentation. Calling pipe.segmenter(bgr) again for the overlay
    # doubled the cost of the expensive stage and drew the picture from a
    # different inference than the record - and it skipped the duplicate
    # suppression the pipeline applies, so the overlay could show two masks on
    # a plant the record had already merged into one.
    res, det = pipe.run_with_detections(
        bgr, depth, valid, K,
        session_id=session.name if session else "", frame_id=path.stem)
    # NAMED reasons, not a substring match on "crop". `crop_protection_
    # unavailable` contains "crop" and means the opposite of what the !CROP tag
    # says: there is no crop mask AT ALL because the model has no onion class.
    # Matching loosely put a hazard marker on every weed in a frame with no
    # onions in it, drawn by a model that cannot predict one - a capability gap
    # dressed as a laser-on-the-crop warning, which is the most alarming thing
    # this overlay can say.
    from perception.safety import R_ONION, R_ONION_CONFLICT
    conflicts = {t.instance_index for t in res.abstentions
                 if R_ONION_CONFLICT in t.rejection_reasons
                 or R_ONION in t.rejection_reasons}
    rec = {
        "n_candidates": len(res.candidates),
        "n_abstentions": len(res.abstentions),
        "reason_counts": res.reason_counts(),
        "timings_ms": res.timings_ms,
        "instances": [
            {"class_name": t.class_name, "score": float(t.class_confidence),
             "area_px": int(t.mask_area_px),
             "safety_status": t.safety_status,
             "lep_uv": t.lep_uv, "xyz_mm": t.xyz_mm,
             "xyz_sigma_mm": t.xyz_sigma_mm,
             "rejection_reasons": list(t.rejection_reasons)}
            for t in res.targets],
    }
    return rec, det, conflicts, res.targets


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images")
    p.add_argument("--dataset", help="a built dataset's OUT_DIR; with --split, "
                                     "runs on that split's frames instead of a "
                                     "folder")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--checkpoint")
    p.add_argument("--out")
    p.add_argument("--backend", choices=["maskrcnn", "rfdetr"])
    p.add_argument("--mode", choices=["segmentation", "full"])
    p.add_argument("--device")
    p.add_argument("--conf", type=float)
    p.add_argument("--limit", type=int)
    p.add_argument("--stride", type=int)
    p.add_argument("--labels", choices=["class_score", "class", "none"])
    p.add_argument("--no-legend", action="store_true")
    a = p.parse_args(argv)

    c = dict(CONFIG)
    for flag, key in (("images", "IMAGES"), ("checkpoint", "CHECKPOINT"),
                      ("dataset", "DATASET"), ("split", "SPLIT"),
                      ("out", "OUT_DIR"), ("backend", "BACKEND"),
                      ("mode", "MODE"), ("device", "DEVICE"),
                      ("conf", "CONF"), ("limit", "LIMIT"),
                      ("stride", "STRIDE"), ("labels", "LABELS")):
        v = getattr(a, flag)
        if v is not None:
            c[key] = v
    if a.no_legend:
        c["LEGEND"] = False
    return predict(c)


if __name__ == "__main__":
    main()
