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
from common.ontology import CROP_CLASS                        # noqa: E402

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
         show_legend=True):
    """Tint and outline every instance, ONE COLOUR PER CLASS.

    Crop proximity is drawn as an extra WHITE outline rather than by recolouring
    the instance. The class and the hazard are two different facts, and
    overwriting one with the other loses information at the moment it matters
    most - you would no longer be able to see WHICH weed is sitting on an onion.
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

    # Resize FIRST, then add the key: a legend drawn before a 0.5 downscale
    # comes out at half the font size and is the one thing you cannot zoom into.
    if scale and scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    if show_legend and drawn:
        out = _legend_strip(out, sorted(set(drawn)))
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
    from perception.segmenter import build_segmenter

    c = dict(CONFIG if cfg is None else cfg)
    device = require_device(c["DEVICE"])
    frames = find_images(c["IMAGES"], c.get("LIMIT", 0), c.get("STRIDE", 1))

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

    print(f"  {len(frames)} frames | {c['BACKEND']} | conf {c['CONF']} | "
          f"mode {mode}")

    records, counts, n_conflict = [], {}, 0
    for path in frames:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  [skip] unreadable: {path}")
            continue

        if pipe is not None:
            rec, det, conflicts = _run_full(pipe, path, bgr)
        else:
            det = seg(bgr)
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
                   show_legend=c.get("LEGEND", True))
        dst = out_dir / "overlays" / f"{path.stem}.png"
        cv2.imwrite(str(dst), vis)
        rec.update({"image": str(path), "overlay": str(dst)})
        records.append(rec)

    (out_dir / "predictions.json").write_text(
        json.dumps({"checkpoint": str(ckpt), "backend": c["BACKEND"],
                    "conf": c["CONF"], "mode": mode, "frames": records},
                   indent=2), encoding="utf-8")

    print(f"\n  detections per class over {len(records)} frames:")
    for k in sorted(counts):
        print(f"    {k:<28}{counts[k]:>6}")
    if not counts:
        print("    (nothing above the confidence threshold)")
    print(f"  weeds overlapping predicted onion: {n_conflict}")
    print(f"\n-> {out_dir / 'overlays'}\n-> {out_dir / 'predictions.json'}")
    print("\nNo ground truth here, so an empty frame means 'found nothing', "
          "NOT 'nothing was there'.\nFor recall, annotate a held-out session "
          "and run evaluation/eval_seg.py.")
    return records


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

    res = pipe.run(bgr, depth, valid, K,
                   session_id=session.name if session else "",
                   frame_id=path.stem)
    det = pipe.segmenter(bgr)
    conflicts = {t.instance_index for t in res.abstentions
                 if any("onion" in r or "crop" in r
                        for r in t.rejection_reasons)}
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
    return rec, det, conflicts


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images")
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
