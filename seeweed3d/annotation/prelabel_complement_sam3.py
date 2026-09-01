#!/usr/bin/env python3
"""
SeeWeed3D - COMPLEMENT prelabeling: onions from a model, weeds from what is left
================================================================================
Runs a trained ONION model over a mixed scene and labels the vegetation it did
NOT claim. Onion-only sessions are the cheapest thing to annotate - one class,
no ambiguity - so a model trained on them can be spent on the expensive scenes.

    python seeweed3d/annotation/prelabel_complement_sam3.py

WHY THIS IS A PRELABELER AND MUST NEVER BE A DEPLOYMENT RULE
-------------------------------------------------------------
"Everything that is not onion is a weed" is arithmetic, and the arithmetic is
right. What is wrong is the direction the errors point.

When the onion model MISSES an onion - occluded, an unusual growth stage, the
edge of the frame, motion blur - those pixels are vegetation, no onion mask
covers them, and the complement calls them weed. Deployed, that is a laser
fired at the crop. And it is not an exotic failure: a missed detection is the
single most common thing a detector does, recall never reaches 1.0, and every
onion in the residue becomes a target.

A two-class model's failure mode is "unsure", which a confidence threshold can
act on. The complement has no such handle - uncertainty defaults to weed, and
weed means fire. So the complement is used only where a human sees the output
before anything acts on it, and it emits a THIRD outcome for the cases it
cannot stand behind:

    onion_plant     vegetation confidently claimed by the model
    <weed class>    vegetation far from any onion, on confident ground
    ignore_region   everything else - near an onion, low score, ambiguous

`ignore_region` is already in the ontology, and the intent is that an uncertain
region costs an annotator one look and costs the model nothing - the error the
complement makes most often landing in a bucket that is inert.

THAT BUCKET IS NOT INERT YET. Ignore regions reach seg_manifest.json and no
further: both trainers iterate `instances` only, so those pixels are supervised
as BACKGROUND - over a real plant, the exact lesson the region exists to
prevent. prepare_dataset.py prints a warning whenever a build contains any.
Until a trainer honours them, treat this prelabeler's output as needing a human
pass over the ignore regions rather than as safe to train on directly.

WHY IT IS STILL BETTER THAN THE SAM 3 MIXED PRELABELER, WHERE IT APPLIES
-------------------------------------------------------------------------
`prelabel_mixed_sam3.py` proposes one homogeneous `plant` class because shape
cannot tell an onion from a grass weed, so every shape needs its class assigned
by hand. Your own checkpoint knows the ontology. Where it is confident, the
class arrives correct and the CVAT pass is mask correction only.

The two are complements rather than rivals: run this where you have a trained
onion model, and use the mixed prelabeler for the regions this one abstains on.

DEPTH, WHERE IT EXISTS
----------------------
Height above the local soil surface is applied exactly as in the mixed
prelabeler - see perception/ground.py. It removes flat, green-tinted mineral
from the weed side before anything is written, which matters more here than
there: a phantom weed in a mixed scene is a shape an annotator has to notice is
wrong, not merely delete.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation.prelabel_weeds_sam3 import (link_or_copy,  # noqa: E402
                                            mask_polygons, pool_frames,
                                            print_pool_report)
from common.ontology import (CROP_CLASS, IGNORE_LABEL,  # noqa: E402
                             coco_categories, prelabel_cvat_labels)
from common.progress import Progress  # noqa: E402
from common.vegetation import vegetation_mask as _vegetation_mask  # noqa: E402
from common.vegetation import white_balance as _white_balance  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

CONFIG = {
    # -- The model that finds the onions ---------------------------------------
    # A checkpoint trained on ONION-ONLY sessions. It does not need to know what
    # a weed is; that is the point.
    "CHECKPOINT": r"E:\Dataset_Vidalia\onions_20260108_1",
    "BACKEND": "rfdetr",              # "maskrcnn" | "rfdetr"
    "DEVICE": "cuda",

    # -- Where ------------------------------------------------------------------
    "DATASET_ROOT": r"E:\Dataset_Vidalia\Mixed_1",
    "OUTPUT_SUBDIR": "auto_labels_complement",
    "ONLY_SESSIONS": [],              # the MIXED sessions
    "ONLY_FRAMES": {},                # same syntax as everywhere else
    "LIMIT_PER_SESSION": 20,          # trial first, then None
    "CVAT_READY_SUBDIR": "cvat_ready",

    # -- The two confidence bands ----------------------------------------------
    # ONION_CONF is what counts as "this is the crop". Set it LOW. A generous
    # onion mask costs an annotator one correction; a stingy one puts crop
    # tissue on the weed side, which is the error this whole module is arranged
    # around not making.
    "ONION_CONF": 0.15,

    # Vegetation within this many pixels of a claimed onion is NOT called weed.
    # A missed leaf of a detected onion is nearly always adjacent to the part
    # that was detected, so the halo catches the most common failure directly.
    # It is measured in pixels; with depth and calibration available the mm
    # equivalent below is used instead.
    "ONION_HALO_PX": 40,
    "ONION_HALO_MM": 30.0,            # used when depth + calibration exist

    # -- What may be called weed -----------------------------------------------
    # Provisional class for confident non-onion vegetation. "other_weed" makes
    # no species claim, which shape cannot support anyway.
    "WEED_CLASS": "other_weed",
    "WEED_MIN_AREA_PX": 250,
    "WEED_MIN_VEG_SCORE": 0.90,

    # Vegetation the model neither claimed nor left clearly alone. Written as
    # ignore_region - which no trainer honours yet, so those pixels currently
    # train as background. See the module docstring before building on it.
    "EMIT_IGNORE": True,
    "IGNORE_MIN_AREA_PX": 150,

    # -- Vegetation prior (identical to the other prelabelers) ------------------
    "WHITE_BALANCE": True,
    "WB_CAST_RATIO": 1.15,
    "EXG_THRESHOLD": 0.05,
    "VEG_MIN_SATURATION": 40,
    "VEG_MORPH_KERNEL": 3,
    "VEG_MIN_COMPONENT_PX": 150,

    # -- Depth (see perception/ground.py and docs/depth_assisted_masking.md) ----
    "USE_DEPTH_HEIGHT": "auto",
    "HEIGHT_MIN_MM": 6.0,
    "HEIGHT_MIN_MEASURED_FRAC": 0.25,
    "HEIGHT_PERCENTILE": 75.0,
    "GROUND_TILE_PX": 32,
    "GROUND_PERCENTILE": 80.0,
    "DEPTH_MIN_CONFIDENCE": 0.30,

    # -- Polygons and previews --------------------------------------------------
    "POLY_APPROX_EPS_FRAC": 0.010,
    "POLY_APPROX_EPS_MIN": 0.5,
    "POLY_APPROX_EPS_MAX": 1.5,
    "POLY_MIN_PART_AREA_PX": 60,
    "POLY_ALL_PARTS": False,
    "SAVE_PREVIEWS": True,
    "PREVIEW_SCALE": 0.5,
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

#: Preview colours. Deliberately not the ontology palette: the question these
#: previews answer is "which of the three buckets did this land in", and the
#: class colours would answer a different one.
BUCKET_BGR = {"onion": (255, 200, 0), "weed": (0, 0, 255),
              "ignore": (140, 140, 140)}


def vegetation(bgr, cfg):
    return _vegetation_mask(bgr, cfg["EXG_THRESHOLD"], cfg["VEG_MIN_SATURATION"],
                            cfg["VEG_MORPH_KERNEL"], cfg["VEG_MIN_COMPONENT_PX"])


def onion_union(det, cfg):
    """Every pixel the model called crop, at ONION_CONF.

    Reads the CROP class by NAME rather than by index. A model trained on a
    reduced class set has its own ordering, and an index assumed here would
    silently point crop safety at whatever class happened to sit in that slot."""
    if det is None or not len(det):
        return None
    h, w = np.asarray(det.masks[0]).shape[:2]
    out = np.zeros((h, w), bool)
    found = False
    for i in range(len(det)):
        if det.class_name(i) != CROP_CLASS:
            continue
        found = True
        if float(det.scores[i]) >= cfg["ONION_CONF"]:
            out |= np.asarray(det.masks[i]).astype(bool)
    return out if found or True else None


def halo_px(cfg, depth_mm=None, fx=None, mask=None):
    """The uncertainty band around a claimed onion, in pixels.

    Expressed in millimetres where depth and calibration allow, so the band is
    a fixed distance on the ground rather than a fixed distance in the image -
    which is what stops it meaning different things at different boom heights."""
    if depth_mm is None or fx is None or mask is None or not mask.any():
        return int(cfg["ONION_HALO_PX"])
    d = np.asarray(depth_mm, np.float32)[mask]
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 8:
        return int(cfg["ONION_HALO_PX"])
    mm_per_px = float(np.median(d)) / float(fx)
    return max(1, int(round(float(cfg["ONION_HALO_MM"]) / max(1e-6, mm_per_px))))


def partition(veg, onions, cfg, score=None, halo=None):
    """Split vegetation into (onion, weed, ignore).

    The halo is the crux. A missed leaf of a DETECTED onion is nearly always
    adjacent to the part that was detected, so widening the onion region before
    taking the complement removes the most common failure directly rather than
    hoping a confidence threshold catches it. Everything inside the band that
    the model did not claim outright becomes ignore, not weed."""
    onion = np.zeros(veg.shape, bool) if onions is None else (onions & veg)
    band = int(cfg["ONION_HALO_PX"] if halo is None else halo)
    if onion.any() and band > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
        near = cv2.dilate(onion.astype(np.uint8), k).astype(bool)
    else:
        near = onion
    rest = veg & ~near

    weed = np.zeros(veg.shape, bool)
    ignore = (veg & near & ~onion)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        rest.astype(np.uint8), 8)
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg["IGNORE_MIN_AREA_PX"]:
            continue                       # below either floor: soil speckle
        confident = area >= cfg["WEED_MIN_AREA_PX"]
        if confident and score is not None:
            confident = float(score[comp].mean()) >= cfg["WEED_MIN_VEG_SCORE"]
        if confident:
            weed |= comp
        else:
            ignore |= comp
    return onion, weed, ignore


def instances_from(mask, cfg, min_area):
    """Connected components of one bucket as separate instances."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        m = labels == i
        x, y, w, h = (int(stats[i, c]) for c in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                       cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        out.append({"mask": m, "area_px": int(m.sum()), "bbox": [x, y, w, h]})
    return out


def overlay(bgr, buckets, scale, cfg=None):
    """Preview coloured by BUCKET, not by class - the question is which of the
    three a region landed in, and a class palette answers a different one."""
    vis = bgr.copy()
    for name, insts in buckets.items():
        col = BUCKET_BGR[name]
        for inst in insts:
            cnts, _ = cv2.findContours(inst["mask"].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
    return cv2.resize(vis, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale != 1.0 else vis


class ComplementCoco:
    """COCO with the real ontology ids, plus ignore_region."""

    def __init__(self, weed_class):
        self.images, self.anns = [], []
        self.classes = [CROP_CLASS, weed_class]
        self.categories = coco_categories(self.classes)
        self._cat = {c["name"]: c["id"] for c in self.categories}
        # ignore_region is not an ontology CLASS - it is an annotation-only
        # label the dataset builder excludes from training - so it gets an id
        # that cannot collide with one.
        self._ignore_id = max(c["id"] for c in self.categories) + 1000
        self.categories.append({"id": self._ignore_id, "name": IGNORE_LABEL,
                                "supercategory": "annotation"})
        self._cat[IGNORE_LABEL] = self._ignore_id
        self._img = self._ann = 0

    def add_image(self, file_name, h, w):
        self._img += 1
        self.images.append({"id": self._img, "file_name": file_name,
                            "height": h, "width": w})
        return self._img

    def add(self, image_id, cls, polygons, bbox, area_px):
        self._ann += 1
        self.anns.append({"id": self._ann, "image_id": image_id,
                          "category_id": self._cat[cls],
                          "segmentation": polygons, "area": float(area_px),
                          "bbox": [float(v) for v in bbox], "iscrowd": 0})

    def dump(self, path):
        Path(path).write_text(json.dumps({
            "info": {"description": "SeeWeed3D complement prelabels - onions "
                                    "from a model, weeds from the remainder, "
                                    "uncertainty as ignore_region",
                     "date_created": datetime.now(timezone.utc).isoformat()},
            "licenses": [], "images": self.images, "annotations": self.anns,
            "categories": self.categories}, indent=2), encoding="utf-8")


def prelabel_session(sid, session_dir, out_root, cfg, seg):
    """One mixed session through the complement. Returns per-session stats."""
    from common.vegetation import vegetation_score
    from perception import ground as gr

    frames = pool_frames(session_dir)
    spec = (cfg.get("ONLY_FRAMES") or {}).get(sid)
    if spec:
        from common.frame_spec import select_filenames
        frames = select_filenames(frames, spec)
        if not frames:
            print(f"  [{sid}] ONLY_FRAMES {spec} matched no pool frame")
            return None
    if cfg["LIMIT_PER_SESSION"]:
        frames = frames[:cfg["LIMIT_PER_SESSION"]]
    print_pool_report(sid, session_dir, len(frames))
    if not frames:
        return None

    want = cfg.get("USE_DEPTH_HEIGHT", "auto")
    use_depth = bool(want) and gr.has_metric_depth(session_dir)
    if want is True and not use_depth:
        sys.exit(f"ERROR: [{sid}] USE_DEPTH_HEIGHT is True but depth_kind is "
                 f"{gr.session_depth_kind(session_dir)!r}, not 'metric'.")
    fx = fy = polarity = None
    if use_depth:
        fx, fy = gr.calibration(session_dir)
        polarity = gr.confidence_polarity(session_dir)
        print(f"  [{sid}] depth: metric | height veto on")

    out = out_root / sid
    (out / "masks").mkdir(parents=True, exist_ok=True)
    cvat_dir = out / cfg["CVAT_READY_SUBDIR"]
    cvat_dir.mkdir(parents=True, exist_ok=True)
    if cfg["SAVE_PREVIEWS"]:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    coco = ComplementCoco(cfg["WEED_CLASS"])
    stats = {"frames": 0, "onion": 0, "weed": 0, "ignore": 0,
             "veg_px": 0, "onion_px": 0, "weed_px": 0, "ignore_px": 0,
             "no_onion_frames": 0}
    rows = []
    prog = Progress(len(frames), f"[{sid}]", unit="frames")

    for fn in frames:
        rgb_path = session_dir / "rgb" / fn
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            prog.update(note="missing frame")
            continue
        proc = (_white_balance(bgr, cfg["WB_CAST_RATIO"])
                if cfg["WHITE_BALANCE"] else bgr)
        veg = vegetation(proc, cfg)
        score = vegetation_score(proc, cfg["EXG_THRESHOLD"],
                                 cfg["VEG_MIN_SATURATION"])

        depth_mm = conf_img = None
        if use_depth:
            dpath = session_dir / "depth" / fn
            if dpath.exists():
                depth_mm = gr.load_depth_mm(dpath)
            cpath = session_dir / "conf" / fn
            if depth_mm is not None and polarity and cpath.exists():
                conf_img = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)

        det = seg(proc)
        onions = onion_union(det, cfg)
        band = halo_px(cfg, depth_mm, fx, onions if onions is not None else None)
        onion, weed, ignore = partition(veg, onions, cfg, score, halo=band)
        if onions is None or not onions.any():
            # No crop found at all. In a MIXED scene that is far more likely to
            # be a model failure than a frame with no onions in it, and the
            # complement would then call every plant a weed - so the whole
            # frame goes to ignore and a human decides.
            stats["no_onion_frames"] += 1
            ignore = ignore | weed
            weed = np.zeros(veg.shape, bool)

        buckets = {
            "onion": instances_from(onion, cfg, cfg["IGNORE_MIN_AREA_PX"]),
            "weed": instances_from(weed, cfg, cfg["WEED_MIN_AREA_PX"]),
            "ignore": (instances_from(ignore, cfg, cfg["IGNORE_MIN_AREA_PX"])
                       if cfg["EMIT_IGNORE"] else []),
        }

        if depth_mm is not None:
            # Flat, green-tinted mineral removed from the WEED side before
            # anything is written. A phantom weed in a mixed scene is worse
            # than in a weed-only one: it is a shape an annotator has to notice
            # is wrong, not merely one to delete.
            from annotation.prelabel_mixed_sam3 import height_veto
            buckets["weed"], _ = height_veto(buckets["weed"], veg, depth_mm,
                                             cfg, conf=conf_img,
                                             polarity=polarity, fx=fx, fy=fy)

        link_or_copy(rgb_path, cvat_dir / fn)
        img_id = coco.add_image(fn, bgr.shape[0], bgr.shape[1])
        label_of = {"onion": CROP_CLASS, "weed": cfg["WEED_CLASS"],
                    "ignore": IGNORE_LABEL}
        union = np.zeros(bgr.shape[:2], bool)
        for name, insts in buckets.items():
            for k, inst in enumerate(insts):
                polys = mask_polygons(inst["mask"], cfg)
                if not polys:
                    continue
                coco.add(img_id, label_of[name], polys, inst["bbox"],
                         inst["area_px"])
                union |= inst["mask"]
                rows.append({"session_id": sid, "filename": fn, "bucket": name,
                             "instance_idx": k, "area_px": inst["area_px"],
                             "height_mm": inst.get("height_mm", ""),
                             "area_mm2": inst.get("area_mm2", "")})
            stats[name] += len(insts)

        stats["veg_px"] += int(veg.sum())
        stats["onion_px"] += int(onion.sum())
        stats["weed_px"] += int(weed.sum())
        stats["ignore_px"] += int(ignore.sum())
        stats["frames"] += 1
        cv2.imwrite(str(out / "masks" / fn), (union.astype(np.uint8) * 255))
        if cfg["SAVE_PREVIEWS"]:
            cv2.imwrite(str(out / "preview" / fn.replace(".png", ".jpg")),
                        overlay(proc, buckets, cfg["PREVIEW_SCALE"], cfg))
        prog.update(note=f"{stats['onion']}on {stats['weed']}wd "
                         f"{stats['ignore']}ig")

    prog.close(note=f"{stats['onion']}on {stats['weed']}wd {stats['ignore']}ig")
    coco.dump(out / "instances_default.json")
    (out / "complement_cvat_labels.json").write_text(
        json.dumps(prelabel_cvat_labels([CROP_CLASS, cfg["WEED_CLASS"]]),
                   indent=2), encoding="utf-8")
    if rows:
        with open(out / "instances.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    veg_px = max(1, stats["veg_px"])
    print(f"  [{sid}] {stats['frames']} frames | {stats['onion']} onion, "
          f"{stats['weed']} weed, {stats['ignore']} ignore")
    print(f"      vegetation: {stats['onion_px'] / veg_px:.0%} onion | "
          f"{stats['weed_px'] / veg_px:.0%} weed | "
          f"{stats['ignore_px'] / veg_px:.0%} uncertain")
    if stats["no_onion_frames"]:
        print(f"  [!] {stats['no_onion_frames']} frame(s) had NO onion "
              f"detected at all. In a mixed scene that is far more likely a "
              f"model failure than a frame without onions, so every plant in "
              f"them was written as {IGNORE_LABEL} rather than as weed.")
    if stats["weed_px"] > stats["veg_px"] * 0.9:
        print(f"  [!] over 90% of vegetation was called weed. Check the "
              f"previews before importing: an onion model that is failing on "
              f"this session looks exactly like a session full of weeds.")
    print(f"      -> {out}")
    return stats


def main():
    cfg = CONFIG
    from common.dataset_paths import require_sessions_root
    from common.torch_utils import require_device
    from perception.segmenter import build_segmenter

    root = Path(cfg["DATASET_ROOT"])
    sessions_root = require_sessions_root(root)
    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    if cfg["ONLY_SESSIONS"]:
        sids = [s for s in sids if s in cfg["ONLY_SESSIONS"]]
    if not sids:
        sys.exit("No sessions selected. Set ONLY_SESSIONS to your mixed sessions.")

    ckpt = Path(cfg["CHECKPOINT"])
    if not ckpt.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt}\n"
                 f"Train an ONION model first - this module has nothing to do "
                 f"without one. See RUNBOOK section 6b.")

    print(f"Complement prelabeling on {len(sids)} mixed session(s).")
    print(f"  Onions from {ckpt.name}; everything else is weed or "
          f"{IGNORE_LABEL}.")
    seg = build_segmenter(cfg["BACKEND"], str(ckpt), conf=cfg["ONION_CONF"],
                          device=require_device(cfg["DEVICE"]))
    seg.load()

    out_root = root / cfg["OUTPUT_SUBDIR"]
    for sid in sids:
        prelabel_session(sid, sessions_root / sid, out_root, cfg, seg)
    print(f"\nDone -> {out_root}")
    print(f"Import instances_default.json into CVAT. Anything labelled "
          f"{IGNORE_LABEL} is a region the model could not stand behind - "
          f"decide those first, they are where the crop/weed boundary lives.")
    print(f"  Decide them, do not leave them: no trainer honours "
          f"{IGNORE_LABEL} yet, so any left\n"
          f"  in place train as bare ground rather than as 'do not learn "
          f"from this'.")


if __name__ == "__main__":
    main()
