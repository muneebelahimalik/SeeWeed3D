#!/usr/bin/env python3
"""
SeeWeed3D - visual results report for Stage A.

    python -m seeweed3d.evaluation.report \
        --checkpoint  D:/runs/seg_v1/best.pt \
        --dataset     D:/training/subset45 \
        --images-root D:/Dataset_Vidalia/sessions \
        --split val --device cuda

Writes ONE self-contained HTML file (images embedded, no sidecar folder) plus
the machine-readable per-frame JSON behind it.

WHY THIS EXISTS SEPARATELY FROM eval_seg.py
-------------------------------------------
eval_seg answers "how good is it" in numbers. Those numbers say nothing about
WHICH plants the model gets wrong, and on a small agricultural dataset that is
the only question that tells you what to annotate next. `small_weed_recall =
0.28` is a fact; a page of the actual missed cotyledons is a plan.

WHAT IT SHOWS, AND WHY EACH PART
--------------------------------
1. RECALL BY INSTANCE SIZE. The single most useful plot for a weeder: recall
   bucketed by ground-truth mask area. A weeder that misses small weeds misses
   the ones worth killing, because a weed is cheapest to destroy before it
   establishes. One aggregate recall hides where the cliff is.

2. PER-FRAME PANELS, WORST FIRST. Ground truth beside prediction, with each
   instance colour-coded by OUTCOME rather than by class:
       green   matched (true positive)
       RED     ground truth the model missed  <- the failures
       magenta predicted where there is nothing
   Sorting by miss count puts the informative frames on the first screen; a
   report you have to scroll to reach the problem is a report nobody reads.

3. MISSED-INSTANCE GALLERY, SMALLEST FIRST. Tight crops of individual missed
   plants. This is what tells you whether a miss is a genuinely hard cotyledon,
   a mislabelled annotation, or a plant that is simply too few pixels at this
   input resolution - three problems with three different fixes.

4. CROP SAFETY, ALWAYS SEPARATE. Missed crop pixels are never folded into an
   averaged score. A model can post excellent mAP and still miss the onion that
   matters.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, CROP_CLASS  # noqa: E402
from evaluation.metrics import match_instances  # noqa: E402

# Outcome colours, BGR. Deliberately NOT the class colours: this view answers
# "what went wrong", and re-using class colours would make a missed primrose
# and a correct primrose look the same.
C_TP = (80, 200, 80)        # green   - matched
C_FN = (60, 60, 235)        # red     - missed ground truth
C_FP = (220, 80, 220)       # magenta - predicted, nothing there
C_CROP = (0, 165, 255)      # orange  - the crop, drawn thicker

#: Area buckets in px for the recall-by-size table. The lowest bucket is the
#: cotyledon end of the range, which is where a weeder earns its keep.
AREA_BUCKETS = [0, 250, 500, 1000, 2000, 4000, 8000, 16000, float("inf")]


def bucket_label(lo, hi):
    return f"{int(lo)}-{'inf' if hi == float('inf') else int(hi)}"


def analyse_frame(pred_masks, pred_names, pred_scores, gt_masks, gt_names,
                  iou_threshold=0.5):
    """Per-instance outcome for one frame.

    Returns (matches, missed_gt_idx, false_pos_idx). Matching is the same
    class-aware, highest-IoU-first greedy rule the metrics module uses, so this
    report can never disagree with the score table beside it."""
    matches, unmatched_pred, unmatched_gt = match_instances(
        pred_masks, pred_names, gt_masks, gt_names, iou_threshold)
    return matches, unmatched_gt, unmatched_pred


def recall_by_size(records, buckets=None):
    """Recall against ground-truth instance area, weeds only.

    The crop is excluded because crop recall is a safety question with its own
    section - averaging it in here would let good weed recall mask a missed
    onion."""
    buckets = buckets or AREA_BUCKETS
    rows = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        n = hit = 0
        for r in records:
            for g in r["gt"]:
                if g["is_crop"] or not (lo <= g["area_px"] < hi):
                    continue
                n += 1
                hit += int(g["matched"])
        rows.append({"range_px": bucket_label(lo, hi), "n_gt": n,
                     "n_found": hit,
                     "recall": (hit / n) if n else None})
    return rows


def draw_outcomes(bgr, masks, names, outcomes, alpha=0.35):
    """Tint and outline every instance by OUTCOME."""
    import cv2
    out = bgr.copy()
    for m, name, kind in zip(masks, names, outcomes):
        m = np.asarray(m).astype(bool)
        if not m.any():
            continue
        colour = {"tp": C_TP, "fn": C_FN, "fp": C_FP}[kind]
        if name == CROP_CLASS:
            colour = C_CROP
        tint = np.zeros_like(out)
        tint[m] = colour
        out = cv2.addWeighted(out, 1.0, tint, alpha, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        # A miss is drawn thickest. It is the thing you are looking for.
        w = 4 if kind == "fn" else (3 if name == CROP_CLASS else 1)
        cv2.drawContours(out, cnts, -1, colour, w)
    return out


def _label(img, text):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return img


def frame_panel(bgr, gt_masks, gt_names, gt_matched, pred_masks, pred_names,
                pred_matched, max_width=1500):
    """[ ground truth | prediction ] with outcome colouring on both sides.

    Two panels rather than one blended image: overlapping tints of a correct
    and an incorrect mask are indistinguishable from a single mask of a third
    colour, which is exactly the case being looked for."""
    import cv2
    left = draw_outcomes(bgr, gt_masks, gt_names,
                         ["tp" if m else "fn" for m in gt_matched])
    right = draw_outcomes(bgr, pred_masks, pred_names,
                          ["tp" if m else "fp" for m in pred_matched])
    n_miss = sum(1 for m in gt_matched if not m)
    n_fp = sum(1 for m in pred_matched if not m)
    _label(left, f"GROUND TRUTH  ({len(gt_masks)} instances, "
                 f"{n_miss} MISSED shown red)")
    _label(right, f"PREDICTION  ({len(pred_masks)} instances, "
                  f"{n_fp} false positives shown magenta)")
    h = min(left.shape[0], right.shape[0])
    pair = np.hstack([left[:h], np.full((h, 6, 3), 255, np.uint8), right[:h]])
    if pair.shape[1] > max_width:
        s = max_width / pair.shape[1]
        pair = cv2.resize(pair, (max_width, max(1, int(pair.shape[0] * s))))
    return pair


def crop_around(bgr, mask, pad_ratio=1.6, min_size=96):
    """A tight, square-ish crop centred on one instance, outlined."""
    import cv2
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if not len(xs):
        return None
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    half = max(min_size / 2.0,
               pad_ratio * max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0)
    h, w = bgr.shape[:2]
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    if x1 <= x0 or y1 <= y0:
        return None
    out = bgr[y0:y1, x0:x1].copy()
    sub = np.asarray(mask).astype(np.uint8)[y0:y1, x0:x1]
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, C_FN, 2)
    if out.shape[0] < 160:
        s = 160 / max(1, out.shape[0])
        out = cv2.resize(out, (max(1, int(out.shape[1] * s)), 160),
                         interpolation=cv2.INTER_NEAREST)
    return out


def png_data_uri(bgr, quality=82):
    """Embed as JPEG so a 30-frame report stays a few MB rather than hundreds."""
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def collect(checkpoint, dataset_dir, images_root, split="val", device="cpu",
            conf=0.5, min_area_px=16, mask_threshold=0.5, iou_threshold=0.5,
            backend="maskrcnn"):
    """Run the model over a split and record per-instance outcomes."""
    import cv2
    from common.torch_utils import require_device
    from perception.segmenter import build_segmenter
    from training.seg_dataset import polygons_to_mask, resolve_image

    device = require_device(device)
    man_path = Path(dataset_dir) / "seg_manifest.json"
    if not man_path.exists():
        raise SystemExit(f"ERROR: {man_path} not found. Build the dataset "
                         f"with prepare_dataset/make_dataset first.")
    doc = json.loads(man_path.read_text(encoding="utf-8"))
    frames = [f for f in doc["frames"] if f.get("split") == split]
    if not frames:
        raise SystemExit(f"ERROR: split {split!r} has no frames.")

    seg = build_segmenter(backend, checkpoint, conf=conf, device=device,
                          **({"mask_threshold": mask_threshold}
                             if backend == "maskrcnn" else {}))
    seg.load()
    classes = list(seg.classes or doc.get("classes") or CLASSES)
    manifest_classes = list(doc.get("classes") or CLASSES)
    if classes != manifest_classes:
        raise SystemExit(
            f"ERROR: checkpoint classes {classes} do not match dataset "
            f"classes {manifest_classes}. Comparing across a class-list "
            f"mismatch silently compares different labels.")

    root = images_root or doc.get("images_root") or "."
    records = []
    for rec in frames:
        try:
            path = resolve_image(rec["image_path"], root, rec.get("session_id"))
        except FileNotFoundError as e:
            raise SystemExit(f"ERROR: {e}")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"ERROR: cannot read {path}")
        h, w = bgr.shape[:2]

        gt_masks, gt_names = [], []
        for inst in rec["instances"]:
            m = polygons_to_mask(inst["polygons"], h, w).astype(bool)
            if int(m.sum()) >= min_area_px:
                gt_masks.append(m)
                gt_names.append(inst["class_name"])

        det = seg(bgr)
        pred_masks = [det.masks[i].astype(bool) for i in range(len(det))]
        pred_names = [det.class_name(i) for i in range(len(det))]
        pred_scores = [float(det.scores[i]) for i in range(len(det))]

        matches, missed, false_pos = analyse_frame(
            pred_masks, pred_names, pred_scores, gt_masks, gt_names,
            iou_threshold)
        matched_gt = {m["gt"] for m in matches}
        matched_pred = {m["pred"] for m in matches}
        iou_of_gt = {m["gt"]: m["iou"] for m in matches}

        records.append({
            "item_id": rec["item_id"], "session_id": rec.get("session_id"),
            "path": str(path), "width": w, "height": h,
            "n_gt": len(gt_masks), "n_pred": len(pred_masks),
            "n_matched": len(matches), "n_missed": len(missed),
            "n_false_pos": len(false_pos),
            "gt": [{"class_name": gt_names[i],
                    "area_px": int(gt_masks[i].sum()),
                    "is_crop": gt_names[i] == CROP_CLASS,
                    "matched": i in matched_gt,
                    "iou": round(float(iou_of_gt.get(i, 0.0)), 3)}
                   for i in range(len(gt_masks))],
            "pred": [{"class_name": pred_names[i],
                      "score": round(pred_scores[i], 3),
                      "matched": i in matched_pred}
                     for i in range(len(pred_masks))],
            "_gt_masks": gt_masks, "_gt_names": gt_names,
            "_pred_masks": pred_masks, "_pred_names": pred_names,
            "_matched_gt": [i in matched_gt for i in range(len(gt_masks))],
            "_matched_pred": [i in matched_pred
                              for i in range(len(pred_masks))],
        })
    return records, classes, doc


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
padding:32px;background:#0f1115;color:#e6e8eb}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;
border-bottom:1px solid #2a2f3a;padding-bottom:6px}
.sub{color:#9aa3b2;margin-bottom:22px}
table{border-collapse:collapse;margin:10px 0 18px;font-variant-numeric:
tabular-nums}
th,td{padding:6px 12px;text-align:right;border-bottom:1px solid #232833}
th{color:#9aa3b2;font-weight:600}td:first-child,th:first-child{text-align:left}
.bar{background:#232833;height:9px;border-radius:5px;min-width:120px;
display:inline-block;vertical-align:middle;overflow:hidden}
.bar>i{display:block;height:100%;background:#4ea1ff}
.warn{background:#3a2416;border-left:4px solid #e08a3c;padding:12px 16px;
margin:14px 0;border-radius:4px}
.bad{background:#3a1a1c;border-left:4px solid #e05561;padding:12px 16px;
margin:14px 0;border-radius:4px}
.ok{background:#16301f;border-left:4px solid #4caf74;padding:12px 16px;
margin:14px 0;border-radius:4px}
.frame{margin:22px 0 30px}.frame img{width:100%;border-radius:6px;
border:1px solid #2a2f3a}
.cap{color:#9aa3b2;font-size:13px;margin:7px 0 0}
.legend span{display:inline-block;margin-right:16px}
.dot{display:inline-block;width:11px;height:11px;border-radius:3px;
margin-right:6px;vertical-align:-1px}
.gal{display:flex;flex-wrap:wrap;gap:10px}
.gal figure{margin:0;width:170px}
.gal img{width:100%;border-radius:5px;border:1px solid #2a2f3a}
.gal figcaption{color:#9aa3b2;font-size:11.5px;margin-top:4px}
code{background:#1a1f29;padding:2px 6px;border-radius:4px}
"""


def _bar(v):
    if v is None:
        return "&ndash;"
    return (f'<span class="bar"><i style="width:{v * 100:.0f}%"></i></span> '
            f'{v:.3f}')


def build_html(records, classes, metrics, split, checkpoint, max_frames=24,
               max_crops=48):
    worst = sorted(records, key=lambda r: (-r["n_missed"], -r["n_false_pos"]))
    n_gt = sum(r["n_gt"] for r in records)
    n_missed = sum(r["n_missed"] for r in records)
    n_fp = sum(r["n_false_pos"] for r in records)

    H = [f"<style>{CSS}</style>",
         "<h1>SeeWeed3D &mdash; Stage A results</h1>",
         f'<p class="sub">split <b>{split}</b> &middot; {len(records)} frames '
         f'&middot; {n_gt} ground-truth instances &middot; checkpoint '
         f'<code>{checkpoint}</code></p>']

    # -- headline ----------------------------------------------------------
    if metrics:
        s = metrics["summary"]
        H.append("<h2>Detection</h2><table><tr><th>class</th><th>n_gt</th>"
                 "<th>AP50</th><th>AP50-95</th><th>P</th><th>R</th>"
                 "<th>IoU</th></tr>")
        op = metrics["operating_point"]
        for c in s["classes"]:
            d, o = metrics["detection"][c], op[c]
            f = lambda v: "&ndash;" if v is None else f"{v:.3f}"
            H.append(f"<tr><td>{c}</td><td>{d['n_gt']}</td>"
                     f"<td>{f(d['ap50'])}</td><td>{f(d['ap50_95'])}</td>"
                     f"<td>{f(o['precision'])}</td><td>{f(o['recall'])}</td>"
                     f"<td>{f(o['mean_iou'])}</td></tr>")
        H.append("</table>")
        m50 = s.get("map50")
        m5095 = s.get("map50_95")
        H.append(f"<p>mAP@50 <b>{'&ndash;' if m50 is None else f'{m50:.3f}'}"
                 f"</b> &nbsp; mAP@50:95 <b>"
                 f"{'&ndash;' if m5095 is None else f'{m5095:.3f}'}</b> "
                 f"&nbsp;<span class='sub'>(P/R/IoU at conf="
                 f"{op['conf']})</span></p>")
        if s.get("classes_without_ground_truth"):
            H.append(f'<div class="warn"><b>No ground truth in this split</b> '
                     f'for: {", ".join(s["classes_without_ground_truth"])}. '
                     f'Their AP is undefined and excluded from the mean &mdash; '
                     f'this is a gap in the annotation set, not a model '
                     f'result.</div>')

    # -- recall by size ----------------------------------------------------
    H.append("<h2>Recall by instance size</h2>")
    H.append("<p class='sub'>Weeds only. A weeder that misses small weeds "
             "misses the ones worth killing &mdash; a weed is cheapest to "
             "destroy before it establishes. One aggregate recall hides where "
             "the cliff is.</p>")
    H.append("<table><tr><th>GT mask area (px)</th><th>n_gt</th>"
             "<th>found</th><th>recall</th></tr>")
    rows = recall_by_size(records)
    for r in rows:
        H.append(f"<tr><td>{r['range_px']}</td><td>{r['n_gt']}</td>"
                 f"<td>{r['n_found']}</td><td>{_bar(r['recall'])}</td></tr>")
    H.append("</table>")

    small = [r for r in rows if r["n_gt"] and r["recall"] is not None
             and r["range_px"] in ("0-250", "250-500", "500-1000")]
    if small:
        worst_small = min(small, key=lambda r: r["recall"])
        if worst_small["recall"] < 0.5:
            H.append(f'<div class="bad"><b>Small-weed recall is '
                     f'{worst_small["recall"]:.2f}</b> in the '
                     f'{worst_small["range_px"]} px bucket &mdash; '
                     f'{worst_small["n_gt"] - worst_small["n_found"]} of '
                     f'{worst_small["n_gt"]} missed. Look at the gallery '
                     f'below before changing anything: a miss caused by too '
                     f'few pixels at this input resolution needs tiling or a '
                     f'higher-resolution model, not more epochs.</div>')

    # -- crop safety -------------------------------------------------------
    H.append("<h2>Crop safety</h2>")
    cs = (metrics or {}).get("crop_safety", {})
    if not cs or cs.get("frames_with_onion", 0) == 0:
        H.append('<div class="warn"><b>UNMEASURED, not passing.</b> This split '
                 'has no <code>onion_plant</code> ground truth, so nothing '
                 'here says the crop is safe. A model trained without onion '
                 'instances cannot predict the crop at all.</div>')
    else:
        frac = cs.get("missed_onion_fraction")
        box = "ok" if (frac is not None and frac < 0.02) else "bad"
        H.append(f'<div class="{box}">Missed onion pixels: '
                 f'<b>{cs["missed_onion_px"]}</b> of {cs["onion_gt_px"]} '
                 f'({"&ndash;" if frac is None else f"{frac:.4f}"}). These are '
                 f'crop pixels the system does not know are crop.</div>')
        # The subset that is damage rather than latent risk. Older
        # metrics_val.json files predate this field; say so rather than
        # printing a zero that would read as a clean bill of health.
        if "weed_on_crop_px" in cs:
            bfrac = cs.get("weed_on_crop_fraction")
            bbox = "ok" if (bfrac is not None and bfrac < 0.002) else "bad"
            H.append(f'<div class="{bbox}">Onion the model called <b>weed</b>: '
                     f'<b>{cs["weed_on_crop_px"]}</b> of {cs["onion_gt_px"]} '
                     f'({"&ndash;" if bfrac is None else f"{bfrac:.4f}"}), in '
                     f'{cs.get("frames_with_burn", 0)} of '
                     f'{cs["frames_with_onion"]} frames. This is the laser '
                     f'firing into the crop &mdash; the only number here that '
                     f'is damage rather than the possibility of it.</div>')
        else:
            H.append('<div class="warn">This <code>metrics_val.json</code> '
                     'predates the <code>weed_on_crop_px</code> measurement, so '
                     'how much onion would actually be fired at is unknown '
                     'here. Re-run <code>eval_seg</code> to get it.</div>')

    # -- missed gallery ----------------------------------------------------
    import cv2
    H.append("<h2>Missed weeds, smallest first</h2>")
    H.append("<p class='sub'>Every one of these is a weed the model did not "
             "find. Ask of each: too few pixels, genuinely ambiguous, or a "
             "labelling error? Three different fixes.</p>")
    misses = []
    for r in records:
        img = None
        for i, g in enumerate(r["gt"]):
            if g["matched"] or g["is_crop"]:
                continue
            if img is None:
                img = cv2.imread(r["path"])
            if img is None:
                break
            misses.append((g["area_px"], g["class_name"], r["item_id"],
                           crop_around(img, r["_gt_masks"][i])))
    misses = [m for m in sorted(misses, key=lambda x: x[0]) if m[3] is not None]
    if not misses:
        H.append('<div class="ok">No missed weeds in this split.</div>')
    else:
        H.append(f"<p class='sub'>{len(misses)} missed; showing the "
                 f"{min(len(misses), max_crops)} smallest.</p><div class='gal'>")
        for area, cls, item, crop in misses[:max_crops]:
            H.append(f"<figure><img src='{png_data_uri(crop)}'>"
                     f"<figcaption>{area} px &middot; {cls}<br>{item}"
                     f"</figcaption></figure>")
        H.append("</div>")

    # -- per-frame panels --------------------------------------------------
    H.append("<h2>Frames, worst first</h2>")
    H.append('<p class="legend">'
             '<span><i class="dot" style="background:#50c850"></i>matched</span>'
             '<span><i class="dot" style="background:#eb3c3c"></i>MISSED ground '
             'truth</span>'
             '<span><i class="dot" style="background:#dc50dc"></i>false '
             'positive</span>'
             '<span><i class="dot" style="background:#ffa500"></i>crop</span>'
             '</p>')
    H.append(f"<p class='sub'>{n_missed} missed and {n_fp} false positives "
             f"across {len(records)} frames; showing the "
             f"{min(len(worst), max_frames)} worst.</p>")
    for r in worst[:max_frames]:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        panel = frame_panel(img, r["_gt_masks"], r["_gt_names"],
                            r["_matched_gt"], r["_pred_masks"],
                            r["_pred_names"], r["_matched_pred"])
        H.append(f"<div class='frame'><img src='{png_data_uri(panel)}'>"
                 f"<p class='cap'><b>{r['item_id']}</b> &mdash; "
                 f"{r['n_matched']} matched, {r['n_missed']} missed, "
                 f"{r['n_false_pos']} false positive</p></div>")

    H.append("<h2>Reading this honestly</h2>")
    H.append("<p class='sub'>If train and val come from the same recording, "
             "these frames share its lighting, soil, growth stage and often "
             "the same individual plants. That makes this a sanity check that "
             "training works &mdash; not evidence of generalisation. Only a "
             "held-out session can support that.</p>")
    return "\n".join(H)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="prepare_dataset output dir")
    p.add_argument("--images-root", nargs="*", default=None,
                   help="sessions root(s); omit to use what the manifest "
                        "recorded")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", default="maskrcnn",
                   choices=["maskrcnn", "rfdetr"],
                   help="must match the backend that produced the checkpoint")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU at which a prediction counts as matching")
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--max-crops", type=int, default=48)
    p.add_argument("--out", default=None, help="output .html path")
    a = p.parse_args(argv)

    images_root = a.images_root if a.images_root else None
    if images_root and len(images_root) == 1:
        images_root = images_root[0]

    records, classes, doc = collect(a.checkpoint, a.dataset, images_root,
                                    a.split, a.device, conf=a.conf,
                                    iou_threshold=a.iou, backend=a.backend)
    try:
        from evaluation.eval_seg import evaluate
        metrics = evaluate(a.checkpoint, a.dataset, images_root, a.split,
                           a.device, conf=a.conf, backend=a.backend)
    except Exception as e:
        print(f"[warn] metric table unavailable: {e}")
        metrics = None

    html = build_html(records, classes, metrics, a.split, a.checkpoint,
                      max_frames=a.max_frames, max_crops=a.max_crops)
    out = Path(a.out) if a.out else Path(a.checkpoint).parent / \
        f"report_{a.split}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    # The per-frame JSON behind the page, so the same analysis can be diffed
    # between runs without re-running inference.
    slim = [{k: v for k, v in r.items() if not k.startswith("_")}
            for r in records]
    jout = out.with_suffix(".json")
    jout.write_text(json.dumps(
        {"split": a.split, "checkpoint": str(a.checkpoint),
         "conf": a.conf, "iou": a.iou, "classes": classes,
         "recall_by_size": recall_by_size(records), "frames": slim}, indent=2),
        encoding="utf-8")

    size_mb = out.stat().st_size / 1e6
    print(f"-> {out}  ({size_mb:.1f} MB, open it in a browser)")
    print(f"-> {jout}")


if __name__ == "__main__":
    main()
