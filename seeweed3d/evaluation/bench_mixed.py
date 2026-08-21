#!/usr/bin/env python3
"""
SeeWeed3D - the mixed-scene benchmark. The ruler for every prelabeling change.

    python -m seeweed3d.evaluation.bench_mixed \\
        --truth  E:/Dataset_Vidalia/mixed_gt \\
        --pred   E:/Dataset_Vidalia/auto_labels_mixed/<session> \\
        --out    E:/Dataset_Vidalia/bench/zeroshot.json

WHY THIS EXISTS
---------------
Every strategy for building the mixed dataset - zero-shot SAM, a trained
identity model, the two-stage arrangement, compositing - is a claim that masks
got better. Without a fixed set of hand-annotated frames and a metric that
measures the right thing, those claims are settled by looking at previews, which
is how a boundary pipeline that improved every number shipped worse masks in the
field (see CHANGELOG #29).

A handful of frames is enough to be that ruler. It is NOT enough to train on,
and this module reports how few it is rather than letting the number look
sturdier than it is.

WHAT IT MEASURES, AND WHAT IT REFUSES TO
-----------------------------------------
Not mean IoU. A mean over all instances buries the one failure that matters
among many harmless ones - a model can look excellent while labelling crop as
weed. Three groups are reported separately and never combined:

  crop      onion-called-weed (a laser at the crop) apart from weed-called-onion
            (a missed weed), and both again restricted to the band around
            onion/weed contacts, where the decision is actually exercised.
  identity  merges and fragments, counted separately because they are opposite
            failures that cancel in any instance-count difference.
  cluster   weed_cluster predicted over ground truth that IS separable -
            annotation policy becoming deployed policy, caught early.

INPUTS
------
`--truth` and `--pred` each accept either:

  * a make_dataset OUT_DIR, read from its seg_manifest.json, or
  * a folder holding a COCO instances_default.json (what the prelabelers emit).

Frames are paired by image file name, so the two sides need not agree on
ordering, on item ids, or on covering the same set of frames - only the
intersection is scored, and both sides' unmatched frames are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CROP_CLASS  # noqa: E402
from evaluation import metrics as met  # noqa: E402

#: Below this many paired frames the numbers are reported with a warning. Not a
#: refusal - a small ruler is still a ruler, and the alternative is judging
#: masks by eye - but a per-frame fraction from a handful of frames moves a lot
#: on one bad frame, and that has to be visible next to the number.
FEW_FRAMES = 20


def _polys_to_mask(polys, h, w):
    import cv2
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        a = np.asarray(p, np.float64).reshape(-1, 2)
        if len(a) >= 3:
            cv2.fillPoly(m, [np.round(a).astype(np.int32)], 1)
    return m.astype(bool)


def _from_manifest(doc):
    """seg_manifest.json -> {file_name: (masks, classes)}."""
    out = {}
    for f in doc.get("frames", []):
        h, w = int(f["height"]), int(f["width"])
        masks, classes = [], []
        for inst in f.get("instances", []):
            masks.append(_polys_to_mask(inst.get("polygons", []), h, w))
            classes.append(inst["class_name"])
        out[Path(f["image_path"]).name] = (masks, classes)
    return out


def _from_coco(doc):
    """COCO instances_default.json -> {file_name: (masks, classes)}."""
    names = {c["id"]: c["name"] for c in doc.get("categories", [])}
    sizes = {im["id"]: (int(im["height"]), int(im["width"]))
             for im in doc.get("images", [])}
    files = {im["id"]: Path(im["file_name"]).name
             for im in doc.get("images", [])}
    per = {i: ([], []) for i in files}
    for a in doc.get("annotations", []):
        iid = a["image_id"]
        if iid not in per:
            continue
        h, w = sizes[iid]
        seg = a.get("segmentation") or []
        # RLE is not decoded here on purpose: every producer in this repo emits
        # polygons, and a silently-skipped RLE instance would understate the
        # prediction rather than fail.
        if isinstance(seg, dict):
            raise SystemExit(
                "ERROR: RLE segmentation found. This benchmark reads polygon "
                "COCO, which is what every prelabeler here emits.")
        per[iid][0].append(_polys_to_mask(seg, h, w))
        per[iid][1].append(names.get(a["category_id"], str(a["category_id"])))
    return {files[i]: per[i] for i in files}


#: Words in a COCO `info.description` that mark a side as machine-generated.
#: Every producer in this repo stamps its own provenance, so this recognises
#: our own files rather than guessing at arbitrary ones.
UNREVIEWED_MARKERS = ("prelabel", "model predictions")


def source_provenance(path):
    """What KIND of labels a side holds: "", "prelabels" or "predictions".

    Returns "" when the file carries no provenance, which is the honest answer
    for a hand-corrected export - CVAT writes its own info block and cannot
    know."""
    p = Path(path)
    cands = ([p] if p.is_file() else
             sorted(p.glob("**/instances_default.json")))
    for c in cands:
        try:
            doc = json.loads(Path(c).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        desc = str((doc.get("info") or {}).get("description", "")).lower()
        if "model predictions" in desc:
            return "predictions"
        if "prelabel" in desc:
            return "prelabels"
    return ""


def provenance_warning(truth_path, pred_path):
    """The caveat that has to sit beside these numbers, or None.

    THE WORD "TRUTH" DOES REAL DAMAGE HERE. Comparing a model against SAM
    prelabels is a legitimate and useful thing to do - it is the only comparison
    available before anything is hand-corrected - but what it measures is
    AGREEMENT BETWEEN TWO PROPOSALS, and neither side is evidence for the other.
    Both can be wrong the same way, and here they are correlated by
    construction: the model was trained on corrected SAM prelabels, so it
    inherits the prelabeler's biases through its training data.

    Reported rather than refused. The alternative to an imperfect ruler is
    judging masks by eye, which is how a boundary pipeline that improved every
    number shipped worse masks in the field (CHANGELOG #29)."""
    kind = source_provenance(truth_path)
    if not kind:
        return None
    what = ("SAM prelabels" if kind == "prelabels" else "model predictions")
    return (f"  [!] --truth is {what}, NOT hand-corrected ground truth.\n"
            f"      Every number below is AGREEMENT BETWEEN TWO PROPOSALS. It "
            f"cannot tell you which side is right, and a high score is also "
            f"what two sources wrong in the SAME way produce.\n"
            f"      If --pred is this project's model, the two are correlated "
            f"by construction: it was trained on corrected SAM prelabels, so it "
            f"inherits the prelabeler's biases. Read agreement as a FLOOR on "
            f"disagreement, and use the per-frame worst rows to choose what to "
            f"annotate.")


def load_side(path):
    """A truth or prediction source, whichever of the two forms it is."""
    p = Path(path)
    man = p / "seg_manifest.json" if p.is_dir() else None
    if man is not None and man.exists():
        return _from_manifest(json.loads(man.read_text(encoding="utf-8")))
    cands = ([p] if p.is_file() else
             sorted(p.glob("**/instances_default.json")) +
             sorted(p.glob("**/*.json")))
    for c in cands:
        try:
            doc = json.loads(Path(c).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and "frames" in doc:
            return _from_manifest(doc)
        if isinstance(doc, dict) and "annotations" in doc and "images" in doc:
            return _from_coco(doc)
    raise SystemExit(
        f"ERROR: no seg_manifest.json or COCO instances_default.json under "
        f"{p}.\\nPoint --truth/--pred at a make_dataset OUT_DIR or at a "
        f"prelabeler output folder.")


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def benchmark(truth, pred, contact_band_px=met.CONTACT_BAND_PX):
    """Per-frame metrics plus a pooled summary.

    Crop numbers are pooled by PIXEL COUNT rather than averaged over frames: a
    frame holding four onions and one holding forty should not carry equal
    weight in a crop-safety number. Rates over frames are averaged, since each
    frame is one observation of the same rate."""
    common = sorted(set(truth) & set(pred))
    frames = []
    tot = {"onion_as_weed_px": 0, "weed_as_onion_px": 0, "onion_unclaimed_px": 0,
           "gt_onion_px": 0, "gt_weed_px": 0}
    for name in common:
        gm, gc = truth[name]
        pm, pc = pred[name]
        r = met.mixed_scene_metrics(pm, pc, gm, gc,
                                    contact_band_px=contact_band_px)
        r["frame"] = name
        frames.append(r)
        for k in tot:
            tot[k] += int(r["crop"].get(k, 0) or 0)

    summary = {
        "n_frames_scored": len(common),
        "n_truth_only": len(set(truth) - set(pred)),
        "n_pred_only": len(set(pred) - set(truth)),
        # THE headline. Crop pixels the system would fire at.
        "onion_as_weed_px": tot["onion_as_weed_px"],
        "onion_as_weed_fraction": (tot["onion_as_weed_px"] / tot["gt_onion_px"]
                                   if tot["gt_onion_px"] else None),
        "weed_as_onion_px": tot["weed_as_onion_px"],
        "weed_as_onion_fraction": (tot["weed_as_onion_px"] / tot["gt_weed_px"]
                                   if tot["gt_weed_px"] else None),
        "onion_unclaimed_px": tot["onion_unclaimed_px"],
        "contact_onion_as_weed_fraction": _mean(
            [f["crop"].get("contact_onion_as_weed_fraction") for f in frames]),
        "contact_weed_as_onion_fraction": _mean(
            [f["crop"].get("contact_weed_as_onion_fraction") for f in frames]),
        "n_frames_with_contact": sum(
            1 for f in frames if f["crop"].get("contact_band_px_count", 0) > 0),
        "merge_rate": _mean([f["identity"]["merge_rate"] for f in frames]),
        "fragment_rate": _mean([f["identity"]["fragment_rate"] for f in frames]),
        "n_merged_gt_instances": sum(f["identity"]["n_merged_gt_instances"]
                                     for f in frames),
        "n_fragmented_gt": sum(f["identity"]["n_fragmented_gt"] for f in frames),
        "instance_count_error": sum(f["identity"]["count_error"]
                                    for f in frames),
        "cluster_over_prediction_rate": _mean(
            [f["cluster"]["cluster_over_prediction_rate"] for f in frames]),
        "few_frames": len(common) < FEW_FRAMES,
    }
    return summary, frames


def format_report(summary):
    def pct(v):
        return "  n/a" if v is None else f"{100.0 * v:5.2f}%"

    L = ["", "  Mixed-scene benchmark", "  " + "-" * 62,
         f"  frames scored          : {summary['n_frames_scored']}"
         f"   (truth-only {summary['n_truth_only']}, "
         f"pred-only {summary['n_pred_only']})",
         f"  frames with contact    : {summary['n_frames_with_contact']}",
         "",
         "  CROP SAFETY  (never averaged with the other direction)",
         f"    onion called weed    : {pct(summary['onion_as_weed_fraction'])}"
         f"   {summary['onion_as_weed_px']} px   <-- laser at the crop",
         f"      of that, at contact: {pct(summary['contact_onion_as_weed_fraction'])}",
         f"    weed called onion    : {pct(summary['weed_as_onion_fraction'])}"
         f"   {summary['weed_as_onion_px']} px   (a missed weed)",
         f"      of that, at contact: {pct(summary['contact_weed_as_onion_fraction'])}",
         f"    onion claimed by     : {summary['onion_unclaimed_px']} px "
         f"neither class",
         "",
         "  IDENTITY  (opposite failures, so never netted off)",
         f"    merge rate           : {pct(summary['merge_rate'])}"
         f"   {summary['n_merged_gt_instances']} true instances swallowed",
         f"    fragment rate        : {pct(summary['fragment_rate'])}"
         f"   {summary['n_fragmented_gt']} true instances shattered",
         f"    instance count error : {summary['instance_count_error']:+d}"
         f"   (a merge and a fragment cancel here - read the two above)",
         "",
         "  CLUSTER USE",
         f"    cluster over separable gt: "
         f"{pct(summary['cluster_over_prediction_rate'])}"]
    if summary["few_frames"]:
        L += ["",
              f"  [!] {summary['n_frames_scored']} frames is a small ruler. One "
              f"bad frame moves every fraction above, so read these as a "
              f"comparison BETWEEN runs rather than as absolute performance.",
              f"      It is still the right ruler: the alternative is judging "
              f"masks by eye."]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True,
                   help="hand-annotated mixed frames: a make_dataset OUT_DIR "
                        "or a folder with COCO instances_default.json")
    p.add_argument("--pred", required=True,
                   help="what to score: a prelabeler output folder or another "
                        "dataset build")
    p.add_argument("--out", default=None,
                   help="write the full per-frame report as JSON")
    p.add_argument("--contact-band-px", type=int, default=met.CONTACT_BAND_PX,
                   help=f"half-width of the onion/weed contact band "
                        f"(default {met.CONTACT_BAND_PX})")
    a = p.parse_args(argv)

    truth, pred = load_side(a.truth), load_side(a.pred)
    summary, frames = benchmark(truth, pred, a.contact_band_px)
    if not summary["n_frames_scored"]:
        raise SystemExit(
            "ERROR: no frames in common between --truth and --pred. They are "
            "paired by image FILE NAME; check both sides describe the same "
            "frames.")
    print(format_report(summary))
    # AFTER the numbers, not before: a caveat printed first is scrolled past,
    # and this one changes what every line above means.
    warn = provenance_warning(a.truth, a.pred)
    if warn:
        print()
        print(warn)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"summary": summary, "frames": frames,
             "truth_provenance": source_provenance(a.truth),
             "pred_provenance": source_provenance(a.pred)},
            indent=2, default=float),
            encoding="utf-8")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
