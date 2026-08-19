#!/usr/bin/env python3
"""
SeeWeed3D - which mixed frames are worth annotating first.

    python -m seeweed3d.annotation.rank_by_contact \\
        --pred E:/Dataset_Vidalia/auto_labels_mixed/<session> \\
        --top 40 --out E:/Dataset_Vidalia/annotate_next.txt

WHY CONTACT
-----------
The dangerous decision in this system is made where onion and weed TOUCH. A
frame of well-separated plants exercises none of it: the model can be right
there for the wrong reasons, and a human correcting it spends time confirming
what was never in doubt. A frame where a weed sits against an onion leaf is
where a mislabel becomes a laser pulse at the crop.

So annotation effort is ranked by how much onion/weed BOUNDARY a frame contains,
not by how many plants it has or how uncertain the model was.

NO MODEL REQUIRED
-----------------
This reads prelabels - whatever you already have, however poor. That is the
point: it is available before anything is trained, and a bad prelabeler still
indicates roughly where the two classes meet. The ranking gets better as the
prelabels do, but it is useful on the first pass.

The active learner already applies this idea downstream: `_crop_risk_score`
scores a weed by its distance to onion. This is the same reasoning moved earlier
- to choosing frames to annotate rather than to ranking predictions.

WHAT IT IS NOT
--------------
Not a replacement for diversity. Ranking by one signal and taking the top N
returns near-duplicate frames from one stretch of one drive, which is the
classic active-learning trap this project already documents. --per-session caps
how many a single session may contribute; keep it on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CROP_CLASS  # noqa: E402
from evaluation.bench_mixed import load_side  # noqa: E402

#: Half-width in pixels of the band that counts as "in contact". Wider than one
#: pixel because the interesting region is the zone a mask boundary could
#: plausibly land in, not the exact ridge between two predicted masks.
CONTACT_BAND_PX = 12

#: Frames from one session, at most, in the output. One drive can dominate a
#: single-signal ranking entirely, and forty near-identical frames of the same
#: two plants is not forty frames of annotation value.
PER_SESSION = 8


def _session_of(name):
    """`<session>_<index>.png` -> `<session>`, the extractor's naming."""
    stem = Path(name).stem
    if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
        return stem.rsplit("_", 1)[0]
    return ""


def contact_score(masks, classes, band_px=CONTACT_BAND_PX):
    """How much onion/weed contact one frame contains.

    Returns a dict; `contact_px` is the ranking key. Zero is a real answer -
    a frame with only one class present has no contact to get wrong - and is
    reported rather than treated as missing."""
    import cv2
    onion = weed = None
    for m, c in zip(masks, classes):
        a = np.asarray(m).astype(bool)
        if c == CROP_CLASS:
            onion = a if onion is None else (onion | a)
        else:
            weed = a if weed is None else (weed | a)
    if onion is None or weed is None or not onion.any() or not weed.any():
        return {"contact_px": 0, "n_onion": int(onion.sum()) if onion is not None else 0,
                "n_weed": int(weed.sum()) if weed is not None else 0,
                "reason": "only one class present"}

    k = 2 * int(band_px) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    band = (cv2.dilate(onion.astype(np.uint8), se).astype(bool)
            & cv2.dilate(weed.astype(np.uint8), se).astype(bool))
    return {"contact_px": int(band.sum()),
            "n_onion": int(onion.sum()), "n_weed": int(weed.sum()),
            "n_instances": len(masks)}


def rank(pred, band_px=CONTACT_BAND_PX, per_session=PER_SESSION, top=None):
    """Frames worth annotating, most contact first, capped per session."""
    rows = []
    for name, (masks, classes) in pred.items():
        s = contact_score(masks, classes, band_px)
        s["frame"] = name
        s["session_id"] = _session_of(name)
        rows.append(s)
    rows.sort(key=lambda r: (-r["contact_px"], r["frame"]))

    kept, seen = [], {}
    for r in rows:
        if r["contact_px"] <= 0:
            continue
        sid = r["session_id"]
        if per_session and seen.get(sid, 0) >= per_session:
            r["skipped"] = "per-session cap"
            continue
        seen[sid] = seen.get(sid, 0) + 1
        kept.append(r)
        if top and len(kept) >= top:
            break
    return kept, rows


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True,
                   help="prelabels to rank: a prelabeler output folder or a "
                        "make_dataset OUT_DIR")
    p.add_argument("--top", type=int, default=None,
                   help="how many frames to return (default: all with contact)")
    p.add_argument("--per-session", type=int, default=PER_SESSION,
                   help=f"cap per session (default {PER_SESSION}). 0 disables "
                        f"the cap, which lets one drive fill the whole list.")
    p.add_argument("--band-px", type=int, default=CONTACT_BAND_PX)
    p.add_argument("--out", default=None,
                   help="write the chosen frame names, one per line")
    a = p.parse_args(argv)

    pred = load_side(a.pred)
    kept, rows = rank(pred, a.band_px, a.per_session, a.top)
    with_contact = sum(1 for r in rows if r["contact_px"] > 0)

    print(f"\n  {len(rows)} frame(s) read | {with_contact} with onion/weed "
          f"contact | {len(kept)} selected")
    if not with_contact:
        print("\n  [!] No frame has BOTH classes predicted. Either these are "
              "single-class scenes, or the prelabeler is not emitting one of "
              "the classes at all - check a preview before annotating from "
              "this ranking.")
    print(f"  {'frame':<44}{'contact px':>12}{'instances':>11}")
    print("  " + "-" * 67)
    for r in kept[:40]:
        print(f"  {r['frame'][:44]:<44}{r['contact_px']:>12}"
              f"{r.get('n_instances', 0):>11}")
    if len(kept) > 40:
        print(f"  ... and {len(kept) - 40} more")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(r["frame"] for r in kept) + "\n",
                               encoding="utf-8")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
