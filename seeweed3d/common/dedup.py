#!/usr/bin/env python3
"""SeeWeed3D - one plant, one detection.

THE FAILURE THIS PREVENTS
-------------------------
RF-DETR is a set-prediction model: every query proposes independently, and
nothing makes two queries that found the same plant agree on what it is. So the
SAME mask comes back twice under two different class labels, at two different
scores:

    other_weed               0.275   bbox [1861.4, 121.1, 24.4, 19.0]
    cutleaf_evening_primrose 0.496   bbox [1861.4, 121.1, 24.4, 19.0]

Identical box, IoU 1.000. Observed on a real weed session in 6 of 16 frames.

Nothing downstream survives that cleanly:

  * A LASER WEEDER fires twice at one plant. The second pulse is spent on
    scorched ground while a weed elsewhere in the frame goes untreated.
  * A PRELABEL EXPORT gives the annotator two overlapping polygons to notice
    and delete, on every affected plant, in every frame.
  * ANY per-instance count - detections per class, instances per frame, the
    cluster rate - is inflated by an amount nobody measured.

WHY IT IS CLASS-AGNOSTIC
------------------------
Suppressing only within a class leaves exactly the case above untouched, which
is the case that actually happens. A plant is one object; if the model cannot
decide what it is, that is one detection with an uncertain label, not two.

The surviving label is the higher-scoring one, which is the model's own answer
to the question it was unsure about.

WHY THE DEFAULT THRESHOLD IS HIGH
---------------------------------
The duplicates seen in the field are EXACT box matches - IoU 1.0. Two genuinely
adjacent plants in a dense frame can reach IoU 0.5-0.6 and must not be merged:
merging them costs a real weed its own treatment point, which is the failure
this project cares about most. 0.85 removes the near-identical pairs and leaves
the ambiguous ones alone.
"""

from __future__ import annotations

import numpy as np

#: Only near-identical detections are merged. See the module docstring: the
#: observed duplicates are IoU 1.0, while two touching plants reach 0.5-0.6 and
#: merging THOSE loses a weed its own treatment point.
DEFAULT_DEDUP_IOU = 0.85


def box_iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a, b):
    """IoU of two boolean masks."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape:
        return 0.0
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def suppress_duplicates(instances, iou=DEFAULT_DEDUP_IOU, score_key="score",
                        box_key="bbox", mask_key="mask"):
    """Drop detections that duplicate a higher-scoring one. Returns (kept, dropped).

    MASKS DECIDE WHERE THEY EXIST. Two plants can share a bounding box while
    overlapping barely at all - an L-shaped weed beside a compact one is the
    ordinary case in a dense frame - so box IoU alone would merge them and cost
    one of them its treatment point. The box is the fallback for callers that
    have already discarded the masks, and it is a coarser instrument.

    Order is by score, descending, so the surviving label is the model's own
    best answer. Ties keep the earlier instance, which makes the result
    deterministic for a given input order.

    Nothing is mutated: `kept` and `dropped` hold the original objects."""
    items = list(instances)
    if len(items) < 2 or iou <= 0:
        return items, []

    def score_of(d):
        try:
            return float(d.get(score_key, 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    order = sorted(range(len(items)), key=lambda i: (-score_of(items[i]), i))
    kept_idx, dropped = [], []
    for i in order:
        cand = items[i]
        for j in kept_idx:
            if _overlap(cand, items[j], box_key, mask_key) >= iou:
                dropped.append(cand)
                break
        else:
            kept_idx.append(i)
    # Restore the caller's original ordering among survivors: callers sort for
    # display, and re-sorting here would silently change what they show.
    return [items[i] for i in sorted(kept_idx)], dropped


def _overlap(a, b, box_key, mask_key):
    ma, mb = a.get(mask_key), b.get(mask_key)
    if ma is not None and mb is not None:
        return mask_iou(ma, mb)
    ba, bb = a.get(box_key), b.get(box_key)
    if ba is not None and bb is not None:
        return box_iou(ba, bb)
    return 0.0


def describe_suppression(kept, dropped, label="duplicate detections"):
    """One line on what was merged, or None when nothing was.

    Names the CLASS PAIRS because that is the diagnostic: a model emitting the
    same mask as two different classes is telling you those two classes are not
    separable on this data, which is a labelling question rather than a
    threshold one."""
    if not dropped:
        return None
    from collections import Counter
    pairs = Counter(str(d.get("class_name", "?")) for d in dropped)
    total = len(kept) + len(dropped)
    body = ", ".join(f"{k}={v}" for k, v in pairs.most_common(5))
    return (f"  [i] suppressed {len(dropped)} {label} of {total} "
            f"({len(dropped) / total:.0%}) - dropped labels: {body}")
