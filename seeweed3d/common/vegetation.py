#!/usr/bin/env python3
"""
SeeWeed3D - shared vegetation / image preprocessing.

Used by both the onion and weed prelabelers so the two cannot drift apart.
Functions take explicit parameters (not a config dict) so they are reusable and
directly testable; each caller passes its own CONFIG values in.
"""

import cv2
import numpy as np


def excess_green(bgr):
    """Excess-Green index (ExG), the standard vegetation index for RGB field
    imagery. Positive on plant tissue, ~0 or negative on soil."""
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    s = b + g + r + 1e-6
    return 2 * (g / s) - (r / s) - (b / s)


def white_balance(bgr, cast_ratio=1.15):
    """Gray-world white balance, applied only when the frame is actually
    colour-cast (channel means diverge past cast_ratio).

    Some ZED frames have a severe green white-balance error - the whole frame
    reads green - which would otherwise be unusable. Neutral frames are returned
    unchanged, so this is safe to leave on."""
    means = bgr.reshape(-1, 3).mean(axis=0) + 1e-6
    if float(means.max() / means.min()) < cast_ratio:
        return bgr
    gray = float(means.mean())
    return np.clip(bgr.astype(np.float32) * (gray / means), 0, 255).astype(np.uint8)


def remove_small(mask, min_px):
    """Drop connected components below min_px."""
    if min_px <= 0 or not mask.any():
        return mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[lbl == i] = True
    return keep


def vegetation_mask(bgr, exg_threshold=0.05, min_saturation=40, morph_kernel=3,
                    min_component_px=80):
    """Vegetation mask: ExG gated by green dominance and saturation.

    ExG alone masks bare soil whenever a frame has a green colour-cast (soil then
    reads as weak green). Requiring green to be the dominant channel AND the
    pixel to be saturated rejects that, because colour-cast soil is desaturated
    while real foliage is saturated green."""
    veg = excess_green(bgr) > exg_threshold
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    veg &= (g >= r) & (g >= b)
    veg &= cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1] >= min_saturation
    veg = veg.astype(np.uint8)
    if morph_kernel > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel,) * 2)
        veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, ker)
        veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, ker)
    return remove_small(veg.astype(bool), min_component_px)


def vegetation_score(bgr, exg_threshold=0.05, min_saturation=40, softness=0.04):
    """Soft plant likelihood in [0, 1], the continuous counterpart of
    vegetation_mask().

    The binary mask is the right tool for deciding what is a plant; for deciding
    exactly WHERE a plant boundary falls, a hard threshold throws away the very
    gradient that locates the edge. This keeps that gradient: a smooth ramp
    across the ExG threshold, gated the same way by green dominance and
    saturation so a colour-cast soil cannot score as plant."""
    exg = excess_green(bgr)
    # Logistic ramp centred on the threshold: 0.5 at the threshold, saturating
    # within a few hundredths of ExG either side.
    score = 1.0 / (1.0 + np.exp(-(exg - exg_threshold) / max(1e-6, softness)))
    b, g, r = [bgr[:, :, i].astype(np.float32) for i in range(3)]
    score *= ((g >= r) & (g >= b)).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    score *= np.clip(sat / max(1.0, float(min_saturation)), 0.0, 1.0)
    return score.astype(np.float32)


def component_boxes(mask, min_area_px, pad_px=0, max_boxes=None,
                    confidence=None, min_confidence=0.0):
    """Bounding boxes [x1,y1,x2,y2] of mask components >= min_area_px, largest
    first. Used to prompt SAM 3 with real plant exemplars.

    confidence, when given, is a per-pixel score in [0, 1] (see
    vegetation_score()): a component is kept only if its MEAN confidence over
    the component reaches min_confidence. min_area_px alone cannot tell a
    faint, small, but genuine seedling from a same-sized patch of gravel,
    lichen or shadow that only marginally crosses the binary vegetation
    threshold - confidence asks a different question, how sure the colour
    evidence is, not how big the blob is. Backward compatible: omitted, this
    behaves exactly as before."""
    if not mask.any():
        return []
    h, w = mask.shape
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        if confidence is not None and float(confidence[lbl == i].mean()) < min_confidence:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append((area, [max(0, x - pad_px), max(0, y - pad_px),
                           min(w, x + bw + pad_px), min(h, y + bh + pad_px)]))
    out.sort(key=lambda t: -t[0])
    boxes = [b for _, b in out]
    return boxes[:max_boxes] if max_boxes else boxes


def distance_peaks(mask, rel_threshold=0.5, min_separation_px=15, max_peaks=32,
                   min_saddle_drop=0.0):
    """Distinct interior maxima of a mask's distance transform.

    For a rosette this lands on the crown; for two rosettes grown into each
    other it lands on both crowns. That is the signal that one connected blob
    holds more than one plant, and it is the only geometric evidence available
    once colour has already merged them.

    Iteratively takes the distance-transform maximum and suppresses a disc
    around it, so a single plant yields one peak however lobed it is. Returns
    [((x, y), radius), ...], strongest first.

    rel_threshold keeps a peak only if its inscribed radius is at least this
    fraction of the largest - a leaf tip is a shallow maximum and must not read
    as a second plant.

    min_saddle_drop IS THE ONE THAT MATTERS ON REAL TISSUE, and 0 (off) is only
    correct for callers that already tolerate over-detection. A LONG LEAF HAS A
    FLAT DISTANCE RIDGE: every point along a ribbon of constant width has the
    same inscribed radius, so relative height alone accepts a dozen "peaks"
    strung along one leaf and shatters it. Between two genuine crowns the
    transform DIPS at the neck where the canopies meet; along one leaf it does
    not. So a candidate is kept only if it is still separated from every
    accepted peak at a level `min_saddle_drop` of its own radius - the standard
    persistence test, and the only thing here that distinguishes "two plants"
    from "one plant twice as long".
    """
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    gmax = float(dt.max())
    if gmax <= 0:
        return []
    work, peaks = dt.copy(), []
    while len(peaks) < max_peaks:
        _, v, _, loc = cv2.minMaxLoc(work)
        if v <= 0 or v < rel_threshold * gmax:
            break
        if peaks and min_saddle_drop > 0:
            # Cut the transform at a level below this candidate. If an already
            # accepted peak survives in the SAME piece, the two sit on one
            # ridge with no neck between them - one plant, not two.
            _, lab = cv2.connectedComponents(
                (dt >= min_saddle_drop * v).astype(np.uint8), 8)
            here = lab[loc[1], loc[0]]
            if here and any(lab[p[1], p[0]] == here for p, _ in peaks):
                cv2.circle(work, loc, int(max(min_separation_px, v)), 0, -1)
                continue
        peaks.append((loc, float(v)))
        cv2.circle(work, loc, int(max(min_separation_px, v)), 0, -1)
    return peaks


#: How far a mask may fall short of its plant before the shortfall counts as
#: something unlabelled. A hand-drawn polygon sits a pixel or two inside the
#: leaf it traces, and a rim that thin is annotation slop, not a missed plant.
CLAIM_DILATE_PX = 4

#: Smallest unclaimed blob worth calling a plant. Below this it is a leaf tip
#: outside its own mask, a speck of moss, or a green fleck of debris - and a
#: report full of those is one nobody reads.
MIN_UNCLAIMED_BLOB_PX = 300


#: A patch this much OUTSIDE the annotation-slop band is a plant rather than a
#: rim. See unclaimed_blobs for why this is a share and not another distance.
MIN_OUTSIDE_BAND_FRAC = 0.10


def unclaimed_blobs(veg, claimed, dilate_px=CLAIM_DILATE_PX,
                    min_blob_px=MIN_UNCLAIMED_BLOB_PX,
                    outside_frac=MIN_OUTSIDE_BAND_FRAC):
    """Plant-sized patches of vegetation that no annotation covers.

    Returns (n_blobs, unclaimed_px, mask).

    A COUNT OF BLOBS, not a fraction of vegetation. The two answer different
    questions and only one of them is useful. Measured on real geometry: masks
    drawn two pixels inside their leaves leave 19% of vegetation unclaimed and
    nothing missing, while one unlabelled seedling among twenty-five labelled
    plants is 1.7% and a real hole. A fraction gate gets BOTH of those wrong,
    and in opposite directions, because its denominator grows with everything
    you labelled correctly - so labelling more plants properly makes a fixed
    miss look smaller.

    HOW A RIM IS TOLD FROM A PLANT. Not by size: a two-pixel rim around a 40 px
    plant is 304 px, bigger than many seedlings. It is told by WHERE it sits.
    A rim lies entirely inside a narrow band along the mask boundary; a plant
    nobody labelled extends past it. So a component is discarded when almost all
    of it falls inside that band, whatever its area.

    Thresholding on area AFTER subtracting the band would be the obvious
    implementation and it is biased in the worst possible direction: it eats the
    near edge of anything adjacent, so an 18x18 seedling TOUCHING a labelled
    plant drops under the size floor and disappears - and a weed touching a crop
    is the exact case this project exists to get right. Measuring the whole
    component and testing only where it lies keeps it.

    It cannot tell you a blob IS a missed plant - the vegetation prior calls
    moss, algae and green debris vegetation too, and misses dark seedlings
    entirely. It tells you where to look."""
    import cv2
    veg = np.asarray(veg, bool)
    claimed = np.asarray(claimed, bool)
    band = claimed
    if dilate_px > 0 and claimed.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1,) * 2)
        band = cv2.dilate(claimed.astype(np.uint8), k).astype(bool)
    left = veg & ~claimed
    if not left.any():
        return 0, 0, left
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), 8)
    keep = np.zeros_like(left)
    blobs = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_blob_px:
            continue
        comp = lbl == i
        outside = int((comp & ~band).sum())
        if outside / area < outside_frac:
            continue                      # a rim along a mask, not a plant
        keep |= comp
        blobs += 1
    return blobs, int(keep.sum()), keep
