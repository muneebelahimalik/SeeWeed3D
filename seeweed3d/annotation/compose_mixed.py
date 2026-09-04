#!/usr/bin/env python3
"""
SeeWeed3D - synthesise MIXED scenes: real weed cut-outs into real onion frames.

    python -m seeweed3d.annotation.compose_mixed

THE GAP THIS CLOSES, AND NOTHING ELSE
--------------------------------------
Every measurement in this project points at one hole. crop_risk showed the weed
model claiming onion tissue almost everywhere it looked. The mixed build's own
warnings showed why nothing can fix that from data: the sessions that contain
onions and weeds TOGETHER are three drives where only the crop was annotated,
plus seven hand-annotated frames. Seven frames cannot teach a decision boundary.

Compositing closes exactly that gap and nothing else. It does not make more
weeds, or more onions - there is plenty of both. It makes the one thing the
recordings do not contain: a weed at a known distance from an onion, with both
masks correct.

WEED INTO ONION, NEVER ONION INTO WEED
---------------------------------------
An onion scene carries row spacing, planting geometry, furrows, crop shadows and
naturally overlapping onions - structure that would have to be synthesised, and
would be synthesised wrongly. A weed cut-out carries a plant. So the real onion
field stays intact and weeds are introduced into it.

This is also why dataset_growth.md's rule against copy-paste is not violated
here. That rule exists because "pasting an onion between frames fabricates crop
geometry no field produced, and this is a crop-SAFETY model". Nothing here
fabricates crop geometry: every onion, every row and every shadow is the one the
camera recorded.

THE SCREEN THAT MATTERS MORE THAN THE COMPOSITING
--------------------------------------------------
A background is only usable if everything green in it is already labelled.

Paste a labelled weed into a frame that also contains an UNLABELLED weed and the
result actively teaches the confusion it was built to fix: two visually
identical plants, one a target, one background. That is worse than no synthetic
data at all, and it is not hypothetical - this project has already found three
"mixed" drives whose weeds were never annotated.

So every candidate background is screened against the vegetation prior first:
vegetation not covered by an annotated mask means something is growing there
that nobody labelled, and the frame is rejected. UNCLAIMED_BLOBS_MAX is the
only gate that can make this whole module counterproductive if set carelessly.

CONTACT IS THE POINT, SO CONTACT IS MEASURED
---------------------------------------------
Placement is not random. Each composite targets a contact band - isolated, near,
very near, touching, overlapping - and the band it ACHIEVED is measured from the
masks and recorded per instance. A generator that says "we pasted weeds near
onions" cannot be checked; one that reports the achieved distance distribution
can.

WHAT COMPOSITES ARE NOT
-----------------------
They have no real occlusion where plants grow through each other, no real shadow
interaction, no co-adapted growth. They bootstrap the contact case; they do not
retire it.

NEVER VALIDATE ON THEM. They share this generator's blind spots, so a score
computed on them measures the generator. The manifest says LABEL_PROVENANCE
"synthetic" and the session is named so it cannot be mistaken for a recording.
"""
from __future__ import annotations

import json
import ntpath
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from common.ontology import CLASSES, CROP_CLASS, LEP_LABEL  # noqa: E402
from common.run_dirs import stamped  # noqa: E402
from common.vegetation import unclaimed_blobs  # noqa: E402
from training import pseudo_label as pl  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: WHERE THE WEED CUT-OUTS COME FROM. Sessions whose weed masks a PERSON drew.
#: Prelabels are the wrong source: a composite made from a machine mask is a
#: machine mask with extra steps, and the whole value here is that the pasted
#: instance is TRUE.
#: A CUT-OUT SOURCE IS NOT THE SAME AS A TRAINING FRAME, and that difference is
#: the reason to use one. A cut-out carries only the pixels somebody drew a mask
#: around; whatever the annotator MISSED in that frame never enters the dataset
#: at all. Used as a whole frame the same miss trains as background, teaching
#: that a plant of that size is soil.
#:
#: So a drive whose annotation is known to be incomplete belongs here and NOT in
#: mixed.py's INCLUDE_FRAMES. Run annotation/missed_plants.py to find out which
#: it is rather than guessing - that audit exists to decide exactly this.
#:
#: The cost of cut-out-only: real weed-beside-weed context, real dense-patch
#: lighting and the drive's own soil all stay behind. Prefer whole frames when
#: the audit says they are clean.
WEED_SOURCES = [
    r"E:\Dataset_Vidalia\Weeds_20260108_3_good\sessions\vid2_20260108_122731",
    r"E:\Dataset_Vidalia\Weeds_20260108_1\sessions\vid3_20260108_110444",
]

#: Which frames of those sessions were actually corrected, in the same
#: `<session>:<range>` form make_dataset uses. Empty means every frame, which is
#: right only when the whole export was reviewed.
#:
#: vid3_20260108_110444 has 326 frames and 75 corrected ones. Cutting instances
#: out of the other 251 would build the bank from SAM's own guesses, and a
#: composite made from a machine mask is a machine mask with extra steps.
SOURCE_FRAMES = ("vid2_20260108_122731:*,"
                 "vid3_20260108_110444:1-75")

#: WHERE THE BACKGROUNDS COME FROM. Onion drives - real soil, real rows, real
#: crop geometry. Every one is screened before use; see UNCLAIMED_BLOBS_MAX.
ONION_BACKGROUNDS = [
    r"E:\Dataset_Vidalia\onions_20260108_1\sessions",
]

#: Where the composites are written: a session-shaped folder the dataset build
#: can consume directly (rgb/ + annotations/default.json, Datumaro 1.0).
OUT_ROOT = r"E:\Dataset_Vidalia\synthetic"

#: HOW MANY. Start small and look at them. A thousand composites nobody opened
#: is a thousand chances to train on a systematic artefact.
N_IMAGES = 200

#: THE POINT OF THE MODULE. Fractions must be > 0 in the bands you care about;
#: they are normalised, so relative sizes are what matter.
#:
#: Weighted hard on purpose. `isolated` teaches weed appearance, which the weed
#: sessions already teach perfectly well; `touching` and `overlap` teach the
#: decision that nothing in the recordings teaches at all.
CONTACT_MIX = {
    "isolated": 0.10,     # > 100 px from any onion
    "near": 0.15,         # 30-100 px
    "very_near": 0.25,    # 4-30 px
    "touching": 0.30,     # within 4 px, no overlap
    "overlap": 0.20,      # foliage overlaps
}

#: Band edges in pixels, matched to CONTACT_MIX. Distance is measured from the
#: pasted weed's mask to the nearest annotated onion pixel.
#: `touching` was ONE PIXEL WIDE and it did not work. The first real run asked
#: for 30% touching and achieved 4%, rejecting 145 of 159 attempts - while
#: very_near and overlap came out over target by exactly the amount that went
#: missing, because that is where the failed attempts landed.
#:
#: A one-pixel window is not a physical distinction at 2208 px. Two plants whose
#: foliage is 2 px apart are touching by any reading, and mask quantisation
#: alone moves a boundary further than that. The band is now 4 px wide, which
#: still means contact and can actually be hit.
CONTACT_BANDS = {
    "isolated": (100, 10 ** 9),
    "near": (30, 100),
    "very_near": (4, 30),
    "touching": (0, 4),
    "overlap": (-10 ** 9, 0),
}

#: How many weeds per composite. Real mixed frames are not one-weed scenes, and
#: a model trained only on those learns that a frame contains exactly one.
#:
#: RAISED FROM (1, 4). A composite carries every onion its background held -
#: about 18 - so at a mean of 2.5 weeds the built dataset came out 30:1 crop to
#: weed, and the three thin weed classes could not be measured at all. A mean
#: of 4 is the cheapest lever on that ratio; N_IMAGES is the other, and the
#: last run used 200 of 209 backgrounds it had merely LOOKED at, not all it
#: has, so more composites are available if this is not enough.
#:
#: Do not push this far past what a field looks like. The model learns scene
#: statistics as well as shapes, and a frame with fifteen weeds among eighteen
#: onions teaches a weed density that no drive will ever show it.
WEEDS_PER_IMAGE = (2, 6)

#: MOST cut-outs to hold in the bank, or 0 for every one there is.
#:
#: This was buried as a function default of 600 while the two source drives
#: hold about 3,300 annotated instances - so five sixths of the hand-drawn
#: weeds in the project were being thrown away before compositing started, by
#: a number nobody could see. It belongs here with the rest of the knobs.
#:
#: A bigger bank is the second half of the reuse fix: cut-outs are drawn
#: without replacement, so reuse only begins once the pastes outnumber the
#: bank. With 800 pastes and a bank of 3,300, it never does.
BANK_MAX = 0

#: A pasted weed may hide at most this fraction of an onion instance. Past it
#: the onion becomes a sliver the annotation cannot honestly describe, and the
#: composite is discarded rather than written with a mangled crop mask.
MAX_ONION_HIDDEN = 0.35

#: Fraction of the pasted weed that may be hidden by onion foliage in front of
#: it. A weed with almost all of its body behind a leaf is not a useful
#: supervised instance.
MAX_WEED_HIDDEN = 0.35

#: How often the pasted weed is drawn IN FRONT of the onion where they overlap.
#: Not 1.0 on purpose: a generator that always puts the paste on top teaches
#: "the pasted object is always in front", which is a property of the generator
#: and not of a field.
WEED_IN_FRONT_P = 0.7

#: THE SAFETY GATE. How many plant-shaped patches of vegetation may sit outside
#: every annotation before a background is refused.
#:
#: A patch nobody labelled is a plant nobody labelled. Compositing onto such a
#: frame teaches that an unlabelled plant is background while an identical
#: pasted one is a target - precisely the confusion this module exists to
#: remove.
#:
#: 1 rather than 0 because the vegetation prior calls moss, algae and green
#: debris vegetation, so a single patch is as likely to be a fleck as a plant.
#: Two or more is a pattern.
UNCLAIMED_BLOBS_MAX = 1

#: Pixels of alpha ramp at the cut-out edge. 1-2 keeps the plant interior
#: untouched while removing the hard sticker edge a network will happily learn.
FEATHER_PX = 2

#: Match the cut-out's brightness to the patch it lands on, at most this far.
#: Mild on purpose: enough to remove an obvious exposure jump, not enough to
#: recolour a species out of recognition.
ILLUMINATION_MATCH = 0.6

#: Scale jitter. OFF by default and that is deliberate: the sources and the
#: backgrounds come from the same rig at the same height, so a pasted weed is
#: already at the right physical size, and random rescaling would invent plants
#: at sizes that never grow. Widen it only with a metric reason.
SCALE_JITTER = (1.0, 1.0)

#: Rotation, degrees. Plants have no canonical heading seen from above.
ROTATION_DEG = 180

#: Deterministic output. Same seed, same composites.
SEED = 1234

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalised_mix(mix=None):
    """CONTACT_MIX as fractions summing to 1, dropping the bands set to zero."""
    m = {k: float(v) for k, v in (mix or CONTACT_MIX).items() if float(v) > 0}
    total = sum(m.values())
    if not total:
        raise SystemExit(
            "ERROR: CONTACT_MIX is all zeros - nothing to generate.\n"
            "The bands that matter are 'touching' and 'overlap': they are the "
            "only ones the recordings do not already contain.")
    return {k: v / total for k, v in m.items()}


def band_plan(n, mix=None, bands=None):
    """`n` band names in the requested proportions, worst-case rounded up.

    Returned as a list rather than sampled per image so the achieved mix is the
    requested one rather than a draw from it - with 200 images a 10% band drawn
    independently can come out at 5%."""
    mix = normalised_mix(mix)
    known = set(bands or CONTACT_BANDS)
    unknown = sorted(set(mix) - known)
    if unknown:
        raise SystemExit(
            f"ERROR: CONTACT_MIX names band(s) with no definition in "
            f"CONTACT_BANDS: {', '.join(unknown)}.\n"
            f"Known: {', '.join(sorted(known))}")
    out = []
    for name, frac in sorted(mix.items()):
        out += [name] * int(round(frac * n))
    while len(out) < n:
        out.append(max(mix, key=mix.get))
    return out[:n]


def unclaimed_vegetation(veg, claimed):
    """Fraction of vegetation no annotated mask covers.

    The number the background screen is built on. 0 means every green pixel is
    accounted for; 1 means nothing in the frame was annotated at all."""
    veg = np.asarray(veg, bool)
    total = int(veg.sum())
    if not total:
        return 0.0
    return float((veg & ~np.asarray(claimed, bool)).sum()) / total


def background_ok(veg, claimed, max_blobs=UNCLAIMED_BLOBS_MAX):
    """(ok, n_blobs, reason) for one candidate background.

    A frame with unlabelled plants in it must never become a composite
    background: the composite would then hold a labelled weed and an unlabelled
    one side by side, teaching that the same plant is both a target and soil.
    That is worse than generating nothing.

    COUNTS PLANT-SHAPED PATCHES, not a fraction. The first real run rejected two
    backgrounds at 19% and 21% unclaimed - which is the signature of masks drawn
    a couple of pixels inside their leaves, not of a missing plant. A fraction is
    wrong in both directions here: 19% can mean nothing missing, and a genuine
    missed seedling among forty labelled plants is under 2%. missed_plants.py
    settled this on real geometry, and the two must not disagree about the same
    frame."""
    veg = np.asarray(veg, bool)
    if not veg.any():
        return False, 0, "no vegetation at all - nothing to composite against"
    n, _, _ = unclaimed_blobs(veg, claimed)
    if n > max_blobs:
        return False, n, (
            f"{n} plant-shaped patch(es) of vegetation are not covered by any "
            f"annotation, so something is growing there that nobody labelled")
    return True, n, ""


def contact_distance(weed_mask, onion_union):
    """Signed distance in pixels from a weed mask to the nearest onion pixel.

    Negative means they overlap, and the magnitude is the overlapping area's
    depth into the weed - so one number orders every band from `isolated`
    through `touching` to `overlap` and the caller needs no special cases."""
    import cv2
    w = np.asarray(weed_mask, bool)
    o = np.asarray(onion_union, bool)
    if not w.any():
        return float("inf")
    if not o.any():
        return float("inf")
    overlap = int((w & o).sum())
    if overlap:
        # Depth of the overlap, so a graze and a burial are not the same band.
        inside = cv2.distanceTransform((w & o).astype(np.uint8), cv2.DIST_L2, 3)
        return -float(inside.max())
    dist = cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[w].min())


def band_of(distance, bands=None):
    """Which contact band a measured distance falls in, or None."""
    for name, (lo, hi) in (bands or CONTACT_BANDS).items():
        if lo <= distance < hi:
            return name
    return None


def feathered_alpha(mask, px=FEATHER_PX):
    """Alpha in [0,1]: 1 inside, ramping to 0 over `px` at the boundary.

    The interior is left exactly 1. A blend that softens the whole instance
    would hand the model a cue that pasted plants are blurry."""
    import cv2
    m = np.asarray(mask, bool).astype(np.uint8)
    if px <= 0 or not m.any():
        return m.astype(np.float32)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return np.clip(d / float(px), 0.0, 1.0).astype(np.float32)


def match_illumination(patch, target_patch, mask, strength=ILLUMINATION_MATCH):
    """Nudge a cut-out's brightness toward the background it lands on.

    Multiplicative on all three channels, so hue is preserved: a species is
    partly identified by colour, and correcting exposure must not recolour the
    plant into a different one."""
    m = np.asarray(mask, bool)
    if not m.any() or strength <= 0:
        return patch
    src = float(np.asarray(patch, np.float32)[m].mean())
    dst = float(np.asarray(target_patch, np.float32)[m].mean()) if m.any() else src
    if src <= 1e-6:
        return patch
    gain = 1.0 + strength * ((dst / src) - 1.0)
    gain = float(np.clip(gain, 0.6, 1.6))
    return np.clip(np.asarray(patch, np.float32) * gain, 0, 255).astype(np.uint8)


def transform_cutout(rgb, mask, scale=1.0, rotation_deg=0.0, lep=None):
    """Rotate and scale a cut-out, carrying its LEP through the same transform.

    The LEP travels because it is the one annotation a composite gets for free
    and could not otherwise buy: Stage B needs growth points, and a pasted weed
    already has one if a person placed it."""
    import cv2
    m = np.asarray(mask, bool).astype(np.uint8)
    h, w = m.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, float(rotation_deg), float(scale))
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - centre[0]
    M[1, 2] += nh / 2.0 - centre[1]
    out_rgb = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderValue=(0, 0, 0))
    out_m = cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_NEAREST,
                           borderValue=0).astype(bool)
    out_lep = None
    if lep is not None:
        p = np.array([float(lep[0]), float(lep[1]), 1.0])
        out_lep = (float(M[0] @ p), float(M[1] @ p))
    return out_rgb, out_m, out_lep


def paste(bg, cutout_rgb, cutout_mask, top_left, feather=FEATHER_PX,
          illumination=ILLUMINATION_MATCH):
    """Composite one cut-out into a background. Returns (image, placed_mask).

    `placed_mask` is full-frame, so every later step - contact distance, z-order,
    the annotation itself - reads the same array rather than re-deriving where
    the instance ended up."""
    out = np.array(bg, copy=True)
    H, W = out.shape[:2]
    h, w = np.asarray(cutout_mask).shape[:2]
    y0, x0 = int(top_left[0]), int(top_left[1])
    ys0, xs0 = max(0, -y0), max(0, -x0)
    y0, x0 = max(0, y0), max(0, x0)
    ys1 = min(h, ys0 + (H - y0))
    xs1 = min(w, xs0 + (W - x0))
    if ys1 <= ys0 or xs1 <= xs0:
        return out, np.zeros((H, W), bool)

    sub_m = np.asarray(cutout_mask, bool)[ys0:ys1, xs0:xs1]
    sub_rgb = np.asarray(cutout_rgb)[ys0:ys1, xs0:xs1]
    y1, x1 = y0 + sub_m.shape[0], x0 + sub_m.shape[1]
    dst = out[y0:y1, x0:x1]

    src = match_illumination(sub_rgb, dst, sub_m, illumination)
    a = feathered_alpha(sub_m, feather)[..., None]
    out[y0:y1, x0:x1] = np.clip(src * a + dst * (1.0 - a), 0, 255).astype(np.uint8)

    placed = np.zeros((H, W), bool)
    placed[y0:y1, x0:x1] = sub_m
    return out, placed


def session_id_for(folder_name):
    """The session id the frames in a composite run will carry.

    Exported rather than kept inside main() because mixed.py has to name this
    session in three places - the source path, INCLUDE_FRAMES and SCENE_HINTS -
    and a build that types the id by hand gets it subtly wrong. This project has
    already lost a build to a folder name and a session id disagreeing, and the
    id here is NOT the folder name: `synth_mixed_20260904_0156` holds frames
    called `synth_20260904_015600_000001.png`, because a session id needs
    seconds to parse as one."""
    stamp = str(folder_name).replace("synth_mixed_", "")
    return f"synth_{stamp}00" if len(stamp) == 13 else f"synth_{stamp}"


def placed_mask(mask, top_left, shape):
    """A cut-out's mask at a frame offset, clipped to the frame."""
    H, W = int(shape[0]), int(shape[1])
    m = np.asarray(mask, bool)
    h, w = m.shape[:2]
    y0, x0 = int(top_left[0]), int(top_left[1])
    ys0, xs0 = max(0, -y0), max(0, -x0)
    y0, x0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(h, ys0 + (H - y0)), min(w, xs0 + (W - x0))
    out = np.zeros((H, W), bool)
    if ys1 > ys0 and xs1 > xs0:
        sub = m[ys0:ys1, xs0:xs1]
        out[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
    return out


def band_target(band, bands=None):
    """A distance to aim at inside a band, rather than at its edge.

    Aiming at an edge and landing a pixel the wrong side of it is a rejected
    placement; aiming at the middle leaves room on both sides. The unbounded
    bands get a sensible finite point instead of infinity."""
    lo, hi = (bands or CONTACT_BANDS)[band]
    if hi > 10 ** 8:
        return float(lo) + 50.0
    if lo < -10 ** 8:
        return -5.0
    return (float(lo) + float(hi)) / 2.0


def refine_offset(union, dist, grad, mask_t, top_left, band, bands=None,
                  steps=3, max_step_px=80):
    """Nudge a placement toward the band it was asked for.

    THE HIT RATE IS THE WHOLE POINT. Anchoring a plant's crown at a pixel whose
    own distance-to-onion is in the band does not put the PLANT in the band: the
    foliage extends outward, so whether the nearest part lands at 2 px or
    overlapping depends on which way the onion happens to lie. The first run
    with a usable band width still rejected 95 of 154 touching attempts.

    So measure and correct. The distance transform's gradient points away from
    the onion, so stepping along it by the error moves the weed's nearest point
    to roughly the distance asked for, and two or three steps converge - which
    turns a coin flip into a placement.

    Returns a corrected top-left, or None if it wandered out of frame."""
    target = band_target(band, bands)
    H, W = dist.shape[:2]
    gy, gx = grad
    top, left = int(top_left[0]), int(top_left[1])
    for _ in range(max(1, steps)):
        placed = placed_mask(mask_t, (top, left), (H, W))
        if not placed.any():
            return None
        d = contact_distance(placed, union)
        if band_of(d, bands) == band:
            return (top, left)
        # Reference point: where the weed is nearest the onion, or its centre
        # once they overlap and "nearest" is inside.
        ys, xs = np.nonzero(placed)
        if d >= 0:
            k = int(np.argmin(dist[ys, xs]))
            ry, rx = int(ys[k]), int(xs[k])
        else:
            ry, rx = int(ys.mean()), int(xs.mean())
        vy, vx = float(gy[ry, rx]), float(gx[ry, rx])
        norm = (vy * vy + vx * vx) ** 0.5
        if norm < 1e-6:
            return None
        step = float(np.clip(d - target, -max_step_px, max_step_px))
        top -= int(round(step * vy / norm))
        left -= int(round(step * vx / norm))
    placed = placed_mask(mask_t, (top, left), (H, W))
    if not placed.any():
        return None
    return (top, left) if band_of(contact_distance(placed, union),
                                  bands) == band else None


def placement_candidates(onion_union, shape, band, rng, n=40,
                         bands=None, margin=8):
    """Top-left positions to try for a cut-out of `shape`, biased to `band`.

    Candidates come from the distance transform: for a band that wants 30-100 px
    of clearance, positions whose own distance-to-onion is in that range are
    where a mask of that size can plausibly land. It is a bias, not a guarantee -
    the achieved distance is measured afterwards and the placement is kept or
    rejected on that."""
    import cv2
    o = np.asarray(onion_union, bool)
    H, W = o.shape[:2]
    h, w = int(shape[0]), int(shape[1])
    if h >= H or w >= W:
        return []
    lo, hi = (bands or CONTACT_BANDS)[band]
    dist = cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3)

    if hi <= 0:                      # overlap: anchor inside an onion
        ys, xs = np.nonzero(o)
    else:
        want = (dist >= max(0.0, lo)) & (dist < min(hi, float(dist.max()) + 1))
        ys, xs = np.nonzero(want)
    if not len(ys):
        return []

    keep = rng.sample(range(len(ys)), min(n, len(ys)))
    out = []
    for i in keep:
        # The anchor is where the plant MEETS THE GROUND, not the corner of its
        # bounding box: a crown placed sensibly with leaves extending outward is
        # a plant, and a bounding box centred anywhere is a sticker.
        top = int(ys[i]) - h + max(1, h // 6)
        left = int(xs[i]) - w // 2
        top = int(np.clip(top, -margin, H - h + margin))
        left = int(np.clip(left, -margin, W - w + margin))
        out.append((top, left))
    return out


def visible_masks(onion_masks, weed_mask, weed_in_front):
    """Masks as the camera would see them once one plant is behind the other.

    Instance segmentation is annotated on what is VISIBLE, so whichever plant is
    behind loses the overlapping pixels. Getting this backwards would paint an
    onion's pixels as weed - and that is the error the whole system is built to
    avoid, arriving through the training data instead of the model."""
    weed = np.asarray(weed_mask, bool)
    if weed_in_front:
        return [np.asarray(m, bool) & ~weed for m in onion_masks], weed
    behind = np.zeros_like(weed)
    for m in onion_masks:
        behind |= np.asarray(m, bool)
    return [np.asarray(m, bool) for m in onion_masks], weed & ~behind


def hidden_fraction(before, after):
    """How much of an instance the composite took away."""
    b = int(np.asarray(before, bool).sum())
    if not b:
        return 0.0
    return 1.0 - int(np.asarray(after, bool).sum()) / b


def summarise(records):
    """Achieved contact bands and rejections, which is what makes this run
    checkable rather than merely repeatable."""
    bands, rejects = {}, {}
    for r in records:
        bands[r.get("band") or "?"] = bands.get(r.get("band") or "?", 0) + 1
    return {"instances": len(records), "bands": dict(sorted(bands.items())),
            "rejects": rejects}


def reuse_note(records, bank_size=None):
    """How much of this set is genuinely different plants, at two granularities.

    CUT-OUTS is the strict one: the same cut-out pasted twice is the same
    pixels, moved and rotated. Drawing without replacement holds it at one
    each, and repeats mean only that the pastes outnumbered the bank.

    SOURCE FRAMES is the honest one, and it is always the smaller number. The
    weed drives are VIDEO: consecutive frames show the same ground, so one
    physical weed becomes a fresh instance in every frame it appears in, and a
    bank of 3,842 cut-outs drawn from 131 frames is nothing like 3,842 plants.
    No id in this pipeline can tell which instances are the same plant, so the
    frame count is the closest available bound - and the true number of plants
    is below it, not above."""
    srcs = [r.get("source") for r in records if r.get("source")]
    if not srcs:
        return []
    uniq = set(srcs)
    counts = {s: srcs.count(s) for s in uniq}
    repeated = sum(1 for n in counts.values() if n > 1)
    most = max(counts.values())
    L = ["", f"  Cut-out reuse: {len(srcs)} paste(s) from {len(uniq)} distinct "
             f"cut-out(s)"
             + (f" (bank held {bank_size})" if bank_size else ""),
         f"    {repeated} cut-out(s) pasted more than once; the most-reused "
         f"appears {most} time(s)"]
    frames = {r.get("source_frame") for r in records if r.get("source_frame")}
    if frames:
        L += [f"    drawn from {len(frames)} source frame(s) of video, so the "
              f"number of DISTINCT",
              f"    PLANTS is below that - one weed recurs in every frame it "
              f"was driven past."]
    if repeated:
        L += ["  [!] a reused cut-out is the SAME PIXELS in several frames, and "
              "the frame-block",
              "      split does not know that - so the same plant can sit in "
              "train and in val,",
              "      where it measures memorisation. Cut-outs are drawn without "
              "replacement, so",
              "      this means the pastes OUTNUMBERED THE BANK: raise "
              "BANK_MAX, lower",
              "      WEEDS_PER_IMAGE, or split on the 'source_instance' "
              "attribute in the",
              "      annotations before quoting a weed score."]
    else:
        L += ["  [i] every paste is a different cut-out, so no weed's exact "
              "pixels appear on",
              "      both sides of a split. Plants recurring across source "
              "frames still can -",
              "      which is one more reason never to validate on these."]
    return L


def format_report(n_images, n_inst, bands, rejected, backgrounds_seen,
                  backgrounds_used, out_dir=None, records=None,
                  bank_size=None):
    L = ["", "  Composited mixed scenes", "  " + "-" * 40,
         f"  {n_images} image(s), {n_inst} pasted weed instance(s)",
         f"  backgrounds: {backgrounds_used} used of {backgrounds_seen} "
         f"screened", ""]
    if bands:
        L.append("  Achieved contact band (measured, not requested):")
        for name, n in sorted(bands.items(), key=lambda kv: -kv[1]):
            share = n / max(1, n_inst)
            L.append(f"    {name:<14}{n:>7}{share:>8.0%}")
    if rejected:
        L += ["", "  Rejected:"]
        for why, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
            L.append(f"    {n:>6}  {why}")
    L += reuse_note(records or [], bank_size)
    L += ["",
          "  [i] provenance is SYNTHETIC. Train on these; never validate on",
          "      them - a score computed on composites measures this "
          "generator's",
          "      blind spots, not the model.",
          "  [i] the onion masks these were composited against are SAM "
          "prelabels,",
          "      so 'distance to crop' is a distance to a machine label."]
    if out_dir:
        L += ["", f"  -> {out_dir}"]
    return "\n".join(L + [""])


def datumaro_doc(items, classes=None):
    """A Datumaro 1.0 document the dataset builder can read directly.

    Datumaro rather than COCO because prepare_dataset refuses COCO, and it is
    right to: COCO cannot carry shape groups, so every mask-to-LEP link would be
    silently discarded - and a pasted weed's LEP is the annotation compositing
    gets for free."""
    names = list(classes or CLASSES) + [LEP_LABEL]
    return {
        "info": {"description": "SeeWeed3D SYNTHETIC COMPOSITE - weed cut-outs "
                                "pasted into real onion frames. Not a "
                                "recording. Train only; never validate.",
                 "label_provenance": "synthetic",
                 "date_created": datetime.now(timezone.utc).isoformat()},
        "categories": {"label": {"labels": [{"name": n, "parent": "",
                                             "attributes": []} for n in names],
                                 "attributes": []}},
        "items": items,
    }


def _item(item_id, image_name, h, w, instances, classes=None):
    """One Datumaro item: polygons, plus a grouped LEP point where one rode in
    with the cut-out."""
    from annotation.mine_pool import mask_to_polygons

    names = list(classes or CLASSES) + [LEP_LABEL]
    idx = {n: i for i, n in enumerate(names)}
    anns, ann_id, group = [], 1, 1
    for inst in instances:
        polys = mask_to_polygons(inst["mask"])
        if not polys:
            continue
        for poly in polys:
            anns.append({"id": ann_id, "type": "polygon",
                         "label_id": idx[inst["class_name"]], "group": group,
                         "points": [float(v) for v in poly], "z_order": 0,
                         "attributes": dict(inst.get("attributes") or {})})
            ann_id += 1
        if inst.get("lep") is not None:
            x, y = inst["lep"]
            anns.append({"id": ann_id, "type": "points",
                         "label_id": idx[LEP_LABEL], "group": group,
                         "points": [float(x), float(y)], "z_order": 0,
                         "attributes": {"lep_visibility": "visible"}})
            ann_id += 1
        group += 1
    return {"id": item_id, "annotations": anns,
            "image": {"path": image_name, "size": [int(h), int(w)]}}


def _find_image(stem, roots):
    for root in roots:
        d = Path(root)
        if not d.is_dir():
            continue
        for suf in sorted(IMAGE_SUFFIXES):
            p = d / f"{stem}{suf}"
            if p.is_file():
                return p
    return None


def load_bank(sources, include_frames="", max_instances=BANK_MAX, rng=None):
    """Hand-drawn weed cut-outs: RGB, mask, class, attributes and LEP.

    Read through load_datumaro rather than by re-parsing, so a cut-out carries
    exactly the attributes and the mask-to-LEP link the annotator produced -
    including the contract fields the dataset build will later check.

    max_instances of 0 keeps every one. It used to default to 600 with the two
    source drives holding some 3,300 between them, which discarded five sixths
    of the project's hand-drawn weeds before compositing began."""
    import cv2
    from training import prepare_dataset as pdz
    from training import datumaro_multitask as dmm

    bank = []
    for root in sources:
        root = Path(root)
        if not root.exists():
            print(f"  [!] source does not exist, skipped: {root}")
            continue
        for f in pdz.find_annotation_files(root):
            frames, _ = dmm.load_datumaro(
                f, fallback_session=dmm.batch_session_id(f))
            # NARROW THE SPEC TO THIS EXPORT. select_frames refuses a spec
            # naming a session it cannot see - correct for a build, where every
            # export is merged first, and wrong here: SOURCE_FRAMES names both
            # drives, so applying it whole to vid2's export errored on vid3.
            sub = pdz.spec_for_sessions(
                include_frames, {r.session_id for r in frames})
            if sub:
                frames, _ = pdz.select_frames(frames, sub, None)
            for rec in frames:
                img = _find_image(Path(rec.image_path or rec.item_id).stem,
                                  [root / "rgb", root,
                                   Path(rec.image_path).parent])
                if img is None:
                    continue
                bgr = cv2.imread(str(img))
                if bgr is None:
                    continue
                H, W = bgr.shape[:2]
                for i, inst in enumerate(rec.instances):
                    if inst.class_name == CROP_CLASS:
                        continue
                    m = _mask_of(inst, H, W)
                    ys, xs = np.nonzero(m)
                    if len(xs) < 3:
                        continue
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    lep = None
                    if inst.lep is not None:
                        lep = (float(inst.lep.x) - x0, float(inst.lep.y) - y0)
                    bank.append({
                        "rgb": bgr[y0:y1, x0:x1].copy(),
                        "mask": m[y0:y1, x0:x1].copy(),
                        "class_name": inst.class_name,
                        "attributes": dict(inst.attributes or {}),
                        "lep": lep,
                        #: THE INSTANCE, not the frame. Without the index
                        #: every weed in one source frame shared an id, so
                        #: the reuse report counted source FRAMES and read
                        #: 794 pastes from 131 plants on a run where every
                        #: paste really was a different cut-out.
                        "source": f"{rec.session_id}/{rec.item_id}#{i}",
                        "source_frame": f"{rec.session_id}/{rec.item_id}",
                    })
    if rng and max_instances and len(bank) > max_instances:
        bank = rng.sample(bank, max_instances)
    return bank


class CutoutDrawer:
    """Draws every hand-drawn plant once before it draws any of them twice.

    rng.choice over the bank samples WITH REPLACEMENT, so 506 pastes were about
    340 distinct plants and roughly 120 of them appeared in several composites -
    the same pixels, moved and rotated. Frame-block splitting cannot see that:
    it separates by frame index, and composites have no video order, so the
    duplicates scattered at random across train, val and test and a memorised
    weed became part of what the model was scored on.

    Sampling without replacement removes it outright rather than making it less
    likely, which is what a bigger bank alone would do. Reuse begins only when
    the pastes outnumber the bank, and `cycles` counts how often that happened
    so the run can say so instead of the next tool discovering it."""

    def __init__(self, bank, rng):
        self._bank = list(bank)
        self._rng = rng
        self._pool = []
        self.cycles = 0

    def __len__(self):
        return len(self._bank)

    def draw(self):
        if not self._pool:
            if not self._bank:
                raise IndexError("the cut-out bank is empty")
            self._pool = list(self._bank)
            self._rng.shuffle(self._pool)
            self.cycles += 1
        return self._pool.pop()


def _mask_of(inst, h, w):
    import cv2
    m = np.zeros((h, w), np.uint8)
    for poly in inst.polygons:
        pts = np.asarray(poly, np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def compose_one(bgr, onion_masks, cutouts, bands_wanted, rng, cfg=None):
    """One composite. Returns (image, instances, records) or (None, None, why).

    Pure apart from the RNG: every rejection is a returned reason rather than a
    silent skip, because a generator that quietly drops most of its attempts
    produces a dataset whose composition nobody can account for."""
    import cv2
    c = dict(cfg or {})
    out = np.array(bgr, copy=True)
    onions = [np.asarray(m, bool) for m in onion_masks]
    union = np.zeros(out.shape[:2], bool)
    for m in onions:
        union |= m
    # Computed ONCE for the frame: the distance to the nearest onion, and its
    # gradient, which is what lets a placement be corrected instead of retried
    # blindly.
    dist = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, 3)
    grad = (cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3),
            cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3))
    placed, records = [], []

    #: A bare list still works and draws without replacement WITHIN the frame;
    #: only a drawer shared across frames can promise it for the whole run, and
    #: main() makes exactly one.
    drawer = cutouts if hasattr(cutouts, "draw") else CutoutDrawer(cutouts, rng)

    for band in bands_wanted:
        cut = drawer.draw()
        scale = rng.uniform(*c.get("scale_jitter", SCALE_JITTER))
        rot = rng.uniform(-c.get("rotation_deg", ROTATION_DEG),
                          c.get("rotation_deg", ROTATION_DEG))
        rgb_t, mask_t, lep_t = transform_cutout(cut["rgb"], cut["mask"],
                                                scale, rot, cut.get("lep"))
        spots = placement_candidates(union, mask_t.shape, band, rng)
        if not spots:
            records.append({"band": None, "reason": f"no {band} position"})
            continue

        for spot in spots:
            # Correct the placement toward the band before judging it. Anchoring
            # a crown at a band pixel does not put the PLANT in the band, and
            # rejecting on that alone threw away most touching attempts.
            spot = refine_offset(union, dist, grad, mask_t, spot, band)
            if spot is None:
                continue
            top, left = spot
            trial, pmask = paste(out, rgb_t, mask_t, (top, left),
                                 c.get("feather", FEATHER_PX),
                                 c.get("illumination", ILLUMINATION_MATCH))
            if not pmask.any():
                continue
            achieved = contact_distance(pmask, union)
            if band_of(achieved) != band:
                continue
            front = rng.random() < c.get("weed_in_front_p", WEED_IN_FRONT_P)
            vis_onions, vis_weed = visible_masks(onions, pmask, front)
            if hidden_fraction(pmask, vis_weed) > c.get("max_weed_hidden",
                                                        MAX_WEED_HIDDEN):
                continue
            if any(hidden_fraction(a, b) > c.get("max_onion_hidden",
                                                 MAX_ONION_HIDDEN)
                   for a, b in zip(onions, vis_onions)):
                continue

            out = trial if front else _paste_behind(out, trial, pmask, union)
            onions = vis_onions
            #: WHICH HAND-DRAWN INSTANCE THIS IS, carried into the dataset.
            #: Cut-outs are drawn WITH REPLACEMENT, so the same plant - the
            #: same pixels, at a different offset and rotation - can land in
            #: several composites. Frame-block splitting cannot see that: it
            #: separates by frame index, and a composite set has no video
            #: order for an index to mean anything. So the same weed can sit
            #: in train and in val, where it measures memorisation exactly.
            #: Recording it in the annotation is what lets a split honour it.
            placed.append({"class_name": cut["class_name"], "mask": vis_weed,
                           "attributes": {**cut["attributes"],
                                          "source_instance": cut["source"]},
                           "lep": None if lep_t is None else
                           (lep_t[0] + left, lep_t[1] + top)})
            records.append({"band": band, "distance_px": round(achieved, 2),
                            "class_name": cut["class_name"],
                            "in_front": bool(front), "source": cut["source"],
                            "source_frame": cut.get("source_frame")})
            break
        else:
            records.append({"band": None,
                            "reason": f"no {band} placement passed the checks"})

    if not placed:
        return None, None, records
    instances = [{"class_name": CROP_CLASS, "mask": m, "attributes": {}}
                 for m in onions if m.any()] + placed
    return out, instances, records


def _paste_behind(original, pasted, weed_mask, onion_union):
    """Keep onion pixels in front where the two overlap."""
    out = np.array(pasted, copy=True)
    behind = np.asarray(weed_mask, bool) & np.asarray(onion_union, bool)
    out[behind] = np.asarray(original)[behind]
    return out


#: Overlay colours, BGR. The crop is orange and drawn thickest, as everywhere
#: else in this project: it is the one thing in the frame that must not be hit.
OVERLAY_CROP = (0, 165, 255)
OVERLAY_WEED = (60, 220, 60)

#: NO BAND MAY LOOK LIKE THE CROP, and none may look like soil. The first
#: version drew `touching` in (0,140,255) - which is the crop's orange to
#: within a shade, and `touching` is 30% of the weeds - and `near`/`isolated`
#: in grey, on grey-green soil. Over half the pasted weeds were camouflaged,
#: and the composites read as one-weed scenes when they hold four.
#: Yellow is out for the same reason: (0,255,255) against the crop's
#: (0,165,255) differs in one channel by 90, which is a shade of the same
#: orange under a canopy shadow.
OVERLAY_BAND = {"overlap": (60, 60, 235),      # red     - on the crop
                "touching": (255, 0, 255),     # magenta - against it
                "very_near": (255, 255, 0),    # cyan
                "near": (255, 128, 0),         # blue
                "isolated": (60, 220, 60)}     # green   - alone


def instance_groups(item, classes=None):
    """A written item's instances, as (class_name, [polygon]) in group order.

    Groups, not annotations: one mask can become several polygons when an onion
    in front splits a pasted weed in two, and drawing those as separate
    instances would show four weeds where the annotation says two."""
    names = list(classes or CLASSES) + [LEP_LABEL]
    by_group = {}
    for a in item.get("annotations", []):
        if a.get("type") != "polygon":
            continue
        name = names[a["label_id"]]
        by_group.setdefault(a.get("group", 0), (name, []))[1].append(a["points"])
    return [by_group[g] for g in sorted(by_group)]


def bands_by_item(report):
    """{item stem: [band per pasted weed, in the order they were placed]}."""
    out = {}
    for r in report.get("records") or []:
        if r.get("band") and r.get("item"):
            out.setdefault(r["item"], []).append(r["band"])
    return out


def draw_composite(bgr, groups, bands=None, scale=1.0, labels=True):
    """One composite with its own annotations drawn on it.

    Colours the weeds BY ACHIEVED CONTACT BAND rather than by class, because
    the thing to check in a composite is not what the plant is - the cut-out
    carries a hand-drawn class - but whether the placement the report claims is
    the placement you can see. A weed labelled `touching` sitting alone in bare
    soil is a generator bug that no other output would show you."""
    import cv2
    out = np.ascontiguousarray(bgr.copy())
    weed_i = 0
    for name, polys in groups:
        crop = name == CROP_CLASS
        if crop:
            colour, thick = OVERLAY_CROP, 4
            text = name
        else:
            band = (bands or [])[weed_i] if weed_i < len(bands or []) else None
            colour = OVERLAY_BAND.get(band, OVERLAY_WEED)
            thick, weed_i = 2, weed_i + 1
            text = f"{name} [{band}]" if band else name
        cnts = [np.asarray(p, np.float32).reshape(-1, 2).round().astype(np.int32)
                for p in polys]
        cv2.polylines(out, cnts, True, colour, thick)
        if labels and cnts is not None and len(cnts):
            p = min((c.reshape(-1, 2) for c in cnts),
                    key=lambda a: (a[:, 1].min(), a[:, 0].min()))
            x, y = int(p[:, 0].min()), int(p[:, 1].min())
            cv2.putText(out, text, (x, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (x, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    if scale and scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return out


def polygon_area(points):
    """Shoelace area of one flat [x, y, x, y, ...] polygon, in pixels."""
    a = np.asarray(points, float).reshape(-1, 2)
    if len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def count_note(per_frame, areas):
    """How many weeds are actually in these frames, and how big they are.

    "It looks like one or two weeds a frame" is a question about the PICTURE,
    and it has two very different answers: the generator pasted one or two, or
    it pasted four and you cannot see them. Only counting separates those, and
    the second is the one that has been true here - half the palette used to
    disappear into the crop colour and the soil.

    Area is reported because a pasted cut-out is not scaled: a real cotyledon
    from the source drive is a few hundred pixels in a 2.7 megapixel frame, and
    at the default overlay scale of 0.5 its outline is a mark you can miss."""
    if not per_frame:
        return []
    n = len(per_frame)
    tot = sum(per_frame)
    L = ["", f"  Weeds drawn: {tot} across {n} frame(s) "
             f"(mean {tot / n:.2f}, min {min(per_frame)}, max {max(per_frame)})"]
    if areas:
        a = sorted(areas)
        med = a[len(a) // 2]
        tiny = sum(1 for v in a if v < 1500)
        L.append(f"    weed area px: median {med:.0f}, smallest {a[0]:.0f}, "
                 f"largest {a[-1]:.0f}")
        if tiny:
            L.append(f"    {tiny} of {len(a)} are under 1500 px - at overlay "
                     f"scale 0.5 those are a few pixels of outline.")
            L.append(f"    Pass --scale 1.0 before concluding a frame is "
                     f"empty.")
    return L


def render_overlays(run_dir, out_dir=None, limit=0, stride=1, scale=0.5):
    """Draw a finished run's own annotations onto its frames.

    Reads a run that already exists rather than drawing while compositing, so
    the composites you have can be looked at without regenerating them - and so
    looking is never a reason to change what was generated."""
    import cv2
    run = Path(run_dir)
    ann = run / "annotations" / "default.json"
    if not ann.exists():
        raise SystemExit(f"ERROR: {ann} not found. Point this at a "
                         f"synth_mixed_* run folder.")
    doc = json.loads(ann.read_text(encoding="utf-8"))
    classes = [c["name"] for c in
               doc.get("categories", {}).get("label", {}).get("labels", [])]
    rep_path = run / "compose_report.json"
    report = (json.loads(rep_path.read_text(encoding="utf-8"))
              if rep_path.exists() else {})
    bands = bands_by_item(report)
    if report and not bands:
        print("  [i] this run's report predates per-frame band records, so "
              "weeds are drawn in one colour.\n      Regenerate to colour them "
              "by achieved contact band.")

    items = sorted(doc.get("items", []), key=lambda i: str(i.get("id")))
    items = items[::max(1, int(stride))]
    if limit:
        items = items[:int(limit)]
    out = Path(out_dir or (run / "overlays"))
    out.mkdir(parents=True, exist_ok=True)

    n, per_frame, areas = 0, [], []
    for item in items:
        stem = str(item.get("id"))
        img = run / "rgb" / f"{stem}.png"
        bgr = cv2.imread(str(img))
        if bgr is None:
            print(f"  [!] missing image, skipped: {img}")
            continue
        groups = instance_groups(item, classes)
        pic = draw_composite(bgr, groups, bands.get(stem), scale)
        cv2.imwrite(str(out / f"{stem}.png"), pic)
        weeds = [g for g in groups if g[0] != CROP_CLASS]
        per_frame.append(len(weeds))
        areas += [polygon_area(p) for _, polys in weeds for p in polys]
        n += 1
    if not n:
        raise SystemExit(f"ERROR: nothing was drawn from {run}.")
    print("\n".join(count_note(per_frame, areas)))
    print(f"\n  {n} overlay(s) -> {out}\n"
          f"  crop is orange and thickest; weeds are coloured by the contact "
          f"band they achieved\n"
          f"  (red overlap, magenta touching, cyan very_near, blue near, "
          f"green isolated).\n"
          f"  WHAT TO LOOK FOR: a paste whose outline does not follow the "
          f"plant, a band label\n"
          f"  that disagrees with what you see, and light or shadow on the "
          f"weed that does not\n"
          f"  match the onions around it - none of which any number in the "
          f"report can show.\n")
    return 0


def main(argv=None):
    import argparse
    import cv2
    from evaluation.crop_risk import load_polygons, rasterise

    ap = argparse.ArgumentParser(description="Composite mixed scenes, or draw "
                                             "a finished run's annotations.")
    ap.add_argument("--overlays", metavar="RUN_DIR",
                    help="draw an existing synth_mixed_* run instead of "
                         "generating a new one")
    ap.add_argument("--out", help="where overlays go (default RUN_DIR/overlays)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.overlays:
        return render_overlays(a.overlays, a.out, a.limit, a.stride, a.scale)

    rng = random.Random(SEED)
    out_dir = Path(stamped(OUT_ROOT, "synth_mixed"))
    session_id = session_id_for(out_dir.name)

    print(f"\n  Building the weed cut-out bank from {len(WEED_SOURCES)} "
          f"source(s)...")
    bank = load_bank(WEED_SOURCES, SOURCE_FRAMES, rng=rng)
    if not bank:
        raise SystemExit(
            "ERROR: no hand-drawn weed instances found under:\n  " +
            "\n  ".join(str(s) for s in WEED_SOURCES) +
            "\nCompositing from prelabels would make a machine mask with extra "
            "steps.")
    by_class = {}
    for b in bank:
        by_class[b["class_name"]] = by_class.get(b["class_name"], 0) + 1
    print(f"  {len(bank)} cut-out(s): " +
          ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())))
    #: ONE drawer for the whole run. Per-frame drawers would each start a fresh
    #: pool, which promises no repeat within a composite and nothing at all
    #: across them - and across them is where a split can be crossed.
    drawer = CutoutDrawer(bank, rng)

    backgrounds = []
    for root in ONION_BACKGROUNDS:
        p = Path(root)
        if not p.is_dir():
            continue
        sessions = [p] if (p / "rgb").is_dir() else [
            d for d in sorted(p.iterdir()) if d.is_dir()]
        for sess in sessions:
            gt = load_polygons(sess, want_class=CROP_CLASS)
            for stem, insts in gt.items():
                img = _find_image(stem, [sess / "rgb", sess])
                if img is not None:
                    backgrounds.append((img, insts))
    if not backgrounds:
        raise SystemExit(
            "ERROR: no onion frames with annotations under:\n  " +
            "\n  ".join(str(s) for s in ONION_BACKGROUNDS))
    rng.shuffle(backgrounds)
    print(f"  {len(backgrounds)} candidate background(s)")

    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    plan = band_plan(N_IMAGES * int(np.mean(WEEDS_PER_IMAGE)), CONTACT_MIX)
    rng.shuffle(plan)
    items, all_records, rejected = [], [], {}
    seen = used = 0
    pi = 0

    for img_path, onion_insts in backgrounds:
        if len(items) >= N_IMAGES:
            break
        seen += 1
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        onion_masks = [rasterise(p, h, w) for _, p, _ in onion_insts]
        claimed = np.zeros((h, w), bool)
        for m in onion_masks:
            claimed |= m
        ok, n_unclaimed, why = background_ok(pl.veg_of(bgr), claimed)
        if not ok:
            rejected[why] = rejected.get(why, 0) + 1
            continue

        k = rng.randint(*WEEDS_PER_IMAGE)
        want = [plan[(pi + i) % len(plan)] for i in range(k)]
        pi += k
        image, instances, records = compose_one(
            bgr, onion_masks, drawer, want, rng)
        kept_records = [r for r in records if r.get("band")]
        for r in records:
            if not r.get("band"):
                rejected[r.get("reason", "?")] = \
                    rejected.get(r.get("reason", "?"), 0) + 1
        if image is None:
            continue

        used += 1
        name = f"{session_id}_{used:06d}.png"
        cv2.imwrite(str(out_dir / "rgb" / name), image)
        items.append(_item(Path(name).stem, name, h, w, instances))
        #: Tagged with the frame it landed in, not just counted. Without this
        #: the report can say a cut-out was used twice but not WHERE, which is
        #: the half that decides whether the reuse crosses a split.
        for r in kept_records:
            r["item"] = Path(name).stem
        all_records += kept_records

    if not items:
        raise SystemExit(
            "ERROR: every background was rejected and nothing was written.\n"
            "The usual cause is UNCLAIMED_BLOBS_MAX: onion frames whose weeds "
            "were never\nannotated cannot be composite backgrounds, because "
            "the result would teach that\nan unlabelled plant is soil and an "
            "identical pasted one is a target.")

    (out_dir / "annotations" / "default.json").write_text(
        json.dumps(datumaro_doc(items), indent=2), encoding="utf-8")
    bands = {}
    for r in all_records:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    report = format_report(len(items), len(all_records), bands, rejected,
                           seen, used, out_dir, records=all_records,
                           bank_size=len(bank))
    print(report)
    (out_dir / "compose_report.json").write_text(json.dumps({
        "session_id": session_id, "n_images": len(items),
        "n_instances": len(all_records), "bands": bands,
        "rejected": rejected, "seed": SEED,
        "contact_mix": CONTACT_MIX, "contact_bands": CONTACT_BANDS,
        "unclaimed_blobs_max": UNCLAIMED_BLOBS_MAX,
        "weed_sources": [str(s) for s in WEED_SOURCES],
        "onion_backgrounds": [str(s) for s in ONION_BACKGROUNDS],
        "label_provenance": "synthetic",
        "records": all_records,
    }, indent=2), encoding="utf-8")
    print(f"  Add it to mixed.py MIXED_SESSIONS, and set\n"
          f"      SCENE_HINTS[{session_id!r}] = 'mixed'\n"
          f"  Never put it in HOLDOUT_VAL or HOLDOUT_TEST.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
