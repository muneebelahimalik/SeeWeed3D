#!/usr/bin/env python3
"""
SeeWeed3D - height above the soil, and metric scale, from stereo depth.

WHAT DEPTH IS GOOD AT HERE, AND WHAT IT IS NOT
----------------------------------------------
Stereo depth answers ONE question well on this imagery: *is this raised above
the ground?* A green-tinted pebble sits at 0 mm. A cotyledon sits at 5-20 mm.
No colour index can separate those - that is precisely why the weed prelabeler
ships its recall backstop off and why the mixed one had to fall back on a size
floor. Height separates them independently of substrate colour, which is
something colour can never do for itself.

It is BAD at the question it looks like it should answer: *where exactly does
this leaf end?* Block matching fails on thin, low-texture structures and
fringes at depth discontinuities, so leaf margins come back holed and haloed at
exactly the boundary a mask is deciding.

So everything here is built to be a VETO AND A SCALE REFERENCE, never a
boundary source. Colour keeps the boundary. Depth deletes things lying on the
dirt, separates things at different heights, and converts pixels to
millimetres.

WHY A LOCAL SURFACE AND NOT A PLANE
------------------------------------
A single fitted plane is wrong on a bed. Vidalia onions grow on raised beds
with furrows between them, so a plane through the whole frame reports the
furrow floor as below ground and the bed top as plant - inventing height where
there is none and hiding it where there is.

`soil_surface()` instead estimates the ground LOCALLY, as a high percentile of
depth in a neighbourhood (the camera looks down, so soil is the FARTHEST
surface and plants are nearer). That follows a furrow, a slope and a tilted
camera for free, because it never assumes the ground is flat - only that it is
smooth at the scale of the window.

THE ONE PARAMETER THAT MATTERS: `tile_px`, AND WHY `veg` DECIDES IT
--------------------------------------------------------------------
Two bounds pull in opposite directions. The window must be LARGER than the
biggest plant, or a big plant defines its own local "soil" and measures itself
as flat ground. It must be SMALLER than the terrain undulation you want to
follow, or a furrow is averaged into the bed beside it.

PASSING `veg` DISSOLVES THE FIRST BOUND, which is why it matters far more than
it looks. Measured on a 140 px plant standing 25 mm proud of flat ground:

    tile_px      32     48     96    160
    veg given  25.0   25.0   25.0   25.0     <- exact at every size
    veg absent 12.1   17.9   17.3   25.0     <- needs a window > the plant

With vegetation excluded the plant contributes no samples at all, so the tile
is filled from its neighbours and the window may be as small as you like. Small
is then strictly better, because the second bound is the one that bites.

Measured on 160 px raised beds with 60 mm furrows and a tilted camera, two
plants each 18 mm tall - one on a bed, one in a furrow:

    tile_px          32     48     64     96
    smooth 0   20.0/20.0  20.8/20.3  35.9/24.6  61.0/21.9
    smooth 1   19.8/19.8  38.2/20.2  47.8/15.4  60.5/14.3

A tile that straddles a bed and a furrow takes its high percentile from the
FURROW, so the bed plant reads as bed-height plus plant-height - 61 mm for an
18 mm plant. Smoothing across tiles makes it worse for the same reason. Hence
the defaults: small tiles, no cross-tile smoothing.

THE FAILURE IS NOT SYMMETRIC, WHICH IS WHY THE DEFAULTS ARE CONSERVATIVE. A
window too large inflates plants on the high ground and DEFLATES plants in the
low ground - and a height veto deletes short things, so the plants it would
silently drop are the ones in the furrows.

INVALID DEPTH IS NOT ZERO DEPTH
--------------------------------
The extractor writes 0 as the invalid sentinel in a 16-bit depth PNG. Read
naively that is "0 mm from the camera", which is nearer than everything and
would read as the tallest object in the frame. Every function here takes depth
with NaN for invalid and returns a `valid` mask alongside its answer, so a
caller can never silently consume a number that was never measured.

CONFIDENCE POLARITY IS NOT GUESSED
-----------------------------------
v2 captures write a confidence map, and `session.json` records which direction
means "good" under `confidence_encoding.polarity`. Guessing that wrong does not
degrade gracefully - it keeps EXACTLY the pixels it should have dropped. So an
unknown polarity disables confidence gating rather than assuming one, and says
so.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

#: The extractor's invalid-depth sentinel in a 16-bit PNG.
DEPTH_INVALID_RAW = 0

#: session.json `depth_kind` values that mean real millimetres. Anything else -
#: a normalised 8-bit preview, or nothing at all - must not reach this module.
METRIC_DEPTH_KINDS = ("metric",)

#: Recognised spellings for which direction of the confidence map means "good".
#: ZED's own CONFIDENCE measure is lower-is-better, but the capture script
#: decides what actually lands in the PNG, so the file is believed over folklore.
CONF_HIGHER_IS_BETTER = {"higher_is_better", "higher", "high", "confidence"}
CONF_LOWER_IS_BETTER = {"lower_is_better", "lower", "low", "uncertainty",
                        "zed_confidence"}


def load_depth_mm(path):
    """A 16-bit depth PNG as float millimetres with NaN for invalid, or None.

    Refuses anything that is not 16-bit rather than scaling an 8-bit preview
    into plausible-looking millimetres - the same fiction REQUIRE_16BIT_DEPTH
    exists to prevent at extraction time."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        return None
    d = raw.astype(np.float32)
    d[raw == DEPTH_INVALID_RAW] = np.nan
    return d


def session_depth_kind(session_dir):
    """`depth_kind` from a session's meta/session.json, or 'unknown'."""
    try:
        doc = json.loads((Path(session_dir) / "meta" / "session.json")
                         .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    return str(doc.get("depth_kind") or "unknown") if isinstance(doc, dict) \
        else "unknown"


def has_metric_depth(session_dir):
    """Is it safe to use this session's depth as millimetres?"""
    return session_depth_kind(session_dir) in METRIC_DEPTH_KINDS


def confidence_polarity(session_dir):
    """Which direction of the confidence map means 'good', or None.

    None means UNKNOWN, and a caller must then not gate at all. Gating the
    wrong way round is worse than not gating: it keeps precisely the pixels it
    was meant to drop, and the result looks like a cleaner depth map."""
    try:
        doc = json.loads((Path(session_dir) / "meta" / "session.json")
                         .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    enc = (doc or {}).get("confidence_encoding") or {}
    pol = str(enc.get("polarity") or "").strip().lower()
    if pol in CONF_HIGHER_IS_BETTER:
        return "higher_is_better"
    if pol in CONF_LOWER_IS_BETTER:
        return "lower_is_better"
    return None


def confidence_mask(conf, polarity, threshold):
    """Pixels whose depth is trustworthy, or None when it cannot be decided.

    `threshold` is expressed as a fraction of the map's full 0-255 range in the
    GOOD direction, so one number means the same thing under either polarity."""
    if conf is None or polarity is None:
        return None
    c = conf.astype(np.float32)
    t = float(np.clip(threshold, 0.0, 1.0)) * 255.0
    return c >= t if polarity == "higher_is_better" else c <= (255.0 - t)


def calibration(session_dir):
    """(fx, fy) in pixels from a session's calibration.json, or (None, None).

    The RECTIFIED left intrinsics, because those are what apply to the images
    the extractor actually wrote."""
    try:
        doc = json.loads((Path(session_dir) / "meta" / "calibration.json")
                         .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    left = (doc or {}).get("left") or {}
    fx, fy = left.get("fx"), left.get("fy")
    if not fx or not fy:
        return None, None
    return float(fx), float(fy)


def _fill_nan_grid(grid):
    """Nearest-valid fill of a small grid, so tiles with no soil samples still
    get a value from their neighbours rather than poisoning the upsample."""
    bad = ~np.isfinite(grid)
    if not bad.any():
        return grid
    if bad.all():
        return grid
    # Distance transform gives, for every invalid cell, the index of the
    # nearest valid one - an exact nearest-neighbour fill in one pass.
    _, idx = cv2.distanceTransformWithLabels(
        bad.astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    flat = grid.ravel()
    valid_flat = np.flatnonzero(np.isfinite(grid).ravel())
    if not len(valid_flat):
        return grid
    # DIST_LABEL_PIXEL labels are 1-based indices into the zero (valid) pixels
    # in raster order.
    lut = np.concatenate([[valid_flat[0]], valid_flat])
    out = flat[lut[np.clip(idx.ravel(), 0, len(lut) - 1)]].reshape(grid.shape)
    return out.astype(np.float32)


def soil_surface(depth_mm, veg=None, valid=None, tile_px=32, percentile=80.0,
                 min_samples_frac=0.10, smooth_tiles=0):
    """Local soil depth in millimetres, same shape as `depth_mm`.

    The camera looks down, so SOIL IS THE FARTHEST SURFACE: a high percentile
    of depth in a neighbourhood is the ground, and anything nearer than it is
    standing on it. A percentile rather than the maximum because the maximum is
    whatever noise reached furthest.

    veg   - known vegetation, EXCLUDED from the samples. Without it a tile
            covered by one big plant would measure the plant as its own ground.
    valid - extra per-pixel gate, normally the confidence mask.

    Returns NaN nowhere unless the whole frame is unusable; tiles with too few
    soil samples are filled from their nearest usable neighbour, because a hole
    in the surface becomes a hole in every height derived from it."""
    d = np.asarray(depth_mm, np.float32)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= valid
    if veg is not None:
        ok &= ~veg
    if not ok.any():
        return np.full(d.shape, np.nan, np.float32)

    h, w = d.shape
    t = max(8, int(tile_px))
    gh, gw = max(1, -(-h // t)), max(1, -(-w // t))
    grid = np.full((gh, gw), np.nan, np.float32)
    need = max(4, int(min_samples_frac * t * t))

    for gy in range(gh):
        y0, y1 = gy * t, min(h, (gy + 1) * t)
        for gx in range(gw):
            x0, x1 = gx * t, min(w, (gx + 1) * t)
            m = ok[y0:y1, x0:x1]
            if int(m.sum()) < need:
                continue
            grid[gy, gx] = np.percentile(d[y0:y1, x0:x1][m], percentile)

    grid = _fill_nan_grid(grid)
    if smooth_tiles and gh > 2 and gw > 2:
        k = 2 * int(smooth_tiles) + 1
        grid = cv2.blur(grid, (k, k))
    # Bilinear so the surface is continuous rather than blocky - a tile-shaped
    # step in the ground estimate becomes a tile-shaped ridge in the height.
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)


def height_map(depth_mm, veg=None, valid=None, **kw):
    """(height_mm, measured) - millimetres above the local soil surface.

    Positive is UP. Nearer the camera means a smaller depth, so height is
    soil-minus-depth. `measured` is True only where the height rests on a real
    depth reading; everywhere else the height is 0 and must not be read as
    'lying on the ground'. Those two are different claims and conflating them
    is how a veto deletes a plant it never measured."""
    d = np.asarray(depth_mm, np.float32)
    measured = np.isfinite(d) & (d > 0)
    if valid is not None:
        measured &= valid
    soil = soil_surface(d, veg=veg, valid=valid, **kw)
    height = np.where(measured & np.isfinite(soil), soil - d, 0.0)
    return height.astype(np.float32), measured


def surface_report(depth_mm, veg=None, valid=None, **kw):
    """What the ground looks like, so the caller can judge whether height means
    anything here before trusting a veto built on it.

    `relief_mm` is the spread of the estimated soil surface across the frame.
    Read it against the height threshold you intend to use: relief of 60 mm
    under a 10 mm threshold is a bed-and-furrow field where the surface
    estimate is doing real work, and if the window is too coarse for it the
    error lands squarely on top of the thing being measured.

    `measured_frac` is how much of the frame carries usable depth at all. Low
    values are not a failure - stereo drops out on texture-poor soil and at
    every depth discontinuity - but they bound how much a height veto can
    decide, and a veto that abstains on most instances should be known to be
    abstaining rather than assumed to be working."""
    d = np.asarray(depth_mm, np.float32)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= valid
    soil = soil_surface(d, veg=veg, valid=valid, **kw)
    s = soil[np.isfinite(soil)]
    return {
        "measured_frac": round(float(ok.mean()), 4),
        "relief_mm": round(float(np.percentile(s, 95) - np.percentile(s, 5)), 1)
        if s.size else 0.0,
        "median_depth_mm": round(float(np.median(d[ok])), 1) if ok.any() else 0.0,
    }


def mm_per_px(depth_mm, fx, fy):
    """Millimetres spanned by one pixel, per pixel, from the pinhole model.

    At depth Z a pixel subtends Z/fx horizontally and Z/fy vertically. This is
    what turns a px^2 area threshold into a mm^2 one and stops every size floor
    in the pipeline depending on how high the boom happens to be."""
    d = np.asarray(depth_mm, np.float32)
    with np.errstate(invalid="ignore"):
        sx = np.where(np.isfinite(d) & (d > 0), d / float(fx), np.nan)
        sy = np.where(np.isfinite(d) & (d > 0), d / float(fy), np.nan)
    return sx.astype(np.float32), sy.astype(np.float32)


def area_mm2(mask, depth_mm, fx, fy):
    """Physical area of a mask in mm^2, or None when it cannot be measured.

    Uses the MEDIAN pixel footprint over the mask rather than summing per
    pixel: stereo noise on a few pixels would otherwise swing the total, and a
    plant is close to planar at this scale so one footprint is a fair
    description of all of it."""
    m = np.asarray(mask, bool)
    if not m.any() or fx is None or fy is None:
        return None
    sx, sy = mm_per_px(depth_mm, fx, fy)
    a = (sx * sy)[m]
    a = a[np.isfinite(a)]
    if a.size < max(4, int(0.05 * m.sum())):
        return None                  # too little valid depth to claim a size
    return float(np.median(a) * int(m.sum()))


def instance_height(mask, height_mm, measured, min_measured_frac=0.20,
                    percentile=75.0):
    """(height_mm, fraction_measured) for one instance, or (None, frac).

    A percentile rather than the mean, because a mask's edge pixels straddle
    the depth discontinuity at the plant's own boundary and read low. The
    number wanted is 'how tall is this thing', which is a property of its body.

    None when too little of the instance carries real depth. That abstention is
    the whole safety property: an instance whose height is unknown must be
    KEPT, not deleted, and a caller cannot make that distinction from a number
    alone."""
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if not n:
        return None, 0.0
    got = m & measured
    frac = float(got.sum()) / n
    if frac < min_measured_frac:
        return None, frac
    return float(np.percentile(height_mm[got], percentile)), frac


def height_veto(instances, veg, depth_mm, cfg, conf=None, polarity=None,
                fx=None, fy=None):
    """Drop instances that are lying on the ground, and say what was decided.

    THE ASYMMETRY THIS IS BUILT AROUND. Stereo drops out on thin, low-texture
    tissue and fringes at every depth discontinuity - which is to say, on
    exactly the small plants a height gate would otherwise delete. So an
    instance whose height cannot be measured is KEPT. The veto only ever fires
    on positive evidence that something is flat, never on absence of evidence
    that it is not.

    Returns (kept, qa). Every instance gains `height_mm` (None when
    unmeasured), `height_measured_frac` and, where calibration allows it,
    `area_mm2` - so the numbers behind a decision travel with the instance
    into instances.csv rather than only into this function."""
    valid = confidence_mask(conf, polarity, cfg["DEPTH_MIN_CONFIDENCE"])
    height, measured = height_map(
        depth_mm, veg=veg, valid=valid,
        tile_px=cfg["GROUND_TILE_PX"], percentile=cfg["GROUND_PERCENTILE"],
        smooth_tiles=0)
    surface = surface_report(depth_mm, veg=veg, valid=valid,
                                tile_px=cfg["GROUND_TILE_PX"],
                                percentile=cfg["GROUND_PERCENTILE"],
                                smooth_tiles=0)

    floor_mm2 = cfg.get("MIN_INSTANCE_AREA_MM2")
    kept, dropped_flat, dropped_small, abstained = [], 0, 0, 0
    for inst in instances:
        h, frac = instance_height(
            inst["mask"], height, measured,
            min_measured_frac=cfg["HEIGHT_MIN_MEASURED_FRAC"],
            percentile=cfg["HEIGHT_PERCENTILE"])
        inst["height_mm"] = None if h is None else round(h, 1)
        inst["height_measured_frac"] = round(frac, 3)
        a2 = area_mm2(inst["mask"], depth_mm, fx, fy)
        inst["area_mm2"] = None if a2 is None else round(a2, 1)

        if h is None:
            abstained += 1
            kept.append(inst)
            continue
        if h < cfg["HEIGHT_MIN_MM"]:
            dropped_flat += 1
            continue
        # The metric floor REPLACES the pixel one where it can be computed, so
        # the threshold stops depending on mount height. Where it cannot be
        # computed the pixel floor has already been applied upstream.
        if floor_mm2 and a2 is not None and a2 < float(floor_mm2):
            dropped_small += 1
            continue
        kept.append(inst)

    return kept, {
        "height_dropped_flat": dropped_flat,
        "height_dropped_small": dropped_small,
        "height_abstained": abstained,
        "ground_relief_mm": surface["relief_mm"],
        "depth_measured_frac": surface["measured_frac"],
        "median_depth_mm": surface["median_depth_mm"],
    }


def mask_height_filter(mask, veg, depth_mm, cfg, conf=None, polarity=None,
                       fx=None, fy=None):
    """Height veto for a caller that holds ONE merged mask, not instances.

    The onion and weed prelabelers export a single boolean mask per frame
    rather than a list, so the veto is applied per connected component and the
    survivors are unioned back. Same evidence, same abstention rule, same
    thresholds as height_veto() - a component whose height cannot be measured
    is KEPT, because stereo drops out on exactly the thin tissue a height gate
    would otherwise delete.

    Returns (mask, qa)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m, {}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    instances = [{"mask": labels == i} for i in range(1, n)]
    kept, qa = height_veto(instances, veg, depth_mm, cfg, conf=conf,
                           polarity=polarity, fx=fx, fy=fy)
    out = np.zeros(m.shape, bool)
    for inst in kept:
        out |= inst["mask"]
    return out, qa


def session_depth_setup(sid, session_dir, cfg, printer=print):
    """Decide ONCE per session whether depth may be used, and say so.

    Returns (use_depth, fx, fy, polarity). Deciding it here rather than per
    frame means a session that cannot support the veto says so before the GPU
    starts, instead of silently skipping it eight hundred times.

    USE_DEPTH_HEIGHT True raises rather than falling back, so a run you believe
    is depth-gated cannot quietly not be."""
    want = cfg.get("USE_DEPTH_HEIGHT", "auto")
    kind = session_depth_kind(session_dir)
    use = bool(want) and kind in METRIC_DEPTH_KINDS
    if want is True and not use:
        raise SystemExit(
            f"ERROR: [{sid}] USE_DEPTH_HEIGHT is True but this session's "
            f"depth_kind is {kind!r}, not 'metric'. Only v2 (MKV/FFV1) captures "
            f"carry real millimetres; a v1 preview was normalised per frame at "
            f"capture and cannot be used as height. Set it to \"auto\" to skip "
            f"depth on sessions that lack it.")
    if not use:
        if want:
            printer(f"  [{sid}] depth: {kind} - height veto off, colour and "
                    f"pixel-size gates only")
        return False, None, None, None

    fx, fy = calibration(session_dir)
    polarity = confidence_polarity(session_dir)
    printer(f"  [{sid}] depth: metric | height veto on "
            f"(>= {cfg['HEIGHT_MIN_MM']:.0f} mm above local soil)")
    if fx is None:
        printer(f"  [!] no calibration.json - a mm^2 floor cannot be applied, "
                f"falling back to the pixel floor.")
    if polarity is None and (Path(session_dir) / "conf").is_dir():
        printer(f"  [!] a confidence map exists but session.json does not "
                f"record its polarity, so it is NOT used. Gating the wrong way "
                f"round keeps exactly the pixels it should drop.")
    return True, fx, fy, polarity


def load_frame_depth(session_dir, filename, use_depth, polarity):
    """(depth_mm, conf) for one frame, or (None, None). Never raises."""
    if not use_depth:
        return None, None
    session_dir = Path(session_dir)
    dpath = session_dir / "depth" / filename
    depth = load_depth_mm(dpath) if dpath.exists() else None
    conf = None
    cpath = session_dir / "conf" / filename
    if depth is not None and polarity and cpath.exists():
        conf = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)
    return depth, conf
