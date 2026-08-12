#!/usr/bin/env python3
"""
SeeWeed3D - what is actually inside a depth video?
==================================================
    python -m seeweed3d.validation.inspect_depth_video \
        --video "E:/.../Session_20250221_130957/Depth_video.avi"

WHY THIS EXISTS
---------------
extract_sessions.py refuses to decode a depth stream that is not 16-bit,
because ffmpeg will happily produce gray16le from an 8-bit source by scaling
up values it invented, and the resulting PNGs are indistinguishable from real
millimetres. That refusal is correct and it is also the end of the
conversation - it says the file is not metric depth, not what it IS.

The difference matters, because "8-bit" covers two very different cases:

  A DEPTH VISUALISATION. The v1 capture app previewed depth scaled to
    `depth_vis_max_mm` (3000) and could write that preview to disk. The
    geometry is still in there, quantised to 256 levels and then lossily
    compressed - recoverable, approximately, IF the scale is known.

  SOMETHING ELSE ENTIRELY. A colour image, a confidence map, a normalised
    per-frame stretch with no fixed scale. Not recoverable at all.

Telling them apart is cheap and the test is decisive, so it should not be a
guess. This reports what is there and what it would be worth; it writes
nothing and changes nothing.

WHAT THE ANSWER BUYS YOU, HONESTLY
-----------------------------------
Even in the best case - a grayscale ramp with a known maximum - the recovered
depth carries `max_mm/255` of quantisation (11.8 mm at 3000) plus DCT ringing
concentrated at exactly the depth discontinuities that matter. That is
plausibly fine for a ground plane and useless for the LEP's canopy-height
channel, where the whole signal is a few centimetres of relief on one plant.

So a positive result here is not "the depth is fine". It is "there is a coarse
range image, and here is what it would cost to use it".
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: OpenCV colormaps a depth preview plausibly uses. JET is the classic ZED
#: preview; TURBO and VIRIDIS are the modern replacements; BONE and OCEAN turn
#: up in hand-rolled viewers.
CANDIDATE_COLORMAPS = ("JET", "TURBO", "VIRIDIS", "MAGMA", "INFERNO",
                       "PLASMA", "HOT", "BONE", "OCEAN", "RAINBOW")

#: Chroma deviation from neutral (128) below which a frame is grayscale. JPEG
#: and MPEG-4 chroma noise on a true gray image stays within a couple of
#: levels; any real colormap swings tens of levels.
GRAY_CHROMA_TOL = 6.0

#: Median per-channel distance to the best-matching colormap entry, below which
#: the match is believable. A wrong colormap on a real image lands far above.
COLORMAP_RESIDUAL_TOL = 12.0


def sixteen_bit(pix_fmt):
    return any(h in (pix_fmt or "").lower()
               for h in ("16le", "16be", "p010", "p016", "10le", "12le"))


def chroma_spread(bgr):
    """How far the colour channels swing from neutral gray, in Cr/Cb levels.

    A grayscale image stored in a colour pixel format has Cr = Cb = 128
    everywhere; lossy compression jitters that by a level or two. A colormap
    swings it by tens. This is the whole grayscale-vs-colour test and it does
    not care which colormap, so it works before anything is matched."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1].astype(np.float32), ycc[:, :, 2].astype(np.float32)
    return float(np.percentile(np.maximum(np.abs(cr - 128), np.abs(cb - 128)), 99))


def colormap_lut(name):
    """(256, 3) BGR lookup table for an OpenCV colormap, index 0..255."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    return cv2.applyColorMap(ramp, getattr(cv2, f"COLORMAP_{name}")).reshape(256, 3)


def match_colormap(bgr, names=CANDIDATE_COLORMAPS, sample=160):
    """Best-matching colormap and how well it fits.

    Returns (name, median_residual, index_image_uint8). The index image is the
    recovered 0..255 depth code - the thing a scale would turn into
    millimetres - so a caller that trusts the match can use it directly.

    Matching is nearest-neighbour in BGR against the colormap's own 256
    entries, on a downsampled copy for speed. The residual is what makes the
    answer trustworthy: a colormap that is not the right one still produces a
    nearest entry for every pixel, and only the distance tells you so.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, sample / float(max(1, w)))
    small = (cv2.resize(bgr, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr)
    px = small.reshape(-1, 3).astype(np.int16)
    best = (None, float("inf"), None)
    for name in names:
        lut = colormap_lut(name).astype(np.int16)
        d = np.abs(px[:, None, :] - lut[None, :, :]).sum(axis=2)
        idx = d.argmin(axis=1)
        resid = float(np.median(d[np.arange(len(idx)), idx]) / 3.0)
        if resid < best[1]:
            best = (name, resid, idx)
    name, resid, idx = best
    if idx is None:
        return None, float("inf"), None
    # Re-run at full resolution only for the winner, so the returned index
    # image lines up with the original frame.
    lut = colormap_lut(name).astype(np.int16)
    full = bgr.reshape(-1, 3).astype(np.int16)
    d = np.abs(full[:, None, :] - lut[None, :, :]).sum(axis=2)
    return name, resid, d.argmin(axis=1).astype(np.uint8).reshape(h, w)


#: How near the top of the range every frame's maximum must sit. NOT 255,
#: because lossy encoding shaves the peak - a faithful replay of the capture
#: code through mpeg4 comes back at 248-250. But close to it, because that is
#: the whole signal: normalisation FORCES the peak to the top of the range
#: every frame, while a fixed scale only reaches it when the scene happens to
#: run past the cut.
NORMALISED_MAX = 245

#: How much the per-frame maxima may differ from each other. Under per-frame
#: normalisation they are all the same value by construction; under a fixed
#: scale they track whatever is actually farthest in each frame.
NORMALISED_MAX_RANGE = 12

#: Fraction of pixels sitting exactly AT the frame's peak, below which the top
#: was reached by outliers rather than by a saturated plateau. A clipped fixed
#: scale pins everything beyond the cut to one value, which is a large uniform
#: region; a normalised frame's peak is the single farthest pixel and the
#: values below it decay smoothly away from it.
NORMALISED_SAT_FRAC = 0.02


def per_frame_normalisation(codes):
    """Was each frame scaled by its OWN maximum?

    This is the one failure a scale cannot fix, so it is worth detecting
    directly rather than inferring it from a calibration that drifts.

    The v1 capture GUI wrote depth as

        depth_norm = depth / depth.max() * 255

    with `depth.max()` recomputed every frame. The scale is therefore a
    property of each individual frame - one distant outlier pixel sets it for
    all the others - and NO constant relates code to millimetres across a
    session. `depth_vis_max_mm` never entered into it.

    Three things have to hold together, because no one of them is decisive:

      every frame reaches the top of the range     - a fixed scale only does
                                                     this when it clips
      the maxima agree with each other             - under a fixed scale they
                                                     track the farthest thing
                                                     actually in each frame
      the peak is an outlier, not a plateau        - a fixed scale that DOES
                                                     clip pins everything
                                                     beyond the cut to one
                                                     value, a large uniform
                                                     region

    Takes CODES, not pixels: for a colourised preview the code is the colormap
    index, and testing the painted luma would test the colormap's brightness
    curve instead.

    Needs several frames - one frame reaching its own maximum says nothing.

    Not infallible. A fixed-scale preview whose scene runs past the cut in a
    small patch every single frame looks the same from the outside. The
    definitive check is the capture code that wrote the file; this is the best
    available answer when that code is not to hand, and it errs toward
    refusing, because recovering garbage depth costs far more than declining
    to recover a file that turns out to have been recoverable.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
             for f in codes]
    if len(grays) < 3:
        return None
    maxima = [int(g.max()) for g in grays]
    top = max(maxima)
    # Measured AT each frame's own peak rather than at 255, so lossy ringing
    # that shaves a few levels off the top does not read as "never saturates".
    sat = [float((g >= m).mean()) for g, m in zip(grays, maxima)]
    return {"frame_maxima": maxima,
            "max_saturated_frac": round(max(sat), 6),
            "normalised": bool(min(maxima) >= NORMALISED_MAX
                               and top - min(maxima) <= NORMALISED_MAX_RANGE
                               and max(sat) < NORMALISED_SAT_FRAC)}


def luma_profile(bgr):
    """What the intensity channel looks like: how much of it is zero, how much
    of the 0..255 range is used, and whether it sits in TV range.

    A depth visualisation has a large exactly-zero population (the invalid
    sentinel) and uses most of the range. Video encoded in limited range never
    goes below 16 or above 235, which silently costs ~14% of the scale and has
    to be undone before any millimetre conversion."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = int(g.min()), int(g.max())
    return {"zero_frac": round(float((g == 0).mean()), 4),
            "min": lo, "max": hi,
            "used_frac": round((hi - lo) / 255.0, 3),
            "looks_tv_range": bool(lo >= 15 and hi <= 236),
            "unique_levels": int(len(np.unique(g)))}


def probe(ffprobe, path):
    import json as _json
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = _json.loads(out)
    s = (d.get("streams") or [{}])[0]
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"),
            "width": s.get("width"), "height": s.get("height"),
            "nb_frames": s.get("nb_frames")}


def read_frames(ffmpeg, path, n, w, h):
    """First n frames as BGR. -vsync 0 keeps one output per coded frame."""
    bpf = w * h * 3
    p = subprocess.Popen(
        [ffmpeg, "-loglevel", "error", "-i", str(path), "-vframes", str(n),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = []
    try:
        while len(frames) < n:
            buf = p.stdout.read(bpf)
            if len(buf) < bpf:
                break
            frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    finally:
        p.stdout.close()
        p.wait(timeout=10)
    return frames


def classify(info, frames):
    """Verdict for one depth video. Pure - no decoding, no I/O."""
    if sixteen_bit(info.get("pix_fmt")):
        return {"verdict": "true_16bit",
                "detail": f"{info['codec']}/{info['pix_fmt']} carries 16 bits "
                          f"per sample - this is metric depth",
                "recoverable": True, "exact": True}
    if not frames:
        return {"verdict": "undecodable",
                "detail": "no frames could be decoded",
                "recoverable": False, "exact": False}

    chroma = max(chroma_spread(f) for f in frames)
    luma = luma_profile(frames[0])

    # Recover the CODE first - the 0..255 depth index - because every later
    # question is about the codes, not about the pixels they were painted as.
    # For a grayscale preview the code is the luma; for a colourised one it is
    # the colormap index, and testing the luma instead would be testing the
    # colormap's own brightness curve.
    if chroma <= GRAY_CHROMA_TOL:
        kind, colormap, resid = "grayscale_8bit", None, None
        codes = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    else:
        colormap, resid, _ = match_colormap(frames[0])
        if resid > COLORMAP_RESIDUAL_TOL:
            return {"verdict": "unknown_8bit", "closest_colormap": colormap,
                    "residual": round(resid, 1),
                    "chroma_spread": round(chroma, 1), "luma": luma,
                    "detail": "8-bit COLOUR matching no known depth colormap. "
                              "This is not a recoverable range image",
                    "recoverable": False, "exact": False}
        kind = "colormapped_8bit"
        codes = [match_colormap(f, names=(colormap,))[2] for f in frames]

    # Per-frame normalisation outranks everything above: a preview scaled by
    # its own maximum is unrecoverable whether it was written gray or
    # colourised, because there is no constant to find in either case.
    norm = per_frame_normalisation(codes)
    if norm and norm["normalised"]:
        return {"verdict": "per_frame_normalised_8bit",
                "written_as": kind, "colormap": colormap,
                "frame_maxima": norm["frame_maxima"],
                "max_saturated_frac": norm["max_saturated_frac"],
                "chroma_spread": round(chroma, 1), "luma": luma,
                "detail": "8-bit, and each frame was scaled by its OWN maximum "
                          "(depth / depth.max() * 255). The scale changes every "
                          "frame, so no constant relates code to millimetres",
                "recoverable": False, "exact": False}

    if kind == "grayscale_8bit":
        return {"verdict": "grayscale_8bit", "chroma_spread": round(chroma, 1),
                "luma": luma,
                "detail": "8-bit GRAYSCALE - a depth preview, not metric depth. "
                          "The geometry is present at 1/256 of whatever range "
                          "it was scaled to, then lossily compressed",
                "recoverable": True, "exact": False}
    return {"verdict": "colormapped_8bit", "colormap": colormap,
            "residual": round(resid, 1), "chroma_spread": round(chroma, 1),
            "luma": luma,
            "detail": f"8-bit COLOURISED with the {colormap} colormap - a "
                      f"preview. The index is invertible, so the same "
                      f"1/256 geometry is present",
            "recoverable": True, "exact": False}


def advise(v):
    """What the verdict means for this project, including what it costs."""
    k = v["verdict"]
    if k == "true_16bit":
        return ["Extract normally. Nothing to do."]
    if k == "undecodable":
        return ["The file cannot be read. Check it is not truncated."]
    if k == "per_frame_normalised_8bit":
        return [
            "Nothing to recover, and no calibration will change that.",
            "",
            "Each frame was divided by its own maximum before being written, "
            "so the scale is a property of the individual frame - one distant "
            "outlier pixel sets it for every other pixel in that frame. Two "
            "frames of identical geometry get different codes whenever their "
            "farthest point differs. There is no constant to find.",
            "",
            "Treat these sessions as RGB-only. They remain fully usable for "
            "segmentation; only 3D and LEP need real depth.",
            "",
            "For future captures the fix is one line at the capture end: write "
            "the raw millimetres as 16-bit rather than a normalised 8-bit "
            "preview. See capture/zed_capture.py, which records depth "
            "losslessly alongside the SVO archive.",
        ]
    if k == "unknown_8bit":
        return ["Nothing to recover. Treat these sessions as RGB-only.",
                "Depth for this visit would have to come from the original "
                "capture output (SVO or the raw stream), if it still exists."]

    out = [
        "There IS geometry in this file, but it is a PREVIEW, not measurement.",
        "",
        "To turn it into millimetres you need the scale it was drawn at - the "
        "v1 capture app's `depth_vis_max_mm`, 3000 in the builds this project "
        "has seen. It is NOT in these session folders (no session_meta.txt), "
        "so it has to come from the capture code or be calibrated against a "
        "session that has real depth.",
        "",
        f"Cost at 3000 mm: {3000 / 255:.1f} mm per level of quantisation, "
        f"before lossy compression. DCT ringing concentrates at depth "
        f"discontinuities - exactly the plant edges that matter.",
        "",
        "Worth it for: ground plane, camera height, coarse row geometry.",
        "Not worth it for: the LEP canopy-height channel, where the entire "
        "signal is a few centimetres of relief on one plant.",
    ]
    luma = v.get("luma", {})
    if luma.get("looks_tv_range"):
        out += ["",
                "NOTE: the luma sits inside 16-235, so this was encoded in TV "
                "range. Undo that before any conversion or every distance is "
                "off by ~14%."]
    if luma.get("zero_frac", 0) < 0.001:
        out += ["",
                "NOTE: almost no exactly-zero pixels, so the invalid sentinel "
                "did not survive the encode. Invalid regions are now "
                "indistinguishable from near distances."]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="a Depth_video.* file")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--save-preview", default=None,
                   help="write the first decoded frame here, to look at")
    a = p.parse_args(argv)

    path = Path(a.video)
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found")

    info = probe(a.ffprobe, path)
    print(f"\n{path.name}")
    print(f"  {info['codec']}/{info['pix_fmt']}  "
          f"{info['width']}x{info['height']}  frames={info['nb_frames']}")

    frames = ([] if sixteen_bit(info["pix_fmt"])
              else read_frames(a.ffmpeg, path, a.frames,
                               info["width"], info["height"]))
    v = classify(info, frames)

    print(f"\n  VERDICT: {v['verdict']}")
    print(f"  {v['detail']}.")
    for key in ("colormap", "closest_colormap", "residual", "chroma_spread",
                "frame_maxima", "max_saturated_frac"):
        if key in v:
            print(f"    {key}: {v[key]}")
    if "luma" in v:
        L = v["luma"]
        print(f"    luma: range {L['min']}..{L['max']} "
              f"({L['unique_levels']} distinct levels), "
              f"{L['zero_frac']:.1%} exactly zero"
              + (", TV range" if L["looks_tv_range"] else ""))
    print()
    for line in advise(v):
        print(f"  {line}" if line else "")
    print()

    if a.save_preview and frames:
        cv2.imwrite(a.save_preview, frames[0])
        print(f"  first frame -> {a.save_preview}\n")
    return v


if __name__ == "__main__":
    main()
