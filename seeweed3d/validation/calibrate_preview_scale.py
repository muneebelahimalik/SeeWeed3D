#!/usr/bin/env python3
"""
SeeWeed3D - measure the scale an 8-bit depth preview was drawn at
==================================================================
    python -m seeweed3d.validation.calibrate_preview_scale \
        --metric-session E:/dataset/sessions/Visit1_20250228_151838 \
        --preview "E:/.../Session_20250221_130957/Depth_video.avi"

WHY
---
inspect_depth_video can say a file is a recoverable depth preview. It cannot
say what range it was drawn against, and without that the codes are not
distances. `depth_vis_max_mm` was the v1 capture GUI's constant - documented as
3000 in the builds this project has seen - but a session folder with no
session_meta.txt gives no way to confirm it, and every recovered millimetre
inherits whatever is assumed.

When some sessions from the same rig DO have metric depth, the scale stops
being an assumption. The soil surface sits at the mount's working distance in
every frame, so the depth distributions of a metric session and a preview
session describe the same geometry - one in millimetres, one in 0..255 codes.
The ratio between them IS the scale.

THE ASSUMPTION THIS RESTS ON, STATED PLAINLY
--------------------------------------------
That the camera was mounted at the same height, pointing the same way, over
similar ground. Different visits, a re-mounted rig or a different boom setting
break it, and nothing here can detect that - the fit will still return a
number. Prefer sessions recorded close together, and treat a result that
disagrees wildly with 3000 as evidence the assumption failed rather than as a
discovery.

WHY IT FITS AT SEVERAL PERCENTILES INSTEAD OF ONE
-------------------------------------------------
A single-point fit always succeeds: pick any statistic from each side, divide,
and a scale falls out whether or not the relationship is linear. That number
means nothing on its own.

Fitting at several percentiles tests the thing that has to be true for the
recovery to work at all - that code and distance are related by ONE linear
map. If the implied scale is stable across the distribution, the preview really
is a linear ramp and the number is trustworthy. If it drifts, the capture GUI
applied something else (a gamma, a clipped near plane, a per-frame stretch) and
NO single scale will recover the depth, which is a result worth having before
spending a day on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Percentiles the fit is evaluated at. Deliberately excludes the extremes:
#: the near and far tails are where clipping and lossy ringing do most damage,
#: and where a preview's scale is least likely to hold.
FIT_PERCENTILES = (20, 30, 40, 50, 60, 70, 80)

#: Spread of the per-percentile scale estimates, relative to their median,
#: above which the linear assumption is rejected. 10% is loose enough to
#: survive quantisation and scene difference, tight enough to catch a gamma.
LINEARITY_TOL = 0.10


def metric_depth_percentiles(session_dir, pcts=FIT_PERCENTILES, max_frames=30):
    """Percentiles of VALID depth, in millimetres, over a metric session.

    Reads depth/ directly rather than the QC columns in frames_index.csv,
    because those are computed on a downscaled copy and only the median is
    kept - and one statistic is exactly what this must not rely on."""
    d = Path(session_dir) / "depth"
    if not d.is_dir():
        raise SystemExit(
            f"ERROR: {d} not found. --metric-session must be an extracted "
            f"session whose depth_kind is 'metric'; a session with only "
            f"depth_approx/ cannot calibrate anything, since that is the "
            f"output being calibrated.")
    files = sorted(d.glob("*.png"))[:max_frames]
    if not files:
        raise SystemExit(f"ERROR: no depth PNGs in {d}")
    vals = []
    for f in files:
        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            continue
        v = raw[raw > 0]
        if v.size:
            vals.append(np.random.default_rng(0).choice(
                v, size=min(v.size, 200_000), replace=False))
    if not vals:
        raise SystemExit(f"ERROR: no valid depth pixels in {d}")
    allv = np.concatenate(vals).astype(np.float64)
    return {p: float(np.percentile(allv, p)) for p in pcts}, len(files)


def preview_code_percentiles(frames, pcts=FIT_PERCENTILES):
    """Percentiles of the non-zero 0..255 codes across preview frames.

    Zero is excluded because it is the invalid sentinel, not a near
    distance - including it would drag every percentile down and inflate the
    fitted scale."""
    vals = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        v = g[g > 0]
        if v.size:
            vals.append(v)
    if not vals:
        return None
    allv = np.concatenate(vals).astype(np.float64)
    return {p: float(np.percentile(allv, p)) for p in pcts}


def fit_scale(metric_pct, code_pct):
    """Scale implied at each percentile, and whether they agree.

    mm = code / 255 * vis_max  ->  vis_max = mm * 255 / code

    Returns a dict with the per-percentile estimates, the median, the relative
    spread, and whether one linear map is consistent with the data at all."""
    per = {}
    for p, mm in metric_pct.items():
        code = code_pct.get(p)
        if code and code > 0:
            per[p] = mm * 255.0 / code
    if not per:
        return {"linear": False, "reason": "no overlapping percentiles"}
    est = np.array(list(per.values()), float)
    med = float(np.median(est))
    spread = float((est.max() - est.min()) / med) if med else float("inf")
    return {"per_percentile": {p: round(v, 1) for p, v in per.items()},
            "vis_max_mm": round(med, 1),
            "relative_spread": round(spread, 4),
            "linear": bool(spread <= LINEARITY_TOL),
            "reason": ("estimates agree across the distribution"
                       if spread <= LINEARITY_TOL else
                       "estimates drift across the distribution, so code and "
                       "distance are NOT related by one linear map")}


def report(fit, assumed=3000.0):
    """What the fit means for DEPTH_VIS_MAX_MM."""
    out = []
    if not fit.get("linear"):
        out += [
            "REJECTED: " + fit.get("reason", "not linear"),
            "",
            "No single DEPTH_VIS_MAX_MM will recover this depth. The capture "
            "GUI applied something other than a plain linear ramp - a gamma, a "
            "clipped near plane, or a per-frame stretch with no fixed scale at "
            "all. Recovering it would need the capture source, not a constant.",
            "",
            "Do not set RECOVER_8BIT_DEPTH on the strength of this.",
        ]
        return out

    v = fit["vis_max_mm"]
    out += [f"DEPTH_VIS_MAX_MM = {v:.0f}",
            f"  estimates agree to {fit['relative_spread']:.1%} across the "
            f"distribution, so one linear map fits."]
    off = abs(v - assumed) / assumed
    if off <= 0.05:
        out += ["",
                f"That is within {off:.0%} of the documented v1 constant "
                f"({assumed:.0f}), which is the corroboration worth having: "
                f"two independent routes to the same number."]
    elif off <= 0.25:
        out += ["",
                f"The documented v1 constant is {assumed:.0f}, so this differs "
                f"by {off:.0%}. Prefer the MEASURED value - the constant is "
                f"what the GUI shipped with, not necessarily what this build "
                f"used."]
    else:
        out += ["",
                f"WARNING: this is {off:.0%} away from the documented v1 "
                f"constant ({assumed:.0f}). That is a large gap, and the more "
                f"likely explanation is that the rig was NOT mounted the same "
                f"way for the two sessions - which breaks the assumption this "
                f"whole fit rests on. Check the mount before trusting either "
                f"number."]
    out += ["",
            f"Quantisation at this scale: {v / 255.0:.1f} mm per level.",
            "Still a preview, still lossily compressed. Fine for a ground "
            "plane; not for the LEP canopy-height channel."]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric-session", required=True,
                   help="an extracted session whose depth_kind is 'metric'")
    p.add_argument("--preview", required=True, help="the 8-bit Depth_video.*")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--assumed", type=float, default=3000.0)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    a = p.parse_args(argv)

    from validation.inspect_depth_video import probe, read_frames

    mm_pct, n_used = metric_depth_percentiles(a.metric_session)
    info = probe(a.ffprobe, a.preview)
    frames = read_frames(a.ffmpeg, a.preview, a.frames,
                         info["width"], info["height"])
    code_pct = preview_code_percentiles(frames)
    if not code_pct:
        raise SystemExit("ERROR: no non-zero codes in the preview frames")

    print(f"\n  metric reference : {Path(a.metric_session).name} "
          f"({n_used} depth frames)")
    print(f"  preview          : {Path(a.preview).name} "
          f"({len(frames)} frames, {info['codec']}/{info['pix_fmt']})\n")
    print(f"  {'pct':>4}  {'metric mm':>10}  {'code':>6}  {'implied scale':>13}")
    fit = fit_scale(mm_pct, code_pct)
    for pc in sorted(mm_pct):
        s = fit.get("per_percentile", {}).get(pc)
        print(f"  {pc:>4}  {mm_pct[pc]:>10.0f}  {code_pct[pc]:>6.0f}  "
              f"{(f'{s:.0f}' if s else '-'):>13}")
    print()
    for line in report(fit, a.assumed):
        print(f"  {line}" if line else "")
    print()
    return fit


if __name__ == "__main__":
    main()
