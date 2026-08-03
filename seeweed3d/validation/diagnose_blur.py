#!/usr/bin/env python3
"""
SeeWeed3D - is the blur MOTION or OPTICS?

WHY THIS IS ANSWERABLE RATHER THAN A GUESS
------------------------------------------
The extraction path is lossless end to end: FFV1 MKV in, full-resolution PNG
out, no resize and no lossy re-encode of the saved frames. (extract_sessions.py
downscales only to compute QC metrics, never the stored image.) So extraction
cannot introduce blur, and the cause is at capture time. This tool decides
which capture cause it is, from the frames themselves.

THE DECISIVE TEST: DIRECTION
----------------------------
Motion blur is ANISOTROPIC. Smearing the image along a direction destroys
detail along that axis while leaving detail perpendicular to it intact, so
gradient energy has a clear minimum along the direction of travel.

Defocus and a dirty/wet lens are ISOTROPIC. They blur equally in every
direction, so gradient energy is flat.

That alone separates the two families. The tool then goes further: it measures
the camera's actual direction of travel by phase correlation between
consecutive frames, and checks whether the blur axis AGREES with it. Blur that
lines up with the measured direction of travel, frame after frame, is motion
blur - there is no other mechanism that would produce that correlation.

Three independent lines of evidence are reported:

  1. ANISOTROPY      - is the blur directional at all?
  2. AXIS AGREEMENT  - does the blur axis match the measured travel direction?
  3. SPEED COUPLING  - does sharpness fall as inter-frame displacement rises?

Plus, when the v2 capture log recorded them, sharpness against EXPOSURE and
GAIN, which is what turns a diagnosis into a fix.

    python seeweed3d/validation/diagnose_blur.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.progress import Progress  # noqa: E402

# #############################################################################
# ##  DATASET_ROOT - the OUTPUT_ROOT you gave extract_sessions.py            ##
# #############################################################################

DATASET_ROOT = r"E:\Dataset_Vidalia"

CONFIG = {
    "DATASET_ROOT": DATASET_ROOT,
    "ONLY_SESSIONS": [],          # empty = every session
    "MAX_FRAMES": 200,            # per session; enough for a stable verdict
    "WORK_WIDTH": 640,            # analysis resolution; blur survives downscaling
    "N_ANGLES": 18,               # directional resolution (10 degree steps)

    # Anisotropy above this means the blur is clearly directional. Below the
    # low bound it is clearly isotropic. Between them the evidence is weak and
    # the tool says so rather than picking a side.
    "ANISO_STRONG": 0.30,
    "ANISO_WEAK": 0.12,

    # Blur axis within this many degrees of the measured travel direction
    # counts as agreement.
    "AXIS_TOLERANCE_DEG": 25.0,

    # A frame is called blurry if its sharpness is below this fraction of the
    # session median. Relative, because Laplacian variance is scene dependent.
    "BLUR_REL_THRESHOLD": 0.55,
}


# --------------------------------------------------------------------------- #
# Per-frame measurements
# --------------------------------------------------------------------------- #
def _gray(path, work_width):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if work_width and w > work_width:
        img = cv2.resize(img, (work_width, max(1, round(h * work_width / w))),
                         interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def sharpness(gray):
    """Laplacian variance - the same measure extract_sessions.py records, so
    numbers here are comparable with pool.csv.

    CV_32F rather than CV_64F because OpenCV 5 has no float32 -> float64 filter
    path; the variance is identical to well past the precision that matters."""
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def directional_energy(gray, n_angles=18):
    """Gradient energy as a function of direction.

    For a direction theta, the directional derivative is
        g_theta = gx*cos(theta) + gy*sin(theta)
    and its mean square is the detail present ALONG that direction. Motion blur
    along an axis destroys detail along it, so this curve dips at the direction
    of travel. Computed from the gradient covariance (the structure tensor's
    global average), which gives every angle in closed form from three sums
    rather than one Sobel pass per angle."""
    g32 = gray.astype(np.float32)
    gx = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    th = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    e = (jxx * np.cos(th) ** 2 + 2.0 * jxy * np.cos(th) * np.sin(th)
         + jyy * np.sin(th) ** 2)
    return th, e, (jxx, jyy, jxy)


def blur_axis(gray, n_angles=18):
    """(anisotropy, blur_axis_deg, sharpness).

    anisotropy   = (max - min) / (max + min) of the directional energy.
                   0 = perfectly isotropic (defocus), high = directional.
    blur_axis_deg = the direction of LEAST detail, i.e. the smear axis,
                   in degrees measured from +x, in [0, 180)."""
    th, e, _ = directional_energy(gray, n_angles)
    lo, hi = float(e.min()), float(e.max())
    aniso = (hi - lo) / (hi + lo) if (hi + lo) > 1e-12 else 0.0
    axis = float(np.degrees(th[int(np.argmin(e))]) % 180.0)
    return aniso, axis, sharpness(gray)


def frame_shift(prev_gray, cur_gray):
    """(dx, dy) between consecutive frames by phase correlation, and the travel
    direction in [0, 180) so it is comparable with a blur AXIS (a smear has no
    sign - travelling left or right blurs identically)."""
    if prev_gray is None or cur_gray is None or prev_gray.shape != cur_gray.shape:
        return None
    win = cv2.createHanningWindow((prev_gray.shape[1], prev_gray.shape[0]),
                                  cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(prev_gray.astype(np.float32),
                                     cur_gray.astype(np.float32), win)
    mag = float(np.hypot(dx, dy))
    ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    return {"dx": float(dx), "dy": float(dy), "px": mag, "angle_deg": ang}


def angle_difference(a, b):
    """Smallest difference between two AXES in degrees, in [0, 90]."""
    d = abs(float(a) - float(b)) % 180.0
    return d if d <= 90.0 else 180.0 - d


# --------------------------------------------------------------------------- #
# Session analysis
# --------------------------------------------------------------------------- #
def read_pool(session_dir):
    p = Path(session_dir) / "meta" / "pool.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r.get("filename")
                and str(r.get("dropped", "0")).strip() not in ("1", "true", "True")]


def _fnum(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyse_session(sid, session_dir, cfg, progress=None):
    rows = read_pool(session_dir)
    if len(rows) < 3:
        return None
    rows = rows[: cfg["MAX_FRAMES"]]
    rgb_dir = Path(session_dir) / "rgb"

    recs, prev = [], None
    for r in rows:
        g = _gray(rgb_dir / r["filename"], cfg["WORK_WIDTH"])
        if g is None:
            continue
        aniso, axis, sharp = blur_axis(g, cfg["N_ANGLES"])
        shift = frame_shift(prev, g)
        recs.append({"filename": r["filename"], "sharpness": sharp,
                     "anisotropy": aniso, "blur_axis_deg": axis,
                     "shift_px": shift["px"] if shift else None,
                     "travel_deg": shift["angle_deg"] if shift else None,
                     "exposure": _fnum(r, "exposure"), "gain": _fnum(r, "gain")})
        prev = g
        if progress:
            progress.update()
    if len(recs) < 3:
        return None

    sharp = np.array([r["sharpness"] for r in recs], float)
    med = float(np.median(sharp))
    blurry = sharp < cfg["BLUR_REL_THRESHOLD"] * med

    out = {"session": sid, "n_frames": len(recs),
           "median_sharpness": med,
           "blurry_fraction": float(blurry.mean()),
           "evidence": {}}

    # --- 1. Is the blur directional at all? --------------------------------
    # Measured over EVERY frame, not only those blurrier than their neighbours.
    # Driving at a constant speed blurs every frame equally, so a relative
    # threshold finds "no blurry frames" precisely when the problem is worst.
    # The blurry subset is reported as a refinement, never as a gate.
    aniso_all = np.array([r["anisotropy"] for r in recs], float)
    aniso_blurry = np.array([r["anisotropy"] for r, b in zip(recs, blurry) if b],
                            float)
    out["evidence"]["anisotropy"] = {
        "median_all": float(np.median(aniso_all)),
        "median_blurry": float(np.median(aniso_blurry)) if aniso_blurry.size
        else None,
        "n_blurry": int(blurry.sum()),
        # A session whose sharpness barely varies is uniformly affected, which
        # is itself informative: constant speed, or fixed optics.
        "sharpness_cv": float(np.std(sharp) / max(1e-9, np.mean(sharp)))}

    # --- 2. Does the blur axis match the direction of travel? --------------
    # ONLY frames whose blur is actually directional take part. On an isotropic
    # frame the energy curve is flat, so its argmin is noise - and a noise axis
    # still "agrees" with the travel direction at chance rate, which is enough
    # to fabricate a confident MOTION verdict on a plainly defocused session.
    # An axis is meaningful only when there is an axis.
    pairs = [(r["blur_axis_deg"], r["travel_deg"]) for r in recs
             if r["travel_deg"] is not None and r["shift_px"]
             and r["shift_px"] > 0.5
             and r["anisotropy"] >= cfg["ANISO_WEAK"]]
    if pairs:
        diffs = np.array([angle_difference(a, t) for a, t in pairs], float)
        agree = float((diffs <= cfg["AXIS_TOLERANCE_DEG"]).mean())
        # A random axis would agree ~28% of the time at a 25 degree tolerance
        # (2*25/180), so that is the number to beat, not 0.
        out["evidence"]["axis_agreement"] = {
            "n": int(diffs.size),
            "median_difference_deg": float(np.median(diffs)),
            "fraction_agreeing": agree,
            "chance_level": float(2.0 * cfg["AXIS_TOLERANCE_DEG"] / 180.0)}
    else:
        n_moving = sum(1 for r in recs if r["shift_px"] and r["shift_px"] > 0.5)
        out["evidence"]["axis_agreement"] = {
            "n": 0,
            "note": ("no frame is directional enough for its blur axis to mean "
                     "anything" if n_moving else "no measurable camera motion")}

    # --- 3. Does sharpness fall as displacement rises? ---------------------
    sp = [(r["shift_px"], r["sharpness"]) for r in recs
          if r["shift_px"] is not None]
    if len(sp) >= 8:
        s = np.array([a for a, _ in sp], float)
        k = np.array([b for _, b in sp], float)
        rs, rk = np.argsort(np.argsort(s)), np.argsort(np.argsort(k))
        rho = float(np.corrcoef(rs, rk)[0, 1])
        out["evidence"]["speed_coupling"] = {
            "n": len(sp), "spearman_shift_vs_sharpness": rho}
    else:
        out["evidence"]["speed_coupling"] = {"n": len(sp), "note": "too few frames"}

    # --- 4. Exposure / gain, when the capture log recorded them ------------
    for key in ("exposure", "gain"):
        vals = [(r[key], r["sharpness"]) for r in recs if r[key] is not None]
        if len(vals) >= 8 and len({v for v, _ in vals}) > 1:
            a = np.array([v for v, _ in vals], float)
            b = np.array([v for _, v in vals], float)
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            out["evidence"][f"{key}_coupling"] = {
                "n": len(vals),
                f"spearman_{key}_vs_sharpness": float(np.corrcoef(ra, rb)[0, 1])}

    out["verdict"] = verdict(out, cfg)
    out["worst_frames"] = [r["filename"] for r in
                           sorted(recs, key=lambda r: r["sharpness"])[:10]]
    return out


def verdict(res, cfg):
    """Weigh the three lines of evidence into a plain-language conclusion.

    Deliberately willing to say 'inconclusive'. A confident wrong diagnosis
    here sends you to re-shoot a whole field for the wrong reason."""
    ev = res["evidence"]
    an = ev["anisotropy"]
    # Prefer the blurry subset when one exists (strongest signal), otherwise
    # judge the session as a whole - a uniformly smeared session has no
    # "blurrier" frames to isolate.
    aniso = an["median_blurry"] if an["median_blurry"] is not None \
        else an["median_all"]
    axis = ev.get("axis_agreement", {})
    speed = ev.get("speed_coupling", {})
    reasons, motion_votes, optics_votes = [], 0, 0

    if an["n_blurry"] == 0 and an["sharpness_cv"] < 0.15:
        reasons.append(f"sharpness is uniform across the session "
                       f"(CV {an['sharpness_cv']:.2f}) - whatever the cause, it "
                       f"affects every frame equally")

    if aniso is None:
        reasons.append("no frames could be analysed")
    elif aniso >= cfg["ANISO_STRONG"]:
        motion_votes += 1
        reasons.append(f"blur is strongly DIRECTIONAL (anisotropy {aniso:.2f}) - "
                       f"defocus and a dirty lens blur equally in all directions")
    elif aniso <= cfg["ANISO_WEAK"]:
        optics_votes += 1
        reasons.append(f"blur is ISOTROPIC (anisotropy {aniso:.2f}) - consistent "
                       f"with defocus, a wet/dirty lens, or too small an aperture, "
                       f"NOT with motion")
    else:
        reasons.append(f"anisotropy {aniso:.2f} is between the thresholds; "
                       f"directionality is unclear")

    if axis.get("n", 0) >= 5:
        frac, chance = axis["fraction_agreeing"], axis["chance_level"]
        if frac >= max(0.6, chance * 2):
            motion_votes += 2          # the decisive test
            reasons.append(f"the blur axis MATCHES the measured direction of "
                           f"travel in {frac*100:.0f}% of blurry frames "
                           f"(chance {chance*100:.0f}%) - nothing but motion "
                           f"produces that")
        elif frac <= chance:
            optics_votes += 1
            reasons.append(f"the blur axis is unrelated to the direction of "
                           f"travel ({frac*100:.0f}% vs chance {chance*100:.0f}%)")

    rho = speed.get("spearman_shift_vs_sharpness")
    if rho is not None:
        if rho <= -0.3:
            motion_votes += 1
            reasons.append(f"sharpness FALLS as inter-frame displacement rises "
                           f"(rho {rho:.2f}) - faster travel, blurrier frames")
        elif rho >= -0.05:
            reasons.append(f"sharpness is unrelated to displacement (rho {rho:.2f})")

    if motion_votes >= 2 and motion_votes > optics_votes:
        label, action = "MOTION BLUR", [
            "Shorten exposure: cap it in the ZED capture settings. Motion blur "
            "length is exposure time x ground speed, so halving exposure halves "
            "the smear.",
            "Compensate the lost light with gain or more illumination rather "
            "than exposure. Sensor noise is far less damaging to segmentation "
            "than smeared leaf margins.",
            "Drive slower over the beds you intend to annotate.",
            "Capture in brighter conditions, which lets the auto-exposure pick "
            "a shorter time by itself."]
    elif optics_votes >= 1 and optics_votes >= motion_votes:
        label, action = "OPTICAL / FOCUS", [
            "Check focus at your actual working distance - a lens focused at "
            "infinity is soft on plants 1 m away.",
            "Clean the lens: dust, spray residue and condensation all read as "
            "uniform blur.",
            "Check for a protective window or housing glass in front of the "
            "lens.",
            "Verify the ZED is in the resolution/mode you expect - a lower "
            "internal mode upscaled to the output size looks uniformly soft."]
    else:
        label, action = "INCONCLUSIVE", [
            "Increase MAX_FRAMES, or run on a session with more consistent "
            "motion.",
            "If the camera was nearly stationary, the direction test cannot "
            "fire - re-run on a session where the rig was moving."]

    return {"label": label, "reasons": reasons, "recommended_actions": action,
            "motion_votes": motion_votes, "optics_votes": optics_votes}


def main():
    cfg = CONFIG
    root = Path(cfg["DATASET_ROOT"]) / "sessions"
    if not root.exists():
        sys.exit(f"ERROR: {root} not found. Run extract_sessions.py first.")
    sids = sorted(p.name for p in root.iterdir() if p.is_dir())
    if cfg["ONLY_SESSIONS"]:
        sids = [s for s in sids if s in cfg["ONLY_SESSIONS"]]
    if not sids:
        sys.exit("No sessions selected.")

    print("Blur diagnosis: MOTION vs OPTICS")
    print("  Extraction is lossless (FFV1 -> full-resolution PNG, no resize),")
    print("  so any blur was captured, not introduced by the pipeline.\n")

    for sid in sids:
        prog = Progress(cfg["MAX_FRAMES"], f"  [{sid}]", unit="frames")
        res = analyse_session(sid, root / sid, cfg, prog)
        prog.close()
        if res is None:
            print(f"  [{sid}] not enough readable frames\n")
            continue
        v = res["verdict"]
        print(f"  [{sid}] {res['n_frames']} frames | median sharpness "
              f"{res['median_sharpness']:.1f} | "
              f"{res['blurry_fraction']*100:.0f}% blurry")
        print(f"      VERDICT: {v['label']}")
        for r in v["reasons"]:
            print(f"        - {r}")
        print("      What to change:")
        for a in v["recommended_actions"]:
            print(f"        * {a}")
        print(f"      worst frames: {', '.join(res['worst_frames'][:5])}\n")


if __name__ == "__main__":
    main()
