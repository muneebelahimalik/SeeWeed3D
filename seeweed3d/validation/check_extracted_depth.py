#!/usr/bin/env python3
"""
SeeWeed3D - is an EXTRACTED session's depth/ real millimetres?

    python -m seeweed3d.validation.check_extracted_depth --sessions E:\\...\\sessions
    python -m seeweed3d.validation.check_extracted_depth --sessions ... --write

WHY THIS EXISTS
---------------
The prelabelers decide whether to use depth by reading `depth_kind` from a
session's meta/session.json. That field was only added to the extractor in #80,
so every session extracted before it has no `depth_kind` at all - which reads
as "unknown", and the height veto correctly refuses to run.

Correctly, because the field is not decoration. ffmpeg will produce gray16le
from ANY source, including an 8-bit preview, by scaling values it made up - so
a 16-bit PNG on disk is not by itself evidence that the numbers are
millimetres. That is the whole reason REQUIRE_16BIT_DEPTH exists at extraction
time, and it is why this module classifies from the DATA rather than assuming
from the container.

Re-extracting those sessions would also answer the question, at the cost of
hours and of whatever curation state the pool carries. This reads the PNGs
instead, says what they are, and - only with --write, and only when the answer
is unambiguous - records it.

HOW A NORMALISED PREVIEW IS RECOGNISED
---------------------------------------
Same signature `inspect_depth_video.py` uses on the source video, applied to
the extracted frames. Per-frame normalisation divides each frame by its OWN
maximum, so:

  * every frame's maximum is pinned near the top of the range, and
  * those maxima agree with each other across frames, and
  * the peak is a lone outlier rather than a saturated plateau.

A fixed metric scale does none of those: its maximum is whatever happened to be
farthest in that frame, and it reaches the top of the range only when the scene
runs past the clip - at which point everything beyond it pins to one value,
which is a large uniform region rather than a single pixel.

AND HOW METRIC DEPTH IS RECOGNISED
-----------------------------------
Plausibility against the rig. A boom-mounted camera over a bed reads a median
of roughly one to two metres, so a median in PLAUSIBLE_MM says these are
millimetres of this scene. A median of 30 000 does not describe any distance
this machine has ever been at.

Neither test alone is trusted to say "metric". A session that satisfies the
normalisation signature is reported as a preview; one that is plainly
millimetres is reported metric; anything else is reported UNCERTAIN and --write
refuses it. Guessing here would reintroduce, by hand, exactly the fabricated
millimetres the extractor's guard exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Frames to sample per session. The signature is a property of the encoding,
#: not of one frame, so a handful spread through the session settles it.
SAMPLE_FRAMES = 12

#: A per-frame-normalised frame's peak sits this close to the top of the range,
#: as a fraction of the dtype maximum. Expressed as a fraction so the same test
#: reads an 8-bit preview and a 16-bit one scaled from the same source.
NORMALISED_MAX_FRAC = 0.96

#: How much the per-frame maxima may differ from each other, as a fraction of
#: the dtype maximum. Under per-frame normalisation they are the same value by
#: construction; under a fixed scale they track whatever is farthest per frame.
NORMALISED_MAX_SPREAD = 0.05

#: Fraction of pixels AT the frame's peak below which the top was reached by
#: outliers rather than by a clipped plateau.
NORMALISED_SAT_FRAC = 0.02

#: Median depth, in millimetres, that a boom-mounted camera over a bed could
#: plausibly report. Deliberately wide: this rejects "these are not distances",
#: not "this is the wrong mount height".
PLAUSIBLE_MM = (250.0, 6000.0)


def sample_depth_frames(session_dir, limit=SAMPLE_FRAMES):
    """Up to `limit` depth PNGs spread through the session, as raw arrays."""
    d = Path(session_dir) / "depth"
    if not d.is_dir():
        return []
    files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
    if not files:
        return []
    step = max(1, len(files) // max(1, limit))
    out = []
    for p in files[::step][:limit]:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            out.append(raw)
    return out


def classify_depth(frames, plausible_mm=PLAUSIBLE_MM):
    """What these extracted depth frames are. Never guesses.

    Returns a dict with `kind` in {metric, preview, not_16bit, missing,
    uncertain} and the evidence behind it, so a caller can print WHY rather
    than only what."""
    if not frames:
        return {"kind": "missing", "reason": "no depth PNGs found"}

    dtypes = {str(f.dtype) for f in frames}
    if dtypes != {"uint16"}:
        return {"kind": "not_16bit",
                "reason": f"depth PNGs are {sorted(dtypes)}, not uint16 - an "
                          f"8-bit file cannot hold a millimetre range",
                "dtypes": sorted(dtypes)}

    full = 65535.0
    maxima, sat, medians = [], [], []
    for f in frames:
        valid = f[f > 0]
        if valid.size < 64:
            continue
        m = float(valid.max())
        maxima.append(m)
        sat.append(float((f >= m).mean()))
        medians.append(float(np.median(valid)))
    if not maxima:
        return {"kind": "uncertain",
                "reason": "every sampled frame is almost entirely the invalid "
                          "sentinel - nothing to classify"}

    top, low = max(maxima), min(maxima)
    normalised = (low >= NORMALISED_MAX_FRAC * full
                  and (top - low) <= NORMALISED_MAX_SPREAD * full
                  and max(sat) < NORMALISED_SAT_FRAC)
    median_mm = float(np.median(medians))
    plausible = plausible_mm[0] <= median_mm <= plausible_mm[1]

    ev = {"frames_sampled": len(maxima),
          "median_value": round(median_mm, 1),
          "max_value": round(top, 1),
          "max_spread": round(top - low, 1),
          "peak_saturated_frac": round(max(sat), 6)}

    if normalised:
        return {"kind": "preview", **ev,
                "reason": f"every frame's maximum is pinned near the top of the "
                          f"16-bit range and they agree within {top - low:.0f} "
                          f"counts, with the peak a lone outlier - the "
                          f"signature of per-frame normalisation at capture"}
    if plausible:
        return {"kind": "metric", **ev,
                "reason": f"median {median_mm:.0f} is a plausible camera "
                          f"distance in millimetres, and the per-frame maxima "
                          f"vary with scene content rather than being pinned"}
    return {"kind": "uncertain", **ev,
            "reason": f"median value {median_mm:.0f} is not a plausible "
                      f"distance in millimetres for this rig "
                      f"({plausible_mm[0]:.0f}-{plausible_mm[1]:.0f}), but the "
                      f"per-frame-normalisation signature does not fit either. "
                      f"If {median_mm:.0f} mm IS your real mount height "
                      f"(a close-mounted rig can be under 250 mm), rerun with "
                      f"--min-mm / --max-mm to say so - the default range is a "
                      f"guess about a typical boom, not a physical law."}


def write_depth_kind(session_dir, kind):
    """Record `depth_kind` in meta/session.json, preserving everything else."""
    path = Path(session_dir) / "meta" / "session.json"
    doc = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8")) or {}
        except ValueError:
            doc = {}
    doc["depth_kind"] = kind
    doc["depth_kind_source"] = "check_extracted_depth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", required=True,
                   help="the sessions root - the folder whose children are "
                        "session ids")
    p.add_argument("--only", nargs="*", default=[],
                   help="restrict to these session ids")
    p.add_argument("--write", action="store_true",
                   help="record depth_kind in each session.json. Refuses to "
                        "write an uncertain classification.")
    p.add_argument("--min-mm", type=float, default=PLAUSIBLE_MM[0],
                   help=f"lower bound of a plausible distance for YOUR rig "
                        f"(default {PLAUSIBLE_MM[0]:.0f}). Widen this only "
                        f"because you know the mount height, not to make an "
                        f"'uncertain' verdict go away.")
    p.add_argument("--max-mm", type=float, default=PLAUSIBLE_MM[1],
                   help=f"upper bound (default {PLAUSIBLE_MM[1]:.0f})")
    a = p.parse_args(argv)
    plausible_mm = (a.min_mm, a.max_mm)
    if plausible_mm != PLAUSIBLE_MM:
        print(f"  Using a custom plausible range: {plausible_mm[0]:.0f}-"
              f"{plausible_mm[1]:.0f} mm (default is {PLAUSIBLE_MM[0]:.0f}-"
              f"{PLAUSIBLE_MM[1]:.0f}).\n")

    root = Path(a.sessions)
    if not root.is_dir():
        raise SystemExit(f"ERROR: not a directory: {root}")
    sids = sorted(q.name for q in root.iterdir() if q.is_dir())
    if a.only:
        sids = [s for s in sids if s in a.only]
    if not sids:
        raise SystemExit(f"ERROR: no sessions under {root}")

    wrote = uncertain = 0
    for sid in sids:
        res = classify_depth(sample_depth_frames(root / sid), plausible_mm)
        kind = res["kind"]
        print(f"\n  {sid}")
        print(f"    depth_kind : {kind}")
        print(f"    because    : {res['reason']}")
        if "median_value" in res:
            print(f"    median {res['median_value']} | max {res['max_value']} "
                  f"| max spread {res['max_spread']} | peak saturated "
                  f"{res['peak_saturated_frac']:.4f}")
        if a.write:
            if kind == "uncertain":
                uncertain += 1
                print(f"    NOT WRITTEN - an uncertain classification must not "
                      f"become a claim that these are millimetres.")
            else:
                write_depth_kind(root / sid, kind)
                wrote += 1
                print(f"    -> wrote depth_kind={kind} to meta/session.json")

    if a.write:
        print(f"\n  wrote {wrote} session(s); {uncertain} left uncertain.")
        print(f"  Sessions now marked 'metric' will have the height veto "
              f"applied by the prelabelers.")
    else:
        print(f"\n  Nothing was written. Re-run with --write to record these "
              f"in each meta/session.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
