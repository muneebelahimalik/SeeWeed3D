"""Blur diagnosis must actually discriminate motion blur from defocus.

Validated against blur it CREATED, so the ground truth is known by
construction rather than assumed."""
import csv

import cv2
import numpy as np
import pytest

from conftest import load_script

db = load_script("validation/diagnose_blur.py")


def _texture(size=256, seed=0):
    """Isotropic random texture - no built-in directional bias that could fake
    an anisotropy result."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size), dtype=np.uint8).astype(np.float32)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _motion_blur(gray, length=15, angle_deg=0.0):
    """Convolve with a line kernel - exactly what camera motion does."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D(((length - 1) / 2.0, (length - 1) / 2.0),
                                angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return cv2.filter2D(gray, -1, k / (s if s > 0 else 1.0))


def _defocus(gray, radius=5):
    """Disc convolution - the standard defocus model. Isotropic."""
    d = 2 * radius + 1
    k = np.zeros((d, d), np.float32)
    cv2.circle(k, (radius, radius), radius, 1.0, -1)
    return cv2.filter2D(gray, -1, k / k.sum())


# --------------------------------------------------------------------------- #
# The core discriminator
# --------------------------------------------------------------------------- #
def test_motion_blur_is_directional_and_defocus_is_not():
    """THE test. If this fails the whole diagnosis is worthless."""
    g = _texture()
    motion = _motion_blur(g, length=15, angle_deg=0.0)
    defocus = _defocus(g, radius=5)

    a_motion, _, _ = db.blur_axis(motion)
    a_defocus, _, _ = db.blur_axis(defocus)

    assert a_motion > 0.30, f"motion blur read as isotropic ({a_motion:.3f})"
    assert a_defocus < 0.12, f"defocus read as directional ({a_defocus:.3f})"
    assert a_motion > 3 * a_defocus


@pytest.mark.parametrize("angle", [0.0, 30.0, 60.0, 90.0, 135.0])
def test_blur_axis_recovers_the_direction_of_the_smear(angle):
    """The axis must be recovered, not merely detected - the whole diagnosis
    rests on comparing it with the direction of travel."""
    blurred = _motion_blur(_texture(), length=17, angle_deg=angle)
    aniso, axis, _ = db.blur_axis(blurred, n_angles=36)
    assert aniso > 0.25
    # getRotationMatrix2D rotates counter-clockwise while image y runs
    # downward, so the smear axis appears mirrored; compare as an unsigned
    # axis, which is all the diagnosis uses.
    assert min(db.angle_difference(axis, angle),
               db.angle_difference(axis, -angle)) <= 15.0


def test_angle_difference_treats_directions_as_unsigned_axes():
    """A smear has no sign: travelling left or right blurs identically."""
    assert db.angle_difference(10.0, 170.0) == pytest.approx(20.0)
    assert db.angle_difference(0.0, 180.0) == pytest.approx(0.0)
    assert db.angle_difference(0.0, 90.0) == pytest.approx(90.0)
    assert db.angle_difference(80.0, 100.0) == pytest.approx(20.0)


def test_sharper_images_score_higher():
    g = _texture()
    assert db.sharpness(g) > db.sharpness(_motion_blur(g, 15))
    assert db.sharpness(g) > db.sharpness(_defocus(g, 5))


def test_directional_energy_dips_along_the_smear_axis():
    """The mechanism: smearing destroys detail ALONG the smear and leaves
    detail perpendicular to it."""
    blurred = _motion_blur(_texture(), length=21, angle_deg=0.0)
    th, e, _ = db.directional_energy(blurred, n_angles=36)
    lo = int(np.argmin(e))
    hi = int(np.argmax(e))
    assert e[hi] > 2 * e[lo]
    assert db.angle_difference(np.degrees(th[lo]), np.degrees(th[hi])) > 60


def test_phase_correlation_recovers_a_known_shift():
    g = _texture()
    shifted = np.roll(g, 8, axis=1)
    s = db.frame_shift(g, shifted)
    assert s is not None
    assert abs(s["px"] - 8.0) < 1.5
    assert db.angle_difference(s["angle_deg"], 0.0) < 15.0


# --------------------------------------------------------------------------- #
# End-to-end verdicts on synthetic sessions with KNOWN cause
# --------------------------------------------------------------------------- #
def _session(tmp_path, sid, kind, n=24, step=9):
    """A synthetic session panning across a large texture.

    `kind` controls the blur applied: 'motion' smears ALONG the pan direction
    (as real camera motion would), 'defocus' applies an isotropic disc, 'sharp'
    applies none."""
    sdir = tmp_path / "sessions" / sid
    (sdir / "rgb").mkdir(parents=True)
    (sdir / "meta").mkdir(parents=True)
    big = _texture(size=600, seed=3)

    rows = []
    for i in range(n):
        x = i * step
        crop = big[100:356, x:x + 256]
        if kind == "motion":
            frame = _motion_blur(crop, length=15, angle_deg=0.0)   # along +x pan
        elif kind == "defocus":
            frame = _defocus(crop, radius=5)
        else:
            frame = crop
        fn = f"{sid}_{i:06d}.png"
        cv2.imwrite(str(sdir / "rgb" / fn),
                    np.clip(frame, 0, 255).astype(np.uint8))
        rows.append({"filename": fn, "video_frame_idx": str(i)})

    with open(sdir / "meta" / "pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "video_frame_idx"])
        w.writeheader()
        w.writerows(rows)
    return sdir


def _cfg(**over):
    c = dict(db.CONFIG)
    c.update(WORK_WIDTH=256, MAX_FRAMES=40)
    c.update(over)
    return c


def test_end_to_end_verdict_identifies_motion_blur(tmp_path):
    """A session panning horizontally with a horizontal smear must be called
    motion blur - the blur axis agrees with the measured travel direction."""
    sdir = _session(tmp_path, "mot", "motion")
    res = db.analyse_session("mot", sdir, _cfg())
    assert res is not None
    v = res["verdict"]
    assert v["label"] == "MOTION BLUR", v
    assert v["motion_votes"] >= 2
    agree = res["evidence"]["axis_agreement"]
    assert agree["n"] > 0
    assert agree["fraction_agreeing"] > agree["chance_level"]
    assert any("exposure" in a.lower() for a in v["recommended_actions"])


def test_end_to_end_verdict_identifies_optical_blur(tmp_path):
    """The same pan with an isotropic disc blur must NOT be blamed on motion."""
    sdir = _session(tmp_path, "opt", "defocus")
    res = db.analyse_session("opt", sdir, _cfg())
    assert res is not None
    v = res["verdict"]
    assert v["label"] != "MOTION BLUR", v
    assert res["evidence"]["anisotropy"]["median_all"] < 0.2
    if v["label"] == "OPTICAL / FOCUS":
        assert any("focus" in a.lower() or "clean" in a.lower()
                   for a in v["recommended_actions"])


def test_verdict_says_inconclusive_rather_than_guessing(tmp_path):
    """A confident wrong diagnosis sends you to re-shoot a field for the wrong
    reason, so weak evidence must be reported as weak."""
    sdir = _session(tmp_path, "sharp", "sharp")
    res = db.analyse_session("sharp", sdir, _cfg())
    assert res is not None
    assert res["verdict"]["label"] in ("INCONCLUSIVE", "OPTICAL / FOCUS")
    # A sharp session has few or no frames below the relative blur threshold.
    assert res["blurry_fraction"] < 0.5


def test_dropped_frames_are_skipped(tmp_path):
    """Respects curate_pool's decisions rather than re-analysing frames the
    user already excluded."""
    sdir = _session(tmp_path, "cur", "motion", n=10)
    p = sdir / "meta" / "pool.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for i, r in enumerate(rows):
        r["dropped"] = "1" if i < 6 else "0"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "video_frame_idx", "dropped"])
        w.writeheader()
        w.writerows(rows)
    assert len(db.read_pool(sdir)) == 4


def test_worst_frames_are_reported_for_inspection(tmp_path):
    sdir = _session(tmp_path, "w", "motion", n=12)
    res = db.analyse_session("w", sdir, _cfg())
    assert res["worst_frames"]
    assert all(f.endswith(".png") for f in res["worst_frames"])
