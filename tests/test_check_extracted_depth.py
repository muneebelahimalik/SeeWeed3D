"""Is an extracted session's depth/ real millimetres?

`depth_kind` was only added to the extractor in #80, so sessions extracted
before it read as "unknown" and the height veto correctly refuses to run. This
answers the question from the DATA rather than by assuming from the container -
ffmpeg produces gray16le from any source, including an 8-bit preview, by
scaling values it invented.
"""
import json

import cv2
import numpy as np

from conftest import load_script

cd = load_script("validation/check_extracted_depth.py")


def _write(session_dir, frames):
    d = session_dir / "depth"
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / f"s_{i:06d}.png"), f)
    return session_dir


def _metric(n=8, h=64, w=64, camera_mm=1200.0, seed=0):
    """A fixed metric scale: values are millimetres, and each frame's maximum
    is whatever happened to be farthest in it."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        d = np.full((h, w), camera_mm, np.float32)
        d += rng.normal(0, 8, d.shape)                 # surface texture
        d[:8, :] = camera_mm + 200 + 40 * k            # a farther band, varying
        out.append(np.clip(d, 0, 65535).astype(np.uint16))
    return out


def _normalised_preview(n=8, h=64, w=64, seed=0):
    """Per-frame normalisation: each frame divided by its OWN maximum, so every
    maximum lands at the top of the range and they agree by construction."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.random((h, w)).astype(np.float32)
        d[0, 0] = 1.0                                  # the lone peak
        out.append((d * 65535).astype(np.uint16))
    return out


# --------------------------------------------------------------------------- #
# The three answers
# --------------------------------------------------------------------------- #
def test_real_millimetres_read_as_metric(tmp_path):
    res = cd.classify_depth(_metric())
    assert res["kind"] == "metric"
    assert 250 <= res["median_value"] <= 6000


def test_a_per_frame_normalised_preview_is_recognised(tmp_path):
    """The signature: maxima pinned at the top of the range, agreeing with each
    other, and the peak a lone outlier rather than a clipped plateau."""
    res = cd.classify_depth(_normalised_preview())
    assert res["kind"] == "preview"
    assert "per-frame normalisation" in res["reason"]


def test_an_8bit_file_cannot_be_metric(tmp_path):
    frames = [np.full((32, 32), 128, np.uint8)]
    res = cd.classify_depth(frames)
    assert res["kind"] == "not_16bit"


def test_no_depth_folder_is_reported_not_crashed(tmp_path):
    assert cd.classify_depth([])["kind"] == "missing"
    assert cd.sample_depth_frames(tmp_path / "nothing") == []


# --------------------------------------------------------------------------- #
# It refuses to guess
# --------------------------------------------------------------------------- #
def test_implausible_distances_are_uncertain_not_metric():
    """30 000 is not a distance this rig has ever been at, and neither is it
    the normalisation signature. Calling it metric would recreate by hand the
    fabricated millimetres the extractor's guard exists to prevent."""
    frames = [np.full((64, 64), 30000, np.uint16) for _ in range(4)]
    for i, f in enumerate(frames):
        f[0, 0] = 30000 + 100 * i          # maxima vary, so not normalised
    res = cd.classify_depth(frames)
    assert res["kind"] == "uncertain"


def test_a_frame_of_only_invalid_sentinel_is_uncertain():
    assert cd.classify_depth([np.zeros((64, 64), np.uint16)])["kind"] \
        == "uncertain"


def test_write_refuses_an_uncertain_classification(tmp_path, capsys):
    sess = _write(tmp_path / "sessions" / "s1",
                  [np.full((64, 64), 30000, np.uint16) for _ in range(4)])
    cd.main(["--sessions", str(tmp_path / "sessions"), "--write"])
    out = capsys.readouterr().out
    assert "NOT WRITTEN" in out
    assert not (sess / "meta" / "session.json").exists()


# --------------------------------------------------------------------------- #
# Backfilling the field
# --------------------------------------------------------------------------- #
def test_write_records_the_kind_and_says_where_it_came_from(tmp_path):
    sess = _write(tmp_path / "sessions" / "s1", _metric())
    cd.main(["--sessions", str(tmp_path / "sessions"), "--write"])
    doc = json.loads((sess / "meta" / "session.json").read_text())
    assert doc["depth_kind"] == "metric"
    assert doc["depth_kind_source"] == "check_extracted_depth"


def test_write_preserves_the_rest_of_session_json(tmp_path):
    """session.json carries the trip, the scene hint and the decode settings.
    Backfilling one field must not cost the others."""
    sess = _write(tmp_path / "sessions" / "s1", _metric())
    meta = sess / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "session.json").write_text(json.dumps(
        {"session_id": "s1", "scene_hint": "onion_only", "trip": "Visit1"}),
        encoding="utf-8")
    cd.main(["--sessions", str(tmp_path / "sessions"), "--write"])
    doc = json.loads((meta / "session.json").read_text())
    assert doc["scene_hint"] == "onion_only" and doc["trip"] == "Visit1"
    assert doc["depth_kind"] == "metric"


def test_nothing_is_written_without_the_flag(tmp_path):
    sess = _write(tmp_path / "sessions" / "s1", _metric())
    cd.main(["--sessions", str(tmp_path / "sessions")])
    assert not (sess / "meta" / "session.json").exists()


def test_a_backfilled_session_then_qualifies_for_the_height_veto(tmp_path):
    """End to end: the reason to run this at all is that the prelabelers then
    stop reporting depth_kind 'unknown' and use the depth that was there."""
    gr = load_script("perception/ground.py")
    sess = _write(tmp_path / "sessions" / "s1", _metric())
    assert not gr.has_metric_depth(sess)
    cd.main(["--sessions", str(tmp_path / "sessions"), "--write"])
    assert gr.has_metric_depth(sess)


def test_only_restricts_which_sessions_are_touched(tmp_path):
    a = _write(tmp_path / "sessions" / "s1", _metric())
    b = _write(tmp_path / "sessions" / "s2", _metric())
    cd.main(["--sessions", str(tmp_path / "sessions"), "--only", "s1",
             "--write"])
    assert (a / "meta" / "session.json").exists()
    assert not (b / "meta" / "session.json").exists()


def test_frames_are_sampled_across_the_session(tmp_path):
    """The signature is a property of the encoding, not of one frame, so a
    handful spread through the session settles it - and reading all 800 would
    make this unusable on a real pool."""
    sess = _write(tmp_path / "sessions" / "s1", _metric(n=100))
    got = cd.sample_depth_frames(sess, limit=6)
    assert 0 < len(got) <= 6
