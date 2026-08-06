"""Resolving a manifest's image path against the real session layout.

    <images_root>/<session_id>/rgb/<session_id>_<idx>.png
    <images_root>/<session_id>/depth/<session_id>_<idx>.png   SAME FILENAME
    <images_root>/<session_id>/meta/...

The identical filename in depth/ is why a blind recursive search is not merely
slow but wrong: it can return a depth PNG as a training image.
"""
import numpy as np
import cv2
import pytest

from conftest import load_script

sd = load_script("training/seg_dataset.py")


@pytest.fixture
def sessions(tmp_path):
    root = tmp_path / "sessions"
    for sess in ("vid2_20260108_122731", "vid3_20260108_103135"):
        for sub in ("rgb", "depth", "meta"):
            (root / sess / sub).mkdir(parents=True)
        for i in (123, 124):
            name = f"{sess}_{i:06d}.png"
            cv2.imwrite(str(root / sess / "rgb" / name),
                        np.full((8, 8, 3), 200, np.uint8))
            cv2.imwrite(str(root / sess / "depth" / name),
                        np.zeros((8, 8, 3), np.uint8))
    sd._INDEX_CACHE.clear()
    return root


def test_bare_filename_resolves_through_the_session_in_its_name(sessions):
    p = sd.resolve_image("vid2_20260108_122731_000123.png", sessions)
    assert p.parent.name == "rgb"
    assert p.parent.parent.name == "vid2_20260108_122731"


def test_rgb_wins_over_the_identically_named_depth_frame(sessions):
    """The failure this guards: a depth PNG silently used as a training image."""
    p = sd.resolve_image("vid2_20260108_122731_000123.png", sessions)
    assert p.parent.name == "rgb"
    assert cv2.imread(str(p)).mean() > 100      # rgb is bright, depth is zeros


def test_explicit_session_id_is_used_when_given(sessions):
    p = sd.resolve_image("vid3_20260108_103135_000124.png", sessions,
                         session_id="vid3_20260108_103135")
    assert p.parent.parent.name == "vid3_20260108_103135"


def test_a_relative_path_that_already_exists_is_used_as_is(sessions):
    rel = "vid2_20260108_122731/rgb/vid2_20260108_122731_000124.png"
    assert sd.resolve_image(rel, sessions).exists()


def test_absolute_path_is_returned_untouched(sessions):
    a = sessions / "vid2_20260108_122731" / "rgb" / "vid2_20260108_122731_000123.png"
    assert sd.resolve_image(a, sessions) == a


def test_frames_directly_under_a_session_still_resolve(tmp_path):
    """Not every layout has an rgb/ subfolder."""
    root = tmp_path / "s"
    (root / "sess_a").mkdir(parents=True)
    cv2.imwrite(str(root / "sess_a" / "sess_a_000001.png"),
                np.zeros((4, 4, 3), np.uint8))
    sd._INDEX_CACHE.clear()
    assert sd.resolve_image("sess_a_000001.png", root).exists()


def test_an_unconventional_name_still_resolves_via_the_index(tmp_path):
    root = tmp_path / "s"
    (root / "odd" / "deep").mkdir(parents=True)
    cv2.imwrite(str(root / "odd" / "deep" / "whatever.png"),
                np.zeros((4, 4, 3), np.uint8))
    sd._INDEX_CACHE.clear()
    assert sd.resolve_image("whatever.png", root).exists()


def test_the_recursive_index_is_built_once_per_root(tmp_path, monkeypatch):
    """It is consulted per image per epoch; rebuilding it each time would
    dominate training on a root holding tens of thousands of PNGs."""
    root = tmp_path / "s"
    (root / "d").mkdir(parents=True)
    for n in ("a.png", "b.png"):
        cv2.imwrite(str(root / "d" / n), np.zeros((4, 4, 3), np.uint8))
    sd._INDEX_CACHE.clear()

    from pathlib import Path
    calls = []
    real = Path.rglob

    def counted(self, pat):
        calls.append(pat)
        return real(self, pat)

    monkeypatch.setattr(Path, "rglob", counted)
    sd.resolve_image("a.png", root)
    sd.resolve_image("b.png", root)
    sd.resolve_image("a.png", root)
    assert len(calls) == 1


def test_a_missing_image_says_what_images_root_should_point_at(sessions):
    with pytest.raises(FileNotFoundError, match="session id folders"):
        sd.resolve_image("nope_000001.png", sessions)


def test_pointing_at_a_session_instead_of_the_sessions_root_fails_clearly(
        sessions):
    """The most likely mistake, and it must not silently half-work."""
    inner = sessions / "vid2_20260108_122731"
    sd._INDEX_CACHE.clear()
    # Still findable via the index here, but the canonical path does not exist,
    # which is the case the error message exists for.
    with pytest.raises(FileNotFoundError):
        sd.resolve_image("vid9_absent_000001.png", inner)


# --------------------------------------------------------------------------- #
# multiple roots: sessions split across separate parent folders
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_roots(tmp_path):
    """Two independent 'sessions folders' - the weed capture set and a
    separately-recorded onion set - each holding a DIFFERENT session."""
    root_a = tmp_path / "weed_sessions"
    root_b = tmp_path / "onion_sessions"
    (root_a / "vid2_weed" / "rgb").mkdir(parents=True)
    (root_b / "onion1" / "rgb").mkdir(parents=True)
    cv2.imwrite(str(root_a / "vid2_weed" / "rgb" / "vid2_weed_000001.png"),
                np.full((4, 4, 3), 10, np.uint8))
    cv2.imwrite(str(root_b / "onion1" / "rgb" / "onion1_000001.png"),
                np.full((4, 4, 3), 20, np.uint8))
    sd._INDEX_CACHE.clear()
    return root_a, root_b


def test_a_frame_resolves_under_whichever_root_actually_has_its_session(
        two_roots):
    root_a, root_b = two_roots
    pa = sd.resolve_image("vid2_weed_000001.png", [root_a, root_b])
    pb = sd.resolve_image("onion1_000001.png", [root_a, root_b])
    assert pa.parent.parent.name == "vid2_weed"
    assert pb.parent.parent.name == "onion1"


def test_root_order_does_not_matter(two_roots):
    root_a, root_b = two_roots
    p1 = sd.resolve_image("onion1_000001.png", [root_a, root_b])
    p2 = sd.resolve_image("onion1_000001.png", [root_b, root_a])
    assert p1 == p2


def test_a_missing_frame_lists_every_root_tried(two_roots):
    root_a, root_b = two_roots
    with pytest.raises(FileNotFoundError) as e:
        sd.resolve_image("nope_000001.png", [root_a, root_b])
    assert str(root_a) in str(e.value) and str(root_b) in str(e.value)


def test_a_single_path_still_works_unchanged(sessions):
    """The common case (one dataset) must not have to pass a list."""
    p = sd.resolve_image("vid2_20260108_122731_000123.png", sessions)
    assert p.exists()


def test_as_roots_normalizes_single_and_multiple():
    assert sd.as_roots("/a") == [sd.Path("/a")]
    assert sd.as_roots(["/a", "/b"]) == [sd.Path("/a"), sd.Path("/b")]
    assert sd.as_roots(("/a", "/b")) == [sd.Path("/a"), sd.Path("/b")]


def test_a_second_root_is_not_scanned_when_the_first_answers_cheaply(
        two_roots, monkeypatch):
    """The cheap canonical-path check must win before ANY root pays for a
    full recursive walk - a scan of a huge first root must not block
    resolving a frame that is trivially found on the second."""
    root_a, root_b = two_roots
    calls = []
    from pathlib import Path as P
    real = P.rglob

    def counted(self, pat):
        calls.append(self)
        return real(self, pat)

    monkeypatch.setattr(P, "rglob", counted)
    sd.resolve_image("onion1_000001.png", [root_a, root_b])
    assert calls == [], "the canonical path must resolve without any scan"
