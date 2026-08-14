"""What a session IS, read from the session itself.

`SessionInfo.scene` existed from the beginning and was always empty, because
nothing read `meta/session.json`. A dataset spanning onion-only, weed-only and
mixed drives was therefore split as if every session were interchangeable.
"""
import json

from conftest import load_script

sm = load_script("common/session_meta.py")


def _session(root, sid, **meta):
    d = root / sid / "meta"
    d.mkdir(parents=True, exist_ok=True)
    if meta:
        (d / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return root / sid


# --------------------------------------------------------------------------- #
# Scene normalisation
# --------------------------------------------------------------------------- #
def test_documented_scene_values_pass_through():
    for s in ("onion_only", "weed_only", "mixed", "unknown"):
        assert sm.normalise_scene(s) == s


def test_real_world_spellings_are_mapped():
    """`scene_hint: "onions"` was actually committed to this repo's trip
    config. Left unmapped it becomes its own stratum of one and quietly splits
    the allocator's quota."""
    assert sm.normalise_scene("onions") == "onion_only"
    assert sm.normalise_scene("Weeds") == "weed_only"
    assert sm.normalise_scene("mix") == "mixed"
    assert sm.normalise_scene("weed-only") == "weed_only"


def test_an_unrecognised_scene_is_unknown_not_a_new_stratum():
    assert sm.normalise_scene("bananas") == "unknown"
    assert sm.normalise_scene(None) == "unknown"


# --------------------------------------------------------------------------- #
# The session id
# --------------------------------------------------------------------------- #
def test_camera_and_date_come_from_the_session_id():
    assert sm.parse_session_id("vid3_20260108_132749") == ("vid3", "2026-01-08")


def test_a_non_conforming_id_yields_nothing_rather_than_a_guess():
    """A half-parsed date would group unrelated sessions, which is the exact
    failure the metadata exists to prevent."""
    assert sm.parse_session_id("some_random_folder") == ("", "")
    assert sm.parse_session_id("") == ("", "")


# --------------------------------------------------------------------------- #
# Reading it
# --------------------------------------------------------------------------- #
def test_reads_scene_and_field_from_session_json(tmp_path):
    _session(tmp_path, "vid3_20260108_132749", scene_hint="mixed",
             field="field_A", trip="Visit1", site="vidalia_1")
    m = sm.read_session_meta(tmp_path, "vid3_20260108_132749")
    assert m["scene"] == "mixed"
    assert m["field_id"] == "field_A"
    assert m["camera"] == "vid3" and m["date"] == "2026-01-08"
    assert m["has_meta"] is True


def test_a_session_without_metadata_is_not_an_error(tmp_path):
    """Older extractions predate session.json. A build must not fail over one
    old recording - it must only avoid inventing relatedness."""
    _session(tmp_path, "vid1_20250221_131902")
    m = sm.read_session_meta(tmp_path, "vid1_20250221_131902")
    assert m["has_meta"] is False and m["scene"] == "unknown"
    assert m["camera"] == "vid1"           # still knowable from the id


def test_corrupt_json_degrades_instead_of_raising(tmp_path):
    d = tmp_path / "vid1_20250221_131902" / "meta"
    d.mkdir(parents=True)
    (d / "session.json").write_text("{not json", encoding="utf-8")
    assert sm.read_session_meta(tmp_path, "vid1_20250221_131902")["scene"] \
        == "unknown"


def test_the_date_is_the_capture_date_not_the_extraction_date(tmp_path):
    """session.json carries extracted_utc, which is when somebody ran the
    extractor - it can be months after the capture and says nothing about
    which drives are near-duplicates of each other."""
    _session(tmp_path, "vid3_20260108_132749", scene_hint="mixed",
             extracted_utc="2026-08-13T00:00:00+00:00")
    assert sm.read_session_meta(tmp_path, "vid3_20260108_132749")["date"] \
        == "2026-01-08"


def test_several_images_roots_are_searched(tmp_path):
    """A build can merge CVAT exports whose images live under different
    sessions roots."""
    a, b = tmp_path / "a", tmp_path / "b"
    _session(b, "vid3_20260108_132749", scene_hint="weed_only")
    m = sm.find_session_meta([a, b], "vid3_20260108_132749")
    assert m["scene"] == "weed_only"


def test_missing_everywhere_still_returns_id_derived_fields(tmp_path):
    m = sm.find_session_meta([tmp_path], "vid3_20260108_132749")
    assert m["scene"] == "unknown" and m["camera"] == "vid3"


def test_the_gaps_are_reportable(tmp_path):
    """An unknown scene does not break a split, it silently removes that
    session from stratification - so it has to be printable."""
    _session(tmp_path, "vid1_20260108_101500", scene_hint="mixed")
    _session(tmp_path, "vid2_20260108_101500", scene_hint="bananas")
    _session(tmp_path, "vid3_20260108_101500")
    metas = [sm.read_session_meta(tmp_path, s) for s in
             ("vid1_20260108_101500", "vid2_20260108_101500",
              "vid3_20260108_101500")]
    rep = sm.unknown_scene_report(metas)
    assert rep["unknown_scene"] == ["vid2_20260108_101500",
                                    "vid3_20260108_101500"]
    assert rep["no_session_json"] == ["vid3_20260108_101500"]
