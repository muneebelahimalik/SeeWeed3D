"""Telling a recordings folder from an extracted dataset.

Two trees look alike from a config block and are nothing alike on disk, and the
config keys do not help: extraction calls them INPUT_ROOTS and OUTPUT_ROOT,
every stage after it calls the second one DATASET_ROOT. Pasting the recordings
path into the next script is the natural mistake.

The old message for it - "ERROR: <path>\\sessions not found. Run
extract_sessions.py first." - is exactly wrong advice when extraction HAS been
run and its output is in another folder. It sends you to re-run a job that
already succeeded and says nothing about the path you typed.
"""
from pathlib import Path

import pytest

from conftest import load_script

dp = load_script("common/dataset_paths.py")


def _recordings(tmp_path, n=3):
    for i in range(n):
        d = tmp_path / f"Session_2025022{i}_130957"
        d.mkdir(parents=True)
        (d / "RGB_video.avi").write_bytes(b"x")
        (d / "Depth_video.avi").write_bytes(b"x")
        (d / "calibration_params.txt").write_text("fx: 700")
    return tmp_path


def _dataset(tmp_path, n=2):
    for i in range(n):
        s = tmp_path / "sessions" / f"Visit1_2025022{i}_130957"
        (s / "rgb").mkdir(parents=True)
        (s / "meta").mkdir(parents=True)
    (tmp_path / "registry.csv").write_text("session_id\n")
    return tmp_path


def test_a_real_dataset_root_resolves():
    pass


def test_an_extracted_dataset_is_accepted(tmp_path):
    root = _dataset(tmp_path)
    assert dp.sessions_root_problem(root) is None
    assert dp.require_sessions_root(root) == root / "sessions"


def test_a_recordings_folder_is_named_as_the_mistake(tmp_path):
    """The whole point. This is the path that used to be told to re-run an
    extraction that had already succeeded."""
    msg = dp.sessions_root_problem(_recordings(tmp_path))
    assert "RECORDINGS folder" in msg
    assert "INPUT_ROOTS" in msg and "OUTPUT_ROOT" in msg
    assert "Run extract_sessions.py first" not in msg


def test_the_recordings_message_shows_which_folders_it_saw(tmp_path):
    """Naming them is what makes the diagnosis checkable rather than a guess."""
    msg = dp.sessions_root_problem(_recordings(tmp_path, n=2))
    assert "Session_20250220_130957" in msg


def test_many_recording_folders_are_summarised_not_listed(tmp_path):
    msg = dp.sessions_root_problem(_recordings(tmp_path, n=9))
    assert "+6 more" in msg


def test_recordings_are_detected_by_structure_not_by_name(tmp_path):
    """A renamed session is still a recording. Matching on 'Session_' would
    miss it and fall through to the generic message."""
    d = tmp_path / "run_one_first_pass"
    d.mkdir(parents=True)
    (d / "RGB_video.mkv").write_bytes(b"x")
    assert "RECORDINGS folder" in dp.sessions_root_problem(tmp_path)


def test_an_extracted_session_is_not_mistaken_for_a_recording(tmp_path):
    """An extracted session has rgb/. Without that check, a dataset whose
    sessions/ was deleted would be diagnosed as a recordings folder."""
    s = tmp_path / "Visit1_20250221_130957"
    (s / "rgb").mkdir(parents=True)
    (s / "rgb" / "f.png").write_bytes(b"x")
    assert dp.looks_like_recordings(tmp_path) == []


def test_a_missing_path_says_so_rather_than_blaming_extraction(tmp_path):
    msg = dp.sessions_root_problem(tmp_path / "typo")
    assert "does not exist" in msg
    assert "OUTPUT_ROOT" in msg


def test_an_empty_folder_gets_the_generic_advice(tmp_path):
    """Exists, holds nothing recognisable - here 'nothing has been extracted'
    really is the right reading."""
    msg = dp.sessions_root_problem(tmp_path)
    assert "no sessions/ folder" in msg
    assert "extract_sessions.py" in msg


def test_require_raises_rather_than_returning_a_bad_path(tmp_path):
    with pytest.raises(SystemExit, match="RECORDINGS folder"):
        dp.require_sessions_root(_recordings(tmp_path))


def test_svo_only_folders_still_read_as_recordings(tmp_path):
    d = tmp_path / "Session_20250221_130957"
    d.mkdir(parents=True)
    (d / "recording.svo2").write_bytes(b"x")
    assert "RECORDINGS folder" in dp.sessions_root_problem(tmp_path)


# --------------------------------------------------------------------------- #
# Every stage that takes a DATASET_ROOT uses it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", [
    "extraction/curate_pool.py",
    "annotation/prelabel_mixed_sam3.py",
    "annotation/prelabel_weeds_sam3.py",
    "annotation/prelabel_onions_sam3.py",
])
def test_the_stage_uses_the_shared_check(script):
    """Four scripts shipped the same misleading message. A fix in one of them
    is not a fix."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "seeweed3d" / script
           ).read_text(encoding="utf-8")
    assert "require_sessions_root" in src
    assert "not found. Run extract_sessions.py first." not in src


# --------------------------------------------------------------------------- #
# A root may BE a session folder, not only the folder of sessions
# --------------------------------------------------------------------------- #
def _session_tree(tmp_path, sid, n=3):
    d = tmp_path / "sessions" / sid
    for sub in ("rgb", "depth"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / "rgb" / f"{sid}_{i:06d}.png").write_bytes(b"")
        (d / "depth" / f"{sid}_{i:06d}.png").write_bytes(b"")
    return d


def test_a_session_folder_works_as_an_images_root(tmp_path):
    """Naming one root per session is a natural way to describe a merged
    build, and it must not cost a recursive scan per image."""
    sd = load_script("training/seg_dataset.py")
    s = _session_tree(tmp_path, "Visit1_20260108_132749")
    got = sd.resolve_image("Visit1_20260108_132749_000001.png", [s],
                           session_id="Visit1_20260108_132749")
    assert Path(got).parent.name == "rgb"
    assert Path(got).name == "Visit1_20260108_132749_000001.png"


def test_the_sessions_root_form_still_works(tmp_path):
    sd = load_script("training/seg_dataset.py")
    _session_tree(tmp_path, "Visit1_20260108_132749")
    got = sd.resolve_image("Visit1_20260108_132749_000001.png",
                           [tmp_path / "sessions"],
                           session_id="Visit1_20260108_132749")
    assert Path(got).parent.name == "rgb"


def test_one_root_per_session_resolves_every_session(tmp_path):
    """The real merged case: several roots, each a session folder."""
    sd = load_script("training/seg_dataset.py")
    a = _session_tree(tmp_path, "Visit1_20260108_132749")
    b = _session_tree(tmp_path, "Visit2_20260210_164149")
    for sid in ("Visit1_20260108_132749", "Visit2_20260210_164149"):
        got = sd.resolve_image(f"{sid}_000002.png", [a, b], session_id=sid)
        assert Path(got).name == f"{sid}_000002.png"
        assert Path(got).parent.parent.name == sid, "resolved from the wrong session"


def test_a_session_folder_root_prefers_rgb_over_depth(tmp_path):
    """depth/ holds a file of the SAME name; picking it would train on depth."""
    sd = load_script("training/seg_dataset.py")
    s = _session_tree(tmp_path, "Visit1_20260108_132749")
    got = sd.resolve_image("Visit1_20260108_132749_000000.png", [s],
                           session_id="Visit1_20260108_132749")
    assert Path(got).parent.name == "rgb"


def test_a_renamed_session_folder_still_resolves(tmp_path):
    """The folder can be renamed after extraction while the frames inside keep
    their original prefix - session id comes from the FILENAME, so
    <root>/<session_id>/rgb/ cannot match and only the session-folder form can.
    This is the layout the onion build actually has: a folder called
    Visit1_20260108_132749 holding frames named vid3_20260108_132749_*.png."""
    sd = load_script("training/seg_dataset.py")
    folder = tmp_path / "sessions" / "Visit1_20260108_132749"
    (folder / "rgb").mkdir(parents=True)
    for i in range(3):
        (folder / "rgb" / f"vid3_20260108_132749_{i:06d}.png").write_bytes(b"")

    got = sd.resolve_image("vid3_20260108_132749_000001.png", [folder],
                           session_id="vid3_20260108_132749")
    assert Path(got).name == "vid3_20260108_132749_000001.png"
    assert Path(got).parent.name == "rgb"
