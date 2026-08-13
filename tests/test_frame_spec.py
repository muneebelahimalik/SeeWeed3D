"""Naming frames by index, in one syntax across every stage.

A single drive can pass from one crop zone into another - weeds only for the
first stretch, then onions only. That is not a mixed scene, it is two
single-class scenes end to end, and each half wants the prelabeler whose
assumption actually holds there. Selecting that half needs a frame spec, and
curation already had one, so the syntax lives in one place rather than being
respelled per stage.
"""
import pytest

from conftest import load_script

fs = load_script("common/frame_spec.py")


def _names(idxs, sid="vid1_20250221_131902", ext=".png"):
    return [f"{sid}_{i:06d}{ext}" for i in idxs]


# --------------------------------------------------------------------------- #
# Token forms
# --------------------------------------------------------------------------- #
def test_a_bare_index_selects_one_frame():
    got = fs.select_filenames(_names([0, 5, 9]), ["5"])
    assert got == _names([5])


def test_a_range_is_inclusive_at_both_ends():
    """Off-by-one here silently drops a frame at a zone boundary, which is
    exactly where the wrong class assumption is most expensive."""
    got = fs.select_filenames(_names(range(10)), ["2-4"])
    assert got == _names([2, 3, 4])


def test_a_reversed_range_still_works():
    assert fs.select_filenames(_names(range(10)), ["4-2"]) == _names([2, 3, 4])


def test_a_source_filename_names_its_own_frame():
    got = fs.select_filenames(_names([1, 7]), ["vid1_20250221_131902_000007.png"])
    assert got == _names([7])


def test_a_preview_jpg_name_works_too():
    """Previews are what you actually look at when finding the transition, and
    a preview is a .jpg beside a .png source. Requiring the source name would
    mean transcribing indices by hand from the thing on screen."""
    got = fs.select_filenames(_names([1, 7]), ["vid1_20250221_131902_000007.jpg"])
    assert got == _names([7])


def test_an_open_ended_range_runs_to_the_end_of_the_session():
    """The split-zone case is 'onions from here on'. Requiring a closing index
    would mean looking up the session's last frame first, and guessing it low
    silently truncates the tail of the run."""
    got = fs.select_filenames(_names(range(10)), ["6-"])
    assert got == _names([6, 7, 8, 9])


def test_an_open_start_runs_from_the_beginning():
    assert fs.select_filenames(_names(range(10)), ["-3"]) == _names([0, 1, 2, 3])


def test_an_open_end_does_not_read_as_zero():
    """'1500-' must not parse as the range 0-1500 - that selects precisely the
    frames it was meant to exclude."""
    _, ranges = fs.parse_frame_tokens(["1500-"])
    assert ranges == [(1500, float("inf"))]


def test_a_bare_dash_is_a_typo_not_a_wildcard(capsys):
    fs.select_filenames(_names(range(5)), ["-"])
    assert "cannot interpret" in capsys.readouterr().out


def test_tokens_combine():
    got = fs.select_filenames(_names(range(20)), ["0-2", "9", "15-16"])
    assert got == _names([0, 1, 2, 9, 15, 16])


# --------------------------------------------------------------------------- #
# The inert / empty cases
# --------------------------------------------------------------------------- #
def test_an_empty_spec_keeps_everything():
    """The option must be inert until somebody sets it, so a stage that never
    uses it behaves exactly as it did before."""
    names = _names(range(5))
    assert fs.select_filenames(names, []) == names
    assert fs.select_filenames(names, None) == names


def test_a_spec_that_matches_nothing_returns_nothing():
    """Not a silent full pass. Selecting no frames is a real answer, and the
    caller reports it with the session id attached."""
    assert fs.select_filenames(_names(range(5)), ["900-999"]) == []


def test_order_is_preserved():
    got = fs.select_filenames(_names(range(10)), ["7", "1", "4"])
    assert got == _names([1, 4, 7])


def test_an_unreadable_token_is_reported_not_silently_ignored(capsys):
    """A typo that selects nothing looks exactly like a range that
    legitimately matched nothing."""
    fs.select_filenames(_names(range(5)), ["not-a-frame"])
    assert "cannot interpret" in capsys.readouterr().out


def test_an_unreadable_token_does_not_poison_the_good_ones(capsys):
    got = fs.select_filenames(_names(range(5)), ["oops", "2"])
    assert got == _names([2])


def test_a_file_with_no_index_is_skipped_rather_than_crashing():
    assert fs.index_of("noindex.png") == -1
    assert fs.select_filenames(["noindex.png"] + _names([3]), ["3"]) == _names([3])


# --------------------------------------------------------------------------- #
# One syntax, not two
# --------------------------------------------------------------------------- #
def test_curate_pool_uses_the_shared_parser():
    """Curation's MANUAL_DROPS and a prelabeler's ONLY_FRAMES must accept the
    same tokens - two spellings of '0-250' would be one too many."""
    cp = load_script("extraction/curate_pool.py")
    assert cp.parse_drop_tokens(["0-3", "9"]) == fs.parse_frame_tokens(["0-3", "9"])


@pytest.mark.parametrize("script", [
    "annotation/prelabel_weeds_sam3.py",
    "annotation/prelabel_onions_sam3.py",
    "annotation/prelabel_mixed_sam3.py",
])
def test_every_prelabeler_offers_the_option(script):
    """A split-zone drive needs whichever prelabeler matches each stretch, so
    all three have to be targetable - a fix in one of them is not a fix."""
    mod = load_script(script)
    assert "ONLY_FRAMES" in mod.CONFIG
    assert mod.CONFIG["ONLY_FRAMES"] == {}       # inert by default


@pytest.mark.parametrize("script", [
    "annotation/prelabel_weeds_sam3.py",
    "annotation/prelabel_onions_sam3.py",
    "annotation/prelabel_mixed_sam3.py",
])
def test_the_range_is_applied_before_the_limit(script):
    """LIMIT_PER_SESSION must trial the SELECTED stretch. Applied the other way
    round, a 20-frame trial of the onion half of a split drive would silently
    return 20 frames of the weed half."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "seeweed3d" / script
           ).read_text(encoding="utf-8")
    assert src.index("ONLY_FRAMES") < src.index('cfg["LIMIT_PER_SESSION"]')
