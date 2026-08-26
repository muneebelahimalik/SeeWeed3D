"""One output folder per run.

Both weed runners used to derive their output folder from ROUND and the session
name alone, so running either one twice overwrote the first run's output. For
overlays that is annoying; for a CVAT batch someone has spent an afternoon
correcting, or for the predictions a set of pseudo-labels was scored from, it
is the loss of the work itself.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from common import run_dirs as rd  # noqa: E402

WHEN = datetime(2026, 8, 26, 11, 16)


# --------------------------------------------------------------------------- #
# stamped
# --------------------------------------------------------------------------- #
def test_the_name_says_when_it_ran(tmp_path):
    assert Path(rd.stamped(tmp_path, "look_vid3", WHEN)).name == \
        "look_vid3_20260826_1116"


def test_two_runs_a_minute_apart_do_not_collide(tmp_path):
    a = rd.stamped(tmp_path, "look_vid3", WHEN)
    b = rd.stamped(tmp_path, "look_vid3", WHEN + timedelta(minutes=1))
    assert a != b


def test_a_second_run_in_the_same_minute_still_gets_its_own_folder(tmp_path):
    """A GPU pass cannot finish twice in a minute, but a crash-and-retry can
    start twice in one, and so can a test."""
    a = rd.stamped(tmp_path, "look_vid3", WHEN)
    Path(a).mkdir()
    b = rd.stamped(tmp_path, "look_vid3", WHEN)
    assert b != a and b.endswith("_2")


def test_the_collision_tail_keeps_counting(tmp_path):
    for _ in range(3):
        Path(rd.stamped(tmp_path, "look_vid3", WHEN)).mkdir()
    assert len(list(tmp_path.iterdir())) == 3


def test_stamped_never_returns_an_existing_path(tmp_path):
    """THE POINT. Anything else and the next run lands on the last one."""
    made = []
    for _ in range(5):
        p = rd.stamped(tmp_path, "selftrain_vid3", WHEN)
        assert not Path(p).exists()
        Path(p).mkdir()
        made.append(p)
    assert len(set(made)) == 5


def test_stamped_does_not_create_the_folder(tmp_path):
    """The caller decides when - predict_images makes its own output tree, and
    a folder created here would defeat the collision check on the next call."""
    p = rd.stamped(tmp_path, "look_vid3", WHEN)
    assert not Path(p).exists()


def test_different_sessions_do_not_share_folders(tmp_path):
    a = rd.stamped(tmp_path, "look_vid3", WHEN)
    b = rd.stamped(tmp_path, "look_vid7", WHEN)
    assert a != b


# --------------------------------------------------------------------------- #
# newest
# --------------------------------------------------------------------------- #
def test_newest_finds_the_most_recent_run(tmp_path):
    for m in (10, 40, 25):
        Path(rd.stamped(tmp_path, "look_vid3",
                        WHEN.replace(minute=m))).mkdir()
    assert Path(rd.newest(tmp_path, "look_vid3")).name.endswith("_1140")


def test_newest_sorts_by_time_not_creation_order(tmp_path):
    """Names sort chronologically, which is why the stamp is zero-padded and
    big-endian. Creating them out of order must not change the answer."""
    Path(rd.stamped(tmp_path, "look_vid3", WHEN.replace(day=9))).mkdir()
    Path(rd.stamped(tmp_path, "look_vid3", WHEN.replace(day=10))).mkdir()
    Path(rd.stamped(tmp_path, "look_vid3", WHEN.replace(day=2))).mkdir()
    assert "20260810" in rd.newest(tmp_path, "look_vid3")


def test_newest_prefers_the_collision_tail_within_a_minute(tmp_path):
    Path(rd.stamped(tmp_path, "look_vid3", WHEN)).mkdir()
    second = rd.stamped(tmp_path, "look_vid3", WHEN)
    Path(second).mkdir()
    assert rd.newest(tmp_path, "look_vid3") == second


def test_newest_is_none_when_nothing_has_run(tmp_path):
    assert rd.newest(tmp_path, "look_vid3") is None


def test_newest_is_none_for_a_missing_parent(tmp_path):
    """A round that has never been trained has no folder at all, and that is
    an ordinary state, not an error."""
    assert rd.newest(tmp_path / "weeds_r9", "look_vid3") is None


def test_newest_ignores_a_different_session(tmp_path):
    Path(rd.stamped(tmp_path, "look_vid7", WHEN)).mkdir()
    assert rd.newest(tmp_path, "look_vid3") is None


def test_newest_ignores_an_unstamped_legacy_folder(tmp_path):
    """Folders from before this existed are named `look_<session>` flat. They
    are real output and must not be picked as 'the newest run' - their date is
    unknown, which is the whole reason stamping was added."""
    (tmp_path / "look_vid3").mkdir()
    assert rd.newest(tmp_path, "look_vid3") is None


def test_newest_ignores_a_file(tmp_path):
    (tmp_path / "look_vid3_20260826_1116").write_text("not a folder")
    assert rd.newest(tmp_path, "look_vid3") is None


def test_a_session_name_that_prefixes_another_is_not_matched(tmp_path):
    """`look_vid3` must not match `look_vid30_...`."""
    Path(rd.stamped(tmp_path, "look_vid30", WHEN)).mkdir()
    assert rd.newest(tmp_path, "look_vid3") is None


def test_is_stamped_tells_the_two_generations_apart(tmp_path):
    assert rd.is_stamped("look_vid3_20260826_1116")
    assert rd.is_stamped("look_vid3_20260826_1116_2")
    assert not rd.is_stamped("look_vid3")
    assert not rd.is_stamped("selftrain_vid3_20260108_110444")


# --------------------------------------------------------------------------- #
# reusing predictions safely
# --------------------------------------------------------------------------- #
def _preds(tmp_path, mtime):
    d = tmp_path / "preds"
    d.mkdir(exist_ok=True)
    f = d / "instances_default.json"
    f.write_text("{}")
    import os
    os.utime(f, (mtime, mtime))
    return d


def _ckpt(tmp_path, mtime):
    f = tmp_path / "checkpoint_best_total.pth"
    f.write_bytes(b"x")
    import os
    os.utime(f, (mtime, mtime))
    return f


def test_predictions_older_than_the_model_are_called_out(tmp_path):
    """The failure this exists for: retrain a round, re-run the scorer, and it
    silently reuses the OLD model's predictions - then writes them into the
    training set as pseudo-labels, teaching the model to agree with a version
    of itself it has already improved on."""
    p = _preds(tmp_path, 1_000)
    c = _ckpt(tmp_path, 2_000)
    warn = rd.stale_predictions_warning(p, c)
    assert warn and "OLDER than the checkpoint" in warn


def test_the_warning_names_both_dates_and_both_paths(tmp_path):
    """A warning you cannot act on is noise."""
    p = _preds(tmp_path, 1_000)
    c = _ckpt(tmp_path, 2_000)
    warn = rd.stale_predictions_warning(p, c)
    assert str(p) in warn and str(c) in warn
    assert warn.count("2026") + warn.count("1970") >= 2


def test_predictions_newer_than_the_model_are_fine(tmp_path):
    p = _preds(tmp_path, 3_000)
    c = _ckpt(tmp_path, 2_000)
    assert rd.stale_predictions_warning(p, c) is None


def test_predictions_written_by_this_very_checkpoint_are_fine(tmp_path):
    """Equal mtimes happen when both land in the same second; that is the
    normal case for a fresh pass, not a stale one."""
    p = _preds(tmp_path, 2_000)
    c = _ckpt(tmp_path, 2_000)
    assert rd.stale_predictions_warning(p, c) is None


def test_no_predictions_yet_is_not_a_staleness_problem(tmp_path):
    c = _ckpt(tmp_path, 2_000)
    assert rd.stale_predictions_warning(tmp_path / "nothing", c) is None


def test_a_missing_checkpoint_is_left_to_the_caller(tmp_path):
    """weeds_selftrain already fails with a message pointing at the trainer;
    a second, vaguer complaint from here would just get in the way."""
    p = _preds(tmp_path, 1_000)
    assert rd.stale_predictions_warning(p, tmp_path / "nope.pth") is None


# --------------------------------------------------------------------------- #
# the runners actually use it
# --------------------------------------------------------------------------- #
def test_the_look_runner_stamps_its_output():
    import inspect
    from training.datasets import weeds_look
    src = inspect.getsource(weeds_look)
    assert "stamped(" in src, "two looks at one session must both survive"
    assert rd.is_stamped(Path(weeds_look.CONFIG["OUT_DIR"]).name)


def test_the_selftrain_runner_stamps_its_batches():
    from training.datasets import weeds_selftrain as st
    assert rd.is_stamped(Path(st.OUT_DIR).name), \
        "a half-corrected CVAT batch must never be overwritten by a re-run"


def test_selftrain_still_reuses_predictions_rather_than_stamping_them():
    """Predictions are the ONE thing that should be reused - re-scoring at a
    different ACCEPT threshold must not cost another GPU pass. If this starts
    stamping too, every re-score silently reruns inference."""
    import inspect
    from training.datasets import weeds_selftrain as st
    assert "newest(" in inspect.getsource(st) or st.PREDICTIONS == ""


def test_selftrain_warns_before_scoring_reused_predictions():
    import inspect
    from training.datasets import weeds_selftrain as st
    assert "stale_predictions_warning" in inspect.getsource(st._predict)
