"""The active-learning ledger: what went out, and what came back.

Two failures appear only once mining is run REPEATEDLY, and neither is visible
to a single pass: frames still sitting in an unfinished CVAT task get selected
again, and nobody ever measures whether a round taught the model anything.
"""
import json

import pytest

from conftest import load_script

al = load_script("training/al_rounds.py")


def _export(d, ids, **kw):
    return al.record_export(d, item_ids=ids, checkpoint="run4/best.pt",
                            conf=0.2, batch_size=len(ids), out_dir=d / "batch",
                            sessions_root=d / "sessions", **kw)


# --------------------------------------------------------------------------- #
# Re-sending frames that are already out
# --------------------------------------------------------------------------- #
def test_exported_frames_are_in_flight(tmp_path):
    """Not annotated yet, so the dataset does not exclude them, and the model
    scores them exactly as it did last round."""
    _export(tmp_path, ["f1", "f2", "f3"])
    assert al.frames_in_flight(al.load_ledger(tmp_path)) == {"f1", "f2", "f3"}


def test_merging_releases_them(tmp_path):
    """Once merged, the dataset's own item ids exclude them - keeping them in
    flight as well would be belt and braces on a fact already established."""
    _export(tmp_path, ["f1", "f2"])
    al.mark_merged(tmp_path)
    assert al.frames_in_flight(al.load_ledger(tmp_path)) == set()


def test_abandoning_releases_them_deliberately(tmp_path):
    """The escape hatch for a CVAT task that will never be finished. Recorded,
    so those frames reappearing in a later batch is explicable."""
    r = _export(tmp_path, ["f1", "f2"])
    al.abandon(tmp_path, r["round"], reason="task never finished")
    assert al.frames_in_flight(al.load_ledger(tmp_path)) == set()
    assert "never finished" in al.load_ledger(tmp_path)["rounds"][0]["notes"]


def test_two_rounds_in_flight_at_once(tmp_path):
    _export(tmp_path, ["a"])
    _export(tmp_path, ["b"])
    assert al.frames_in_flight(al.load_ledger(tmp_path)) == {"a", "b"}


def test_round_numbers_increment(tmp_path):
    assert _export(tmp_path, ["a"])["round"] == 1
    assert _export(tmp_path, ["b"])["round"] == 2


def test_merge_targets_the_newest_exported_round(tmp_path):
    """A linear annotate-merge-retrain loop always means the newest one."""
    _export(tmp_path, ["a"])
    _export(tmp_path, ["b"])
    assert al.mark_merged(tmp_path)["round"] == 2
    assert al.frames_in_flight(al.load_ledger(tmp_path)) == {"a"}


def test_merging_with_nothing_out_is_an_error_not_a_silent_noop(tmp_path):
    _export(tmp_path, ["a"])
    al.mark_merged(tmp_path)
    with pytest.raises(ValueError, match="no exported round"):
        al.mark_merged(tmp_path)


# --------------------------------------------------------------------------- #
# Did the round buy anything
# --------------------------------------------------------------------------- #
def test_metric_deltas_are_computed(tmp_path):
    r = _export(tmp_path, ["a", "b"])
    al.mark_merged(tmp_path)
    al.attach_metrics(tmp_path, r["round"],
                      before={"mAP": 0.41, "weed_recall": 0.62},
                      after={"mAP": 0.47, "weed_recall": 0.71})
    row = al.history(tmp_path)[0]
    assert row["deltas"]["mAP"] == pytest.approx(0.06)
    assert row["deltas"]["weed_recall"] == pytest.approx(0.09)


def test_a_round_that_moved_nothing_is_still_reported(tmp_path):
    """Information, not a failure - it means the bottleneck is somewhere the
    frame selection cannot reach."""
    r = _export(tmp_path, ["a"])
    al.attach_metrics(tmp_path, r["round"], before={"mAP": 0.5},
                      after={"mAP": 0.5})
    assert al.history(tmp_path)[0]["deltas"]["mAP"] == 0.0


def test_a_metric_present_on_only_one_side_is_skipped(tmp_path):
    """Half a comparison is not a delta, and printing one would invite reading
    an absolute number as a change."""
    r = _export(tmp_path, ["a"])
    al.attach_metrics(tmp_path, r["round"], before={"mAP": 0.5},
                      after={"mAP": 0.6, "mAP50": 0.8})
    assert set(al.history(tmp_path)[0]["deltas"]) == {"mAP"}


def test_history_without_metrics_has_empty_deltas(tmp_path):
    _export(tmp_path, ["a"])
    assert al.history(tmp_path)[0]["deltas"] == {}


def test_format_history_is_readable(tmp_path):
    _export(tmp_path, ["a", "b"])
    out = al.format_history(al.history(tmp_path))
    assert "exported" in out and "2" in out


def test_format_history_says_so_when_empty(tmp_path):
    assert "no active-learning rounds" in al.format_history(al.history(tmp_path))


# --------------------------------------------------------------------------- #
# The holdout, recorded per round
# --------------------------------------------------------------------------- #
def test_the_holdout_is_recorded_with_each_round(tmp_path):
    """A holdout quietly redefined between rounds turns a test score into a
    training score, and the diff between two rounds is the only place that
    becomes visible."""
    _export(tmp_path, ["a"], holdout_sessions=["vid9_20260101_100000"])
    _export(tmp_path, ["b"], holdout_sessions=[])
    holdouts = [r["holdout_sessions"]
                for r in al.load_ledger(tmp_path)["rounds"]]
    assert holdouts == [["vid9_20260101_100000"], []]


# --------------------------------------------------------------------------- #
# Robustness - a ledger is written beside a dataset people delete and rebuild
# --------------------------------------------------------------------------- #
def test_a_missing_ledger_is_the_normal_starting_state(tmp_path):
    assert al.load_ledger(tmp_path) == {"rounds": []}


def test_a_corrupt_ledger_does_not_take_the_run_down(tmp_path):
    al.ledger_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert al.load_ledger(tmp_path) == {"rounds": []}


def test_the_ledger_is_plain_readable_json(tmp_path):
    _export(tmp_path, ["a"])
    raw = json.loads(al.ledger_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["rounds"][0]["state"] == "exported"


def test_an_unknown_round_number_is_refused(tmp_path):
    _export(tmp_path, ["a"])
    for fn in (lambda: al.mark_merged(tmp_path, 99),
               lambda: al.abandon(tmp_path, 99, ""),
               lambda: al.attach_metrics(tmp_path, 99, before={"mAP": 1})):
        with pytest.raises(ValueError):
            fn()


# --------------------------------------------------------------------------- #
# mine_pool wiring
# --------------------------------------------------------------------------- #
def test_mine_pool_offers_the_in_flight_switch():
    mp = load_script("annotation/mine_pool.py")
    assert isinstance(mp.CONFIG["SKIP_FRAMES_IN_FLIGHT"], bool)
    assert isinstance(mp.CONFIG["RECORD_ROUND"], bool)


def test_mine_pool_excludes_in_flight_frames_from_the_pool(tmp_path):
    """pool_frames takes the union of annotated and in-flight ids."""
    mp = load_script("annotation/mine_pool.py")
    sess = tmp_path / "vid1_20260101_100000" / "rgb"
    sess.mkdir(parents=True)
    for i in (1, 2, 3):
        (sess / f"vid1_20260101_100000_00000{i}.png").write_bytes(b"x")

    all_ids = {f[1].stem for f in mp.pool_frames(tmp_path)}
    assert len(all_ids) == 3
    kept = mp.pool_frames(tmp_path,
                          exclude_ids={"vid1_20260101_100000_000002"})
    assert len(kept) == 2
