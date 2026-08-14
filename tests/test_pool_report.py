"""What the pool held, and what curation removed.

A prelabeler reads the CURATED pool - dropped rows in meta/pool.csv are skipped,
which is the point of curation - but nothing ever said how much that removed. A
754-frame session curated down to 15 then prelabels 15 frames and reports
success, and the only symptom is a number that looks small if you happen to
remember what the session held.
"""
import csv

from conftest import load_script

wd = load_script("annotation/prelabel_weeds_sam3.py")


def _pool(tmp_path, total, kept, reason="redundant with the previous frame"):
    d = tmp_path / "sess" / "meta"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"filename": f"s_{i:06d}.png",
             "dropped": "0" if i < kept else "1",
             "drop_reason": "" if i < kept else reason}
            for i in range(total)]
    with open(d / "pool.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "dropped", "drop_reason"])
        w.writeheader()
        w.writerows(rows)
    return d.parent


def test_it_reports_the_pool_and_what_curation_took(tmp_path):
    kept, total, reasons = wd.pool_report(_pool(tmp_path, 754, 15))
    assert (kept, total) == (15, 754)
    assert sum(reasons.values()) == 739


def test_heavy_curation_is_called_out(capsys, tmp_path):
    """754 -> 15 is the case that actually happened, from a stale
    MIN_SHIFT_FRAC. It looked like a successful run."""
    wd.print_pool_report("sess", _pool(tmp_path, 754, 15), 15)
    out = capsys.readouterr().out
    assert "754 frame(s)" in out and "739 dropped by curation" in out
    assert "98% of this session" in out and "RESTORE_ALL" in out


def test_light_curation_is_reported_without_an_alarm(capsys, tmp_path):
    """Dropping a few redundant frames is what curation is FOR."""
    wd.print_pool_report("sess", _pool(tmp_path, 100, 85), 85)
    out = capsys.readouterr().out
    assert "15 dropped by curation" in out
    assert "[!]" not in out


def test_an_uncurated_pool_says_nothing_about_dropping(capsys, tmp_path):
    wd.print_pool_report("sess", _pool(tmp_path, 40, 40), 40)
    out = capsys.readouterr().out
    assert "40 frame(s)" in out and "dropped" not in out


def test_a_narrowed_run_is_separable_from_curation(capsys, tmp_path):
    """Three reasons a frame can be absent - curated out, outside ONLY_FRAMES,
    past LIMIT_PER_SESSION - and they need telling apart at a glance."""
    wd.print_pool_report("sess", _pool(tmp_path, 100, 80), 20)
    out = capsys.readouterr().out
    assert "20 dropped by curation" in out and "20 selected by this run" in out


def test_the_drop_reasons_are_named(capsys, tmp_path):
    """'Why did I lose 700 frames' is answered by the reason, not the count."""
    wd.print_pool_report("sess", _pool(tmp_path, 100, 10, reason="too blurred"),
                         10)
    assert "too blurred=90" in capsys.readouterr().out


def test_a_missing_pool_is_silent_not_a_crash(tmp_path):
    assert wd.pool_report(tmp_path / "nothing") == (0, 0, {})


def test_a_pool_predating_curation_reads_as_keep_everything(tmp_path):
    """A missing `dropped` column must not read as 'everything dropped'."""
    d = tmp_path / "sess" / "meta"
    d.mkdir(parents=True)
    with open(d / "pool.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename"])
        w.writeheader()
        w.writerows([{"filename": f"s_{i:06d}.png"} for i in range(12)])
    assert wd.pool_report(d.parent)[:2] == (12, 12)


def test_every_prelabeler_reports_its_pool():
    """A fix in one of them is not a fix - all three read the curated pool."""
    from pathlib import Path
    for s in ("prelabel_weeds_sam3.py", "prelabel_onions_sam3.py",
              "prelabel_mixed_sam3.py"):
        src = (Path(__file__).resolve().parents[1] / "seeweed3d" / "annotation"
               / s).read_text(encoding="utf-8")
        assert "print_pool_report(sid, session_dir, len(frames))" in src, s
        # after the narrowing, so the printed count is what actually ran.
        # Matched on the full call - the helper's DEFINITION sits earlier in
        # the file and a prefix match would find that instead.
        assert src.index('cfg["LIMIT_PER_SESSION"]') < \
            src.index("print_pool_report(sid, session_dir, len(frames))")
