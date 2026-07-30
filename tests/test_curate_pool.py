"""Checks for pool curation: dropping redundant / bad frames must never touch
image files, must be reversible, and must actually take effect downstream."""
import csv

import cv2
import numpy as np

from conftest import load_script

cp = load_script("extraction/curate_pool.py")
wd = load_script("annotation/prelabel_weeds_sam3.py")
sb = load_script("extraction/select_batches.py")


def _session(tmp_path, n=10, with_pose=True, travel_mm=5.0, sid="sess"):
    """Session whose frames advance by a fixed pose step, so the expected
    keep/drop pattern is exactly predictable."""
    sdir = tmp_path / "sessions" / sid
    (sdir / "rgb").mkdir(parents=True)
    (sdir / "meta").mkdir(parents=True)
    rows = []
    for i in range(n):
        fn = f"{sid}_{i:06d}.png"
        img = np.full((60, 80, 3), 40, np.uint8)
        cv2.imwrite(str(sdir / "rgb" / fn), img)
        row = {"video_frame_idx": str(i), "filename": fn, "session_id": sid,
               "sharpness": "100", "veg_frac": "0.2"}
        if with_pose:
            row |= {"pose_state": "OK", "tx_mm": f"{i * travel_mm}",
                    "ty_mm": "0", "tz_mm": "0"}
        rows.append(row)
    with open(sdir / "meta" / "pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return sdir


def _cfg(**over):
    base = dict(cp.CONFIG)
    base.update(DRY_RUN=False, MANUAL_DROPS={}, RESTORE_ALL=False,
                DROP_REDUNDANT=False)
    base.update(over)
    return base


def test_curation_never_deletes_or_renames_image_files(tmp_path):
    """The filename encodes the video frame index and is the join key across
    rgb/depth/right/conf and back to the source video. Curation must be a
    manifest edit only - deleting or renaming would desynchronise the streams."""
    sdir = _session(tmp_path, n=10)
    before = sorted(p.name for p in (sdir / "rgb").iterdir())

    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=100.0))

    after = sorted(p.name for p in (sdir / "rgb").iterdir())
    assert after == before                 # every file still present, same names


def test_redundant_frames_dropped_by_pose_travel(tmp_path):
    """5 mm per frame with a 20 mm threshold: keep frame 0, then every 4th."""
    sdir = _session(tmp_path, n=13, travel_mm=5.0)
    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=20.0))

    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    kept = [cp.frame_index(r) for r in rows if not cp.is_dropped(r)]
    assert kept == [0, 4, 8, 12]
    dropped = [r for r in rows if cp.is_dropped(r)]
    assert all(r["drop_reason"] == "redundant" for r in dropped)


def test_travel_accumulates_from_last_kept_not_previous_frame(tmp_path):
    """The core of the algorithm. Crawling at 5 mm/frame, every CONSECUTIVE pair
    moves only 5 mm, so a pairwise rule with a 20 mm threshold would drop
    nothing at all. Accumulating from the last KEPT frame is what actually
    thins a slow segment."""
    sdir = _session(tmp_path, n=13, travel_mm=5.0)
    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=20.0))
    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    assert sum(1 for r in rows if cp.is_dropped(r)) == 9     # not 0


def test_fast_motion_keeps_every_frame(tmp_path):
    """Moving faster than the threshold means no frame is redundant."""
    sdir = _session(tmp_path, n=8, travel_mm=250.0)
    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=100.0))
    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    assert all(not cp.is_dropped(r) for r in rows)


def test_unreliable_pose_is_refused_not_trusted(tmp_path):
    """A pose recorded while tracking was lost is worse than none - it would
    report a bogus jump. pose_xyz must refuse it so the caller falls back to
    measuring the images."""
    ok = {"pose_state": "OK", "tx_mm": "10", "ty_mm": "0", "tz_mm": "0"}
    lost = {"pose_state": "SEARCHING", "tx_mm": "10", "ty_mm": "0", "tz_mm": "0"}
    states = cp.CONFIG["POSE_OK_STATES"]
    assert cp.pose_xyz(ok, states) == (10.0, 0.0, 0.0)
    assert cp.pose_xyz(lost, states) is None
    assert cp.pose_xyz({"tx_mm": "", "ty_mm": "", "tz_mm": ""}, states) is None


def test_image_shift_fallback_when_no_pose(tmp_path):
    """With no pose at all the run must still work, measuring image shift."""
    sdir = tmp_path / "sessions" / "sess"
    (sdir / "rgb").mkdir(parents=True)
    (sdir / "meta").mkdir(parents=True)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    rows = []
    for i in range(6):
        fn = f"sess_{i:06d}.png"
        # Each frame is the same ground shifted 2 px - a deliberately slow crawl.
        cv2.imwrite(str(sdir / "rgb" / fn), np.roll(base, i * 2, axis=1))
        rows.append({"video_frame_idx": str(i), "filename": fn})
    with open(sdir / "meta" / "pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_frame_idx", "filename"])
        w.writeheader()
        w.writerows(rows)

    st = cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True,
                                              MIN_SHIFT_FRAC=0.5))
    assert st["signal"] == "image"          # fell back, did not silently use pose
    assert st["after"] < st["before"]       # the crawl was thinned


def test_manual_drops_accept_index_range_and_preview_name(tmp_path):
    """You decide a frame is bad by looking at a PREVIEW (.jpg), so the tool has
    to accept that name as readily as the source .png, a bare index, or a
    range."""
    sdir = _session(tmp_path, n=20)
    cfg = _cfg(MANUAL_DROPS={"sess": ["0-4", "9", "sess_000015.jpg"]})
    cp.curate_session("sess", sdir, cfg)

    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    dropped = sorted(cp.frame_index(r) for r in rows if cp.is_dropped(r))
    assert dropped == [0, 1, 2, 3, 4, 9, 15]
    assert all(r["drop_reason"] == "manual" for r in rows if cp.is_dropped(r))


def test_dry_run_writes_nothing(tmp_path):
    sdir = _session(tmp_path, n=10)
    original = (sdir / "meta" / "pool.csv").read_text()
    cp.curate_session("sess", sdir, _cfg(DRY_RUN=True, DROP_REDUNDANT=True,
                                         MIN_TRAVEL_MM=1000.0))
    assert (sdir / "meta" / "pool.csv").read_text() == original


def test_restore_all_undoes_every_drop(tmp_path):
    """Nothing was deleted, so curation must be fully reversible."""
    sdir = _session(tmp_path, n=13)
    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=20.0))
    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    assert any(cp.is_dropped(r) for r in rows)

    cp.curate_session("sess", sdir, _cfg(RESTORE_ALL=True))
    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    assert all(not cp.is_dropped(r) for r in rows)
    assert len(rows) == 13                  # no row was ever removed


def test_manual_and_redundant_drops_compose(tmp_path):
    """Both mechanisms in one pass: a manually dropped frame must not become an
    anchor for the redundancy pass, and must stay dropped."""
    sdir = _session(tmp_path, n=13, travel_mm=5.0)
    cfg = _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=20.0,
               MANUAL_DROPS={"sess": ["0"]})
    cp.curate_session("sess", sdir, cfg)

    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    by_idx = {cp.frame_index(r): r for r in rows}
    assert cp.is_dropped(by_idx[0]) and by_idx[0]["drop_reason"] == "manual"
    kept = sorted(i for i, r in by_idx.items() if not cp.is_dropped(r))
    assert kept[0] == 1                     # anchoring restarted at the first live frame


def test_dropped_frames_are_skipped_by_the_prelabelers(tmp_path):
    """The whole point: a curated pool must actually shrink what the annotation
    stages process."""
    sdir = _session(tmp_path, n=13, travel_mm=5.0)
    assert len(wd.pool_frames(sdir)) == 13

    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=20.0))
    frames = wd.pool_frames(sdir)
    assert len(frames) == 4
    assert frames == [f"sess_{i:06d}.png" for i in (0, 4, 8, 12)]


def test_dropped_frames_are_skipped_by_batch_selection(tmp_path):
    sdir = _session(tmp_path, n=13, travel_mm=5.0)
    assert len(sb.load_pool(tmp_path)) == 13
    cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True, MIN_TRAVEL_MM=20.0))
    assert len(sb.load_pool(tmp_path)) == 4


def test_uncurated_pool_behaves_exactly_as_before(tmp_path):
    """Backward compatibility: a pool.csv with no `dropped` column at all (every
    session extracted before curation existed) must read as keep-everything."""
    sdir = _session(tmp_path, n=6)
    rows = list(csv.DictReader(open(sdir / "meta" / "pool.csv", encoding="utf-8")))
    assert "dropped" not in rows[0]
    assert len(wd.pool_frames(sdir)) == 6
    assert len(sb.load_pool(tmp_path)) == 6
