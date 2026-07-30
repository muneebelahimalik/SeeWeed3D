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


def test_shift_is_summed_as_a_vector_not_a_magnitude(tmp_path):
    """A jittering but stationary camera must not accumulate its way past the
    threshold. Summing |shift| would count every wobble - and phase
    correlation's strictly positive noise floor on near-identical frames - as
    forward progress, so exact duplicates would be kept. Summing the vector
    cancels the wobble, which is what actually happened physically."""
    sdir = tmp_path / "sessions" / "sess"
    (sdir / "rgb").mkdir(parents=True)
    (sdir / "meta").mkdir(parents=True)
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    rows = []
    for i in range(12):
        fn = f"sess_{i:06d}.png"
        # Oscillate between two offsets: net displacement stays ~0 forever,
        # while the sum of per-step magnitudes grows without bound.
        cv2.imwrite(str(sdir / "rgb" / fn), np.roll(base, 6 * (i % 2), axis=1))
        rows.append({"video_frame_idx": str(i), "filename": fn})
    with open(sdir / "meta" / "pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_frame_idx", "filename"])
        w.writeheader()
        w.writerows(rows)

    st = cp.curate_session("sess", sdir, _cfg(DROP_REDUNDANT=True,
                                              MIN_SHIFT_FRAC=0.10))
    # 12 jittering frames of the same ground: at most a couple survive.
    assert st["after"] <= 2, f"jitter accumulated as travel: kept {st['after']}"


def test_image_shift_vec_returns_signed_components():
    """The vector must be signed and directional, not a magnitude."""
    rng = np.random.default_rng(2)
    base = rng.integers(0, 255, (128, 256), dtype=np.uint8).astype(np.float32)
    right = np.roll(base, 8, axis=1).astype(np.float32)

    fwd = cp.image_shift_vec(base, right)
    back = cp.image_shift_vec(right, base)
    assert fwd is not None and back is not None
    # Opposite directions must have opposite sign, so they cancel when summed.
    assert fwd[0] * back[0] < 0
    assert abs(fwd[0] + back[0]) < 0.01


def test_drop_histogram_localises_a_slow_start():
    """The histogram is the diagnostic that decides whether a threshold is
    right, so it must actually show WHERE drops fall, not just how many."""
    rows = [{"dropped": "1" if i < 20 else "0"} for i in range(100)]
    lines = cp.drop_histogram(rows, buckets=10)
    assert len(lines) == 10
    assert "100.0%" in lines[0] and "100.0%" in lines[1]   # slow start
    assert "0.0%" in lines[5]                               # clean later on


def test_diagnose_pose_distinguishes_missing_from_untrusted():
    """'Fell back to image shift' has two very different causes and they need
    different responses, so they must be reported differently."""
    states = cp.CONFIG["POSE_OK_STATES"]
    none = cp.diagnose_pose([{"filename": "a.png"}], states)
    assert "no pose recorded" in none

    lost = cp.diagnose_pose(
        [{"tx_mm": "1", "ty_mm": "0", "tz_mm": "0", "pose_state": "SEARCHING"}],
        states)
    assert "no frame had a trusted pose_state" in lost and "SEARCHING" in lost

    good = cp.diagnose_pose(
        [{"tx_mm": "1", "ty_mm": "0", "tz_mm": "0", "pose_state": "OK"}], states)
    assert "usable on all 1" in good


def test_sweep_is_monotonic_and_matches_the_real_selection():
    """The sweep must agree exactly with what applying that threshold would
    actually do - otherwise it is a table you cannot act on."""
    # Straight-line travel, 1 unit per frame.
    positions = [np.array([float(i), 0.0]) for i in range(21)]
    forced = [False] * 21

    sweep = cp.sweep_thresholds(positions, forced, [1, 2, 5, 10], "frac")
    assert [s["kept"] for s in sweep] == [21, 11, 5, 3]     # strictly decreasing

    for s in sweep:
        real = cp.select_keeps(positions, forced, s["threshold"])
        assert len(real) == s["kept"]
        assert s["kept"] + s["dropped"] == len(positions)


def test_sweep_reports_overlap_only_for_image_mode():
    """Overlap is 1 - shift as a fraction of frame width; it is meaningless for
    a pose threshold in millimetres, so it must not be invented there."""
    positions = [np.array([float(i), 0.0]) for i in range(10)]
    forced = [False] * 10

    frac = cp.sweep_thresholds(positions, forced, [0.25], "frac")
    assert abs(frac[0]["overlap_pct"] - 75.0) < 1e-6

    mm = cp.sweep_thresholds(positions, forced, [100], "mm")
    assert "overlap_pct" not in mm[0]


def test_unmeasurable_frames_are_never_dropped():
    """A frame whose shift could not be measured must be kept, at any
    threshold - a measurement failure is not evidence of redundancy."""
    positions = [np.zeros(2) for _ in range(5)]      # no travel at all
    forced = [False, False, True, False, False]      # frame 2 unmeasurable
    keep = cp.select_keeps(positions, forced, 999.0)
    assert keep == [0, 2]                            # anchor + the forced one


def test_pose_mode_used_only_when_every_frame_has_one(tmp_path):
    """Mixing millimetres and frame-widths inside one session would make the
    threshold change meaning partway through, so a single missing pose must
    demote the whole session to image mode rather than silently mixing."""
    sdir = _session(tmp_path, n=6, travel_mm=50.0)
    rows, _ = cp.read_pool(sdir)
    _, _, signal, unit = cp.frame_positions(rows, sdir / "rgb", cp.CONFIG)
    assert signal == "pose" and unit == "mm"

    rows[3]["pose_state"] = "SEARCHING"               # one bad frame
    _, _, signal, unit = cp.frame_positions(rows, sdir / "rgb", cp.CONFIG)
    assert signal == "image" and unit == "frac"
