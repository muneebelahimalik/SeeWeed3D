"""End-to-end checks for extract_sessions.py and select_batches.py against
synthetic v1/v2 sessions built with real FFV1 MKVs (see conftest.py)."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from conftest import load_script


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_cvat_labels_every_attribute_has_nonempty_values():
    """CVAT's Raw label editor rejects the whole schema on paste if any
    attribute - including 'text' ones, where CVAT still expects the default
    wrapped in a single-element list - is missing 'values' or has it empty.
    Regression guard for the same bug class that broke weed_cvat_labels.json."""
    s2 = load_script("extraction/select_batches.py")
    for label in s2.CVAT_LABELS:
        for attr in label["attributes"]:
            assert "values" in attr, (
                f"{label['name']}.{attr['name']} has no 'values' key")
            assert isinstance(attr["values"], list) and len(attr["values"]) > 0, (
                f"{label['name']}.{attr['name']} has an empty 'values' array")


def test_registry_detects_both_formats(extracted_root):
    reg = {r["session_id"]: r for r in _rows(extracted_root / "registry.csv")}
    assert len(reg) == 3
    assert reg["vid_20250304_142804"]["capture_format"] == "v1"
    assert reg["vid_20260108_103135"]["capture_format"] == "v2"
    assert reg["vid_20260108_103135"]["has_confidence"] == "True"
    assert reg["vid_20260108_103135"]["has_right"] == "True"
    assert reg["vid_20250304_142804"]["has_confidence"] == "False"


def test_depth_lossless_and_index_aligned(extracted_root, raw_root):
    for sid, truth in raw_root["truth"].items():
        sdir = extracted_root / "sessions" / sid
        for r in _rows(sdir / "meta" / "pool.csv"):
            v = int(r["video_frame_idx"])
            d = cv2.imread(str(sdir / "depth" / r["filename"]), cv2.IMREAD_UNCHANGED)
            assert d is not None and d.dtype == np.uint16
            assert int(d.sum()) == truth[v]["depth_sum"]
            assert int(d[d.shape[0] // 2, d.shape[1] // 2]) == truth[v]["depth_at"]


def test_dropped_frame_index_recovered(extracted_root, raw_root):
    sid = "vid_20260108_103135"
    pool = _rows(extracted_root / "sessions" / sid / "meta" / "pool.csv")
    truth = raw_root["truth"][sid]
    for r in pool:
        assert int(r["capture_frame_idx"]) == truth[int(r["video_frame_idx"])]["capture_idx"]
    assert 7 not in {int(r["capture_frame_idx"]) for r in pool}   # the dropped one


def test_confidence_sentinel_excluded(extracted_root):
    idx = _rows(extracted_root / "sessions" / "vid_20260108_103135" / "meta" / "frames_index.csv")
    means = [float(r["conf_mean"]) for r in idx if r.get("conf_mean") not in ("", None)]
    assert means and max(means) <= 100     # 255 sentinel removed


def test_depth_not_rescaled_and_calibration_parsed(extracted_root):
    sdir = extracted_root / "sessions" / "vid_20250304_142804"
    pool = _rows(sdir / "meta" / "pool.csv")
    d = cv2.imread(str(sdir / "depth" / pool[0]["filename"]), cv2.IMREAD_UNCHANGED)
    assert 700 < int(np.median(d[d > 0])) < 1100      # raw mm, not ×3000
    calib = json.loads((sdir / "meta" / "calibration.json").read_text())
    assert abs(calib["mm_per_px_at_1000mm"] - 1000 / 350) < 1e-3


def _select(extracted_root, holdout):
    s2 = load_script("extraction/select_batches.py")
    s2.CONFIG.update({
        "DATASET_ROOT": str(extracted_root), "HOLDOUT_SESSIONS": holdout,
        "GATES": {"min_sharpness": 5.0, "min_veg_frac": 0.005, "max_clip_frac": 0.95,
                  "max_dark_frac": 0.95, "min_depth_valid_frac_veg": 0.05},
        "MIN_PHASH_DISTANCE": 2, "MAX_SESSION_SHARE": 0.9,
        "BATCHES": [{"name": "b01", "n": 10, "stress_fraction": 0.3, "pool": "train"},
                    {"name": "b_test", "n": 6, "stress_fraction": 0.2, "pool": "holdout"}],
    })
    s2.main()

    def files(name):
        d = extracted_root / "batches" / name / "images"
        return sorted(p.name for p in d.iterdir()) if d.exists() else []
    return files


def test_batches_respect_holdout_and_reproducible(extracted_root):
    holdout = ["vid_20250305_090000"]
    files = _select(extracted_root, holdout)
    train, test = files("b01"), files("b_test")
    assert train and test
    assert all(not f.startswith("vid_20250305_090000") for f in train)  # no leak
    assert all(f.startswith("vid_20250305_090000") for f in test)
    assert not (set(train) & set(test))                                 # disjoint

    import shutil
    shutil.rmtree(extracted_root / "batches")
    files2 = _select(extracted_root, holdout)
    assert files2("b01") == train and files2("b_test") == test          # stable seed
