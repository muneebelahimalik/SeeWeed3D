"""prepare_dataset end-to-end on a synthetic Datumaro export, plus the
evaluation metrics. No GPU, no real data, no heavy dependency."""
import json

import numpy as np
import pytest

from conftest import load_script

prep = load_script("training/prepare_dataset.py")
met = load_script("evaluation/metrics.py")

from common.ontology import CLASSES, CROP_CLASS  # noqa: E402

LABELS = list(CLASSES) + ["weed_LEP", "ignore_region"]
L = {n: i for i, n in enumerate(LABELS)}


def _sq(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


def _export(tmp_path, n_sessions=4, frames_per=2):
    """A minimal but contract-valid Datumaro export across several sessions."""
    items = []
    for s in range(n_sessions):
        for f in range(frames_per):
            sid = f"sess{s:02d}"
            anns = [
                {"id": 1, "type": "polygon", "label_id": L["wild_radish"],
                 "group": 1, "points": _sq(20, 20, 60), "z_order": 0,
                 "attributes": {"lep_visibility": "visible", "targetable": "yes",
                                "growth_stage": "3-5-leaf"}},
                {"id": 2, "type": "points", "label_id": L["weed_LEP"],
                 "group": 1, "points": [50.0, 50.0], "z_order": 0,
                 "attributes": {"lep_visibility": "visible"}},
                {"id": 3, "type": "polygon", "label_id": L[CROP_CLASS],
                 "group": 2, "points": _sq(200, 150, 80), "z_order": 0,
                 "attributes": {}},
            ]
            items.append({"id": f"{sid}_{f:06d}",
                          "annotations": anns,
                          "image": {"path": f"{sid}_{f:06d}.png",
                                    "size": [480, 640]}})
    doc = {"info": {},
           "categories": {"label": {"labels": [{"name": n, "parent": "",
                                                "attributes": []}
                                               for n in LABELS],
                                    "attributes": []}},
           "items": items}
    root = tmp_path / "export"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "default.json").write_text(json.dumps(doc))
    return root


def test_prepare_dataset_produces_every_required_artefact(tmp_path):
    root = _export(tmp_path)
    out = tmp_path / "out"
    report, split_map, rows = prep.build(root, tmp_path / "images", out,
                                         val_fraction=0.25, test_fraction=0.25,
                                         seed=1, strict=False)

    assert (out / "data.yaml").exists()
    assert (out / "lep_manifest.json").exists()
    assert (out / "dataset_report.json").exists()
    assert (out / "annotations_needing_correction.json").exists()
    for split in ("train", "val", "test"):
        assert (out / "splits" / f"{split}_sessions.txt").exists()
        assert (out / "splits" / f"{split}_images.txt").exists()
    assert (out / "splits" / "splits_summary.json").exists()

    # data.yaml carries the ontology order verbatim.
    y = (out / "data.yaml").read_text()
    assert f"nc: {len(CLASSES)}" in y
    for i, n in enumerate(CLASSES):
        assert f"  {i}: {n}" in y

    # A contract-valid export must be error-free.
    assert report.ok, report.errors
    assert report.n_leps == 8                       # 4 sessions x 2 frames
    assert len(rows) == 8


def test_yolo_labels_are_written_per_split(tmp_path):
    root = _export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.25,
               test_fraction=0.25, seed=1, strict=False)
    files = list((out / "labels").rglob("*.txt"))
    assert files
    body = files[0].read_text().strip().splitlines()
    assert body
    for line in body:
        parts = line.split()
        assert 0 <= int(parts[0]) < len(CLASSES)
        coords = [float(v) for v in parts[1:]]
        assert all(0.0 <= c <= 1.0 for c in coords)


def test_sessions_never_span_splits(tmp_path):
    """The leakage guarantee, enforced end to end."""
    root = _export(tmp_path, n_sessions=6)
    out = tmp_path / "out"
    _, split_map, rows = prep.build(root, tmp_path / "images", out,
                                    val_fraction=0.2, test_fraction=0.2,
                                    seed=3, strict=False)
    seen = {}
    for split, sessions in split_map.items():
        for s in sessions:
            assert s not in seen, f"{s} is in {seen.get(s)} and {split}"
            seen[s] = split
    # Every manifest row inherits its session's split.
    for r in rows:
        assert r["split"] == seen[r["session_id"]]


def test_broken_contract_is_reported_and_blocks_by_default(tmp_path):
    """A visible, targetable weed with no grouped LEP must stop the build."""
    root = _export(tmp_path, n_sessions=2)
    p = root / "annotations" / "default.json"
    doc = json.loads(p.read_text())
    doc["items"][0]["annotations"] = [a for a in doc["items"][0]["annotations"]
                                      if a["type"] != "points"]
    p.write_text(json.dumps(doc))

    with pytest.raises(SystemExit):
        prep.build(root, tmp_path / "images", tmp_path / "out",
                   val_fraction=0.25, test_fraction=0.25, seed=1, strict=True)

    out = tmp_path / "out2"
    report, _, _ = prep.build(root, tmp_path / "images", out,
                              val_fraction=0.25, test_fraction=0.25,
                              seed=1, strict=False)
    assert not report.ok
    needs = json.loads((out / "annotations_needing_correction.json").read_text())
    assert any(n["kind"] == "missing_lep" for n in needs)


def test_missing_export_fails_with_instructions(tmp_path):
    with pytest.raises(SystemExit) as e:
        prep.find_annotation_files(tmp_path / "nothing")
    assert "Datumaro 1.0" in str(e.value)


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def _disc(size, cx, cy, r):
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def test_onion_recall_is_reported_separately_from_iou():
    """Crop safety is asymmetric: missing onion tissue can destroy the crop,
    while a false onion merely skips a weed. IoU averages both directions and
    can look healthy while crop pixels are missed."""
    gt = _disc(100, 50, 50, 20)
    partial = _disc(100, 50, 50, 14)          # under-segmented: misses a rim
    m = met.onion_safety_metrics(partial, gt)
    assert m["onion_recall"] < 1.0
    assert m["missed_onion_px"] > 0
    assert m["onion_precision"] == pytest.approx(1.0, abs=1e-6)

    perfect = met.onion_safety_metrics(gt, gt)
    assert perfect["onion_recall"] == pytest.approx(1.0)
    assert perfect["missed_onion_px"] == 0


def test_segmentation_metrics_match_instances_by_class():
    a = _disc(100, 30, 30, 12)
    b = _disc(100, 70, 70, 12)
    res = met.segmentation_metrics([a, b], ["wild_radish", "grass_weed"],
                                   [a, b], ["wild_radish", "grass_weed"])
    assert res["mask_ap50_95"] == pytest.approx(1.0)
    assert res["per_class"]["wild_radish"]["recall"] == pytest.approx(1.0)

    # A class mismatch must not be matched.
    wrong = met.segmentation_metrics([a], ["grass_weed"], [a], ["wild_radish"])
    assert wrong["per_class"]["wild_radish"]["recall"] == 0.0


def test_lep_error_metrics_and_thresholds():
    gt = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    pred = gt + np.array([[1.0, 0.0], [0.0, 6.0], [12.0, 0.0]])
    m = met.lep_errors(pred, gt)
    assert m["n"] == 3
    assert m["pct_within_2px"] == pytest.approx(100 / 3, abs=1e-6)
    assert m["pct_within_10px"] == pytest.approx(200 / 3, abs=1e-6)
    assert m["pct_within_15px"] == pytest.approx(100.0)


def test_normalised_lep_error_compares_across_growth_stages():
    """5px on a cotyledon is a miss; 5px on a large rosette is a hit."""
    gt = np.array([[10.0, 10.0], [50.0, 50.0]])
    pred = gt + np.array([[5.0, 0.0], [5.0, 0.0]])
    m = met.lep_errors(pred, gt, plant_radius_px=[10.0, 100.0])
    assert m["pct_within_0.1_radius"] == pytest.approx(50.0)


def test_lep_inside_mask_rate_detects_wrong_instance():
    m1 = _disc(100, 30, 30, 12)
    inside = met.lep_inside_mask_rate([(30, 30)], [m1])
    outside = met.lep_inside_mask_rate([(80, 80)], [m1])
    assert inside == 1.0 and outside == 0.0


def test_method_comparison_covers_every_required_baseline():
    """The plan requires the learned model to be compared against all four."""
    gt = np.array([[10.0, 10.0], [20.0, 20.0]])
    methods = {"bbox_center": gt + 6.0, "centroid": gt + 3.0,
               "dt_peak": gt + 1.5, "lep_py": gt + 1.0, "learned": gt + 0.5}
    cmp = met.compare_lep_methods(gt, methods)
    assert set(cmp) == set(methods)
    assert cmp["learned"]["median_px"] < cmp["bbox_center"]["median_px"]


def test_uncertainty_calibration_detects_a_useful_sigma():
    """A sigma that does not track error is worse than none - the abstention
    threshold would reject good targets and pass bad ones."""
    rng = np.random.default_rng(0)
    sig = np.linspace(1, 20, 60)
    good = met.uncertainty_calibration(sig, sig + rng.normal(0, 0.5, 60))
    assert good["spearman"] > 0.8
    bad = met.uncertainty_calibration(sig, rng.permutation(sig))
    assert abs(bad["spearman"]) < 0.5


def test_3d_metrics_refuse_to_invent_accuracy_without_reference_labels():
    """Depth self-consistency is NOT metric accuracy."""
    out = met.metrics_3d([[1.0, 2.0, 900.0]], None)
    assert out["n"] == 0 and "no reference 3D labels" in out["note"]


def test_safety_metrics_count_reasons_and_abstentions():
    targets = [
        {"safety_status": "candidate", "abstained": False,
         "rejection_reasons": []},
        {"safety_status": "abstain", "abstained": True,
         "rejection_reasons": ["onion_safety_conflict"]},
        {"safety_status": "abstain", "abstained": True,
         "rejection_reasons": ["weed_cluster", "low_lep_confidence"]}]
    m = met.safety_metrics(targets)
    assert m["n"] == 3 and m["n_candidates"] == 1
    assert m["abstention_rate"] == pytest.approx(2 / 3)
    assert m["rejection_reasons"]["onion_safety_conflict"] == 1
    assert m["onion_conflict_rate"] == pytest.approx(1 / 3)


def test_latency_summary_reports_percentiles():
    t = [{"total": 10.0, "segmentation": 6.0},
         {"total": 20.0, "segmentation": 12.0},
         {"total": 30.0, "segmentation": 18.0}]
    s = met.latency_summary(t)
    assert s["total"]["p50_ms"] == pytest.approx(20.0)
    assert s["total"]["n"] == 3


# --------------------------------------------------------------------------- #
# Merging separate CVAT tasks (one task per session - the normal workflow)
# --------------------------------------------------------------------------- #
def _one_session_export(tmp_path, sid, name, frames_per=2, label_order=None):
    """A single-session export, optionally with a DIFFERENT label order, as two
    independently-created CVAT tasks genuinely can have."""
    labels = label_order or LABELS
    idx = {n: i for i, n in enumerate(labels)}
    items = []
    for f in range(frames_per):
        items.append({"id": f"{sid}_{f:06d}",
                      "image": {"path": f"{sid}_{f:06d}.png", "size": [480, 640]},
                      "annotations": [
                          {"id": 1, "type": "polygon",
                           "label_id": idx["wild_radish"], "group": 1,
                           "points": _sq(20, 20, 60), "z_order": 0,
                           "attributes": {"lep_visibility": "visible",
                                          "targetable": "yes"}},
                          {"id": 2, "type": "points", "label_id": idx["weed_LEP"],
                           "group": 1, "points": [50.0, 50.0], "z_order": 0,
                           "attributes": {}},
                          {"id": 3, "type": "polygon", "label_id": idx[CROP_CLASS],
                           "group": 2, "points": _sq(200, 150, 80), "z_order": 0,
                           "attributes": {}}]})
    doc = {"info": {},
           "categories": {"label": {"labels": [{"name": n, "parent": "",
                                                "attributes": []} for n in labels],
                                    "attributes": []}},
           "items": items}
    root = tmp_path / name
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "default.json").write_text(json.dumps(doc))
    return root


def test_several_cvat_exports_merge_into_one_dataset(tmp_path):
    """One CVAT task per session is good practice; merging them is the normal
    path, not an edge case."""
    roots = [_one_session_export(tmp_path, f"sess{i:02d}", f"task{i}")
             for i in range(4)]
    report, split_map, rows = prep.build(roots, tmp_path / "images",
                                         tmp_path / "out", val_fraction=0.25,
                                         test_fraction=0.25, seed=1, strict=False)
    assert report.ok, report.errors
    assert report.n_frames == 8                       # 4 sessions x 2 frames
    assert len(rows) == 8
    all_sessions = sorted(s for v in split_map.values() for s in v)
    assert all_sessions == ["sess00", "sess01", "sess02", "sess03"]


def test_a_parent_folder_of_exports_is_discovered(tmp_path):
    parent = tmp_path / "all_exports"
    parent.mkdir()
    for i in range(3):
        _one_session_export(parent, f"sess{i:02d}", f"task{i}")
    files = prep.find_annotation_files(parent)
    assert len(files) == 3


def test_merging_resolves_labels_by_NAME_not_by_index(tmp_path):
    """Two CVAT tasks can legitimately order their labels differently, so
    label_id 2 may mean different classes in each. A merge keyed on the index
    would silently relabel half the dataset."""
    normal = _one_session_export(tmp_path, "sessA", "taskA")
    shuffled = list(reversed(LABELS))
    assert shuffled != LABELS
    other = _one_session_export(tmp_path, "sessB", "taskB",
                                label_order=shuffled)

    report, _, rows = prep.build([normal, other], tmp_path / "images",
                                 tmp_path / "out", val_fraction=0.0,
                                 test_fraction=0.5, seed=1, strict=False)
    assert report.ok, report.errors
    # Every instance must be the class its NAME says, from both exports.
    assert set(report.per_class) == {"wild_radish", CROP_CLASS}
    assert report.per_class["wild_radish"] == 4
    assert all(r["class_name"] == "wild_radish" for r in rows)


def test_the_same_frame_in_two_exports_is_reported(tmp_path):
    """Duplicate frames would be trained on twice and could span two splits -
    exactly the leakage the session rule exists to prevent."""
    a = _one_session_export(tmp_path, "sessX", "taskA")
    b = _one_session_export(tmp_path, "sessX", "taskB")     # same session again
    report, _, _ = prep.build([a, b], tmp_path / "images", tmp_path / "out",
                              val_fraction=0.0, test_fraction=0.0, seed=1,
                              strict=False)
    assert any(e["kind"] == "duplicate_frame_across_exports"
               for e in report.errors)


def test_seg_manifest_is_written_for_the_permissive_backend(tmp_path):
    """The BSD-3 Mask R-CNN path trains from this, so it must exist alongside
    the YOLO labels rather than instead of them."""
    root = _export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.25,
               test_fraction=0.25, seed=1, strict=False)
    doc = json.loads((out / "seg_manifest.json").read_text())
    assert doc["classes"] == list(CLASSES)
    assert doc["n_frames"] == 8
    f = doc["frames"][0]
    assert f["split"] in ("train", "val", "test")
    assert {i["class_name"] for i in f["instances"]} == {"wild_radish", CROP_CLASS}
    assert f["instances"][0]["polygons"]
