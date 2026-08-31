"""prepare_dataset end-to-end on a synthetic Datumaro export, plus the
evaluation metrics. No GPU, no real data, no heavy dependency."""
import json

import numpy as np
import pytest

from conftest import load_script

prep = load_script("training/prepare_dataset.py")
met = load_script("evaluation/metrics.py")
sp = load_script("training/splits.py")

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
    # Enough frames that the single-session fallback split is viable; this test
    # is about duplicate detection, not about splitting.
    a = _one_session_export(tmp_path, "sessX", "taskA", frames_per=6)
    b = _one_session_export(tmp_path, "sessX", "taskB", frames_per=6)  # same session
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


# --------------------------------------------------------------------------- #
# Single-session fallback (35 frames from one recording - the real case)
# --------------------------------------------------------------------------- #
def test_single_session_falls_back_to_frame_blocks_not_empty_splits(tmp_path):
    """One recording cannot be split by session. Silently producing empty
    val/test would mean training blind with no signal that it is learning."""
    root = _one_session_export(tmp_path, "vid3_20260108_103135", "task0",
                               frames_per=35)
    out = tmp_path / "out"
    report, split_map, rows = prep.build(root, tmp_path / "images", out,
                                         val_fraction=0.2, test_fraction=0.2,
                                         seed=1, strict=False)
    summary = json.loads((out / "splits" / "splits_summary.json").read_text())
    assert summary["split_mode"] == "frame_block"
    assert "warning" in summary and "generalisation" in summary["warning"]

    blocks = summary["frame_blocks"]
    assert len(blocks["train"]) > 0
    assert len(blocks["val"]) > 0
    assert len(blocks["test"]) > 0

    # Every split must actually receive rows. Frames in the discarded gap
    # legitimately have no split - that is the buffer doing its job.
    counts = {s: sum(1 for r in rows if r["split"] == s)
              for s in ("train", "val", "test")}
    assert all(v > 0 for v in counts.values()), counts
    n_gap = sum(1 for r in rows if r["split"] == "unassigned")
    assert n_gap == len(blocks["_dropped_gap"])
    assert n_gap > 0, "the temporal buffer should have excluded some frames"


def test_frame_blocks_are_contiguous_and_gap_separated():
    """Blocks, not a random frame split: adjacent frames are near-identical, so
    a random split scores memorisation. The gap buys real separation."""
    ids = [f"s_{i:06d}" for i in range(40)]
    out = sp.assign_frame_blocks(ids, 0.2, 0.2, gap_frames=3)

    for split in ("train", "val", "test"):
        idx = [ids.index(f) for f in out[split]]
        assert idx == list(range(idx[0], idx[-1] + 1)), f"{split} is not contiguous"

    assert max(ids.index(f) for f in out["train"]) < min(ids.index(f)
                                                         for f in out["val"])
    assert max(ids.index(f) for f in out["val"]) < min(ids.index(f)
                                                       for f in out["test"])
    assert len(out["_dropped_gap"]) == 6            # two 3-frame buffers

    everything = out["train"] + out["val"] + out["test"] + out["_dropped_gap"]
    assert sorted(everything) == sorted(ids)        # nothing invented or lost
    assert len(set(everything)) == len(ids)         # nothing in two blocks


def test_frame_block_split_refuses_when_there_are_too_few_frames():
    with pytest.raises(sp.SplitError) as e:
        sp.assign_frame_blocks([f"f{i}" for i in range(4)], 0.2, 0.2)
    assert "at least 5" in str(e.value)


def test_class_counts_are_reported_so_imbalance_is_visible(tmp_path):
    """A class with zero instances cannot be learned, and a model that never
    sees it will silently never predict it. per_class must show exactly what
    was annotated so the gap is obvious before training, not after."""
    root = _one_session_export(tmp_path, "sessOnly", "t", frames_per=8)
    report, _, _ = prep.build(root, tmp_path / "images", tmp_path / "out",
                              val_fraction=0.2, test_fraction=0.2, seed=1,
                              strict=False)
    # This export contains only wild_radish and onion_plant, so every other
    # ontology class must be absent from the counts.
    assert set(report.per_class) == {"wild_radish", CROP_CLASS}
    missing = [c for c in CLASSES if c not in report.per_class]
    assert "grass_weed" in missing and "weed_cluster" in missing


# --------------------------------------------------------------------------- #
# Dropping classes for one build, and excluding un-annotated frames
# --------------------------------------------------------------------------- #
def _mixed_export(tmp_path, name="mixed", frames_per=10, n_empty=0):
    """Frames with wild_radish + grass_weed, plus optionally some with no
    annotations at all (the un-annotated case)."""
    items = []
    for f in range(frames_per):
        anns = [
            {"id": 1, "type": "polygon", "label_id": L["wild_radish"],
             "group": 1, "points": _sq(20, 20, 60), "z_order": 0,
             "attributes": {"lep_visibility": "visible", "targetable": "yes"}},
            {"id": 2, "type": "points", "label_id": L["weed_LEP"], "group": 1,
             "points": [50.0, 50.0], "z_order": 0, "attributes": {}},
            {"id": 3, "type": "polygon", "label_id": L["grass_weed"],
             "group": 2, "points": _sq(200, 20, 60), "z_order": 0,
             "attributes": {"lep_visibility": "visible", "targetable": "yes"}},
            {"id": 4, "type": "points", "label_id": L["weed_LEP"], "group": 2,
             "points": [230.0, 50.0], "z_order": 0, "attributes": {}},
        ]
        items.append({"id": f"sessM_{f:06d}", "annotations": anns,
                      "image": {"path": f"sessM_{f:06d}.png", "size": [480, 640]}})
    for e in range(n_empty):
        i = frames_per + e
        items.append({"id": f"sessM_{i:06d}", "annotations": [],
                      "image": {"path": f"sessM_{i:06d}.png", "size": [480, 640]}})
    doc = {"info": {},
           "categories": {"label": {"labels": [{"name": n, "parent": "",
                                                "attributes": []} for n in LABELS],
                                    "attributes": []}},
           "items": items}
    root = tmp_path / name
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "default.json").write_text(json.dumps(doc))
    return root


def test_dropped_classes_leave_the_ontology_untouched(tmp_path):
    """The ontology fixes COCO ids, the CVAT schema and every file already
    exported. Dropping must be local to the build so a future dataset that DOES
    contain the class still merges with this one."""
    root = _mixed_export(tmp_path)
    out = tmp_path / "out"
    report, _, rows = prep.build(root, tmp_path / "images", out,
                                 val_fraction=0.2, test_fraction=0.2, seed=1,
                                 strict=False, drop_classes=["wild_radish"])

    assert "wild_radish" not in report.per_class
    assert report.per_class.get("grass_weed", 0) > 0
    assert all(r["class_name"] != "wild_radish" for r in rows)

    from common.ontology import CLASSES as LIVE
    assert "wild_radish" in LIVE, "the ontology must NOT be mutated"

    mapping = json.loads((out / "class_mapping.json").read_text())
    assert mapping["dropped"] == ["wild_radish"]
    assert "wild_radish" not in mapping["active_classes"]
    assert mapping["ontology"] == list(LIVE)


def test_dropping_keeps_training_indices_contiguous(tmp_path):
    """A gap in the label indices would silently shift every class above it."""
    root = _mixed_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False,
               drop_classes=["wild_radish", "weed_cluster"])

    active = json.loads((out / "seg_manifest.json").read_text())["classes"]
    assert "wild_radish" not in active and "weed_cluster" not in active
    assert len(active) == len(CLASSES) - 2

    y = (out / "data.yaml").read_text()
    assert f"nc: {len(active)}" in y
    for i, n in enumerate(active):
        assert f"  {i}: {n}" in y

    # Every YOLO label index must be inside the reduced, contiguous range.
    for txt in (out / "labels").rglob("*.txt"):
        for line in txt.read_text().split("\n"):
            if line.strip():
                assert 0 <= int(line.split()[0]) < len(active)

    # And the per-instance index in the manifest agrees with that list.
    for f in json.loads((out / "seg_manifest.json").read_text())["frames"]:
        for inst in f["instances"]:
            assert active[inst["class_index"]] == inst["class_name"]


def test_unannotated_frames_are_excluded_by_default(tmp_path):
    """An empty frame in a hand-annotated export is almost always one you did
    not reach. Training on it teaches the model that plants are background."""
    root = _mixed_export(tmp_path, frames_per=10, n_empty=4)
    out = tmp_path / "out"
    report, _, _ = prep.build(root, tmp_path / "images", out,
                              val_fraction=0.2, test_fraction=0.2, seed=1,
                              strict=False)
    assert report.n_frames == 14                 # all 14 were read...
    seg = json.loads((out / "seg_manifest.json").read_text())
    # ...the 4 un-annotated ones were excluded, and the single-session
    # frame-block split additionally discards its gap frames as a buffer.
    assert 0 < seg["n_frames"] <= 10
    kept = {f["item_id"] for f in seg["frames"]}
    assert not any(int(i.rsplit("_", 1)[1]) >= 10 for i in kept), \
        "an un-annotated frame reached the training set"
    for txt in (out / "labels").rglob("*.txt"):
        assert txt.read_text().strip(), f"{txt.name} is an empty label file"


def test_empty_frames_can_be_kept_deliberately(tmp_path):
    """Genuinely bare ground is a legitimate negative example - but it has to
    be an explicit choice."""
    root = _mixed_export(tmp_path, frames_per=10, n_empty=4)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False, keep_empty_frames=True)
    labels = list((out / "labels").rglob("*.txt"))
    assert any(not t.read_text().strip() for t in labels)


def test_unknown_drop_class_fails_clearly(tmp_path):
    root = _mixed_export(tmp_path)
    with pytest.raises(SystemExit) as e:
        prep.build(root, tmp_path / "images", tmp_path / "out",
                   drop_classes=["nonexistent_weed"], strict=False)
    assert "nonexistent_weed" in str(e.value)


# --------------------------------------------------------------------------- #
# Naming what a build IS, rather than what it lacks
# --------------------------------------------------------------------------- #
def _crop_and_weeds_export(tmp_path, name="cw", frames_per=10):
    """Onions AND weeds in every frame - the export a crop-only build has to
    narrow down."""
    items = []
    for f in range(frames_per):
        anns = [
            {"id": 1, "type": "polygon", "label_id": L["grass_weed"],
             "group": 1, "points": _sq(20, 20, 60), "z_order": 0,
             "attributes": {"lep_visibility": "visible", "targetable": "yes"}},
            {"id": 2, "type": "points", "label_id": L["weed_LEP"], "group": 1,
             "points": [50.0, 50.0], "z_order": 0, "attributes": {}},
            {"id": 3, "type": "polygon", "label_id": L[CROP_CLASS],
             "group": 2, "points": _sq(200, 20, 60), "z_order": 0,
             "attributes": {}},
        ]
        items.append({"id": f"sessC_{f:06d}", "annotations": anns,
                      "image": {"path": f"sessC_{f:06d}.png",
                                "size": [480, 640]}})
    doc = {"info": {},
           "categories": {"label": {"labels": [{"name": n, "parent": "",
                                                "attributes": []} for n in LABELS],
                                    "attributes": []}},
           "items": items}
    root = tmp_path / name
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "default.json").write_text(json.dumps(doc))
    return root


def test_keep_classes_builds_a_crop_only_dataset(tmp_path):
    """Combining the onion-only sessions into one build: name the class the
    build is FOR and everything else goes, without enumerating the weeds."""
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    report, _, rows = prep.build(root, tmp_path / "images", out,
                                 val_fraction=0.2, test_fraction=0.2, seed=1,
                                 strict=False, keep_classes=[CROP_CLASS])

    assert set(report.per_class) == {CROP_CLASS}
    assert all(r["class_name"] == CROP_CLASS for r in rows)

    mapping = json.loads((out / "class_mapping.json").read_text())
    assert mapping["active_classes"] == [CROP_CLASS]
    assert mapping["train_index_to_ontology_name"] == {"0": CROP_CLASS}


def test_an_allow_list_does_not_admit_a_class_appended_later(tmp_path):
    """The reason to prefer it over the equivalent deny-list. The ontology
    grows by APPENDING, and a deny-list written today does not mention a class
    added tomorrow - so that class would silently enter a build that named
    itself onion-only. Asserting against the LIVE ontology is what makes this
    test keep testing the property as CLASSES grows."""
    from common.ontology import CLASSES as LIVE
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False,
               keep_classes=[CROP_CLASS])

    mapping = json.loads((out / "class_mapping.json").read_text())
    assert mapping["dropped"] == sorted(set(LIVE) - {CROP_CLASS})
    assert mapping["selection_mode"] == "keep"
    assert mapping["kept_requested"] == [CROP_CLASS]


def test_a_drop_build_still_records_itself_as_one(tmp_path):
    """The mode is recorded so a later reader can tell 'onion only, by intent'
    from 'these happened to be dropped that day'."""
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False,
               drop_classes=["wild_radish"])
    mapping = json.loads((out / "class_mapping.json").read_text())
    assert mapping["selection_mode"] == "drop"
    assert mapping["kept_requested"] is None


def test_drop_applies_on_top_of_keep(tmp_path):
    """Narrowing an allow-list without rewriting it."""
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False,
               keep_classes=[CROP_CLASS, "grass_weed"],
               drop_classes=["grass_weed"])
    mapping = json.loads((out / "class_mapping.json").read_text())
    assert mapping["active_classes"] == [CROP_CLASS]


def test_an_empty_keep_list_is_refused_not_read_as_keep_everything(tmp_path):
    """`[]` is a mistake worth naming. Treating it as 'no allow-list' would
    build the full-ontology dataset the caller was trying to narrow."""
    root = _crop_and_weeds_export(tmp_path)
    with pytest.raises(SystemExit) as e:
        prep.build(root, tmp_path / "images", tmp_path / "out",
                   keep_classes=[], strict=False)
    assert "keep-classes" in str(e.value).lower()


def test_unknown_keep_class_fails_clearly(tmp_path):
    """A typo must not quietly produce an empty dataset."""
    root = _crop_and_weeds_export(tmp_path)
    with pytest.raises(SystemExit) as e:
        prep.build(root, tmp_path / "images", tmp_path / "out",
                   keep_classes=["onion_plants"], strict=False)   # trailing s
    assert "onion_plants" in str(e.value)


def test_no_keep_list_keeps_every_class(tmp_path):
    """None is not the same as an empty list."""
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False)
    mapping = json.loads((out / "class_mapping.json").read_text())
    assert mapping["dropped"] == []
    assert mapping["selection_mode"] == "drop"


# --------------------------------------------------------------------------- #
# What the labels ARE decides what the metrics MEAN
# --------------------------------------------------------------------------- #
def test_label_provenance_travels_with_the_manifest(tmp_path):
    """A build months later is read by someone who was not in the conversation
    where 'these are just SAM outputs' was said out loud."""
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False,
               label_provenance="prelabel_unreviewed")
    for name in ("seg_manifest.json", "lep_manifest.json"):
        doc = json.loads((out / name).read_text())
        assert doc["label_provenance"] == "prelabel_unreviewed", name


def test_provenance_defaults_to_hand_corrected(tmp_path):
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.2,
               test_fraction=0.2, seed=1, strict=False)
    doc = json.loads((out / "seg_manifest.json").read_text())
    assert doc["label_provenance"] == "hand_corrected"


def test_an_unknown_provenance_is_refused(tmp_path):
    """A typo must not silently record 'these were reviewed'."""
    root = _crop_and_weeds_export(tmp_path)
    with pytest.raises(SystemExit) as e:
        prep.build(root, tmp_path / "images", tmp_path / "out",
                   label_provenance="sam3", strict=False)
    assert "label_provenance" in str(e.value)


# --------------------------------------------------------------------------- #
# An export can outlive the frames it describes
# --------------------------------------------------------------------------- #
def test_frames_whose_image_is_gone_are_excluded_and_counted(tmp_path, capsys):
    """Pool frames deleted after extraction leave annotations pointing at
    nothing. Without this the mismatch surfaces one frame at a time from COCO
    export, which cannot say whether one frame is missing or nine hundred."""
    import cv2
    import numpy as np
    root = _crop_and_weeds_export(tmp_path, frames_per=6)
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    for i in range(4):                      # only 4 of the 6 survive
        cv2.imwrite(str(rgb / f"sessC_{i:06d}.png"),
                    np.zeros((16, 16, 3), np.uint8))

    out = tmp_path / "out"
    prep.build(root, root, out, val_fraction=0.0, test_fraction=0.0, seed=1,
               strict=False, verify_images=True, gap_frames=0)
    printed = capsys.readouterr().out
    assert "EXCLUDED 2 frame(s) whose image is not on disk" in printed
    assert "sessC" in printed

    kept = {f["item_id"] for f in
            json.loads((out / "seg_manifest.json").read_text())["frames"]}
    assert all(int(i.rsplit("_", 1)[1]) < 4 for i in kept)


def test_verify_images_is_off_by_default_so_build_needs_no_images(tmp_path):
    """build() records where to find an image rather than reading one, which
    is what lets it be tested without a dataset on disk."""
    root = _crop_and_weeds_export(tmp_path)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "no_images_here", out, val_fraction=0.0,
               test_fraction=0.0, seed=1, strict=False)
    assert json.loads((out / "seg_manifest.json").read_text())["n_frames"] > 0


# --------------------------------------------------------------------------- #
# Writing the manifests
# --------------------------------------------------------------------------- #
def test_the_frame_manifests_are_not_pretty_printed(tmp_path):
    """indent=2 puts every polygon COORDINATE on its own line. On a few
    thousand masks that is hundreds of megabytes of whitespace, and the whole
    string then goes through a single write() - which Windows rejects with
    OSError 22 on some volumes while succeeding on others."""
    root = _crop_and_weeds_export(tmp_path, frames_per=4)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.0,
               test_fraction=0.0, seed=1, strict=False)

    for name in ("seg_manifest.json", "lep_manifest.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert json.loads(text), f"{name} must still be valid JSON"
        # One line, or close to it - certainly not one line per coordinate.
        assert text.count("\n") <= 1, f"{name} is pretty-printed"

    # The small reports stay readable: they are read by people.
    report = (out / "dataset_report.json").read_text(encoding="utf-8")
    assert report.count("\n") > 1


def test_the_manifests_survive_a_round_trip(tmp_path):
    """Dropping the indent must not change the CONTENT."""
    root = _crop_and_weeds_export(tmp_path, frames_per=4)
    out = tmp_path / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.0,
               test_fraction=0.0, seed=1, strict=False)
    doc = json.loads((out / "seg_manifest.json").read_text(encoding="utf-8"))
    assert doc["frames"] and doc["classes"]
    assert doc["frames"][0]["instances"][0]["polygons"]
    assert "label_provenance" in doc and "images_root" in doc


def test_a_manifest_is_written_even_if_the_directory_is_new(tmp_path):
    root = _crop_and_weeds_export(tmp_path, frames_per=4)
    out = tmp_path / "deep" / "nested" / "out"
    prep.build(root, tmp_path / "images", out, val_fraction=0.0,
               test_fraction=0.0, seed=1, strict=False)
    assert (out / "seg_manifest.json").exists()


# --------------------------------------------------------------------------- #
# A hand-curated batch folder: annotations/ + rgb/, filenames that no longer
# name a drive. It is the only shape that holds onions and weeds in one frame,
# so it must be buildable rather than an error nobody can act on.
# --------------------------------------------------------------------------- #
def _batch_export(tmp_path, folder="Mix_raj Batch 01", frames=3):
    """Frames named the way a person names them, not the way extraction does."""
    items = []
    for f in range(frames):
        items.append({"id": f"photo{f}",
                      "image": {"path": f"photo{f}.png", "size": [480, 640]},
                      "annotations": [
                          {"id": 1, "type": "polygon",
                           "label_id": L["wild_radish"], "group": 1,
                           "points": _sq(20, 20, 60), "z_order": 0,
                           "attributes": {"lep_visibility": "visible",
                                          "targetable": "yes"}},
                          {"id": 2, "type": "points", "label_id": L["weed_LEP"],
                           "group": 1, "points": [50.0, 50.0], "z_order": 0,
                           "attributes": {}},
                          {"id": 3, "type": "polygon", "label_id": L[CROP_CLASS],
                           "group": 2, "points": _sq(200, 150, 80),
                           "z_order": 0, "attributes": {}}]})
    doc = {"info": {},
           "categories": {"label": {"labels": [{"name": n, "parent": "",
                                                "attributes": []} for n in LABELS],
                                    "attributes": []}},
           "items": items}
    root = tmp_path / folder
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "default.json").write_text(json.dumps(doc))
    return root


def test_a_hand_curated_batch_builds_under_its_folder_name(tmp_path):
    root = _batch_export(tmp_path)
    report, split_map, rows = prep.build(root, tmp_path / "images",
                                         tmp_path / "out", val_fraction=0.34,
                                         test_fraction=0.33, seed=1,
                                         strict=False)
    assert report.ok, report.errors
    sessions = sorted(s for v in split_map.values() for s in v)
    assert sessions == ["Mix_raj_Batch_01"], "the space becomes an underscore"
    assert not any(e["kind"] == "unresolvable_session" for e in report.errors)


def test_two_batches_stay_two_sessions(tmp_path):
    """One shared id would let a split put near-copies of the same plant on
    both sides, which shows up only as a validation score that is too good."""
    roots = [_batch_export(tmp_path, "Batch 01"),
             _batch_export(tmp_path, "Batch 02")]
    report, split_map, _ = prep.build(roots, tmp_path / "images",
                                      tmp_path / "out", val_fraction=0.25,
                                      test_fraction=0.25, seed=1, strict=False)
    assert sorted(s for v in split_map.values() for s in v) == \
        ["Batch_01", "Batch_02"]


def test_a_batch_frame_that_names_its_drive_keeps_it(tmp_path):
    """A curated batch drawn from real drives must stay those drives - the
    fallback is a fallback, never an override."""
    root = _batch_export(tmp_path, "Mix_raj Batch 01", frames=0)
    doc = json.loads((root / "annotations" / "default.json").read_text())
    doc["items"] = [
        {"id": f"vid3_20260108_103135_{i:06d}",
         "image": {"path": f"vid3_20260108_103135_{i:06d}.png",
                   "size": [480, 640]},
         "annotations": [{"id": 3, "type": "polygon", "label_id": L[CROP_CLASS],
                          "group": 2, "points": _sq(200, 150, 80),
                          "z_order": 0, "attributes": {}}]}
        for i in range(4)]
    (root / "annotations" / "default.json").write_text(json.dumps(doc))
    report, split_map, _ = prep.build(root, tmp_path / "images",
                                      tmp_path / "out", val_fraction=0.25,
                                      test_fraction=0.25, seed=1, strict=False,
                                      keep_classes=[CROP_CLASS])
    assert sorted(s for v in split_map.values() for s in v) == \
        ["vid3_20260108_103135"]
