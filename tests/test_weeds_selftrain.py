"""The self-training round, end to end on synthetic predictions.

The scoring is tested in test_pseudo_label.py. This exercises the plumbing:
that a prediction COCO in, produces two batches out, with the right frames in
each and provenance stamped on both.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import pseudo_label as pl                       # noqa: E402


def soil(shape=(80, 80)):
    return np.full(shape + (3,), (70, 45, 60), np.uint8)


def with_plants(bgr, boxes):
    for (y0, y1, x0, x1) in boxes:
        bgr[y0:y1, x0:x1] = (35, 160, 45)
    return bgr


def poly_of(y0, y1, x0, x1):
    return [float(x0), float(y0), float(x1), float(y0),
            float(x1), float(y1), float(x0), float(y1)]


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A pool session, a prediction COCO over it, and the runner pointed there."""
    pool = tmp_path / "pool"
    sess = pool / "vid_test"
    (sess / "rgb").mkdir(parents=True)
    # Named and placed the way the runner discovers them: a stamped look folder
    # under the round directory. Handing it the path directly would test a
    # wiring that no longer exists.
    pred = tmp_path / "runs" / "weeds_r0" / "look_vid_test_20260101_0000"
    (pred / "overlays").mkdir(parents=True)

    boxes = [(20, 40, 20, 40), (20, 40, 50, 70)]
    images, anns, ann_id = [], [], 1

    def frame(name, plant_boxes, pred_boxes, score=0.9):
        nonlocal ann_id
        bgr = with_plants(soil(), plant_boxes)
        cv2.imwrite(str(sess / "rgb" / name), bgr)
        cv2.imwrite(str(pred / "overlays" / name), bgr)
        iid = len(images) + 1
        images.append({"id": iid, "file_name": name, "height": 80, "width": 80})
        for b in pred_boxes:
            anns.append({"id": ann_id, "image_id": iid, "category_id": 35,
                         "segmentation": [poly_of(*b)], "iscrowd": 0,
                         "bbox": [b[2], b[0], b[3] - b[2], b[1] - b[0]],
                         "area": float((b[1] - b[0]) * (b[3] - b[2])),
                         "score": score})
            ann_id += 1

    # GOOD: predictions cover every plant.
    for i in range(4):
        frame(f"good_{i}.png", boxes, boxes)
    # BAD: two plants, one predicted. Every detection correct, half the
    # vegetation would become background.
    for i in range(2):
        frame(f"miss_{i}.png", boxes, boxes[:1])
    # BAD: a mask on bare soil.
    frame("soil_0.png", [boxes[0]], [(55, 75, 55, 75)])

    (pred / "instances_default.json").write_text(json.dumps({
        "info": {"description": "SeeWeed3D MODEL PREDICTIONS - not ground truth"},
        "images": images, "annotations": anns,
        "categories": [{"id": 35, "name": "grass_weed"}]}))

    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    monkeypatch.setattr(mod, "WEED_POOL_ROOT", str(pool))
    monkeypatch.setattr(mod, "SESSIONS", ["vid_test"])
    monkeypatch.setattr(mod, "IMAGES", "")
    monkeypatch.setattr(mod, "RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(mod, "ROUND", 0)
    monkeypatch.setattr(mod, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(mod, "HOLDOUT_TEST", [])
    monkeypatch.setattr(mod, "N_HAND", 40)
    # The session's own folder: every per-session assertion below is about one
    # batch, and the run folder now holds one of these per drive.
    return mod, tmp_path / "out" / "vid_test"


def batch(out, which):
    p = out / which / "instances_default.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_it_runs_and_writes_both_batches(run):
    mod, out = run
    assert mod.main() == 0
    assert batch(out, "accept") is not None
    assert batch(out, "review") is not None
    assert (out / "selftrain_report.json").exists()


def test_frames_covering_their_vegetation_are_accepted(run):
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert any(f.startswith("good_") for f in rep["accepted"])


def test_frames_that_miss_plants_go_to_review_not_accept(run):
    """THE case this exists for: every detection is correct, and the missed
    plants would become BACKGROUND in the pseudo-label."""
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert not any(f.startswith("miss_") for f in rep["accepted"])
    assert any(f.startswith("miss_") for f in rep["review"])


def test_a_mask_on_soil_goes_to_review(run):
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert not any(f.startswith("soil_") for f in rep["accepted"])


def test_both_batches_carry_provenance_in_the_coco(run):
    """A batch that cannot say where its labels came from is indistinguishable
    from a hand-corrected export six months later."""
    mod, out = run
    mod.main()
    assert "PSEUDO-LABELS" in batch(out, "accept")["info"]["description"]
    assert "FOR CORRECTION" in batch(out, "review")["info"]["description"]


def test_the_images_are_placed_beside_the_coco_for_cvat(run):
    mod, out = run
    mod.main()
    ready = out / "accept" / "cvat_ready"
    assert ready.is_dir() and list(ready.glob("*.png"))


def test_a_spot_check_sample_is_always_written(run):
    """Minutes of work, and the only thing between a bad threshold and a
    poisoned dataset."""
    mod, out = run
    mod.main()
    assert (out / "spot_check").is_dir()
    assert list((out / "spot_check").glob("*.png"))


def test_it_refuses_to_pseudo_label_a_holdout(run, monkeypatch):
    """The model's own output in its own test set: every later round would
    score against what it already believes.

    Naming it in SESSIONS explicitly must NOT override this. An explicit list
    is how someone says 'score these' without remembering which are held out,
    which is exactly when the guardrail has to hold."""
    mod, out = run
    monkeypatch.setattr(mod, "HOLDOUT_TEST", ["vid_test"])
    chosen, skipped = mod.session_plan("pool", "ds", ["vid_test"], ["vid_test"])
    assert chosen == []
    assert "held out" in skipped[0][1]
    with pytest.raises(SystemExit, match="held out"):
        mod.main()


def test_no_predictions_and_no_frames_names_both(run, monkeypatch, tmp_path):
    """It generates predictions itself when it can, so the only unrecoverable
    case is having neither. The error has to name both, or the reader fixes the
    wrong one."""
    mod, out = run
    monkeypatch.setattr(mod, "IMAGES", str(tmp_path / "also-nothing"))
    monkeypatch.setattr(mod, "RUNS_ROOT", str(tmp_path / "no-runs"))
    with pytest.raises(SystemExit) as e:
        mod.main()
    msg = str(e.value)
    assert "no predictions at" in msg and "do not exist either" in msg


def test_a_missing_checkpoint_names_the_trainer(run, monkeypatch, tmp_path):
    """Predictions can be generated, but only if a model exists. Sending
    someone to the scorer when the real gap is an untrained round wastes the
    one thing this loop is short of."""
    mod, out = run
    images = tmp_path / "frames"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    monkeypatch.setattr(mod, "IMAGES", str(images))
    monkeypatch.setattr(mod, "RUNS_ROOT", str(tmp_path / "no-runs"))
    with pytest.raises(SystemExit, match="weeds_train"):
        mod.main()


def test_the_annotation_areas_are_per_instance(run):
    """They were indexed by the running annotation count, which paired an
    instance with another instance's area."""
    mod, out = run
    mod.main()
    doc = batch(out, "accept")
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        assert a["area"] == pytest.approx(w * h, rel=0.01)


def test_the_report_records_the_thresholds_it_used(run):
    """A batch whose threshold is not recorded cannot be compared with the next
    round's."""
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert rep["accept_threshold"] == pl.ACCEPT_SCORE
    assert rep["n_hand_corrected"] == 40
    assert rep["pseudo_budget"] == 80


# --------------------------------------------------------------------------- #
# The crash this cost a GPU pass to find
# --------------------------------------------------------------------------- #
def test_it_does_not_build_a_mask_per_instance(run):
    """bench_mixed._from_coco materialises one FULL-FRAME mask per annotation.
    That is right for a benchmark of a few frames and fatal here: a real weed
    session came back with 79 frames and 2,840 instances, which at 1242x2208
    bool is 7.8 GB held at once - and the process died AFTER the GPU pass had
    already been paid for.

    The scorer only ever needs the union per frame, so importing that helper is
    the bug itself."""
    src = (ROOT / "training" / "datasets" / "weeds_selftrain.py").read_text()
    assert "_from_coco" not in src, (
        "importing _from_coco rebuilds a full-frame mask per instance")
    assert "fillPoly" in src, "the union has to be rasterised in one array"


def test_a_frame_dense_with_instances_still_scores(run, tmp_path, monkeypatch):
    """Peak memory is not directly assertable, but the shape of the failure is:
    many instances on one frame. Under the old path this allocated one array
    per instance."""
    mod, out = run
    pred = Path(tmp_path / "runs" / "weeds_r0" / "look_vid_test_20260101_0000")
    doc = json.loads((pred / "instances_default.json").read_text())
    base = doc["annotations"][0]
    nxt = max(a["id"] for a in doc["annotations"]) + 1
    for i in range(300):
        a = dict(base)
        a["id"] = nxt + i
        doc["annotations"].append(a)
    (pred / "instances_default.json").write_text(json.dumps(doc))
    assert mod.main() == 0
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert rep["summary"]["n_frames"] >= 7


# --------------------------------------------------------------------------- #
# What CVAT needs, in the folder rather than in a chat message
# --------------------------------------------------------------------------- #
def test_each_batch_carries_the_cvat_label_schema(run):
    """Without the schema there is nothing for the COCO's category NAMES to
    match against, and CVAT matches by name."""
    mod, out = run
    mod.main()
    for which in ("accept", "review"):
        p = out / which / "weed_cvat_labels.json"
        assert p.is_file(), which
        assert json.loads(p.read_text()), which


def test_the_schema_is_the_full_ontology_not_the_models_classes(run):
    """The model predicts three classes. An annotator must be able to correct an
    instance INTO one it cannot predict - and without onion_plant, someone who
    finds crop in a weed-only frame is forced to call it a weed, which is the
    one error this project cannot afford."""
    mod, out = run
    mod.main()
    names = {l["name"] for l in
             json.loads((out / "accept" / "weed_cvat_labels.json").read_text())}
    predicted = {c["name"] for c in
                 json.loads((out / "accept" / "instances_default.json").read_text())
                 ["categories"]}
    assert {"wild_radish", "weed_cluster", "onion_plant"} <= names
    assert predicted < names, "the schema must be wider than what was predicted"


def test_each_batch_carries_its_own_instructions(run):
    """A batch opened three weeks later, or on another machine, has to say what
    to do with itself - including that the schema goes in BEFORE the import."""
    mod, out = run
    mod.main()
    txt = (out / "accept" / "README.txt").read_text()
    assert "Raw label editor" in txt and "COCO 1.0" in txt
    assert "Datumaro 1.0" in txt
    assert "duplicate" in txt          # the silent failure the order prevents


# --------------------------------------------------------------------------- #
# Using the round you meant to use
# --------------------------------------------------------------------------- #
def test_newest_trained_round_ignores_a_round_with_no_checkpoint(tmp_path):
    """An interrupted run leaves the directory and no weights. Calling that the
    latest model points everything downstream at a file that is not there."""
    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    for n, has_ckpt in ((0, True), (1, True), (2, False)):
        d = tmp_path / f"weeds_r{n}"
        d.mkdir()
        if has_ckpt:
            (d / "checkpoint_best_total.pth").write_bytes(b"x")
    assert mod.newest_trained_round(tmp_path) == 1


def test_no_runs_directory_is_not_an_error(tmp_path):
    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    assert mod.newest_trained_round(tmp_path / "nope") is None
    assert mod.stale_round_warning(tmp_path / "nope", 0) is None


def test_a_newer_trained_round_is_reported(tmp_path):
    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    for n in (0, 1, 2):
        d = tmp_path / f"weeds_r{n}"
        d.mkdir()
        (d / "checkpoint_best_total.pth").write_bytes(b"x")
    warn = mod.stale_round_warning(tmp_path, 0)
    assert warn and "weeds_r2" in warn and "ROUND = 2" in warn


def test_being_on_the_newest_round_is_silent(tmp_path):
    """Firing when nothing is wrong is how a warning gets ignored."""
    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    d = tmp_path / "weeds_r3"
    d.mkdir()
    (d / "checkpoint_best_total.pth").write_bytes(b"x")
    assert mod.stale_round_warning(tmp_path, 3) is None
    assert mod.stale_round_warning(tmp_path, 4) is None


def test_the_warning_allows_a_deliberate_older_round(tmp_path):
    """Comparing rounds means loading an old checkpoint on purpose, so this
    reports and never refuses."""
    import importlib
    mod = importlib.import_module("training.datasets.weeds_selftrain")
    for n in (0, 5):
        d = tmp_path / f"weeds_r{n}"
        d.mkdir()
        (d / "checkpoint_best_total.pth").write_bytes(b"x")
    assert "legitimate" in mod.stale_round_warning(tmp_path, 0)


# --------------------------------------------------------------------------- #
# Which sessions get scored, and why the others do not
# --------------------------------------------------------------------------- #
def _pool(tmp_path, *names, empty=()):
    root = tmp_path / "pool"
    for n in names:
        d = root / n / "rgb"
        d.mkdir(parents=True)
        if n not in empty:
            cv2.imwrite(str(d / "f0.png"), soil())
    return root


def _manifest(tmp_path, sessions):
    d = tmp_path / "ds"
    d.mkdir(exist_ok=True)
    (d / "seg_manifest.json").write_text(json.dumps({"sessions": sessions}))
    return d


def test_it_finds_every_session_in_the_pool(tmp_path):
    import training.datasets.weeds_selftrain as mod
    root = _pool(tmp_path, "vid1", "vid2", "vid3")
    assert mod.discover_sessions(root) == ["vid1", "vid2", "vid3"]


def test_an_aborted_recording_with_no_frames_is_not_a_session(tmp_path):
    """An empty rgb/ would produce an empty batch folder that looks exactly
    like a session someone forgot to correct."""
    import training.datasets.weeds_selftrain as mod
    root = _pool(tmp_path, "vid1", "vid2", empty=("vid2",))
    assert mod.discover_sessions(root) == ["vid1"]


def test_a_missing_pool_is_empty_not_an_exception(tmp_path):
    import training.datasets.weeds_selftrain as mod
    assert mod.discover_sessions(tmp_path / "nope") == []


def test_a_trained_session_is_excluded_from_the_pool(tmp_path):
    """THE case. Scoring a session the model trained on gives ceiling
    agreement, which reads as a great batch and teaches nothing - and nothing
    in the output would look wrong."""
    import training.datasets.weeds_selftrain as mod
    root = _pool(tmp_path, "vid1", "vid2")
    ds = _manifest(tmp_path, ["vid2"])
    chosen, skipped = mod.session_plan(root, ds, [])
    assert chosen == ["vid1"]
    assert skipped == [("vid2", skipped[0][1])]
    assert "already in the training build" in skipped[0][1]


def test_the_trained_list_comes_from_the_manifest_not_a_second_list(tmp_path):
    """A hand-kept list of trained sessions is a thing to forget to update,
    and forgetting it is silent."""
    import training.datasets.weeds_selftrain as mod
    assert mod.trained_sessions(_manifest(tmp_path, ["a", "b"])) == {"a", "b"}


def test_no_manifest_yet_excludes_nothing(tmp_path):
    """Before the first build there is no manifest, and that is an ordinary
    state - not a reason to refuse to score anything."""
    import training.datasets.weeds_selftrain as mod
    assert mod.trained_sessions(tmp_path / "nothing") == set()


def test_a_corrupt_manifest_does_not_take_the_run_down(tmp_path):
    import training.datasets.weeds_selftrain as mod
    d = tmp_path / "ds"
    d.mkdir()
    (d / "seg_manifest.json").write_text("{not json")
    assert mod.trained_sessions(d) == set()


def test_every_exclusion_carries_its_reason(tmp_path):
    """A run that silently scores three of seven sessions is indistinguishable
    from a run that found only three."""
    import training.datasets.weeds_selftrain as mod
    root = _pool(tmp_path, "vid1", "vid2", "vid3")
    chosen, skipped = mod.session_plan(root, _manifest(tmp_path, ["vid2"]),
                                       ["vid3"])
    assert chosen == ["vid1"]
    assert {s for s, _ in skipped} == {"vid2", "vid3"}
    assert all(why for _, why in skipped)


def test_naming_a_session_explicitly_does_not_override_the_holdout(tmp_path):
    import training.datasets.weeds_selftrain as mod
    root = _pool(tmp_path, "vid1")
    chosen, _ = mod.session_plan(root, _manifest(tmp_path, []), ["vid1"],
                                 ["vid1"])
    assert chosen == []


# --------------------------------------------------------------------------- #
# One folder per session, and the instructions that go with them
# --------------------------------------------------------------------------- #
def test_each_session_gets_its_own_batch_folder(run, tmp_path):
    """Pooling would be one less CVAT task and would break the round trip: the
    build takes one session folder per source, and the gap accounting is
    computed from session identity."""
    mod, out = run
    mod.main()
    assert (tmp_path / "out" / "vid_test" / "accept").is_dir()
    assert (tmp_path / "out" / "vid_test" / "review").is_dir()


def test_the_run_writes_a_pooled_report_beside_the_sessions(run, tmp_path):
    mod, out = run
    mod.main()
    doc = json.loads((tmp_path / "out" / "selftrain_report.json").read_text())
    assert [s["session"] for s in doc["sessions"]] == ["vid_test"]
    assert doc["infer_stride"] == mod.INFER_STRIDE


def test_next_steps_is_written_into_the_run_folder(run, tmp_path):
    """A batch outlives the terminal it was produced in."""
    mod, out = run
    mod.main()
    txt = (tmp_path / "out" / "NEXT_STEPS.txt").read_text()
    assert "vid_test" in txt
    assert "Raw" in txt and "COCO 1.0" in txt and "Datumaro 1.0" in txt


def test_next_steps_names_where_each_export_must_land(run, tmp_path):
    """The corrected frames have to go back to the drive they came from, or
    the build cannot resolve their images and the split logic loses session
    identity."""
    mod, out = run
    mod.main()
    txt = (tmp_path / "out" / "NEXT_STEPS.txt").read_text()
    assert "annotations" in txt and "default.json" in txt
    assert "WEED_SESSIONS" in txt
    assert f"ROUND = {mod.ROUND + 1}" in txt


def test_next_steps_says_to_change_the_provenance(run, tmp_path):
    """It stops being 'hand_corrected' the first time it stops being true, and
    every score computed later is read through that field."""
    mod, out = run
    mod.main()
    assert "mixed" in (tmp_path / "out" / "NEXT_STEPS.txt").read_text()


def test_pooling_past_the_budget_is_called_out():
    """The budget is enforced per session, so several sessions can pass it
    together. That is exactly how a mostly-pseudo dataset gets built without
    anyone deciding to build one.

    Tested directly rather than through a run: with one session the per-session
    budget already caps the write, so the pooled case is unreachable from a
    single-session fixture - which is the guardrail working, not a gap."""
    import training.datasets.weeds_selftrain as mod
    w = mod.pooled_budget_warning(120, 96, 48)
    assert w and "exceeds the budget of 96" in w
    assert "PER SESSION" in w


def test_a_pooled_total_within_budget_is_silent():
    import training.datasets.weeds_selftrain as mod
    assert mod.pooled_budget_warning(96, 96, 48) is None
    assert mod.pooled_budget_warning(5, 96, 48) is None


def test_no_budget_means_no_warning():
    """pseudo_budget returns 0 when there are no hand-corrected frames, and
    dividing by that opinion would be worse than staying quiet."""
    import training.datasets.weeds_selftrain as mod
    assert mod.pooled_budget_warning(50, 0, 0) is None


def test_the_runner_uses_the_written_total_for_the_budget():
    """Against the classified count this fired on a run that exported five
    frames against a budget of ninety-six - a false alarm on the very number it
    exists to protect."""
    import inspect
    import training.datasets.weeds_selftrain as mod
    src = inspect.getsource(mod.main)
    assert 'n_acc = sum(r["written"]["accept"]' in src
    assert "pooled_budget_warning(n_acc" in src


# --------------------------------------------------------------------------- #
# Reused predictions that do not match the settings asked for
# --------------------------------------------------------------------------- #
def _pred_coco(tmp_path, n):
    d = tmp_path / "look_x_20260101_0000"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "instances_default.json"
    p.write_text(json.dumps({
        "images": [{"id": i + 1, "file_name": f"f{i:03d}.png",
                    "height": 8, "width": 8} for i in range(n)],
        "annotations": [], "categories": []}))
    return p


def _frames(tmp_path, n):
    d = tmp_path / "sess" / "rgb"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"f{i:03d}.png"), soil((8, 8)))
    return tmp_path / "sess"


def test_reusing_a_stride_5_folder_at_stride_1_is_called_out(tmp_path):
    """THE CASE. INFER_STRIDE went 5 -> 1 to score a whole 393-frame drive, the
    run reused a folder built at stride 5, scored the same 79 frames and
    reported them without a word. The setting had no effect and the output
    looked entirely normal."""
    import training.datasets.weeds_selftrain as mod
    coco = _pred_coco(tmp_path, 79)
    imgs = _frames(tmp_path, 393)
    w = mod.reuse_mismatch_warning(coco, str(imgs), 0, 1)
    assert w and "cover 79 frame(s)" in w and "select 393" in w


def test_it_says_the_settings_had_no_effect(tmp_path):
    """Naming the discrepancy is not enough - the reader has to understand that
    the number they just changed did nothing."""
    import training.datasets.weeds_selftrain as mod
    w = mod.reuse_mismatch_warning(_pred_coco(tmp_path, 79),
                                   str(_frames(tmp_path, 393)), 0, 1)
    assert "NO EFFECT" in w


def test_the_warning_names_the_folder_to_delete(tmp_path):
    """A warning you cannot act on is noise. The fix is one folder."""
    import training.datasets.weeds_selftrain as mod
    coco = _pred_coco(tmp_path, 79)
    w = mod.reuse_mismatch_warning(coco, str(_frames(tmp_path, 393)), 0, 1)
    assert str(coco.parent) in w


def test_matching_predictions_are_silent(tmp_path):
    """Reuse is the point, and a warning on every correct reuse would train
    the reader to skip past it."""
    import training.datasets.weeds_selftrain as mod
    w = mod.reuse_mismatch_warning(_pred_coco(tmp_path, 40),
                                   str(_frames(tmp_path, 40)), 0, 1)
    assert w is None


def test_a_stride_that_matches_what_was_generated_is_silent(tmp_path):
    import training.datasets.weeds_selftrain as mod
    w = mod.reuse_mismatch_warning(_pred_coco(tmp_path, 20),
                                   str(_frames(tmp_path, 100)), 0, 5)
    assert w is None


def test_frames_added_to_the_session_since_are_caught_too(tmp_path):
    """It compares counts, not settings, so every cause is caught at once -
    including the one nobody thinks of."""
    import training.datasets.weeds_selftrain as mod
    w = mod.reuse_mismatch_warning(_pred_coco(tmp_path, 40),
                                   str(_frames(tmp_path, 55)), 0, 1)
    assert w and "cover 40" in w and "select 55" in w


def test_an_unreadable_coco_is_left_to_the_caller(tmp_path):
    import training.datasets.weeds_selftrain as mod
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert mod.reuse_mismatch_warning(p, str(_frames(tmp_path, 10)), 0, 1) is None


def test_missing_frames_are_not_a_mismatch(tmp_path):
    """The images being gone is a different failure with its own message."""
    import training.datasets.weeds_selftrain as mod
    w = mod.reuse_mismatch_warning(_pred_coco(tmp_path, 40),
                                   str(tmp_path / "nowhere"), 0, 1)
    assert w is None


def test_the_runner_checks_before_it_scores(tmp_path):
    """After scoring is too late: the frames are already chosen."""
    import inspect
    import training.datasets.weeds_selftrain as mod
    assert "reuse_mismatch_warning" in inspect.getsource(mod._predict)


# --------------------------------------------------------------------------- #
# The counts the run reports must be the counts in the folders
# --------------------------------------------------------------------------- #
def test_the_pooled_total_counts_files_not_classifications(run, tmp_path):
    """THE CASE. A real run classified 292 frames as accept and wrote 4, then
    reported '360 accepted' and warned about a budget of 96. Separation and the
    class cap sit between the two numbers and the gap is a factor of fifty."""
    mod, out = run
    mod.main()
    doc = json.loads((tmp_path / "out" / "selftrain_report.json").read_text())
    on_disk = len(list((out / "accept" / "cvat_ready").glob("*.png")))
    assert doc["written"]["accept"] == on_disk


def test_the_per_session_report_records_both_numbers(run):
    """Both are worth having - one says what the model thought, the other what
    you can upload - so the report keeps them side by side and named."""
    mod, out = run
    mod.main()
    rep = json.loads((out / "selftrain_report.json").read_text())
    assert "written" in rep and "summary" in rep
    assert rep["written"]["accept"] == len(rep["accepted"])


def test_next_steps_promises_what_the_folders_hold(run, tmp_path):
    """This document tells someone what to upload to CVAT. Promising 292 and
    handing over 4 is worse than saying nothing."""
    mod, out = run
    mod.main()
    txt = (tmp_path / "out" / "NEXT_STEPS.txt").read_text()
    rep = json.loads((out / "selftrain_report.json").read_text())
    n = rep["written"]["accept"]
    row = [l for l in txt.splitlines() if "vid_test" in l and "annotations" not in l]
    assert row and str(n) in row[0]


def test_next_steps_still_shows_how_many_were_scored(run, tmp_path):
    """Dropping it would hide that a 391-frame drive yielded four frames, which
    is the number that decides whether to lower MIN_FRAME_GAP."""
    mod, out = run
    mod.main()
    txt = (tmp_path / "out" / "NEXT_STEPS.txt").read_text()
    assert "scored" in txt


def test_the_budget_is_checked_against_what_was_written(run, tmp_path,
                                                        monkeypatch, capsys):
    """Checking the classified count fired on a run that exported five frames
    against a budget of ninety-six."""
    mod, out = run
    monkeypatch.setattr(mod, "N_HAND", 40)     # budget 80, way above any write
    mod.main()
    assert "exceeds the budget" not in capsys.readouterr().out


def test_the_class_cap_says_what_it_dropped(run, tmp_path, monkeypatch, capsys):
    """It has quietly removed a third of a batch - 6 frames in, 4 out - while
    the printed accept count two lines above still read 292."""
    mod, out = run
    monkeypatch.setattr(pl, "BALANCE_CAP_FRAC", 0.5)
    monkeypatch.setattr(mod, "MIN_FRAME_GAP", 0)   # keep enough to hit the cap
    mod.main()
    txt = capsys.readouterr().out
    assert "class balance" in txt and "per-class" in txt


def test_a_batch_within_the_cap_prints_no_balance_line(run, capsys):
    """A line on every run is a line nobody reads."""
    mod, out = run
    mod.main()
    assert "class balance" not in capsys.readouterr().out
