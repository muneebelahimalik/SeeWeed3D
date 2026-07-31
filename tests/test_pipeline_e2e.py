"""End-to-end RGB-D inference with a MOCKED segmenter and synthetic depth.

Exercises the whole runtime path - crop-safety union, batched ROI LEP, 3D
localisation, safety abstention, structured output - without a GPU, trained
weights, or a real dataset."""
import numpy as np
import pytest

from conftest import load_script

seg_mod = load_script("perception/segmenter.py")
pipe_mod = load_script("perception/pipeline.py")
schema = load_script("perception/schema.py")
sf = load_script("perception/safety.py")
cfgm = load_script("training/config.py")

from common.ontology import CLASSES  # noqa: E402

K = (700.0, 700.0, 320.0, 240.0)


def _scene(w=320, h=240):
    """Soil frame with one weed and one onion, plus consistent depth."""
    bgr = np.full((h, w, 3), (70, 60, 55), np.uint8)
    weed = np.zeros((h, w), bool)
    weed[80:130, 40:90] = True
    onion = np.zeros((h, w), bool)
    onion[80:140, 200:260] = True
    bgr[weed] = (40, 170, 60)
    bgr[onion] = (60, 190, 90)

    depth = np.full((h, w), 1000.0, np.float32)
    depth[weed] = 950.0
    depth[onion] = 940.0
    valid = np.ones((h, w), bool)
    return bgr, weed, onion, depth, valid


def _detections(masks, classes, scores, w, h):
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([float(xs.min()), float(ys.min()),
                      float(xs.max() - xs.min() + 1),
                      float(ys.max() - ys.min() + 1)])
    return seg_mod.Detections(np.stack(masks), np.array(boxes, float),
                              np.array(classes, int), np.array(scores, float),
                              w, h)


def _pipeline(det):
    cfg = cfgm.PipelineConfig()
    return pipe_mod.InferencePipeline(seg_mod.MockSegmenter(det), cfg), cfg


# --------------------------------------------------------------------------- #
def test_onion_masks_are_unioned_into_one_safety_mask():
    """Crop safety is a FRAME-level property: deciding it per weed would let an
    onion detected late fail to protect a weed decided early."""
    bgr, weed, onion, _, _ = _scene()
    onion2 = np.zeros_like(onion)
    onion2[10:40, 10:40] = True
    det = _detections([weed, onion, onion2],
                      [CLASSES.index("wild_radish"),
                       CLASSES.index("onion_plant"),
                       CLASSES.index("onion_plant")],
                      [0.9, 0.9, 0.8], bgr.shape[1], bgr.shape[0])
    union = det.onion_safety_mask()
    assert union is not None
    assert union.sum() == int(onion.sum() + onion2.sum())
    assert not union[weed].any()                 # the weed is not in the union
    assert det.weed_indices() == [0]


def test_full_inference_produces_the_structured_schema():
    bgr, weed, onion, depth, valid = _scene()
    det = _detections([weed, onion],
                      [CLASSES.index("wild_radish"), CLASSES.index("onion_plant")],
                      [0.92, 0.88], bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    res = pipe.run(bgr, depth, valid, K, session_id="sess01", frame_id="f0001")

    # Structural check, not isinstance: conftest.load_script imports the module
    # under a synthetic name while the pipeline imports it normally, so the two
    # class objects differ even though they come from the same file.
    assert type(res).__name__ == "FrameResult"
    assert res.session_id == "sess01" and res.frame_id == "f0001"
    assert res.n_instances == 2
    assert len(res.targets) == 1                 # the onion is not a target
    t = res.targets[0]
    assert t.class_name == "wild_radish"
    assert t.lep_uv is not None and len(t.lep_uv) == 2
    assert t.used_depth is True
    assert t.safety_status in (schema.STATUS_CANDIDATE, schema.STATUS_ABSTAIN)
    assert "total" in res.timings_ms and res.timings_ms["n_weed_rois"] == 1
    d = res.to_dict()
    assert "n_candidates" in d and "reason_counts" in d


def test_weed_on_top_of_an_onion_abstains_with_the_conflict_reason():
    """The crop-safety veto, end to end."""
    bgr, weed, onion, depth, valid = _scene()
    overlapping = np.zeros_like(weed)
    overlapping[85:135, 205:255] = True          # sits inside the onion
    det = _detections([overlapping, onion],
                      [CLASSES.index("wild_radish"), CLASSES.index("onion_plant")],
                      [0.95, 0.95], bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    res = pipe.run(bgr, depth, valid, K)
    t = res.targets[0]
    assert t.abstained
    assert sf.R_ONION_CONFLICT in t.rejection_reasons
    assert res.candidates == []


def test_weed_cluster_abstains_end_to_end():
    bgr, weed, onion, depth, valid = _scene()
    det = _detections([weed], [CLASSES.index("weed_cluster")], [0.9],
                      bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    res = pipe.run(bgr, depth, valid, K)
    assert sf.R_CLUSTER in res.targets[0].rejection_reasons
    assert res.reason_counts()[sf.R_CLUSTER] == 1


def test_pipeline_runs_without_depth_and_marks_it():
    """RGB-only operation is a hard requirement - field depth is often
    unavailable."""
    bgr, weed, onion, _, _ = _scene()
    det = _detections([weed], [CLASSES.index("other_weed")], [0.9],
                      bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    res = pipe.run(bgr, None, None, None)
    t = res.targets[0]
    assert t.used_depth is False
    assert t.xyz_mm is None
    assert t.lep_uv is not None                  # a 2D LEP is still produced


def test_frame_with_no_weeds_returns_cleanly():
    bgr, weed, onion, depth, valid = _scene()
    det = _detections([onion], [CLASSES.index("onion_plant")], [0.9],
                      bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    res = pipe.run(bgr, depth, valid, K)
    assert res.targets == [] and res.candidates == []
    assert res.onion_area_px > 0


def test_fallback_estimator_is_used_when_no_learned_model_is_loaded():
    """The hand-engineered perception/lep.py estimator keeps the system
    runnable - and comparable - before any learned model exists. It is also the
    baseline the learned model must beat."""
    bgr, weed, onion, depth, valid = _scene()
    det = _detections([weed], [CLASSES.index("wild_radish")], [0.9],
                      bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    assert pipe.lep_model is None
    res = pipe.run(bgr, depth, valid, K)
    t = res.targets[0]
    assert t.lep_uv is not None
    # The LEP must land on its own plant.
    u, v = int(round(t.lep_uv[0])), int(round(t.lep_uv[1]))
    assert weed[v, u], "fallback LEP left its own mask"


def test_learned_model_path_runs_batched_over_all_weeds():
    """Deployment must not loop the ROI model per weed; one batch per frame."""
    torch = pytest.importorskip("torch")
    net_mod = load_script("training/lep_roinet.py")

    bgr, weed, onion, depth, valid = _scene()
    w2 = np.zeros_like(weed)
    w2[150:200, 120:170] = True
    w3 = np.zeros_like(weed)
    w3[30:70, 120:165] = True
    det = _detections([weed, w2, w3, onion],
                      [CLASSES.index("wild_radish"), CLASSES.index("grass_weed"),
                       CLASSES.index("other_weed"), CLASSES.index("onion_plant")],
                      [0.9, 0.85, 0.8, 0.9], bgr.shape[1], bgr.shape[0])

    cfg = cfgm.PipelineConfig()
    cfg.model.width = 8
    model = net_mod.build_model(cfg.model).eval()

    calls = {"n": 0}
    original = model.predict

    def counting_predict(rgb, geom=None):
        calls["n"] += 1
        assert rgb.shape[0] == 3, "all weed ROIs must arrive in ONE batch"
        return original(rgb, geom)

    model.predict = counting_predict
    pipe = pipe_mod.InferencePipeline(seg_mod.MockSegmenter(det), cfg,
                                      lep_model=model)
    res = pipe.run(bgr, depth, valid, K, session_id="s", frame_id="f")

    assert calls["n"] == 1, "the ROI model was called more than once per frame"
    assert len(res.targets) == 3
    for t in res.targets:
        assert t.visibility in ("visible", "partially_occluded_inferable",
                                "not_visible")
        assert t.visibility_probs and len(t.visibility_probs) == 3
        assert t.targetability_probs and len(t.targetability_probs) == 3
        assert t.lep_sigma_px >= 0.0


def test_every_target_records_a_machine_readable_verdict():
    """Nothing may be silently dropped: a target is either a candidate or has
    at least one explicit reason."""
    bgr, weed, onion, depth, valid = _scene()
    det = _detections([weed, onion],
                      [CLASSES.index("wild_radish"), CLASSES.index("onion_plant")],
                      [0.3, 0.9], bgr.shape[1], bgr.shape[0])
    pipe, _ = _pipeline(det)
    res = pipe.run(bgr, depth, valid, K)
    for t in res.targets:
        if t.safety_status == schema.STATUS_CANDIDATE:
            assert t.rejection_reasons == []
        else:
            assert t.rejection_reasons
            for r in t.rejection_reasons:
                assert r in sf.ALL_REJECTION_REASONS
