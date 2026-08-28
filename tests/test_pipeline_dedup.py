"""Two defects in the deployed pipeline, both invisible in its output.

The segmentation path deduplicated; the FULL pipeline did not. On a real weed
session 14% of detections came back as duplicates at IoU >= 0.85 - RF-DETR is a
set-prediction model and nothing makes two queries that found the same plant
agree about what it is. Left in, each copy becomes its own WeedTarget with its
own LEP and its own 3D point.

And predict_images called the segmenter a SECOND time to draw the overlay,
after the pipeline had already segmented. That doubled the cost of the
expensive stage and drew the picture from a different inference than the
record - and it skipped whatever the pipeline had done to its detections.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from common.ontology import CLASSES, CROP_CLASS      # noqa: E402
from perception import segmenter as seg              # noqa: E402
from perception.pipeline import InferencePipeline    # noqa: E402
from training.config import PipelineConfig           # noqa: E402

HW = (80, 80)


def det(specs):
    """specs: [(class_name, (y0, y1, x0, x1), score)]"""
    masks, cls, sc = [], [], []
    for name, (y0, y1, x0, x1), s in specs:
        m = np.zeros(HW, bool)
        m[y0:y1, x0:x1] = True
        masks.append(m)
        cls.append(CLASSES.index(name))
        sc.append(s)
    n = len(specs)
    boxes = np.array([[x0, y0, x1 - x0, y1 - y0]
                      for _, (y0, y1, x0, x1), _ in specs], float) \
        if n else np.zeros((0, 4))
    return seg.Detections(
        np.asarray(masks, bool) if n else np.zeros((0,) + HW, bool),
        boxes, np.asarray(cls, int), np.asarray(sc, float),
        HW[1], HW[0], names=list(CLASSES))


class Counting:
    """A segmenter that returns fixed detections and counts its calls."""

    classes = list(CLASSES)

    def __init__(self, d):
        self.d, self.calls = d, 0

    def load(self):
        return self

    def __call__(self, _bgr):
        self.calls += 1
        return self.d


def pipe(d, **cfgkw):
    cfg = PipelineConfig()
    for k, v in cfgkw.items():
        setattr(cfg, k, v)
    return InferencePipeline(Counting(d), cfg), cfg


def frame():
    return np.full(HW + (3,), 60, np.uint8)


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def test_one_plant_found_twice_becomes_one_target():
    """THE CASE. Two queries, one plant, two labels - and downstream each copy
    is a separate LEP and a separate 3D point, so the laser is aimed twice at
    one weed."""
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.9),
                     ("cutleaf_evening_primrose", (10, 40, 10, 40), 0.7)]))
    res, d = p.run_with_detections(frame())
    assert len(d) == 1
    assert len(res.targets) == 1


def test_the_higher_scoring_copy_is_the_one_kept():
    p, _ = pipe(det([("cutleaf_evening_primrose", (10, 40, 10, 40), 0.6),
                     ("grass_weed", (10, 40, 10, 40), 0.95)]))
    res, _ = p.run_with_detections(frame())
    assert res.targets[0].class_name == "grass_weed"


def test_two_different_plants_both_survive():
    """Suppression that removes real detections is worse than none."""
    p, _ = pipe(det([("grass_weed", (5, 25, 5, 25), 0.9),
                     ("grass_weed", (50, 70, 50, 70), 0.9)]))
    res, d = p.run_with_detections(frame())
    assert len(d) == 2 and len(res.targets) == 2


def test_dedup_runs_before_the_onion_safety_mask_is_built():
    """A duplicate labelled onion puts those pixels in the crop-safety mask
    while its twin is on the target list - the same plant both protected and
    fired at. Order is the whole fix."""
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.95),
                     (CROP_CLASS, (10, 40, 10, 40), 0.5)]))
    res, d = p.run_with_detections(frame())
    assert CROP_CLASS not in [d.class_name(i) for i in range(len(d))]
    assert res.onion_area_px == 0, \
        "the suppressed copy must not still be protecting its own twin"


def test_what_was_suppressed_is_recoverable():
    """A silent drop is indistinguishable from a model that found less."""
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.9),
                     ("other_weed", (10, 40, 10, 40), 0.7)]))
    p.run_with_detections(frame())
    assert len(p.last_duplicates) == 1


def test_suppression_can_be_turned_off():
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.9),
                     ("other_weed", (10, 40, 10, 40), 0.7)]), dedup_iou=0)
    _, d = p.run_with_detections(frame())
    assert len(d) == 2


def test_an_empty_frame_still_works():
    p, _ = pipe(det([]))
    res, d = p.run_with_detections(frame())
    assert len(d) == 0 and res.targets == []


def test_the_config_carries_the_threshold():
    """A second, independently chosen IoU for the same physical question is how
    the scorer and the pipeline come to disagree about one frame."""
    assert hasattr(PipelineConfig(), "dedup_iou")


# --------------------------------------------------------------------------- #
# Segmenting once
# --------------------------------------------------------------------------- #
def test_one_frame_costs_one_segmentation():
    """Stage A is the expensive stage; running it twice per frame doubled the
    cost of the whole pipeline."""
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.9)]))
    p.run_with_detections(frame())
    assert p.segmenter.calls == 1


def test_run_returns_just_the_result():
    """The old signature has callers; it must keep working."""
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.9)]))
    res = p.run(frame())
    assert hasattr(res, "targets") and not isinstance(res, tuple)


def test_the_detections_returned_are_the_ones_the_result_was_built_from():
    """This is what makes an overlay honest: the picture and the record are the
    same inference, so they cannot disagree about a frame."""
    p, _ = pipe(det([("grass_weed", (10, 40, 10, 40), 0.9),
                     ("other_weed", (10, 40, 10, 40), 0.7)]))
    res, d = p.run_with_detections(frame())
    assert len(d) == res.n_instances


def test_an_empty_frame_also_returns_its_detections():
    """The early return skipped them, so a frame with nothing found used to
    take a different path out."""
    p, _ = pipe(det([]))
    out = p.run_with_detections(frame())
    assert isinstance(out, tuple) and len(out) == 2


def test_predict_images_does_not_segment_a_second_time():
    """Structural: the second call was one line and easy to reintroduce."""
    import inspect
    from perception import predict_images as pi
    # Comments stripped: the comment explaining why the second call is gone
    # naturally mentions it, and matching that would make this pass forever.
    src = "\n".join(l for l in inspect.getsource(pi._run_full).splitlines()
                    if not l.strip().startswith("#"))
    assert "run_with_detections" in src
    assert "pipe.segmenter(" not in src


# --------------------------------------------------------------------------- #
# The overlay has to SHOW the thing the pipeline was run for
# --------------------------------------------------------------------------- #
def _target(uv, status, xyz=None):
    from perception.schema import WeedTarget
    return WeedTarget(lep_uv=list(uv) if uv else None,
                      safety_status=status, xyz_mm=xyz)


def test_a_full_mode_overlay_marks_the_growth_points():
    """Without this a full-mode overlay was pixel-identical to a segmentation
    one: the LEP and the 3D point lived only in the JSON, so the one thing a
    person runs the whole pipeline to LOOK at was the one thing the picture did
    not show."""
    from perception.predict_images import draw
    from perception.schema import STATUS_CANDIDATE
    d = det([("grass_weed", (10, 40, 10, 40), 0.9)])
    plain = draw(frame(), d, set(), show_legend=False)
    marked = draw(frame(), d, set(), show_legend=False,
                  targets=[_target((25, 25), STATUS_CANDIDATE, [1, 2, 300])])
    assert not np.array_equal(plain, marked)


def test_the_marker_colour_carries_the_safety_verdict():
    """Whether the laser would fire is the question the pipeline exists to
    answer, and a refused point looks identical to an approved one unless the
    picture says which."""
    from perception.predict_images import draw
    from perception.schema import STATUS_ABSTAIN, STATUS_CANDIDATE
    d = det([("grass_weed", (10, 40, 10, 40), 0.9)])
    ok = draw(frame(), d, set(), show_legend=False, labels="none",
              targets=[_target((25, 25), STATUS_CANDIDATE)])
    no = draw(frame(), d, set(), show_legend=False, labels="none",
              targets=[_target((25, 25), STATUS_ABSTAIN)])
    assert not np.array_equal(ok, no)


def test_a_target_with_no_lep_is_skipped_not_drawn_at_the_origin():
    """lep_uv is None when the estimator abstained. Falling back to (0,0) would
    put a marker in the corner of every such frame."""
    from perception.predict_images import draw
    from perception.schema import STATUS_ABSTAIN
    d = det([("grass_weed", (10, 40, 10, 40), 0.9)])
    plain = draw(frame(), d, set(), show_legend=False, labels="none")
    with_none = draw(frame(), d, set(), show_legend=False, labels="none",
                     targets=[_target(None, STATUS_ABSTAIN)])
    assert np.array_equal(plain, with_none)


def test_segmentation_mode_is_unchanged():
    """targets defaults to None, so nothing about the 2D path moves."""
    from perception.predict_images import draw
    d = det([("grass_weed", (10, 40, 10, 40), 0.9)])
    assert np.array_equal(draw(frame(), d, set(), show_legend=False),
                          draw(frame(), d, set(), show_legend=False,
                               targets=None))


def test_run_full_returns_its_targets_to_the_caller():
    """Structural: the overlay can only draw what _run_full hands back, and it
    used to hand back three values."""
    import inspect
    from perception import predict_images as pi
    src = inspect.getsource(pi._run_full)
    assert "return rec, det, conflicts, res.targets" in src
