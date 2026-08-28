"""Boundary quality in the MIXED prelabeler.

The two prelabelers diverged. The weed one grew edge refinement and
anti-aliasing; the mixed one grew vegetation bridging, seeding and splitting.
Neither inherited the other's work, so running "the improved prelabeler" on a
mixed scene ran a pipeline without the boundary half of the improvements.

It matters MORE here than in a weed-only frame. Onion leaves are thin tubes a
few pixels wide crossing everything, so a boundary that bleeds two pixels does
not merely blur an edge - it swallows tissue belonging to the crop, and the crop
is the thing that must not be hit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from annotation import prelabel_mixed_sam3 as mx     # noqa: E402
from annotation import prelabel_weeds_sam3 as wd     # noqa: E402


def cfg(**kw):
    c = dict(mx.CONFIG)
    c.update(kw)
    return c


def disc(shape=(60, 60), c=(30, 30), r=10):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - c[0]) ** 2 + (xx - c[1]) ** 2) <= r * r


# --------------------------------------------------------------------------- #
# The functions are shared, not copied
# --------------------------------------------------------------------------- #
def test_the_refinement_is_the_weed_prelabelers_own():
    """A second implementation tuned separately drifts from the first, and
    these thresholds were chosen against real ground."""
    assert mx.refine_boundary is wd.refine_boundary
    assert mx.smooth_boundary is wd.smooth_boundary


def test_the_thresholds_match_the_weed_build():
    """Two numbers for one physical question is how two stages come to disagree
    about what a boundary is."""
    for k in ("BOUNDARY_REFINE_BAND_PX", "BOUNDARY_REFINE_VEG_MIN",
              "BOUNDARY_REFINE_MAX_AREA_PX"):
        assert mx.CONFIG[k] == wd.CONFIG[k], k


def test_the_small_weed_definition_is_the_projects_own():
    """1500 px is eval_seg's definition of a small weed, so the prelabeler and
    the metric agree on what 'small' means."""
    assert mx.CONFIG["BOUNDARY_REFINE_MAX_AREA_PX"] == 1500


# --------------------------------------------------------------------------- #
# What it does to an instance
# --------------------------------------------------------------------------- #
def test_the_edge_moves_onto_the_image_evidence():
    """THE POINT. SAM decides WHAT the object is; the image decides exactly
    where it ends."""
    m = disc(r=8)
    score = disc(r=10).astype(np.float32)      # the plant is really bigger
    out = mx.clean_instance(m, cfg(MIN_INSTANCE_AREA_PX=1), score)
    assert int(out.sum()) > int(m.sum())


def test_a_big_instance_is_left_alone():
    """The size gate is the whole reason this is usable: big plants are the half
    that already works, and #29 is what happens without one."""
    m = disc(shape=(200, 200), c=(100, 100), r=40)     # ~5000 px, over the cap
    score = np.ones((200, 200), np.float32)
    out = mx.clean_instance(m, cfg(MIN_INSTANCE_AREA_PX=1), score)
    assert int(out.sum()) == int(m.sum())


def test_refinement_cannot_bridge_to_a_neighbouring_plant():
    """An onion leaf passing within the band must not be absorbed into a weed.
    In a mixed scene that is not a cosmetic error - it is crop tissue labelled
    as a target."""
    m = disc(c=(30, 20), r=8)
    score = np.zeros((60, 60), np.float32)
    score[disc(c=(30, 20), r=9)] = 1.0
    score[:, 40:44] = 1.0                       # a separate strip, not touching
    out = mx.clean_instance(m, cfg(MIN_INSTANCE_AREA_PX=1), score)
    assert not out[:, 40:44].any()


def test_nothing_happens_without_a_veg_score():
    """The score is what the refinement decides on. Without one the only honest
    thing is to leave the mask as the watershed produced it."""
    m = disc(r=8)
    out = mx.clean_instance(m, cfg(MIN_INSTANCE_AREA_PX=1), None)
    assert int(out.sum()) == int(m.sum())


def test_turning_the_band_off_restores_the_old_behaviour():
    """Existing configurations must be unaffected until someone opts in."""
    m = disc(r=8)
    score = disc(r=10).astype(np.float32)
    out = mx.clean_instance(m, cfg(MIN_INSTANCE_AREA_PX=1,
                                   BOUNDARY_REFINE_BAND_PX=0), score)
    assert int(out.sum()) == int(m.sum())


def test_the_area_floor_is_applied_after_the_boundary_work():
    """An instance has to be measured at the size it will be EXPORTED at, or
    the floor is checked against a shape nobody ever sees."""
    import ast
    import inspect
    # The BODY only. The docstring names these steps in a different order than
    # the code runs them, and matching prose instead of code is how a structural
    # test comes to assert nothing - it bit once already in test_pipeline_dedup.
    fn = ast.parse(inspect.getsource(mx.clean_instance)).body[0]
    body = [ast.unparse(n) for n in fn.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    src = "\n".join(body)
    assert src.index("refine_boundary") < src.index("MIN_INSTANCE_AREA_PX")
    assert src.index("smooth_boundary") < src.index("drop_fragments"), \
        "a speck the refinement created must still be caught"


def test_a_frame_passes_its_score_through():
    """Structural: clean_instance can only refine with what analyze_frame hands
    it, and it used to be called with two arguments."""
    import inspect
    assert "clean_instance(m, cfg, score)" in inspect.getsource(mx.analyze_frame)
