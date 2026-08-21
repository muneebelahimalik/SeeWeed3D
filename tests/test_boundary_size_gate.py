"""Edge snapping, and the size gate that keeps it away from what already works.

THE OBSERVATION THIS IS BUILT ON
--------------------------------
Field judgement on the SAM weed prelabels: BIG weeds already have correct
boundaries and need no correction at all, while SMALL ones do not.

That asymmetry is the diagnosis. SAM decodes masks on a fixed grid spread over
the whole frame, so a big rosette gets many cells across it and a seedling gets
a handful - and a boundary error of a fixed number of pixels is a large FRACTION
of a small plant and a negligible one of a large plant.

Two consequences, and this module pins both:

  1. A fixed-width refinement band lands on exactly that asymmetry, so it helps
     where the error is without being asked to.
  2. It must not touch the big instances anyway. They are the half that already
     works, and #29 is the case in this project where a boundary pipeline that
     improved every number produced worse masks in the field.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.prelabel_weeds_sam3 import CONFIG, refine_boundary  # noqa: E402


def plant(size, halo=2):
    """A square plant, a SAM mask that overshoots it by `halo` px on every side,
    and an ExG score that is exactly right. Returns (truth, sam, score)."""
    n = size + 2 * halo + 8
    truth = np.zeros((n, n), bool)
    truth[4 + halo:4 + halo + size, 4 + halo:4 + halo + size] = True
    sam = cv2.dilate(truth.astype(np.uint8),
                     np.ones((2 * halo + 1,) * 2, np.uint8)).astype(bool)
    return truth, sam, truth.astype(np.float32)


def cfg(**kw):
    return dict(CONFIG, BOUNDARY_REFINE_BAND_PX=2,
                BOUNDARY_REFINE_VEG_MIN=0.5, **kw)


def err(mask, truth):
    return int((mask ^ truth).sum()) / int(truth.sum())


# --------------------------------------------------------------------------- #
# The asymmetry itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size,floor", [(20, 0.35), (30, 0.20), (120, 0.10)])
def test_a_fixed_overshoot_hurts_small_plants_far_more(size, floor):
    """Why big weeds look right and small ones do not, with no model involved:
    the same 2 px of overshoot is 44% of a 20 px plant and 7% of a 120 px one."""
    truth, sam, _ = plant(size)
    e = err(sam, truth)
    assert e >= floor if size <= 30 else e <= floor


def test_the_error_falls_monotonically_as_the_plant_grows():
    errs = [err(plant(s)[1], plant(s)[0]) for s in (20, 30, 60, 120)]
    assert errs == sorted(errs, reverse=True), errs


# --------------------------------------------------------------------------- #
# Snapping fixes the small case
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [20, 30])
def test_snapping_recovers_a_small_plant_boundary(size):
    """With a clean vegetation score there is nothing left to recover. Real ExG
    is noisier than this - the test pins the MECHANISM, not a field result."""
    truth, sam, score = plant(size)
    out = refine_boundary(sam, score, cfg(BOUNDARY_REFINE_MAX_AREA_PX=1500))
    assert err(out, truth) < err(sam, truth)


def test_snapping_never_eats_into_the_interior():
    """The refinement re-decides a RING. An instance that lost interior tissue
    would be a worse failure than the bloat it is fixing."""
    truth, sam, score = plant(24)
    out = refine_boundary(sam, score, cfg(BOUNDARY_REFINE_MAX_AREA_PX=1500))
    core = cv2.erode(sam.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert (out | ~core).all(), "refinement removed core pixels"


# --------------------------------------------------------------------------- #
# The gate leaves what works alone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [60, 120])
def test_a_big_instance_passes_through_untouched(size):
    """Byte-for-byte, not merely 'close'. The guarantee is the point: field
    judgement is that these are already correct."""
    _, sam, score = plant(size)
    out = refine_boundary(sam, score, cfg(BOUNDARY_REFINE_MAX_AREA_PX=1500))
    assert np.array_equal(out, sam)


def test_the_gate_boundary_is_the_projects_own_small_weed_threshold():
    """eval_seg reports 'small-weed recall (<=1500 px)'. The prelabeler and the
    metric must mean the same thing by 'small', or the two disagree about which
    instances the work was aimed at."""
    assert CONFIG["BOUNDARY_REFINE_MAX_AREA_PX"] == 1500


def test_zero_disables_the_gate_and_refines_everything():
    _, sam, score = plant(120)
    out = refine_boundary(sam, score, cfg(BOUNDARY_REFINE_MAX_AREA_PX=0))
    assert not np.array_equal(out, sam)


def test_snapping_is_never_armed_without_the_gate():
    """Snapping shipped off and was turned on deliberately after field
    judgement that small weeds need help and big ones do not.

    What must never happen is snapping ON with the gate OFF: that is free rein
    over the boundaries that are already right, which is #29's failure exactly.
    So this pins the PAIR rather than either value - it stays true whether
    snapping is on or off, and fails only on the combination that is unsafe."""
    if CONFIG["BOUNDARY_REFINE_BAND_PX"]:
        assert CONFIG["BOUNDARY_REFINE_MAX_AREA_PX"] > 0


def test_the_band_stays_narrow():
    """The band is re-decided from a colour index, and colour indices
    false-positive on green-tinted mineral (#24). A wide band turns that from a
    rim into a takeover - and on a small weed the band IS most of the plant."""
    assert CONFIG["BOUNDARY_REFINE_BAND_PX"] <= 3


def test_the_gate_does_nothing_while_snapping_is_off():
    _, sam, score = plant(20)
    off = dict(CONFIG, BOUNDARY_REFINE_BAND_PX=0)
    assert np.array_equal(refine_boundary(sam, score, off), sam)


def test_a_plant_too_thin_to_erode_is_left_alone():
    """A cotyledon a few px across has no core after erosion, and refining from
    an empty core could delete it outright."""
    m = np.zeros((30, 30), bool)
    m[14:16, 10:20] = True                       # 2 px tall
    out = refine_boundary(m, np.zeros((30, 30), np.float32),
                          cfg(BOUNDARY_REFINE_MAX_AREA_PX=1500))
    assert out.any(), "a thin instance must survive refinement"


def test_refinement_cannot_bridge_to_a_neighbouring_plant():
    """Added pixels must stay connected to the original core, or snapping could
    merge two weeds into one - which costs an individual LEP."""
    score = np.zeros((40, 60), np.float32)
    score[10:30, 5:20] = 1.0                     # this plant
    score[10:30, 24:40] = 1.0                    # a neighbour, 4 px away
    m = np.zeros((40, 60), bool)
    m[12:28, 7:18] = True
    out = refine_boundary(m, score, cfg(BOUNDARY_REFINE_MAX_AREA_PX=1500))
    n, _ = cv2.connectedComponents(out.astype(np.uint8), 8)
    assert n == 2, "refinement must remain one connected instance"
    assert not out[:, 24:].any(), "refinement reached the neighbour"
