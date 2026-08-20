"""The per-dataset runners.

They exist because a single shared CONFIG, edited back and forth between an
onion build, a weed build and a mixed build, is how a stale DATASET_DIR reaches
a training run and how the same key ended up in one CONFIG twice. These tests
pin the properties that make the split worth having.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets import common as loc          # noqa: E402
from training.datasets import weeds                  # noqa: E402
from training.datasets import weeds_mine             # noqa: E402
from training.datasets import weeds_train            # noqa: E402
from training.make_dataset import CONFIG as BASE     # noqa: E402


# --------------------------------------------------------------------------- #
# The holdout, which is the one mistake that invalidates every later number
# --------------------------------------------------------------------------- #
def test_the_mining_holdout_matches_the_dataset_holdout():
    """Mining picks the frames the model finds HARDEST, which are exactly the
    frames it would most benefit from having seen. Mining a test session does
    not merely leak it - it leaks it in the most flattering direction. The two
    lists are separate settings in separate files, so nothing but this makes
    them agree."""
    assert (sorted(weeds_mine.CONFIG["HOLDOUT_SESSIONS"])
            == sorted(weeds.CONFIG["HOLDOUT_TEST_SESSIONS"]))


def test_the_seed_is_shared_so_rounds_stay_comparable():
    """A new seed re-draws every split, which makes round 3 incomparable with
    round 1 for a reason that has nothing to do with the model."""
    assert weeds.CONFIG["SEED"] == BASE["SEED"]


# --------------------------------------------------------------------------- #
# Each build describes itself honestly
# --------------------------------------------------------------------------- #
def test_the_weed_build_keeps_weeds_and_not_the_crop():
    """A weed-only drive has no crop in it, so an onion_plant instance is a
    mislabel worth seeing rather than training on."""
    from common.ontology import CROP_CLASS, WEED_CLASSES
    keep = weeds.CONFIG["KEEP_CLASSES"]
    assert CROP_CLASS not in keep
    assert set(keep) == set(WEED_CLASSES)


def test_the_weed_build_declares_hand_corrected_labels():
    """Unlike the onion build, these frames were corrected by a person - so
    their val and test scores measure performance rather than agreement with a
    prelabeler, and the manifest has to say so."""
    assert weeds.CONFIG["LABEL_PROVENANCE"] == "hand_corrected"


def test_the_runners_inherit_everything_they_do_not_override():
    """The point of a thin override: a fix to the shared defaults reaches every
    dataset without being copied into it."""
    for key in ("VERIFY_IMAGES", "GAP_FRAMES", "STRATIFY_BY_SCENE",
                "KEEP_EMPTY_FRAMES", "ALLOW_ERRORS"):
        assert key in weeds.CONFIG, key
    # And an override really overrides.
    assert weeds.CONFIG["OUT_DIR"] != BASE["OUT_DIR"]


def test_no_runner_reuses_another_runners_output_directory():
    """A shared OUT_DIR silently rebuilds one dataset over another."""
    outs = [weeds.CONFIG["OUT_DIR"], BASE["OUT_DIR"]]
    assert len(set(outs)) == len(outs)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_reads_the_dataset_the_build_writes():
    """The mismatch that has killed a run twice: DATASET_DIR pointing at a
    previous build's OUT_DIR."""
    assert weeds_train.CONFIG["DATASET_DIR"] == weeds.CONFIG["OUT_DIR"]


def test_mining_reads_the_same_dataset_it_will_grow():
    assert weeds_mine.CONFIG["DATASET_DIR"] == weeds.CONFIG["OUT_DIR"]


def test_each_round_gets_its_own_run_directory():
    """Reusing a run directory overwrites the checkpoint you would have
    compared against, and the comparison is the whole point of the loop."""
    a = dict(weeds_train.CONFIG)
    assert f"weeds_r{weeds_train.ROUND}" in a["RUN_DIR"]


def test_the_resolution_is_valid_for_the_variant():
    """rfdetr requires a multiple of patch_size*num_windows - 24 for
    medium/large. An invalid value fails deep inside the package rather than
    here."""
    assert weeds_train.CONFIG["VARIANT"] in ("medium", "large")
    assert weeds_train.CONFIG["RESOLUTION"] % 24 == 0


def test_the_effective_batch_stays_near_sixteen():
    c = weeds_train.CONFIG
    assert 8 <= c["BATCH"] * c["GRAD_ACCUM"] <= 32


def test_the_mining_confidence_is_below_a_deployment_threshold():
    """A spurious mask costs one delete; a missing one costs the annotator
    noticing an absence, which is far harder."""
    assert weeds_mine.CONFIG["CONF"] <= 0.3


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_the_data_root_is_overridable_by_environment(monkeypatch):
    """The annotation machine and the training machine hold the same data on
    different drives, and neither should have to edit the other's file."""
    import importlib
    monkeypatch.setenv("SEEWEED3D_DATA_ROOT", str(Path("/tmp/somewhere")))
    reloaded = importlib.reload(loc)
    assert reloaded.DATA_ROOT == Path("/tmp/somewhere")
    assert reloaded.campaign("X") == Path("/tmp/somewhere/X/sessions")
    monkeypatch.delenv("SEEWEED3D_DATA_ROOT")
    importlib.reload(loc)


def test_each_source_uses_one_path_for_both_roots():
    """The annotations live INSIDE the session folder, beside rgb/, so one path
    serves as both: DATUMARO_ROOT finds annotations/default.json under it, and
    IMAGES_ROOT resolves the frames. Two different paths here is a typo, not a
    configuration."""
    assert weeds.CONFIG["SOURCES"], "at least one source is required"
    for src in weeds.CONFIG["SOURCES"]:
        assert src["IMAGES_ROOT"] == src["DATUMARO_ROOT"]


def test_a_single_session_build_cannot_pin_a_holdout():
    """Holding out your only session leaves nothing to train on. The build
    falls back to frame blocks within the session and says so - an upper bound,
    not generalisation, and a known limitation rather than an oversight."""
    if len(weeds.WEED_SESSIONS) < 2:
        assert weeds.HOLDOUT_TEST == [], (
            "with one session a holdout would empty the training set")
