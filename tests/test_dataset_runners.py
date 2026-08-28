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


def test_every_configured_path_is_absolute():
    """A relative path here silently writes beside the working directory, and
    the working directory is wherever the terminal happened to be. It bit once
    already: Path(<windows path>).parent evaluated on posix returns '.', so a
    batch destined for E:\\...\\batches became './batches'."""
    import ntpath
    paths = [weeds.CONFIG["OUT_DIR"],
             weeds_train.CONFIG["DATASET_DIR"], weeds_train.CONFIG["RUN_DIR"],
             weeds_mine.CONFIG["DATASET_DIR"], weeds_mine.CONFIG["OUT_DIR"],
             weeds_mine.CONFIG["CHECKPOINT"], weeds_mine.CONFIG["SESSIONS_ROOT"]]
    paths += [s["DATUMARO_ROOT"] for s in weeds.CONFIG["SOURCES"]]
    for p in paths:
        # Checked as a WINDOWS path regardless of the host running the tests:
        # these configs name drives, and posixpath calls "E:\\x" relative.
        assert ntpath.isabs(p), f"not an absolute path: {p!r}"


# --------------------------------------------------------------------------- #
# Looking at a session the model never saw
# --------------------------------------------------------------------------- #
def test_looking_and_mining_read_the_same_pool():
    """Pointing these at different campaigns is how you inspect one session and
    mine another without noticing - and then conclude the model transfers based
    on frames from the wrong drive."""
    from training.datasets import weeds_look
    assert weeds_mine.CONFIG["SESSIONS_ROOT"] == weeds.WEED_POOL_ROOT
    assert weeds_look.CONFIG["IMAGES"].startswith(weeds.WEED_POOL_ROOT)


def test_the_pool_still_holds_something_untrained():
    """A pool containing ONLY sessions you trained on has nothing new to find,
    and every frame it returns is one the model has already seen.

    Overlap itself is normal and used to fail here: a session is mined from the
    pool, corrected, and becomes training data while staying physically in the
    pool folder. That is the lifecycle, not a mistake. What would be a mistake
    is EVERY source living there with nothing left over."""
    under = [s for s in weeds.CONFIG["SOURCES"]
             if s["DATUMARO_ROOT"].startswith(weeds.WEED_POOL_ROOT)]
    assert len(under) < len(weeds.CONFIG["SOURCES"]) or not under, (
        "every training source is inside the pool - either the pool has "
        "nothing unseen left, or WEED_POOL_ROOT is pointing at the training "
        "campaign")


def test_a_trained_session_is_excluded_from_scoring_by_the_runner():
    """The real guardrail, now that pool and training campaigns overlap. It
    reads the build's own manifest rather than a list kept in step by hand,
    because scoring a session the model trained on gives ceiling agreement
    that reads as a great batch and teaches nothing."""
    from training.datasets import weeds_selftrain as st
    chosen, skipped = st.session_plan(
        weeds.WEED_POOL_ROOT, weeds.CONFIG["OUT_DIR"], [],
        requested=["vid_trained", "vid_fresh"])
    # No manifest on this machine, so nothing is excluded - the point is that
    # the runner consults one at all, and names its reason when it does.
    import inspect
    src = inspect.getsource(st.session_plan)
    assert "trained_sessions(dataset_dir)" in src
    assert "already in the training build" in src
    assert chosen == ["vid_trained", "vid_fresh"]


def test_looking_uses_the_same_backend_the_training_run_writes():
    """rfdetr weights loaded as maskrcnn fail deep inside the backend, and a
    dropped --backend has cost this project a run before."""
    from training.datasets import weeds_look
    assert weeds_look.CONFIG["BACKEND"] == "rfdetr"
    assert weeds_look.CONFIG["CHECKPOINT"].endswith("checkpoint_best_total.pth")


def test_looking_defaults_to_the_round_being_trained():
    from training.datasets import weeds_look
    assert f"weeds_r{weeds_train.ROUND}" in weeds_look.CONFIG["CHECKPOINT"]


def test_looking_strides_rather_than_taking_the_first_n_frames():
    """Consecutive ZED frames are near-identical, so LIMIT alone gives you 40
    pictures of the same plant and no evidence about the drive."""
    from training.datasets import weeds_look
    assert weeds_look.CONFIG["STRIDE"] > 1


def test_looking_is_rgb_only():
    """Stage A is what is being judged; 'full' folds depth, LEP and the safety
    decision into the picture and three more failure modes with them."""
    from training.datasets import weeds_look
    assert weeds_look.CONFIG["MODE"] == "segmentation"


def test_look_paths_are_absolute():
    from training.datasets import weeds_look
    import ntpath
    for key in ("IMAGES", "CHECKPOINT", "OUT_DIR"):
        assert ntpath.isabs(weeds_look.CONFIG[key]), key


# --------------------------------------------------------------------------- #
# Which model each runner loads
# --------------------------------------------------------------------------- #
def test_every_runner_derives_its_checkpoint_from_one_round():
    """RUNS_ROOT and ROUND live in weeds_train.py and everything else imports
    them, so bumping the round moves training, inference, mining and
    self-training together. A path written out by hand in one of them is how
    they drift."""
    from training.datasets import weeds_look, weeds_selftrain
    for path in (weeds_train.CONFIG["RUN_DIR"],
                 weeds_look.CONFIG["CHECKPOINT"],
                 weeds_look.CONFIG["OUT_DIR"],
                 weeds_mine.CONFIG["CHECKPOINT"],
                 weeds_selftrain.OUT_DIR):
        assert path.startswith(weeds_train.RUNS_ROOT), path

    # PREDICTIONS is the newest existing look folder, so it is legitimately ""
    # on a machine where this round has never run. Empty means "make one", and
    # the name it makes still has to come from the same two constants.
    import inspect
    src = inspect.getsource(weeds_selftrain._predict)
    assert 'newest(round_dir, f"look_{session}")' in src, \
        "the prediction folder must be derived, not written out by hand"


def test_mining_ranks_with_the_PREVIOUS_rounds_model():
    """Round N's model does not exist until round N's batch is corrected and
    trained on, so mining round N must rank with round N-1.

    This was hard-coded to weeds_r0: bumping ROUND to 2 moved the output folder
    and left the ranking model at round 0. The batch would still have been
    complete and plausible - chosen by a model two rounds stale - and nothing
    in the run would have said so."""
    assert f"weeds_r{weeds_mine.ROUND - 1}" in weeds_mine.CONFIG["CHECKPOINT"]
    assert f"weeds_round{weeds_mine.ROUND}" in weeds_mine.CONFIG["OUT_DIR"]


def test_mining_at_round_zero_does_not_ask_for_round_minus_one():
    import ntpath
    assert ntpath.isabs(weeds_mine.CONFIG["CHECKPOINT"])
    assert "weeds_r-1" not in weeds_mine.CONFIG["CHECKPOINT"]


def test_every_runner_uses_the_total_checkpoint_not_the_ema():
    """rfdetr keeps three files and _total is copied from whichever actually
    won, so naming _ema silently scores the loser."""
    from training.datasets import weeds_look
    for path in (weeds_look.CONFIG["CHECKPOINT"], weeds_mine.CONFIG["CHECKPOINT"]):
        assert path.endswith("checkpoint_best_total.pth"), path


def test_the_run_names_the_model_it_loaded():
    """"Which model just ran" is the first thing anyone asks of a prediction
    folder, and every path is derived from a ROUND edited by hand three files
    away. The run has to say it rather than leave it to be reconstructed."""
    src = (ROOT / "perception" / "predict_images.py").read_text(encoding="utf-8")
    assert 'print(f"  model : {ckpt}")' in src
    assert "st_mtime" in src, "the date distinguishes two rounds' checkpoints"


# --------------------------------------------------------------------------- #
# Three builds, one shared base
# --------------------------------------------------------------------------- #
def _builds():
    from training.datasets import mixed, onions
    return {"weeds": weeds, "onions": onions, "mixed": mixed}


def test_the_onion_build_is_a_runner_not_the_shared_base():
    """It used to live IN make_dataset.py's CONFIG - the base weeds.py
    overrides. One CONFIG edited back and forth between an onion build and a
    weed build is how a stale DATASET_DIR reaches a training run, which is the
    mistake the per-dataset runners exist to prevent."""
    from training.datasets import onions
    from common.ontology import CROP_CLASS
    assert onions.CONFIG["KEEP_CLASSES"] == [CROP_CLASS]
    assert onions.CONFIG["OUT_DIR"] != BASE["OUT_DIR"]


def test_no_two_builds_write_to_the_same_place():
    """A shared OUT_DIR silently rebuilds one dataset over another."""
    outs = [m.CONFIG["OUT_DIR"] for m in _builds().values()]
    assert len(set(outs)) == len(outs)


def test_the_mixed_build_covers_the_whole_ontology():
    """It is the build the deployed model comes from, so a class left out here
    is a class the pipeline scores as background."""
    from common.ontology import CLASSES
    from training.datasets import mixed
    assert set(mixed.CONFIG["KEEP_CLASSES"]) == set(CLASSES)


def test_the_mixed_build_includes_the_crop():
    """Without it the safety decision rejects every candidate - see
    perception/preflight.py. That is the entire reason this build exists."""
    from common.ontology import CROP_CLASS
    from training.datasets import mixed
    assert CROP_CLASS in mixed.CONFIG["KEEP_CLASSES"]


def test_the_mixed_build_drops_nothing_by_default():
    """weeds.py drops weed_cluster and wild_radish on counts that are
    properties of THAT build. Merging sessions changes them, so carrying the
    list over would drop classes that now have examples."""
    from training.datasets import mixed
    assert mixed.CONFIG["DROP_CLASSES"] == []


def test_the_mixed_build_declares_mixed_provenance():
    """Hand-corrected weeds beside unreviewed crop masks. Neither answers for
    the other, and every score is read through this field."""
    from training.datasets import mixed
    assert mixed.CONFIG["LABEL_PROVENANCE"] == "mixed"


def test_the_mixed_build_imports_its_sources_rather_than_repeating_them():
    """A session added to the weed build must reach the mixed build without a
    second edit, or the two silently come to disagree about which drives
    exist."""
    from training.datasets import mixed, onions
    roots = [s["DATUMARO_ROOT"] for s in mixed.CONFIG["SOURCES"]]
    for p in list(weeds.WEED_SESSIONS) + list(onions.ONION_SESSIONS):
        assert p in roots, p


def test_the_mixed_build_stratifies_by_scene():
    """A split that put every onion frame in train and every weed frame in val
    would report a confident number about a question the model was never
    asked."""
    from training.datasets import mixed
    assert mixed.CONFIG["STRATIFY_BY_SCENE"] is True


def test_every_build_shares_the_seed():
    for name, m in _builds().items():
        assert m.CONFIG["SEED"] == BASE["SEED"], name


def test_the_mixed_trainer_reads_the_mixed_build():
    from training.datasets import mixed, mixed_train
    assert mixed_train.CONFIG["DATASET_DIR"] == mixed.CONFIG["OUT_DIR"]


def test_the_mixed_trainer_does_not_overwrite_the_weed_runs():
    """A mixed round landing in the weed runs root would overwrite the
    checkpoint it is supposed to be compared against."""
    from training.datasets import mixed_train
    assert not mixed_train.CONFIG["RUN_DIR"].startswith(weeds_train.RUNS_ROOT)
    assert f"mixed_r{mixed_train.ROUND}" in mixed_train.CONFIG["RUN_DIR"]


def test_the_two_trainers_share_their_recipe():
    """Changing the recipe in the same step that adds the crop class makes the
    comparison between the two models uninterpretable - and that comparison is
    how you learn whether adding onions cost you weed recall."""
    from training.datasets import mixed_train
    for k in ("RESOLUTION", "VARIANT", "BATCH", "GRAD_ACCUM", "EPOCHS",
              "PATIENCE", "TVERSKY_ALPHA", "TVERSKY_BETA"):
        assert mixed_train.CONFIG[k] == weeds_train.CONFIG[k], k


def test_every_new_configured_path_is_absolute():
    """Path(<windows path>).parent on posix returns '.', so a relative path
    here writes beside the working directory."""
    import ntpath
    from training.datasets import mixed, mixed_train, onions
    paths = [onions.CONFIG["OUT_DIR"], mixed.CONFIG["OUT_DIR"],
             mixed_train.CONFIG["DATASET_DIR"], mixed_train.CONFIG["RUN_DIR"]]
    paths += [s["DATUMARO_ROOT"] for s in mixed.CONFIG["SOURCES"]]
    for p in paths:
        assert ntpath.isabs(p), f"not an absolute path: {p!r}"
