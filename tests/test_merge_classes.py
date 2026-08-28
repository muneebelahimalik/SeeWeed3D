"""Relabelling a thin class instead of deleting it.

A class with too few examples had two answers and both are wrong. Dropping it
removes the annotations, so real plants are trained as BACKGROUND - a weeder
that has learned a radish is soil, which is the failure this project exists to
avoid. Keeping it asks the model a question the build cannot answer: 91
instances across 135 frames reports an AP whose error bars are wider than the
number, and that lands on the round establishing the baseline.

Merging keeps every instance a target and asks only what the data supports.
It is not a fudge: other_weed MEANS "a weed I cannot name more precisely", so
a radish labelled other_weed is true, merely less specific.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from common.ontology import CLASSES, CROP_CLASS   # noqa: E402


def test_the_weed_build_merges_rather_than_drops_the_thin_class():
    """THE decision. Dropping was the previous answer; it taught the model that
    91 real plants were soil."""
    from training.datasets import weeds
    assert weeds.CONFIG["MERGE_CLASSES"] == {"wild_radish": "other_weed"}
    assert "wild_radish" not in weeds.CONFIG["DROP_CLASSES"]


def test_the_merge_target_is_a_weed_class():
    """Merging into the crop would put a weed in the safety mask."""
    from training.datasets import weeds
    for tgt in weeds.CONFIG["MERGE_CLASSES"].values():
        assert tgt != CROP_CLASS
        assert tgt in CLASSES


def test_a_merged_class_is_still_kept():
    """KEEP_CLASSES turns into a deny-list, so a class merged but not kept
    would be dropped before the merge could save it."""
    from training.datasets import weeds
    keep = set(weeds.CONFIG["KEEP_CLASSES"])
    for src, tgt in weeds.CONFIG["MERGE_CLASSES"].items():
        assert src in keep, f"{src} would be dropped before it is merged"
        assert tgt in keep, f"{tgt} is the target and must survive"


def test_a_class_is_not_both_merged_and_dropped():
    """Both would be silently contradictory - the merge runs first, so the
    instances would arrive under the target name and the drop would miss."""
    from training.datasets import weeds
    assert not (set(weeds.CONFIG["MERGE_CLASSES"])
                & set(weeds.CONFIG["DROP_CLASSES"]))


def test_weed_cluster_is_dropped_and_not_merged():
    """A different question. weed_cluster means 'no separable single growth
    point' - a statement about TARGETABILITY, not species. Calling one
    other_weed would assert it can be aimed at individually."""
    from training.datasets import weeds
    assert "weed_cluster" in weeds.CONFIG["DROP_CLASSES"]
    assert "weed_cluster" not in weeds.CONFIG["MERGE_CLASSES"]


def test_the_merge_runs_before_the_drop():
    """Order is load-bearing: a class can be merged and its old label dropped
    in the same build, and the reverse order would delete it first."""
    import inspect
    from training import prepare_dataset as pd
    src = inspect.getsource(pd.build)
    assert src.index("if merge_classes:") < src.index("if drop:")


def test_the_base_config_offers_it_and_defaults_to_off():
    from training.make_dataset import CONFIG as BASE
    assert BASE["MERGE_CLASSES"] == {}


def test_an_unknown_class_is_refused(tmp_path):
    """Naming a class the ontology does not define is a typo, and a silent
    no-op would leave the instances under their original label."""
    from training import prepare_dataset as pd
    with pytest.raises(SystemExit, match="not in the ontology"):
        pd.build(str(tmp_path), str(tmp_path), str(tmp_path / "o"),
                 merge_classes={"not_a_plant": "other_weed"})


def test_the_manifest_records_what_was_merged():
    """Six months on, a dataset whose other_weed silently contains radishes is
    unauditable - the same reason label provenance exists."""
    import inspect
    from training import prepare_dataset as pd
    assert '"merged"' in inspect.getsource(pd.build)


def test_a_chained_merge_is_refused(tmp_path):
    """A -> B while B -> C is applied once, not repeatedly, so where A lands
    would depend on dict iteration order. Silently ordering-dependent is worse
    than refused."""
    from training import prepare_dataset as pd
    with pytest.raises(SystemExit, match="chains"):
        pd.build(str(tmp_path), str(tmp_path), str(tmp_path / "o"),
                 merge_classes={"wild_radish": "other_weed",
                                "other_weed": "grass_weed"})


def test_an_unknown_target_is_refused_too(tmp_path):
    """Only the source was checked at first, so a typo in the TARGET produced a
    class name nothing downstream could index."""
    from training import prepare_dataset as pd
    with pytest.raises(SystemExit, match="not in the ontology"):
        pd.build(str(tmp_path), str(tmp_path), str(tmp_path / "o"),
                 merge_classes={"wild_radish": "other_wed"})


def test_it_fails_before_reading_any_export(tmp_path):
    """A config typo must not cost a full merge of every export first. The
    error names the ontology, not a missing Datumaro file."""
    from training import prepare_dataset as pd
    with pytest.raises(SystemExit) as e:
        pd.build(str(tmp_path), str(tmp_path), str(tmp_path / "o"),
                 merge_classes={"nope": "other_weed"})
    assert "Datumaro" not in str(e.value)
