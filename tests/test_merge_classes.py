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
import json
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


# --------------------------------------------------------------------------- #
# End to end - the half that shipped broken
# --------------------------------------------------------------------------- #
def _export(tmp_path, labels_by_frame):
    """A minimal Datumaro export plus its images. labels_by_frame maps a frame
    number to the ontology class index its single instance carries."""
    import cv2
    import numpy as np
    from training import prepare_dataset as pd
    items = [{
        "id": f"s_one_{i:06d}", "media": {"path": f"s_one_{i:06d}.png"},
        "image": {"size": [200, 200]},
        "annotations": [{"id": i, "type": "polygon", "label_id": lab,
                         "group": i,
                         "points": [10, 10, 60, 10, 60, 60, 10, 60],
                         "attributes": {}}]}
        for i, lab in sorted(labels_by_frame.items())]
    ann = tmp_path / "exports" / "annotations"
    ann.mkdir(parents=True)
    (ann / "default.json").write_text(json.dumps(
        {"info": {}, "categories": {"label": {"labels": [
            {"name": c} for c in pd.CLASSES]}}, "items": items}))
    d = tmp_path / "sessions" / "s_one" / "rgb"
    d.mkdir(parents=True)
    for i in labels_by_frame:
        cv2.imwrite(str(d / f"s_one_{i:06d}.png"),
                    np.zeros((200, 200, 3), np.uint8))
    return ann.parent, tmp_path / "sessions"


def _built(tmp_path, **kw):
    from training import prepare_dataset as pd
    radish = CLASSES.index("wild_radish")
    grass = CLASSES.index("grass_weed")
    labels = {i: (radish if i % 2 else grass) for i in range(1, 13)}
    root, images = _export(tmp_path, labels)
    pd.build(root, images, tmp_path / "ds", val_fraction=0.25,
             test_fraction=0.0, strict=False, **kw)
    return json.loads((tmp_path / "ds" / "seg_manifest.json").read_text())


def test_a_merged_class_gets_no_detection_head(tmp_path):
    """THE REGRESSION, and it shipped. Renaming the instances empties the class
    but left it in active_classes, so the model was built with one more class
    than it had data for - capacity spent on something that can never be
    predicted, an AP pinned at zero dragging the mean, and a model with a
    different class count from the round before it, which breaks the only
    comparison the loop exists to make.

    A real run reached the trainer showing 4 classes and `wild_radish 0 0`."""
    man = _built(tmp_path, merge_classes={"wild_radish": "other_weed"})
    assert "wild_radish" not in man["classes"]


def test_the_merge_target_survives_as_a_class(tmp_path):
    man = _built(tmp_path, merge_classes={"wild_radish": "other_weed"})
    assert "other_weed" in man["classes"]


def test_the_instances_really_moved(tmp_path):
    """Not merely relabelled in the report - the frames must carry them."""
    man = _built(tmp_path, merge_classes={"wild_radish": "other_weed"})
    names = {i["class_name"] for f in man["frames"] for i in f["instances"]}
    assert "wild_radish" not in names
    assert "other_weed" in names


def test_nothing_is_lost_to_a_merge(tmp_path):
    """The whole reason to merge rather than drop: every instance survives."""
    plain = _built(tmp_path / "a", merge_classes={})
    merged = _built(tmp_path / "b", merge_classes={"wild_radish": "other_weed"})
    n = lambda m: sum(len(f["instances"]) for f in m["frames"])
    assert n(merged) == n(plain)


def test_a_dropped_class_does_lose_its_instances(tmp_path):
    """The contrast that makes the merge worth having."""
    plain = _built(tmp_path / "a", merge_classes={})
    dropped = _built(tmp_path / "b", drop_classes=["wild_radish"])
    n = lambda m: sum(len(f["instances"]) for f in m["frames"])
    assert n(dropped) < n(plain)


def test_merging_leaves_no_empty_class_in_the_report(tmp_path):
    """`wild_radish 0 0` in a preflight table is the symptom someone has to
    notice by eye. There should be nothing to notice."""
    man = _built(tmp_path, merge_classes={"wild_radish": "other_weed"})
    counts = man.get("per_class") or {}
    assert all(v for v in counts.values()), counts
