"""A benchmark whose "truth" side is not ground truth has to say so.

THE FAILURE THIS PREVENTS
-------------------------
Comparing the trained model against the SAM prelabels is a legitimate and
useful thing to do - it is the only comparison available before anything is
hand-corrected. What it measures is AGREEMENT BETWEEN TWO PROPOSALS, and the
flag is called `--truth`.

Here the two sides are correlated by construction: the model was trained on
corrected SAM prelabels, so it inherits the prelabeler's biases through its
training data. A high agreement number is therefore also exactly what two
sources wrong in the same way produce, and nothing in a report headed "truth"
suggests that.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "seeweed3d"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import bench_mixed as bm                     # noqa: E402


def coco(tmp_path, description, name="instances_default.json"):
    d = tmp_path / description.replace(" ", "_")[:20]
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({
        "info": {"description": description},
        "images": [{"id": 1, "file_name": "a.png", "height": 8, "width": 8}],
        "annotations": [], "categories": [{"id": 1, "name": "grass_weed"}]}))
    return d


def test_sam_prelabels_are_recognised(tmp_path):
    d = coco(tmp_path, "SeeWeed3D SAM 3 weed instance prelabels")
    assert bm.source_provenance(d) == "prelabels"


def test_model_predictions_are_recognised(tmp_path):
    d = coco(tmp_path, "SeeWeed3D MODEL PREDICTIONS - not ground truth")
    assert bm.source_provenance(d) == "predictions"


def test_a_corrected_export_carries_no_marker(tmp_path):
    """CVAT writes its own info block and cannot know it was corrected, so ""
    is the honest answer rather than a guess."""
    assert bm.source_provenance(coco(tmp_path, "exported from CVAT")) == ""


def test_a_missing_path_does_not_crash(tmp_path):
    assert bm.source_provenance(tmp_path / "nope") == ""


def test_the_warning_fires_on_prelabelled_truth(tmp_path):
    t = coco(tmp_path, "SeeWeed3D SAM 3 weed instance prelabels")
    p = coco(tmp_path, "SeeWeed3D MODEL PREDICTIONS - not ground truth")
    warn = bm.provenance_warning(t, p)
    assert warn and "AGREEMENT BETWEEN TWO PROPOSALS" in warn
    assert "SAM prelabels" in warn


def test_the_warning_names_the_shared_bias(tmp_path):
    """The specific reason agreement is inflated here, not a generic caution."""
    t = coco(tmp_path, "SeeWeed3D SAM 3 weed instance prelabels")
    warn = bm.provenance_warning(t, t)
    assert "trained on corrected SAM prelabels" in warn
    assert "FLOOR" in warn


def test_the_warning_is_silent_on_a_corrected_truth_side(tmp_path):
    """Firing on every run is how a warning gets ignored."""
    t = coco(tmp_path, "exported from CVAT")
    p = coco(tmp_path, "SeeWeed3D MODEL PREDICTIONS - not ground truth")
    assert bm.provenance_warning(t, p) is None


def test_the_markers_match_what_the_writers_actually_stamp():
    """The heuristic recognises THIS repo's files. If a writer's description
    changes, the check must fail here rather than go quiet in the field."""
    src = (ROOT / "annotation" / "prelabel_weeds_sam3.py").read_text()
    assert '"description": "SeeWeed3D SAM 3 weed instance prelabels"' in src
    pred = (ROOT / "perception" / "predict_images.py").read_text()
    assert "MODEL PREDICTIONS - not ground truth" in pred
