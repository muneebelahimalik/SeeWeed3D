"""The CVAT label schema, against the one actually deployed.

WHY THIS IS PINNED TO A CAPTURED FILE
-------------------------------------
CVAT matches annotations to labels BY NAME. A schema that has drifted from the
tasks already in CVAT does not fail loudly - it creates a DUPLICATE label and
leaves the prelabels sitting in it, unreviewed, next to the empty label the
annotator is looking at. That is failure #40 in this project's history and it
was found by eye, weeks later.

tests/data/cvat_deployed_labels.json is the schema in use on the annotation
machine, captured from CVAT's Raw editor. Comparing against it caught two real
drifts:

  * `other_weed` was #aaaaaa - the exact grey predict_images uses for "a class
    not in the ontology" - so the model's MOST-predicted class rendered as the
    unknown-class colour, and against dry soil it is the one colour that
    vanishes.
  * `weed_LEP`'s lep_visibility defaulted to "visible" rather than
    "partially_occluded_inferable", so every point an annotator dropped and did
    not touch recorded a certainty nobody asserted.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "seeweed3d") not in sys.path:
    sys.path.insert(0, str(ROOT / "seeweed3d"))

from common.ontology import CLASSES, CLASS_COLORS, cvat_labels  # noqa: E402

DEPLOYED = json.loads(
    (ROOT / "tests" / "data" / "cvat_deployed_labels.json").read_text())
GENERATED = cvat_labels()

BY_NAME_D = {l["name"]: l for l in DEPLOYED}
BY_NAME_G = {l["name"]: l for l in GENERATED}


def test_the_label_names_match_exactly():
    """The only field CVAT matches on. A name here that CVAT does not have is a
    duplicate label and a silently unreviewed prelabel."""
    assert [l["name"] for l in GENERATED] == [l["name"] for l in DEPLOYED]


@pytest.mark.parametrize("name", sorted(BY_NAME_D))
def test_each_label_has_the_deployed_type_and_colour(name):
    d, g = BY_NAME_D[name], BY_NAME_G[name]
    assert g["type"] == d["type"], name
    assert g["color"] == d["color"], name


@pytest.mark.parametrize("name", ["cutleaf_evening_primrose", "weed_LEP",
                                  "ignore_region"])
def test_the_attributes_match_the_deployed_schema(name):
    """Values and defaults both. A default that differs changes what an
    UNTOUCHED annotation records, which is most of them."""
    def strip(attrs):
        return [{k: a[k] for k in
                 ("name", "input_type", "mutable", "values", "default_value")}
                for a in attrs]
    assert strip(BY_NAME_G[name]["attributes"]) == \
           strip(BY_NAME_D[name]["attributes"]), name


def test_the_lep_default_is_the_cautious_one():
    """An annotator who drops a point and moves on should leave behind
    "I inferred this", not "I could see it"."""
    lep = BY_NAME_G["weed_LEP"]["attributes"][0]
    assert lep["default_value"] == "partially_occluded_inferable"


def test_every_class_has_an_explicit_colour():
    """The guard for what went wrong: other_weed fell through to the #aaaaaa
    fallback, which is also what predict_images draws an UNKNOWN class in."""
    missing = [c for c in CLASSES if c not in CLASS_COLORS]
    assert not missing, f"these fall back to the unknown-class grey: {missing}"


def test_no_class_uses_the_unknown_class_grey():
    grey = [c for c, v in CLASS_COLORS.items() if v.lower() == "#aaaaaa"]
    assert not grey, f"{grey} are coloured as 'not in the ontology'"


def test_the_generated_schema_omits_ids():
    """CVAT assigns label and attribute ids on paste. Emitting our own would
    collide with whatever the server already has."""
    assert "id" not in GENERATED[0]
    assert all("id" not in a for a in GENERATED[0]["attributes"])
