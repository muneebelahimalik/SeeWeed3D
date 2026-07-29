"""The ontology is the single source of truth for class names and category IDs.
These tests lock the contract that everything else depends on."""
from conftest import load_script

onto = load_script("common/ontology.py")


def test_classes_are_the_agreed_taxonomy():
    assert onto.CLASSES == [
        "cutleaf_evening_primrose",
        "wild_radish",
        "grass_weed",
        "weed_cluster",
        "other_weed",
        "onion_plant",
    ]


def test_category_ids_are_stable_and_one_based():
    """COCO ids must never be reordered once annotation has started - doing so
    would silently remap every existing label."""
    assert onto.CATEGORY_ID["cutleaf_evening_primrose"] == 1
    assert onto.CATEGORY_ID["onion_plant"] == 6
    ids = [c["id"] for c in onto.coco_categories()]
    assert ids == sorted(ids) == list(range(1, len(onto.CLASSES) + 1))


def test_weed_classes_exclude_the_crop():
    assert onto.CROP_CLASS == "onion_plant"
    assert onto.CROP_CLASS not in onto.WEED_CLASSES
    assert len(onto.WEED_CLASSES) == len(onto.CLASSES) - 1


def test_species_are_not_auto_assignable():
    """Both named species are rosette-forming, so shape cannot separate them."""
    for species in ("cutleaf_evening_primrose", "wild_radish"):
        assert species not in onto.AUTO_ASSIGNABLE
    assert onto.CROP_CLASS not in onto.AUTO_ASSIGNABLE
    assert onto.AUTO_ASSIGNABLE == {"grass_weed", "weed_cluster", "other_weed"}


def test_names_are_snake_case_without_spaces():
    for n in onto.CLASSES + [onto.LEP_LABEL, onto.IGNORE_LABEL]:
        assert " " not in n, f"{n} contains a space"
        assert n == n.strip()


def test_cvat_schema_covers_every_class_plus_lep():
    labels = onto.cvat_labels()
    names = [l["name"] for l in labels]
    for c in onto.CLASSES:
        assert c in names
    assert onto.LEP_LABEL in names and onto.IGNORE_LABEL in names
    lep = next(l for l in labels if l["name"] == onto.LEP_LABEL)
    assert lep["type"] == "points"
    # Every class label carries the shared instance attributes.
    cls_label = next(l for l in labels if l["name"] == "wild_radish")
    attr_names = {a["name"] for a in cls_label["attributes"]}
    assert {"growth_stage", "lep_visibility", "targetable", "difficulty"} <= attr_names


def test_every_attribute_has_a_nonempty_values_array():
    """CVAT's Raw label editor rejects the WHOLE schema on paste if any
    attribute - including 'text' ones, where CVAT still expects the default
    wrapped in a single-element list - is missing 'values' or has it empty.
    Regression guard: species_note shipped without 'values' and broke every
    weed_cvat_labels.json paste until this was caught."""
    for label in onto.cvat_labels():
        for attr in label["attributes"]:
            assert "values" in attr, (
                f"{label['name']}.{attr['name']} has no 'values' key - "
                f"CVAT will reject the entire schema")
            assert isinstance(attr["values"], list) and len(attr["values"]) > 0, (
                f"{label['name']}.{attr['name']} has an empty 'values' array")


def test_coco_categories_can_be_restricted_but_keep_global_ids():
    """A weed-only export omits the crop, but the ids it does emit must match
    the global ontology so datasets merge without remapping."""
    cats = onto.coco_categories(onto.WEED_CLASSES)
    assert [c["name"] for c in cats] == onto.WEED_CLASSES
    for c in cats:
        assert c["id"] == onto.CATEGORY_ID[c["name"]]


def test_rosette_classes_used_by_the_lep_estimator():
    assert "wild_radish" in onto.ROSETTE_CLASSES
    assert "cutleaf_evening_primrose" in onto.ROSETTE_CLASSES
    assert "grass_weed" not in onto.ROSETTE_CLASSES
