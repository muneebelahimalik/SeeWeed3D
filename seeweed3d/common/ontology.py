#!/usr/bin/env python3
"""
SeeWeed3D - project ontology (single source of truth for class names).

Every stage - prelabelers, CVAT schemas, COCO exports, training - imports these
names from here, so a class can never be spelled two different ways in two
places, and category IDs stay stable across the weed, onion and future mixed
datasets. Stable IDs are what let those datasets be merged later without
remapping every annotation.

NAMING RULE: lower_snake_case, no spaces. CVAT, COCO and YOLO all tolerate it,
and it survives being used as a filename or a column header.
"""

# Ordered: the index+1 is the COCO category_id. NEVER reorder or renumber these
# once annotation has started - it would silently remap existing labels. Append
# new classes at the end instead.
CLASSES = [
    "cutleaf_evening_primrose",   # Oenothera laciniata - rosette
    "wild_radish",                # Raphanus raphanistrum - rosette
    "grass_weed",                 # grasses / tillers
    "weed_cluster",               # intermingled weeds, no separable single LEP
    "other_weed",                 # any weed not in the classes above
    "onion_plant",                # the CROP. Never a treatment target.
]

CATEGORY_ID = {name: i + 1 for i, name in enumerate(CLASSES)}

# The crop. Present in the ontology so that an onion appearing in a weed scene
# can be labelled as an onion rather than forced into a weed class - a
# crop-safety protection, not an expectation.
CROP_CLASS = "onion_plant"

# Classes a weed-only prelabeler may emit. Species are excluded because species
# is an appearance question that shape cannot answer, and the crop is excluded
# because it should not be present in a weed-only scene.
WEED_CLASSES = [c for c in CLASSES if c != CROP_CLASS]
AUTO_ASSIGNABLE = {"grass_weed", "weed_cluster", "other_weed"}

# Rosette-forming classes: leaves radiate from a central meristem, so the LEP
# estimator weights phyllotactic evidence highly for them.
ROSETTE_CLASSES = {"cutleaf_evening_primrose", "wild_radish", "other_weed"}

# Point label for the annotated leaf emergence point.
LEP_LABEL = "weed_LEP"
IGNORE_LABEL = "ignore_region"

# Display colours, shared by CVAT schemas and preview overlays.
CLASS_COLORS = {
    "cutleaf_evening_primrose": "#ff8c00",
    "wild_radish":              "#ff6037",
    "grass_weed":               "#ffcc00",
    "weed_cluster":             "#8a2be2",
    "other_weed":               "#aaaaaa",
    "onion_plant":              "#33ddff",
}

# BGR equivalents for OpenCV previews.
CLASS_COLORS_BGR = {
    "cutleaf_evening_primrose": (0, 140, 255),
    "wild_radish":              (55, 96, 255),
    "grass_weed":               (0, 204, 255),
    "weed_cluster":             (226, 43, 138),
    "other_weed":               (170, 170, 170),
    "onion_plant":              (255, 221, 51),
}

INSTANCE_ATTRIBUTES = [
    {"name": "growth_stage", "input_type": "select", "mutable": True,
     "values": ["cotyledon", "2-leaf", "3-5-leaf", "later", "unknown"],
     "default_value": "unknown"},
    {"name": "lep_visibility", "input_type": "select", "mutable": True,
     "values": ["visible", "partially_occluded_inferable", "not_visible"],
     "default_value": "visible"},
    {"name": "targetable", "input_type": "select", "mutable": True,
     "values": ["yes", "no", "uncertain"], "default_value": "yes"},
    {"name": "difficulty", "input_type": "select", "mutable": True,
     "values": ["normal", "overlapping", "blurred", "shadowed", "wet", "truncated"],
     "default_value": "normal"},
    {"name": "species_note", "input_type": "text", "mutable": True,
     "default_value": ""},
]


def coco_categories(classes=None):
    """COCO category block with the project's stable IDs."""
    names = classes if classes is not None else CLASSES
    return [{"id": CATEGORY_ID[n], "name": n,
             "supercategory": "crop" if n == CROP_CLASS else "weed"}
            for n in names]


def cvat_labels(classes=None, include_lep=True, shape_type="polygon"):
    """CVAT label schema to paste into the Raw label editor."""
    names = classes if classes is not None else CLASSES
    labels = [{"name": n, "type": shape_type,
               "color": CLASS_COLORS.get(n, "#aaaaaa"),
               "attributes": [dict(a) for a in INSTANCE_ATTRIBUTES]}
              for n in names]
    if include_lep:
        labels.append({
            "name": LEP_LABEL, "type": "points", "color": "#fffc00",
            "attributes": [
                {"name": "lep_visibility", "input_type": "select", "mutable": True,
                 "values": ["visible", "partially_occluded_inferable"],
                 "default_value": "visible"}]})
    labels.append({"name": IGNORE_LABEL, "type": shape_type, "color": "#000000",
                   "attributes": [
                       {"name": "reason", "input_type": "select", "mutable": False,
                        "values": ["severe_blur", "ambiguity",
                                   "labeling_uncertainty", "out_of_range"],
                        "default_value": "ambiguity"}]})
    return labels
