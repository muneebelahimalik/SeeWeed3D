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

# The single homogeneous class a MIXED-scene prelabeler may emit.
#
# Deliberately OUTSIDE `CLASSES`, and its id is far above them, for two
# reasons. It must never reach training - a model trained on "plant" has
# learned nothing this project needs, and the crop-safety metrics have no crop
# to measure. And it is the workflow's own progress signal: any shape still
# carrying it is a shape nobody has reviewed, which is a property worth being
# able to count rather than infer.
#
# In a mixed scene, guessing the class is worse than declining to. Shape
# separates a blade from a rosette, and an onion IS a blade - so the one
# morphology call the weed prelabeler can make confidently would label the crop
# as grass_weed. A wrong prelabel also costs more than a neutral one: an
# annotator confirms a plausible label and re-examines a blank one.
PRELABEL_CLASS = "plant"
PRELABEL_CATEGORY_ID = 100

# Attributes for a class-assignment pass. Deliberately short: the job is to
# press one key per shape, and every extra attribute is a field that defaults
# quietly and is never looked at. `difficulty` earns its place because it is
# the one thing the annotator can see and the pipeline cannot.
PRELABEL_ATTRIBUTES = [
    {"name": "difficulty", "input_type": "select", "mutable": True,
     "values": ["normal", "overlapping", "blurred", "shadowed", "wet",
                "truncated"],
     "default_value": "normal"},
]

# Display colours, shared by CVAT schemas and preview overlays.
CLASS_COLORS = {
    "cutleaf_evening_primrose": "#ff8c00",
    "wild_radish":              "#ff6037",
    "grass_weed":               "#ffcc00",
    "weed_cluster":             "#8a2be2",
    # NOT #aaaaaa. That is the exact grey predict_images uses for "a class not
    # in the ontology", so the model's MOST-predicted class was rendering as the
    # unknown-class colour - and against dry soil, stubble and shadow it is the
    # one colour that vanishes. This value is what the deployed CVAT tasks
    # already use, so old and new tasks agree.
    "other_weed":               "#3d3df5",
    "onion_plant":              "#33ddff",
}

# BGR equivalents for OpenCV previews.
CLASS_COLORS_BGR = {
    "cutleaf_evening_primrose": (0, 140, 255),
    "wild_radish":              (55, 96, 255),
    "grass_weed":               (0, 204, 255),
    "weed_cluster":             (226, 43, 138),
    "other_weed":               (245, 61, 61),      # BGR of #3d3df5
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
    # CVAT's Raw label editor requires "values" as a non-empty array for EVERY
    # attribute, including "text" ones - it holds the default, wrapped in a
    # single-element list. Omitting it (as an earlier version of this file did)
    # makes CVAT reject the whole label schema with "attribute values must be a
    # non-empty array" as soon as it is pasted in.
    {"name": "species_note", "input_type": "text", "mutable": True,
     "default_value": "", "values": [""]},
]


def coco_categories(classes=None):
    """COCO category block with the project's stable IDs."""
    names = classes if classes is not None else CLASSES
    return [{"id": CATEGORY_ID[n], "name": n,
             "supercategory": "crop" if n == CROP_CLASS else "weed"}
            for n in names]


def prelabel_categories():
    """COCO categories for a mixed-scene prelabel export: one class.

    The sentinel keeps its own high id so that a prelabel file and a corrected
    file can sit side by side without either shadowing the other's ids, and so
    that a stray `plant` annotation surviving into a merged dataset shows up as
    an unknown category rather than silently becoming class 1."""
    return [{"id": PRELABEL_CATEGORY_ID, "name": PRELABEL_CLASS,
             "supercategory": "unreviewed"}]


def prelabel_cvat_labels(classes=None, shape_type="polygon"):
    """CVAT schema for the class-assignment pass.

    Real classes come FIRST because CVAT's label shortcuts are assigned in list
    order - the whole point of the pass is that reassigning is one keystroke,
    so the keystrokes have to land on the classes actually used. The sentinel
    goes last: it is never chosen, only cleared.

    No LEP point label. Adding a shape type nobody will draw costs a slot in
    that same shortcut list."""
    names = classes if classes is not None else CLASSES
    labels = [{"name": n, "type": shape_type,
               "color": CLASS_COLORS.get(n, "#aaaaaa"),
               "attributes": [dict(a) for a in PRELABEL_ATTRIBUTES]}
              for n in names]
    labels.append({"name": PRELABEL_CLASS, "type": shape_type,
                   "color": "#ffffff",
                   "attributes": [dict(a) for a in PRELABEL_ATTRIBUTES]})
    labels.append({"name": IGNORE_LABEL, "type": shape_type, "color": "#000000",
                   "attributes": [
                       {"name": "reason", "input_type": "select",
                        "mutable": False,
                        "values": ["severe_blur", "ambiguity",
                                   "labeling_uncertainty", "out_of_range"],
                        "default_value": "ambiguity"}]})
    return labels


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
                 # The CAUTIOUS default, matching the deployed CVAT schema. An
                 # annotator who drops a LEP point and moves on should leave
                 # behind "I inferred this", not "I could see it": defaulting to
                 # `visible` records a certainty nobody asserted, and every
                 # untouched point then claims the crown was in view.
                 "default_value": "partially_occluded_inferable"}]})
    labels.append({"name": IGNORE_LABEL, "type": shape_type, "color": "#000000",
                   "attributes": [
                       {"name": "reason", "input_type": "select", "mutable": False,
                        "values": ["severe_blur", "ambiguity",
                                   "labeling_uncertainty", "out_of_range"],
                        "default_value": "ambiguity"}]})
    return labels
