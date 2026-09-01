"""Datumaro 1.0 multitask ingestion: mask/LEP ownership, the annotation
contract, and the exports. No datumaro package, no GPU, no real dataset."""
import json

import numpy as np
import pytest

from conftest import load_script

dm = load_script("training/datumaro_multitask.py")
cfgm = load_script("training/config.py")

LABELS = ["cutleaf_evening_primrose", "wild_radish", "grass_weed",
          "weed_cluster", "other_weed", "onion_plant", "weed_LEP",
          "ignore_region"]
L = {n: i for i, n in enumerate(LABELS)}


def _square(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


def _doc(items, labels=LABELS):
    return {"info": {},
            "categories": {"label": {"labels": [{"name": n, "parent": "",
                                                 "attributes": []}
                                                for n in labels],
                                     "attributes": []}},
            "items": items}


def _item(item_id="sess01_000001", annotations=(), w=640, h=480,
          path=None):
    return {"id": item_id, "annotations": list(annotations),
            "image": {"path": path or f"{item_id}.png", "size": [h, w]}}


def _poly(label, points, group=0, attrs=None, ann_id=1):
    return {"id": ann_id, "type": "polygon", "label_id": L[label],
            "group": group, "points": list(points), "z_order": 0,
            "attributes": dict(attrs or {})}


def _point(x, y, group=0, attrs=None, ann_id=99, label="weed_LEP"):
    return {"id": ann_id, "type": "points", "label_id": L[label],
            "group": group, "points": [x, y], "z_order": 0,
            "attributes": dict(attrs or {})}


def _write(tmp_path, doc, name="default.json"):
    d = tmp_path / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
def test_polygon_ingestion_and_class_ids(tmp_path):
    """Polygons become instances with ontology-stable class identity."""
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 40), group=1, ann_id=1),
        _poly("onion_plant", _square(200, 200, 60), group=2, ann_id=2),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    assert len(frames) == 1
    rec = frames[0]
    assert len(rec.instances) == 2
    assert {i.class_name for i in rec.instances} == {"wild_radish", "onion_plant"}
    assert len(rec.weeds) == 1 and len(rec.onions) == 1
    assert rec.width == 640 and rec.height == 480
    assert rec.session_id == "sess01"          # derived from <session>_<idx>

    from common.ontology import CATEGORY_ID
    weed = rec.weeds[0]
    assert weed.category_id == CATEGORY_ID["wild_radish"]


def test_grouped_lep_binds_to_its_owning_mask(tmp_path):
    """Ownership comes from group_id - the whole point of using Datumaro."""
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 60), group=7, ann_id=1),
        _poly("grass_weed", _square(200, 10, 60), group=8, ann_id=2),
        _point(40, 40, group=7, ann_id=3),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rec = frames[0]
    radish = next(i for i in rec.instances if i.class_name == "wild_radish")
    grass = next(i for i in rec.instances if i.class_name == "grass_weed")
    assert radish.lep is not None and radish.lep.uv == (40.0, 40.0)
    assert grass.lep is None                    # not the nearest-point winner
    assert not rec.orphan_leps


def test_ungrouped_lep_is_rejected_never_auto_assigned(tmp_path):
    """An ungrouped point must NOT be silently attached to the nearest weed:
    in a dense frame the nearest crown is often the neighbouring plant, and a
    wrong owner aims the laser at the wrong tissue."""
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 60), group=1, ann_id=1),
        _point(45, 45, group=0, ann_id=2),      # group 0 == Datumaro "no group"
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rec = frames[0]
    assert rec.instances[0].lep is None
    assert len(rec.orphan_leps) == 1
    assert any(e["kind"] == "ungrouped_lep" for e in rep.errors)

    # A nearest-instance hint may be offered, but only as a suggestion.
    dm.validate_frames(frames, report=rep)
    assert rep.suggestions and "not applied" in rep.suggestions[0]["note"]
    assert rec.instances[0].lep is None          # still untouched


def test_visibility_and_targetability_attributes_are_preserved(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("other_weed", _square(10, 10, 50), group=3, ann_id=1,
              attrs={"lep_visibility": "partially_occluded_inferable",
                     "targetable": "uncertain", "growth_stage": "cotyledon",
                     "species_note": "unsure"}),
        _point(30, 30, group=3, ann_id=2,
               attrs={"lep_visibility": "partially_occluded_inferable"}),
    ])])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    inst = frames[0].instances[0]
    assert inst.visibility == "partially_occluded_inferable"
    assert inst.targetable == "uncertain"
    assert inst.growth_stage == "cotyledon"
    assert inst.attributes["species_note"] == "unsure"
    assert inst.lep.visibility == "partially_occluded_inferable"


def test_not_visible_weed_is_not_forced_to_have_a_lep(tmp_path):
    """Absence of a LEP is legitimate, not an error, when it is not visible."""
    doc = _doc([_item(annotations=[
        _poly("other_weed", _square(10, 10, 50), group=1, ann_id=1,
              attrs={"lep_visibility": "not_visible"}),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rep = dm.validate_frames(frames, report=rep)
    assert not any(e["kind"] == "missing_lep" for e in rep.errors)


def test_visible_targetable_weed_without_lep_is_an_error(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 50), group=1, ann_id=1,
              attrs={"lep_visibility": "visible", "targetable": "yes"}),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rep = dm.validate_frames(frames, report=rep)
    assert any(e["kind"] == "missing_lep" for e in rep.errors)
    assert not rep.ok


def test_weed_cluster_must_not_carry_a_single_lep(tmp_path):
    """A cluster has no separable single growth point by definition."""
    doc = _doc([_item(annotations=[
        _poly("weed_cluster", _square(10, 10, 120), group=4, ann_id=1),
        _point(60, 60, group=4, ann_id=2),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rep = dm.validate_frames(frames, report=rep)
    assert any(e["kind"] == "lep_on_cluster" for e in rep.errors)


def test_onion_never_carries_a_weed_lep(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("onion_plant", _square(10, 10, 80), group=5, ann_id=1),
        _point(40, 40, group=5, ann_id=2),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    assert any(e["kind"] == "lep_on_onion" for e in rep.errors)
    assert frames[0].instances[0].lep is None


def test_lep_outside_owning_mask_is_flagged(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 40), group=1, ann_id=1),
        _point(300, 300, group=1, ann_id=2),       # far outside
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rep = dm.validate_frames(frames, report=rep)
    assert any(e["kind"] == "lep_outside_owning_mask" for e in rep.errors)


def test_lep_just_outside_mask_is_within_annotation_tolerance(tmp_path):
    """Annotators click at finite precision; a 2px miss is a warning, not a
    rejected annotation."""
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 40), group=1, ann_id=1),
        _point(52, 30, group=1, ann_id=2),        # ~2px outside x=50 edge
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    rep = dm.validate_frames(frames, report=rep)
    assert not any(e["kind"] == "lep_outside_owning_mask" for e in rep.errors)
    assert any(w["kind"] == "lep_near_mask_edge" for w in rep.warnings)


def test_duplicate_group_id_is_reported(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 40), group=1, ann_id=1),
        _poly("grass_weed", _square(100, 10, 40), group=1, ann_id=2),
        _point(30, 30, group=1, ann_id=3),
    ])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    assert any(e["kind"] == "duplicate_group_id" for e in rep.errors)


def test_ignore_regions_are_captured_and_kept_out_of_instances(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 40), group=1, ann_id=1),
        _poly("ignore_region", _square(300, 300, 100), group=0, ann_id=2,
              attrs={"reason": "severe_blur"}),
    ])])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    rec = frames[0]
    assert len(rec.instances) == 1                 # ignore is not an instance
    assert len(rec.ignore_regions) == 1


def test_unknown_label_fails_clearly(tmp_path):
    doc = _doc([_item(annotations=[_poly("wild_radish", _square(1, 1, 9),
                                         ann_id=1)])],
               labels=LABELS + ["mystery_plant"])
    with pytest.raises(dm.DatumaroFormatError) as e:
        dm.load_datumaro(_write(tmp_path, doc))
    assert "mystery_plant" in str(e.value)
    assert "ontology" in str(e.value)


def test_a_separator_slip_in_a_label_name_is_repaired(tmp_path):
    """'ignore region' and 'ignore_region' are the same label typed two ways.
    CVAT's label field is free text, and refusing the export over one character
    costs a re-annotation round trip."""
    labels = LABELS[:-1] + ["ignore region"]        # the slip REPLACES it
    ann = {"id": 1, "type": "polygon", "label_id": labels.index("ignore region"),
           "group": 0, "points": _square(1, 1, 9), "z_order": 0,
           "attributes": {}}
    doc = _doc([_item(annotations=[ann])], labels=labels)
    frames, report = dm.load_datumaro(_write(tmp_path, doc))
    assert len(frames[0].ignore_regions) == 1      # read as the real label
    assert frames[0].instances == []               # and not as a class
    kinds = [w["kind"] for w in report.warnings]
    assert "label_name_normalised" in kinds, "the repair must be recorded"


def test_case_and_hyphen_slips_are_repaired_too(tmp_path):
    labels = [("Grass-Weed" if n == "grass_weed" else n) for n in LABELS]
    ann = {"id": 1, "type": "polygon", "label_id": labels.index("Grass-Weed"),
           "group": 0, "points": _square(1, 1, 9), "z_order": 0,
           "attributes": {"lep_visibility": "not_visible", "targetable": "no"}}
    doc = _doc([_item(annotations=[ann])], labels=labels)
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    assert [i.class_name for i in frames[0].instances] == ["grass_weed"]


def test_a_label_that_is_a_guess_about_intent_still_fails(tmp_path):
    """The repair resolves only on an EXACT match after normalising. 'weeds' is
    not a separator slip, it is a guess about which class was meant - and
    guessing that is how a dataset silently trains on the wrong label."""
    doc = _doc([_item(annotations=[_poly("wild_radish", _square(1, 1, 9),
                                         ann_id=1)])],
               labels=LABELS + ["weeds"])
    with pytest.raises(dm.DatumaroFormatError) as e:
        dm.load_datumaro(_write(tmp_path, doc))
    assert "weeds" in str(e.value)


def test_a_clean_export_records_no_normalisation_warning(tmp_path):
    doc = _doc([_item(annotations=[_poly("grass_weed", _square(1, 1, 9),
                                         ann_id=1)])])
    _, report = dm.load_datumaro(_write(tmp_path, doc))
    assert not any(w["kind"] == "label_name_normalised"
                   for w in report.warnings)


def test_missing_categories_fails_clearly(tmp_path):
    p = _write(tmp_path, {"info": {}, "items": []})
    with pytest.raises(dm.DatumaroFormatError) as e:
        dm.load_datumaro(p)
    assert "Datumaro 1.0" in str(e.value)


def test_uncompressed_rle_mask_is_decoded(tmp_path):
    """Datumaro may export masks as RLE rather than polygons."""
    h = w = 20
    m = np.zeros((h, w), np.uint8)
    m[5:15, 5:15] = 1
    flat = m.reshape(-1, order="F")
    counts, prev, run = [], 0, 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev, run = v, 1
    counts.append(run)

    doc = _doc([_item(w=w, h=h, annotations=[{
        "id": 1, "type": "mask", "label_id": L["other_weed"], "group": 1,
        "z_order": 0, "attributes": {},
        "rle": {"counts": counts, "size": [h, w]}}])])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    inst = frames[0].instances[0]
    assert inst.polygons
    assert 60 <= inst.area_px() <= 120          # ~10x10 square


def test_compressed_rle_without_pycocotools_fails_clearly(tmp_path):
    pytest.importorskip  # noqa - readability
    try:
        import pycocotools  # noqa: F401
        pytest.skip("pycocotools installed; the clear-failure path is not taken")
    except ImportError:
        pass
    doc = _doc([_item(annotations=[{
        "id": 1, "type": "mask", "label_id": L["other_weed"], "group": 1,
        "z_order": 0, "attributes": {},
        "rle": {"counts": "d0d0:F\\0", "size": [10, 10]}}])])
    with pytest.raises(dm.DatumaroFormatError) as e:
        dm.load_datumaro(_write(tmp_path, doc))
    assert "pycocotools" in str(e.value)


def test_unresolvable_session_is_an_error(tmp_path):
    """A frame with no session cannot be placed in a leak-free split."""
    doc = _doc([_item(item_id="mystery", path="mystery.png",
                      annotations=[_poly("other_weed", _square(1, 1, 20),
                                         ann_id=1)])])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    assert any(e["kind"] == "unresolvable_session" for e in rep.errors)


def test_yolo_labels_are_normalised_and_use_ontology_indices(tmp_path):
    from common.ontology import CLASSES
    doc = _doc([_item(w=640, h=480, annotations=[
        _poly("grass_weed", _square(64, 48, 64), group=1, ann_id=1)])])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    body = dm.to_yolo_segmentation(frames[0])
    parts = body.split()
    assert int(parts[0]) == CLASSES.index("grass_weed")
    coords = [float(v) for v in parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)
    assert abs(coords[0] - 64 / 640) < 1e-6
    assert abs(coords[1] - 48 / 480) < 1e-6


def test_lep_manifest_references_images_without_copying(tmp_path):
    doc = _doc([_item(annotations=[
        _poly("wild_radish", _square(10, 10, 60), group=1, ann_id=1,
              attrs={"growth_stage": "2-leaf"}),
        _point(40, 40, group=1, ann_id=2),
        _poly("onion_plant", _square(300, 300, 60), group=2, ann_id=3),
    ])])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    rows = dm.to_lep_manifest(frames)
    assert len(rows) == 1                        # onion excluded, no LEP on it
    r = rows[0]
    assert r["class_name"] == "wild_radish"
    assert r["lep_x"] == 40.0 and r["lep_y"] == 40.0
    assert r["session_id"] == "sess01"
    assert "/" in r["image_path"] or r["image_path"].endswith(".png")
    assert r["growth_stage"] == "2-leaf"
    assert r["polygons"]                         # geometry travels in the manifest


def test_the_filename_is_found_when_size_and_path_are_in_different_blobs(tmp_path):
    """An export carrying BOTH `image` and `media` may split the fields between
    them. Taking the first key that exists and stopping loses the filename, and
    the fallback - the item id - has no extension, so the frame resolves to
    nothing and the loss is silent until something opens it."""
    item = {"id": "sess_a_000001",
            "image": {"size": [480, 640]},
            "media": {"path": "sess_a_000001.png"},
            "annotations": [_poly("grass_weed", _square(1, 1, 9), ann_id=1)]}
    frames, _ = dm.load_datumaro(_write(tmp_path, _doc([item])))
    assert frames[0].image_path == "sess_a_000001.png"
    assert (frames[0].width, frames[0].height) == (640, 480)


# --------------------------------------------------------------------------- #
# A hand-curated batch: frames someone gathered into one folder and annotated,
# whose filenames no longer name the drive they came from.
# --------------------------------------------------------------------------- #
def test_a_frame_that_names_no_session_is_unresolvable_without_a_fallback(tmp_path):
    """The existing guard: a frame with no session cannot be placed in a
    leak-free split, so it is an error rather than a guess."""
    doc = _doc([_item(item_id="raj_photo", path="raj_photo.png")])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc))
    assert frames[0].session_id == ""
    assert any(f["kind"] == "unresolvable_session" for f in rep.errors)


def test_a_batch_folder_supplies_the_session_a_filename_cannot(tmp_path):
    doc = _doc([_item(item_id="raj_photo", path="raj_photo.png")])
    frames, rep = dm.load_datumaro(_write(tmp_path, doc),
                                   fallback_session="Mix_raj_Batch_01")
    assert frames[0].session_id == "Mix_raj_Batch_01"
    assert not any(f["kind"] == "unresolvable_session" for f in rep.errors)


def test_the_fallback_never_overrides_a_session_the_filename_names(tmp_path):
    """A batch drawn from three drives must stay three sessions. One id over
    frames metres apart lets a split put near-copies on both sides of it."""
    doc = _doc([_item(item_id="vid3_20260108_103135_000012",
                      path="vid3_20260108_103135_000012.png")])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc),
                                 fallback_session="Mix_raj_Batch_01")
    assert frames[0].session_id == "vid3_20260108_103135"


def test_batch_session_id_is_derived_from_the_export_folder(tmp_path):
    p = tmp_path / "Mix_raj_Batch 01" / "annotations" / "default.json"
    assert dm.batch_session_id(p) == "Mix_raj_Batch_01"
    assert dm.batch_session_id(tmp_path / "Batch-02" / "default.json") \
        == "Batch_02"


def test_two_batches_get_different_session_ids(tmp_path):
    """One shared config value would have merged them, and that shows up only
    as a validation score that is too good."""
    a = dm.batch_session_id(tmp_path / "Mix_raj_Batch 01" / "annotations" / "d.json")
    b = dm.batch_session_id(tmp_path / "Mix_raj_Batch 02" / "annotations" / "d.json")
    assert a != b


def test_cvats_frame_000001_naming_does_not_become_a_session_called_frame(tmp_path):
    """CVAT renames uploaded images to frame_000001.png. That parses perfectly
    and yields the session id "frame" - a bucket EVERY such batch lands in,
    silently merging unrelated drives into one session. It broke the first real
    mixed build: the batch's frames became 'frame' and the holdout never
    matched."""
    doc = _doc([_item(item_id="frame_000001", path="frame_000001.png")])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc),
                                 fallback_session="Mix_raj_Batch_01")
    assert frames[0].session_id == "Mix_raj_Batch_01"


def test_a_real_session_name_still_beats_the_folder(tmp_path):
    """The rule is not 'the folder always wins' - it is 'the filename must
    actually name a session'. vid3_20260108_103135 does; frame does not."""
    doc = _doc([_item(item_id="vid3_20260108_103135_000012",
                      path="vid3_20260108_103135_000012.png")])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc),
                                 fallback_session="Mix_raj_Batch_01")
    assert frames[0].session_id == "vid3_20260108_103135"


def test_without_a_fallback_a_weak_id_is_still_better_than_nothing(tmp_path):
    """Only when there is a folder to prefer does the stricter test apply -
    otherwise the alternative is an unresolvable_session error."""
    doc = _doc([_item(item_id="frame_000001", path="frame_000001.png")])
    frames, _ = dm.load_datumaro(_write(tmp_path, doc))
    assert frames[0].session_id == "frame"


def test_two_frame_named_batches_do_not_merge(tmp_path):
    """The consequence that matters: without this they are one session, and a
    split puts near-copies of the same plant on both sides of it."""
    ids = []
    for folder in ("Batch 01", "Batch 02"):
        d = tmp_path / folder
        d.mkdir()
        doc = _doc([_item(item_id="frame_000001", path="frame_000001.png")])
        p = _write(d, doc)
        frames, _ = dm.load_datumaro(p, fallback_session=dm.batch_session_id(p))
        ids.append(frames[0].session_id)
    assert ids == ["Batch_01", "Batch_02"]
