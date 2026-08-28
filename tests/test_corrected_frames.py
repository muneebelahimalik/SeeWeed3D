"""Which frames did a person actually correct?

A CVAT task pre-loaded with prelabels has annotations on EVERY frame, including
the ones nobody opened. Correct 75 of 393 and export, and 318 of the exported
frames are the model's own output. Merged as hand_corrected, the model trains on
its own predictions while the manifest says a person verified them - and nothing
downstream can detect it, because the only difference between a verified frame
and a machine one is whether a human looked, which no file records.

So this compares the export against the prelabels that went in. It is a
cross-check on a list someone already has, not a replacement for knowing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from annotation import corrected_frames as cf   # noqa: E402

BOX = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
NOISY = [10.00001, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 49.9999]
MOVED = [10.0, 10.0, 90.0, 10.0, 90.0, 50.0, 10.0, 50.0]
CATS = {"a": 1, "b": 2}


def coco(frames):
    imgs, anns, i = [], [], 1
    for n, (name, insts) in enumerate(frames.items(), 1):
        imgs.append({"id": n, "file_name": name + ".png",
                     "height": 80, "width": 80})
        for cls, poly in insts:
            anns.append({"id": i, "image_id": n, "category_id": CATS[cls],
                         "segmentation": [poly]})
            i += 1
    return {"images": imgs, "annotations": anns,
            "categories": [{"id": v, "name": k} for k, v in CATS.items()]}


def write(tmp_path, name, frames):
    p = tmp_path / name
    p.write_text(json.dumps(coco(frames)))
    return p


def diff(tmp_path, before, after):
    return cf.compare(write(tmp_path, "b.json", before),
                      write(tmp_path, "a.json", after))


# --------------------------------------------------------------------------- #
# What counts as edited
# --------------------------------------------------------------------------- #
def test_an_untouched_frame_is_not_reported_as_edited(tmp_path):
    """THE case that would make this useless. CVAT round-trips coordinates
    through its own float formatting, so a frame nobody opened comes back with
    412.0 written as 412.00001. Reporting every frame as edited is the same as
    reporting none."""
    r = diff(tmp_path, {"f1": [("a", BOX)]}, {"f1": [("a", NOISY)]})
    assert r["changed"] == [] and r["unchanged"] == ["f1"]


def test_a_class_change_alone_counts(tmp_path):
    """The most common weed correction - primrose vs radish - moves no
    coordinate at all. Comparing geometry only would miss every one."""
    r = diff(tmp_path, {"f1": [("a", BOX)]}, {"f1": [("b", BOX)]})
    assert r["changed"] == ["f1"]


def test_a_moved_boundary_counts(tmp_path):
    r = diff(tmp_path, {"f1": [("a", BOX)]}, {"f1": [("a", MOVED)]})
    assert r["changed"] == ["f1"]


def test_an_added_instance_counts(tmp_path):
    """A weed the model missed and a person drew - the single most valuable
    correction in this project."""
    r = diff(tmp_path, {"f1": [("a", BOX)]},
             {"f1": [("a", BOX), ("a", MOVED)]})
    assert r["changed"] == ["f1"]


def test_a_deleted_instance_counts(tmp_path):
    r = diff(tmp_path, {"f1": [("a", BOX), ("a", MOVED)]},
             {"f1": [("a", BOX)]})
    assert r["changed"] == ["f1"]


def test_reordered_instances_are_not_an_edit(tmp_path):
    """CVAT renumbers and reorders annotations on export. Comparing lists would
    call every frame edited."""
    r = diff(tmp_path, {"f1": [("a", BOX), ("b", MOVED)]},
             {"f1": [("b", MOVED), ("a", BOX)]})
    assert r["changed"] == []


def test_a_frame_with_everything_deleted_counts(tmp_path):
    """Deleting every mask is a real judgement - 'the model found nothing real
    here' - and an empty frame must not read as an untouched one."""
    r = diff(tmp_path, {"f1": [("a", BOX)]}, {"f1": []})
    assert r["changed"] == ["f1"]


def test_only_the_frames_in_both_files_are_compared(tmp_path):
    r = diff(tmp_path, {"f1": [("a", BOX)], "gone": [("a", BOX)]},
             {"f1": [("a", BOX)], "new": [("a", BOX)]})
    assert r["only_in_prelabels"] == ["gone"]
    assert r["only_in_export"] == ["new"]


def test_a_sub_tolerance_nudge_is_not_an_edit(tmp_path):
    """Below what anyone can do with a mouse, above any formatting difference."""
    small = [v + cf.COORD_TOLERANCE_PX / 4 for v in BOX]
    r = diff(tmp_path, {"f1": [("a", BOX)]}, {"f1": [("a", small)]})
    assert r["changed"] == []


def test_a_deliberate_nudge_is_an_edit(tmp_path):
    big = [v + cf.COORD_TOLERANCE_PX * 4 for v in BOX]
    r = diff(tmp_path, {"f1": [("a", BOX)]}, {"f1": [("a", big)]})
    assert r["changed"] == ["f1"]


# --------------------------------------------------------------------------- #
# Reading both formats, because the round trip goes COCO in, Datumaro out
# --------------------------------------------------------------------------- #
def test_it_reads_a_datumaro_export(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({
        "categories": {"label": {"labels": [{"name": "a"}, {"name": "b"}]}},
        "items": [{"id": "f1", "annotations": [
            {"type": "polygon", "label_id": 0, "points": BOX}]}]}))
    assert cf.load_annotations(p) == {"f1": [("a", [BOX])]}


def test_coco_in_and_datumaro_out_compare(tmp_path):
    """The only direction the round trip actually goes. A comparison that
    worked one way only would be unusable."""
    b = write(tmp_path, "b.json", {"f1": [("a", BOX)]})
    a = tmp_path / "a.json"
    a.write_text(json.dumps({
        "categories": {"label": {"labels": [{"name": "a"}, {"name": "b"}]}},
        "items": [{"id": "f1", "annotations": [
            {"type": "polygon", "label_id": 1, "points": BOX}]}]}))
    assert cf.compare(b, a)["changed"] == ["f1"], "label 1 is 'b', not 'a'"


def test_non_polygon_datumaro_shapes_are_ignored(tmp_path):
    """LEP points are annotations too, and a task that has them is not a task
    whose masks were all edited."""
    p = tmp_path / "d.json"
    p.write_text(json.dumps({
        "categories": {"label": {"labels": [{"name": "a"}]}},
        "items": [{"id": "f1", "annotations": [
            {"type": "polygon", "label_id": 0, "points": BOX},
            {"type": "points", "label_id": 0, "points": [20.0, 20.0]}]}]}))
    assert cf.load_annotations(p) == {"f1": [("a", [BOX])]}


def test_an_unrecognised_file_says_which_two_it_wanted(tmp_path):
    import pytest
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"something": []}))
    with pytest.raises(SystemExit, match="COCO 1.0"):
        cf.load_annotations(p)


# --------------------------------------------------------------------------- #
# The report, and the caveat that has to travel with the number
# --------------------------------------------------------------------------- #
def test_the_report_says_edited_is_a_lower_bound():
    """A frame opened, examined and correctly left alone is identical to one
    never opened. Reporting the count without that turns it into a claim the
    files cannot support."""
    text = cf.format_report({"changed": ["a"], "unchanged": ["b"],
                             "only_in_export": [], "only_in_prelabels": []})
    assert "LOWER BOUND" in text
    assert "correctly left alone" in text


def test_claiming_more_than_the_files_show_is_explained():
    r = {"changed": ["a"] * 68, "unchanged": [], "only_in_export": [],
         "only_in_prelabels": []}
    text = cf.format_report(r, claimed=75)
    assert "You said 75" in text and "68" in text
    assert "LEAVE THEM OUT" in text, "the safe direction has to be named"


def test_claiming_fewer_than_the_files_show_is_a_different_warning():
    """More edits than you made means a stray drag or the wrong prelabel file -
    and it is not fixed by adding frames to a list."""
    r = {"changed": ["a"] * 90, "unchanged": [], "only_in_export": [],
         "only_in_prelabels": []}
    text = cf.format_report(r, claimed=75)
    assert "edited more than you did" in text


def test_no_claim_means_no_comparison_line():
    r = {"changed": ["a"], "unchanged": [], "only_in_export": [],
         "only_in_prelabels": []}
    assert "You said" not in cf.format_report(r)


# --------------------------------------------------------------------------- #
# The @file INCLUDE_FRAMES reads
# --------------------------------------------------------------------------- #
def test_the_include_file_scopes_every_id_to_its_session(tmp_path):
    """Positions restart in each session, so an unscoped token applies to every
    session and would pull in frames from drives nobody corrected."""
    p = tmp_path / "inc.txt"
    cf.write_include_file(p, "vid3_x", ["f1", "f2"])
    body = [l for l in p.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    assert body == ["vid3_x:f1", "vid3_x:f2"]


def test_the_include_file_is_parsed_by_the_build(tmp_path):
    """The two ends of this are in different modules and nothing else makes
    them agree on the syntax."""
    from training.prepare_dataset import parse_frame_spec
    p = tmp_path / "inc.txt"
    cf.write_include_file(p, "vid3_x", ["frame_0007", "frame_0009"])
    groups = parse_frame_spec(f"@{p}")
    assert "vid3_x" in groups
    _, patterns = groups["vid3_x"]
    assert "frame_0007" in patterns and "frame_0009" in patterns


def test_the_include_file_says_where_it_came_from(tmp_path):
    """Hand-editing it without reading the lower-bound caveat is how an
    unreviewed frame gets added back."""
    p = tmp_path / "inc.txt"
    cf.write_include_file(p, "s", ["f1"])
    head = p.read_text()
    assert "corrected_frames.py" in head and "do not hand-edit" in head
