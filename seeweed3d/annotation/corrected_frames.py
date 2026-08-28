#!/usr/bin/env python3
"""
SeeWeed3D - which frames did a person ACTUALLY correct? Diff export vs prelabel.

    python -m seeweed3d.annotation.corrected_frames

THE FAILURE THIS EXISTS FOR
---------------------------
A CVAT task pre-loaded with prelabels has annotations on EVERY frame, including
the ones nobody opened. Correct 75 of 393 and export, and the export contains
393 annotated frames - 318 of which are the model's own output. Merge that as
hand_corrected and the model is trained on its own predictions while the
manifest says a person verified them.

Nothing downstream can detect it. The masks are plausible, the classes are
plausible, the file is well-formed, and the only difference between a verified
frame and a machine one is whether a human looked - which is not recorded
anywhere in the export.

So compare the export against the prelabel COCO that was IMPORTED into the task.
A frame whose annotations differ was edited. That is a fact about the files, not
a claim about anyone's memory of which frames they did.

THE ONE THING IT CANNOT TELL YOU
---------------------------------
A frame someone opened, examined, and correctly judged to need no change is
IDENTICAL to one never opened. There is no way to tell them apart from the
files, so this reports a LOWER BOUND on what was reviewed.

If you corrected 75 and this finds 68, the other 7 were verified-unchanged and
only you know which. Add them by hand - and if you cannot remember, leave them
out. An unreviewed frame in a hand_corrected build costs more than a verified
frame left on the floor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Polygon coordinates are rounded to this many pixels before comparing.
#:
#: CVAT round-trips coordinates through its own float formatting, so a frame
#: nobody touched can come back with 412.0 written as 412.00001. Comparing raw
#: floats reports every frame as edited, which is the same as reporting none.
#: Half a pixel is far below any correction a person makes with a mouse and far
#: above any formatting difference.
COORD_TOLERANCE_PX = 0.5


def _round(v):
    return round(float(v) / COORD_TOLERANCE_PX)


def frame_signature(instances):
    """An order-independent signature for one frame's annotations.

    A SET of per-instance signatures, because CVAT renumbers annotation ids and
    reorders them on export - comparing lists would call every frame edited.

    Class is part of it: correcting only the label, which is the single most
    common weed correction, moves no coordinate at all."""
    sig = set()
    for cls, polys in instances:
        pts = tuple(sorted(
            tuple(_round(v) for v in poly) for poly in polys if poly))
        sig.add((str(cls), pts))
    return frozenset(sig)


def _from_coco(doc):
    """{item_id: [(class_name, [polygon, ...])]} from a COCO 1.0 document."""
    cat = {c["id"]: c["name"] for c in doc.get("categories", [])}
    by_img = {}
    for im in doc.get("images", []):
        by_img[im["id"]] = (Path(im["file_name"]).stem, [])
    for a in doc.get("annotations", []):
        if a.get("image_id") not in by_img:
            continue
        seg = a.get("segmentation") or []
        polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
        by_img[a["image_id"]][1].append(
            (cat.get(a.get("category_id"), "?"), polys))
    return {name: inst for name, inst in by_img.values()}


def _from_datumaro(doc):
    """The same, from a Datumaro 1.0 document.

    CVAT exports Datumaro and the prelabels went in as COCO, so this has to
    read both - a comparison that only worked one way would be unusable in the
    only direction the round trip actually goes."""
    names = {}
    for cat in (doc.get("categories") or {}).get("label", {}).get("labels", []):
        names[len(names)] = cat.get("name", "?")
    out = {}
    for item in doc.get("items", []):
        inst = []
        for a in item.get("annotations", []):
            if a.get("type") != "polygon":
                continue
            pts = a.get("points") or []
            if len(pts) >= 6:
                inst.append((names.get(a.get("label_id"), "?"), [list(pts)]))
        out[Path(str(item.get("id", ""))).stem] = inst
    return out


def load_annotations(path):
    """Item id -> instances, from COCO 1.0 or Datumaro 1.0, detected by shape."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if "items" in doc:
        return _from_datumaro(doc)
    if "images" in doc:
        return _from_coco(doc)
    raise SystemExit(
        f"ERROR: {path} is neither COCO 1.0 ('images') nor Datumaro 1.0 "
        f"('items').")


def compare(before, after):
    """What changed between the prelabels and the export.

    Returns a dict of sorted item-id lists. `unchanged` is the one to read
    carefully: it holds both the frames nobody opened AND the frames someone
    checked and correctly left alone, and nothing here can separate them."""
    b, a = load_annotations(before), load_annotations(after)
    common = set(b) & set(a)
    changed = sorted(f for f in common
                     if frame_signature(b[f]) != frame_signature(a[f]))
    unchanged = sorted(common - set(changed))
    return {"changed": changed, "unchanged": unchanged,
            "only_in_export": sorted(set(a) - set(b)),
            "only_in_prelabels": sorted(set(b) - set(a))}


def format_report(result, claimed=None):
    """The readout, with the lower-bound caveat attached to the number."""
    n_ch, n_un = len(result["changed"]), len(result["unchanged"])
    L = ["",
         f"  {n_ch + n_un} frame(s) in both files",
         f"    edited      {n_ch:>5}",
         f"    unchanged   {n_un:>5}   (never opened, OR opened and correctly "
         f"left alone)"]
    if result["only_in_export"]:
        L.append(f"    new         {len(result['only_in_export']):>5}   in the "
                 f"export but not the prelabels")
    if result["only_in_prelabels"]:
        L.append(f"    missing     {len(result['only_in_prelabels']):>5}   in "
                 f"the prelabels but not the export")
    L += ["",
          "  'edited' is a LOWER BOUND on what was reviewed. A frame opened,",
          "  examined and correctly left alone is identical to one never",
          "  opened, and no file records the difference."]
    if claimed is not None and claimed > n_ch:
        L += ["",
              f"  [!] You said {claimed} and the files show {n_ch} edited.",
              f"      The other {claimed - n_ch} were verified-unchanged, and "
              f"only you know which.",
              f"      Add them by hand. If you cannot remember, LEAVE THEM "
              f"OUT - an unreviewed",
              f"      frame in a hand_corrected build costs more than a "
              f"verified one left out."]
    elif claimed is not None and claimed < n_ch:
        L += ["",
              f"  [!] You said {claimed} but {n_ch} frames differ. Something "
              f"edited more than you did -",
              f"      a stray drag, or the wrong prelabel file compared. Check "
              f"before merging."]
    return "\n".join(L)


def write_include_file(path, session, item_ids):
    """An @file for INCLUDE_FRAMES: one session-scoped item id per line.

    Item ids rather than positions. Positions are 1-based counts within a
    session and a merge that renumbers them silently redirects a carefully
    checked selection at different frames; an item id names the frame itself."""
    lines = ["# SeeWeed3D - frames a person actually corrected.",
             "# Generated by annotation/corrected_frames.py - do not hand-edit",
             "# without re-reading the lower-bound caveat in that module.",
             f"# session: {session}", ""]
    lines += [f"{session}:{i}" if session else str(i) for i in item_ids]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(item_ids)


# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

#: The prelabel COCO that was IMPORTED into the CVAT task - the one the batch
#: folder carries, NOT a later re-run's. Comparing against different prelabels
#: reports every frame as edited.
PRELABELS = r"E:\Dataset_Vidalia\runs_1_only_weeds\weeds_r0\selftrain_20260827_2305\vid3_20260108_110444\accept\instances_default.json"

#: What you EXPORTED from CVAT (Datumaro 1.0, or COCO 1.0 - either is read).
EXPORT = r"E:\Dataset_Vidalia\Weeds_20260108_1\sessions\vid3_20260108_110444\annotations\default.json"

#: The session these frames belong to, used to scope the INCLUDE_FRAMES tokens.
SESSION = "vid3_20260108_110444"

#: Where to write the @file. Point INCLUDE_FRAMES at it:
#:     INCLUDE_FRAMES = "@E:\\...\\corrected_frames.txt"
OUT_FILE = r"E:\Dataset_Vidalia\datasets\corrected_vid3_20260108_110444.txt"

#: How many you believe you corrected. The run compares and says so if the
#: files disagree. None to skip the check.
CLAIMED = None

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################


def main():
    for label, p in (("PRELABELS", PRELABELS), ("EXPORT", EXPORT)):
        if not Path(p).is_file():
            raise SystemExit(f"ERROR: {label} not found: {p}")
    result = compare(PRELABELS, EXPORT)
    print(format_report(result, CLAIMED))
    n = write_include_file(OUT_FILE, SESSION, result["changed"])
    print(f"\n  -> {OUT_FILE}  ({n} frame(s))")
    print(f"\n  In training/datasets/weeds.py:")
    print(f'      INCLUDE_FRAMES = "@{OUT_FILE}"')
    print(f"\n  Then run the build with LIST_FRAMES = True FIRST and confirm "
          f"it selects\n"
          f"  {n} frames from {SESSION} before you train on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
