#!/usr/bin/env python3
"""
SeeWeed3D - multitask ingestion of verified CVAT annotations (Datumaro 1.0)
===========================================================================
Reads a CVAT **Datumaro 1.0** export of a mixed onion+weed scene and produces
the two training targets the supervised baseline needs, plus an integrity
report.

WHY DATUMARO AND NOT COCO
-------------------------
`annotation/cvat_roundtrip.py` reads COCO 1.0, which is correct for
segmentation alone but is lossy for this task: COCO has no representation for
CVAT's **shape groups**, so the link between a weed mask and its LEP point is
destroyed on export. It also drops per-shape attributes. Datumaro is CVAT's own
native format and preserves masks/polygons, point shapes, `group`, attributes
and ignore regions together, which is exactly the multitask contract here.
COCO remains the right format for the segmentation stage alone.

WHY THIS PARSES THE JSON DIRECTLY
---------------------------------
The Datumaro 1.0 on-disk format is a small, stable, documented JSON schema.
Parsing it here rather than importing the `datumaro` package means:

  * the unit suite runs with numpy + stdlib only, satisfying the project rule
    that tests need no GPU, no heavy dependency and no real dataset;
  * SAM 3's `numpy<2` pin cannot be broken by a transitive requirement of a
    training-only library. That pin is load-bearing for the annotation stage.

`datumaro` is still listed in requirements-training.txt and, when installed,
`cross_check_with_datumaro()` re-reads the same file through the official
library and compares counts, so the hand-rolled reader is verified against the
reference implementation rather than trusted.

OWNERSHIP IS BY GROUP ID, NEVER BY PROXIMITY
--------------------------------------------
A weed mask and its LEP are associated **only** through the Datumaro `group`
field (CVAT's shape group). Nearest-neighbour matching is offered strictly as a
*diagnostic suggestion* in the report and never modifies a training target: in
a dense frame the nearest crown is frequently the neighbouring plant, and a
mislabelled owner trains the model to aim a 60 W laser at the wrong tissue.

Note `group == 0` is Datumaro's "no group" sentinel, so it is treated as
ungrouped, not as a group whose id happens to be zero.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import (CLASSES, CATEGORY_ID, CROP_CLASS,  # noqa: E402
                             IGNORE_LABEL, LEP_LABEL)
from training.config import AnnotationContract  # noqa: E402

# Datumaro's sentinel for "this shape belongs to no group".
NO_GROUP = 0


class DatumaroFormatError(ValueError):
    """The export cannot be interpreted. Always raised with the offending file
    and a concrete instruction, never swallowed - a silently skipped annotation
    is a hole in the training target that nothing downstream can detect."""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class LEPAnnotation:
    """A verified leaf emergence point."""
    x: float
    y: float
    group_id: int
    visibility: str = "visible"
    attributes: dict = field(default_factory=dict)

    @property
    def uv(self):
        return (self.x, self.y)


@dataclass
class PlantInstance:
    """One annotated plant: a mask/polygon, its class, and (maybe) its LEP."""
    class_name: str
    polygons: list                      # list of flat [x1,y1,x2,y2,...]
    group_id: int
    attributes: dict = field(default_factory=dict)
    lep: Optional[LEPAnnotation] = None
    rle: Optional[dict] = None          # when the shape was exported as a mask
    _bbox: Optional[tuple] = None

    @property
    def is_crop(self):
        return self.class_name == CROP_CLASS

    @property
    def category_id(self):
        return CATEGORY_ID[self.class_name]

    @property
    def visibility(self):
        return str(self.attributes.get("lep_visibility", "visible"))

    @property
    def targetable(self):
        return str(self.attributes.get("targetable", "yes"))

    @property
    def growth_stage(self):
        return str(self.attributes.get("growth_stage", "unknown"))

    def bbox(self):
        """(x, y, w, h) over every polygon part, or None if unavailable."""
        if self._bbox is not None:
            return self._bbox
        pts = []
        for p in self.polygons:
            a = np.asarray(p, np.float64).reshape(-1, 2)
            if a.size:
                pts.append(a)
        if not pts:
            return None
        a = np.concatenate(pts, 0)
        x0, y0 = a.min(0)
        x1, y1 = a.max(0)
        self._bbox = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
        return self._bbox

    def area_px(self):
        """Shoelace area summed over parts. Cheap, and enough for the
        'is this a real annotation or a stray click' check."""
        total = 0.0
        for p in self.polygons:
            a = np.asarray(p, np.float64).reshape(-1, 2)
            if len(a) < 3:
                continue
            x, y = a[:, 0], a[:, 1]
            total += 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                                     np.dot(y, np.roll(x, -1))))
        return total


@dataclass
class FrameRecord:
    """One annotated image and everything attached to it."""
    item_id: str
    image_path: str
    width: int
    height: int
    session_id: str = ""
    instances: list = field(default_factory=list)     # PlantInstance
    ignore_regions: list = field(default_factory=list)  # list of flat polygons
    orphan_leps: list = field(default_factory=list)   # LEPAnnotation, no owner

    @property
    def weeds(self):
        return [i for i in self.instances if not i.is_crop]

    @property
    def onions(self):
        return [i for i in self.instances if i.is_crop]


@dataclass
class MultitaskDatasetReport:
    """Integrity findings. `errors` block training; `warnings` need review."""
    n_frames: int = 0
    n_instances: int = 0
    n_leps: int = 0
    per_class: dict = field(default_factory=dict)
    per_session: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    needs_correction: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors

    def add_error(self, frame, kind, detail):
        self.errors.append({"frame": frame, "kind": kind, "detail": detail})
        self.needs_correction.append({"frame": frame, "kind": kind,
                                      "detail": detail, "severity": "error"})

    def add_warning(self, frame, kind, detail):
        self.warnings.append({"frame": frame, "kind": kind, "detail": detail})
        self.needs_correction.append({"frame": frame, "kind": kind,
                                      "detail": detail, "severity": "warning"})

    def to_dict(self):
        return asdict(self)

    def summary(self):
        return (f"{self.n_frames} frames | {self.n_instances} instances | "
                f"{self.n_leps} LEPs | {len(self.errors)} errors | "
                f"{len(self.warnings)} warnings")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _label_names(doc, path):
    """Ordered label names from the categories block."""
    try:
        labels = doc["categories"]["label"]["labels"]
    except (KeyError, TypeError):
        raise DatumaroFormatError(
            f"{path}: no categories.label.labels block. This does not look like "
            f"a Datumaro 1.0 export. In CVAT choose Export -> 'Datumaro 1.0'.")
    if not labels:
        raise DatumaroFormatError(f"{path}: the label category list is empty.")
    return [str(l["name"]) for l in labels]


def _item_image(item, path):
    """(image_path, width, height). Datumaro has used both `image` and `media`;
    both are accepted so an export from either CVAT generation works."""
    blob = item.get("image") or item.get("media") or {}
    img_path = str(blob.get("path") or blob.get("filename") or item.get("id", ""))
    size = blob.get("size")
    h = w = 0
    if isinstance(size, (list, tuple)) and len(size) == 2:
        h, w = int(size[0]), int(size[1])       # Datumaro stores [height, width]
    return img_path, w, h


def _decode_rle(rle, path, item_id):
    """Uncompressed Datumaro/COCO RLE -> bool mask.

    Compressed (string `counts`) RLE needs pycocotools; it is attempted when
    available and otherwise fails loudly. A mask that cannot be decoded must
    never become an empty training target - that would teach the model the
    plant is background."""
    counts, size = rle.get("counts"), rle.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise DatumaroFormatError(
            f"{path}: item {item_id!r} has an RLE mask with no valid size.")
    h, w = int(size[0]), int(size[1])
    if isinstance(counts, (list, tuple)):
        flat = np.zeros(h * w, np.uint8)
        idx, val = 0, 0
        for run in counts:
            run = int(run)
            if val:
                flat[idx:idx + run] = 1
            idx += run
            val ^= 1
        return flat.reshape((h, w), order="F").astype(bool)
    try:
        from pycocotools import mask as coco_mask       # optional
    except ImportError:
        raise DatumaroFormatError(
            f"{path}: item {item_id!r} uses COMPRESSED RLE masks, which need "
            f"pycocotools to decode. Either `python -m pip install pycocotools`, "
            f"or re-export from CVAT with polygons instead of masks.")
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode()
    return coco_mask.decode(r).astype(bool)


def _mask_to_polygons(mask, min_points=3):
    """Outer contours of a decoded mask as flat polygons."""
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if len(c) >= min_points:
            out.append(c.reshape(-1).astype(float).tolist())
    return out


def _session_of(item_id, image_path, session_from):
    """Session id for a frame.

    Extraction names every pooled frame `<session_id>_<frame_idx>.png`, so the
    session is recoverable from the filename. An explicit map wins when given.
    Returning "" means unresolvable, which the caller turns into a hard error -
    a frame with no session cannot be placed in a leak-free split."""
    if session_from:
        for key in (item_id, Path(image_path).name, Path(image_path).stem):
            if key in session_from:
                return session_from[key]
    stem = Path(image_path or item_id).stem
    if "_" in stem:
        head = stem.rsplit("_", 1)
        if head[-1].isdigit():
            return head[0]
    return ""


def load_datumaro(json_path, contract=None, session_from=None, report=None):
    """Parse one Datumaro 1.0 annotation file into FrameRecords.

    Returns (frames, report). The report carries every integrity finding; the
    caller decides whether to proceed. Structural problems (bad file, unknown
    label under a strict contract) raise instead, because they mean the export
    itself is wrong."""
    contract = contract or AnnotationContract()
    report = report or MultitaskDatasetReport()
    path = Path(json_path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DatumaroFormatError(f"{path}: cannot read as JSON ({e}).")

    names = _label_names(doc, path)
    unknown = sorted(set(names) - set(CLASSES) - {LEP_LABEL, IGNORE_LABEL})
    if unknown and contract.strict_unknown_labels:
        raise DatumaroFormatError(
            f"{path}: labels not defined in common/ontology.py: {unknown}. "
            f"Known: {CLASSES + [LEP_LABEL, IGNORE_LABEL]}. Fix the CVAT label "
            f"schema (annotation/regen_cvat_labels.py regenerates it) or add the "
            f"class to the ontology - do not rename it here.")

    items = doc.get("items")
    if items is None:
        raise DatumaroFormatError(f"{path}: no 'items' array.")

    frames = []
    for item in items:
        frames.append(_parse_item(item, names, path, contract, session_from,
                                  report))
    return frames, report


def _parse_item(item, names, path, contract, session_from, report):
    item_id = str(item.get("id", ""))
    img_path, w, h = _item_image(item, path)
    session = _session_of(item_id, img_path, session_from)
    rec = FrameRecord(item_id=item_id, image_path=img_path, width=w, height=h,
                      session_id=session)
    if not session:
        report.add_error(item_id, "unresolvable_session",
                         "cannot derive a session id from the item id or image "
                         "path; splits would leak. Pass session_from={...}.")

    by_group_shape = defaultdict(list)      # group -> [PlantInstance]
    leps_by_group = defaultdict(list)       # group -> [LEPAnnotation]
    ungrouped_leps = []

    for ann in item.get("annotations", []):
        a_type = str(ann.get("type", ""))
        label_idx = ann.get("label_id")
        if label_idx is None or not (0 <= int(label_idx) < len(names)):
            report.add_error(item_id, "bad_label_id",
                             f"annotation {ann.get('id')} has label_id "
                             f"{label_idx!r}, outside the category list.")
            continue
        label = names[int(label_idx)]
        attrs = dict(ann.get("attributes") or {})
        group = int(ann.get("group", NO_GROUP) or NO_GROUP)

        if label == IGNORE_LABEL:
            if a_type == "polygon":
                rec.ignore_regions.append(list(ann.get("points", [])))
            elif a_type == "mask" and ann.get("rle"):
                m = _decode_rle(ann["rle"], path, item_id)
                rec.ignore_regions.extend(_mask_to_polygons(m))
            continue

        if label == LEP_LABEL or a_type == "points":
            pts = list(ann.get("points", []))
            if len(pts) < 2:
                report.add_error(item_id, "empty_lep",
                                 f"point annotation {ann.get('id')} has no coords.")
                continue
            lep = LEPAnnotation(x=float(pts[0]), y=float(pts[1]), group_id=group,
                                visibility=str(attrs.get("lep_visibility",
                                                         "visible")),
                                attributes=attrs)
            if group == NO_GROUP:
                ungrouped_leps.append(lep)
            else:
                leps_by_group[group].append(lep)
            continue

        if label not in CLASSES:
            report.add_error(item_id, "unknown_label",
                             f"'{label}' is not in common/ontology.py CLASSES.")
            continue

        polys, rle = [], None
        if a_type == "polygon":
            p = list(ann.get("points", []))
            if len(p) >= 6:
                polys = [p]
        elif a_type == "mask" and ann.get("rle"):
            rle = ann["rle"]
            polys = _mask_to_polygons(_decode_rle(rle, path, item_id))
        elif a_type == "bbox" and ann.get("bbox"):
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            polys = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
            report.add_warning(item_id, "bbox_only_instance",
                               f"'{label}' was annotated as a box, not a mask; "
                               f"its segmentation target is the box rectangle.")
        else:
            report.add_warning(item_id, "unsupported_shape",
                               f"annotation {ann.get('id')} of type '{a_type}' "
                               f"for '{label}' was not converted.")
            continue

        if not polys:
            report.add_error(item_id, "empty_shape",
                             f"'{label}' annotation {ann.get('id')} produced no "
                             f"polygon (too few points, or an empty mask).")
            continue

        inst = PlantInstance(class_name=label, polygons=polys, group_id=group,
                             attributes=attrs, rle=rle)
        rec.instances.append(inst)
        if group != NO_GROUP:
            by_group_shape[group].append(inst)

    _attach_leps(rec, by_group_shape, leps_by_group, ungrouped_leps, report,
                 contract)
    return rec


def _attach_leps(rec, by_group_shape, leps_by_group, ungrouped_leps, report,
                 contract):
    """Bind each LEP to its owner strictly by group id."""
    for group, leps in leps_by_group.items():
        owners = by_group_shape.get(group, [])
        if not owners:
            for lep in leps:
                rec.orphan_leps.append(lep)
                report.add_error(rec.item_id, "lep_group_has_no_shape",
                                 f"group {group} contains an LEP but no plant "
                                 f"shape. Group the point WITH its weed mask in "
                                 f"CVAT.")
            continue
        if len(owners) > 1:
            report.add_error(rec.item_id, "duplicate_group_id",
                             f"group {group} contains {len(owners)} plant shapes; "
                             f"LEP ownership is ambiguous. Give each plant its "
                             f"own group.")
        if len(leps) > 1:
            report.add_error(rec.item_id, "multiple_leps_in_group",
                             f"group {group} contains {len(leps)} LEP points; a "
                             f"plant has exactly one growth point.")
        owner = owners[0]
        if owner.is_crop:
            report.add_error(rec.item_id, "lep_on_onion",
                             f"group {group}: an LEP is attached to an "
                             f"'{CROP_CLASS}'. The crop is never a target.")
            continue
        owner.lep = leps[0]

    for lep in ungrouped_leps:
        rec.orphan_leps.append(lep)
        report.add_error(rec.item_id, "ungrouped_lep",
                         "an LEP point has no group id, so its owning weed is "
                         "unknown. It is NOT auto-assigned - group it in CVAT.")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _point_in_polygons(x, y, polygons, tolerance_px):
    """(inside, distance_px) of a point w.r.t. a multi-part polygon."""
    import cv2
    best = -1e18
    for p in polygons:
        a = np.asarray(p, np.float32).reshape(-1, 2)
        if len(a) < 3:
            continue
        d = cv2.pointPolygonTest(a, (float(x), float(y)), True)
        best = max(best, d)
    if best <= -1e17:
        return False, float("inf")
    return bool(best >= -tolerance_px), float(-best if best < 0 else 0.0)


def validate_frames(frames, contract=None, report=None, suggest_owners=True):
    """Apply the annotation contract. Returns the report."""
    contract = contract or AnnotationContract()
    report = report or MultitaskDatasetReport()
    per_class, per_session = Counter(), defaultdict(Counter)

    for rec in frames:
        report.n_frames += 1
        for inst in rec.instances:
            report.n_instances += 1
            per_class[inst.class_name] += 1
            per_session[rec.session_id][inst.class_name] += 1

            area = inst.area_px()
            if area < contract.min_instance_area_px:
                report.add_warning(rec.item_id, "tiny_instance",
                                   f"'{inst.class_name}' area {area:.0f}px is "
                                   f"below {contract.min_instance_area_px}px; "
                                   f"verify it is a plant and not a stray click.")

            if inst.is_crop:
                if inst.lep is not None:
                    report.add_error(rec.item_id, "lep_on_onion",
                                     "an onion_plant carries a weed LEP.")
                continue

            cluster = inst.class_name in contract.non_targetable_classes
            needs = (inst.visibility in contract.visibility_requiring_lep
                     and inst.targetable == "yes" and not cluster)

            if inst.lep is not None:
                report.n_leps += 1
                if cluster:
                    report.add_error(rec.item_id, "lep_on_cluster",
                                     f"'{inst.class_name}' has an LEP, but a "
                                     f"cluster has no single growth point. "
                                     f"Either split it into plants or remove "
                                     f"the point.")
                inside, dist = _point_in_polygons(
                    inst.lep.x, inst.lep.y, inst.polygons,
                    contract.lep_inside_mask_tolerance_px)
                if not inside:
                    report.add_error(
                        rec.item_id, "lep_outside_owning_mask",
                        f"LEP for '{inst.class_name}' (group "
                        f"{inst.group_id}) lies {dist:.1f}px outside its own "
                        f"mask, beyond the "
                        f"{contract.lep_inside_mask_tolerance_px}px tolerance. "
                        f"It may be grouped with the wrong plant.")
                elif dist > 0:
                    report.add_warning(rec.item_id, "lep_near_mask_edge",
                                       f"LEP is {dist:.1f}px outside its mask, "
                                       f"within tolerance.")
            elif needs:
                report.add_error(
                    rec.item_id, "missing_lep",
                    f"'{inst.class_name}' (group {inst.group_id}) is "
                    f"{inst.visibility} and targetable but has no grouped LEP.")

        if rec.orphan_leps and suggest_owners:
            _suggest_owners(rec, report)

    report.per_class = dict(per_class)
    report.per_session = {s: dict(c) for s, c in per_session.items()}
    return report


def _suggest_owners(rec, report):
    """DIAGNOSTIC ONLY: name the nearest weed to each orphan LEP.

    Deliberately writes to `suggestions` and never touches an instance. In a
    dense frame the nearest crown is often the neighbour, and silently binding
    it would produce a confidently wrong laser target."""
    weeds = [i for i in rec.weeds if i.bbox()]
    for lep in rec.orphan_leps:
        best, best_d = None, float("inf")
        for inst in weeds:
            x, y, w, h = inst.bbox()
            cx, cy = x + w / 2.0, y + h / 2.0
            d = float(np.hypot(cx - lep.x, cy - lep.y))
            if d < best_d:
                best, best_d = inst, d
        if best is not None:
            report.suggestions.append({
                "frame": rec.item_id, "lep": [lep.x, lep.y],
                "nearest_class": best.class_name,
                "nearest_group": best.group_id,
                "distance_px": round(best_d, 1),
                "note": "SUGGESTION ONLY - not applied. Group them in CVAT."})


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def to_yolo_segmentation(rec, classes=None):
    """One Ultralytics segmentation label file body for a frame.

    Format: `class_idx x1 y1 x2 y2 ...` with coordinates normalised to [0,1].
    Index is into `classes`, which defaults to the full ontology. A build that
    dropped classes passes its reduced ACTIVE list so indices stay contiguous -
    a gap in the label indices would silently shift every class above it."""
    if not rec.width or not rec.height:
        raise DatumaroFormatError(
            f"frame {rec.item_id!r} has no image size; YOLO labels are "
            f"normalised and cannot be written. Re-export from CVAT with image "
            f"info, or pass explicit sizes.")
    names = list(classes) if classes else list(CLASSES)
    lines = []
    for inst in rec.instances:
        if inst.class_name not in names:
            continue
        idx = names.index(inst.class_name)
        for poly in inst.polygons:
            a = np.asarray(poly, np.float64).reshape(-1, 2)
            if len(a) < 3:
                continue
            a[:, 0] = np.clip(a[:, 0] / rec.width, 0.0, 1.0)
            a[:, 1] = np.clip(a[:, 1] / rec.height, 0.0, 1.0)
            coords = " ".join(f"{v:.6f}" for v in a.reshape(-1))
            lines.append(f"{idx} {coords}")
    return "\n".join(lines)


def to_lep_manifest(frames):
    """Per-instance LEP training rows.

    A MANIFEST, not a copied dataset: each row points at the existing image via
    a relative path and carries the polygon, so no large image tree is
    duplicated and Windows paths survive (paths are stored posix-style and
    resolved against a caller-supplied root)."""
    rows = []
    for rec in frames:
        for k, inst in enumerate(rec.instances):
            if inst.is_crop or inst.lep is None:
                continue
            bb = inst.bbox()
            if bb is None:
                continue
            rows.append({
                "session_id": rec.session_id,
                "item_id": rec.item_id,
                "image_path": Path(rec.image_path).as_posix(),
                "width": rec.width, "height": rec.height,
                "instance_index": k,
                "group_id": inst.group_id,
                "class_name": inst.class_name,
                "class_index": CLASSES.index(inst.class_name),
                "bbox_xywh": [round(v, 2) for v in bb],
                "lep_x": round(inst.lep.x, 3), "lep_y": round(inst.lep.y, 3),
                "lep_visibility": inst.visibility,
                "targetable": inst.targetable,
                "growth_stage": inst.growth_stage,
                "polygons": [[round(float(v), 2) for v in p]
                             for p in inst.polygons],
                "attributes": dict(inst.attributes),
            })
    return rows


def cross_check_with_datumaro(json_path, frames):
    """Re-read the same export through the official `datumaro` package and
    compare item/annotation counts against this module's reader.

    Returns None when datumaro is not installed, so it never blocks the
    lightweight test suite. When it IS installed this is what keeps the
    hand-rolled reader honest instead of merely assumed correct."""
    try:
        import datumaro as dm
    except ImportError:
        return None
    ds = dm.Dataset.import_from(str(Path(json_path).parent.parent), "datumaro")
    ref_items = 0
    ref_anns = 0
    for it in ds:
        ref_items += 1
        ref_anns += len(it.annotations)
    ours_anns = sum(len(f.instances) + len(f.ignore_regions) + len(f.orphan_leps)
                    + sum(1 for i in f.instances if i.lep) for f in frames)
    return {"datumaro_items": ref_items, "our_items": len(frames),
            "datumaro_annotations": ref_anns, "our_annotations": ours_anns,
            "items_match": ref_items == len(frames)}
