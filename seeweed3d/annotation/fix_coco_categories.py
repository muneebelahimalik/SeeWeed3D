#!/usr/bin/env python3
"""
SeeWeed3D - repair category names in an OLD COCO export before importing it
=============================================================================
`common/ontology.py` renamed every class to `lower_snake_case`
("onion plant" -> "onion_plant", "ignore region" -> "ignore_region"). A
`instances_default.json` written by a SAM 3 prelabeler run BEFORE that rename
still carries the old spaced name.

Importing such a file into a CVAT task whose label schema already uses the
NEW name does not fail loudly - CVAT's "Upload annotations -> COCO 1.0" import
matches by category NAME, so an unrecognised one silently creates a SECOND,
duplicate label instead of filling the one your prelabels are meant to
correct. You would end up hand-relabelling everything from scratch without
knowing why the prelabels never showed up in the right place.

This rewrites `categories[*].name` in place to the current ontology name and
refuses to write anything it cannot confidently match - a class it does not
recognise is left for you to rename by hand rather than guessed at.

    python seeweed3d/annotation/fix_coco_categories.py \\
        --in  E:/Dataset_Vidalia/auto_labels_onion/vid3_20260108_132749/instances_default.json \\
        --out E:/Dataset_Vidalia/auto_labels_onion/vid3_20260108_132749/instances_default.json

--out may equal --in: the file is only overwritten after every category name
resolves, so a failed run never leaves a half-fixed file on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES, IGNORE_LABEL, LEP_LABEL  # noqa: E402

KNOWN_NAMES = set(CLASSES) | {IGNORE_LABEL, LEP_LABEL}

# Renames seen in the wild, from before common/ontology.py's snake_case rule.
# Extend this if another old export turns up a name not covered here - do not
# guess a mapping automatically, because a wrong guess would relabel a class
# to a DIFFERENT one instead of failing.
KNOWN_RENAMES = {
    "onion plant": "onion_plant",
    "ignore region": "ignore_region",
    "weed LEP": LEP_LABEL,
}


def resolve_name(name):
    """The current ontology name for a (possibly stale) category name, or
    None if it cannot be resolved with confidence."""
    if name in KNOWN_NAMES:
        return name
    if name in KNOWN_RENAMES:
        return KNOWN_RENAMES[name]
    # A conservative fallback: spaces -> underscores, but ONLY accepted if the
    # result is an actual ontology name. This catches an un-catalogued rename
    # of the same shape without guessing at anything genuinely unfamiliar.
    guess = name.strip().replace(" ", "_")
    if guess in KNOWN_NAMES:
        return guess
    return None


def fix_categories(coco):
    """Returns (fixed_coco, renamed[list of (old, new)], unresolved[list]).

    Never mutates the input; the caller decides whether the result is safe to
    write."""
    coco = dict(coco)
    cats = [dict(c) for c in coco.get("categories", [])]
    renamed, unresolved = [], []
    for c in cats:
        old = c.get("name", "")
        new = resolve_name(old)
        if new is None:
            unresolved.append(old)
        elif new != old:
            renamed.append((old, new))
            c["name"] = new
    coco["categories"] = cats
    return coco, renamed, unresolved


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True,
                   help="the instances_default.json to repair")
    p.add_argument("--out", required=True,
                   help="where to write the result; may equal --in")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change; write nothing")
    a = p.parse_args(argv)

    src = Path(a.inp)
    if not src.exists():
        raise SystemExit(f"ERROR: {src} not found.")
    try:
        coco = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: {src} is not valid JSON ({e}).")
    if not isinstance(coco.get("categories"), list):
        raise SystemExit(f"ERROR: {src} has no 'categories' array - is this "
                         f"actually a COCO file?")

    fixed, renamed, unresolved = fix_categories(coco)

    if unresolved:
        raise SystemExit(
            f"ERROR: {len(unresolved)} categor{'y is' if len(unresolved) == 1 else 'ies are'} "
            f"not in common/ontology.py and not a known pre-rename alias: "
            f"{unresolved}\n"
            f"Known classes: {sorted(KNOWN_NAMES)}\n"
            f"Known renames: {KNOWN_RENAMES}\n"
            f"Add the mapping to KNOWN_RENAMES if this is a genuine old name, "
            f"or fix the export if it is not - nothing was written.")

    if not renamed:
        print(f"No stale category names found in {src}. Nothing to do.")
        if not a.dry_run and Path(a.out) != src:
            Path(a.out).write_text(json.dumps(coco, indent=None), encoding="utf-8")
        return

    for old, new in renamed:
        print(f"  '{old}'  ->  '{new}'")
    if a.dry_run:
        print(f"\n[dry run] {len(renamed)} categor{'y' if len(renamed) == 1 else 'ies'} "
              f"would be renamed. Nothing written.")
        return

    Path(a.out).write_text(json.dumps(fixed, indent=None), encoding="utf-8")
    print(f"\n-> {a.out}  ({len(renamed)} categor"
          f"{'y' if len(renamed) == 1 else 'ies'} renamed, geometry untouched)")


if __name__ == "__main__":
    main()
