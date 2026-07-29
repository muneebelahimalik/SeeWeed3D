#!/usr/bin/env python3
"""
SeeWeed3D - regenerate CVAT label schema files without rerunning SAM 3.

The label schema (weed_cvat_labels.json / onion_cvat_labels.json) is pure data
derived from common/ontology.py - it does not depend on any per-frame
inference. When the ontology changes (a class renamed, an attribute fixed) the
label file goes stale, but re-running the full SAM 3 prelabeler just to get a
fresh JSON can cost hours. This script rewrites the label file inside every
session folder that already exists, in seconds, and touches nothing else -
not instances_default.json, not masks, not previews, not cvat_ready/.

Run this after pulling a change to common/ontology.py or the label schemas,
instead of re-running a full prelabeling pass.
"""

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # for common.ontology
sys.path.insert(0, str(_HERE))          # for the sibling cvat_roundtrip import below
from common.ontology import cvat_labels  # noqa: E402

# #############################################################################
# ##   DATASET_ROOT  -  same DATASET_ROOT you gave the prelabeler(s)         ##
# #############################################################################

DATASET_ROOT = r"E:\Dataset_Vidalia"

CONFIG = {
    "DATASET_ROOT": DATASET_ROOT,
    # Which auto-label trees to refresh, and what to name the file in each
    # existing session folder. Onion labels are imported lazily below so this
    # module has no import-time dependency on cvat_roundtrip.py.
    "TARGETS": {
        "auto_labels_weeds": "weed_cvat_labels.json",
        "auto_labels_onion": "onion_cvat_labels.json",
    },
}


def onion_labels():
    """Imported lazily so this script has no hard dependency on
    cvat_roundtrip.py's own CONFIG block being valid on this machine."""
    from cvat_roundtrip import ONION_CVAT_LABELS
    return ONION_CVAT_LABELS


def schema_for(subdir_name):
    if subdir_name == "auto_labels_weeds":
        return cvat_labels()
    if subdir_name == "auto_labels_onion":
        return onion_labels()
    raise ValueError(f"no label schema known for {subdir_name}")


def regenerate(cfg):
    root = Path(cfg["DATASET_ROOT"])
    written, skipped = [], []
    for subdir_name, filename in cfg["TARGETS"].items():
        parent = root / subdir_name
        if not parent.exists():
            print(f"  [skip] {parent} does not exist - nothing to refresh")
            continue
        schema = schema_for(subdir_name)
        text = json.dumps(schema, indent=2)
        for session_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
            target = session_dir / filename
            if not (session_dir / "instances_default.json").exists():
                skipped.append(target)
                continue
            target.write_text(text)
            written.append(target)
            print(f"  wrote {target}")
    if skipped:
        print(f"\nSkipped {len(skipped)} folder(s) with no instances_default.json "
              f"(not a processed session).")
    print(f"\nRegenerated {len(written)} label file(s). No inference re-run; "
          f"instances_default.json, masks, previews and cvat_ready/ untouched.")
    return written


def main():
    regenerate(CONFIG)


if __name__ == "__main__":
    main()
