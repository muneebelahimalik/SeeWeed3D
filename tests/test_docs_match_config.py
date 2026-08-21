"""The prelabeling record must not drift from the code it describes.

docs/sam_prelabeling.md quotes ~40 config values as the concrete record of what
the pipeline does. A document that says MIN_INSTANCE_AREA_PX is 250 while the
code says 700 is worse than no document: it is read as evidence, and it is the
kind of wrong answer nothing else in this repo would catch.

The values were checked by hand once. This keeps them checked.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "seeweed3d") not in sys.path:
    sys.path.insert(0, str(ROOT / "seeweed3d"))

from annotation.prelabel_weeds_sam3 import CONFIG  # noqa: E402

DOC = ROOT / "docs" / "sam_prelabeling.md"


def quoted_values(doc, key):
    """Every value the doc places next to `key`, in prose, tables or code."""
    out = []
    for m in re.finditer(re.escape(key) + r"`?[\"']?\s*[:|]\s*`?([^\s`,|]+)", doc):
        out.append(m.group(1).strip().rstrip('.,`"\'').lstrip('"\''))
    return out


def test_the_doc_exists_and_is_linked_from_the_readme():
    assert DOC.is_file()
    assert "sam_prelabeling.md" in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("key", sorted(
    k for k, v in CONFIG.items() if isinstance(v, (int, float, bool, str))))
def test_every_config_value_the_doc_quotes_matches_the_code(key):
    doc = DOC.read_text(encoding="utf-8")
    want = CONFIG[key]
    for got in quoted_values(doc, key):
        if got in ("", "-"):
            continue
        # Compare numerically where both sides are numbers, so 0.010 and 0.01
        # agree - the source writes one and arithmetic writes the other.
        try:
            assert float(got) == float(want), f"{key}: doc {got!r} vs code {want!r}"
            continue
        except (TypeError, ValueError):
            pass
        assert got == str(want), f"{key}: doc {got!r} vs code {want!r}"


def test_the_decisive_settings_are_stated_at_all():
    """A record that omits the settings the field decisions were about is not a
    record of those decisions."""
    doc = DOC.read_text(encoding="utf-8")
    for key in ("MIN_INSTANCE_AREA_PX", "SPLIT_TOUCHING_INSTANCES",
                "RECOVER_MISSED_PLANTS", "BOUNDARY_REFINE_BAND_PX",
                "BOUNDARY_REFINE_MAX_AREA_PX", "EXEMPLAR_MIN_VEG_SCORE",
                "USE_DEPTH_HEIGHT", "CLUSTER_MIN_PEAKS", "GRASS_MIN_ASPECT"):
        assert key in doc, f"{key} is not mentioned in the record"
