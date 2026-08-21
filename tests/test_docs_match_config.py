"""The records must not drift from the code they describe.

docs/sam_prelabeling.md and docs/training.md quote config values as the
concrete record of what the pipeline does. A document that says
MIN_INSTANCE_AREA_PX is 250 while the code says 700 is worse than no document:
it is read as evidence, and it is the kind of wrong answer nothing else in this
repo would catch.

The values were checked by hand once. This keeps them checked.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "seeweed3d") not in sys.path:
    sys.path.insert(0, str(ROOT / "seeweed3d"))

from annotation.prelabel_weeds_sam3 import CONFIG as PRELABEL   # noqa: E402
from training.train_model_rfdetr import CONFIG as TRAIN         # noqa: E402

PRELABEL_DOC = ROOT / "docs" / "sam_prelabeling.md"
TRAIN_DOC = ROOT / "docs" / "training.md"

SCALARS = (int, float, bool, str)


def lines_naming(doc, key):
    """Lines that mention `key` as a whole word.

    Whole-word, because `LR` is a substring of `LR_SCHEDULER` and `TRACK` of
    `MLFLOW_TRACKING_URI`. Matching loosely made a correct document look wrong
    five times over, which is the fastest way to get a check switched off."""
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])")
    return [ln for ln in doc.splitlines() if pat.search(ln)]


def mentions_value(line, want):
    """Does this line carry `want` anywhere in it?

    The line, not a position within it: a table row may pair two settings
    ("BATCH / GRAD_ACCUM | 2 x 8") and prose may put the value before the name.
    What matters is that the right number is on the row and the wrong one is
    not."""
    if isinstance(want, bool) or isinstance(want, str):
        return str(want) in line
    for tok in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", line.replace(",", "")):
        try:
            if float(tok) == float(want):
                return True
        except ValueError:
            continue
    # 1e-4 written as 1e-4 is caught above; 0.0001 written as 1e-4 is not.
    return f"{want:g}" in line or f"{want:e}" in line


def check(doc_path, config, key):
    doc = doc_path.read_text(encoding="utf-8")
    named = lines_naming(doc, key)
    if not named:
        pytest.skip(f"{key} is not quoted in {doc_path.name}")
    want = config[key]
    assert any(mentions_value(ln, want) for ln in named), (
        f"{doc_path.name} names {key} but never with its value {want!r}.\n"
        + "\n".join(f"    {ln.strip()}" for ln in named[:4]))


# --------------------------------------------------------------------------- #
def test_both_records_exist_and_are_linked_from_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for p in (PRELABEL_DOC, TRAIN_DOC):
        assert p.is_file(), p
        assert p.name in readme, f"{p.name} is not linked from the README"


@pytest.mark.parametrize("key", sorted(
    k for k, v in PRELABEL.items() if isinstance(v, SCALARS)))
def test_prelabeling_doc_matches_the_code(key):
    check(PRELABEL_DOC, PRELABEL, key)


@pytest.mark.parametrize("key", sorted(
    k for k, v in TRAIN.items() if isinstance(v, SCALARS)))
def test_training_doc_matches_the_code(key):
    check(TRAIN_DOC, TRAIN, key)


# --------------------------------------------------------------------------- #
# The settings each record exists to explain must actually appear in it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", [
    "MIN_INSTANCE_AREA_PX", "SPLIT_TOUCHING_INSTANCES", "RECOVER_MISSED_PLANTS",
    "BOUNDARY_REFINE_BAND_PX", "BOUNDARY_REFINE_MAX_AREA_PX",
    "EXEMPLAR_MIN_VEG_SCORE", "USE_DEPTH_HEIGHT", "CLUSTER_MIN_PEAKS",
    "GRASS_MIN_ASPECT"])
def test_the_decisive_prelabel_settings_are_stated(key):
    assert key in PRELABEL_DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("key", [
    "RESOLUTION", "VARIANT", "GRAD_ACCUM", "LR_SCHEDULER", "PATIENCE",
    "TVERSKY_ALPHA", "TVERSKY_BETA", "USE_EMA", "WORKERS"])
def test_the_decisive_training_settings_are_stated(key):
    assert key in TRAIN_DOC.read_text(encoding="utf-8")


def test_the_training_record_names_the_backend_actually_in_use():
    """Both the onion and weed models train on RF-DETR-Seg; Mask R-CNN remains
    the torchvision default in the segmenter. A record that conflates the two
    would send someone to the wrong runner."""
    doc = TRAIN_DOC.read_text(encoding="utf-8")
    assert "RF-DETR-Seg" in doc and "maskrcnn" in doc
    assert "Apache-2.0" in doc and "AGPL" in doc


def test_the_training_record_keeps_the_checkpoint_warning():
    """rfdetr keeps three checkpoints and _total is copied from whichever won.
    Naming _ema silently scores the loser, and it has happened once."""
    doc = TRAIN_DOC.read_text(encoding="utf-8")
    assert "checkpoint_best_total.pth" in doc and "_ema" in doc
