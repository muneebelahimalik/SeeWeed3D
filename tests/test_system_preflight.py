"""Is the DEPLOYED system ready?

Every failure this checks for produces output that looks completely ordinary.
A weed-only model abstains on every target and reports zero candidates, which
in the JSON is indistinguishable from a clean frame with no weeds in it. A
checkpoint with no recorded class list maps its labels through the full
ontology and mislabels every plant with total confidence. A session with no
calibration returns every target with xyz_mm None, which reads as "nothing
found" rather than "nothing measured".

None of them raise. That is the entire reason this module exists, and it is
why every finding here carries its consequence rather than just its name.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seeweed3d"))
from common.ontology import CLASSES, CROP_CLASS   # noqa: E402
from perception import preflight as pf            # noqa: E402

WEEDS = [c for c in CLASSES if c != CROP_CLASS][:3]
ALL = list(CLASSES)


def codes(findings):
    return {f.code for f in findings}


def one(findings, code):
    return next(f for f in findings if f.code == code)


def base(**kw):
    """A system with nothing wrong, so each test changes exactly one thing."""
    args = dict(checkpoint="ck.pth", model_classes=ALL, lep_checkpoint="lep.pt",
                allow_missing_crop_mask=False, has_depth=True,
                has_calibration=True, holdout_sessions=["vid9"],
                label_provenance="hand_corrected", dedup_iou=None)
    args.update(kw)
    return pf.system_findings(**args)


# --------------------------------------------------------------------------- #
# The one that makes a weed-only model undeployable
# --------------------------------------------------------------------------- #
def test_a_weed_only_model_is_blocking():
    """THE CASE. SafetyConfig.allow_missing_crop_mask defaults to False, so a
    model with no onion class rejects EVERY candidate - and the run completes,
    reporting zero candidates, exactly as a clean field would."""
    f = base(model_classes=WEEDS)
    assert "no_crop_class" in codes(f)
    assert one(f, "no_crop_class").level == pf.ERROR


def test_the_blocking_message_says_every_candidate_is_rejected():
    """Naming the missing class is not enough - someone reading zero candidates
    has to connect the two."""
    msg = one(base(model_classes=WEEDS), "no_crop_class").message
    assert "EVERY candidate" in msg


def test_it_explains_that_the_default_is_deliberate():
    """A safety default that reads as a bug gets turned off."""
    fix = one(base(model_classes=WEEDS), "no_crop_class").fix
    assert "must never have its silence read as" in fix
    assert "allow_missing_crop_mask" in fix


def test_turning_the_crop_check_off_is_a_warning_not_silence():
    """It is a claim about the field, and only whoever is standing in it can
    make that claim - so it is recorded rather than accepted quietly."""
    f = base(model_classes=WEEDS, allow_missing_crop_mask=True)
    assert "no_crop_class" not in codes(f)
    assert one(f, "crop_blind_allowed").level == pf.WARN


def test_a_model_with_the_crop_class_clears_it():
    assert "no_crop_class" not in codes(base(model_classes=ALL))


# --------------------------------------------------------------------------- #
# The class list, which decides what every prediction is called
# --------------------------------------------------------------------------- #
def test_a_checkpoint_with_no_class_list_is_blocking():
    """The segmenter falls back to the FULL ontology, so a 3-class model has
    every prediction mapped to the wrong name - confidently."""
    f = base(model_classes=None)
    assert one(f, "no_class_list").level == pf.ERROR
    assert "FULL ontology" in one(f, "no_class_list").fix


def test_a_class_the_ontology_does_not_define_is_blocking():
    """Every downstream stage indexes by name, so a name nothing else knows is
    a prediction that cannot be acted on."""
    f = base(model_classes=ALL + ["mystery_plant"])
    assert one(f, "class_not_in_ontology").level == pf.ERROR
    assert "mystery_plant" in one(f, "class_not_in_ontology").message


def test_classes_the_model_cannot_predict_are_worth_knowing():
    """Legitimate while a class has too few examples to train on, and it stops
    being legitimate the round it has enough."""
    f = base(model_classes=WEEDS + [CROP_CLASS])
    assert one(f, "ontology_classes_absent").level == pf.WARN
    assert "scored as background" in one(f, "ontology_classes_absent").fix


def test_the_full_ontology_flags_nothing_about_classes():
    assert not {"no_class_list", "class_not_in_ontology",
                "ontology_classes_absent"} & codes(base())


# --------------------------------------------------------------------------- #
# 3D - the half that makes a pixel into somewhere to aim
# --------------------------------------------------------------------------- #
def test_missing_depth_is_blocking():
    f = base(has_depth=False)
    assert one(f, "no_depth").level == pf.ERROR
    assert "nothing measured" in one(f, "no_depth").fix


def test_missing_calibration_is_blocking():
    """Depth without intrinsics is present and unusable."""
    f = base(has_calibration=False)
    assert one(f, "no_calibration").level == pf.ERROR


def test_a_folder_of_loose_images_is_not_reported_as_missing_depth():
    """None means 'not checked', which is the honest state for a plain image
    folder - flagging it would be a blocking error on every 2D look."""
    f = base(has_depth=None, has_calibration=None)
    assert not {"no_depth", "no_calibration"} & codes(f)


# --------------------------------------------------------------------------- #
# What the numbers will and will not mean
# --------------------------------------------------------------------------- #
def test_no_holdout_is_worth_knowing():
    f = base(holdout_sessions=[])
    assert one(f, "no_holdout").level == pf.WARN
    assert "upper bound" in one(f, "no_holdout").fix


def test_machine_labels_are_flagged_so_scores_are_read_correctly():
    for prov in ("prelabel_unreviewed", "pseudo_label", "mixed"):
        f = base(label_provenance=prov)
        assert "machine_labels" in codes(f), prov
        assert "agreement" in one(f, "machine_labels").fix


def test_hand_corrected_labels_are_not_flagged():
    assert "machine_labels" not in codes(base(label_provenance="hand_corrected"))


def test_a_missing_lep_model_names_the_fallback_for_what_it_is():
    """The hand-engineered estimator is the baseline a learned model must beat,
    not the deployed stage."""
    f = base(lep_checkpoint="")
    assert one(f, "no_lep_model").level == pf.WARN
    assert "baseline" in one(f, "no_lep_model").fix


def test_dedup_disabled_is_flagged_with_its_consequence():
    f = base(dedup_iou=0)
    assert "aimed twice at one plant" in one(f, "dedup_off").fix


def test_the_default_dedup_is_not_flagged():
    assert "dedup_off" not in codes(base(dedup_iou=None))


# --------------------------------------------------------------------------- #
# No checkpoint at all
# --------------------------------------------------------------------------- #
def test_no_checkpoint_names_the_trainer_and_stops_there():
    """Listing every other problem when there is no model yet buries the one
    that has to be fixed first."""
    f = pf.system_findings(checkpoint=None)
    assert codes(f) == {"no_checkpoint"}
    assert "weeds_train" in f[0].fix


# --------------------------------------------------------------------------- #
# Reading the class list off disk
# --------------------------------------------------------------------------- #
def test_the_class_list_is_read_from_the_train_config(tmp_path):
    (tmp_path / "rfdetr_train_config.json").write_text(
        json.dumps({"classes": WEEDS}))
    assert pf.checkpoint_classes(tmp_path / "ck.pth") == WEEDS


def test_it_falls_back_to_the_coco_categories(tmp_path):
    d = tmp_path / "coco" / "annotations"
    d.mkdir(parents=True)
    (d / "instances_train.json").write_text(json.dumps(
        {"categories": [{"id": 2, "name": "b"}, {"id": 1, "name": "a"}]}))
    assert pf.checkpoint_classes(tmp_path / "ck.pth") == ["a", "b"], \
        "ascending category id is the order rfdetr assigns labels in"


def test_a_checkpoint_with_neither_records_nothing(tmp_path):
    assert pf.checkpoint_classes(tmp_path / "ck.pth") is None


def test_a_corrupt_train_config_does_not_take_the_check_down(tmp_path):
    (tmp_path / "rfdetr_train_config.json").write_text("{not json")
    assert pf.checkpoint_classes(tmp_path / "ck.pth") is None


def test_depth_readiness_is_read_from_the_session_layout(tmp_path):
    s = tmp_path / "sess"
    (s / "depth").mkdir(parents=True)
    (s / "meta").mkdir()
    assert pf.session_depth_ready(s) == (False, False), "an empty depth/ is not depth"
    (s / "depth" / "f0.png").write_bytes(b"x")
    (s / "meta" / "calibration.json").write_text("{}")
    assert pf.session_depth_ready(s) == (True, True)


# --------------------------------------------------------------------------- #
# The readout
# --------------------------------------------------------------------------- #
def test_a_clean_system_says_ready():
    assert "READY" in pf.format_report(base())


def test_the_report_separates_blocking_from_worth_knowing():
    text = pf.format_report(base(model_classes=WEEDS, lep_checkpoint="",
                                 holdout_sessions=[]))
    assert "BLOCKING" in text and "WORTH KNOWING" in text
    assert text.index("BLOCKING") < text.index("WORTH KNOWING"), \
        "the thing that stops the system comes first"


def test_the_report_says_that_nothing_will_raise():
    """Someone who sees a blocking finding and runs anyway must not expect an
    exception to save them."""
    text = pf.format_report(base(model_classes=WEEDS))
    assert "does not raise" in text


def test_the_report_names_the_model_and_its_classes():
    text = pf.format_report(base(), checkpoint="E:/r/ck.pth",
                            model_classes=WEEDS)
    assert "E:/r/ck.pth" in text
    assert all(c in text for c in WEEDS)


def test_every_finding_carries_a_fix():
    """A finding without one is a complaint."""
    f = base(model_classes=None, has_depth=False, has_calibration=False,
             lep_checkpoint="", holdout_sessions=[], label_provenance="mixed",
             dedup_iou=0)
    assert f and all(x.fix.strip() for x in f)
