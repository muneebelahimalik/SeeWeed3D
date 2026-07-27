"""Checks for the weed instance prelabeler's non-GPU logic: instance filtering
and NMS, shape descriptors, morphology proposal, LEP/treatment points, and the
multi-class COCO export. SAM 3 is stubbed (needs a GPU + gated weights)."""
import json

import cv2
import numpy as np

from conftest import load_script

wd = load_script("annotation/prelabel_weeds_sam3.py")


def _rosette(h=200, w=200, cx=100, cy=100, r=40, arms=8):
    """Synthetic rosette: a disc with radiating leaves, like the Brassica
    rosettes in the field images. Its growth point is the centre."""
    m = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    m |= (yy - cy) ** 2 + (xx - cx) ** 2 < (r * 0.45) ** 2      # compact centre
    for k in range(arms):                                        # leaves
        a = 2 * np.pi * k / arms
        for t in np.linspace(0, r, r * 2):
            x, y = int(cx + t * np.cos(a)), int(cy + t * np.sin(a))
            cv2.circle(m.view(np.uint8), (x, y), 6, 1, -1)
    return m.astype(bool)


def _blade(h=200, w=200):
    """Synthetic grass blade: long, thin, low solidity."""
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (20, 100), (180, 90), 1, 7)
    return m.astype(bool)


def test_shape_features_and_morphology_separate_rosette_from_grass():
    rose_f = wd.shape_features(_rosette())
    blade_f = wd.shape_features(_blade())
    assert rose_f and blade_f
    # A blade is far more elongated than a rosette - the key separating signal.
    assert blade_f["aspect_ratio"] > rose_f["aspect_ratio"]
    assert wd.classify_morphology(blade_f, wd.CONFIG)[0] == "weed grass"
    # Rosette must never be called grass; broadleaf or unknown are both fine
    # (the heuristic is deliberately conservative).
    assert wd.classify_morphology(rose_f, wd.CONFIG)[0] != "weed grass"


def test_classify_is_conservative_in_the_ambiguous_band():
    """Aspect between BROADLEAF_MAX_ASPECT and GRASS_MIN_ASPECT is the
    deliberately undecided band: neither class is claimed, so the annotator is
    never nudged toward a confident wrong label."""
    m = np.zeros((160, 160), np.uint8)
    cv2.ellipse(m, (80, 80), (65, 25), 0, 0, 360, 1, -1)      # aspect ~2.6
    f = wd.shape_features(m.astype(bool))
    assert wd.CONFIG["BROADLEAF_MAX_ASPECT"] < f["aspect_ratio"] < wd.CONFIG["GRASS_MIN_ASPECT"]
    cls, conf = wd.classify_morphology(f, wd.CONFIG)
    assert cls == "weed unknown" and conf == 0.0

    # A thin cross: compact overall (aspect ~1) but far too sparse to be a
    # rosette, so the solidity gate keeps it out of "broadleaf".
    cross = np.zeros((120, 120), np.uint8)
    cross[58:63, 15:105] = 1
    cross[15:105, 58:63] = 1
    fc = wd.shape_features(cross.astype(bool))
    assert fc["aspect_ratio"] < wd.CONFIG["BROADLEAF_MAX_ASPECT"]
    assert fc["solidity"] < wd.CONFIG["BROADLEAF_MIN_SOLIDITY"]
    assert wd.classify_morphology(fc, wd.CONFIG)[0] == "weed unknown"


def test_treatment_points_lep_at_rosette_centre():
    """The distance-transform peak must land at the rosette centre - that is the
    growth point (LEP/AMT) this whole project targets."""
    m = _rosette(cx=120, cy=80, r=40)
    p = wd.treatment_points(m)
    x, y = p["lep_dt"]
    assert abs(x - 120) < 12 and abs(y - 80) < 12
    # All three candidate points exist for the plan's LEP-method comparison.
    for key in ("lep_dt", "centroid", "bbox_ctr"):
        assert len(p[key]) == 2
    assert p["dt_radius_px"] > 0


def test_filter_instances_rejects_and_dedupes():
    veg = np.zeros((200, 200), bool)
    veg[50:150, 50:150] = True
    good = np.zeros((200, 200), bool); good[60:140, 60:140] = True   # on veg
    dup = good.copy()                                                # duplicate
    off = np.zeros((200, 200), bool); off[0:30, 0:30] = True         # off veg
    tiny = np.zeros((200, 200), bool); tiny[60:63, 60:63] = True     # too small
    huge = np.ones((200, 200), bool)                                 # whole frame
    kept = wd.filter_instances([good, dup, off, tiny, huge], veg, wd.CONFIG)
    assert len(kept) == 1                       # dup NMS'd, others rejected
    assert kept[0].sum() == good.sum()


def test_analyze_frame_and_coco_export(tmp_path):
    bgr = np.full((200, 200, 3), (70, 40, 60), np.uint8)      # soil
    m = _rosette()
    bgr[m] = (40, 200, 40)                                     # green rosette
    instances, veg = wd.analyze_frame(bgr, [m], wd.CONFIG)
    assert len(instances) == 1
    inst = instances[0]
    assert inst["cls"] in wd.WEED_CLASSES
    assert inst["growth_stage"] in ("cotyledon", "2-leaf", "3-5-leaf")

    coco = wd.WeedCoco()
    img_id = coco.add_image("f_000001.png", 200, 200)
    poly = wd.mask_polygon(inst["mask"], wd.CONFIG["POLY_APPROX_EPS"])
    coco.add_instance(img_id, inst["cls"], poly, inst["features"]["bbox"])
    out = tmp_path / "instances_default.json"
    coco.dump(out)

    d = json.loads(out.read_text())
    assert [c["name"] for c in d["categories"]] == wd.WEED_CLASSES
    a = d["annotations"][0]
    assert 1 <= a["category_id"] <= len(wd.WEED_CLASSES)
    seg = a["segmentation"][0]
    assert len(seg) >= 6 and len(seg) % 2 == 0
    assert 0 <= min(seg[0::2]) and max(seg[0::2]) <= 200


def test_cvat_label_schema_covers_classes_and_lep():
    names = [l["name"] for l in wd.weed_cvat_labels()]
    for c in wd.WEED_CLASSES:
        assert c in names
    assert "weed LEP" in names
    lep = next(l for l in wd.weed_cvat_labels() if l["name"] == "weed LEP")
    assert lep["type"] == "points"
