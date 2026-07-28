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


def test_grass_separated_by_elongation():
    rose_f = wd.shape_features(_rosette())
    blade_f = wd.shape_features(_blade())
    assert rose_f and blade_f
    # A blade is far more elongated than a rosette - the key separating signal.
    assert blade_f["aspect_ratio"] > rose_f["aspect_ratio"]
    assert wd.classify_morphology(blade_f, wd.CONFIG)[0] == "grass"
    assert wd.classify_morphology(rose_f, wd.CONFIG)[0] != "grass"


def test_species_are_never_auto_assigned():
    """brassica vs primrose is an appearance question shape cannot answer, so a
    rosette must fall back to the default class with zero confidence rather than
    nudging the annotator toward a confident wrong species."""
    cls, conf = wd.classify_morphology(wd.shape_features(_rosette()), wd.CONFIG)
    assert cls == wd.CONFIG["DEFAULT_SPECIES_CLASS"] == "other weed"
    assert conf == 0.0
    assert cls not in ("brassica", "primrose")
    # Only shape-supported classes are ever proposed.
    assert wd.AUTO_CLASSES == {"grass", "weed cluster", "other weed"}


def test_growth_peaks_one_per_plant():
    """One rosette -> one growth point; several intermingled -> several."""
    single = _rosette(cx=100, cy=100, r=40)
    assert len(wd.growth_peaks(single, wd.CONFIG)) == 1

    merged = np.zeros((300, 300), bool)
    for cx, cy in ((90, 90), (200, 100), (140, 210)):
        merged |= _rosette(300, 300, cx, cy, 45)
    assert len(wd.growth_peaks(merged, wd.CONFIG)) >= 3


def test_cluster_only_declared_for_large_multi_peak_blobs():
    """The cluster threshold is deliberately high: a single plant, however big,
    must never be called a cluster."""
    single = _rosette(cx=100, cy=100, r=40)
    f_single = wd.shape_features(single)
    peaks_single = wd.growth_peaks(single, wd.CONFIG)
    assert wd.classify_morphology(f_single, wd.CONFIG, peaks_single)[0] != "weed cluster"

    merged = np.zeros((400, 400), bool)
    for cx, cy in ((110, 110), (250, 130), (170, 260), (290, 280)):
        merged |= _rosette(400, 400, cx, cy, 60)
    f_m = wd.shape_features(merged)
    peaks_m = wd.growth_peaks(merged, wd.CONFIG)
    cfg = dict(wd.CONFIG)
    cfg["CLUSTER_MIN_AREA_PX"] = min(cfg["CLUSTER_MIN_AREA_PX"], f_m["area_px"])
    cls, conf = wd.classify_morphology(f_m, cfg, peaks_m)
    assert cls == "weed cluster" and conf > 0


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
    # A non-cluster instance carries a usable single LEP.
    assert inst["lep_valid"] is True and inst["peaks"]

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


def test_fused_lep_is_attached_to_instances():
    """The multi-evidence LEP must reach each instance, carry a visibility
    verdict and uncertainty, and keep the geometric baselines alongside it for
    the plan's LEP-method comparison."""
    bgr = np.full((240, 240, 3), (70, 45, 60), np.uint8)
    m = _rosette(240, 240, 120, 120, 55)
    bgr[m] = (35, 110, 40)
    core = np.zeros((240, 240), np.uint8)
    cv2.circle(core, (120, 120), 14, 1, -1)
    bgr[core.astype(bool) & m] = (70, 190, 120)     # pale young tissue

    instances, _ = wd.analyze_frame(bgr, [m], wd.CONFIG)
    inst = instances[0]
    r = inst["lep"]
    assert r is not None
    assert abs(r.uv[0] - 120) < 20 and abs(r.uv[1] - 120) < 20
    assert r.visibility in ("visible", "partially_occluded_inferable", "not_visible")
    assert 0.0 <= r.confidence <= 1.0 and r.sigma_px >= 0
    # Baselines preserved for the comparison study.
    for key in ("lep_dt", "centroid", "bbox_ctr"):
        assert key in inst["points"]
    row = r.as_row("lep")
    for k in ("lep_x", "lep_y", "lep_confidence", "lep_visibility", "lep_method"):
        assert k in row


def test_cluster_gets_no_fused_lep():
    """A cluster has several growth points, so a single LEP must not be
    emitted for it."""
    merged = np.zeros((400, 400), bool)
    for cx, cy in ((110, 110), (250, 130), (170, 260), (290, 280)):
        merged |= _rosette(400, 400, cx, cy, 60)
    bgr = np.full((400, 400, 3), (70, 45, 60), np.uint8)
    bgr[merged] = (35, 110, 40)
    cfg = dict(wd.CONFIG)
    cfg["CLUSTER_MIN_AREA_PX"] = 1000
    instances, _ = wd.analyze_frame(bgr, [merged], cfg)
    clusters = [i for i in instances if i["cls"] == "weed cluster"]
    assert clusters, "setup should produce a cluster"
    assert clusters[0].get("lep") is None and clusters[0]["lep_valid"] is False


def test_cvat_label_schema_covers_classes_and_lep():
    names = [l["name"] for l in wd.weed_cvat_labels()]
    for c in wd.WEED_CLASSES:
        assert c in names
    assert "weed LEP" in names
    lep = next(l for l in wd.weed_cvat_labels() if l["name"] == "weed LEP")
    assert lep["type"] == "points"
