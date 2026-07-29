"""Checks for the weed instance prelabeler's non-GPU logic: instance filtering
and NMS, shape descriptors, morphology proposal, LEP/treatment points, and the
multi-class COCO export. SAM 3 is stubbed (needs a GPU + gated weights)."""
import csv
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
    assert wd.classify_morphology(blade_f, wd.CONFIG)[0] == "grass_weed"
    assert wd.classify_morphology(rose_f, wd.CONFIG)[0] != "grass_weed"


def test_species_are_never_auto_assigned():
    """brassica vs primrose is an appearance question shape cannot answer, so a
    rosette must fall back to the default class with zero confidence rather than
    nudging the annotator toward a confident wrong species."""
    cls, conf = wd.classify_morphology(wd.shape_features(_rosette()), wd.CONFIG)
    assert cls == wd.CONFIG["DEFAULT_SPECIES_CLASS"] == "other_weed"
    assert conf == 0.0
    assert cls not in ("cutleaf_evening_primrose", "wild_radish")
    # Only shape-supported classes are ever proposed.
    assert wd.AUTO_CLASSES == {"grass_weed", "weed_cluster", "other_weed"}


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
    assert wd.classify_morphology(f_single, wd.CONFIG, peaks_single)[0] != "weed_cluster"

    merged = np.zeros((400, 400), bool)
    for cx, cy in ((110, 110), (250, 130), (170, 260), (290, 280)):
        merged |= _rosette(400, 400, cx, cy, 60)
    f_m = wd.shape_features(merged)
    peaks_m = wd.growth_peaks(merged, wd.CONFIG)
    cfg = dict(wd.CONFIG)
    cfg["CLUSTER_MIN_AREA_PX"] = min(cfg["CLUSTER_MIN_AREA_PX"], f_m["area_px"])
    cls, conf = wd.classify_morphology(f_m, cfg, peaks_m)
    assert cls == "weed_cluster" and conf > 0


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


def test_small_seedlings_are_not_filtered_out():
    """Regression guard. MIN_INSTANCE_AREA_PX was once raised to 700 on the
    assumption that small detections were noise; in the field imagery they are
    real cotyledon/2-leaf weeds, and 700 deleted every plant under ~29 px
    diameter. For a laser weeder a missed small weed is worse than an extra
    instance the annotator deletes, so the default must keep them."""
    assert wd.CONFIG["MIN_INSTANCE_AREA_PX"] <= 300

    veg = np.zeros((300, 300), bool)
    masks = []
    for i, d in enumerate((18, 22, 30, 60)):        # cotyledon -> rosette
        m = np.zeros((300, 300), np.uint8)
        cv2.circle(m, (40 + i * 70, 150), d // 2, 1, -1)
        mb = m.astype(bool)
        veg |= mb
        masks.append(mb)
    kept = wd.filter_instances(masks, veg, wd.CONFIG)
    assert len(kept) == 4, "small real seedlings must survive the size gate"


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
    emitted for it. Splitting is disabled here so the cluster path is tested in
    isolation - separable plants are split into individual instances instead,
    which is covered by test_touching_plants_are_split_not_clustered."""
    merged = np.zeros((400, 400), bool)
    for cx, cy in ((110, 110), (250, 130), (170, 260), (290, 280)):
        merged |= _rosette(400, 400, cx, cy, 60)
    bgr = np.full((400, 400, 3), (70, 45, 60), np.uint8)
    bgr[merged] = (35, 110, 40)
    cfg = dict(wd.CONFIG)
    cfg["CLUSTER_MIN_AREA_PX"] = 1000
    cfg["SPLIT_TOUCHING_INSTANCES"] = False
    instances, _ = wd.analyze_frame(bgr, [merged], cfg)
    clusters = [i for i in instances if i["cls"] == "weed_cluster"]
    assert clusters, "setup should produce a cluster"
    assert clusters[0].get("lep") is None and clusters[0]["lep_valid"] is False


def test_instances_csv_survives_mixed_rows(tmp_path):
    """Regression guard for a real crash: rows do NOT all share a key set - a
    weed_cluster instance carries no lep_* columns, and canopy_height only
    appears when the frame has valid depth. Taking the CSV header from row 0
    raised 'dict contains fields not in fieldnames' as soon as the first
    instance was a cluster."""
    sess = tmp_path / "sess"
    (sess / "rgb").mkdir(parents=True)
    (sess / "meta").mkdir(parents=True)

    # One frame containing a big multi-peak clump (-> cluster, no LEP) AND a
    # single rosette (-> has LEP), so both row shapes occur in one session.
    size = 500
    bgr = np.full((size, size, 3), (70, 45, 60), np.uint8)
    clump = np.zeros((size, size), bool)
    for cx, cy in ((110, 110), (200, 130), (150, 220), (240, 240)):
        clump |= _rosette(size, size, cx, cy, 55)
    single = _rosette(size, size, 400, 400, 45)
    bgr[clump | single] = (35, 110, 40)
    cv2.imwrite(str(sess / "rgb" / "f1.png"), bgr)
    with open(sess / "meta" / "pool.csv", "w", newline="") as f:
        f.write("filename\nf1.png\n")

    def stub(pred, image, cfg, exemplars=None):
        return [clump, single]

    cfg = dict(wd.CONFIG)
    cfg["CLUSTER_MIN_AREA_PX"] = 5000        # make the clump register as a cluster
    cfg["SPLIT_TOUCHING_INSTANCES"] = False  # isolate the CSV-schema behaviour
    out = tmp_path / "out"
    st = wd.prelabel_session("sess", sess, out, cfg, predictor="STUB", sam_fn=stub)
    assert st["instances"] == 2

    rows = list(csv.DictReader(open(out / "sess" / "instances.csv", encoding="utf-8")))
    assert len(rows) == 2
    classes = {r["class"] for r in rows}
    assert "weed_cluster" in classes           # the row with no LEP
    # Every row shares one schema; the cluster's LEP cells are simply blank.
    assert all(set(r.keys()) == set(rows[0].keys()) for r in rows)
    assert "lep_x" in rows[0]
    cluster_row = next(r for r in rows if r["class"] == "weed_cluster")
    assert cluster_row["lep_x"] == ""


def test_touching_plants_are_split_not_clustered():
    """Two rosettes growing into each other are ONE connected blob, so SAM
    returns them as a single instance. Marker-controlled watershed seeded on the
    detected growth points must cut them into two plants with a boundary
    between them - that separation is what a trained model has to learn."""
    size = 300
    a = _rosette(size, size, 105, 150, 55)
    b = _rosette(size, size, 195, 150, 55)
    merged = a | b
    assert merged.sum() < a.sum() + b.sum(), "setup should actually overlap"

    peaks = wd.growth_peaks(merged, wd.CONFIG)
    parts = wd.split_touching_instances(merged, peaks, wd.CONFIG)
    assert len(parts) >= 2, "touching plants must be separated"
    # Parts are disjoint and together cover essentially the whole blob.
    assert (parts[0] & parts[1]).sum() == 0
    covered = np.logical_or.reduce(parts).sum()
    assert covered >= 0.80 * merged.sum()
    # Each part belongs to ONE plant: compare against the EXCLUSIVE region of
    # each (the plants overlap, so shared pixels legitimately count for both).
    # Geodesic assignment - tissue goes to the growth point it connects to
    # through the plant - makes this essentially pure, so hold it to that.
    only_a, only_b = a & ~b, b & ~a
    claims = [((p & only_a).sum(), (p & only_b).sum()) for p in parts[:2]]
    for ca, cb in claims:
        assert max(ca, cb) > 20 * max(1, min(ca, cb)), "a part straddles both plants"
    # ...and the two parts claim opposite plants, not the same one twice.
    assert (claims[0][0] > claims[0][1]) != (claims[1][0] > claims[1][1])


def test_split_falls_back_rather_than_inventing_a_boundary():
    """A single plant has one growth point, so nothing to split - and a split
    that loses tissue must be rejected rather than fabricating a cut."""
    single = _rosette(cx=100, cy=100, r=45)
    peaks = wd.growth_peaks(single, wd.CONFIG)
    assert wd.split_touching_instances(single, peaks, wd.CONFIG) == [single] or \
        len(wd.split_touching_instances(single, peaks, wd.CONFIG)) == 1
    # Disabled by config -> always the original mask.
    cfg = dict(wd.CONFIG); cfg["SPLIT_TOUCHING_INSTANCES"] = False
    merged = _rosette(300, 300, 105, 150, 55) | _rosette(300, 300, 195, 150, 55)
    out = wd.split_touching_instances(merged, wd.growth_peaks(merged, cfg), cfg)
    assert len(out) == 1 and np.array_equal(out[0], merged)


def test_boundary_refinement_snaps_to_plant_evidence():
    """SAM's edge can bleed onto soil. Only the narrow boundary band is
    re-decided from the image's own plant/soil evidence; the interior and the
    overall shape are untouched, and refinement cannot absorb a neighbour."""
    from common.vegetation import vegetation_score
    size = 160
    bgr = np.full((size, size, 3), (70, 45, 60), np.uint8)      # soil
    true_plant = np.zeros((size, size), bool)
    cv2.circle(true_plant.view(np.uint8), (80, 80), 30, 1, -1)
    bgr[true_plant] = (35, 160, 45)                              # saturated green

    bloated = np.zeros((size, size), bool)                       # SAM overshoots
    cv2.circle(bloated.view(np.uint8), (80, 80), 36, 1, -1)

    score = vegetation_score(bgr, wd.CONFIG["EXG_THRESHOLD"],
                             wd.CONFIG["VEG_MIN_SATURATION"])
    refined = wd.refine_boundary(bloated, score, wd.CONFIG)
    # Closer to the true plant than the bloated input was.
    before = np.logical_xor(bloated, true_plant).sum()
    after = np.logical_xor(refined, true_plant).sum()
    assert after < before, f"refinement made it worse: {after} vs {before}"
    assert refined[80, 80], "interior must be preserved"


def test_polygons_keep_all_parts_and_scale_tolerance():
    """Exporting only the largest contour silently dropped real tissue that was
    separated by an occlusion, and a fixed simplification tolerance erases shape
    on seedlings while bloating vertices on rosettes."""
    m = np.zeros((200, 200), bool)
    cv2.circle(m.view(np.uint8), (60, 100), 30, 1, -1)      # main body
    cv2.circle(m.view(np.uint8), (150, 100), 14, 1, -1)     # detached leaf
    polys = wd.mask_polygons(m, wd.CONFIG)
    assert len(polys) == 2, "a detached part must not be dropped"
    for p in polys:
        assert len(p) >= 6 and len(p) % 2 == 0

    # Tolerance scales with size, and stays inside the configured bounds.
    small = wd.polygon_epsilon(300, wd.CONFIG)
    large = wd.polygon_epsilon(40000, wd.CONFIG)
    assert small < large
    assert wd.CONFIG["POLY_APPROX_EPS_MIN"] <= small
    assert large <= wd.CONFIG["POLY_APPROX_EPS_MAX"]


def test_coco_accepts_multipart_segmentation_and_true_area():
    """COCO's segmentation field is a list so a multi-part instance survives;
    area must be the real mask area, not the bbox area which badly overstates a
    thin or lobed plant."""
    coco = wd.WeedCoco()
    img = coco.add_image("f.png", 200, 200)
    coco.add_instance(img, "other_weed",
                      [[10, 10, 50, 10, 50, 50, 10, 50],
                       [120, 10, 160, 10, 160, 50, 120, 50]],
                      bbox=[10, 10, 150, 40], area_px=1234)
    ann = coco.anns[0]
    assert len(ann["segmentation"]) == 2
    assert ann["area"] == 1234.0 and ann["area"] != 150 * 40


def test_cvat_label_schema_covers_classes_and_lep():
    names = [l["name"] for l in wd.weed_cvat_labels()]
    for c in wd.WEED_CLASSES:
        assert c in names
    assert wd.LEP_LABEL in names
    lep = next(l for l in wd.weed_cvat_labels() if l["name"] == wd.LEP_LABEL)
    assert lep["type"] == "points"


# --------------------------------------------------------------------------- #
# Recall backstop
# --------------------------------------------------------------------------- #
def _scene_with_two_rosettes(size=320):
    """Soil frame with two well-separated green rosettes."""
    bgr = np.full((size, size, 3), (70, 45, 60), np.uint8)
    a = _rosette(size, size, 90, 90, 45)
    b = _rosette(size, size, 230, 230, 45)
    bgr[a | b] = (35, 110, 40)
    return bgr, a, b


def test_plant_missed_by_sam_is_recovered():
    """The failure this guards against: SAM returns one of two plants, so the
    other is silently absent from the export and the annotator never sees it -
    teaching a model that such plants do not exist. In a weed-only scene every
    vegetation blob IS a plant, so the unclaimed one must come back."""
    bgr, a, b = _scene_with_two_rosettes()

    instances, _ = wd.analyze_frame(bgr, [a], wd.CONFIG)      # SAM found only `a`
    assert len(instances) == 2

    sources = sorted(i["source"] for i in instances)
    assert sources == ["sam", "vegetation"]

    # The recovered instance is the plant SAM missed, not a fragment of the one
    # it found.
    rec = next(i for i in instances if i["source"] == "vegetation")
    ys, xs = np.nonzero(rec["mask"])
    assert abs(float(xs.mean()) - 230) < 20 and abs(float(ys.mean()) - 230) < 20
    assert float((rec["mask"] & b).sum()) / rec["mask"].sum() > 0.9


def test_recovery_is_a_no_op_when_sam_finds_everything():
    bgr, a, b = _scene_with_two_rosettes()
    instances, _ = wd.analyze_frame(bgr, [a, b], wd.CONFIG)
    assert len(instances) == 2
    assert all(i["source"] == "sam" for i in instances)


def test_recovery_can_be_disabled():
    bgr, a, _ = _scene_with_two_rosettes()
    cfg = dict(wd.CONFIG, RECOVER_MISSED_PLANTS=False)
    instances, _ = wd.analyze_frame(bgr, [a], cfg)
    assert len(instances) == 1 and instances[0]["source"] == "sam"


def test_recovery_does_not_duplicate_an_already_detected_plant():
    """The backstop must not turn a leaf tip poking past the edge of a good
    detection into a second plant. That residual is large enough to pass the
    area test, so the guard has to be the fraction of the WHOLE vegetation blob
    that is already claimed - here ~85%, far above RECOVER_MAX_CLAIMED_FRAC."""
    size = 260
    plant = np.zeros((size, size), np.uint8)
    cv2.circle(plant, (110, 120), 30, 1, -1)                 # crown
    cv2.line(plant, (135, 120), (215, 120), 1, 13)           # one long leaf
    plant = plant.astype(bool)

    clipped = plant.copy()
    clipped[:, 170:] = False                                  # SAM clipped the tip
    residual_px = int((plant & ~clipped).sum())
    assert residual_px > wd.CONFIG["MIN_INSTANCE_AREA_PX"]    # area alone would pass

    assert wd.recover_missed_plants(plant, [clipped], wd.CONFIG) == []


def test_recovery_returns_a_plant_no_instance_touches():
    veg = np.zeros((260, 260), bool)
    veg[30:90, 30:90] = True                                  # detected
    veg[160:220, 160:220] = True                              # missed entirely
    detected = np.zeros((260, 260), bool)
    detected[30:90, 30:90] = True

    out = wd.recover_missed_plants(veg, [detected], wd.CONFIG)
    assert len(out) == 1
    assert out[0][180, 180] and not out[0][50, 50]


def test_session_summary_reports_vegetation_coverage(tmp_path):
    """Recall has to be visible in the run output, because instance count alone
    cannot distinguish a thorough pass from one that missed half the plants."""
    sess = tmp_path / "sess"
    (sess / "rgb").mkdir(parents=True)
    (sess / "meta").mkdir(parents=True)
    bgr, a, b = _scene_with_two_rosettes()
    cv2.imwrite(str(sess / "rgb" / "f1.png"), bgr)
    with open(sess / "meta" / "pool.csv", "w", newline="") as f:
        f.write("filename\nf1.png\n")

    st = wd.prelabel_session("sess", sess, tmp_path / "out", dict(wd.CONFIG),
                             predictor="STUB",
                             sam_fn=lambda p, im, cfg, ex=None: [a])
    assert st["instances"] == 2 and st["recovered"] == 1
    assert st["veg_px"] > 0
    # With the backstop on, nearly all vegetation ends up inside an instance.
    assert st["veg_covered_px"] / st["veg_px"] > 0.9

    rows = list(csv.DictReader(
        open(tmp_path / "out" / "sess" / "instances.csv", encoding="utf-8")))
    assert sorted(r["source"] for r in rows) == ["sam", "vegetation"]


# --------------------------------------------------------------------------- #
# Boundary anti-aliasing (smooth_boundary)
# --------------------------------------------------------------------------- #
def _jagged_disc(h=200, w=200, cx=100, cy=100, r=50, seed=0):
    """A disc with single-pixel staircase noise stitched onto its boundary -
    the kind of noise a per-pixel threshold decision (refine_boundary) leaves
    behind, as opposed to a genuinely lobed leaf margin."""
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    rng = np.random.default_rng(seed)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for (x, y) in cnts[0][:, 0, :]:
        if rng.random() < 0.5:
            cv2.circle(m, (int(x), int(y)), 1, int(rng.random() < 0.5), -1)
    return m.astype(bool)


def test_smooth_boundary_reduces_perimeter_without_losing_area():
    """Blur-and-rethreshold should erase the staircase noise (shorter, cleaner
    perimeter) while the disc's actual footprint survives almost intact."""
    noisy = _jagged_disc()
    smooth = wd.smooth_boundary(noisy, wd.CONFIG)

    p_noisy = cv2.arcLength(
        max(cv2.findContours(noisy.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea), True)
    p_smooth = cv2.arcLength(
        max(cv2.findContours(smooth.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea), True)
    assert p_smooth < p_noisy
    assert abs(int(smooth.sum()) - int(noisy.sum())) < 0.05 * noisy.sum()


def test_smooth_boundary_preserves_grass_elongation():
    """Sigma is sub-pixel so it must not blur away the elongation signal that
    classify_morphology relies on to separate grass from a rosette."""
    blade = wd.smooth_boundary(_blade(), wd.CONFIG)
    f = wd.shape_features(blade)
    assert f is not None and f["aspect_ratio"] >= wd.CONFIG["GRASS_MIN_ASPECT"]


def test_smooth_boundary_falls_back_rather_than_sever_a_thin_neck():
    """A blur that would cut a genuinely thin connection into two pieces must
    not silently discard one of them - the unsmoothed mask is safer, exactly
    like split_touching_instances()'s own fallback."""
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (25, 50), 15, 1, -1)
    cv2.circle(m, (75, 50), 15, 1, -1)
    cv2.line(m, (40, 50), (60, 50), 1, 1)          # 1px isthmus
    m = m.astype(bool)
    cfg = dict(wd.CONFIG, BOUNDARY_SMOOTH_SIGMA_PX=3.0)   # aggressive on purpose
    out = wd.smooth_boundary(m, cfg)
    assert np.array_equal(out, m)                  # fallback returned as-is


def test_smooth_boundary_disabled_by_zero_sigma():
    m = _rosette()
    cfg = dict(wd.CONFIG, BOUNDARY_SMOOTH_SIGMA_PX=0)
    assert np.array_equal(wd.smooth_boundary(m, cfg), m)


def test_smooth_boundary_reduces_spurious_skeleton_junctions():
    """The reason this step exists: PetioleConvergenceEvidence
    (perception/lep.py) finds the LEP by locating skeleton junctions. Boundary
    staircase noise fabricates short spurious skeleton branches - noise
    injected directly into the growth-point estimate. Smoothing the mask
    before skeletonising must reduce that, not just tidy the outline."""
    lp = load_script("perception/lep.py")
    noisy = _jagged_disc(r=40)
    smooth = wd.smooth_boundary(noisy, wd.CONFIG)

    def junction_count(mask):
        skel = lp.zhang_suen_thin(mask).astype(np.uint8)
        if not skel.any():
            return 0
        deg = cv2.filter2D(skel, -1, np.ones((3, 3), np.uint8),
                           borderType=cv2.BORDER_CONSTANT) - skel
        return int(((deg >= 3) & (skel > 0)).sum())

    assert junction_count(smooth) < junction_count(noisy)
