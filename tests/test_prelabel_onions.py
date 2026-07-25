"""Checks for the SAM-3 onion prelabeler's non-GPU logic (vegetation prior,
fusion, polygon/COCO export) using a SAM stub. SAM 3 itself needs a GPU and
gated weights and is never loaded here."""
import json

import cv2
import numpy as np

from conftest import load_script

pre = load_script("annotation/prelabel_onions_sam3.py")


def test_vegetation_mask_green_in_soil_out():
    bgr = np.full((100, 100, 3), (70, 40, 60), np.uint8)   # brown soil
    bgr[30:70, 30:70] = (40, 200, 40)                      # green patch
    veg = pre.vegetation_mask(bgr, pre.CONFIG)
    assert veg[50, 50] and not veg[5, 5]


def test_vegetation_mask_rejects_desaturated_green_cast_soil():
    # Soil with a green colour-cast: green-dominant and high ExG, but LOW
    # saturation. This is exactly what masked whole frames before the fix.
    soil = np.full((100, 100, 3), (120, 140, 120), np.uint8)  # BGR, desaturated
    veg = pre.vegetation_mask(soil, pre.CONFIG)
    assert veg.mean() < 0.02          # essentially nothing masked
    # A saturated onion-green leaf on the same frame must still be kept.
    soil[40:60, 40:60] = (40, 200, 40)
    veg2 = pre.vegetation_mask(soil, pre.CONFIG)
    assert veg2[50, 50] and veg2.mean() < 0.10


def test_fuse_rejects_whole_frame_sam_mask():
    veg = np.zeros((100, 100), bool); veg[30:70, 30:70] = True
    whole = np.ones((100, 100), bool)          # SAM false positive over everything
    final, st = pre.fuse([whole], veg, pre.CONFIG)
    assert st["sam_kept"] == 0 and st["fallback_veg_only"]   # oversized mask dropped


def test_fuse_keeps_on_veg_drops_off_veg():
    veg = np.zeros((100, 100), bool); veg[30:70, 30:70] = True
    on = np.zeros((100, 100), bool); on[35:65, 35:65] = True
    off = np.zeros((100, 100), bool); off[0:10, 0:10] = True
    final, st = pre.fuse([on, off], veg, pre.CONFIG)
    assert st["sam_kept"] == 1 and not st["fallback_veg_only"] and final[50, 50]


def test_fuse_falls_back_to_veg_when_no_sam():
    veg = np.zeros((100, 100), bool); veg[30:70, 30:70] = True
    off = np.zeros((100, 100), bool); off[0:10, 0:10] = True
    final, st = pre.fuse([off], veg, pre.CONFIG)
    assert st["fallback_veg_only"] and final[50, 50]


def test_polygons_valid():
    m = np.zeros((100, 100), bool); m[20:80, 20:80] = True
    polys = pre.mask_to_polygons(m, pre.CONFIG)
    assert polys and all(len(p) >= 6 and len(p) % 2 == 0 for p in polys)


def _stub_sam(predictor, image_path, cfg, exemplars=None):
    bgr = cv2.imread(str(image_path))
    v = pre.vegetation_mask(bgr, cfg)
    return [cv2.dilate(v.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)]


def test_prelabel_session_produces_valid_coco(extracted_root, tmp_path):
    sid = "vid_20250304_142804"
    sess = extracted_root / "sessions" / sid
    cfg = dict(pre.CONFIG)
    cfg["LIMIT_PER_SESSION"] = 5
    st = pre.prelabel_session(sid, sess, tmp_path, cfg, predictor="STUB", sam_fn=_stub_sam)
    assert st["frames"] == 5 and st["polys"] > 0

    coco = json.loads((tmp_path / sid / "instances_default.json").read_text())
    assert coco["categories"][0]["name"] == "onion plant"
    assert len(coco["images"]) == 5 and coco["annotations"]
    by_id = {im["id"]: im for im in coco["images"]}
    for a in coco["annotations"]:
        im = by_id[a["image_id"]]
        seg = a["segmentation"][0]
        xs, ys = seg[0::2], seg[1::2]
        assert 0 <= min(xs) and max(xs) <= im["width"]
        assert 0 <= min(ys) and max(ys) <= im["height"]
    for im in coco["images"]:
        assert (sess / "rgb" / im["file_name"]).exists()
    assert any((tmp_path / sid / "masks").iterdir())
    assert any((tmp_path / sid / "preview").iterdir())
