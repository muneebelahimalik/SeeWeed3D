"""Round-trip checks for cvat_roundtrip.py: COCO polygon rasterization, mask +
manifest output, and auto-vs-verified agreement metrics. No CVAT/GPU needed."""
import json

import numpy as np

from conftest import load_script

rt = load_script("annotation/cvat_roundtrip.py")


def _coco(file_name, polys, w=64, h=48):
    anns = [{"id": i + 1, "image_id": 1, "category_id": 1,
             "segmentation": [p], "iscrowd": 0} for i, p in enumerate(polys)]
    return {"images": [{"id": 1, "file_name": file_name, "width": w, "height": h}],
            "annotations": anns,
            "categories": [{"id": 1, "name": "onion plant"}]}


def test_coco_masks_rasterizes_polygon():
    # a 20x20 square polygon at (10,10)-(30,30)
    sq = [10, 10, 30, 10, 30, 30, 10, 30]
    masks = rt.coco_masks(_coco("f_000001.png", [sq]), "onion plant")
    m = masks["f_000001.png"]
    assert m.shape == (48, 64) and m.dtype == bool
    assert m[20, 20] and not m[2, 2]
    assert 350 < int(m.sum()) < 450        # ~20x20 filled


def test_agreement_metrics():
    a = np.zeros((10, 10), bool); a[2:8, 2:8] = True     # 36 px
    v = np.zeros((10, 10), bool); v[2:8, 2:6] = True     # 24 px, subset of a
    r = rt.agreement(a, v)
    assert abs(r["iou"] - 24 / 36) < 1e-6
    assert abs(r["recall"] - 1.0) < 1e-6                 # v fully covered
    assert abs(r["precision"] - 24 / 36) < 1e-6
    # empty vs empty is perfect
    z = np.zeros((5, 5), bool)
    assert rt.agreement(z, z)["iou"] == 1.0


def test_ingest_end_to_end(tmp_path):
    sid = "s1"
    root = tmp_path / "dataset"
    (root / "sessions" / sid / "rgb").mkdir(parents=True)
    (root / "auto_labels_onion" / sid).mkdir(parents=True)
    vroot = tmp_path / "verified" / sid
    vroot.mkdir(parents=True)

    sq = [10, 10, 30, 10, 30, 30, 10, 30]
    big = [8, 8, 32, 8, 32, 32, 8, 32]      # slightly larger = the auto version
    (vroot / "instances_default.json").write_text(json.dumps(_coco("f_000001.png", [sq])))
    (root / "auto_labels_onion" / sid / "instances_default.json").write_text(
        json.dumps(_coco("f_000001.png", [big])))

    cfg = dict(rt.CONFIG)
    cfg.update({"DATASET_ROOT": str(root), "VERIFIED_ROOT": str(tmp_path / "verified")})
    res = rt.ingest(cfg)

    out = root / "training_onion"
    # label schema + training mask written
    assert (out / "onion_cvat_labels.json").exists()
    assert (out / "masks" / sid / "f_000001.png").exists()
    # manifest points at the real rgb path and the written mask
    assert res["manifest"][0]["session_id"] == sid
    assert res["manifest"][0]["image"].endswith("f_000001.png")
    # agreement computed: auto (big) vs verified (sq) -> partial IoU, full recall
    agr = res["agreement"][0]
    assert 0.5 < agr["iou"] < 1.0 and agr["recall"] == 1.0
    assert res["summary"][0]["frames_compared"] == 1
    assert (out / "agreement_summary.csv").exists()
