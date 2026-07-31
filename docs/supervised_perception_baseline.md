# Supervised perception baseline

First supervised perception milestone for the laser weeder: crop-vs-weed
instance segmentation, per-weed LEP localization, RGB-D conversion to a 3D
camera-frame point, and safety-aware abstention.

> **Status.** The infrastructure below is implemented and unit-tested against
> synthetic fixtures. **No model has been trained and no accuracy, latency or
> 3D-error number is reported anywhere in this repository**, because no
> verified annotations exist yet. Every table that would hold such a number is
> marked *not measured*. See [Status by component](#status-by-component).

---

## 1. Architecture

```
                       ┌─────────────────────────────────────────┐
   RGB frame ─────────▶│ STAGE A   YOLO26n-seg (Ultralytics)      │
   (2208×1242)         │  full-frame instance segmentation        │
                       └────────────┬────────────────────────────┘
                                    │ per-instance mask, box, class, score
                       ┌────────────▼────────────────────────────┐
                       │ union of every onion_plant mask          │
                       │  ==> ONION SAFETY MASK (frame-level)     │
                       └────────────┬────────────────────────────┘
                                    │
              weeds only ───────────┤
                                    ▼
   ┌────────────────────────────────────────────────────────────┐
   │ ROI assembly (training/roi.py)                              │
   │  expand box ×1.35 → square → letterbox to 128×128           │
   │  RGB + owning mask + depth transformed IDENTICALLY          │
   └────────────┬───────────────────────────────────────────────┘
                │  ALL weed ROIs of the frame as ONE batch
   ┌────────────▼───────────────────────────────────────────────┐
   │ STAGE B   LEPRoiNet                                         │
   │   RGB encoder (MobileNetV3-style inverted residuals)        │
   │   geometry branch [mask, height_norm, depth_valid] ─┐       │
   │   fused at S/8 ────────────────────────────────────┘       │
   │   → 1-channel LEP heatmap (32×32)                           │
   │   → visibility logits (3)                                   │
   │   → targetability logits (3)                                │
   └────────────┬───────────────────────────────────────────────┘
                │ soft-argmax + spatial covariance  (lep_targets.decode_lep)
   ┌────────────▼───────────────────────────────────────────────┐
   │ 3D localization (perception/depth3d.py)                     │
   │   confidence-weighted depth sample ∩ owning mask            │
   │   bimodality check → MAD → back-projection → 3D covariance  │
   └────────────┬───────────────────────────────────────────────┘
   ┌────────────▼───────────────────────────────────────────────┐
   │ SAFETY DECISION (perception/safety.py)  — can only REJECT   │
   └────────────┬───────────────────────────────────────────────┘
                ▼
        WeedTarget[]  (candidate | abstain + machine-readable reasons)
```

**This produces candidates only.** `perception/safety.py` has no I/O, no
actuator handle, and a test asserts it never acquires one. Turning a candidate
into a laser command is a separate, deliberate act by the control layer.

### Key architectural decisions

| Decision | Why |
|---|---|
| **Two stages, not one multi-head net** | The LEP needs a high-resolution view of one plant's crown. Predicting it from full-frame features at 1/8 resolution throws away the pixels that carry the phyllotactic structure. Two stages also let each be replaced independently — which matters given the Ultralytics licence. |
| **Batch all ROIs per frame** | A dense frame holds 30–60 weeds. Per-instance forward passes spend most of a Jetson latency budget on kernel-launch overhead, not arithmetic. |
| **Geometry in a separate branch, not a 4th RGB channel** | Concatenating depth to RGB makes the first conv mix a metric quantity into colour features at full resolution, and makes "depth missing" indistinguishable from "height zero". A separate branch fused at low resolution keeps the modalities separable, which is what makes depth dropout survivable. |
| **Height above *local* soil, never raw depth** | Raw depth encodes camera mount height — a model trained at one height silently fails at another. Height above the soil ring around *this* plant is a property of the plant, transferable across rigs. |
| **Heatmap + soft-argmax, not direct coordinate regression** | Gives sub-pixel accuracy *and* a free uncertainty estimate (the heatmap's spatial covariance), which is the abstention signal. Direct regression gives a point with no confidence. |
| **Soft-argmax kept OUT of the exported graph** | Decoding in `lep_targets.decode_lep()` keeps the ONNX/TensorRT graph simple and guarantees training-time and deployment-time decode are the same code. |
| **Datumaro, not COCO, as the multitask master** | COCO cannot represent CVAT shape *groups*, so the weed-mask↔LEP link is destroyed on export. See §3. |
| **MobileNetV3-style convolutions, not a transformer** | TensorRT constraint: these ops have long-standing optimised kernels. Attention still needs care to convert cleanly on Jetson. |

**DINOv3 is explicitly out of scope** for this milestone. It is documented as a
future *offline* teacher / dataset-curation tool in §11 — not part of the
runtime path.

---

## 2. Researched sources and tested versions

Research was done against current official documentation (July 2026).

| Component | Version used / verified | Source |
|---|---|---|
| Ultralytics YOLO26 | `yolo26n-seg.pt` confirmed to exist; `-seg` suffix, COCO-pretrained, n/s/m/l/x scales | [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26), [Segment task](https://docs.ultralytics.com/tasks/segment) |
| Datumaro | 1.0 JSON schema: `categories.label.labels[]`, `items[].annotations[]` with `type`/`label_id`/`group`/`points`/`rle` | [CVAT Datumaro format](https://docs.cvat.ai/docs/dataset_management/formats/format-datumaro/) |
| Datumaro Python API | `dm.Dataset.import_from(path, 'datumaro')` | [CVAT dataset formats](https://docs.cvat.ai/docs/dataset_management/formats/) |
| PyTorch | **2.13.0+cpu — actually installed and used to run every Stage B test in this repo** | pytorch.org |
| numpy | Repo pins `>=1.26,<2` for SAM 3; code verified to also run on numpy 2.4.6 (this container) | — |
| OpenCV | 5.0.0 (container); repo requires `opencv-python` | — |
| TensorRT / Jetson | Engines are architecture- and version-specific and must be built on the Orin; measure→optimise→remeasure | NVIDIA TensorRT docs |

### Licensing — read before shipping

**Ultralytics is AGPL-3.0.** Per Ultralytics' own guidance:

- AGPL-3.0 compliance requires publicly releasing the **complete corresponding
  source of the entire derivative work** — this repository, the larger
  application, and, where applicable, model weights.
- **Proprietary or commercial use requires an Ultralytics Enterprise License**,
  and Ultralytics states this applies even to internal company R&D unless the
  whole project is open-sourced under AGPL-3.0.

A commercial laser weeder is squarely in the Enterprise-License case. This is a
**business decision, not a technical one**, and it needs resolving before Stage A
weights ship in a product.

Mitigation already in place: Stage A sits behind `perception/segmenter.py`,
which exposes a plain `Detections` numpy structure. Replacing Ultralytics with
an Apache/BSD-licensed segmenter touches that one file. Stage B (`LEPRoiNet`) is
original code in this repository and carries no such obligation.

Sources: [Ultralytics License](https://www.ultralytics.com/license),
[AGPL-3.0 terms](https://www.ultralytics.com/legal/agpl-3-0-software-license).

---

## 3. Annotation contract (CVAT → training)

**Master format: CVAT "Datumaro 1.0" export.** Not COCO — `cvat_roundtrip.py`
remains correct for segmentation alone, but COCO has no representation for
shape groups, so the weed↔LEP association is lost.

### The ownership rule

> A weed mask and its LEP are associated **only** through the Datumaro `group`
> field. Nearest-neighbour matching is **never** applied to a training target.

In a dense frame the nearest crown is frequently the *neighbouring* plant. An
ungrouped LEP is reported as an error; a nearest-instance hint may appear in
`report.suggestions` marked *"SUGGESTION ONLY — not applied"*, and the importer
does not touch the instance. Note `group: 0` is Datumaro's "no group" sentinel.

### Validated rules

| Rule | Severity |
|---|---|
| Visible + targetable weed has exactly one grouped LEP | error |
| `not_visible` weed is **not** required to have an LEP | — |
| `weed_cluster` must not carry a single LEP | error |
| `onion_plant` never carries a weed LEP | error |
| LEP lies inside its owning mask (tolerance `4 px`) | error beyond tolerance, warning within |
| Duplicate `group_id` within one image | error |
| Ungrouped LEP | error |
| Unknown label / missing categories | **raises** — the schema and code have diverged |
| Compressed RLE without pycocotools | **raises** with the install command |
| Frame with no resolvable session | error — splits would leak |
| Ignore regions | captured, excluded from instances |
| Attributes (`growth_stage`, `targetable`, `species_note`, …) | preserved |

Outputs: `data.yaml` + Ultralytics labels, `lep_manifest.json`,
`dataset_report.json`, `annotations_needing_correction.json`, per-split session
lists and image manifests.

**Images are never copied.** Manifests reference existing paths, stored
posix-style and resolved against `--images-root`, so a large dataset is not
duplicated and Windows paths keep working.

---

## 4. Session-safe splits

Adjacent video frames are near-identical, so a frame-level split measures
memorisation. `training/splits.py` enforces whole-session splits in code:

- A session appears in **exactly one** split (`check_no_leakage` raises otherwise).
- Explicit holdouts always win.
- Sessions sharing `(date, field, camera)` are treated as **one indivisible
  group** — two recordings of the same bed on the same morning are
  near-duplicates and leak just as frames do.
- Sessions with *no* metadata are each their own group (grouping requires
  positive evidence of relatedness, not its absence).
- Deterministic via `zlib.crc32`, **not** `hash()` — Python's string hash is
  salted per process, so a `hash()`-based split silently differs every run.
- The training split is never left empty, even when a few large groups would
  otherwise consume the val/test quotas.

---

## 5. Depth representation

```
height_norm = clip(local_soil_reference − plant_depth, −50mm … 400mm) → [0,1]
```

The reference is the **median depth of an annulus outside the instance**
(radii scale with plant size, so a seedling and a large rosette both sample soil
just beyond their own canopy). If too little of that ring is valid, **no
reference exists and the channel is reported invalid** — an invented reference
biases every height on the plant.

Three geometry channels, each independently meaningful:

| Channel | Meaning |
|---|---|
| `mask` | which pixels belong to **this** plant (ownership) |
| `height_norm` | camera-transferable elevation above local soil |
| `depth_valid` | where `height_norm` may be believed **at all** |

Keeping validity separate is what lets the network distinguish *flat ground*
from *no measurement*.

### Simulated degradation (training only)

Field stereo depth is not the clean map a synthetic pipeline produces. Training
applies holes, range-dependent noise (`5 mm/m`), quantisation, and **complete
depth dropout (p=0.25)** so the RGB path never becomes dependent on a stream
that is frequently invalid.

### Ablations

`rgb` · `rgb_mask` · `rgb_mask_geom` — all three are implemented and tested. The
first two are also the runtime fallbacks when depth is unavailable.

---

## 6. Losses

| Term | Weight | Job |
|---|---|---|
| `heatmap` | 1.0 | Per-pixel localisation (MSE on the Gaussian). Main signal. |
| `soft_argmax` | 0.1 | Direct error on the **decoded** coordinate — the quantity actually used at inference. |
| `visibility` | 0.5 | 3-way classification; the abstention gate. |
| `targetability` | 0.5 | 3-way classification; the annotator's treat/don't verdict. |
| `outside_mask` | 0.2 | Penalises probability mass outside the owning instance — teaches *"this plant's crown"*. |

**Sub-pixel truth is preserved**: the Gaussian is rendered about the exact float
coordinate. Rounding to the heatmap grid would cap accuracy at the stride (4 px)
before training starts.

**Missing is not zero**: `not_visible` samples get weight **0**, not a zero
heatmap. Supervising zeros would teach that the plant has no growth point, which
is false — the point exists, the annotator could not see it.
`partially_occluded_inferable` gets weight 0.5.

**Forbidden augmentations**: Mosaic, MixUp, CutMix and copy-paste are **refused
with an exception** for ROI LEP training — they composite pixels from different
plants into one ROI, destroying the ownership the task depends on. (Mosaic/MixUp
remain fine for Stage A full-frame segmentation; `copy_paste` is disabled there
too, since pasting an onion fabricates crop-safety geometry.)

---

## 7. Safety rules

`decide()` starts from **reject** and every check can veto. There is no branch
that promotes a candidate — acceptance is the absence of any veto, so a
forgotten case fails **closed**. All reasons accumulate; nothing short-circuits.

| Rejection reason | Trigger |
|---|---|
| `onion_plant` | the object is the crop |
| `weed_cluster` | no separable single growth point |
| `lep_not_visible` | visibility says not visible, or below confidence |
| `low_lep_confidence` | heatmap peak below threshold |
| `high_heatmap_uncertainty` | `sigma_px` above limit |
| `outside_owning_mask` | LEP not on the plant it claims |
| `onion_safety_conflict` | laser spot **+ margin** intersects crop tissue |
| `insufficient_valid_depth` | too few valid depth samples |
| `depth_discontinuity` | window straddles a surface boundary |
| `not_targetable` | annotator/model verdict `no` or `uncertain` |
| `class_uncertain` | segmentation confidence too low |
| `high_3d_uncertainty` | 3D sigma above limit |
| `no_lep_predicted` | no LEP produced |

Two details that matter:

- **Crop conflict tests a disc, not a pixel.** The beam has physical extent, so
  the whole spot plus a margin must clear onion tissue.
- **Snapping is bounded and recorded.** A LEP up to `allow_snap_to_mask_px`
  (2 px) outside its mask is nudged in and the correction is written to the
  result. Beyond that it is **rejected**, never silently reprojected.

---

## 8. Output schema

`perception/schema.py` — `WeedTarget` carries provenance (session, frame,
instance), segmentation (class, confidence, bbox, mask ref, area), the 2D LEP
(uv, peak, `sigma_px`, covariance, visibility/targetability probabilities, any
snap), depth/3D (`used_depth`, valid fraction, spread, stats, `xyz_mm`,
`xyz_sigma_mm`, covariance, `is_3d_fallback`), and the verdict
(`safety_status`, `abstained`, `rejection_reasons`, notes).

`FrameResult` adds the onion mask reference, per-stage `timings_ms`, and
`.candidates` / `.abstentions` / `.reason_counts()`.

---

## 9. Commands

PowerShell, into the active `dl` conda env. Use `python -m pip` (not bare `pip`)
so packages install into the interpreter you are running.

```powershell
# Optional training/deployment stack (NOT needed for the unit suite)
python -m pip install -r requirements-training.txt
```

```powershell
# 1. Verified CVAT "Datumaro 1.0" export -> trainable dataset
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root  D:/exports/verified_mixed `
    --images-root    D:/Dataset_Vidalia/sessions `
    --out            D:/Dataset_Vidalia/training/mixed_v1 `
    --holdout-test   vid3_20260108_103135

# 2. Stage A
python -m seeweed3d.training.train_seg `
    --data D:/Dataset_Vidalia/training/mixed_v1/data.yaml `
    --model yolo26n-seg.pt --epochs 100 --imgsz 1024 --device 0

# 3. Stage B  (--input-mode selects the ablation)
python -m seeweed3d.training.train_lep `
    --manifest    D:/Dataset_Vidalia/training/mixed_v1/lep_manifest.json `
    --images-root D:/Dataset_Vidalia/sessions `
    --out         D:/Dataset_Vidalia/runs/lep_v1 `
    --input-mode rgb_mask_geom --device cuda

# 4. Export + parity check
python -m seeweed3d.deploy.export `
    --checkpoint D:/Dataset_Vidalia/runs/lep_v1/best.pt `
    --out        D:/Dataset_Vidalia/runs/lep_v1/export --precision fp16

# 5. Benchmark  (ON THE JETSON — desktop numbers do not transfer)
python -m seeweed3d.deploy.benchmark `
    --checkpoint runs/lep_v1/best.pt --device cuda --out bench.json
```

---

## 10. Evaluation protocol

Evaluate **by whole session** and by difficulty condition (the `difficulty`
attribute: normal / overlapping / blurred / shadowed / wet / truncated).

**Segmentation** — mask mAP50-95, per-class AP/precision/recall, boundary
F-score, small-weed recall. Crop safety is reported **separately and
asymmetrically**: onion recall, **missed onion pixels**, IoU, Dice. Missing crop
tissue can destroy the crop; a false onion merely skips a weed, and IoU averages
those two very different errors together.

**LEP** — pixel error; plant-size-normalised error; % within 2/5/10/15 px;
% within a fraction of plant radius; LEP-inside-owning-mask rate; error by
visibility, class and growth stage; **uncertainty calibration** (does predicted
`sigma_px` track actual error? — an uncertainty that doesn't is worse than none,
since the abstention threshold would reject good targets and pass bad ones).

**Required baseline comparison** — the learned model is compared against bbox
centre, mask centroid, DT peak, and the existing `perception/lep.py`
multi-evidence estimator. `perception/lep.py` is **preserved, not replaced**; it
is also the runtime fallback when no learned model is loaded.

**Safety** — unsafe-target rate, onion-conflict rate, wrong-instance LEP rate,
abstention rate, recall among targetable weeds, false-candidate rate.

**3D** — only when reference 3D labels exist. `metrics_3d()` returns
`{"n": 0, "note": "no reference 3D labels…"}` rather than a number.
**Depth self-consistency is not metric accuracy** and the covariance must never
be reported as one.

**Performance** — per-stage and end-to-end p50/p95, GPU memory, weed-ROI count,
FP32/FP16 (INT8 only after FP16 accuracy is verified).

---

## 11. Known limitations

1. **Nothing is trained.** No accuracy, latency or 3D number exists in this
   repository. Every metric function is unit-tested on synthetic inputs only.
2. **No metric 3D ground truth.** 3D accuracy is unmeasurable until reference
   labels exist. The reported covariance is depth-sample spread.
3. **Ultralytics AGPL-3.0** — see §2. Unresolved business decision.
4. **Jetson performance is unmeasured.** No number in this repo came from an
   Orin. TensorRT engines must be built and benchmarked on the device.
5. **Bed-plane fallback is a fallback.** It assumes the crown lies on the bed
   surface — wrong by the plant's own height — so it returns a deliberately
   large uncertainty and `is_3d_fallback=True`.
6. **`weed_cluster` is always rejected.** Correct for safety, but clusters are
   common in dense field imagery, so recall among *all* weeds will be limited by
   how often the annotator must use that class.
7. **Compressed-RLE exports need pycocotools**; polygon exports do not.
8. **Single-frame only.** No temporal tracking yet — see below.

## 12. Future work

- **DINOv3 as an offline teacher** — dataset curation, cluster-then-label for
  species assignment, and pseudo-labelling the unverified pool. Explicitly *not*
  in the runtime path.
- **Teacher–student semi-supervised training** on the large unverified frame
  pool, re-verifying only low-confidence and disagreeing cases.
- **Temporal tracking** — a weed seen over several frames gives several LEP
  estimates; fusing them should cut variance and allow a targeting decision to
  be deferred rather than abstained.
- **INT8 calibration** on real field frames, after FP16 accuracy is verified.
- **Metric 3D validation rig** to make §10's 3D metrics computable.
