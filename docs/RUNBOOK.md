# SeeWeed3D — complete runbook

Every step from raw ZED recordings to running inference, with the exact
commands. Windows/PowerShell throughout; on Linux swap `` ` `` line-continuations
for `\` and `D:/...` for your paths.

**Use `python -m pip`, never bare `pip`** — that guarantees packages land in the
interpreter you are actually running (your `dl` conda env), not some other one
on PATH.

---

## Contents

| Stage | What it does | Needs GPU? |
|---|---|---|
| [0. Install](#0-install) | environments | no |
| [1. Extract](#1-extract-recordings) | recordings → indexed frame pool | no |
| [2. Curate](#2-curate-the-pool-optional) | drop redundant/bad frames | no |
| [3. Prelabel with SAM 3](#3-prelabel-with-sam-3) | auto-annotations to correct | **yes** |
| [3a. Mixed scenes](#3a-mixed-scenes-onions-and-weeds--prelabel_mixed_sam3py) | precise masks only, classes assigned in CVAT | **yes** |
| [3b. Mine the pool](#3b-mine-the-pool-for-the-next-batch) | pick the frames worth annotating next | **yes** |
| [3c. Split-zone drives](#3c-one-drive-that-changes-scene--weeds-first-then-onions) | weeds for a stretch, then onions — one session, two runs | **yes** |
| [4. CVAT](#4-annotate-and-correct-in-cvat) | human verification | no |
| [5. Merge + prepare](#5-merge-exports-and-build-the-training-dataset) | many CVAT tasks → one dataset, split by session + scene | no |
| [6. Train Stage A](#6-train-stage-a-segmentation) | crop-vs-weed segmentation | **yes** |
| [6b. RF-DETR-Seg](#6b-train-stage-a-on-rf-detr-seg-the-advanced-path) | same, on the real-time transformer backend | **yes** |
| [7. Train Stage B](#7-train-stage-b-lep) | LEP localization | **yes** |
| [8. Evaluate](#8-evaluate) | metrics by session | no |
| [9. Inference](#9-run-inference) | predict on unlabelled frames; full RGB-D pipeline | yes (practically) |
| [10. Deploy](#10-export-and-benchmark-jetson) | ONNX/TensorRT + benchmark | on Jetson |

**Related:** [project history](../CHANGELOG.md) ·
[extraction fidelity](extraction_quality.md) ·
[growing the dataset](dataset_growth.md) ·
[dataset assembly + splits](dataset_assembly.md) ·
[depth-assisted masking](depth_assisted_masking.md) ·
[mixed-scene prelabeling](mixed_prelabeling.md) ·
[experiment tracking](experiment_tracking.md) ·
[edge model research](edge_model_research.md) ·
[LEP localization explained](lep_localization_explained.md)

---

## 0. Install

Two environments, deliberately separate.

```powershell
# --- Environment A: data pipeline + SAM 3 prelabeling -----------------------
conda activate dl
python -m pip install -r requirements.txt

# SAM 3 (only needed for stage 3). Pulls torch.
python -m pip install "git+https://github.com/facebookresearch/sam3.git"
# Windows + torch 2.5:
python -m pip install "triton-windows>=3.1,<3.2"
```

> **Do not upgrade numpy past 2.0 in this environment.** `requirements.txt`
> pins `numpy>=1.26,<2` because SAM 3 requires it. Breaking that pin breaks
> prelabeling.

```powershell
# --- Environment B: training + deployment ----------------------------------
# Separate env so a training dependency can never break SAM 3's numpy pin.
conda create -n sw-train python=3.11 -y
conda activate sw-train
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt

# CUDA torch matching your driver (check with: nvidia-smi)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pytest tests/ -q
```

**Licence note:** the default segmentation backend is torchvision Mask R-CNN
(BSD-3-Clause) — nothing to install beyond the above, and no obligation on a
commercial product. Ultralytics is **not** installed by default because it is
AGPL-3.0. See [§6](#6-train-stage-a-segmentation).

---

## 1. Extract recordings

Turns raw ZED recordings into an indexed, QC'd frame pool.

Edit the CONFIG block at the top of `seeweed3d/extraction/extract_sessions.py`:

```python
INPUT_ROOTS = [r"E:\Field_Recordings\2026-01-08"]   # one entry per visit
OUTPUT_ROOT = r"E:\Dataset_Vidalia"
DRY_RUN     = True                                   # run this first
```

```powershell
conda activate dl
python seeweed3d/extraction/extract_sessions.py     # DRY_RUN=True: inspect
# then set DRY_RUN = False and run again
python seeweed3d/extraction/extract_sessions.py
```

**Produces** `E:\Dataset_Vidalia\sessions\<session_id>\` containing `rgb/`,
`depth/`, `conf/`, `meta/pool.csv`, `meta/frames_index.csv`,
`meta/calibration.json`, plus a top-level `registry.csv`.

Read `registry.csv` before continuing — check `median_depth_valid_frac_veg` and
pick which sessions you will hold out for testing.

> **Depth is raw uint16 millimetres, `0` = invalid.** Never rescale it. Always
> read it via `common/depth_utils.load_depth_mm()`.

### Are the extracted frames the best quality available?

Yes, and the parts that could silently not be are pinned by tests.

- **PNG output is lossless.** `PNG_COMPRESSION` trades file size against write
  speed and cannot cost quality at any level.
- **Nothing is resized, and nothing is re-encoded.** Frames come out at the
  source resolution and bit depth.
- **v2 (FFV1/MKV) captures decode bit-exact** — RGB and depth both, asserted by
  tests. For those sessions the extracted PNGs are byte-identical to what the
  camera wrote, which is the ceiling.
- **v1 (AVI/mpeg4) captures are lossy at the source.** That loss happened in
  the field when `cv2.VideoWriter` re-encoded with XVID, and no extraction
  setting can undo it. What extraction *can* control is not adding more.

`SWS_FLAGS: "accurate_rnd"` is the one setting that matters for v1 sources.
swscale truncates by default, leaving a systematic negative bias — measured at
a uniform **−2 per channel** on flat patches, dropping to about **−0.8** with
rounding enabled. That is not cosmetic in this pipeline: the vegetation prior
thresholds ExG and compares green against blue, so a channel-dependent error
moves real segmentation decisions. Measured on a synthetic field frame it cut
`vegetation_mask()` disagreement against the uncompressed original from
**0.607% to 0.474%** of pixels, at no measurable decode cost.

**Three things were measured and deliberately *not* changed**, so they don't
get re-tried:

| Tried | Result |
|---|---|
| `full_chroma_int` | **No-op.** It governs chroma interpolation during *resizing*, and extraction never resizes |
| Forcing BT.709 | **Worse.** Flat-patch error swings −21 on green, +7 on red — a matrix mismatch signature. The default (BT.601) leaves a colour-neutral offset, so the defaults at both ends already agree |
| Forcing full-range input | **3.6× worse.** The streams really are limited-range, and ffmpeg's default already reads them that way |

v1 AVIs carry **no colour metadata at all** (`color_space`, `color_range` and
`color_primaries` are all absent), so both encoder and decoder fall back to
their own defaults — and those defaults happen to agree. Leave this alone
unless a future capture writes tagged colour metadata.

`meta/session.json` records the decode settings under `decode`, since two
sessions extracted under different flags are not strictly comparable and the
difference is invisible in the PNGs afterwards.

**The real quality lever is at capture, not extraction.** A v1 AVI session has
already lost detail that cannot be recovered; a v2 FFV1 session has lost
nothing. See `docs/capture_changelog.md`.

**→ [`docs/extraction_quality.md`](extraction_quality.md)** is the full
reference account of extraction fidelity — what is guaranteed, what was
measured, and what is explicitly not claimed. Written to be quotable in a
methods section.

### Mixed containers, and checking depth before you trust it

A campaign may have written **AVI** in its early sessions and **MKV** in its
later ones under one parent folder. Both are discovered — `VIDEO_SUFFIXES`
covers `.mkv .avi .mp4 .mov`, and patterns match the file *stem*, so
`RGB_video.avi` and `RGB_video.mkv` are found by the same rule. Anything in a
container not on that list is reported as a `[SKIP]` line naming the extension,
never silently dropped.

**The container is not the issue; the depth pixel format is.** Depth is decoded
by asking ffmpeg for `gray16le`, and ffmpeg will produce `gray16le` from *any*
source — including an 8-bit lossy one, by scaling up values it invented. The
result is a valid PNG full of plausible millimetres that are **fiction**, and
nothing downstream can detect it: the QC fractions compute, the range looks
sane, and the LEP's canopy-height channel reads the noise as terrain.

AVI is why this matters. Its usual codecs (MJPEG, XVID, DIVX) have **no 16-bit
grayscale format at all**. But FFV1 in AVI is perfectly fine and a badly
remuxed MKV is not, so this is checked per session rather than assumed from the
extension.

Check yours before a full run:

```powershell
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt `
    -of csv=p=0 "E:\...\Session_20250221_130957\Depth_video.avi"
```

| Output | Meaning |
|---|---|
| `ffv1,gray16le` | real millimetres — extract normally |
| `mjpeg,yuvj420p` / `mpeg4,yuv420p` / anything 8-bit | **not depth data.** RGB is still extracted; the session has no usable depth |

`REQUIRE_16BIT_DEPTH: True` (the default) drops the depth stream for such a
session, extracts RGB, and prints why. That session is still fully usable for
segmentation — only 3D and LEP need real depth. Set it to `False` only if you
have verified the writer actually stored millimetres.

### If the 8-bit file is a depth *preview*

"Not 16-bit" covers two very different things, and the difference decides
whether the geometry is coarsely recoverable or gone. Find out which:

```powershell
python -m seeweed3d.validation.inspect_depth_video `
    --video "E:\...\Depth_video.avi" --save-preview depth_frame.png
```

| Verdict | Meaning |
|---|---|
| `true_16bit` | metric depth — extract normally |
| `grayscale_8bit` | a preview. Geometry present at 1/256 of its scale |
| `colormapped_8bit` | a colourised preview. Same, and the colormap is named |
| `per_frame_normalised_8bit` | **nothing to recover, ever.** See below |
| `unknown_8bit` | not a range image. Nothing to recover |

**`per_frame_normalised_8bit` is the one that cannot be fixed by calibration.**
Some v1 capture builds wrote depth as

```python
max_val = depth_data.max()                       # recomputed EVERY frame
depth_norm = (depth_data / max_val * 255).astype(np.uint8)
```

The scale is a property of the individual frame — one distant outlier pixel
sets it for every other pixel in that frame — so two frames of identical
geometry get different codes whenever their farthest point differs. There is no
constant to find, and `depth_vis_max_mm` never entered into it. The detector
looks for the signature: every frame pinned at the top of the range, the maxima
agreeing with each other, and the peak an outlier rather than a saturated
plateau. It errs toward refusing, because recovering garbage depth costs far
more than declining a file that turns out to have been recoverable.

The fix is at the capture end, one line: write raw millimetres as 16-bit
instead of a normalised 8-bit preview — which is what `capture/zed_capture.py`
already does.

For the two recoverable cases, `RECOVER_8BIT_DEPTH: True` inverts the capture
GUI's display scaling and writes approximate millimetres:

```python
CONFIG["RECOVER_8BIT_DEPTH"] = True
CONFIG["DEPTH_VIS_MAX_MM"]   = 3000.0   # the scale it was DRAWN at
```

**Read this before turning it on.**

- Output goes to **`depth_approx/`, never `depth/`**. A file under `depth/` is
  metric millimetres and everything downstream is entitled to assume so;
  putting an approximation there would recreate by hand the exact silent
  corruption `REQUIRE_16BIT_DEPTH` exists to prevent. The two directories hold
  identical-looking uint16 PNGs, so the distinction lives in
  `meta/session.json` → `depth_kind` (`metric` | `approximate` | `none`) and in
  a `depth_kind` column in `registry.csv`. **Check it before using depth.**
- `DEPTH_VIS_MAX_MM` is an *assumption* unless your sessions have a
  `session_meta.txt` to take it from. Every recovered distance inherits it.
  **Measure it instead**, if any session from the same rig has metric depth —
  the soil sits at the mount's working distance in both, so the ratio between
  the two depth distributions *is* the scale:

  ```powershell
  # Straight from the source videos - nothing needs to be extracted yet, so
  # the whole visit goes through in ONE extraction pass.
  python -m seeweed3d.validation.calibrate_preview_scale `
      --metric-video "E:\...\Session_20250226_202127\Depth_video.mkv" `
      --preview      "E:\...\Session_20250221_130957\Depth_video.avi"

  # Or against an already-extracted session, if you have one:
  #   --metric-session E:\dataset\sessions\Visit1_20250228_151838
  ```

  It fits at seven percentiles rather than one, because a single-point ratio
  always returns a number whether or not code and distance are linearly
  related. Agreement across the distribution is what makes the answer mean
  something.

  **Always pass `--control-video`** — a *second* metric video from the same
  rig. Drift has two possible causes and they look identical: the preview is
  non-linear, or the two sessions simply do not frame the same geometry (a
  different day, a different patch of row, more or less of the machine in
  view). Two known-linear sessions compared the same way measure the second
  cause on its own, since the map from real millimetres to real millimetres is
  the identity. If the preview drifts no more than they do, the verdict is
  **inconclusive**, not rejected — and abandoning recoverable data on a
  rejection that was really about the weather is the expensive mistake.

  ```powershell
  python -m seeweed3d.validation.calibrate_preview_scale `
      --metric-video  "E:\...\Session_20250226_202127\Depth_video.mkv" `
      --preview       "E:\...\Session_20250221_131902\Depth_video.avi" `
      --control-video "E:\...\Session_20250228_151838\Depth_video.mkv"
  ```

  Match the **capture mode** between preview and reference where you can. A
  session recorded at 1920x1080@30 frames different ground from one at
  2208x1242@15, and that difference alone can produce a bimodal code
  distribution that has nothing to do with the preview's linearity.

  Inconclusive is not permission: the scale was never confirmed, so leave
  `RECOVER_8BIT_DEPTH` off either way.
- At 3000 mm one level is **11.8 mm**, before DCT ringing that concentrates at
  exactly the depth discontinuities that matter. Fine for a ground plane,
  camera height or row geometry. **Not** usable for the LEP canopy-height
  channel, where the whole signal is a few centimetres of relief on one plant.
- Approximate depth is deliberately **kept out of
  `median_depth_valid_frac_veg`**, so that column keeps meaning one thing
  across sessions.
- If the run warns that the invalid sentinel did not survive the encode,
  invalid regions are now indistinguishable from near distances, and any 3D
  point built there is fiction.

---

## 2. Curate the pool (optional)

Drops near-duplicate frames (from driving slowly) and bad frames. **Edits
`pool.csv` only — no image is ever deleted or renamed**, and `RESTORE_ALL`
undoes everything.

Edit the top of `seeweed3d/extraction/curate_pool.py`:

```python
DATASET_ROOT = r"E:\Dataset_Vidalia"
DRY_RUN      = True
CONFIG["MIN_SHIFT_FRAC"] = 0.6      # see the sweep table it prints
```

```powershell
python seeweed3d/extraction/curate_pool.py          # dry run: read the sweep
# choose MIN_SHIFT_FRAC from the "overlap between kept frames" column,
# then set DRY_RUN = False and run again
python seeweed3d/extraction/curate_pool.py
```

Pick the threshold from the **overlap** column, not the drop percentage: at
`0.15`, 85% of every kept frame is ground you already annotated in the previous
one. Since you hand-correct everything in CVAT, annotation cost dominates —
`0.4–0.6` is usually the right range.

To undo: set `RESTORE_ALL = True` and re-run.

---

## 3. Prelabel with SAM 3

Auto-generates annotations so CVAT is *correction*, not annotation from scratch.

**Get the weights** (gated — request access on HuggingFace first):

```powershell
huggingface-cli login
huggingface-cli download facebook/sam3 --local-dir E:\Models\sam3
```

Edit the top of the prelabeler you need:

```python
DATASET_ROOT   = r"E:\Dataset_Vidalia"
SAM_VERSION    = "sam3"
SAM_CHECKPOINT = r"E:\Models\sam3\sam3.pt"
CONFIG["ONLY_SESSIONS"]      = ["vid3_20260108_103135"]   # IMPORTANT
CONFIG["LIMIT_PER_SESSION"]  = 20                          # trial first, then None
```

```powershell
conda activate dl

# Weed-only scenes
python seeweed3d/annotation/prelabel_weeds_sam3.py

# Onion-only scenes
python seeweed3d/annotation/prelabel_onions_sam3.py
```

### "Why did it only prelabel 15 frames?"

Every prelabeler reads the **curated** pool — rows marked `dropped` in
`meta/pool.csv` are skipped, which is the whole point of §2. Each session now
says what that cost:

```
  [Visit1_20250221_142227] pool: 754 frame(s) | 739 dropped by curation
      (redundant with the previous frame=739) -> prelabelling 15
  [!] curation removed 98% of this session. If that was not deliberate, restore
      it with extraction/curate_pool.py (RESTORE_ALL = True) and re-curate with
      a lower MIN_SHIFT_FRAC.
```

Three separate things can shrink a run, and the line separates them:

| Cause | Fix |
|---|---|
| **curation dropped them** — `MIN_SHIFT_FRAC` too high is the usual reason | `curate_pool.py` with `RESTORE_ALL = True`, then re-curate lower |
| **`ONLY_FRAMES`** restricted the run to one stretch | clear it, or widen the range |
| **`LIMIT_PER_SESSION`** capped it for a trial | set it to `None` |

A fourth: only the sessions under `DATASET_ROOT/sessions/` are seen at all, and
`ONLY_SESSIONS` narrows that further. `dir E:\...\sessions` tells you how many
exist; the run's first line tells you how many it selected.

**Check `auto_labels_weeds/<session>/preview/` before doing a full run.** Look
for red outlines on bare ground — if you see them, the vegetation prior is
false-positiving on your substrate and you should tighten
`RECOVER_MIN_VEG_SCORE` / leave `RECOVER_MISSED_PLANTS` off.

Weed previews draw every outline **red** with no LEP dot, because the default
class's ontology colour is a mid grey that disappears against dry soil and the
dot sits on the part of the mask you most need to see. Neither reaches CVAT —
class colours there come from the label schema. To judge boundaries rather than
just detections, set `PREVIEW_SCALE = 1.0`; at `0.5` you are inspecting a mask
at half the resolution it was drawn. `PREVIEW_OUTLINE_BGR = None` restores the
per-class palette, `PREVIEW_SHOW_LEP = True` the dots.

**Produces** per session, under `auto_labels_weeds/<session>/`:

| Item | Use |
|---|---|
| `cvat_ready/` | **upload this folder to CVAT** |
| `instances_default.json` | COCO import for CVAT |
| `weed_cvat_labels.json` | paste into CVAT's Raw label editor |
| `preview/` | sanity-check before committing to a full run |
| `flagged_rgb/` | colour-cast/glare frames — annotate separately or skip |

> `vegetation == weed` holds only in **weed-only** recordings, and
> `vegetation == onion` only in **onion-only** ones. Never point either of these
> two at a mixed scene — use §3a.
>
> A session that is weed-only for its first stretch and onion-only after it is
> **not** a mixed scene — it is two single-class scenes end to end, and each
> half wants its own prelabeler. See §3c.

---

### 3a. Mixed scenes (onions *and* weeds) — `prelabel_mixed_sam3.py`

Neither prelabeler above can be pointed at a mixed scene: one would call every
onion a weed, the other would call every weed crop. This module drops the class
proposal entirely and does one job — **one precise mask per plant** — emitting a
single class, `plant`, that you reassign in CVAT with one keystroke per shape.

```powershell
# edit DATASET_ROOT and CONFIG["ONLY_SESSIONS"] at the top, then
python seeweed3d/annotation/prelabel_mixed_sam3.py
```

Output lands under `auto_labels_mixed/<session>/` in the same layout as above,
plus `review_first.txt` — the frames whose numbers say the masks are wrong.
Correct those first.

The line to watch in the console is **coverage**, not the instance count:

```
coverage: 97.3% of vegetation inside an instance | 402 from SAM + 29 recovered
```

Vegetation outside every instance is plant the annotator is never shown, because
in a prelabelled task nobody draws what is not already there.

Previews are coloured **by instance index**, not by class — every instance has
the same class here, so a per-class palette would show nothing. Two adjacent
plants in two different colours is the signal.

**If onion comes out as a scatter of tiny speckle instances instead of one
clean leaf per plant**, that is onion's glossy, glaucous leaf surface
fragmenting the vegetation prior, not a SAM problem — its waxy blue-green
cuticle both lifts blue reflectance past green and throws specular highlights
that carry no leaf colour at all, and both defects cut straight across a
leaf's width where the existing pinhole-filling cannot reach them. Tune
`VEG_BLUE_TOLERANCE` and `VEG_BRIDGE_PX` first; see "Onion's glaucous leaf" in
`docs/mixed_prelabeling.md` for the full mechanism and why `VEG_CLOSE_KERNEL_PX`
is the last resort, not the first.

**`docs/mixed_prelabeling.md` is the full guide**: why one homogeneous class is
correct rather than a shortcut (shape says *blade → grass*, and an onion **is** a
blade), how the vegetation prior and SAM are combined so each only answers what
it is good at, why the split that fragmented plants in `prelabel_weeds_sam3.py`
cannot happen on this path, and which knob to turn for each mask symptom.

**The onion prelabeler does not write its own label file.** It produces
`cvat_ready/`, `instances_default.json`, `preview/` and `flagged_rgb/` under
`auto_labels_onion/<session>/`, same as above, but the label JSON to paste comes
from `annotation/cvat_roundtrip.ONION_CVAT_LABELS` — get it either way:

```powershell
python seeweed3d/annotation/regen_cvat_labels.py
# -> auto_labels_onion/<session>/onion_cvat_labels.json
```

or in Python: `from annotation.cvat_roundtrip import write_labels`.

**⚠️ Do not reuse an old hand-copied label file.** `common/ontology.py` fixed
every class name to `lower_snake_case` (`onion_plant`, `ignore_region`, …). A
schema from before that rename (`"onion plant"` with a space, sometimes found
copied out of the now-superseded `extraction/select_batches.py`) will not match
what the SAM 3 prelabeler's own COCO category name uses, so the import creates
a **second, duplicate label** instead of filling the one already prelabelled —
and a Datumaro export under that old name fails with `unknown label`. Always
regenerate rather than reuse a saved copy.

**The same rename can bite an old `instances_default.json` itself**, not just
the label file — a SAM 3 onion prelabeling run from before the rename wrote
`"category_id"` entries named `"onion plant"`. That is invisible until CVAT
silently creates a duplicate label on import. Check it before uploading:

```powershell
python -m seeweed3d.annotation.fix_coco_categories `
    --in  auto_labels_onion/<session>/instances_default.json `
    --out auto_labels_onion/<session>/instances_default.json `
    --dry-run
```

Drop `--dry-run` to write the fix. It only ever touches `categories[*].name`;
every mask, box and image reference is untouched. It refuses to write if any
category name is neither current nor a known pre-rename alias, rather than
guessing — extend `KNOWN_RENAMES` in the script if a genuinely new one turns up.

If you change the ontology later, you do **not** need to re-run SAM 3 — the
same command above refreshes every existing session's label file in seconds
without touching any mask, preview, or `instances_default.json`.

---

## 3b. Mine the pool for the next batch

**Once a model exists, this replaces §3 for every subsequent round.** SAM 3
proposes masks with generic classes; your own checkpoint knows this ontology, so
the export is *correction* rather than annotation.

```powershell
conda activate sw-train
python seeweed3d/annotation/mine_pool.py     # edit the config block first
```

It runs your model over the unlabelled pool, ranks every frame by how much
annotating it would teach (uncertainty + class rarity + crop risk, then a
diversity pass so you do not get 60 pictures of one plant), and writes
`cvat_ready/` plus `instances_default.json` — the same layout the SAM 3
prelabelers produce, so §4 is unchanged.

| Setting | Note |
|---|---|
| `HOLDOUT_SESSIONS` | **use it.** A session you intend as a test set stops being one the moment it is annotated into training, and this is the last point where that is preventable |
| `CONF` | 0.20 — *below* deployment. A spurious mask costs one delete; a missing one costs you noticing an absence |
| `STRIDE` | 5. Consecutive ZED frames are near-identical; scanning all of them wastes inference on duplicates |
| `BATCH_SIZE` | size it to what you will actually annotate this round |

Frames already in `seg_manifest.json` are excluded automatically — the same
frame in two CVAT tasks produces two versions of the truth.

### The round ledger

Every batch is recorded in `al_rounds.json` under `DATASET_DIR`, which does two
things a single mining pass cannot.

**Frames still out with an annotator are skipped.** A frame exported into a
CVAT task nobody has finished is not *annotated* yet, so nothing in the dataset
excludes it, and the same model scores it exactly as before — so without the
ledger it is selected again and you get a batch you are half way through.

**A round can be measured.** Active learning claims these frames teach more
than random ones; the test is the metric before against the metric after, on an
unchanged test set.

```powershell
# after the corrected export is merged with make_dataset.py
python -m seeweed3d.training.al_round merge  --dataset E:\Dataset_Vidalia\training1

# after retraining and evaluating on the SAME test set
python -m seeweed3d.training.al_round metrics --dataset E:\Dataset_Vidalia\training1 `
    --round 3 --before ...\run4\eval\metrics.json --after ...\run5\eval\metrics.json

python -m seeweed3d.training.al_round status  --dataset E:\Dataset_Vidalia\training1
```

A round whose deltas are ~0 is information, not a failure: the bottleneck is
somewhere frame selection cannot reach, usually a class with too few instances
to be learnable. Release a task that will never be finished with
`al_round abandon --round N`, rather than switching `SKIP_FRAMES_IN_FLIGHT` off.

> These are **predictions, not truth**. The model's recall is well under 1.0, so
> every frame is missing instances — and the ones it missed are worth the most.
> Open `analysis/report.html` and look at the missed-weed gallery before
> starting a batch; it re-tunes your eye for what to look for.

The full strategy — active learning, teacher–student, what to target and in what
order — is in **[growing the dataset](dataset_growth.md)**.

---

## 3c. One drive that *changes* scene — weeds first, then onions

A drive that starts in a weedy stretch and later runs into clean onion rows is
**not** a mixed scene. A mixed scene is one where both classes are present *in
the same frame*, and that is the only reason §3a gives up the class proposal.
Here every frame still has exactly one class in it — the class just changes
partway through the session.

So do not reach for `prelabel_mixed_sam3.py`. Split the session by frame index
and run the prelabeler whose assumption actually holds on each stretch:
each half then comes out of SAM already carrying the **right class**, and the
CVAT pass is mask correction only — no per-shape reassignment at all.

**1. Find the transition.** The previews are the fastest way; the frame index is
the number at the end of the filename, and it is the same number in `preview/`,
`rgb/` and `meta/pool.csv`.

```powershell
explorer E:\Dataset_Vidalia\sessions\<session>\preview
```

Note the last clearly-weed frame and the first clearly-onion one.

**2. Run each prelabeler over its own range.** `ONLY_FRAMES` takes the same
tokens curation's `MANUAL_DROPS` does — a bare index, an inclusive `a-b` range,
an open-ended `1500-` or `-250`, or a filename pasted straight out of `preview/`
(`.jpg` is accepted, not just the `.png` source):

```python
# prelabel_weeds_sam3.py
CONFIG["ONLY_SESSIONS"] = ["vid3_20260108_103135"]
CONFIG["ONLY_FRAMES"]   = {"vid3_20260108_103135": ["0-1200"]}
```

```python
# prelabel_onions_sam3.py
CONFIG["ONLY_SESSIONS"] = ["vid3_20260108_103135"]
CONFIG["ONLY_FRAMES"]   = {"vid3_20260108_103135": ["1500-"]}
```

```powershell
python seeweed3d/annotation/prelabel_weeds_sam3.py
python seeweed3d/annotation/prelabel_onions_sam3.py
```

The two runs write to `auto_labels_weeds/<session>/` and
`auto_labels_onion/<session>/` — separate folders, so they are two CVAT tasks
and neither overwrites the other.

**3. Leave a gap at the transition.** The `1200 → 1500` hole above is
deliberate. Somewhere in there the two crops overlap, or you cannot tell from a
preview which you are looking at, and a single-class prelabeler run over that
stretch will state a class confidently and wrongly. **Calling an onion a weed
is the worst error this project can make** — it is the one that puts a laser on
the crop. Give the ambiguous stretch to `prelabel_mixed_sam3.py`, which proposes
masks without proposing a class, or leave it out of this round entirely:

```python
# prelabel_mixed_sam3.py — only the uncertain stretch
CONFIG["ONLY_FRAMES"] = {"vid3_20260108_103135": ["1201-1499"]}
```

**Do not let two ranges overlap.** A frame prelabelled under both assumptions
reaches CVAT twice with contradictory classes, and nothing downstream will tell
you which one you corrected.

`ONLY_FRAMES` is applied **before** `LIMIT_PER_SESSION`, so a
`LIMIT_PER_SESSION = 20` trial samples the stretch you selected. It is inert
until you set it — every existing single-zone run behaves exactly as before.

---

## 4. Annotate and correct in CVAT

**One CVAT task per session.** Tasks stay manageable, and a session is the unit
that train/val/test splits must respect anyway. They get merged in [§5](#5-merge-exports-and-build-the-training-dataset).

### 4.1 Create the task

1. **Tasks → + → Create new task**
2. Name it **exactly the session id** (e.g. `vid3_20260108_103135`) — this is
   how sessions are traced later.
3. **Select files** → upload `auto_labels_weeds/<session>/cvat_ready/`
4. Open the **Raw** tab in the label editor and paste the entire contents of
   `weed_cvat_labels.json`.
5. Submit.

### 4.2 Import the prelabels

**Task menu → Upload annotations → COCO 1.0** → `instances_default.json`.

*(COCO is fine for this direction — it only has to carry masks in. The
export direction must be Datumaro, because COCO cannot represent the mask↔LEP
grouping.)*

### 4.3 Correct the annotations

For every plant:

1. **Fix the class.** SAM proposes only `grass_weed`, `weed_cluster` or
   `other_weed`. `cutleaf_evening_primrose` vs `wild_radish` is an *appearance*
   question that shape cannot answer — those are always yours to assign.
2. **Fix the mask boundary** where it is wrong.
3. **Place the `weed_LEP` point** at the centre of the youngest emerging tissue
   (the crown / apical meristem). *Optional on a first pass — see
   "LEPs are optional" below.*
4. **⚠️ GROUP the LEP with its weed mask.** Select both shapes → press **`G`**.

> **Grouping is the single most important step.** The mask↔LEP link is carried
> *only* by the group id. An ungrouped LEP is **rejected** by the importer — it
> is never auto-assigned to the nearest weed, because in a dense frame the
> nearest crown is frequently the neighbouring plant, and a wrong owner aims a
> 60 W laser at the wrong tissue.

5. Set the attributes: `lep_visibility`, `targetable`, `growth_stage`.

**Rules the importer enforces** (violations are listed for you to fix):

| Rule | Why |
|---|---|
| A `visible` + `targetable=yes` weed has exactly one grouped LEP | otherwise there is nothing to train on |
| A `not_visible` weed needs **no** LEP | absence is legitimate; never invent one |
| `weed_cluster` must **not** have an LEP | a cluster has no single growth point |
| `onion_plant` never has an LEP | the crop is never a target |
| The LEP lies inside its own mask (±4 px) | otherwise it is grouped with the wrong plant |
| Each plant has its own group id | duplicate group ids make ownership ambiguous |

Use `ignore_region` for areas that should be excluded from training entirely
(severe blur, glare, the rig in frame).

### LEPs are optional — you can annotate masks first

**Stage A (crop-vs-weed segmentation) does not use LEPs at all.** It trains from
masks and classes only. Stage B (`LEPRoiNet`) is the only thing that needs them,
and until it exists the pipeline uses the hand-engineered estimator in
`perception/lep.py`, which needs no training data whatsoever.

So a mask-only annotation round is a legitimate, complete deliverable.
`prepare_dataset.py` detects it automatically:

| LEP points in the export | What happens |
|---|---|
| **zero**, with weeds present | `dataset_kind: segmentation_only`. The LEP requirement is lifted, no `missing_lep` errors, no `lep_manifest.json`. Stage A trains normally. |
| **some but not all** | A real contract gap — every weed still missing one is reported in `annotations_needing_correction.json`. Half-labelled is worse than not labelled, because Stage B would learn from a biased subset. |
| **all of them** | `dataset_kind: multitask`. Both stages train. |

Pass `--no-require-lep` to force segmentation-only even when a few LEPs exist
(they are kept in the records, just not required).

> **Note on prelabels:** the SAM 3 prelabeler estimates an LEP per weed, but the
> CVAT import in §4.2 is **COCO**, and COCO has no point annotation type — so
> those proposals do not reach CVAT and every LEP placed there is placed by hand.
> That is why deferring them is reasonable: do the masks now, and add LEPs in a
> later pass on the frames you actually intend to train Stage B on.

### 4.4 Export

**Task menu → Export annotations → `Datumaro 1.0`** → download → **unzip**.

> **Must be Datumaro 1.0, not COCO.** COCO has no representation for shape
> groups, so exporting as COCO silently destroys every mask↔LEP association you
> just created.

Put every unzipped export under one parent folder:

```
E:\CVAT_exports\
  vid3_20260108_103135\annotations\default.json
  vid3_20260108_110444\annotations\default.json
  onion1_20260112_090000\annotations\default.json
```

---

## 5. Merge exports and build the training dataset

This is where the separate CVAT tasks become one dataset.

> **`docs/dataset_assembly.md` is the full guide** — how merging resolves
> labels by name rather than by id, why the split is by session and never by
> frame, how scene stratification keeps onion-only / weed-only / mixed
> represented in every split, and how to pin a test session so scores stay
> comparable across rounds.

Two settings decide whether the numbers this dataset produces mean anything:

```python
# A FIXED test set. A fraction re-draws it every time the dataset grows, so two
# rounds' scores are computed on different rulers and no improvement can be
# attributed to anything.
"HOLDOUT_TEST_SESSIONS": ["vid3_20260108_110444"],

# Allocate within each scene, so val and test each contain onion-only,
# weed-only AND mixed. Measured over 40 seeds on a six-session set, the
# scene-blind allocator gave validation a single scene one time in four.
"STRATIFY_BY_SCENE": True,
```

The scene comes from each session's `meta/session.json` (`scene_hint`, set per
input folder in `extract_sessions.py`). If you never set it, the build says so
and allocates those sessions without scene information.

### What `--images-root` / `IMAGES_ROOT` means

The **sessions** folder — the one whose *children* are session ids. Not a
session itself, and not its `rgb/` subfolder.

```
<IMAGES_ROOT>\
  vid2_20260108_122731\
    rgb\    vid2_20260108_122731_000123.png    ← training images
    depth\  vid2_20260108_122731_000123.png    ← same filename, NOT an image
    meta\   pool.csv  frames_index.csv  session.json  calibration.json
  vid3_20260108_103135\
    ...
```

CVAT tasks are flat uploads, so an export's media path is usually a bare
filename with no session folder in it. Resolution therefore reconstructs the
canonical path from the session id embedded in the name
(`<session_id>_<index>.png`) and looks in `rgb/` **first** — the `depth/` frame
has the identical filename, so a blind recursive search could return it as a
training image.

**More than one sessions folder is fine.** `--images-root` (and `IMAGES_ROOT`
in the config-block form) accepts several paths. A frame is found by trying
each one in turn, so a session need only exist under ONE of them — the ordinary
case when a weed capture campaign and a separately-recorded onion-only campaign
were never stored under a common parent directory.

### The easy way: edit a config block, no command line

`prepare_dataset.py`'s flags are all available in a config block, the same
convention the extraction and prelabel stages use. Works identically in cmd.exe,
PowerShell and VS Code's Run button — no backticks, no quoting rules.

> **cmd.exe note:** a trailing `` ` `` is **not** a line continuation there —
> that's PowerShell. In cmd it's `^`. A pasted PowerShell command arrives as a
> series of broken commands, which is exactly what it looks like.

```
1. Open seeweed3d/training/make_dataset.py in VS Code
2. Add one SOURCES entry per CVAT export - each names its OWN DATUMARO_ROOT
   and IMAGES_ROOT, since they don't have to share a parent folder
3. Set OUT_DIR
4. Leave LIST_FRAMES = True     → python seeweed3d/training/make_dataset.py
5. Read the table (grouped by session across every source), set INCLUDE_FRAMES
6. Set LIST_FRAMES = False      → python seeweed3d/training/make_dataset.py
```

```python
"SOURCES": [
    {"DATUMARO_ROOT": r"E:\CVAT_exports\vid2_20260108_122731",
     "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Weeds_3_good\sessions"},
    {"DATUMARO_ROOT": r"E:\CVAT_exports\onion1_20260115_090000",
     "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Onion_only\sessions"},   # different parent - fine
],
```

Then `seeweed3d/training/train_model.py` the same way — it trains and evaluates
in one run. Leave its `IMAGES_ROOT` as `""` to reuse exactly what
`make_dataset.py` already recorded in `seg_manifest.json`, so the roots are
typed out once, not kept in sync by hand across two files.

The command-line form below stays supported and takes exactly the same options.

### The command-line way

```powershell
conda activate sw-train

# Point at the PARENT folder - every export under it is found and merged
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root  E:/CVAT_exports `
    --images-root    E:/Dataset_Vidalia/sessions `
    --out            E:/Dataset_Vidalia/training/mixed_v1 `
    --holdout-test   vid3_20260108_103135 `
    --val-fraction 0.2 --test-fraction 0.2 --seed 1234
```

Or list exports explicitly:

```powershell
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root  E:/CVAT_exports/vid3_20260108_103135 `
                     E:/CVAT_exports/vid3_20260108_110444 `
    --images-root    E:/Dataset_Vidalia/sessions `
    --out            E:/Dataset_Vidalia/training/mixed_v1
```

Both `--datumaro-root` and `--images-root` take one or more paths (`nargs="+"`).
Pass several `--images-root` values when the sessions being merged are not all
under one parent:

```powershell
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root  E:/CVAT_exports/vid2_20260108_122731 `
                     E:/CVAT_exports/onion1_20260115_090000 `
    --images-root    E:/Dataset_Vidalia/Weeds_3_good/sessions `
                     E:/Dataset_Vidalia/Onion_only/sessions `
    --out            E:/Dataset_Vidalia/training/mixed_v1
```

### 5.1 ⚠️ If the CVAT task was pre-loaded with SAM prelabels

Then **every frame in the export has annotations**, including the ones you never
reached. Those frames are *not* empty — they carry SAM's masks with whatever
class it guessed. Nothing else in the pipeline filters them, and the
un-annotated-frame exclusion will not catch them.

Training on them is **worse than having no data**: the mask geometry is
basically correct and only the label is wrong, so the loss is confident and
consistent, and the model reliably learns the wrong class.

Select the frames you actually verified. **List first:**

```powershell
python -m seeweed3d.training.prepare_dataset `
    --datumaro-root E:/CVAT_exports --list-frames
```

That prints a numbered table — position, item id, instance count, class
breakdown. Positions are **1-based** over item-id order. Check the numbering
against what you remember from CVAT *before* selecting, rather than discovering
a one-off in a trained model.

Then build with the selection:

```powershell
    --include-frames "1-26,28-36,50-59"
```

| Syntax | Meaning |
|---|---|
| `1-26` | inclusive position range |
| `12` | a single position |
| `frame_0007` | a literal item id |
| `*_001*` | fnmatch glob over item ids |
| `<session>:1-26` | any of the above, scoped to one session |
| `<session>:*` | every frame of that session |
| `@keep.txt` | read the list from a file, one token per line, `#` comments |
| `--exclude-frames` | same syntax, applied **after** `--include-frames` |

> **Positions restart at 1 in each session.** That is what makes a range stable:
> merging a second CVAT export must not renumber the first one and quietly
> redirect a checked selection at different frames. Once more than one session
> is present, a bare position is ambiguous and is **refused**, not guessed —
> scope it with the session id.

Selection happens **before** validation, so the frames you discarded cannot fill
`annotations_needing_correction.json` with errors about annotations you are
throwing away. A position beyond the end of the export is an **error**, not a
silent no-op — quietly ignoring it would build the wrong dataset.

**Merging is safe across tasks:** each export is resolved through its *own*
`categories` block, so two CVAT tasks that happen to order their labels
differently still merge correctly — only the label **name** is carried forward.
The same frame appearing in two exports is reported as an error, because it
would be trained on twice and could land in two splits.

**Produces:**

| File | Purpose |
|---|---|
| `seg_manifest.json` | Stage A training (permissive backend) |
| `lep_manifest.json` | Stage B training — **omitted for a segmentation-only dataset** |
| `data.yaml` + `labels/` | Stage A training (Ultralytics backend only) |
| `splits/{train,val,test}_sessions.txt` | which session went where |
| `splits/splits_summary.json` | per-split session/class counts |
| `dataset_report.json` | full integrity report |
| **`annotations_needing_correction.json`** | **read this** |

**If it exits with errors**, open `annotations_needing_correction.json`, fix
those shapes in CVAT, re-export, and re-run. To inspect without blocking, add
`--allow-errors` — but do **not** train on the result.

Images are never copied. Manifests reference your existing files.

---

## 6. Train Stage A (segmentation)

**Default backend: torchvision Mask R-CNN (BSD-3-Clause).** No extra install, no
licence obligation.

```powershell
conda activate sw-train
python -m seeweed3d.training.train_seg_torchvision `
    --dataset     E:/Dataset_Vidalia/training/mixed_v1 `
    --images-root E:/Dataset_Vidalia/sessions `
    --out         E:/Dataset_Vidalia/runs/seg_v1 `
    --epochs 20 --batch 2 --device cuda `
    --track all --preview-every 5
```

Produces `runs/seg_v1/best.pt`, `last.pt`, `history.json`, `params.json`.

### Watching the run

`--track` enables **local** experiment tracking — nothing is uploaded and no
account is needed. `auto` (the default) uses whatever is installed and never
fails; `all` requires both and errors if either is missing.

```powershell
python -m pip install tensorboard mlflow psutil nvidia-ml-py

tensorboard --logdir E:/Dataset_Vidalia/runs/seg_v1/tb
# the URI, not the directory - training prints the exact line to use
mlflow ui --backend-store-uri sqlite:///E:/Dataset_Vidalia/runs/mlruns/mlflow.db
```

| Flag | Effect |
|---|---|
| `--preview-every N` | **GT-vs-prediction overlay panels** every N epochs on a fixed sample of val frames |
| `--eval-every N` | val `mAP@50`, `mAP@50:95` and `missed_onion_fraction` every N epochs (slow — a full pass over val) |

The preview panels are the highest-value signal on a small dataset. A loss curve
cannot tell you that every mask is one plant too large, or that the model calls
every onion a weed; ten seconds looking at the overlays will. Full rationale and
the tools that were rejected: [experiment tracking](experiment_tracking.md).

### Choosing a different backend

| Backend | Licence | Real-time? | Configurable losses? | When |
|---|---|---|---|---|
| **`maskrcnn`** (default) | **BSD-3** | no | no — internal to torchvision | **prototyping — start here** |
| `rfdetr` | **Apache-2.0** | **yes** | **yes** — Dice/CE/cls coefficients | the upgrade, ships commercially |
| `ultralytics` | **AGPL-3.0** | yes | partial | research only — see below |

`rtmdet` is **not** implemented. RTMDet-Ins is a credible Apache-2.0 alternative
on paper, but MMDetection's last release was v3.3.0 in **May 2024** and mmcv's
CUDA build is a recurring problem on Jetson, so it was not wired up. Don't pass
`backend="rtmdet"` expecting it to work.

> **Ultralytics is AGPL-3.0.** Commercial or proprietary use requires an
> Ultralytics Enterprise Licence — they state this applies even to internal
> company R&D unless the whole project is released under AGPL. For a commercial
> laser weeder that is a real cost, and unlike a code defect it cannot be fixed
> after distribution. It is not installed by default; `build_segmenter()` prints
> a loud warning if you select it.

---

## 6b. Train Stage A on RF-DETR-Seg (the advanced path)

Same dataset, same `seg_manifest.json`, same evaluation table — so the two
backends are directly comparable. Nothing here changes or invalidates a
Mask R-CNN run already in progress.

```powershell
conda activate sw-train
python -m pip install rfdetr
```

Then **edit the config block** at the top of
`seeweed3d/training/train_model_rfdetr.py` and run it:

```powershell
python seeweed3d/training/train_model_rfdetr.py
```

It converts `seg_manifest.json` into the Roboflow COCO layout RF-DETR expects
(`train/`, `valid/`, `test/`, each with `_annotations.coco.json`), hardlinking
images rather than copying where the filesystem allows, then trains.

### Pre-flight — read this before a long run

Between the COCO export and training, the dataset and your schedule are checked
for the things that let a run finish, print a plausible metric, and mean
nothing. The report goes to the console and to `RUN_DIR/preflight.json`.

```
  class                         train    valid     test
  --------------------------------------------------------
  onion_plant                     312       71       64
  grass_weed                      188       44       39
  other_weed                     1104      241      210
  cutleaf_evening_primrose         11        0        4
  FRAMES                          412       98       86

  [!] 'cutleaf_evening_primrose' has 11 training instance(s) (floor 20). It will
      report an AP near zero and drag the mean down for a reason unrelated to
      the model.
        -> Add "cutleaf_evening_primrose" to DROP_CLASSES in make_dataset.py
  [!] 'cutleaf_evening_primrose' has 11 training instance(s) but NONE in val, so
      nothing measures it. Early stopping and best-checkpoint selection are both
      blind to this class.
     412 train frames | effective batch 16 | 26 step(s)/epoch | 60 epochs = 1560 steps.
```

| Finding | Why it matters |
|---|---|
| class missing from **train** | *error*. The model can never predict it; downstream that reads as "looked and found nothing" |
| class missing from **val** | nothing measures it, so early stopping and best-checkpoint selection are blind to it — and if it is the crop, "best" may be the checkpoint that segments onions worst |
| class under ~20 instances | not learnable; reports AP≈0 and drags the mean for a reason unrelated to the model |
| val under ~10 frames | epoch-to-epoch noise exceeds the gap between checkpoints, so "best" is largely chance |
| no test split | every number comes from the set used to pick the checkpoint |
| patience ≥ epochs | early stopping can never fire, whatever the config says |
| effective batch ≥ dataset | an epoch is one optimiser step, so warmup/cosine/patience are all counting single steps |

An error-level finding stops the run. `SKIP_PREFLIGHT = True` (or
`--skip-preflight`) trains anyway — correct for a deliberate smoke test, which
trips several of these on purpose. The findings are recorded either way.

Standalone, without training:

```powershell
python -m seeweed3d.training.preflight --coco E:\Dataset_Vidalia\training1\rfdetr_v4\coco `
    --epochs 60 --batch 2 --grad-accum 8 --patience 25
```

### As the dataset grows

`EPOCHS` is a count of passes, so the same number means very different amounts
of work at different dataset sizes — 60 epochs over 60 frames is ~240 optimiser
steps at effective batch 16, and over 600 frames it is ~2,400. Pre-flight prints
the step count; use that, not the epoch count, when comparing runs.

| Setting | At ~60 frames | At ~600 frames | Why |
|---|---|---|---|
| `EPOCHS` | 60 | 30–40 | steps, not passes, are what the schedule spends; ten times the data does not need ten times the passes |
| `PATIENCE` | 25 | 8–12 | patience counts epochs, and an epoch is now ten times longer |
| `WORKERS` | 0 on Windows | 2–4 once a run is known to work | data loading starts to matter once an epoch is more than a handful of steps |
| `COCO_DIR` | inside `RUN_DIR` | one shared path | the conversion is per-run otherwise, and it copies or links every image |
| `LR` | 1e-4 | 1e-4 | leave it unless you change the effective batch; if you do, scale roughly linearly with `BATCH × GRAD_ACCUM` |

Keep `SEED` fixed across rounds. Changing it re-draws every split and
invalidates comparison with every earlier run.

### ⚠️ RESOLUTION is the setting that decides whether this is an upgrade

RF-DETR-Seg's own default is **432×432**. On a 2208×1242 ZED frame that is a
5× downscale — worse than the 1333 px Mask R-CNN default that already cost this
project most of its small-weed recall. Adopting the model at its default would
**regress the exact metric it looks like an upgrade for**, which is why
`train_seg_rfdetr.py` refuses to start at the default and requires an explicit
`--allow-default-resolution` to override.

Resolution must be a multiple of `patch_size × num_windows` for the variant —
**24** for medium/large, **12** for nano/small. The runner reads those values
out of the installed package rather than hard-coding them, and an invalid value
names the valid neighbours instead of failing deep inside training.

| Value | = | Note |
|---|---|---|
| `1008` | 24 × 42 | sensible first try (the shipped default) |
| `1248` | 24 × 52 | close to the 1333 the Mask R-CNN path used |
| `1344` | 24 × 56 | above it |

VRAM cost grows with the **square** of resolution. If you run out, raise
`GRAD_ACCUM` — do **not** lower `RESOLUTION`. `BATCH × GRAD_ACCUM` is the
effective batch and should stay near 16; the shipped `BATCH: 2, GRAD_ACCUM: 8`
trains at 1008 px with an effective batch of 16 on modest VRAM.

### What the config block exposes that Mask R-CNN cannot

| Setting | What it does |
|---|---|
| `MASK_CE_COEF` / `MASK_DICE_COEF` | weights the mask loss. torchvision computes Mask R-CNN's losses internally — changing them there means forking its ROI heads |
| `CLS_COEF` | classification loss weight (IA-BCE, designed for DETR set prediction) |
| `USE_EMA` | exponential moving average weights — usually a small free gain |
| `MULTI_SCALE` | trains across scales; helps small objects |
| `EARLY_STOPPING` / `PATIENCE` | built in |
| `GRAD_ACCUM` | effective batch 16 on small VRAM, instead of actually training at batch 2 |
| `LR_SCHEDULER` | **`cosine`.** rfdetr defaults to `step` with `lr_drop=100`, so on any run under 100 epochs the step never fires and the LR is **constant start to finish** |
| `WARMUP_EPOCHS` | the detection head is re-initialised for your class count, so step 0 is a random classifier at full LR beside a pretrained backbone |
| `PATIENCE` | **25, not rfdetr's 10.** The re-initialised head leaves whole classes at AP 0.000 for the first several epochs; patience 10 stopped a 60-epoch run at 23 with a class still improving |
| — | no anchors at all: DETR set prediction, so the anchor-size trap that cost the Mask R-CNN path its small weeds cannot occur in the same form |

### Tversky: breaking Dice's symmetry

Dice weights a false positive and a false negative **equally**. This system does
not — a missed weed survives to set seed, a spurious one costs one laser pulse,
and onion the model fails to mark is onion nothing protects.

```
TI = TP / (TP + ALPHA*FP + BETA*FN)          loss = (1 - TI)^GAMMA
```

| Setting | Effect |
|---|---|
| `TVERSKY_ALPHA` / `TVERSKY_BETA` | `0.5/0.5` **is exactly Dice** and patches nothing at all. `0.3/0.7` penalises missed pixels harder — the natural first try here. `0.2/0.8` is aggressive; watch precision and the burn fraction |
| `FOCAL_GAMMA` | `>1` concentrates gradient on the masks the model gets *wrong*, which here means small weeds. Noisier on 62 frames, so off by default; 1.33–2.0 is the usual range |

The equivalence at `0.5/0.5` is pinned by a test against rfdetr's own
`dice_loss`, so a Tversky run and a Dice run stay comparable. It works by
rebinding rfdetr's module-level `dice_loss_jit` rather than forking its
criterion; if a future version computes its mask loss differently, training
**stops and says so** rather than quietly using Dice while the config claims
otherwise.

Change one thing at a time. `0.3/0.7` with `GAMMA 1.0` first, and read
`small_weed_recall` and `weed_on_crop_fraction` — not overall mAP, which will
barely move.

**Read precision alongside recall, or the comparison lies.** Raising `BETA`
buys recall *with* precision — that is the whole mechanism, not a side effect.
The RF-DETR runs already emit far more low-score predictions per frame than the
Mask R-CNN run does at the same confidence, so run `eval_seg --sweep` on both
and compare `weed_recall` and `weed_precision` at a *matched* operating point.
If precision falls further than recall rises, the fix is a higher operating
confidence or a return to `0.5/0.5` — not a higher `BETA`.

Because the loss is patched into rfdetr at runtime, nothing rfdetr itself
writes records which one trained. `rfdetr_train_config.json` carries a
`mask_loss` block for exactly this reason — on Dice runs too, so a missing
block never has to be read as "probably the default":

```json
"mask_loss": {"kind": "tversky", "tversky_alpha": 0.3,
              "tversky_beta": 0.7, "focal_gamma": 1.0}
```

Check it before comparing two runs. Sidecars that differ only in `run` name
mean the loss did not actually change.

Leave the three loss coefficients at `None` for the first run — that uses the
model's own defaults (`mask_ce 5.0`, `mask_dice 5.0`, `cls 1.0`) and gives you a
baseline to move from. Raise `MASK_DICE_COEF` relative to `MASK_CE_COEF` when
the report shows masks roughly the right shape but with poor boundaries: Dice is
computed over the whole mask and is insensitive to how many background pixels
surround it, so it does not get swamped by a large empty frame the way per-pixel
cross-entropy can.

### Scoring it against the Mask R-CNN run

`TRACK: "auto"` routes RF-DETR's own tensorboard/mlflow output into the **same**
MLflow store as the Mask R-CNN runs, so both appear in one comparison table.
The evaluation path is identical apart from one flag:

```powershell
python -m seeweed3d.evaluation.eval_seg --backend rfdetr `
    --checkpoint E:/Dataset_Vidalia/training1/rfdetr_v1/checkpoint_best_total.pth `
    --dataset E:/Dataset_Vidalia/training1 --split val --device cuda --sweep

python -m seeweed3d.evaluation.report --backend rfdetr `
    --checkpoint E:/Dataset_Vidalia/training1/rfdetr_v1/checkpoint_best_total.pth `
    --dataset E:/Dataset_Vidalia/training1 --split val --device cuda
```

**`checkpoint_best_total.pth`, not `_ema`.** rfdetr keeps three files —
`_regular` (best live weights), `_ema` (best averaged weights) and `_total`,
copied from whichever actually won. Scoring `_ema` silently reports the loser
whenever the live weights were better; the runner prints the right path.

Loading it back needs the **variant, the resolution and the class list**, and
none of the three are inside the `.pth`. `train_seg_rfdetr.py` writes them to
`rfdetr_train_config.json` beside the checkpoint and the segmenter reads them
from there. Move a checkpoint without that file and loading fails with an
explanation rather than guessing — every available default is wrong: `Preview`
is a different architecture, 432 is the wrong resolution, and rfdetr's own
`class_names` is the COCO 80, which would point crop safety at "bicycle".

Compare the **recall-by-size** table and `weed_on_crop_fraction` at a matched
confidence, not overall mAP. Overall mAP is dominated by the large easy
instances; small-weed recall is the number this system is actually limited by.

RF-DETR prints its own per-class table each epoch, but that one is **box** AP
while `eval_seg` reports **mask** AP, and the two use different matching. Only
`eval_seg --backend rfdetr` compares like with like.

Sizing, the licence caveat on the XL variants (Nano..Large are Apache-2.0;
XL/2XL may fall under Roboflow's Platform Model License and are deliberately not
offered), and why **INT8 is the wrong move on Orin for a transformer**:
[edge model research](edge_model_research.md).

---

## 7. Train Stage B (LEP)

> **Skip this section if your dataset is `segmentation_only`.** With no LEP
> annotations there is nothing to fit, and `perception/lep.py` supplies growth
> points at inference time without any training. Come back here after an
> annotation round that places LEPs.

```powershell
python -m seeweed3d.training.train_lep `
    --manifest    E:/Dataset_Vidalia/training/mixed_v1/lep_manifest.json `
    --images-root E:/Dataset_Vidalia/sessions `
    --out         E:/Dataset_Vidalia/runs/lep_v1 `
    --input-mode rgb_mask_geom --device cuda --epochs 50
```

`--input-mode` selects the ablation:

| Mode | Inputs | Use |
|---|---|---|
| `rgb` | RGB only | baseline; works with no depth at all |
| `rgb_mask` | + owning instance mask | no depth stream available |
| `rgb_mask_geom` | + normalised height + depth validity | **default**, full model |

Run all three to report the ablation. Depth dropout (p=0.25) is applied during
training, so the full model still works when depth fails at runtime.

---

## 8. Evaluate

### 8.1 Stage A

Training prints loss, which is not a model-quality number — it is not comparable
across runs with different class counts, and it says nothing about whether small
weeds are found or onions are protected. Run this instead:

```powershell
python -m seeweed3d.evaluation.eval_seg `
    --checkpoint E:/Dataset_Vidalia/runs/seg_v1/best.pt `
    --dataset    E:/Dataset_Vidalia/training/mixed_v1 `
    --split val --device cuda --conf 0.5
```

`--images-root` is optional in both `eval_seg` and `report`: omit it and they use
what `make_dataset.py` recorded in `seg_manifest.json`, which is normally right.
Add `--backend rfdetr` when scoring an RF-DETR checkpoint — it must match the
backend that produced the file.

### Choosing the deployment confidence — `--sweep`

The single most misleading thing a metrics table can do is report one threshold.
`run4` moved small-weed recall from **0.28 at conf 0.5 to 0.73 at conf 0.25** on
unchanged weights: the number was measuring the operating point, not the model.

```powershell
python -m seeweed3d.evaluation.eval_seg --checkpoint <run>/best.pt `
    --dataset <dir> --split val --device cuda --sweep
```

Bare `--sweep` walks `0.15 … 0.7`; pass explicit values to narrow it. The model
runs **once per frame** either way — only the thresholding and matching repeat —
so the whole table costs almost nothing beyond the single-threshold run.

Read it as a decision, not a score. A missed weed survives to set seed; a false
weed costs one laser pulse. That asymmetry favours recall — as far as
**ONION BURNED** lets you go, and no further.

Three tables, deliberately not combined into one score:

| Table | What it answers |
|---|---|
| **detection** | `mAP@50`, `mAP@50:95` per class — score-ranked, 101-point interpolated, so comparable with published numbers |
| **operating point** | precision/recall/IoU at the confidence you would actually deploy at. mAP is threshold-free; a robot is not |
| **crop safety** | **missed onion pixels** and **onion called weed** — see below |

Crop safety is reported separately on purpose. A model can post excellent mAP and
still miss the one onion that matters, and averaging that into a headline number
hides it. A class with no ground truth in the split reports AP as `-` (undefined)
and is excluded from the mean, rather than counted as zero.

The crop-safety block reports **two different failures, and they are not equally
bad**:

| Number | Meaning | Consequence |
|---|---|---|
| `missed_onion_fraction` | crop the model did not label as crop | the laser has no reason to aim there — mostly **latent risk** |
| `weed_on_crop_fraction` | crop the model labelled as **weed** | the laser **fires into the onion** — actual **damage** |

`weed_on_crop_px` is a strict subset of `missed_onion_px`. Reporting only the
first makes a model that *ignores* onions look identical to one that *shoots*
them. Pixels claimed by both a crop and a weed prediction are **not** counted as
a burn — the pipeline's onion-conflict check suppresses that shot — so this
measures what the robot would actually do, not what the raw masks overlap.

Judge a run on `weed_on_crop_fraction` first. A high `missed_onion_fraction` with
a near-zero burn fraction means the masks are too tight, which is a mask-quality
problem; a non-zero burn fraction is a crop-loss problem and outranks every other
metric on the page.

Writes `metrics_<split>.json` next to the checkpoint.

### 8.1b Everything at once — `analyze_run.py`

Both trainers run this automatically when they finish. To redo it, or to
analyse an older run, edit the config block and:

```powershell
python seeweed3d/evaluation/analyze_run.py
```

Point `RUN_DIR` at **either** backend's run directory — it detects which from
what training wrote, reads that backend's history (`history.json` for
Mask R-CNN, lightning's `metrics.csv` for RF-DETR), and produces one set of
figures for both:

| `analysis/` | Answers |
|---|---|
| `training_curves.png` | is it still learning, or was the schedule cut short? |
| `per_class_ap.png` | which class is holding the headline number down (with `n_gt` on each bar) |
| `confidence_sweep.png` | **the deployment-threshold decision** |
| `crop_safety.png` | did it get *safer*, or just better at weeds? |
| `recall_by_size.png` | the failure this system is actually limited by |
| `report.html` | GT-vs-prediction panels and the missed-weed gallery |

Everything is also logged to MLflow, so both backends' runs sit in one
comparison table.

> **On hosted trackers (W&B, Comet).** They would not save any of this work.
> Every figure above has to be **computed** here either way — no tracker knows
> what a missed onion is, that recall at conf 0.5 and 0.25 are different
> questions, or how to draw a mask on a plant. A tracker only decides where the
> resulting PNG is filed, which is one call in `_track()`. What a hosted service
> *does* buy is a shared URL and a comparison UI, against uploading field
> imagery and a customer's row geometry to a vendor's cloud. That is a business
> decision, not a technical gap, which is why the default is local.

### 8.2 The visual report — look at this before changing anything

Numbers say how good the model is. They say nothing about **which** plants it
gets wrong, and on a small agricultural dataset that is the only question that
tells you what to annotate next.

```powershell
python -m seeweed3d.evaluation.report `
    --checkpoint  E:/Dataset_Vidalia/runs/seg_v1/best.pt `
    --dataset     E:/Dataset_Vidalia/training/subset45 `
    --split val --device cuda --conf 0.5
```

One self-contained HTML file (images embedded — open it, mail it, no sidecar
folder) plus the per-frame JSON behind it. Four sections:

| Section | What it answers |
|---|---|
| **Recall by instance size** | where the small-weed cliff actually is. `small_weed_recall = 0.28` is a fact; this table says whether it falls off below 250 px or below 2000 px, which are different problems |
| **Missed weeds, smallest first** | tight crops of each plant the model failed to find. Too few pixels? genuinely ambiguous? a labelling error? Three different fixes, and only the image distinguishes them |
| **Frames, worst first** | ground truth beside prediction, coloured by **outcome** — <span>green = matched, **red = missed**, magenta = false positive</span>, orange = crop. Sorted by miss count, so the informative frames are on the first screen |
| **Crop safety** | separate and never averaged in. No onion ground truth reports **UNMEASURED, not passing** |

Instances are coloured by outcome rather than by class deliberately: with class
colours a missed primrose and a correct primrose look identical, which is
exactly the comparison you are trying to make.

`--max-frames` / `--max-crops` bound the page size (defaults 24 / 48).

### 8.3 Everything else

Metrics live in `seeweed3d/evaluation/metrics.py` and are computed from stored
results, so you can re-evaluate without re-running inference.

```python
from seeweed3d.evaluation import metrics as m

m.segmentation_metrics(pred_masks, pred_classes, gt_masks, gt_classes)
m.onion_safety_metrics(pred_onion_union, gt_onion_union)   # crop safety
m.lep_errors(pred_uv, gt_uv, plant_radius_px)
m.compare_lep_methods(gt_uv, {"bbox_center": ..., "centroid": ...,
                              "dt_peak": ..., "lep_py": ..., "learned": ...})
m.uncertainty_calibration(sigmas, errors)
m.safety_metrics(targets, gt_lookup)
```

**Evaluate by whole session.** Watch **onion recall** and **missed onion pixels**
separately from IoU — missing crop tissue can destroy the crop, while a false
onion merely skips a weed, and IoU averages those two very different errors
together.

The learned LEP must be compared against all four baselines (bbox centre,
centroid, DT peak, and the hand-engineered `perception/lep.py`). That estimator
is preserved and is also the runtime fallback.

---

## 9. Run inference

### 9.1 On a folder of images — the quick look

Needs **no ground truth**, so it works on a held-out session, a new field, or a
different time of day. Edit the config block and run:

```powershell
python seeweed3d/perception/predict_images.py
```

| Setting | Note |
|---|---|
| `IMAGES` | a session folder (uses its `rgb/`), a folder of images, or one file |
| `STRIDE` | **use this.** Consecutive ZED frames are near-identical, so `LIMIT` alone gives you N pictures of the same plant |
| `CONF` | lower than the 0.5 the metrics table uses — to see failures you want what the model *nearly* said |
| `LABELS` | `class_score` / `class` / `none`. A dense frame carries 50 instances; at that point the text is the noise |
| `BACKEND` | `maskrcnn` or `rfdetr`, must match the checkpoint |
| `MODE` | `segmentation` (RGB only) or `full` (depth → LEP → 3D → safety decision) |

Writes `overlays/*.png` and `predictions.json`. **One colour per class** (see
`CLASS_COLOURS`), with `onion_plant` orange and drawn thickest, and a key
appended below the frame rather than painted over a corner of it. A weed
touching predicted onion gets an extra **white** outline and a `!CROP` tag
instead of being recoloured — the class and the hazard are separate facts, and
you want to know *which* weed is sitting on the crop.

> **What this cannot tell you.** Without labels there is no recall. An empty
> frame means *the model found nothing*, which is indistinguishable from *there
> was nothing to find*. Use this to see **how** the model fails and `eval_seg`
> to learn **how often**. On a held-out session the useful question is usually
> not the score but whether the masks still land on plants at all.

`MODE: "full"` runs the deployed pipeline and returns each weed as a
`candidate` or an `abstention` with reasons. It needs `<session>/rgb`,
`<session>/depth` and `<session>/meta/calibration.json`; a frame missing depth
degrades to segmentation for that frame rather than aborting the run. With
`LEP_CHECKPOINT` empty it uses the hand-engineered growth-point estimator, so it
works before Stage B is trained.

### 9.2 From Python — the full pipeline

```python
import cv2, torch
from seeweed3d.perception.segmenter import MaskRCNNSegmenter
from seeweed3d.perception.pipeline import InferencePipeline
from seeweed3d.training.config import PipelineConfig
from seeweed3d.training.lep_roinet import build_model
from seeweed3d.common.depth_utils import load_depth_mm, load_intrinsics

cfg = PipelineConfig()
seg = MaskRCNNSegmenter(r"E:/Dataset_Vidalia/runs/seg_v1/best.pt",
                        conf=0.25, device="cuda")

blob = torch.load(r"E:/Dataset_Vidalia/runs/lep_v1/best.pt", weights_only=False)
lep = build_model(cfg.model); lep.load_state_dict(blob["model"]); lep.to("cuda")

pipe = InferencePipeline(seg, cfg, lep_model=lep, torch_device="cuda")

session = r"E:/Dataset_Vidalia/sessions/vid3_20260108_103135"
bgr = cv2.imread(f"{session}/rgb/vid3_20260108_103135_000100.png")
depth, valid = load_depth_mm(f"{session}/depth/vid3_20260108_103135_000100.png")
K = load_intrinsics(session)

result = pipe.run(bgr, depth, valid, K,
                  session_id="vid3_20260108_103135", frame_id="000100")

for t in result.candidates:                    # SAFE to consider for treatment
    print(t.class_name, t.lep_uv, t.xyz_mm, "sigma", t.xyz_sigma_mm)

for t in result.abstentions:                   # refused, with reasons
    print("SKIP", t.class_name, t.rejection_reasons)

print(result.timings_ms, result.reason_counts())
```

**The pipeline produces candidates only.** `perception/safety.py` has no
actuator surface and never commands anything. Turning a candidate into a laser
command is a separate, deliberate act by your control layer — and it must
consult `t.safety_status == "candidate"`, never just the presence of `lep_uv`.

Without a trained LEP model, `lep_model=None` falls back to the hand-engineered
estimator, so the pipeline runs end-to-end before Stage B exists.

---

## 10. Export and benchmark (Jetson)

```powershell
# ONNX + PyTorch-vs-ONNX numerical parity check
python -m seeweed3d.deploy.export `
    --checkpoint E:/Dataset_Vidalia/runs/lep_v1/best.pt `
    --out        E:/Dataset_Vidalia/runs/lep_v1/export `
    --precision fp16
```

```bash
# ON THE JETSON ORIN - engines are architecture- and version-specific,
# so one built on your desktop GPU will not work here.
python -m seeweed3d.deploy.benchmark \
    --checkpoint runs/lep_v1/best.pt --device cuda --out jetson_bench.json
```

> A successful export is **not** a correct export. ONNX export succeeding only
> means the graph was traceable. Check `export_report.json` for
> `torch_vs_onnx.passed` before trusting the artefact, and verify FP16 accuracy
> before attempting INT8 — calibrating on an already-wrong engine bakes the
> error in.

Desktop numbers do not transfer to a Jetson. `benchmark.py` records the device
in every result and warns if it is not running on one.

---

## Quick reference

```powershell
# Environment A (dl): data pipeline + SAM 3
python seeweed3d/extraction/extract_sessions.py
python seeweed3d/extraction/curate_pool.py
python seeweed3d/annotation/prelabel_weeds_sam3.py
python seeweed3d/annotation/prelabel_onions_sam3.py
python seeweed3d/annotation/regen_cvat_labels.py

# --- CVAT: one task per session, group LEP with mask (G), export Datumaro 1.0 ---

# Environment B (sw-train): training + deployment
python -m seeweed3d.training.prepare_dataset --datumaro-root <exports> --images-root <sessions> --out <dir>
python -m seeweed3d.training.train_seg_torchvision --dataset <dir> --images-root <sessions> --out <run>
python seeweed3d/training/train_model_rfdetr.py                    # RF-DETR-Seg, config block
python -m seeweed3d.evaluation.eval_seg --checkpoint <run>/best.pt --dataset <dir> --split val
python -m seeweed3d.evaluation.report   --checkpoint <run>/best.pt --dataset <dir> --split val
python seeweed3d/perception/predict_images.py                       # predict on unlabelled frames
python -m seeweed3d.training.train_lep --manifest <dir>/lep_manifest.json --images-root <sessions> --out <run>
python -m seeweed3d.deploy.export --checkpoint <run>/best.pt --out <run>/export --precision fp16
python -m seeweed3d.deploy.benchmark --checkpoint <run>/best.pt --device cuda
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no Datumaro JSON found` | Exported as COCO instead of Datumaro 1.0, or didn't unzip. |
| `ungrouped_lep` errors | LEP points not grouped with their masks in CVAT — select both, press **G**. |
| `missing_lep` errors | Some weeds have LEPs and some don't. Finish the pass, or pass `--no-require-lep` to build a segmentation-only dataset. (A dataset with *zero* LEPs never raises this.) |
| `unknown label 'X'` | CVAT schema drifted from `common/ontology.py`. Run `regen_cvat_labels.py` and re-paste. |
| `duplicate_frame_across_exports` | The same frame is in two CVAT tasks. Delete one. |
| `compressed RLE ... needs pycocotools` | `python -m pip install pycocotools`, or re-export with polygons. |
| `unresolvable_session` | Task not named after the session id. Rename the CVAT task and re-export. |
| `training split has no rows` | Too few sessions for the requested fractions. Lower them or annotate more sessions. |
| SAM 3 `numpy` errors | numpy was upgraded past 2.0 in env A. `python -m pip install "numpy<2"`. |
| Phantom detections on bare ground | Vegetation prior false-positiving. Keep `RECOVER_MISSED_PLANTS=False`. |
