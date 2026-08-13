# Extraction fidelity — what is guaranteed, what is measured, and what is not

Reference account of how SeeWeed3D turns raw ZED recordings into an annotated
dataset, and of the fidelity of that conversion. Written to be quotable in a
methods section: every claim here is either enforced by code, asserted by a
test, or reported as a measurement with its conditions.

Operational instructions live in [`RUNBOOK.md`](RUNBOOK.md) §1. This document
is about *why the output can be trusted*, not how to run it.

---

## 1. The headline claim

**For current (v2, MKV/FFV1) captures, extraction is bit-exact.** The PNGs in
`sessions/<id>/rgb/` and `sessions/<id>/depth/` are byte-for-byte identical to
the frames the camera produced. Nothing is resized, re-encoded, colour-shifted,
resampled in time, or rounded. This is asserted by tests, not assumed.

For legacy (v1, AVI/XVID) captures, extraction is **as faithful as the source
allows** — the loss happened in the field at record time and is not
recoverable. §6 covers what is and is not salvageable there.

---

## 2. Two capture generations

The pipeline reads both and tags every session `capture_format` in
`registry.csv`, so no later analysis can silently mix them.

| | **v2 — current** | **v1 — legacy** |
|---|---|---|
| Container / codec | MKV / **FFV1 (lossless)** | AVI / **XVID (lossy)** |
| RGB | `bgr24`, lossless | 8-bit YUV 4:2:0, lossy |
| Depth | `gray16le`, **true millimetres** | 8-bit *preview*, not measurement |
| Confidence map | recorded | absent |
| Right image | recorded | absent |
| SVO2 archive | recorded | absent |
| Per-frame log | `frames.csv` — exposure, gain, WB, pose, IMU, 3 clocks | absent |
| Calibration | `calibration.json` — raw + rectified, both cameras, distortion, stereo rotation | `calibration_params.txt` — rectified only |
| Extraction fidelity | **bit-exact** | best-effort; source already degraded |

**v1 is not a supported recording format going forward.** It is supported as an
*input* because real field data exists in it. Everything in §6 is archaeology,
not a recommended path.

The v1→v2 rationale is in [`capture_changelog.md`](capture_changelog.md); each
change there exists because something in v1 recordings could not be recovered
afterwards.

---

## 3. Fidelity guarantees

### 3.1 No lossy re-encoding anywhere

Frames are decoded to raw arrays and written as **PNG**, which is lossless at
every compression level. `PNG_COMPRESSION` trades file size against write speed
and *cannot* affect pixel values — pinned by a test that round-trips random
data at the shipped setting.

### 3.2 No resampling

- **No spatial resizing.** Frames are written at source resolution. The one
  place a resize appears is `frame_metrics()`, which computes QC statistics on
  a downscaled *copy*; that copy is never written and never reaches training.
- **No temporal resampling.** Frames are selected by **frame index**, never by
  `-vf fps=`. Decoding uses `-vsync 0`, which guarantees exactly one output
  frame per coded frame — no duplication, no dropping.

This matters because capture can drop frames while the encoder assumes constant
fps, so **video time is not real time**. Selecting by time would silently
misalign streams; selecting by index cannot.

### 3.3 Stream alignment is by index, by construction

Every stream (`rgb`, `depth`, `right`, `conf`) is written from the same capture
queue item and shares the video frame index. The filename
`<session_id>_<video_frame_idx:06d>.png` **is** the join key, and the same name
appears in every stream's directory.

Gaps in the numbering are correct and expected — they mean "this frame exists
in the video but is not in the pool". Curation therefore never deletes or
renames a file; it records `dropped` / `drop_reason` columns in `pool.csv`
instead, because renaming to close gaps would destroy the link back to the
video and to `frames_index.csv`.

### 3.4 Depth is measurement, and is protected as such

Depth is stored as **16-bit PNG, 1 count = 1 mm, 0 = invalid**. It is never
rescaled. Two specific hazards are guarded:

- **`depth_vis_max_mm` is a GUI preview constant**, not a data scale. Rescaling
  by it corrupts every value. It was removed in v2 and is explicitly ignored
  for v1. `common/depth_utils.load_depth_mm()` is the only sanctioned reader.
- **A depth stream that is not genuinely 16-bit is refused** rather than
  decoded (`REQUIRE_16BIT_DEPTH`). ffmpeg will happily produce `gray16le` from
  an 8-bit source by scaling up values it invented; the result is a valid PNG
  full of plausible millimetres that are fiction, and nothing downstream can
  detect it. See §6.2.

### 3.5 Colour conversion is correct, and was verified rather than assumed

Only relevant to v1 (lossy YUV) sources — FFV1 stores BGR directly and needs no
conversion. Measured on flat colour patches, where chroma subsampling and codec
loss are negligible so any residual is the conversion itself:

| Setting | Result | Verdict |
|---|---|---|
| Default (BT.601, limited-range in) | colour-neutral offset, uniform across channels | **correct — kept** |
| Forced BT.709 | −21 on green, +7 on red | matrix mismatch — rejected |
| Forced full-range input | 3.6× worse | streams really are limited-range — rejected |
| `full_chroma_int` | no measurable change | governs *resizing*; extraction never resizes — rejected |
| **`accurate_rnd`** | bias −2.0 → −0.8 per channel | **kept** |

v1 AVIs carry **no colour metadata at all** (`color_space`, `color_range` and
`color_primaries` all absent), so both encoder and decoder fall back to their
own defaults — and those defaults agree. This is why no override is applied.

`accurate_rnd` makes swscale round rather than truncate. The default truncation
leaves a systematic negative bias. That is not cosmetic in this pipeline: the
vegetation prior thresholds Excess-Green and compares green against blue, so a
channel-dependent error moves real segmentation decisions.

Measured on a synthetic field frame, against the uncompressed original:

| | default | `accurate_rnd` |
|---|---|---|
| Mean absolute pixel error | 3.30 | **2.78** |
| `vegetation_mask()` disagreement | 0.607% of pixels | **0.474%** |

Performance-neutral (an initial timing suggesting a speedup proved to be a
cold-cache artifact). Verified **not** to disturb bit-exactness on the lossless
paths.

---

## 4. What the pipeline refuses to do

Guards exist because each one corresponds to a failure that is *silent* — the
run completes, the output looks plausible, and the error surfaces much later.

| Guard | The silent failure it prevents |
|---|---|
| Non-16-bit depth refused | Fabricated millimetres that pass every QC check |
| Per-frame-normalised depth detected | An 8-bit preview whose scale changes every frame; no constant can recover it |
| Approximate depth written to `depth_approx/`, never `depth/` | An approximation being read as measurement — identical uint16 PNGs otherwise |
| Unsupported container reported, not skipped | Whole sessions vanishing with no output while the run reports success |
| Recordings-vs-dataset path confusion named explicitly | Being told to re-run an extraction that already succeeded |
| Over-curation warning | A stale threshold deleting 96% of a pool while reporting success |
| Session-level split validation | A build with no validation set, or one that starves a class |
| Empty-crop-mask distinction (`None` vs empty) | A model with no crop class clearing every shot in a crop row |

---

## 5. Provenance and reproducibility

### `meta/session.json` — per session

Records source paths and ffprobe results for every stream; **decode settings**
(`sws_flags`, `png_compression`) since these change pixel values on a lossy
source and are otherwise invisible afterwards; full calibration; `depth_kind`
(`metric` / `approximate` / `none`); frame counts, stride, `timestamp_source`,
frames dropped at capture; pool QC summary; and a `warnings` list naming
everything missing or degraded.

### `meta/frames_index.csv` — per frame

`video_frame_idx`, `capture_frame_idx`, `timestamp_ns`, plus (v2) exposure,
gain, white balance, pose. QC per frame: `sharpness` (variance of Laplacian),
`mean_luma`, `clip_frac`, `dark_frac`, `veg_frac`, `phash`, and where depth
exists `depth_valid_frac`, `depth_inrange_frac`, `depth_valid_frac_veg`,
`depth_median_mm`.

`depth_valid_frac_veg` — valid depth **on vegetation** — is the one that
matters for growth-point work: whole-frame depth validity can look fine while
the plants themselves have no depth at all.

### `registry.csv` — per campaign

One row per session: format, trip, site, field, scene hint, decoded/pool
counts, timestamp source, stream inventory, `depth_kind`, median depth validity
on vegetation, median sharpness, warning count, source folder.

> **`median_sharpness` is not comparable across capture formats.** Lossy XVID
> blocking artifacts inflate Laplacian variance, so v1 sessions score far
> higher than v2 sessions of the same scene. Compare within a format, never
> across. The same caution applies to any absolute-threshold gate built on it.

---

## 6. Legacy AVI data — what was salvaged

Kept because real field data exists in this format, and documented because the
answers were non-obvious and cost real investigation.

### 6.1 RGB — usable, permanently degraded

XVID is lossy and the loss is baked in. The frames remain fully usable for
segmentation: the dataset is built from them, and the only measurable
consequence is compression artifacts that inflate sharpness metrics (§5) and
slightly degrade the chroma signal the vegetation prior reads.

### 6.2 Depth — not recoverable, and provably so

The v1 capture wrote depth as:

```python
max_val = depth_data.max()                        # recomputed EVERY frame
depth_norm = (depth_data / max_val * 255).astype(np.uint8)
```

The scale is a property of the **individual frame** — one distant outlier pixel
sets it for every other pixel — so two frames of identical geometry get
different codes whenever their farthest point differs. **No constant relates
code to millimetres**, and `depth_vis_max_mm` never entered into it.

This was established in three stages, all of which are tooling that survives:

1. `REQUIRE_16BIT_DEPTH` refused the stream, correctly, but could only say
   *"not metric depth"*, not *what it was*.
2. `validation/inspect_depth_video.py` distinguishes a recoverable preview
   (grayscale or colour-mapped, fixed scale) from an unrecoverable one, and now
   detects per-frame normalisation directly from the file — every frame pinned
   near the top of the range, maxima agreeing, peak an outlier rather than a
   saturated plateau.
3. `validation/calibrate_preview_scale.py` measures a preview's scale against a
   session with real depth, fitting at seven percentiles rather than one
   because a single-point ratio always returns a number whether or not the
   relationship is linear. It reported drift — correctly — before the capture
   source confirmed why.

**Consequence:** the affected sessions are RGB-only. That costs nothing for
segmentation, dataset building or Stage A training, none of which read depth.
It costs everything for 3D localisation and the LEP canopy-height channel on
those sessions.

`RECOVER_8BIT_DEPTH` exists for genuine *fixed-scale* previews and writes to
`depth_approx/` with its terms recorded. It does **not** apply to per-frame
normalised data, and is off by default.

### 6.3 Mixed containers in one campaign

A single visit had AVI in its early sessions and MKV in its later ones.
Discovery matches on the file **stem** with the extension checked separately
against `VIDEO_SUFFIXES`, so both are found by one rule. Anything in an
unlisted container is reported with its extension named.

---

## 7. What is *not* claimed

Stated explicitly so the guarantees above are not read as broader than they
are.

- **v1 RGB is not lossless.** No extraction setting recovers XVID's loss.
- **v1 depth is gone**, not merely awkward.
- **Timestamps on v1 sessions are synthesised** from nominal fps — no
  `frames.csv` exists — and are approximate. Not usable for pose sync;
  `timestamp_source` records this per session.
- **Sharpness is not comparable across capture formats** (§5).
- **Bit-exactness is verified for FFV1 `bgr24` and `gray16le`** specifically,
  which is what the v2 capture writes. It is not a claim about arbitrary codecs.
- **Colour-conversion measurements are on synthetic frames** with known ground
  truth, since real footage has no uncompressed reference. The flat-patch
  method isolates conversion error from codec loss, but the absolute numbers
  are conditioned on that synthetic content.

---

## 8. Recommended practice for future captures

1. **Record with the FFV1/MKV path** (or `capture/zed_capture.py`, which adds
   the SVO2 archive, confidence map, right image and per-frame log). Every
   fidelity guarantee in §3 that says *bit-exact* applies only to this path.
2. **Never write depth as an 8-bit preview.** One line at the capture end —
   `gray16le` rather than a normalised `uint8` — is the difference between
   millimetres and an unrecoverable picture of millimetres.
3. **Keep the SVO2 archive.** It is the only route back to raw stereo if
   anything downstream needs re-deriving.
4. **Verify a new capture before a full field day**: run
   `validation/inspect_depth_video.py` on one depth file and confirm
   `true_16bit`. Ten seconds, and it is the check whose absence cost this
   project a visit's depth.
