# SeeWeed3D — ZED dataset extraction & annotation prep

Edit only the `CONFIG` block at the top of each script.

```
python seeweed3d/capture/zed_capture.py          # field capture (v2 format)
python seeweed3d/extraction/extract_sessions.py  # recordings -> indexed, QC'd pool
python seeweed3d/extraction/select_batches.py    # pool -> CVAT-ready batches
python seeweed3d/annotation/prelabel_onions_sam3.py  # onion-only SAM 3 prelabels
```

## Capture formats

The extractor reads both and tags each session in `registry.csv`:

| | v1 (legacy, March 2025) | v2 (`capture/zed_capture.py`) |
|---|---|---|
| Left rectified | yes | yes |
| Right image | no | via SVO (+ optional MKV) |
| Depth | yes (mm, 0 = invalid) | same |
| Confidence | **no** | yes, polarity probed from data |
| SVO archive | **no** | yes — depth is recomputable |
| Exposure / gain / WB | **not logged** | per frame, lockable |
| Pose / IMU | **no** | per frame + ROS 2 join key |
| Dropped frames | counted then discarded | listed in `dropped_frames.csv` |

Existing v1 data needs no migration. See `docs/capture_changelog.md`.

---

## 1. Depth semantics — read this before touching a depth file

Verified directly against the v1 capture app (`legacy/zed_app_v1.py`), not assumed:

| Property | Value | Source |
|---|---|---|
| Unit | **1 PNG count = 1 mm** | `np.clip(depth_mm, 0, 65535).astype(np.uint16)` |
| Invalid | **0** | `nan_to_num(depth, nan=0, posinf=0, neginf=0)` |
| Measure | `MEASURE.DEPTH`, `NEURAL` mode | `retrieve_measure(...)` |
| Image | `VIEW.LEFT`, **rectified** | `retrieve_image(..., sl.VIEW.LEFT)` |
| Intrinsics | rectified, zero distortion | `camera_configuration.calibration_parameters` |

**`depth_vis_max_mm: 3000` in `session_meta.txt` is a GUI preview constant.**
It never touches stored data. Rescaling by it corrupts every depth value.
Use `depth_utils.load_depth_mm()` and this cannot happen.

`0` is not a distance. It merges *no stereo match*, *too near* and *too far*
into one sentinel — you can tell that depth is missing, but not why.

---

## 2. Why frames are selected by index, not by time

The capture thread drops frames when the writer queue fills
(`except queue.Full: self.dropped_enqueue += 1`), and the encoder is told the
stream is a constant 15 fps. So **elapsed video time ≠ elapsed real time**
wherever drops occurred.

The old extractor used `-vf fps=1/0.4348` on RGB and depth *independently*.
Two consequences: sampling drifts against real time, and RGB frame *i* can be
paired with a different depth frame. Since both streams are written from the
same queue item, they are aligned **by index** — so index-based selection makes
pairing exact. Verified bit-exact on a synthetic session encoded with the
capture app's own ffmpeg settings.

`frames.csv`, where present, is written one row per *encoded* frame in write
order, so **video frame `i` ↔ CSV row `i`**, which recovers true capture indices
and nanosecond timestamps despite the drops.

---

## 3. Output layout

```
dataset/
  registry.csv                      one row per session — start split planning here
  sessions/<session_id>/
    rgb/    <session_id>_000123.png    8-bit BGR, lossless
    depth/  <session_id>_000123.png    16-bit, millimetres, 0 = invalid
    meta/session.json                  provenance, calibration, depth encoding, warnings
    meta/calibration.json
    meta/frames_index.csv              QC metrics for EVERY decoded frame
    meta/pool.csv                      the subset written to disk
    meta/orig_*                        untouched copies of the source sidecar files
  batches/<batch>/
    images/                            RGB only — CVAT cannot read 16-bit PNG
    manifest.csv                       full provenance per frame
    cvat_labels.json                   paste into CVAT's Raw label editor
    batch.json
  reports/pool_stats.csv
```

`session_id` = `<trip>_<YYYYMMDD>_<HHMMSS>`, taken from the folder name. Filenames
embed it, so IDs are stable and traceable with no global counter file to lose or
corrupt. The number in the filename is the **video frame index** — the join key
back to `frames_index.csv` and to the depth file.

The MKVs remain the archive. Everything under `dataset/` is regenerable, so you
can re-run with a different stride without re-collecting anything.

---

## 4. Handling all three folder variants

One script, no forks. Discovery is a case-insensitive glob
(`RGB_PATTERNS` / `DEPTH_PATTERNS`), so `RGB_video.mkv`, `rgb.mkv` and other
spellings all match. Everything else is read from the **video header** via
ffprobe rather than trusting sidecar files, so a missing `session_meta.txt`
costs nothing. Missing files are recorded as warnings in `session.json`, not
crashes. A folder with no date in its name falls back to its mtime — check
`registry.csv` and rename if you want a meaningful ID.

---

## 5. QC metrics

Computed for every frame, so selection is evidence-based rather than a fixed
time interval:

| Metric | Use |
|---|---|
| `sharpness` | Laplacian variance — motion blur at speed |
| `clip_frac` / `dark_frac` | glare from wet leaves; deep shadow |
| `veg_frac` | excess-green coverage — drops bare-soil frames |
| **`depth_valid_frac_veg`** | **fraction of valid depth *on vegetation*** |
| `depth_median_mm`, `depth_inrange_frac` | working-distance sanity |
| `conf_mean`, `conf_mean_veg` | v2 only — SDK confidence, incl. on vegetation |
| `phash` | 64-bit near-duplicate detection |

`depth_valid_frac_veg` is the one to watch. Whole-frame validity is dominated by
easy flat soil and looks fine while thin seedling leaves have no depth at all.
If the median is low, bed-plane fallback becomes the primary 3D path rather than
the backup — which is worth knowing before annotating hundreds of frames.

---

## 6. Selection and split discipline

Stage 2 gates on quality, removes near-duplicates by pHash Hamming distance,
then greedily picks visually distinct frames with a per-session cap.

`stress_fraction` deliberately reserves part of each batch for **hard** frames —
glare, shadow, blur, depth holes. A gold set drawn only from clean frames will
overstate how well SAM 3 prelabelling works, and you will find out at exactly
the wrong moment.

`HOLDOUT_SESSIONS` removes whole sessions from every training batch. Adjacent
video frames are near-identical, so a random frame split leaks test data into
training and inflates every number you report. Hold out whole sessions —
ideally a different date and field. Verified: no frame appears in two batches,
and no holdout session reaches a training batch.

Start with `b01_gold` (~80 frames, manual, no zero-shot help) to measure
prelabel quality and annotation time honestly, then `b02_seed`.

---

## 6b. Curating the pool — dropping redundant and bad frames

Two things go wrong on a real capture run: you move slowly (often at the start),
so consecutive pooled frames show almost the same ground; and some frames are
simply bad and you want them gone after seeing the previews.

```bash
python seeweed3d/extraction/curate_pool.py     # DRY_RUN = True by default
```

### Never delete or rename the image files

The filename `<session_id>_<video_frame_idx:06d>.png` **is the join key**. The
same name appears in `rgb/`, `depth/`, `right/` and `conf/`, and the number is
the frame's index in the source video.

- Deleting the `rgb` file but not the `depth` file silently desynchronises the
  pair.
- Renaming files to close the gaps destroys the link back to the video and to
  `meta/frames_index.csv`, and breaks the invariant in §2 that every stream is
  aligned **by index**.

**Gaps in the numbering are correct.** They mean "this frame exists in the video
but is not in the pool" — that is exactly what a stride, a QC gate, or a curation
drop is supposed to produce.

So curation is recorded in `meta/pool.csv` — the manifest every later stage
already reads — as two columns:

| Column | Meaning |
|---|---|
| `dropped` | `0` = use it, `1` = skip it |
| `drop_reason` | `redundant` / `manual`, so a decision is auditable later |

`select_batches.py` and both prelabelers skip dropped rows automatically. A
pool written before curation existed has no `dropped` column, which reads as
keep-everything, so old sessions are unaffected.

Because nothing is deleted, every drop is reversible: `RESTORE_ALL = True`
clears the flags and the frames come straight back.

### How redundant frames are found

Overlap is a question about how far the **camera travelled**, so that is what
gets measured, preferring real physical evidence over a proxy:

1. **Pose** (best) — v2 captures record `tx_mm/ty_mm/tz_mm` from ZED positional
   tracking, so the distance between two frames is literal camera travel in
   millimetres. Set `MIN_TRAVEL_MM`. A pose recorded while tracking was *lost*
   is refused rather than trusted — a bogus jump is worse than no pose at all.
2. **Image shift** (fallback) — phase correlation gives the dominant
   translation in pixels, as a fraction of frame width so the threshold is
   resolution independent. Set `MIN_SHIFT_FRAC`.

Travel accumulates **from the last kept frame, not the previous frame**. That
distinction is the whole point: crawling at 5 mm/frame, every consecutive pair
looks "different enough" pairwise, so a pairwise rule with a 20 mm threshold
drops *nothing*. Accumulating from the last kept frame keeps one frame per
`MIN_TRAVEL_MM` of ground actually covered — which is the property you want.

### Choosing the threshold

The before → after count alone cannot tell you whether a threshold is right, so
a dry run also prints a **sweep** — every candidate threshold evaluated against
your own footage. The measurement is done once, so the extra rows are nearly
free:

```
        threshold sweep (current = 0.15):
          value    kept   dropped   overlap between kept frames
          0.05      120         0      95% of each frame re-annotated
          0.1        60        60      90% of each frame re-annotated
          0.15       42        78      85% of each frame re-annotated <-- current
          0.25       30        90      75% of each frame re-annotated
          0.4        20       100      60% of each frame re-annotated
          0.6        14       106      40% of each frame re-annotated
          1.0         9       111       0% of each frame re-annotated
```

**Overlap is the column to choose on.** `MIN_SHIFT_FRAC` is how far the camera
travels between kept frames as a fraction of frame width, so `1 - value` is how
much of each kept frame shows ground you already annotated in the previous one.
At `0.15`, 85% of every frame is a re-annotation of the frame before it.

There is no universally correct value — it is a trade-off you own:

| | Low value (0.05–0.15) | High value (0.5–1.0) |
|---|---|---|
| Frames | many | few |
| Annotation cost | high — the same plant labelled repeatedly | low |
| Ground covered | same | same |
| Useful when | you want many views of each plant | you want maximum distinct ground per labelling hour |

Since the point of this stage is to build a **manually corrected** training set,
annotation cost usually dominates, which argues for a higher value than the
`0.15` default. Start around `0.4–0.6` and check the previews.

A flat drop histogram is **not** a sign the threshold is wrong here — it just
means you moved at a steady speed, so the threshold alone sets the sampling
rate. Bunched drops mean a genuinely slow patch was thinned.

`MIN_TRAVEL_MM` (pose mode) must be **calibrated to your mount height**: it
should be a meaningful fraction of the ground footprint of one frame. Its sweep
is `SWEEP_TRAVEL_MM`.

### Dropping specific bad frames

`MANUAL_DROPS` takes a per-session list and accepts whichever form you have to
hand — a bare index, an inclusive range, or any filename whose stem ends in the
index. Preview names (`.jpg`) work as well as the source `.png`, because
previews are what you actually look at when deciding a frame is bad:

```python
"MANUAL_DROPS": {
    "weed1_20260108_143022": ["0-250", "1187", "weed1_20260108_143022_001900.jpg"],
},
```

---

## 7. Suggested run order

1. Set the two prominent blocks at the top of `extraction/extract_sessions.py` —
   `INPUT_ROOTS` (one entry per visit; paths are searched recursively) and
   `OUTPUT_ROOT`. `ffmpeg`/`ffprobe` are found on PATH automatically; only set
   `CONFIG["FFMPEG"]` if they are not on PATH. Run stage 1 with `DRY_RUN=True`.
2. Confirm session IDs and warnings, then run for real.
3. Read `registry.csv` — pick holdout sessions, sanity-check `median_depth_valid_frac_veg`.
4. *(optional)* Run `extraction/curate_pool.py` to thin slow/overlapping
   segments and drop bad frames — see §6b. Dry-run it first.
5. Run stage 2 with only `b01_gold` enabled; the gate report tells you which
   threshold is rejecting frames and what value the data suggests.
6. Create a CVAT task from `batches/b01_gold/images/`, paste `cvat_labels.json`.
7. Annotate the gold set manually. Keep CVAT/Datumaro as the master format;
   generate COCO/YOLO exports from it, never the reverse.
8. Double-label 10–15% of LEPs with a second annotator before trusting any
   pixel-level LEP target.

---

## 8. Gaps in the v1 data (fixed in v2 capture, not retroactively)

These constrain what the March 2025 recordings can support. All are addressed in
`capture/zed_capture.py` going forward, but none can be recovered from existing files:

- **No right image.** Only `VIEW.LEFT` was saved, so stereo cannot be
  re-processed with different matching settings, and the plan's "retain raw left
  and right" requirement cannot be met retroactively.
- **No confidence map.** `MEASURE.CONFIDENCE` was never retrieved. The only
  validity signal is the 0 sentinel — no left–right consistency or confidence
  filtering is possible on this data.
- **NEURAL depth is a learned filler.** It interpolates smoothly across thin
  structures, so depth near a leaf edge can be confidently wrong rather than
  absent. Treat ZED depth as a working signal, not as 3D ground truth for LEP
  error measurement.
- **No pose, speed, exposure, gain or white balance logged.** Timestamps exist
  (when `frames.csv` does) but there is nothing to sync them against.
- **No SVO retained.** Re-extraction with a different depth mode is impossible.

All five are fixed in v2, primarily by recording an SVO archive: it stores the
raw stereo stream, so right image and re-computable depth both come back, and it
is written inside `grab()` where the writer queue cannot drop it.

The same applies to the Bumblebee when it arrives — record the raw pair and the
per-frame settings, and derived products stay regenerable.
