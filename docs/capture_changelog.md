# Capture changelog — v1 → v2

`seeweed3d/capture/zed_capture.py` replaces the v1 app (now `legacy/zed_app_v1.py`).

Every change below exists because something in the v1 recordings could not be
recovered afterwards. Nothing here is stylistic.

---

## Summary

| # | v1 problem | Consequence | v2 fix |
|---|---|---|---|
| 1 | Only `VIEW.LEFT` saved | Stereo could never be re-processed | **SVO2 archive** of the raw stereo stream |
| 2 | No `MEASURE.CONFIDENCE` | Only validity signal was the `0` sentinel | Confidence map recorded, **polarity probed from data** |
| 3 | No right image | No independent disparity check | `VIEW.RIGHT` optional stream (redundant once SVO is on) |
| 4 | Auto exposure, nothing logged | Appearance tracked the sun, not the plants | Exposure/gain/WB **lockable**, actual values logged per frame |
| 5 | Silent frame drops | Gaps invisible; video time ≠ real time | `dropped_frames.csv`; SVO written inside `grab()` and cannot drop |
| 6 | No pose, no vehicle clock | Nothing to sync depth against | Pose + IMU + **three clocks** per frame |
| 7 | `depth_vis_max_mm` in metadata | Read as a data scale; corrupts depth | Removed; encoding stated explicitly in `session.json` |
| 8 | Calibration = 8 numbers | Could not re-rectify or model raw stereo | Full dump: raw + rectified, both cameras, stereo rotation |
| 9 | Disk checked once at start | Sessions died mid-row | Live free-space and queue monitoring, clean abort |

---

## 1. SVO2 archive — the single most important change

```python
zed.enable_recording(sl.RecordingParameters(path, mode))
```

The SDK writes the **raw stereo stream** on every `grab()`. This retroactively
fixes problems 1 and 3, and means depth can be recomputed later with any
`DEPTH_MODE` — so a future comparison of `NEURAL` against `NEURAL_PLUS` on the
*same* frames becomes possible, which is exactly the kind of controlled
comparison the experiment plan calls for.

It also **cannot be dropped by our writer queue**, because it happens inside
`grab()` rather than downstream of it. When the MKV writer falls behind, the SVO
is still complete. That is why the drop warning says the archive survived.

Compression is attempted in order (`H265_LOSSLESS` → `H264_LOSSLESS` →
`LOSSLESS`) because NVENC support varies by machine. Whichever the SDK accepts
is recorded in `session.json`. If none work, the app says so loudly rather than
silently recording without an archive.

## 2. Confidence, with polarity determined empirically

`MEASURE.CONFIDENCE` is retrieved and stored as an 8-bit stream.

Documentation on which direction the confidence scale runs is easy to
misremember, and getting it backwards would silently invert every filtering
decision. So the app **measures it**: for the first 30 frames it compares mean
confidence where depth is invalid against where depth is valid, and writes the
result into `session.json`:

```json
"polarity": {
  "mean_confidence_where_depth_invalid": 95.0,
  "mean_confidence_where_depth_valid": 12.0,
  "lower_value_means_more_confident": true
}
```

Use that field. Do not assume a polarity from memory or from docs for a
different SDK version.

## 3. Exposure discipline

Auto exposure makes brightness a function of cloud cover, so the same weed looks
different at two ends of a row and the model learns the sun. `LOCK_EXPOSURE`
turns off AEC/AGC and fixes exposure and gain; `LOCK_WHITE_BALANCE` does the
same for colour temperature.

The *requested* values live in `session.json`; the **actual** per-frame values
live in `frames.csv`. Trust the latter — a lock can silently fail on some
firmware, and now you can tell.

Setting exposure is also the lever on motion blur. At 0.60 m/s a 1 ms exposure
smears about 0.6 mm and 5 ms about 3 mm. Against a PCK@2mm target, exposure is
not a comfort setting.

## 4. Complete frame accounting

v1 did `except queue.Full: self.dropped_enqueue += 1` — the count was printed at
the end and then lost. v2 records **which** frames were lost, in
`dropped_frames.csv`, and `frames.csv` now carries `video_frame_idx` explicitly
alongside `capture_frame_idx` so the mapping is stated rather than implied by
row order.

## 5. Clock alignment

Three clocks per frame:

| Column | Meaning |
|---|---|
| `image_timestamp_ns` | ZED IMAGE clock — use for intra-camera timing |
| `host_monotonic_ns` | immune to NTP steps — use for durations |
| `host_realtime_ns` | **join key** for ROS 2 bags and Amiga odometry |

ZED positional tracking is enabled for convenience, but it drifts badly over
repetitive crop rows with little parallax. **Record Amiga/ROS 2 odometry in
parallel and join on `host_realtime_ns`.** Treat the ZED pose as a sanity check,
not as vehicle ground truth.

## 6. Disk budget — read before enabling every stream

Measured estimate at HD2K (2208×1242) @ 15 fps, printed at startup:

| Streams enabled | Raw | Expected FFV1 write |
|---|---|---|
| left + right + depth + confidence | 370 MB/s | ~204 MB/s |
| left + depth + confidence | 247 MB/s | ~136 MB/s |
| left + depth | 214 MB/s | ~118 MB/s |
| SVO only | — | ~30–60 MB/s |

**This is the main practical constraint.** Sustained 136 MB/s needs a decent
NVMe; a USB drive will drop frames continuously. Recommended configuration:

- **Field default:** SVO + left MKV + depth MKV + confidence MKV, on NVMe.
- **Slow disk:** SVO only. Everything else is derivable from it offline.
- Leave `RECORD_RIGHT_MKV = False` whenever SVO is on — it is pure duplication.

If drops appear in the status bar, reduce streams rather than accepting gaps.

## 7. Metadata

`session_meta.txt` is replaced by `session.json`, which carries field context
(site, field, plot, row, growth stage, weather, mount height, vehicle speed),
stream inventory, exposure policy, clock definitions, integrity counters and
full calibration.

`depth_vis_max_mm` is **gone**. It was a GUI preview constant that sat in the
metadata file looking exactly like a data scale, and rescaling depth by it
corrupts every value. `session.json` now states the encoding directly:
1 unit = 1 mm, 0 = invalid.

`calibration.json` replaces `calibration_params.txt` with both rectified and raw
parameters for both cameras, full distortion vectors, stereo translation *and*
rotation, serial number and firmware — plus a derived `mm_per_px_at_1000mm`,
which is the number to check before trusting any sub-millimetre LEP target.

---

## Backward compatibility

`extraction/extract_sessions.py` reads **both** formats and tags each session
`capture_format: v1 | v2` in `registry.csv`. Existing March 2025 data needs no
migration and no re-processing. v1 sessions get a warning listing what is
missing, so a later analysis cannot silently assume confidence data exists.

The left MKV is still written as `RGB_video.mkv` so any existing script keeps
working unchanged.

---

## Before the next field session

1. Set `BENCH_CHECK = True` and run. It opens the camera, applies settings,
   lists available SVO modes, grabs one frame and reports left/right shapes,
   depth valid fraction and confidence range. Fix anything that fails **here**,
   not in a field.
2. Set `BENCH_CHECK = False`. Fill in `SITE`, `FIELD`, `PLOT`, `ROW`,
   `GROWTH_STAGE`, `WEATHER`, `MOUNT_HEIGHT_MM`, `VEHICLE_SPEED_MPS`.
3. Set exposure with the rig at working height under the day's lighting, then
   lock it. Re-check after any large lighting change.
4. Record 20 s, stop, and confirm: `recording.svo2` exists, `dropped_frames.csv`
   has only a header, and `frames.csv` shows constant exposure/gain.
5. Watch the status bar during the run. Rising queue depth means the disk is
   losing; stop and drop a stream.

## Known limitations of v2

- **Untested against hardware.** The data path is unit-tested against a stubbed
  SDK — writer, CSV schema, drop accounting, calibration dump and the confidence
  probe all verified. The SDK calls themselves have not run against a camera.
  Do the bench check.
- SDK version differences are handled defensively (`get_camera_settings` returns
  a tuple on 4.x and a bare int on 3.x), but `WHITEBALANCE_AUTO` and some SVO
  modes may be absent on older SDKs. Missing enums are skipped with a warning
  rather than crashing.
- Vehicle pose still has to come from the Amiga. The app logs the join key; it
  cannot log what it is not given.
