import os
import json
import csv
import math
import subprocess
import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
SESSION_DIR = r"C:\ZED\DataCollection\Vidalia3\Session_20260108_103135"

# How many frames to scan from the start for global stats (keeps it fast)
SCAN_FIRST_N_FRAMES = 300

# How many sample frames to export as images (evenly spaced across the video)
EXPORT_SAMPLE_FRAMES = 12

# Fixed visualization range for depth (mm)
DEPTH_VIS_MAX_MM = 3000


# =========================
# Helpers
# =========================
def run_cmd(cmd):
    """Run a command and return (returncode, stdout, stderr)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def ffprobe_stream_info(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,pix_fmt,nb_frames,r_frame_rate,avg_frame_rate",
        "-of", "json",
        str(video_path)
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}\n{err}")
    info = json.loads(out)
    if "streams" not in info or len(info["streams"]) == 0:
        raise RuntimeError(f"No video stream found in {video_path}")
    return info["streams"][0]

def parse_fps(rate_str):
    """rate_str like '30/1'."""
    if not rate_str or rate_str == "0/0":
        return None
    num, den = rate_str.split("/")
    num = float(num)
    den = float(den) if float(den) != 0 else 1.0
    return num / den

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_frames_csv(csv_path: Path):
    if not csv_path.exists():
        return None, None
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None, None
    # best-effort parse fps from file
    try:
        fps = int(float(rows[0].get("fps", "0")))
    except Exception:
        fps = None
    # count rows as recorded frames
    return rows, fps

def depth_to_vis_u8(depth_u16, vis_max_mm=3000):
    depth_clip = np.clip(depth_u16.astype(np.float32), 0, float(vis_max_mm))
    u8 = (depth_clip / float(vis_max_mm) * 255.0).astype(np.uint8)
    return u8

def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def append_text(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

def ffmpeg_extract_frame_by_time(video_path: Path, t_sec: float, out_path: Path, pix_fmt=None):
    """
    Extract one frame near time t_sec as PNG.
    If pix_fmt is provided, force output pixel format.
    """
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t_sec:.6f}", "-i", str(video_path), "-frames:v", "1"]
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]
    cmd += [str(out_path)]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg frame extract failed:\n{err}")

def ffmpeg_decode_first_n_depth_frames_to_stats(depth_video: Path, width: int, height: int, n_frames: int, out_report: Path):
    """
    Decode first n_frames as raw gray16le and compute stats.
    Returns a dict with summary stats.
    """
    frame_bytes = width * height * 2  # uint16
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(depth_video),
        "-frames:v", str(n_frames),
        "-f", "rawvideo",
        "-pix_fmt", "gray16le",
        "pipe:1"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    total_frames = 0
    total_px = 0
    total_valid_px = 0
    total_zero_px = 0
    sum_valid = 0.0
    global_min = None
    global_max = None

    # For a quick histogram (sampled)
    hist_values = []

    try:
        while True:
            raw = p.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint16).reshape((height, width))

            total_frames += 1
            total_px += frame.size

            zero_mask = (frame == 0)
            valid_mask = ~zero_mask

            zero_count = int(np.sum(zero_mask))
            valid_count = frame.size - zero_count

            total_zero_px += zero_count
            total_valid_px += valid_count

            if valid_count > 0:
                valid_vals = frame[valid_mask]
                vmin = int(valid_vals.min())
                vmax = int(valid_vals.max())
                vmean = float(valid_vals.mean())

                sum_valid += float(valid_vals.sum())

                if global_min is None or vmin < global_min:
                    global_min = vmin
                if global_max is None or vmax > global_max:
                    global_max = vmax

                # For histogram, subsample to keep memory reasonable
                if len(hist_values) < 2_000_000:
                    # take every ~20th pixel from valid region
                    hist_values.append(valid_vals[::20])

            # optional progress
            if total_frames % 50 == 0:
                append_text(out_report, f"Scanned {total_frames} frames...\n")

        stderr = p.stderr.read().decode("utf-8", errors="ignore")
        p.wait(timeout=10)

    finally:
        try:
            p.kill()
        except Exception:
            pass

    valid_fraction = (total_valid_px / total_px) if total_px > 0 else 0.0
    zero_fraction = (total_zero_px / total_px) if total_px > 0 else 1.0
    mean_valid = (sum_valid / total_valid_px) if total_valid_px > 0 else float("nan")

    # collapse histogram values
    if hist_values:
        hist_values = np.concatenate(hist_values).astype(np.float32)
    else:
        hist_values = None

    return {
        "frames_scanned": total_frames,
        "total_pixels": total_px,
        "valid_pixels": total_valid_px,
        "zero_pixels": total_zero_px,
        "valid_fraction": valid_fraction,
        "zero_fraction": zero_fraction,
        "min_valid_mm": global_min,
        "max_valid_mm": global_max,
        "mean_valid_mm": mean_valid,
        "hist_values": hist_values
    }


# =========================
# Main analysis
# =========================
def main():
    session_dir = Path(SESSION_DIR)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    depth_video = session_dir / "Depth_video.mkv"
    rgb_video = session_dir / "RGB_video.mkv"
    frames_csv = session_dir / "frames.csv"
    calib_txt = session_dir / "calibration_params.txt"
    meta_txt = session_dir / "session_meta.txt"

    out_dir = session_dir / "Depth_Analysis"
    ensure_dir(out_dir)

    report_path = out_dir / "report.txt"
    write_text(report_path, f"Depth Validation Report\nCreated: {datetime.datetime.now().isoformat()}\n\n")

    # --- Existence checks ---
    missing = []
    for p in [depth_video, rgb_video]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        append_text(report_path, "Missing required files:\n" + "\n".join(missing) + "\n")
        print("Missing required files. See report:", report_path)
        return

    append_text(report_path, f"Session folder: {session_dir}\n")
    append_text(report_path, f"Depth video: {depth_video.name}\n")
    append_text(report_path, f"RGB video: {rgb_video.name}\n")
    append_text(report_path, f"frames.csv present: {frames_csv.exists()}\n")
    append_text(report_path, f"calibration_params.txt present: {calib_txt.exists()}\n")
    append_text(report_path, f"session_meta.txt present: {meta_txt.exists()}\n\n")

    # --- ffprobe info ---
    depth_info = ffprobe_stream_info(depth_video)
    rgb_info = ffprobe_stream_info(rgb_video)

    dw, dh = int(depth_info["width"]), int(depth_info["height"])
    d_pix = depth_info.get("pix_fmt", "")
    d_nb = depth_info.get("nb_frames", None)
    d_fps = parse_fps(depth_info.get("avg_frame_rate") or depth_info.get("r_frame_rate"))

    rw, rh = int(rgb_info["width"]), int(rgb_info["height"])
    r_pix = rgb_info.get("pix_fmt", "")
    r_nb = rgb_info.get("nb_frames", None)
    r_fps = parse_fps(rgb_info.get("avg_frame_rate") or rgb_info.get("r_frame_rate"))

    append_text(report_path, "FFprobe video stream info\n")
    append_text(report_path, f"Depth: {dw}x{dh}, pix_fmt={d_pix}, fps≈{d_fps}, nb_frames={d_nb}\n")
    append_text(report_path, f"RGB  : {rw}x{rh}, pix_fmt={r_pix}, fps≈{r_fps}, nb_frames={r_nb}\n\n")

    # --- CSV info (if exists) ---
    csv_rows, csv_fps = read_frames_csv(frames_csv)
    if csv_rows is not None:
        append_text(report_path, f"frames.csv rows: {len(csv_rows)}\n")
        append_text(report_path, f"frames.csv fps (parsed): {csv_fps}\n\n")
    else:
        append_text(report_path, "frames.csv not found or empty.\n\n")

    # Prefer a sane FPS estimate for time-to-frame mapping
    fps = None
    for candidate in [csv_fps, d_fps, r_fps]:
        if candidate and candidate > 0:
            fps = float(candidate)
            break
    if fps is None:
        fps = 15.0
        append_text(report_path, "WARNING: Could not determine FPS. Defaulting to 15.\n\n")

    # --- Scan first N frames for quality stats ---
    append_text(report_path, f"Scanning first {SCAN_FIRST_N_FRAMES} depth frames via FFmpeg raw decode...\n")
    stats = ffmpeg_decode_first_n_depth_frames_to_stats(depth_video, dw, dh, SCAN_FIRST_N_FRAMES, report_path)

    append_text(report_path, "\nDepth stats (first N frames)\n")
    append_text(report_path, f"frames_scanned: {stats['frames_scanned']}\n")
    append_text(report_path, f"valid_fraction: {stats['valid_fraction']*100:.2f}%\n")
    append_text(report_path, f"zero_fraction : {stats['zero_fraction']*100:.2f}%\n")
    append_text(report_path, f"min_valid_mm  : {stats['min_valid_mm']}\n")
    append_text(report_path, f"max_valid_mm  : {stats['max_valid_mm']}\n")
    append_text(report_path, f"mean_valid_mm : {stats['mean_valid_mm']:.2f}\n\n")

    # Quick decision messages
    if stats["frames_scanned"] == 0:
        append_text(report_path, "ERROR: No depth frames decoded. The depth video may be unreadable.\n")
        print("No frames decoded. See report:", report_path)
        return

    if stats["valid_fraction"] < 0.01:
        append_text(report_path, "ALERT: Depth is almost entirely zeros. This usually indicates capture/config problems.\n")
    elif stats["min_valid_mm"] is not None and stats["max_valid_mm"] is not None:
        if stats["max_valid_mm"] < 50:
            append_text(report_path, "ALERT: Depth values are extremely small. Units or encoding may be wrong.\n")

    # --- Histogram plot ---
    hist_vals = stats["hist_values"]
    if hist_vals is not None and hist_vals.size > 0:
        plt.figure(figsize=(8, 6))
        # robust range for plot
        lo = np.percentile(hist_vals, 1)
        hi = np.percentile(hist_vals, 99)
        if hi <= lo:
            lo, hi = float(hist_vals.min()), float(hist_vals.max()) + 1.0

        plt.hist(hist_vals, bins=80, range=(lo, hi), alpha=0.8)
        plt.xlabel("Depth (mm)")
        plt.ylabel("Frequency")
        plt.title(f"Depth Histogram (sampled from first {stats['frames_scanned']} frames)")
        plt.tight_layout()
        plt.savefig(out_dir / "depth_histogram.png", dpi=150)
        plt.close()
        append_text(report_path, "Saved: depth_histogram.png\n\n")
    else:
        append_text(report_path, "Histogram skipped (no valid depth values).\n\n")

    # --- Export sample frames across the video ---
    samples_dir = out_dir / "samples"
    ensure_dir(samples_dir)

    # Estimate total frames (best effort)
    total_frames_est = None
    if csv_rows is not None:
        total_frames_est = len(csv_rows)
    else:
        try:
            total_frames_est = int(d_nb) if d_nb not in (None, "N/A", "") else None
        except Exception:
            total_frames_est = None

    if total_frames_est is None:
        # fallback: assume a few seconds; still export from early part
        total_frames_est = max(stats["frames_scanned"], 1)
        append_text(report_path, "WARNING: Total frame count unknown. Sampling from early part of video.\n")

    append_text(report_path, f"Exporting {EXPORT_SAMPLE_FRAMES} sample frames (total_frames_est={total_frames_est}, fps≈{fps:.3f})...\n")

    # Choose evenly spaced frame indices
    if total_frames_est <= 1:
        sample_indices = [0]
    else:
        sample_indices = sorted(set(
            int(round(i * (total_frames_est - 1) / max(EXPORT_SAMPLE_FRAMES - 1, 1)))
            for i in range(EXPORT_SAMPLE_FRAMES)
        ))

    for idx in sample_indices:
        t_sec = idx / fps

        depth_png16 = samples_dir / f"depth_frame_{idx:06d}_16bit.png"
        depth_vis_png = samples_dir / f"depth_frame_{idx:06d}_vis.png"
        depth_bone_png = samples_dir / f"depth_frame_{idx:06d}_bone.png"
        rgb_png = samples_dir / f"rgb_frame_{idx:06d}.png"
        side_by_side = samples_dir / f"rgb_depth_{idx:06d}.png"

        # Extract depth as 16-bit PNG (gray16le)
        try:
            ffmpeg_extract_frame_by_time(depth_video, t_sec, depth_png16, pix_fmt="gray16le")
        except Exception as e:
            append_text(report_path, f"[WARN] Depth frame extract failed for idx={idx}: {e}\n")
            continue

        # Extract RGB frame
        try:
            ffmpeg_extract_frame_by_time(rgb_video, t_sec, rgb_png, pix_fmt=None)
        except Exception as e:
            append_text(report_path, f"[WARN] RGB frame extract failed for idx={idx}: {e}\n")
            # continue anyway with depth-only outputs

        # Load depth as uint16
        dimg = cv2.imread(str(depth_png16), cv2.IMREAD_UNCHANGED)
        if dimg is None:
            append_text(report_path, f"[WARN] Could not read extracted depth PNG for idx={idx}\n")
            continue

        if dimg.dtype != np.uint16:
            # Some FFmpeg builds may output 8-bit if the chain is wrong
            append_text(report_path, f"[WARN] Depth PNG dtype is {dimg.dtype} (expected uint16). Check ffmpeg output.\n")

        # Stats for this frame
        frame = dimg.astype(np.uint16)
        valid = frame[frame > 0]
        if valid.size > 0:
            vmin, vmax, vmean = int(valid.min()), int(valid.max()), float(valid.mean())
            append_text(report_path, f"Frame {idx:06d}: valid%={(valid.size/frame.size)*100:.2f}  min={vmin}  max={vmax}  mean={vmean:.2f}\n")
        else:
            append_text(report_path, f"Frame {idx:06d}: valid%=0.00 (all zeros)\n")

        # Create viewable visualization
        u8 = depth_to_vis_u8(frame, vis_max_mm=DEPTH_VIS_MAX_MM)
        cv2.imwrite(str(depth_vis_png), u8)
        bone = cv2.applyColorMap(u8, cv2.COLORMAP_BONE)
        cv2.imwrite(str(depth_bone_png), bone)

        # Side-by-side with RGB (if RGB extracted)
        rgb = cv2.imread(str(rgb_png), cv2.IMREAD_COLOR)
        if rgb is not None:
            # Ensure same height for side-by-side
            bone_bgr = bone
            if rgb.shape[0] != bone_bgr.shape[0]:
                bone_bgr = cv2.resize(bone_bgr, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)

            combo = np.hstack([rgb, bone_bgr])
            cv2.putText(combo, f"idx={idx}  t={t_sec:.2f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(side_by_side), combo)

    append_text(report_path, "\nSaved sample frames to: samples/\n")
    append_text(report_path, f"Depth visualization uses DEPTH_VIS_MAX_MM={DEPTH_VIS_MAX_MM} mm\n\n")

    # Explain black-video symptom
    append_text(
        report_path,
        "Note on 'black depth video':\n"
        "Depth_video.mkv is typically 16-bit grayscale (gray16le). Many players display it as black.\n"
        "Use the exported *_vis.png or *_bone.png images for a correct visual check.\n"
    )

    print("Done. Outputs written to:", out_dir)
    print("Open the report:", report_path)


if __name__ == "__main__":
    main()
