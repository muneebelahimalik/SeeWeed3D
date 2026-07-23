import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pyzed.sl as sl
import numpy as np
import cv2
import os
import datetime
import threading
import time
import csv
import shutil
import queue
import subprocess  # FFmpeg pipelines
from pydub import AudioSegment
from pydub.playback import play

###############################################################################
# Utility functions
###############################################################################

def create_save_directory(base_directory: str) -> str:
    if not os.path.exists(base_directory):
        os.makedirs(base_directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Session_{timestamp}"
    save_directory = os.path.join(base_directory, folder_name)
    os.makedirs(save_directory, exist_ok=True)
    return save_directory

def save_calibration_parameters(zed: sl.Camera, save_directory: str) -> None:
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters

    intrinsic_left = calibration_params.left_cam
    intrinsic_str_left = (
        f"Left Camera Intrinsic:\n"
        f"  fx: {intrinsic_left.fx:.2f}, fy: {intrinsic_left.fy:.2f}\n"
        f"  cx: {intrinsic_left.cx:.2f}, cy: {intrinsic_left.cy:.2f}\n"
        f"  Distortion: {intrinsic_left.disto}\n\n"
    )

    intrinsic_right = calibration_params.right_cam
    intrinsic_str_right = (
        f"Right Camera Intrinsic:\n"
        f"  fx: {intrinsic_right.fx:.2f}, fy: {intrinsic_right.fy:.2f}\n"
        f"  cx: {intrinsic_right.cx:.2f}, cy: {intrinsic_right.cy:.2f}\n"
        f"  Distortion: {intrinsic_right.disto}\n\n"
    )

    translation = calibration_params.stereo_transform.get_translation()
    extrinsic_str = (
        f"Extrinsic Translation:\n"
        f"  Tx: {translation.get()[0]:.2f}\n"
        f"  Ty: {translation.get()[1]:.2f}\n"
        f"  Tz: {translation.get()[2]:.2f}\n\n"
    )

    hfov_str = f"Horizontal FOV (Left Cam): {intrinsic_left.h_fov}\n"

    out_path = os.path.join(save_directory, "calibration_params.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(intrinsic_str_left + intrinsic_str_right + extrinsic_str + hfov_str)

def play_wav_file(wav_path: str) -> None:
    """Safely loads and plays a WAV file once."""
    if not os.path.exists(wav_path):
        print(f"[WARNING] WAV file not found: {wav_path}")
        return
    try:
        sound = AudioSegment.from_wav(wav_path)
        play(sound)
    except Exception as e:
        print(f"[WARNING] Could not load/play WAV: {e}")

def check_ffmpeg_available() -> bool:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        return completed.returncode == 0
    except Exception:
        return False

def get_free_space_bytes(path: str) -> int:
    """Returns free bytes on the drive containing path."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except Exception:
        return -1

###############################################################################
# Threaded capture + writer
###############################################################################

class CaptureThread(threading.Thread):
    """
    Continuously grabs frames at camera rate. Stores the latest frame for preview.
    If recording is enabled, it pushes frames to a bounded queue for the writer.
    """
    def __init__(
        self,
        zed: sl.Camera,
        runtime_params: sl.RuntimeParameters,
        image_mat: sl.Mat,
        depth_mat: sl.Mat,
        frame_queue: "queue.Queue",
        latest_lock: threading.Lock,
        latest_holder: dict,
        stop_event: threading.Event,
        recording_flag: threading.Event,
        depth_vis_max_mm: int = 3000
    ):
        super().__init__(daemon=True)
        self.zed = zed
        self.runtime_params = runtime_params
        self.image_mat = image_mat
        self.depth_mat = depth_mat
        self.frame_queue = frame_queue
        self.latest_lock = latest_lock
        self.latest_holder = latest_holder
        self.stop_event = stop_event
        self.recording_flag = recording_flag
        self.depth_vis_max_mm = max(1, int(depth_vis_max_mm))

        self.frame_idx = 0
        self.dropped_enqueue = 0

    def run(self):
        while not self.stop_event.is_set():
            err = self.zed.grab(self.runtime_params)
            if err != sl.ERROR_CODE.SUCCESS:
                # small sleep to avoid tight loop on errors
                time.sleep(0.002)
                continue

            # Retrieve data
            self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)          # BGRA
            self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)    # float32 depth in mm (because units set to mm)

            # Timestamp (nanoseconds in SDK, depends on version; keep as int)
            try:
                ts = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            except Exception:
                ts = int(time.time() * 1e9)

            rgb_bgra = self.image_mat.get_data()
            rgb_bgr = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR)

            depth = self.depth_mat.get_data()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # Update latest for preview (copy to avoid lifetime issues)
            with self.latest_lock:
                self.latest_holder["rgb_bgr"] = rgb_bgr
                self.latest_holder["depth_mm"] = depth
                self.latest_holder["timestamp_ns"] = ts
                self.latest_holder["frame_idx"] = self.frame_idx

            # If recording, enqueue full-res data for writer
            if self.recording_flag.is_set():
                try:
                    # Keep queue items lightweight but safe. Copy because mats reuse buffers.
                    item = (self.frame_idx, ts, rgb_bgr.copy(), depth.copy())
                    self.frame_queue.put_nowait(item)
                except queue.Full:
                    self.dropped_enqueue += 1  # drop frames when writer falls behind

            self.frame_idx += 1


class WriterThread(threading.Thread):
    """
    Pulls frames from a queue and writes to FFmpeg. Also writes a per-frame CSV log.
    """
    def __init__(
        self,
        frame_queue: "queue.Queue",
        stop_event: threading.Event,
        save_directory: str,
        width: int,
        height: int,
        fps: int,
        depth_clip_max_mm: int = 65535
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.save_directory = save_directory
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_clip_max_mm = max(1, int(depth_clip_max_mm))

        self.color_proc = None
        self.depth_proc = None

        self.csv_path = os.path.join(save_directory, "frames.csv")
        self.csv_file = None
        self.csv_writer = None

        self.frames_written = 0
        self.broken_pipe = False

    def _start_ffmpeg(self):
        rgb_video_path = os.path.join(self.save_directory, "RGB_video.mkv")
        depth_video_path = os.path.join(self.save_directory, "Depth_video.mkv")

        color_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "ffv1",
            "-level", "3",
            "-pix_fmt", "bgr24",
            rgb_video_path
        ]
        depth_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "gray16le",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "ffv1",
            "-level", "3",
            "-pix_fmt", "gray16le",
            depth_video_path
        ]

        # bufsize=0 to reduce latency; still may block if disk is slow, but thread isolates GUI/capture
        self.color_proc = subprocess.Popen(color_cmd, stdin=subprocess.PIPE, bufsize=0)
        self.depth_proc = subprocess.Popen(depth_cmd, stdin=subprocess.PIPE, bufsize=0)

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_idx",
            "timestamp_ns",
            "system_time_iso",
            "width",
            "height",
            "fps"
        ])

    def _close_ffmpeg(self):
        # Close CSV
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        self.csv_file = None
        self.csv_writer = None

        # Close FFmpeg pipes
        for proc_name in ["color_proc", "depth_proc"]:
            proc = getattr(self, proc_name)
            if proc:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.wait(timeout=10)
                except Exception as e:
                    print(f"[WARN] Error closing {proc_name}: {e}")
                setattr(self, proc_name, None)

    def run(self):
        self._start_ffmpeg()

        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame_idx, ts_ns, rgb_bgr, depth_mm = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.color_proc is None or self.depth_proc is None:
                self.frame_queue.task_done()
                continue

            # Depth encode: clip to uint16 and write as gray16le
            depth_u16 = np.clip(depth_mm, 0, self.depth_clip_max_mm).astype(np.uint16)

            try:
                self.color_proc.stdin.write(rgb_bgr.tobytes())
                self.depth_proc.stdin.write(depth_u16.tobytes())
            except (BrokenPipeError, OSError) as e:
                self.broken_pipe = True
                print(f"[ERROR] FFmpeg pipe error: {e}")
                # Stop consuming quickly; still close cleanly
                self.stop_event.set()

            # CSV log per frame
            if self.csv_writer:
                system_time_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
                self.csv_writer.writerow([frame_idx, ts_ns, system_time_iso, self.width, self.height, self.fps])

            self.frames_written += 1
            self.frame_queue.task_done()

        self._close_ffmpeg()

###############################################################################
# Main Application Class
###############################################################################

class ZEDTkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Collection Application")

        # ---------------------------
        # Configuration Variables
        # ---------------------------
        #self.base_directory = "C:/ZED/DataCollection/Vidalia_onions_nov_right_plot"
        self.wave_start_path = "C:/Users/mm17889/Zed2iData/working.wav"
        self.wave_stop_path = "C:/Users/mm17889/Zed2iData/save.wav"
   
        self.base_directory = "M:/Research/Data/Vidalia_onions_visit2_day2"
        #self.wave_start_path = "C:/Users/mm17889/Zed2iData/working.wav"
        #self.wave_stop_path = "C:/Users/mm17889/Zed2iData/save.wav"

        # Preview settings
        self.preview_scale = 0.5
        self.preview_update_ms = 50  # GUI refresh only
        self.depth_vis_max_mm = 3000  # fixed depth range for stable preview

        # Recording safety
        self.frame_queue_max = 120  # bounded queue to prevent RAM blow-up
        self.min_free_space_gb = 5  # block recording if free space below this

        # ZED objects
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.runtime_params = sl.RuntimeParameters()

        # Flags/states
        self.camera_opened = False
        self.save_directory = None

        # Threading primitives
        self.capture_stop_event = threading.Event()
        self.recording_event = threading.Event()  # set => recording enabled
        self.latest_lock = threading.Lock()
        self.latest_holder = {
            "rgb_bgr": None,
            "depth_mm": None,
            "timestamp_ns": None,
            "frame_idx": None
        }

        self.frame_queue = queue.Queue(maxsize=self.frame_queue_max)
        self.capture_thread = None
        self.writer_thread = None

        # Resolution / FPS options
        self.res_options = {
            "HD2K (2208x1242)": (sl.RESOLUTION.HD2K, 15),
            "HD1080 (1920x1080)": (sl.RESOLUTION.HD1080, 30),
            "HD720 (1280x720)": (sl.RESOLUTION.HD720, 60),
            "VGA (672x376)": (sl.RESOLUTION.VGA, 100),
        }

        # -------------------------------------------------------------------
        # ttk styling (same as yours)
        # -------------------------------------------------------------------
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("AppBar.TFrame", background="#E9ECEF")
        style.configure("AppBar.TLabel",
                        font=("Segoe UI", 16, "bold"),
                        background="#E9ECEF",
                        foreground="black")
        style.configure("AppBarSub.TFrame", background="#E9ECEF")
        style.configure("TLabel",
                        font=("Segoe UI", 11),
                        background="white",
                        foreground="black")
        style.configure("TButton",
                        font=("Segoe UI", 10, "bold"),
                        foreground="white",
                        background="#0078D7",
                        padding=6)
        style.map("TButton", background=[("active", "#005A9E")])
        style.configure("TCombobox", font=("Segoe UI", 10), padding=3)
        style.configure("TEntry", font=("Segoe UI", 10), padding=3)
        style.configure("WeedScan.TLabel",
                        font=("Times New Roman", 18, "bold"),
                        foreground="#BA0C2F",
                        background="#E9ECEF")

        self.root.configure(bg="white")
        self.create_gui_layout()

        # Mats allocated on open
        self.image_zed = None
        self.depth_zed = None

    def create_gui_layout(self):
        # Title area
        title_frame = ttk.Frame(self.root, padding=(10, 10, 10, 0), style="AppBar.TFrame")
        title_frame.pack(side=tk.TOP, fill=tk.X)

        title_center_frame = ttk.Frame(title_frame, style="AppBarSub.TFrame")
        title_center_frame.pack(expand=True, anchor="center")

        weedscan_label = ttk.Label(title_center_frame, text="WeedScan", style="WeedScan.TLabel", anchor="center")
        weedscan_label.pack(side=tk.LEFT, padx=(5, 0))

        rest_label = ttk.Label(
            title_center_frame,
            text=": Data Collection System for Vidalia Onion Weeds",
            style="AppBar.TLabel",
            anchor="center"
        )
        rest_label.pack(side=tk.LEFT, padx=5)

        subtitle_frame = ttk.Frame(self.root, padding=5, style="AppBar.TFrame")
        subtitle_frame.pack(side=tk.TOP, fill=tk.X)

        subtitle_center_frame = ttk.Frame(subtitle_frame, style="AppBarSub.TFrame")
        subtitle_center_frame.pack(expand=True, anchor="center")

        subtitle_label = ttk.Label(
            subtitle_center_frame,
            text="Precision Crop Protection Lab",
            style="AppBar.TLabel",
            anchor="center",
            font=("Segoe UI", 14, "italic")
        )
        subtitle_label.pack(side=tk.TOP)

        # Top controls
        top_frame = ttk.Frame(self.root, padding=5, style="AppBar.TFrame")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        top_center_frame = ttk.Frame(top_frame, style="AppBarSub.TFrame")
        top_center_frame.pack(expand=True, anchor="center")

        ttk.Label(top_center_frame, text="Resolution:", style="AppBar.TLabel").pack(side=tk.LEFT, padx=5)

        self.res_var = tk.StringVar(value="HD2K (2208x1242)")
        self.res_menu = ttk.Combobox(
            top_center_frame,
            textvariable=self.res_var,
            values=list(self.res_options.keys()),
            width=18,
            state="readonly"
        )
        self.res_menu.pack(side=tk.LEFT, padx=5)

        ttk.Label(top_center_frame, text="FPS:", style="AppBar.TLabel").pack(side=tk.LEFT, padx=5)

        self.fps_var = tk.StringVar(value="15")
        self.fps_entry = ttk.Entry(top_center_frame, textvariable=self.fps_var, width=5)
        self.fps_entry.pack(side=tk.LEFT, padx=5)

        self.open_cam_btn = ttk.Button(top_center_frame, text="Open Camera", command=self.open_camera)
        self.open_cam_btn.pack(side=tk.LEFT, padx=(15, 5))

        self.close_cam_btn = ttk.Button(top_center_frame, text="Close Camera", command=self.close_camera, state=tk.DISABLED)
        self.close_cam_btn.pack(side=tk.LEFT, padx=5)

        # Mid controls
        mid_frame = ttk.Frame(self.root, padding=5, style="AppBar.TFrame")
        mid_frame.pack(side=tk.TOP, fill=tk.X)

        mid_center_frame = ttk.Frame(mid_frame, style="AppBarSub.TFrame")
        mid_center_frame.pack(expand=True, anchor="center")

        self.start_btn = ttk.Button(mid_center_frame, text="Start Recording", command=self.start_recording, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(10, 10))

        self.stop_btn = ttk.Button(mid_center_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Video display
        self.video_label = ttk.Label(self.root, background="white")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status line
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.root, textvariable=self.status_var, background="white")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))

    ############################################################################
    # Camera / threads
    ############################################################################

    def open_camera(self):
        if self.camera_opened:
            print("[INFO] Camera is already opened.")
            return

        if not check_ffmpeg_available():
            messagebox.showerror(
                "FFmpeg not found",
                "FFmpeg was not found on PATH.\n\nInstall FFmpeg or add it to PATH, then restart."
            )
            return

        chosen_res_label = self.res_var.get()
        res_enum, default_fps = self.res_options[chosen_res_label]
        try:
            user_fps = int(self.fps_var.get())
        except ValueError:
            user_fps = default_fps
            self.fps_var.set(str(default_fps))

        print(f"[INFO] Opening ZED with {chosen_res_label} at {user_fps} FPS")

        self.init_params.camera_resolution = res_enum
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        self.init_params.camera_fps = user_fps

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            messagebox.showerror("ZED Open Error", f"Cannot open ZED camera:\n{err}")
            return

        self.image_zed = sl.Mat()
        self.depth_zed = sl.Mat()

        self.camera_opened = True
        self.open_cam_btn.config(state=tk.DISABLED)
        self.close_cam_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)

        # Start capture thread
        self.capture_stop_event.clear()
        self.recording_event.clear()

        # Reset queue
        self._drain_queue()

        self.capture_thread = CaptureThread(
            zed=self.zed,
            runtime_params=self.runtime_params,
            image_mat=self.image_zed,
            depth_mat=self.depth_zed,
            frame_queue=self.frame_queue,
            latest_lock=self.latest_lock,
            latest_holder=self.latest_holder,
            stop_event=self.capture_stop_event,
            recording_flag=self.recording_event,
            depth_vis_max_mm=self.depth_vis_max_mm
        )
        self.capture_thread.start()

        # Start GUI preview loop
        self.status_var.set("Camera opened")
        self.update_preview()

    def close_camera(self):
        if not self.camera_opened:
            return

        # Stop recording first
        if self.recording_event.is_set():
            self.stop_recording()

        # Stop capture thread
        self.capture_stop_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.capture_thread = None

        self.zed.close()
        self.camera_opened = False

        self.open_cam_btn.config(state=tk.NORMAL)
        self.close_cam_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

        self.video_label.config(image="")
        self.status_var.set("Camera closed")
        print("[INFO] ZED camera closed.")

    def _drain_queue(self):
        try:
            while True:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
        except queue.Empty:
            pass

    def update_preview(self):
        if not self.camera_opened:
            return

        rgb_bgr = None
        depth_mm = None
        frame_idx = None

        with self.latest_lock:
            rgb_bgr = self.latest_holder.get("rgb_bgr")
            depth_mm = self.latest_holder.get("depth_mm")
            frame_idx = self.latest_holder.get("frame_idx")

        if rgb_bgr is not None and depth_mm is not None:
            # Stable depth visualization (fixed max range)
            depth_vis = np.clip(depth_mm, 0, self.depth_vis_max_mm)
            depth_u8 = (depth_vis / float(self.depth_vis_max_mm) * 255.0).astype(np.uint8)
            depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

            combined = np.hstack((rgb_bgr, depth_bgr))

            # Scale down for display
            new_width = int(combined.shape[1] * self.preview_scale)
            new_height = int(combined.shape[0] * self.preview_scale)
            combined_small = cv2.resize(combined, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Overlay info
            rec = "REC" if self.recording_event.is_set() else "PREVIEW"
            qsize = self.frame_queue.qsize()
            overlay = f"{rec} | frame {frame_idx} | Q {qsize}/{self.frame_queue_max}"
            cv2.putText(
                combined_small,
                overlay,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if self.recording_event.is_set() else (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            frame_rgb = cv2.cvtColor(combined_small, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk

        # Schedule next preview update (does NOT control capture rate)
        self.root.after(self.preview_update_ms, self.update_preview)

    ############################################################################
    # Recording / Saving
    ############################################################################

    def start_recording(self):
        if not self.camera_opened:
            print("[WARN] Camera not opened, cannot start recording.")
            return
        if self.recording_event.is_set():
            print("[INFO] Already recording.")
            return

        # Disk space check
        free_bytes = get_free_space_bytes(self.base_directory)
        if free_bytes >= 0:
            free_gb = free_bytes / (1024**3)
            if free_gb < self.min_free_space_gb:
                messagebox.showerror(
                    "Low disk space",
                    f"Free space is {free_gb:.2f} GB.\n"
                    f"Recording requires at least {self.min_free_space_gb} GB free."
                )
                return

        self.save_directory = create_save_directory(self.base_directory)
        save_calibration_parameters(self.zed, self.save_directory)

        # Record a small session metadata file
        try:
            info = self.zed.get_camera_information()
            cam_cfg = info.camera_configuration
            meta_path = os.path.join(self.save_directory, "session_meta.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"created: {datetime.datetime.now().isoformat()}\n")
                f.write(f"resolution: {cam_cfg.resolution.width}x{cam_cfg.resolution.height}\n")
                f.write(f"fps: {cam_cfg.fps}\n")
                f.write(f"depth_mode: NEURAL\n")
                f.write(f"units: MILLIMETER\n")
                f.write(f"depth_vis_max_mm: {self.depth_vis_max_mm}\n")
        except Exception as e:
            print(f"[WARN] Could not write session metadata: {e}")

        # Start sound (async)
        threading.Thread(target=play_wav_file, args=(self.wave_start_path,), daemon=True).start()

        # Camera resolution/fps for writer
        cam_info = self.zed.get_camera_information()
        cam_config = cam_info.camera_configuration
        width = cam_config.resolution.width
        height = cam_config.resolution.height
        fps = cam_config.fps

        # Reset queue so writer starts cleanly
        self._drain_queue()

        # Start writer thread
        writer_stop_event = threading.Event()
        self.writer_stop_event = writer_stop_event  # keep ref for stop

        self.writer_thread = WriterThread(
            frame_queue=self.frame_queue,
            stop_event=writer_stop_event,
            save_directory=self.save_directory,
            width=width,
            height=height,
            fps=fps,
            depth_clip_max_mm=65535
        )
        self.writer_thread.start()

        # Enable recording in capture thread
        self.recording_event.set()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Recording to {self.save_directory}")
        print(f"[INFO] Recording started. Saving to {self.save_directory}")

    def stop_recording(self):
        if not self.recording_event.is_set():
            return

        # Stop capture enqueue
        self.recording_event.clear()

        # Signal writer to finish remaining queued frames
        if getattr(self, "writer_stop_event", None):
            self.writer_stop_event.set()

        # Wait for writer thread to drain
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=15.0)

        # Collect stats
        frames_written = getattr(self.writer_thread, "frames_written", 0) if self.writer_thread else 0
        broken_pipe = getattr(self.writer_thread, "broken_pipe", False) if self.writer_thread else False
        dropped = getattr(self.capture_thread, "dropped_enqueue", 0) if self.capture_thread else 0

        self.writer_thread = None
        self.writer_stop_event = None

        # Stop sound (async)
        threading.Thread(target=play_wav_file, args=(self.wave_stop_path,), daemon=True).start()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        msg = f"Recording stopped. Written frames: {frames_written}. Dropped (queue full): {dropped}."
        if broken_pipe:
            msg += " FFmpeg encountered a pipe error. Check disk speed/space and FFmpeg logs."
        self.status_var.set(msg)

        print("[INFO] Recording stopped.")
        print(f"[INFO] Files saved to {self.save_directory}")
        print(f"[INFO] Written frames: {frames_written}, Dropped enqueue: {dropped}, Broken pipe: {broken_pipe}")

###############################################################################
# Main
###############################################################################

def on_closing(app: ZEDTkApp, root: tk.Tk):
    try:
        if app.recording_event.is_set():
            app.stop_recording()
        if app.camera_opened:
            app.close_camera()
    finally:
        root.destroy()

def main():
    root = tk.Tk()
    app = ZEDTkApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(app, root))
    root.mainloop()

if __name__ == "__main__":
    main()
