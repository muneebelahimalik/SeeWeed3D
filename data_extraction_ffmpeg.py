import os
import shutil
import subprocess
import json

###############################################################################
# CONFIGURE YOUR FFMPEG PATH HERE
###############################################################################
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"  # <-- Change to your ffmpeg.exe location

# Global frame counter file (placed in a fixed location)
GLOBAL_COUNTER_FILE = r"C:\ZED\DataCollection\global_frame_counter.json"  # <-- Change this path as needed

def get_next_frame_number():
    """Get the next available frame number from the global counter file."""
    if os.path.exists(GLOBAL_COUNTER_FILE):
        with open(GLOBAL_COUNTER_FILE, 'r') as f:
            data = json.load(f)
            return data.get('next_frame', 0)
    return 0

def save_next_frame_number(next_frame):
    """Save the next frame number to the global counter file."""
    with open(GLOBAL_COUNTER_FILE, 'w') as f:
        json.dump({'next_frame': next_frame}, f)

def extract_frames_ffmpeg(video_path, save_folder, fps=15, interval=1, is_depth=False, start_frame=0):
    """
    Extract frames using FFmpeg from a video file (FFV1 in MKV).
    - video_path: path to the .mkv file
    - save_folder: folder to save extracted frames
    - fps: the *original* capture FPS (not always used, but kept for reference)
    - interval: extract 1 frame every 'interval' seconds
    - is_depth: if True, treat as 16-bit depth video and preserve 16-bit PNG
    - start_frame: absolute frame number to start from
    
    Returns the next available frame number
    """
    print(f"Opening video with FFmpeg: {video_path}")
    if not os.path.exists(video_path):
        print(f"❌ Error: Could not find {video_path}")
        return start_frame

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Calculate how many frames per second to extract
    extraction_fps = 1.0 / interval

    # Use a temporary directory for extraction
    temp_dir = os.path.join(save_folder, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Output temporary file pattern
    temp_pattern = os.path.join(temp_dir, "temp_%04d.png")

    # Build the ffmpeg command:
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vf", f"fps={extraction_fps}",
        "-vsync", "0"
    ]

    # If this is a 16-bit depth video, we force gray16le output
    if is_depth:
        cmd += ["-pix_fmt", "gray16le"]

    # Specify the temporary output pattern
    cmd += [temp_pattern]

    print(f"Running FFmpeg command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Frames extracted to temporary folder")
        
        # Get the number of frames extracted (to return for synchronization)
        temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("temp_")])
        num_frames = len(temp_files)
        
        # Rename files with absolute frame numbers
        current_frame = start_frame
        for temp_file in temp_files:
            temp_path = os.path.join(temp_dir, temp_file)
            new_name = f"frame_{current_frame:08d}.png"  # 8 digits for larger datasets
            new_path = os.path.join(save_folder, new_name)
            shutil.move(temp_path, new_path)
            current_frame += 1
            
        # Clean up temporary directory
        os.rmdir(temp_dir)
        print(f"✅ Frames renamed with absolute numbering starting from {start_frame}")
        
        return start_frame, num_frames  # Return start_frame and count for synchronization
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e}")
        return start_frame, 0

def organize_data(main_directory, output_directory, fps=15, interval=1):
    """Organize RGB and Depth video data into structured directories, using FFmpeg to decode."""
    print(f"Processing data from {main_directory} and saving to {output_directory}")
    print(f"Using parameters - FPS: {fps}, Frame Extraction Interval: {interval} second(s)")

    if not os.path.exists(main_directory):
        print(f"❌ Error: Directory {main_directory} does not exist!")
        return

    # Create subfolders for the extracted frames
    os.makedirs(output_directory, exist_ok=True)
    rgb_output = os.path.join(output_directory, "rgb_frames")
    depth_output = os.path.join(output_directory, "depth_frames")
    calibration_output = os.path.join(output_directory, "calibration")

    os.makedirs(rgb_output, exist_ok=True)
    os.makedirs(depth_output, exist_ok=True)
    os.makedirs(calibration_output, exist_ok=True)

    # Get the next available frame number from the GLOBAL counter
    next_frame = get_next_frame_number()
    print(f"Starting extraction from absolute frame number: {next_frame}")

    # Expect these filenames in main_directory
    rgb_video_path = os.path.join(main_directory, "RGB_video.mkv")
    depth_video_path = os.path.join(main_directory, "Depth_video.mkv")
    calibration_file = os.path.join(main_directory, "calibration_params.txt")

    # Extract RGB frames first to determine frame count
    if os.path.exists(rgb_video_path):
        print(f"✅ RGB Video found: {rgb_video_path}")
        start_frame, num_frames = extract_frames_ffmpeg(rgb_video_path, rgb_output, fps, interval, is_depth=False, start_frame=next_frame)
        # Update the next frame number
        next_frame = start_frame + num_frames
    else:
        print(f"❌ RGB Video missing in {main_directory}")
        start_frame, num_frames = next_frame, 0

    # Extract Depth frames using the SAME frame numbers as RGB
    if os.path.exists(depth_video_path):
        print(f"✅ Depth Video found: {depth_video_path}")
        extract_frames_ffmpeg(depth_video_path, depth_output, fps, interval, is_depth=True, start_frame=start_frame)
        print(f"✅ Depth frames numbered to match RGB frames ({start_frame} to {start_frame + num_frames - 1})")
    else:
        print(f"❌ Depth Video missing in {main_directory}")

    # Copy calibration file if present
    if os.path.exists(calibration_file):
        dest_calibration_file = os.path.join(calibration_output, "calibration_params.txt")
        shutil.copy(calibration_file, dest_calibration_file)
        print(f"✅ Copied calibration file to {dest_calibration_file}")
    else:
        print(f"❌ Calibration File missing in {main_directory}")

    # Save the next frame number for future runs to the GLOBAL counter
    save_next_frame_number(next_frame)
    print(f"✅ Data extraction complete! Global frame counter updated. Next frame number will be: {next_frame}")

    # Also save a local record for reference
    local_info_file = os.path.join(output_directory, "frame_info.txt")
    with open(local_info_file, 'w') as f:
        f.write(f"First frame: {start_frame}\n")
        f.write(f"Last frame: {next_frame - 1}\n")
        f.write(f"Total frames in this batch: {num_frames}\n")
    print(f"✅ Frame information saved to {local_info_file}")


# Example usage
if __name__ == "__main__":
    main_directory = r"C:\ZED\DataCollection\Vidalia2\Session_20250304_142804"  # Example path
    output_directory = r"M:\Research\Data\Weeds\Vidalia2\12"                    # Example output
    fps = 15       # For reference, original camera FPS
    interval = 0.4348   # Extract 1 frame every 0.4348 seconds = 2.3 FPS

    organize_data(main_directory, output_directory, fps, interval)