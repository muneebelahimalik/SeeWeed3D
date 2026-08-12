#!/usr/bin/env python3
"""
SeeWeed3D - resolving a dataset root, and saying what is wrong when it is not one.

Two directory trees in this pipeline look alike from a config block and are
nothing alike on disk:

    RECORDINGS  (extract_sessions.py INPUT_ROOTS)
        <root>/Session_20250221_130957/RGB_video.avi
                                       Depth_video.avi
                                       calibration_params.txt

    DATASET     (extract_sessions.py OUTPUT_ROOT, and every stage after it)
        <root>/sessions/Visit1_20250221_130957/rgb/
                                               meta/pool.csv
        <root>/registry.csv

Every stage after extraction wants the second, and its config key is called
DATASET_ROOT while extraction's is INPUT_ROOTS/OUTPUT_ROOT - so pasting the
recordings path into the next script is the natural mistake, not a careless one.

The old message for it was "ERROR: <path>\\sessions not found. Run
extract_sessions.py first." That is exactly wrong advice when extraction HAS
been run and its output is sitting in another folder: it sends you to re-run a
job that already succeeded, and says nothing about the path you actually typed.
"""
from pathlib import Path


def looks_like_recordings(root):
    """Does this directory hold raw Session_* recording folders?

    Detected by structure rather than by name, so it also catches a folder
    whose sessions were renamed: a child directory containing a video file and
    no rgb/ subfolder is a recording, not an extracted session."""
    root = Path(root)
    if not root.is_dir():
        return []
    hits = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or (child / "rgb").is_dir():
            continue
        try:
            if any(f.suffix.lower() in (".mkv", ".avi", ".mp4", ".mov", ".svo",
                                        ".svo2")
                   for f in child.iterdir() if f.is_file()):
                hits.append(child.name)
        except (PermissionError, OSError):
            continue
    return hits


def sessions_root_problem(dataset_root):
    """Why this is not a dataset root, or None if it is one.

    Ordered by what is actually most likely: the recordings-path mix-up first,
    because it is the one the old message actively misdirected."""
    root = Path(dataset_root)
    sessions = root / "sessions"
    if sessions.is_dir():
        return None

    if not root.exists():
        return (f"ERROR: {root} does not exist.\n"
                f"  DATASET_ROOT is the OUTPUT_ROOT you gave "
                f"extract_sessions.py, not a recordings folder.")

    rec = looks_like_recordings(root)
    if rec:
        shown = ", ".join(rec[:3]) + (f" (+{len(rec) - 3} more)"
                                      if len(rec) > 3 else "")
        return (f"ERROR: {root} is a RECORDINGS folder, not an extracted "
                f"dataset.\n"
                f"  It holds raw session folders - {shown} - each with video "
                f"files in it.\n"
                f"  That path belongs in extract_sessions.py's INPUT_ROOTS. "
                f"DATASET_ROOT wants the\n"
                f"  OUTPUT_ROOT you gave extract_sessions.py, which is where "
                f"sessions/ and registry.csv were written.")

    return (f"ERROR: {sessions} not found.\n"
            f"  {root} exists but holds no sessions/ folder, so nothing has "
            f"been extracted here.\n"
            f"  Either run extract_sessions.py with OUTPUT_ROOT set to this "
            f"path, or point\n  DATASET_ROOT at the folder where it already "
            f"wrote its output.")


def require_sessions_root(dataset_root):
    """The <dataset_root>/sessions path, or exit with a message that names the
    actual problem."""
    problem = sessions_root_problem(dataset_root)
    if problem:
        raise SystemExit(problem)
    return Path(dataset_root) / "sessions"
