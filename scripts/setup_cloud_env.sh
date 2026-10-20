#!/usr/bin/env bash
# SeeWeed3D - bring a fresh cloud container up to a FULL GREEN TEST SUITE.
#
#   bash scripts/setup_cloud_env.sh
#
# Point the cloud environment's Setup script at this file. A container is
# rebuilt often and starts bare every time; without this, each new session
# reinstalls by hand, and the parts that are tedious to install get skipped -
# which is how 85 tests spent a whole working session being excluded as
# "environmental" rather than run.
#
# WHY THIS LIVES IN THE REPO rather than in the environment settings box: it is
# reviewable, it changes with the code that needs it, and a test can read it.
# The settings box then holds one line that never changes again.
#
# Idempotent. Safe to re-run, and re-running is the normal case.
set -euo pipefail

say() { printf "\n==> %s\n" "$*"; }

# -- 1. ffmpeg AND ffprobe -----------------------------------------------------
# Not optional and not pip-installable. The extraction tests build real FFV1
# recordings and decode them, so without these, 48 tests do not run - including
# every check that depth survives a round trip, which is the one measurement
# this project cannot afford to get wrong.
#
# imageio-ffmpeg bundles ffmpeg but NOT ffprobe, and the extractor needs both,
# so this uses apt. `apt-get update` first: the image ships a stale index and
# the install 404s on moved packages without it.
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  say "installing ffmpeg + ffprobe"
  apt-get update -qq
  apt-get install -y -qq ffmpeg
fi

# -- 2. Python packages --------------------------------------------------------
say "installing Python packages"

# HEADLESS OpenCV, deliberately against requirements.txt. That file names
# `opencv-python`, which drags in libGL and fails to import on a container with
# no display. Same cv2 API, no GUI backend. requirements.txt is right for a
# workstation; this is right for here.
# NOT `pip install --upgrade pip`: pip here is Debian-managed and has no
# RECORD file, so the upgrade aborts with "Cannot uninstall pip" and takes the
# whole script down with it under `set -e`. The shipped pip is fine.
python -m pip install -q "numpy>=1.26,<2" "opencv-python-headless<5" pillow pytest

# numpy is pinned <2 because Meta's sam3 requires it, and recent
# opencv-python-headless declares numpy>=2. pip resolves that by upgrading
# numpy and silently breaking the pin, so numpy goes LAST and wins.
python -m pip install -q "numpy>=1.26,<2"

# CPU torch: torchvision for the Mask R-CNN backend, matplotlib for the report
# figures. Without them 31 more tests fail on ImportError. The CPU index keeps
# this to a few hundred MB instead of a CUDA build nothing here can use.
python -m pip install -q torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q matplotlib

# -- 3. Prove it, rather than hope ---------------------------------------------
say "verifying"
command -v ffmpeg  >/dev/null || { echo "ffmpeg missing";  exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe missing"; exit 1; }
python - <<'PYEOF'
import sys
import numpy, cv2, pytest, torch, torchvision, matplotlib
assert numpy.__version__.startswith("1."), (
    f"numpy {numpy.__version__} - the <2 pin was clobbered, and sam3 needs it")
print(f"  numpy {numpy.__version__}  cv2 {cv2.__version__}  "
      f"torch {torch.__version__}")
print(f"  torchvision {torchvision.__version__}  "
      f"matplotlib {matplotlib.__version__}  pytest {pytest.__version__}")
PYEOF

say "ready - run: python -m pytest tests/ -q"
