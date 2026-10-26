"""The cloud setup script.

A container here is rebuilt often and starts bare, so the difference between
"the suite runs" and "most of the suite is skipped as environmental" is one
script staying correct. These pin the parts that fail SILENTLY - a missing
binary looks like an unrelated error, and a clobbered pin looks like nothing
at all until SAM 3 is loaded.
"""
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_cloud_env.sh"


@pytest.fixture(scope="module")
def body():
    assert SCRIPT.exists(), f"{SCRIPT} is missing"
    return SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def code(body):
    """The script with comments stripped.

    The comments explain the traps by NAMING them - "not `pip install
    --upgrade pip`", and so on - so a check run against the whole file fires
    on the explanation of the rule rather than on a breach of it. A test reads
    the code; the prose is for people."""
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_it_is_valid_shell():
    """A syntax error here fails every future session at startup, where the
    message is far from the cause."""
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_it_stops_on_the_first_failure(body):
    """Without `set -e` a failed install is a warning in a log nobody reads,
    and the session continues with a broken environment."""
    assert re.search(r"set -euo pipefail", body)


def test_it_installs_ffprobe_not_only_ffmpeg(body):
    """The extractor probes streams as well as decoding them, and
    imageio-ffmpeg - the obvious pip answer - bundles ffmpeg ALONE. Half the
    pair is what makes the extraction tests fail with FileNotFoundError from
    deep inside subprocess."""
    assert "ffprobe" in body
    assert "apt-get install" in body and "ffmpeg" in body


def test_it_refreshes_the_apt_index_before_installing(body):
    """The image ships a stale index; installing without updating 404s on
    packages that have moved."""
    assert body.index("apt-get update") < body.index("apt-get install")


def test_numpy_is_pinned_below_2_and_installed_last(body):
    """opencv declares numpy>=2, so pip resolves the conflict by upgrading
    numpy and silently breaking the pin that sam3 depends on. The pin only
    holds if it is applied after everything that fights it."""
    pins = [m.start() for m in re.finditer(r'"numpy>=1\.26,<2"', body)]
    assert len(pins) >= 2, "numpy is not re-pinned after the packages that move it"
    assert pins[-1] > body.index("opencv-python-headless")


def test_opencv_is_headless(body, code):
    """`opencv-python` needs libGL and will not import on a container with no
    display - which reads as a broken cv2 rather than a missing system lib."""
    assert "opencv-python-headless" in body
    assert not re.search(r'"opencv-python[<>=]', code)


def test_it_does_not_try_to_upgrade_pip(code):
    """pip here is Debian-managed and has no RECORD file, so the upgrade dies
    with 'Cannot uninstall pip' and takes the whole script with it."""
    assert not re.search(r"install\s+(-q\s+)?--upgrade\s+pip", code)


def test_it_verifies_instead_of_assuming(body):
    """A setup script that exits 0 without checking is worth little: the
    failure then surfaces an hour later as a confusing test error."""
    assert "command -v ffprobe" in body
    assert "import numpy, cv2" in body or "import numpy" in body


def test_the_torch_build_is_cpu_only(body):
    """Nothing in this container has a GPU, and the CUDA wheels are gigabytes
    of download for an identical test result."""
    assert "download.pytorch.org/whl/cpu" in body


def test_pillow_carries_a_version_floor(code):
    """Debian's pillow 10.2 lives in /usr/lib/python3/dist-packages, which is on
    3.11's sys.path, and its C extensions are built for another ABI. An
    unpinned `pip install pillow` treats it as already satisfied and installs
    nothing, so `import PIL.Image` dies on a cpython-312 .so and takes
    torchvision down with it. The floor is what forces a real cp311 wheel."""
    m = re.search(r"pillow[^\s\"']*", code)
    assert m, "pillow is not installed at all"
    assert re.search(r"\"pillow>=\d+\"", code), (
        f"pillow must carry a version floor, found: {m.group(0)}")


def test_every_install_goes_through_python_dash_m_pip(code):
    """`python` is 3.11 under /usr/local; `pip` on PATH is Debian's 3.13. A bare
    `pip install` succeeds loudly and installs into the interpreter that never
    runs the tests."""
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("pip ") or s.startswith("pip3 "):
            pytest.fail(f"bare pip invocation: {s}")
        if "pip install" in s:
            assert "python -m pip install" in s, f"not via python -m pip: {s}"


def test_the_verify_step_rejects_packages_from_debians_tree(body):
    """The whole class of failure is a package resolving out of another
    interpreter's site directory, so the check is on the path, not on one
    package name."""
    assert "/usr/lib/python3/dist-packages" in body
    assert "PIL.Image" in body, "importing PIL alone does not load the C extension"
