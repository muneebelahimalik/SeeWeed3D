#!/usr/bin/env python3
"""
SeeWeed3D - experiment tracking, local and optional.

Backends are chosen for one property above all: **nothing leaves the machine**.
Field imagery and the geometry of a commercial onion operation are not data to
upload to a vendor's cloud for the convenience of a nicer chart. Both supported
backends write to a local directory and neither needs an account.

    tensorboard  scalar curves + images. `pip install tensorboard`. Already a
                 torch dependency in most installs, so usually free.
    mlflow       runs, parameters, metrics, artifacts, and - the reason it is
                 here - a run COMPARISON table. Apache-2.0, `pip install
                 mlflow`, local `./mlruns` file store by default.

Use both: TensorBoard answers "is this run learning", MLflow answers "which of
my eleven runs was best and what was different about it". At 45 annotated
frames the second question is the one that actually costs you time.

Every backend is optional. `Tracker(backend="none")` is a working no-op, so the
training script has exactly one code path whether or not anything is installed.
An explicitly requested backend that is missing raises - silently degrading a
requested backend to no-op would let you finish a run believing it was logged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BACKENDS = ("none", "tensorboard", "mlflow", "all")


def _flatten(d, prefix=""):
    """MLflow params are scalars. Nested config becomes dotted keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(list(v))
        else:
            out[key] = v
    return out


#: MLflow experiment every backend logs into, so a Mask R-CNN run and an
#: RF-DETR run appear in ONE comparison table rather than two.
EXPERIMENT = "seeweed3d"


def mlflow_store_uri(out_dir, create=False):
    """The one definition of where MLflow runs are stored.

    Returns ``(uri, artifacts_dir_or_None)``. artifacts is None when the URI
    came from the environment, since then the store is not ours to lay out.

    SQLite, NOT the bare './mlruns' file store: MLflow 3 refuses the filesystem
    backend outright ("in maintenance mode"), so the obvious local choice raises
    on start. This lives here rather than inline in Tracker because the RF-DETR
    backend has to point pytorch-lightning's own MLFlowLogger at the SAME store
    - and a second copy of this path expression is a second store that only
    looks like the first one.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env, None
    store = (Path(out_dir).parent / "mlruns").resolve()
    artifacts = store / "artifacts"
    if create:
        store.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + store.joinpath("mlflow.db").as_posix(), artifacts


class Tracker:
    """One handle over zero or more local tracking backends.

    Deliberately not a context manager around the whole training loop: a crash
    mid-training should leave the metrics logged so far on disk, which the file
    backends already guarantee. `close()` is idempotent."""

    def __init__(self, backend="auto", out_dir=".", run_name=None,
                 experiment="seeweed3d"):
        self.out_dir = Path(out_dir)
        self.run_name = run_name or self.out_dir.name
        self.experiment = experiment
        self._tb = None
        self._mlflow = None
        self._closed = False
        self._uri = None
        self.system_metrics = False

        if backend not in BACKENDS + ("auto",):
            raise ValueError(f"backend must be one of {BACKENDS + ('auto',)}, "
                             f"got {backend!r}")

        want_tb = backend in ("tensorboard", "all")
        want_mf = backend in ("mlflow", "all")
        if backend == "auto":
            # Use whatever is installed; never fail, because the user did not
            # ask for anything specific.
            want_tb = _available("tensorboard")
            want_mf = _available("mlflow")

        self.active = []
        if want_tb:
            self._tb = self._start_tensorboard(required=(backend != "auto"))
        if want_mf:
            self._mlflow = self._start_mlflow(required=(backend != "auto"))

    # ----------------------------------------------------------------- setup
    def _start_tensorboard(self, required):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            if required:
                raise SystemExit(_missing_package_error("tensorboard"))
            return None
        w = SummaryWriter(str(self.out_dir / "tb"))
        self.active.append("tensorboard")
        return w

    def _start_mlflow(self, required):
        try:
            import mlflow
        except ImportError:
            if required:
                raise SystemExit(_missing_package_error("mlflow"))
            return None
        try:
            return self._start_mlflow_inner(mlflow)
        except SystemExit:
            raise
        except Exception as e:
            # Anything else - an unreadable store, a locked database, a backend
            # the installed mlflow refuses - must not take the training run
            # with it. Only a MISSING explicitly-requested backend is fatal;
            # a broken one degrades to a warning, because losing three hours of
            # training to a charting library is never the right trade.
            print(f"  [warn] mlflow disabled: {type(e).__name__}: {e}")
            return None

    def _start_mlflow_inner(self, mlflow):
        # SQLite, NOT the bare './mlruns' file store - see mlflow_store_uri.
        uri, artifacts = mlflow_store_uri(self.out_dir, create=True)
        mlflow.set_tracking_uri(uri)

        # With a database backend the artifact root is NOT implied by the
        # tracking URI; left unset it resolves to ./mlruns relative to the
        # working directory, so preview images would land wherever you
        # happened to run python from.
        exp = mlflow.get_experiment_by_name(self.experiment)
        if exp is None and artifacts is not None:
            mlflow.create_experiment(self.experiment,
                                     artifact_location=artifacts.as_uri())
        mlflow.set_experiment(self.experiment)

        # log_system_metrics samples GPU/CPU/RAM/disk during the run, which is
        # what distinguishes "the model is slow" from "the dataloader is
        # starving the GPU" - the two have completely different fixes and are
        # indistinguishable from a loss curve.
        #
        # It is an ENHANCEMENT, so its failure must cost only itself. The
        # dependency is CHECKED UP FRONT rather than caught: mlflow creates the
        # run in the store and only then starts the metrics monitor, so a
        # failure there leaves an orphan run that mlflow.active_run() no longer
        # reports and nothing can clean up - an empty row in exactly the
        # comparison table MLflow is here for. Not asking for what cannot work
        # is the only way to avoid it.
        want_sys = _available("psutil")
        if not want_sys:
            print("  [warn] mlflow system metrics off (psutil missing). "
                  "Everything else is still logged; `python -m pip install "
                  "psutil pynvml` enables GPU/CPU/RAM sampling.")
        try:
            mlflow.start_run(run_name=self.run_name,
                             log_system_metrics=True if want_sys else None)
            self.system_metrics = want_sys
        except TypeError:
            # An mlflow too old to know the argument at all.
            mlflow.start_run(run_name=self.run_name)
            self.system_metrics = False
        self._uri = uri
        self.active.append("mlflow")
        return mlflow

    # ---------------------------------------------------------------- logging
    def log_params(self, params):
        flat = _flatten(dict(params))
        (self.out_dir / "params.json").write_text(
            json.dumps(flat, indent=2, default=str), encoding="utf-8")
        if self._mlflow:
            # Params are immutable in MLflow; a resumed run would otherwise
            # abort the whole training on a duplicate key.
            try:
                self._mlflow.log_params(flat)
            except Exception:
                pass
        if self._tb:
            text = "\n".join(f"{k} = {v}" for k, v in sorted(flat.items()))
            self._tb.add_text("params", "```\n" + text + "\n```", 0)

    def log_metrics(self, metrics, step=None):
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and v == v}   # drop None/NaN
        if self._tb:
            for k, v in clean.items():
                self._tb.add_scalar(k, v, step)
        if self._mlflow:
            self._mlflow.log_metrics(clean, step=step)

    def log_image(self, tag, rgb, step=None):
        """`rgb` is an (H,W,3) uint8 array in RGB order.

        Prediction previews matter more than any scalar on a 45-frame dataset:
        a loss curve cannot tell you that every mask is one plant too large,
        and looking at eight overlays for ten seconds can."""
        import numpy as np
        a = np.asarray(rgb)
        if self._tb:
            self._tb.add_image(tag, a.transpose(2, 0, 1), step,
                               dataformats="CHW")
        if self._mlflow:
            try:
                self._mlflow.log_image(a, f"{tag}_{step or 0:04d}.png")
            except Exception:
                pass

    def log_artifact(self, path):
        p = Path(path)
        if self._mlflow and p.exists():
            self._mlflow.log_artifact(str(p))

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._tb:
            self._tb.flush(); self._tb.close()
        if self._mlflow:
            try:
                self._mlflow.end_run()
            except Exception:
                pass

    def hint(self):
        """What to run to actually look at the results."""
        lines = []
        if "tensorboard" in self.active:
            lines.append(f"  tensorboard --logdir {self.out_dir / 'tb'}")
        if "mlflow" in self.active:
            # The exact URI the run was written to. Printing the DIRECTORY
            # instead would send you to a store mlflow 3 refuses to open.
            lines.append(f"  mlflow ui --backend-store-uri {self._uri}")
        if not lines:
            return ("tracking: none active (pip install tensorboard mlflow "
                    "to get curves and a run-comparison table)")
        return "tracking active:\n" + "\n".join(lines)


def git_commit(repo_root=None):
    """Current commit, with '-dirty' when the tree has uncommitted changes.

    The dirty flag is the important half: a commit hash alone claims a run is
    reproducible, and if the working tree differed from that commit it is
    not."""
    import subprocess
    root = str(repo_root or Path(__file__).resolve().parents[2])
    def _run(args):
        return subprocess.run(args, cwd=root, capture_output=True, text=True,
                              timeout=10)
    try:
        r = _run(["git", "rev-parse", "HEAD"])
        if r.returncode != 0:
            return None
        sha = r.stdout.strip()
        d = _run(["git", "status", "--porcelain"])
        if d.returncode == 0 and d.stdout.strip():
            sha += "-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return None


def file_digest(path, chunk=1 << 20):
    """SHA-256 of a file, or None. Used on the dataset manifest so a run is
    tied to the EXACT dataset it saw - a path plus a date is not enough, since
    rebuilding into the same OUT_DIR silently changes what a path refers to."""
    import hashlib
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()[:16]


def environment_params(dataset_dir=None):
    """Versions, hardware and code state - everything needed to explain a
    number months later that is not a hyperparameter.

    Every field is best-effort: a missing CUDA build or no git binary must
    never stop a training run."""
    import platform
    out = {
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = torch.version.cuda
        out["cudnn_version"] = (torch.backends.cudnn.version()
                                if torch.backends.cudnn.is_available() else None)
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            out["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    except Exception:
        pass
    try:
        import torchvision
        out["torchvision_version"] = torchvision.__version__
    except Exception:
        pass
    if dataset_dir:
        d = Path(dataset_dir)
        out["dataset_dir"] = str(d)
        out["seg_manifest_sha256"] = file_digest(d / "seg_manifest.json")
        out["class_mapping_sha256"] = file_digest(d / "class_mapping.json")
    return {k: v for k, v in out.items() if v is not None}


def _missing_package_error(package):
    """`pip install X` is not actionable on its own when several interpreters
    are on the machine.

    SeeWeed3D deliberately uses two conda environments - `dl` pins numpy<2 for
    SAM 3, `sw-train` carries the training stack - so "not installed" almost
    always means "you ran the other one", and the fix is to switch interpreter,
    NOT to install into whichever one happens to be current. Installing the
    training stack into `dl` is how SAM 3's numpy pin gets broken.

    Naming sys.executable makes the mismatch visible immediately, because a
    shell prompt showing (sw-train) says nothing about which python an
    explicit path or an IDE's configured interpreter actually launched."""
    import sys as _sys
    exe = Path(_sys.executable)
    env = exe.parent.name if exe.parent.name != "bin" else exe.parents[1].name
    return (
        f"ERROR: {package} is not installed in the interpreter that is "
        f"running:\n"
        f"    {exe}\n"
        f"    (environment: {env})\n\n"
        f"SeeWeed3D uses TWO environments on purpose:\n"
        f"    dl        data pipeline + SAM 3   (numpy<2 pin, do NOT add the "
        f"training stack here)\n"
        f"    sw-train  training + deployment   (torch, mlflow, tensorboard)\n\n"
        f"If the above is not your training environment, run with that one "
        f"instead - in VS Code, Ctrl+Shift+P -> 'Python: Select Interpreter'.\n"
        f"If it IS the right one, install there:\n"
        f"    \"{exe}\" -m pip install -r requirements-training.txt")


def _available(mod):
    import importlib.util
    if mod == "tensorboard":
        return importlib.util.find_spec("torch.utils.tensorboard") is not None \
            and importlib.util.find_spec("tensorboard") is not None
    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- #
# prediction previews
# --------------------------------------------------------------------------- #
CROP_BGR = (0, 140, 255)          # orange - the crop, drawn thicker
WEED_BGR = (60, 220, 60)          # green


def overlay_masks(bgr, masks, names, crop_class, alpha=0.45):
    """Tint each instance and outline it. Crop instances are outlined thicker
    because a missed crop is the failure that matters."""
    import cv2
    import numpy as np
    out = bgr.copy()
    for m, n in zip(masks, names):
        m = np.asarray(m).astype(bool)
        if not m.any():
            continue
        colour = CROP_BGR if n == crop_class else WEED_BGR
        tint = np.zeros_like(out); tint[m] = colour
        out = cv2.addWeighted(out, 1.0, tint, alpha, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colour,
                         3 if n == crop_class else 1)
    return out


def side_by_side(gt_bgr, pred_bgr, max_width=1400):
    """Ground truth left, prediction right, with a divider.

    Two panels rather than one overlay of both: overlapping tints of a correct
    and an incorrect mask are indistinguishable from a single mask of a third
    colour, which is exactly the case you are looking for."""
    import cv2
    import numpy as np
    h = min(gt_bgr.shape[0], pred_bgr.shape[0])
    g = gt_bgr[:h]; p = pred_bgr[:h]
    pair = np.hstack([g, np.full((h, 4, 3), 255, np.uint8), p])
    if pair.shape[1] > max_width:
        s = max_width / pair.shape[1]
        pair = cv2.resize(pair, (max_width, int(pair.shape[0] * s)))
    cv2.putText(pair, "GT", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    cv2.putText(pair, "PRED", (pair.shape[1] // 2 + 8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return pair
