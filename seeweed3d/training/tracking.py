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
                raise SystemExit(
                    "ERROR: --track tensorboard requested but tensorboard is "
                    "not installed:\n    python -m pip install tensorboard")
            return None
        w = SummaryWriter(str(self.out_dir / "tb"))
        self.active.append("tensorboard")
        return w

    def _start_mlflow(self, required):
        try:
            import mlflow
        except ImportError:
            if required:
                raise SystemExit(
                    "ERROR: --track mlflow requested but mlflow is not "
                    "installed:\n    python -m pip install mlflow")
            return None
        # A local file store under the run's parent, so `mlflow ui` in that
        # folder sees every run of this project and nothing is uploaded.
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            store = (self.out_dir.parent / "mlruns").resolve()
            store.mkdir(parents=True, exist_ok=True)
            uri = store.as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=self.run_name)
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
            lines.append(f"  mlflow ui --backend-store-uri "
                         f"{(self.out_dir.parent / 'mlruns').resolve()}")
        if not lines:
            return ("tracking: none active (pip install tensorboard mlflow "
                    "to get curves and a run-comparison table)")
        return "tracking active:\n" + "\n".join(lines)


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
