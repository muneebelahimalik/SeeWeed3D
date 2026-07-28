#!/usr/bin/env python3
"""
SeeWeed3D - lightweight progress reporting.

Dependency-free on purpose: these scripts already need ffmpeg, torch and SAM 3,
and a progress bar is not worth another install. Writes a single self-updating
line to the terminal and degrades to periodic plain lines when output is
redirected to a file, so logs stay readable.
"""

import shutil
import sys
import time


def _hms(seconds):
    if seconds is None or seconds != seconds or seconds in (float("inf"), -float("inf")):
        return "--:--"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    """Progress line with rate and ETA.

        p = Progress(len(frames), "vid3_...", unit="frames")
        for f in frames:
            ...
            p.update(note=f"{n} instances")
        p.close()
    """

    def __init__(self, total, label="", unit="frames", stream=None,
                 min_interval=0.2, file_interval=15.0):
        self.total = int(total) if total else 0
        self.label = label
        self.unit = unit
        self.stream = stream or sys.stdout
        self.min_interval = min_interval
        self.file_interval = file_interval
        self.n = 0
        self.t0 = time.monotonic()
        self._last = 0.0
        self._len = 0
        self._rendered_n = None
        # A self-updating line only makes sense on a terminal; when redirected,
        # emit occasional full lines instead of thousands of \r fragments.
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def update(self, step=1, note=""):
        self.n += step
        now = time.monotonic()
        interval = self.min_interval if self.tty else self.file_interval
        if now - self._last < interval and self.n < self.total:
            return
        self._last = now
        self._render(note)

    def _render(self, note=""):
        self._rendered_n = self.n
        elapsed = time.monotonic() - self.t0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if (rate > 0 and self.total) else None
        pct = (100.0 * self.n / self.total) if self.total else 0.0

        head = f"  {self.label} " if self.label else "  "
        body = (f"{self.n}/{self.total} {self.unit} {pct:5.1f}% | "
                f"{rate:.2f} {self.unit}/s | elapsed {_hms(elapsed)} | "
                f"ETA {_hms(eta)}")
        if note:
            body += f" | {note}"
        line = head + body

        if self.tty:
            width = shutil.get_terminal_size((100, 20)).columns - 1
            line = line[:width]
            pad = " " * max(0, self._len - len(line))
            self._len = len(line)
            self.stream.write("\r" + line + pad)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self, note=""):
        """Finish the line so following output is not overwritten.

        Re-renders only if the last update did not already show the final
        count, so a redirected log does not get a duplicated final line."""
        if self._rendered_n != self.n or note:
            self._render(note)
        if self.tty:
            self.stream.write("\n")
        self.stream.flush()
        return time.monotonic() - self.t0
