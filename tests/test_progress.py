"""Progress reporting must never be able to break a long pipeline run, and must
stay readable when output is redirected to a log file."""
import io

from conftest import load_script

prog = load_script("common/progress.py")


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_reports_counts_rate_and_eta_on_a_terminal():
    s = _Tty()
    p = prog.Progress(10, "[sess]", unit="frames", stream=s, min_interval=0.0)
    for _ in range(10):
        p.update(note="42 instances")
    p.close()
    out = s.getvalue()
    assert "10/10 frames" in out and "100.0%" in out
    assert "ETA" in out and "elapsed" in out and "42 instances" in out
    assert "\r" in out and out.endswith("\n")   # in-place line, then finished


def test_redirected_output_uses_plain_lines_not_carriage_returns():
    """A log file full of \\r fragments is unreadable; when not a tty the
    reporter emits occasional whole lines instead."""
    s = io.StringIO()                    # isatty() is False
    p = prog.Progress(5, "[sess]", stream=s, file_interval=0.0)
    for _ in range(5):
        p.update()
    p.close()
    out = s.getvalue()
    assert "\r" not in out
    assert out.count("\n") >= 2


def test_throttles_updates_so_it_cannot_dominate_runtime():
    s = _Tty()
    p = prog.Progress(1000, stream=s, min_interval=3600.0)   # effectively never
    for _ in range(999):
        p.update()
    assert s.getvalue() == "", "throttled updates should not render"
    p.update()                                                # final -> renders
    assert "1000/1000" in s.getvalue()


def test_survives_zero_total_and_unknown_eta():
    """Frame counts are not always known ahead of time (ffprobe may not report
    nb_frames), so a zero total must not divide by zero."""
    s = _Tty()
    p = prog.Progress(0, "[unknown]", stream=s, min_interval=0.0)
    p.update()
    p.close()
    assert "--:--" in s.getvalue() or "0/0" in s.getvalue()


def test_hms_formats_and_handles_undefined():
    assert prog._hms(0) == "00:00"
    assert prog._hms(65) == "01:05"
    assert prog._hms(3725) == "1:02:05"
    assert prog._hms(None) == "--:--"
    assert prog._hms(float("inf")) == "--:--"


def test_throttling_does_not_depend_on_how_long_the_machine_has_been_up():
    """time.monotonic() counts from an arbitrary origin - system boot on both
    Linux and Windows - so comparing `now - 0.0` against min_interval was
    really asking "has this machine been up longer than the interval". The
    first update rendered on a workstation up for an hour and stayed silent on
    a freshly-booted container: same code, same config, different output, and
    nothing on screen to say which you were getting.

    Simulated here by moving the clock's origin, which is what a different
    uptime does."""
    import time as _time

    seen = []
    for origin in (0.1, 5.0, 86400.0 * 7):        # fresh boot .. up a week
        s = _Tty()
        real = prog.time.monotonic
        prog.time.monotonic = lambda o=origin: o
        try:
            p = prog.Progress(1000, stream=s, min_interval=3600.0)
            p.update()
        finally:
            prog.time.monotonic = real
        seen.append(s.getvalue())

    assert all(v == seen[0] for v in seen), (
        f"throttling changed with the clock origin: {[bool(v) for v in seen]}")
    assert seen[0] == "", "an update inside the interval should not render"


def test_the_final_update_always_renders_however_throttled():
    """Otherwise a run that finishes inside one interval prints no total."""
    s = _Tty()
    p = prog.Progress(3, stream=s, min_interval=3600.0)
    for _ in range(3):
        p.update()
    assert "3/3" in s.getvalue()
