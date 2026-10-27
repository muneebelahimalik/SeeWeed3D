#!/usr/bin/env python3
"""
SeeWeed3D - what happened to a recording that will not extract?
==============================================================
    python -m seeweed3d.validation.inspect_broken_recording \
        --folder "E:/Dataset_Vidalia/Data_15_16_October_2026/Session_20261016_104209"

    # or sweep a whole visit and report only the broken ones:
    python -m seeweed3d.validation.inspect_broken_recording \
        --root "E:/Dataset_Vidalia/Data_15_16_October_2026"

WHY THIS EXISTS
---------------
extract_sessions.py reports a session it cannot probe and moves on, which is
right - it must not guess at a malformed container. But ffprobe's own message
is a wall of EBML parser complaints that cannot distinguish the two cases that
matter:

  A FALSE START - header written, no frame ever arrived. Nothing to recover.
  AN UNFINALISED RECORDING - every frame is on disk, only the index is missing.
    A remux gets all of it back, losslessly.

Both print the same "Duplicate element / invalid as first byte of an EBML
number" noise. Guessing wrong in one direction throws away a good drive;
guessing wrong in the other sends you chasing frames that were never recorded.

This reads a few bytes per file and reports the verdict. It writes nothing and
changes nothing, including on the files it is pointed at.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.mkv_integrity import diagnose_session  # noqa: E402

VERDICT_ORDER = ["unfinalised", "false_start", "readable", "no_video"]


def session_folders(root):
    """Any folder holding a video file, the root itself included."""
    root = Path(root)
    out = []
    for folder in [root] + sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(p.suffix.lower() in (".mkv", ".avi", ".mp4", ".mov")
               for p in folder.iterdir() if p.is_file()):
            out.append(folder)
    return out


def report(d, verbose=True):
    print(f"\n{d['folder']}")
    print(f"  verdict : {d['verdict']}")
    if verbose:
        for f in d["files"]:
            print(f"  {f['path'].name:28s} {f['detail']}")
        if d["frames_csv_rows"] is not None:
            extra = ("" if d["frames_csv_written"] is None
                     else f", {d['frames_csv_written']} written to MKV")
            print(f"  {'frames.csv':28s} {d['frames_csv_rows']} rows{extra}")
        else:
            print(f"  {'frames.csv':28s} absent")
        if d["svo_files"]:
            mb = d["svo_bytes"] / 1e6
            print(f"  {'SVO archive':28s} "
                  f"{len(d['svo_files'])} file(s), {mb:.1f} MB")
        else:
            print(f"  {'SVO archive':28s} none")
    print(f"  -> {d['recovery']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", help="one recording folder")
    g.add_argument("--root", help="a visit folder; every session under it is checked")
    ap.add_argument("--all", action="store_true",
                    help="with --root, also report the sessions that are fine")
    a = ap.parse_args()

    if a.folder:
        report(diagnose_session(a.folder))
        return

    folders = session_folders(a.root)
    if not folders:
        sys.exit(f"No video files under {a.root}")
    results = [diagnose_session(f) for f in folders]
    shown = [d for d in results if a.all or d["verdict"] != "readable"]
    for d in sorted(shown, key=lambda d: (VERDICT_ORDER.index(d["verdict"])
                                          if d["verdict"] in VERDICT_ORDER
                                          else 9, str(d["folder"]))):
        report(d)

    counts = {}
    for d in results:
        counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1
    print(f"\n{len(results)} session folder(s): "
          + ", ".join(f"{n} {v}" for v, n in sorted(counts.items())))
    if counts.get("unfinalised"):
        print("\nTo repair the unfinalised ones in place (no re-encode, so the "
              "pixels are untouched), for each broken file:")
        print("  ffmpeg -err_detect ignore_err -i Broken.mkv -c copy Fixed.mkv")
        print("then re-run extraction with OVERWRITE=True.")


if __name__ == "__main__":
    main()
