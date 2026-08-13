#!/usr/bin/env python3
"""
SeeWeed3D - naming frames by index, in one syntax everywhere.

Several stages need to name a subset of a session's frames: curation drops bad
ones by hand, and a prelabeler may need to run over only part of a drive. They
should not each invent a spelling for it, so the tokens are parsed here.

ACCEPTED TOKENS
---------------
    "1187"                a single video frame index
    "0-250"               an INCLUSIVE index range
    "1500-"               from 1500 to the end of the session
    "-250"                from the start of the session through 250
    "sess_001900.png"     a filename - the trailing number is the index
    "sess_001900.jpg"     a PREVIEW name works too

The open-ended forms exist because the common case is a drive that CHANGES at
one point - weeds up to some frame, onions after it. Naming that as "1500-"
does not require you to first go and look up the last index in the session,
and it cannot silently truncate the tail if you guess that number too low.

The filename forms matter because previews are what you actually look at when
deciding a frame is bad or where a drive changes crop, and a preview is a .jpg
beside a .png source. Requiring the exact source name would mean transcribing
indices by hand from the thing you are looking at.

WHY INDEX AND NOT POSITION
--------------------------
`<session_id>_<video_frame_idx:06d>.png` is the join key across rgb/, depth/,
right/ and conf/, and the number is the frame's index IN THE SOURCE VIDEO.
Gaps in it are meaningful - they mean "this frame exists in the video but is
not in the pool" - so counting positions in the pool would drift away from the
numbers you can actually read off a filename or a preview.
"""
from pathlib import Path


def parse_frame_tokens(tokens):
    """(indices, ranges) from a list of tokens. Unreadable tokens are reported
    and skipped rather than silently ignored, because a typo that selects
    nothing looks exactly like a range that legitimately matched nothing."""
    indices, ranges = set(), []
    for tok in tokens or []:
        t = str(tok).strip()
        if not t:
            continue
        if t.isdigit():
            indices.add(int(t))
            continue
        lo_hi = [p.strip() for p in t.split("-")]
        if len(lo_hi) == 2 and any(lo_hi) and all(p.isdigit() or not p
                                                  for p in lo_hi):
            # An omitted end is open, not zero: "1500-" runs to the end of the
            # session, "-250" from its start. Both ends omitted would be a
            # spec that selects everything, which is what leaving it unset
            # already means, so `any(lo_hi)` rejects a bare "-" as a typo.
            lo = int(lo_hi[0]) if lo_hi[0] else 0
            hi = int(lo_hi[1]) if lo_hi[1] else float("inf")
            ranges.append((min(lo, hi), max(lo, hi)))
            continue
        tail = Path(t).stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            indices.add(int(tail))
        else:
            print(f"      [WARN] cannot interpret frame token {tok!r} - ignored")
    return indices, ranges


def matches(idx, indices, ranges):
    """Is this frame index named by the parsed spec?"""
    return idx in indices or any(lo <= idx <= hi for lo, hi in ranges)


def index_of(filename):
    """Video frame index encoded in a frame filename, or -1.

    Accepts the source .png or its .jpg preview - both end in the index."""
    tail = Path(str(filename)).stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def select_filenames(filenames, tokens):
    """Keep only the frames a spec names, preserving order.

    An empty or absent spec keeps EVERYTHING - so the option is inert until
    somebody sets it, and a stage that never uses it behaves exactly as before.

    A spec that matches nothing is a caller-visible empty list rather than a
    silent full pass: selecting no frames is a real answer, and the caller is
    better placed to say so with its session id in the message."""
    indices, ranges = parse_frame_tokens(tokens)
    if not indices and not ranges:
        return list(filenames)
    return [f for f in filenames
            if (i := index_of(f)) >= 0 and matches(i, indices, ranges)]
