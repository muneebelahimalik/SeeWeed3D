#!/usr/bin/env python3
"""
SeeWeed3D - what a session IS, read from the session itself.

`extract_sessions.py` writes `<session>/meta/session.json` recording the trip,
site, field, capture format and - the one that matters most for training - the
`scene_hint`: whether that drive is onion_only, weed_only, mixed, or unknown.

Until this module existed nothing downstream read it. The split allocator has
carried a `scene` field since it was written and it was always empty, so a
dataset spanning onion-only, weed-only and mixed drives was split as if every
session were interchangeable. Two things follow from that, both silent:

  * A validation set can end up with no mixed scene in it at all. Mixed is the
    only scene where the crop-vs-weed decision is actually exercised, so a
    model can look excellent on val while never having been measured on the
    question the machine exists to answer.
  * A test set can end up all-onion. Weed recall then goes unmeasured, and
    weed recall is the number that decides whether a weed survives the pass.

The date/field/camera fields matter for a second reason. Two drives of the same
bed on the same morning are near-duplicates of each other; separating them puts
what is nearly the same ground on both sides of the train/test line. The
allocator already knows how to keep such sessions together - it just needs to
be told which sessions those are.

WHERE EACH FIELD COMES FROM
---------------------------
    scene     session.json `scene_hint`, normalised
    field_id  session.json `field`
    trip/site  session.json `trip` / `site`
    date      the session id, `<cam>_<YYYYMMDD>_<HHMMSS>` - this is the capture
              date, whereas session.json's `extracted_utc` is the date somebody
              ran the extractor, which is not the same thing and can differ by
              months
    camera    the session id prefix (`vid1`, `vid3`, ...), which is how the
              capture rig names its recorders

A session with no `meta/session.json` is not an error. Older extractions
predate it, and a session that carries no evidence about itself must not be
grouped with other sessions that carry none - "both unknown" is not evidence of
relatedness. Such a session gets empty metadata and the allocator treats it as
its own group.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

#: The scene values the extractor documents. Anything else is normalised to
#: "unknown" rather than silently becoming its own stratum - a typo like
#: "onions" would otherwise split the allocator's quota across a stratum of one
#: and be invisible in the report.
SCENES = ("onion_only", "weed_only", "mixed", "unknown")

#: Spellings seen in real trip configs, mapped to the documented value. Add to
#: this rather than loosening the check: an unrecognised scene should be
#: reported, because the cost of it is a split that silently fails to measure
#: what you think it measures.
SCENE_ALIASES = {
    "onion": "onion_only", "onions": "onion_only", "onlyonions": "onion_only",
    "onion_only": "onion_only", "crop_only": "onion_only",
    "weed": "weed_only", "weeds": "weed_only", "weed_only": "weed_only",
    "mixed": "mixed", "mix": "mixed", "both": "mixed",
    "": "unknown", "unknown": "unknown",
}

_SESSION_RE = re.compile(r"^([A-Za-z0-9]+)_(\d{8})_(\d{6})$")


def normalise_scene(value):
    """A documented scene value, or 'unknown'. Never raises - a bad hint is a
    metadata problem, not a reason to fail a dataset build."""
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return SCENE_ALIASES.get(key, "unknown")


def parse_session_id(session_id):
    """(camera, date) from `<cam>_<YYYYMMDD>_<HHMMSS>`, or ('', '').

    Anything that does not match returns empties rather than a guess: a
    half-parsed date would group unrelated sessions, which is the exact failure
    this metadata exists to prevent."""
    m = _SESSION_RE.match(str(session_id).strip())
    if not m:
        return "", ""
    cam, ymd, _ = m.groups()
    return cam, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"


def read_session_meta(sessions_root, session_id):
    """Metadata for one session as a plain dict, always with the same keys.

    Missing file, unreadable JSON and missing keys all degrade to empty values.
    A dataset build must not fail because one old session predates the
    extractor's metadata; it must only avoid *inventing* relatedness."""
    camera, date = parse_session_id(session_id)
    out = {"session_id": session_id, "scene": "unknown", "field_id": "",
           "camera": camera, "date": date, "trip": "", "site": "",
           "capture_format": "", "has_meta": False}
    if not sessions_root:
        return out
    path = Path(sessions_root) / session_id / "meta" / "session.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(raw, dict):
        return out
    out["has_meta"] = True
    out["scene"] = normalise_scene(raw.get("scene_hint"))
    out["field_id"] = str(raw.get("field") or "")
    out["trip"] = str(raw.get("trip") or "")
    out["site"] = str(raw.get("site") or "")
    out["capture_format"] = str(raw.get("capture_format") or "")
    return out


def find_session_meta(sessions_roots, session_id):
    """First root that actually has this session's metadata.

    A build can merge CVAT exports whose images live under different sessions
    roots, so the session is looked for in each. The first root carrying real
    metadata wins; if none does, the id-derived fields are still returned."""
    roots = [r for r in (sessions_roots or []) if r]
    fallback = read_session_meta(None, session_id)
    for root in roots:
        meta = read_session_meta(root, session_id)
        if meta["has_meta"]:
            return meta
        fallback = meta if meta["camera"] else fallback
    return fallback


def unknown_scene_report(metas):
    """Sessions whose scene could not be determined, for a caller to print.

    Worth surfacing every time rather than only on failure: an unknown scene
    does not break a split, it just removes that session from stratification,
    and a quiet 'we could not tell' is how a val set ends up with no mixed
    scene in it."""
    unknown = sorted(m["session_id"] for m in metas
                     if m.get("scene", "unknown") == "unknown")
    no_meta = sorted(m["session_id"] for m in metas if not m.get("has_meta"))
    return {"unknown_scene": unknown, "no_session_json": no_meta}
