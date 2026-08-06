#!/usr/bin/env python3
"""
SeeWeed3D - session-safe dataset splits.

THE FAILURE THIS PREVENTS
-------------------------
Adjacent video frames are near-identical. Splitting by FRAME puts a frame and
its near-duplicate on both sides of the train/test boundary, so the test set
measures memorisation and reports an accuracy the field will not reproduce.
The only safe unit is the whole session, and this module is the code that
enforces it rather than a convention someone has to remember.

Assignment is deterministic given a seed, explicit holdouts always win, and
when session metadata is available the allocator prefers to put *different*
dates / fields / cameras in validation and test - a val split from the same
morning as training measures far less than one from a different day.
"""

from __future__ import annotations

import json
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLITS = ("train", "val", "test")


class SplitError(ValueError):
    """A split cannot be produced safely."""


@dataclass
class SessionInfo:
    """What the allocator knows about one session."""
    session_id: str
    date: str = ""
    field_id: str = ""
    camera: str = ""
    scene: str = ""
    n_frames: int = 0
    class_counts: dict = field(default_factory=dict)

    def group_key(self):
        """Sessions sharing this key are near-duplicates of each other for
        generalisation purposes: same day, same field, same camera."""
        return (self.date, self.field_id, self.camera)


def _stable_rank(session_id, seed):
    """Deterministic pseudo-random ordering key.

    zlib.crc32, not hash(): Python's str hash is salted per process by
    PYTHONHASHSEED, so hash() would silently produce a DIFFERENT split on every
    run and destroy reproducibility."""
    return zlib.crc32(f"{seed}:{session_id}".encode()) / 0xFFFFFFFF


def assign_splits(sessions, val_fraction=0.2, test_fraction=0.2, seed=1234,
                  holdout_val=(), holdout_test=(), holdout_train=()):
    """Assign every session to exactly one split.

    sessions: [SessionInfo] or [str]. Returns {split: [session_id]}.

    Explicit holdouts are honoured first and unconditionally. The remainder is
    allocated by a deterministic rank, walking whole METADATA GROUPS (same
    date+field+camera) so two recordings of the same bed on the same morning
    cannot be separated - they are near-duplicates and would leak just as a
    frame split does."""
    infos = [SessionInfo(session_id=s) if isinstance(s, str) else s
             for s in sessions]
    if not infos:
        raise SplitError("no sessions supplied.")

    ids = [i.session_id for i in infos]
    dup = [s for s, n in Counter(ids).items() if n > 1]
    if dup:
        raise SplitError(f"duplicate session ids: {sorted(dup)}")

    forced = {}
    for split, names in (("val", holdout_val), ("test", holdout_test),
                         ("train", holdout_train)):
        for name in names or ():
            if name in forced and forced[name] != split:
                raise SplitError(
                    f"session {name!r} is pinned to both {forced[name]} and "
                    f"{split}. A session belongs to exactly one split.")
            if name not in ids:
                raise SplitError(
                    f"holdout session {name!r} is not in the dataset. Known: "
                    f"{sorted(ids)}")
            forced[name] = split

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise SplitError(
            f"val_fraction + test_fraction must be < 1.0 (got "
            f"{val_fraction} + {test_fraction}); training would be empty.")

    out = {s: [] for s in SPLITS}
    for name, split in forced.items():
        out[split].append(name)

    free = [i for i in infos if i.session_id not in forced]

    # Group ONLY on positive evidence of relatedness. Two sessions from the same
    # date+field+camera are near-duplicates and must not be separated. Sessions
    # with no metadata carry no such evidence, so each is its own group -
    # bucketing them together on a shared empty key would put every session in
    # one group and hand the whole dataset to a single split.
    groups = defaultdict(list)
    for i in free:
        key = i.group_key()
        groups[key if any(key) else ("__solo__", i.session_id, "")].append(
            i.session_id)

    ordered = sorted(groups.items(),
                     key=lambda kv: _stable_rank(sorted(kv[1])[0], seed))

    total_free = len(free)
    want_test = int(round(test_fraction * total_free))
    want_val = int(round(val_fraction * total_free))
    n_test = n_val = 0
    n_groups = len(ordered)

    for gi, (_, members) in enumerate(ordered):
        members = sorted(members)
        remaining_after = n_groups - gi - 1
        # A whole group is indivisible, so a few large groups can consume the
        # val/test quotas and leave training empty. Reserve the last group for
        # training when nothing has reached it yet - a dataset with no training
        # data is never the intended outcome of a fraction.
        must_train = (not out["train"]) and remaining_after == 0

        if must_train:
            out["train"].extend(members)
        elif n_test < want_test:
            out["test"].extend(members)
            n_test += len(members)
        elif n_val < want_val:
            out["val"].extend(members)
            n_val += len(members)
        else:
            out["train"].extend(members)

    for s in SPLITS:
        out[s] = sorted(out[s])

    seen = Counter(sum(out.values(), []))
    leaked = [k for k, v in seen.items() if v > 1]
    if leaked:
        raise SplitError(f"internal error: sessions in several splits: {leaked}")
    if not out["train"]:
        raise SplitError(
            "the training split is empty. Lower val_fraction/test_fraction, or "
            "record more sessions - with very few sessions a session-safe split "
            "may be impossible.")
    return out


def assign_frame_blocks(frame_ids, val_fraction=0.2, test_fraction=0.2,
                        gap_frames=2):
    """Split ONE session into contiguous frame blocks, with a discarded buffer.

    USE ONLY WHEN A SESSION-LEVEL SPLIT IS IMPOSSIBLE (a single recording). It
    is strictly weaker than splitting by session and the caller must say so.

    Why blocks and not a random frame split: adjacent video frames are
    near-identical, so a random split puts a frame and its near-duplicate on
    both sides of the boundary and the score measures memorisation. Contiguous
    blocks put train, val and test in *different parts of the drive*, so at
    least they see different ground.

    Why a gap: the frames either side of a block boundary still overlap
    heavily. `gap_frames` at each boundary are dropped from every split rather
    than assigned, which is the cheapest way to buy real separation.

    What this still cannot give you: the val/test frames come from the same
    recording, so the same lighting, the same soil, the same plants at the same
    growth stage, and often the same individual plants seen again. Treat the
    resulting numbers as a sanity check that training is working - NOT as
    evidence the model generalises. Only a held-out SESSION can support that.

    Returns {split: [frame_id]} preserving the input order.
    """
    ids = list(frame_ids)
    n = len(ids)
    if n < 5:
        raise SplitError(
            f"only {n} frames; a within-session split needs at least 5 to leave "
            f"anything in each block. Annotate more frames.")
    if val_fraction + test_fraction >= 1.0:
        raise SplitError("val_fraction + test_fraction must be < 1.0")

    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    gap = max(0, int(gap_frames))

    # Layout: [ train | gap | val | gap | test ]
    # Test last so it sits at one end of the drive, furthest from training.
    need = n_val + n_test + 2 * gap
    if need >= n:
        gap = 0                                    # too small to afford buffers
        need = n_val + n_test
    if need >= n:
        raise SplitError(
            f"{n} frames cannot be split into train/val/test at "
            f"val={val_fraction}, test={test_fraction}. Lower the fractions or "
            f"annotate more frames.")

    n_train = n - need
    i = 0
    train = ids[i:i + n_train]; i += n_train + gap
    val = ids[i:i + n_val]; i += n_val + gap
    test = ids[i:i + n_test]
    return {"train": train, "val": val, "test": test,
            "_dropped_gap": [f for f in ids
                             if f not in set(train) | set(val) | set(test)]}


def check_no_leakage(split_map, frame_sessions):
    """Raise if any session, or any frame, appears in more than one split.

    frame_sessions: {frame_key: session_id}. This is the assertion that makes
    the guarantee testable rather than assumed."""
    owner = {}
    for split, sessions in split_map.items():
        for s in sessions:
            if s in owner:
                raise SplitError(
                    f"session {s!r} is in both {owner[s]} and {split}.")
            owner[s] = split

    unknown = sorted({s for s in frame_sessions.values() if s not in owner})
    if unknown:
        raise SplitError(
            f"frames reference sessions with no split: {unknown}. Every frame "
            f"must resolve to a session, otherwise it cannot be placed safely.")

    frames_per_split = defaultdict(set)
    for frame, session in frame_sessions.items():
        frames_per_split[owner[session]].add(frame)
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            both = frames_per_split[a] & frames_per_split[b]
            if both:
                raise SplitError(
                    f"{len(both)} frames appear in both {a} and {b}: "
                    f"{sorted(both)[:5]}")
    return True


def write_splits(out_dir, split_map, frames_by_session=None, sessions=None):
    """Write the split files, per-split image manifests and a JSON summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {i.session_id: i for i in (sessions or [])
            if isinstance(i, SessionInfo)}

    for split in SPLITS:
        (out / f"{split}_sessions.txt").write_text(
            "\n".join(split_map.get(split, [])) + "\n", encoding="utf-8")

    manifests = {}
    if frames_by_session:
        for split in SPLITS:
            paths = []
            for s in split_map.get(split, []):
                paths.extend(frames_by_session.get(s, []))
            # Posix separators so a manifest written on Windows is readable on
            # Linux and vice versa; consumers resolve against their own root.
            body = "\n".join(Path(p).as_posix() for p in paths)
            (out / f"{split}_images.txt").write_text(body + "\n" if body else "",
                                                     encoding="utf-8")
            manifests[split] = len(paths)

    summary = {"seed_note": "assignment is deterministic for a given seed",
               "splits": {}}
    for split in SPLITS:
        rows, classes = [], Counter()
        for s in split_map.get(split, []):
            i = info.get(s)
            rows.append({"session_id": s,
                         "date": i.date if i else "",
                         "field": i.field_id if i else "",
                         "camera": i.camera if i else "",
                         "scene": i.scene if i else "",
                         "n_frames": i.n_frames if i else
                         len((frames_by_session or {}).get(s, []))})
            if i:
                classes.update(i.class_counts or {})
        summary["splits"][split] = {
            "n_sessions": len(rows),
            "n_frames": manifests.get(split, sum(r["n_frames"] for r in rows)),
            "sessions": rows,
            "class_counts": dict(classes)}
    (out / "splits_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    return summary
