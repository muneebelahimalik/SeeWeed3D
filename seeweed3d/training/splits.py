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


def _quota_walk(ordered, val_fraction, test_fraction, out, guarantee_train,
                size_by_session=None):
    """Fill test, then val, then train from an ordered list of groups.

    BEST FIT, NOT FIRST FIT. Groups are indivisible, and the obvious loop -
    "add this group if the split is still under quota" - checks the quota
    BEFORE adding and so overshoots by the whole size of the group. Asking for
    20% test from groups of 1, 2 and 3 sessions put all three of the last group
    in test: 50% of the data, and its entire capture date absent from training.

    So a group is taken only if it FITS the remaining quota. When nothing fits,
    the split takes the smallest remaining group rather than the first one it
    happens to see - a split has to contain something, but it does not have to
    swallow whatever came first.

    Sizes are FRAMES when they are known. A 500-frame session and a 50-frame
    one are not interchangeable, and counting sessions makes a 20% split mean
    anything between 5% and 60% of the actual data. Falls back to counting
    sessions when no frame counts were supplied.

    `guarantee_train` reserves a group for training when nothing has reached it
    - a dataset with no training data is never the intended outcome."""
    units = [(sorted(m), _unit_size(m, size_by_session)) for _, m in ordered]
    total = sum(w for _, w in units)
    if not total:
        return
    want = {"test": test_fraction * total, "val": val_fraction * total}
    taken = {"test": 0.0, "val": 0.0}
    remaining = list(range(len(units)))

    for split in ("test", "val"):
        if want[split] <= 0:
            continue
        while remaining:
            room = want[split] - taken[split]
            if room <= 0:
                break
            fits = [i for i in remaining if units[i][1] <= room]
            if fits:
                i = fits[0]                     # keep the interleaved order
            elif taken[split] == 0:
                # Nothing fits and the split is empty: take the SMALLEST, not
                # the first. This is the whole difference between a test set of
                # 20% and one of 50%.
                i = min(remaining, key=lambda j: (units[j][1], units[j][0][0]))
            else:
                break
            taken[split] += units[i][1]
            out[split].extend(units[i][0])
            remaining.remove(i)

    for i in remaining:
        out["train"].extend(units[i][0])

    if guarantee_train and not out["train"]:
        # Every unit was consumed by val/test. Give the largest back rather
        # than shipping an empty training set.
        biggest = max((u for u in units), key=lambda u: (u[1], u[0][0]))
        for split in ("val", "test"):
            out[split] = [s for s in out[split] if s not in biggest[0]]
        out["train"].extend(biggest[0])


def _unit_size(members, size_by_session=None):
    """Frames in a group when known, else its session count."""
    if size_by_session:
        n = sum(int(size_by_session.get(s, 0)) for s in members)
        if n:
            return n
    return len(members)


def scene_of(members, scene_by_session):
    """One scene label for a whole group.

    A group is sessions of the same bed on the same morning, so they normally
    share a scene. When they do not, the group is called `mixed`: the group is
    indivisible, so whichever split receives it receives every scene in it, and
    claiming it as onion_only would overstate what the split contains."""
    seen = {scene_by_session.get(s, "unknown") for s in members}
    seen.discard("unknown")
    if not seen:
        return "unknown"
    return seen.pop() if len(seen) == 1 else "mixed"


def scene_representation(split_map, sessions):
    """{split: {scene: n_sessions}} plus the scenes missing from each split.

    This is the check the whole stratification exists for: a val set with no
    mixed scene in it never measures the crop-vs-weed decision, and a test set
    with no weed_only scene never measures weed recall."""
    scene_by = {i.session_id: (i.scene or "unknown")
                for i in sessions if isinstance(i, SessionInfo)}
    present = {s for s in scene_by.values() if s != "unknown"}
    counts, missing = {}, {}
    for split in SPLITS:
        c = Counter(scene_by.get(s, "unknown") for s in split_map.get(split, []))
        counts[split] = dict(sorted(c.items()))
        # Only meaningful for a split that received anything at all: an empty
        # test set is a separate, louder problem and is reported elsewhere.
        missing[split] = (sorted(present - set(c)) if split_map.get(split)
                          else [])
    return {"counts": counts, "missing": missing}


#: Split granularity, best first. The unit that is held out decides what a
#: validation number can possibly mean.
#:
#:   group        a whole date+field+camera. Val/test share NO conditions with
#:                train - different day, light, growth stage. The only split
#:                that estimates generalisation.
#:   session      a whole recording, from a date that training also saw. No
#:                frame-level leakage, but the conditions are shared, so the
#:                score is an upper bound on a new drive.
#:   frame_block  contiguous blocks WITHIN each session. Val/test come from the
#:                same recording as train; adjacent frames are near-duplicates.
#:                Measures fit, not generalisation.
GRANULARITIES = ("group", "session")


def _interleave(strata, seed):
    """One order that visits every stratum before repeating any.

    Round-robin rather than one quota walk per stratum. Independent per-stratum
    quotas inflate the small splits on small strata - three sessions at a 20%
    test quota rounds to one, which is 33% - and those errors compound across
    strata until the global ratio bears no relation to what was asked for.
    Interleaving keeps the global quota exact AND still hands the early
    positions (test, then val) to different strata.

    Strata are visited largest-first so a scene with many sessions absorbs the
    rounding a one-session stratum cannot."""
    ranked = {k: sorted(v, key=lambda kv: _stable_rank(sorted(kv[1])[0], seed))
              for k, v in strata.items()}
    order = sorted(ranked, key=lambda k: (-sum(len(m) for _, m in ranked[k]), k))
    out, i = [], 0
    while any(len(ranked[k]) > i for k in order):
        for k in order:
            if len(ranked[k]) > i:
                out.append(ranked[k][i])
        i += 1
    return out


def plan_splits(sessions, val_fraction=0.2, test_fraction=0.2, seed=1234,
                holdout_val=(), holdout_test=(), holdout_train=(),
                stratify_by_scene=True, granularity="auto"):
    """assign_splits, plus what it had to settle for. Returns (map, info).

    granularity "auto" tries `group` and falls back to `session` when a
    requested split comes out empty - which happens as soon as the dataset has
    few metadata groups. Six sessions recorded on two days are TWO groups, and
    two indivisible units cannot fill three splits: test ends up empty and
    training sees one date.

    Falling back to `session` is a real loss and is reported as one: val and
    test then share a date with training, so the score is an upper bound. It is
    still far better than frame blocks, which share the recording itself.

    info: {"granularity", "fell_back", "reason", "n_units"}."""
    want = str(granularity or "auto").lower()
    if want not in ("auto",) + GRANULARITIES:
        raise SplitError(
            f"granularity must be 'auto', 'group' or 'session'; got "
            f"{granularity!r}")

    tries = GRANULARITIES if want == "auto" else (want,)
    last = None
    for g in tries:
        out = assign_splits(sessions, val_fraction, test_fraction, seed,
                            holdout_val, holdout_test, holdout_train,
                            stratify_by_scene, _granularity=g)
        empty = [s for s, frac in (("val", val_fraction),
                                   ("test", test_fraction))
                 if frac > 0 and not out.get(s)]
        info = {"granularity": g, "fell_back": g != tries[0],
                "n_units": _n_units(sessions, g), "reason": None}
        if not empty:
            if info["fell_back"]:
                info["reason"] = (
                    "too few metadata groups to fill every split, so whole "
                    "SESSIONS were held out instead of whole days. Val and "
                    "test share a date with training: no frame leaks between "
                    "them, but the conditions are shared, so read the score as "
                    "an upper bound on a new drive.")
            return out, info
        last = (out, info, empty)
    out, info, empty = last
    info["reason"] = (f"even at {info['granularity']} granularity "
                      f"{', '.join(empty)} came out empty")
    return out, info


def _n_units(sessions, granularity):
    infos = [SessionInfo(session_id=s) if isinstance(s, str) else s
             for s in sessions]
    if granularity == "session":
        return len(infos)
    return len({i.group_key() if any(i.group_key()) else ("__solo__",
                                                          i.session_id, "")
                for i in infos})


def assign_splits(sessions, val_fraction=0.2, test_fraction=0.2, seed=1234,
                  holdout_val=(), holdout_test=(), holdout_train=(),
                  stratify_by_scene=True, _granularity="group"):
    """Assign every session to exactly one split.

    sessions: [SessionInfo] or [str]. Returns {split: [session_id]}.

    Explicit holdouts are honoured first and unconditionally. The remainder is
    allocated by a deterministic rank, walking whole METADATA GROUPS (same
    date+field+camera) so two recordings of the same bed on the same morning
    cannot be separated - they are near-duplicates and would leak just as a
    frame split does.

    stratify_by_scene (default True) runs that allocation SEPARATELY WITHIN
    each scene - onion_only, weed_only, mixed - so each split gets its share of
    each. Without it the allocation is blind to what a session contains, and
    with a handful of sessions it is entirely ordinary for every mixed drive to
    land in train: the model then looks excellent on a validation set that
    never exercises the crop-vs-weed decision at all. Sessions whose scene is
    unknown form their own stratum and are allocated as before.

    Stratification changes only WHICH sessions go where, never the guarantees:
    holdouts still win, the result is still deterministic for a seed, and
    training is still never left empty."""
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
        if _granularity == "session":
            # Each recording is its own unit. Deliberate, not a relaxation of
            # the leakage rule: no FRAME is shared between splits either way.
            # What is given up is that val/test now share a date with training.
            groups[("__session__", i.session_id, "")].append(i.session_id)
            continue
        key = i.group_key()
        groups[key if any(key) else ("__solo__", i.session_id, "")].append(
            i.session_id)

    ordered = sorted(groups.items(),
                     key=lambda kv: _stable_rank(sorted(kv[1])[0], seed))

    if not stratify_by_scene:
        _quota_walk(ordered, val_fraction, test_fraction, out,
                    guarantee_train=True)
    else:
        scene_by = {i.session_id: (i.scene or "unknown") for i in infos}
        date_by = {i.session_id: (i.date or "") for i in infos}
        strata = defaultdict(list)
        for key, members in ordered:
            stratum = scene_of(members, scene_by)
            if _granularity == "session":
                # Date joins the stratum key ONLY here. At group granularity the
                # date IS the grouping, so adding it would say nothing; at
                # session granularity it is the axis that actually varies, and
                # without it a split can end up holding one day entirely -
                # which on this data means training never sees January.
                dates = {date_by.get(s, "") for s in members}
                stratum = (stratum, dates.pop() if len(dates) == 1 else "mixed")
            strata[stratum].append((key, members))
        # ONE quota walk over an interleaved order, not one walk per stratum.
        # Per-stratum quotas inflate the small splits on small strata (three
        # sessions at 20% rounds to one, which is 33%) and the errors compound.
        _quota_walk(_interleave(strata, seed), val_fraction, test_fraction,
                    out, guarantee_train=True)
        if not out["train"]:
            # Every stratum was thin enough to be consumed by val/test. Give
            # back the largest group rather than shipping an empty training set.
            biggest = max(ordered, key=lambda kv: (len(kv[1]), sorted(kv[1])[0]))
            for split in ("val", "test"):
                out[split] = [s for s in out[split] if s not in biggest[1]]
            out["train"].extend(sorted(biggest[1]))

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


#: A session with fewer frames than this cannot be block-split at all, so it
#: goes wholly to train rather than aborting a build over one short recording.
MIN_FRAMES_FOR_BLOCKS = 5


def assign_frame_blocks_per_session(frame_ids_by_session, val_fraction=0.2,
                                    test_fraction=0.2, gap_frames=2,
                                    n_blocks=1, seed=1234, rotate=True):
    """Contiguous frame blocks WITHIN each session, merged across sessions.

    USE WHEN A SESSION-LEVEL SPLIT IS IMPOSSIBLE OR HARMFUL. Splitting by whole
    session is the only way to measure generalisation, but it requires that no
    class lives in just one session. When a weed-only recording and an
    onion-only recording are the whole dataset, holding out either removes an
    entire class from training - and a crop-safety model that never saw the
    crop is the exact failure this pipeline exists to prevent.

    Blocking within each session instead keeps every class present in every
    split. What it costs is stated plainly by the caller: val then shares each
    session's lighting, soil and often its individual plants, so the scores are
    a sanity check, not evidence of generalisation.

    Returns {split: [frame_id]} plus `_train_only_sessions`: sessions too short
    to block-split, which went wholly to train."""
    out = {"train": [], "val": [], "test": [], "_dropped_gap": [],
           "_train_only_sessions": []}
    for sess in sorted(frame_ids_by_session):
        ids = list(frame_ids_by_session[sess])
        if len(ids) < MIN_FRAMES_FOR_BLOCKS:
            out["train"].extend(ids)
            out["_train_only_sessions"].append(sess)
            continue
        # Keyed by session, so two sessions do not rotate identically and the
        # test set is not the same stretch of every drive.
        part = assign_frame_blocks(ids, val_fraction, test_fraction,
                                   gap_frames=gap_frames, n_blocks=n_blocks,
                                   seed=seed, rotate=rotate, _key=sess)
        for key in ("train", "val", "test", "_dropped_gap"):
            out[key].extend(part[key])
    return out


def missing_from_train(split_map, class_counts_by_session):
    """Classes present in the dataset but absent from every TRAIN session.

    A class the model never sees in training is one it can never predict, and
    when that class is the crop the resulting model reports an empty crop mask
    - indistinguishable, downstream, from 'looked and found no crop'."""
    everywhere, in_train = set(), set()
    train_sessions = set(split_map.get("train", []))
    for sess, counts in class_counts_by_session.items():
        present = {c for c, n in counts.items() if n}
        everywhere |= present
        if sess in train_sessions:
            in_train |= present
    return sorted(everywhere - in_train)


def _split_order(key, seed, rotate):
    """Which split occupies which position along a block."""
    order = ["train", "val", "test"]
    if not rotate:
        return order
    r = _rotation(key, seed)
    return order[r:] + order[:r]


def _edge_splits(chunk, val_fraction, test_fraction, seed, key, rotate):
    """(first, last) split along this chunk, without laying it out.

    Derived from the layout rather than read back from the result: the result
    is a dict of frame lists, and recovering position from it would depend on
    ids sorting in positional order - true of the extractor's zero-padded names
    and not a property worth relying on. Returning the order as an extra dict
    key is worse still: callers sum the dict to account for every frame, and a
    list of split names in there is silently counted as frames.

    A chunk too short to block-split goes wholly to train."""
    n = len(chunk)
    if n < MIN_FRAMES_FOR_BLOCKS:
        return "train", "train"
    sizes = {"val": int(round(val_fraction * n)),
             "test": int(round(test_fraction * n))}
    order = _split_order(key, seed, rotate)
    present = [s for s in order if s == "train" or sizes.get(s, 0) > 0]
    if not present:
        return "train", "train"
    return present[0], present[-1]


def _rotation(key, seed):
    """Which of train/val/test starts this block. Deterministic from the key."""
    return int(zlib.crc32(f"{seed}:{key}".encode()) % 3)


def assign_frame_blocks(frame_ids, val_fraction=0.2, test_fraction=0.2,
                        gap_frames=2, n_blocks=1, seed=1234, rotate=True,
                        _key=""):
    """Split ONE session into contiguous frame blocks, with a discarded buffer.

    USE ONLY WHEN A SESSION-LEVEL SPLIT IS IMPOSSIBLE (a single recording). It
    is strictly weaker than splitting by session and the caller must say so.

    n_blocks cuts the session into that many equal stretches and applies the
    layout inside EACH of them, so val and test are sampled from several places
    along the drive instead of only its tail.

    That matters more than it sounds. With one block, test is always the LAST
    stretch of every recording - which on this data is a systematically
    different thing: later in the day, further along the bed, and often the
    headland where the rig turns. A test set made only of drive-ends measures
    drive-ends. More blocks cost more gap frames (one seam per block boundary),
    which is the price of a test set that looks like the dataset.

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

    # Clamp to what this session can actually support. A stretch shorter than
    # MIN_FRAMES_FOR_BLOCKS cannot be split at all and goes wholly to train, so
    # asking for more blocks than a short session has room for would quietly
    # send EVERY frame to train and leave val empty - training then runs blind,
    # saves no checkpoint and reports no metric, hours later. The request is a
    # ceiling, not a demand.
    k = max(1, min(int(n_blocks), n // MIN_FRAMES_FOR_BLOCKS))
    if k > 1:
        # Each stretch is split independently by the single-block layout below,
        # so every guarantee it makes - contiguity, a gap at every real seam,
        # nothing shared between splits - holds per stretch and therefore
        # overall. A stretch too short to split contributes wholly to train
        # rather than failing the build.
        out = {"train": [], "val": [], "test": [], "_dropped_gap": []}
        edges = [round(j * n / k) for j in range(k + 1)]
        # `_order` is per-chunk and meaningless once chunks are merged, so it is
        # read for the seam decision and not accumulated.
        gap = max(0, int(gap_frames))
        prev_last = None
        for j, (a, b) in enumerate(zip(edges, edges[1:])):
            chunk = ids[a:b]
            # THE SEAM BETWEEN CHUNKS NEEDS A BUFFER - but only when the split
            # actually changes across it. Each chunk used to end with test and
            # begin with train, so a buffer was always required; with the
            # layout rotated per chunk, two chunks often meet at the SAME
            # split, and dropping frames to separate train from train buys
            # nothing. At five blocks and a twelve-frame gap this was throwing
            # away a third of every session.
            nxt, this_last = _edge_splits(chunk, val_fraction, test_fraction,
                                          seed, f"{_key}#{j}", rotate)
            if j and gap and len(chunk) > gap and prev_last != nxt:
                out["_dropped_gap"].extend(chunk[:gap])
                chunk = chunk[gap:]
            if len(chunk) < MIN_FRAMES_FOR_BLOCKS:
                out["train"].extend(chunk)
                prev_last = "train"
                continue
            part = assign_frame_blocks(chunk, val_fraction, test_fraction,
                                       gap_frames, n_blocks=1, seed=seed,
                                       rotate=rotate, _key=f"{_key}#{j}")
            for key in out:
                out[key].extend(part[key])
            prev_last = this_last
        return out

    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    gap = max(0, int(gap_frames))

    # Layout: [ train | gap | val | gap | test ]
    # Test last so it sits at one end of the drive, furthest from training.
    #
    # A gap is spent only where a REAL boundary exists. With test_fraction=0
    # there is no train|val|test seam to buffer, only train|val, and charging
    # for the absent one would discard annotated frames to separate a block
    # from nothing. On a 45-frame dataset that is two hand-corrected frames
    # thrown away for no separation at all.
    n_gaps = (1 if n_val else 0) + (1 if n_test else 0)
    need = n_val + n_test + n_gaps * gap
    if need >= n:
        gap = 0                                    # too small to afford buffers
        need = n_val + n_test
    if need >= n:
        raise SplitError(
            f"{n} frames cannot be split into train/val/test at "
            f"val={val_fraction}, test={test_fraction}. Lower the fractions or "
            f"annotate more frames.")

    n_train = n - need
    sizes = {"train": n_train, "val": n_val, "test": n_test}

    # WHICH SPLIT SITS WHERE ALONG THE DRIVE IS ROTATED PER BLOCK.
    #
    # The layout is still [ a | gap | b | gap | c ] - contiguous, buffered, no
    # frame shared - but which of train/val/test is `a` rotates deterministically
    # with the block. Fixing the order put val in the middle of every stretch of
    # every session and test at the end of every one, so the test set was made
    # entirely of block-ends: later in the pass, further along the bed, often
    # the headland where the rig turns. A test set of drive-ends measures
    # drive-ends.
    #
    # Rotating costs nothing - the same three segment sizes, the same two seams
    # - and buys a test set drawn from beginnings, middles and ends. It is NOT
    # a random frame shuffle: that would put a frame and its near-duplicate on
    # opposite sides of the boundary, which is the failure blocks exist to
    # prevent.
    order = _split_order(_key, seed, rotate)

    parts, i = {}, 0
    for j, name in enumerate(order):
        k = sizes[name]
        parts[name] = ids[i:i + k]
        i += k
        # A gap only where a REAL boundary follows: a zero-length segment is
        # not a seam, and charging for an absent one throws away annotated
        # frames to separate a block from nothing.
        if k and any(sizes[o] for o in order[j + 1:]):
            i += gap
    used = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    return {"train": parts["train"], "val": parts["val"], "test": parts["test"],
            "_dropped_gap": [f for f in ids if f not in used]}


#: Below this many VIDEO frames between a test frame and the nearest training
#: frame, the two are the same ground from centimetres apart and the test set is
#: measuring memorisation. At ~30 fps and walking pace this is roughly two
#: seconds of travel - deliberately modest, because it is a floor for "not
#: literally the same photograph", not a claim of independence.
MIN_SEAM_SEPARATION = 60


def seam_separation(split_frames, frame_index_of=None):
    """How far apart the splits actually are, in VIDEO frames.

    A gap is expressed in POOL frames, and the pool is usually strided - so
    `gap_frames=2` can mean 10 video frames or 2, depending on how the pool was
    curated. That distinction decides whether a within-session split measures
    anything, and it is invisible in the configuration.

    So this measures the thing itself: for each session, the smallest video-
    frame distance between a test frame and a training frame, and the same for
    val. That number is the honest summary of a within-session split. If it is
    2, the test set is the training set photographed again.

    split_frames: {split: [frame_id]}. frame_index_of maps a frame id to its
    video frame index; the default reads the trailing number of the id, which
    is the extractor's own convention.

    Returns {"test": {session: min_distance}, "val": {...}} - sessions with
    nothing in a split are absent rather than reported as infinitely far."""
    if frame_index_of is None:
        from common.frame_spec import index_of as frame_index_of

    def by_session(ids):
        out = defaultdict(list)
        for fid in ids or ():
            idx = frame_index_of(fid)
            if idx >= 0:
                # `<camera>_<date>_<time>_<index>` - the session is everything
                # before the trailing index.
                out[str(fid).rsplit("_", 1)[0]].append(idx)
        return out

    train = by_session(split_frames.get("train"))
    out = {}
    for split in ("val", "test"):
        got = {}
        for sess, idxs in by_session(split_frames.get(split)).items():
            others = train.get(sess)
            if not others:
                continue          # no training frames here: nothing to be near
            got[sess] = min(abs(a - b) for a in idxs for b in others)
        out[split] = got
    return out


def seam_separation_problems(report, floor=MIN_SEAM_SEPARATION):
    """Sessions whose val/test frames sit too close to training frames."""
    bad = []
    for split in ("val", "test"):
        for sess, dist in sorted((report.get(split) or {}).items()):
            if dist < floor:
                bad.append((split, sess, dist))
    return bad


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
