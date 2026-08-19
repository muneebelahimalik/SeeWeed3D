#!/usr/bin/env python3
"""
SeeWeed3D - Stage 0: verified CVAT/Datumaro export -> trainable dataset.

Produces, from one verified export:
  1. Ultralytics segmentation labels (labels/<split>/*.txt) + data.yaml
  2. A per-instance LEP manifest (lep_manifest.json)
  3. A dataset integrity report (dataset_report.json)
  4. Per-session and per-class statistics
  5. annotations_needing_correction.json

Images are NOT copied. Manifests reference the existing files, so a large
dataset is not duplicated and Windows paths keep working (paths are stored
posix-style and resolved against --images-root).

    python -m seeweed3d.training.prepare_dataset \\
        --datumaro-root  D:/exports/verified_mixed \\
        --images-root    D:/Dataset_Vidalia/sessions \\
        --out            D:/Dataset_Vidalia/training/mixed_v1
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ontology import CLASSES  # noqa: E402
from training import datumaro_multitask as dmm  # noqa: E402
from training import seg_dataset as sd  # noqa: E402
from training import splits as sp  # noqa: E402
from training.config import AnnotationContract  # noqa: E402

RANGE_RE = re.compile(r"\d+-\d+")


def find_annotation_files(roots):
    """Every Datumaro JSON under one or several export roots.

    MERGING SEPARATE CVAT TASKS IS THE NORMAL CASE. Annotating one session per
    CVAT task is good practice - tasks stay small, and a session is the unit
    that splits must respect anyway - so this accepts either a parent folder
    holding many unzipped exports, or several explicit roots.

    Merging across tasks is safe because each file is resolved through its OWN
    `categories` block: `label_id` 2 can legitimately mean different classes in
    two tasks, and only the label NAME is carried forward. A merge keyed on
    label_id would silently relabel half the dataset."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    candidates = []
    for root in roots:
        root = Path(root)
        found = sorted(root.rglob("annotations/*.json"))
        if not found:
            found = sorted(root.rglob("*.json"))
        candidates.extend(found)

    # Sniff each file rather than trusting its location. The SAM 3 prelabel
    # output directory holds `instances_default.json` in COCO format plus
    # several bookkeeping JSONs, and pointing at that folder is an easy mistake
    # - it is where the CVAT round trip started. Parsing COCO as Datumaro would
    # fail somewhere deep with a message about a missing 'items' array.
    files, coco, other = [], [], []
    for p in sorted(set(candidates)):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get("items"), list):
            files.append(p)
        elif isinstance(doc.get("images"), list) and "annotations" in doc:
            coco.append(p)
        else:
            other.append(p)

    if not files:
        shown = ", ".join(str(Path(r)) for r in roots)
        msg = [f"ERROR: no Datumaro JSON found under: {shown}"]
        if coco:
            msg += [
                f"",
                f"Found {len(coco)} COCO file(s) instead, e.g. {coco[0]}.",
                f"COCO is the format you IMPORT into CVAT (the SAM 3 "
                f"prelabels). The EXPORT must be 'Datumaro 1.0' - COCO cannot "
                f"carry shape groups, so it would silently discard every "
                f"mask-to-LEP link.",
                f"In CVAT: Task menu -> Export annotations -> 'Datumaro 1.0' "
                f"-> download -> UNZIP, then point at the unzipped folder.",
            ]
        else:
            if other:
                msg += [f"",
                        f"Found {len(other)} JSON file(s) that are neither "
                        f"Datumaro nor COCO, e.g. {other[0]}."]
            msg += [
                f"",
                f"Expected <root>/annotations/*.json. In CVAT use "
                f"Export annotations -> 'Datumaro 1.0' -> download -> UNZIP, "
                f"then point at the unzipped folder(s), or at one parent "
                f"folder containing all of them.",
            ]
        raise SystemExit("\n".join(msg))
    return sorted(set(files))


def _spec_tokens(spec):
    """Split a spec into raw tokens, honouring the `@file` form."""
    if not spec:
        return []
    if isinstance(spec, str) and spec.startswith("@"):
        p = Path(spec[1:])
        if not p.exists():
            raise SystemExit(f"ERROR: frame list file not found: {p}")
        toks = [ln.split("#")[0].strip() for ln in
                p.read_text(encoding="utf-8").splitlines()]
    else:
        toks = []
        for part in ([spec] if isinstance(spec, str) else list(spec)):
            toks.extend(str(part).split(","))
    return [t.strip() for t in toks if t.strip()]


def parse_frame_spec(spec):
    """Parse a frame selection into {session_or_None: (positions, patterns)}.

    Tokens are comma-separated and are one of:
        12                a single 1-based POSITION within its session
        1-26              an inclusive position range
        frame_0007        a literal item id
        *_00[0-3]*        an fnmatch glob over item ids
        vid2_2026...:1-26 any of the above, scoped to ONE session
        vid2_2026...:*    every frame of that session

    Positions are 1-based because that is how a person counts images in the
    CVAT frame slider, and an off-by-one here silently trains on the wrong
    frames.

    POSITIONS ARE NUMBERED WITHIN A SESSION, not across the merged dataset.
    That is what makes a range stable: merging a second CVAT export must not
    renumber the first one and quietly redirect a carefully-checked selection
    at different frames.

    `@path` reads the spec from a file instead, one token per line, `#`
    comments allowed - use it when the list is long enough to mistype.

    The None key holds unscoped tokens, which apply to every session."""
    groups = {}
    for t in _spec_tokens(spec):
        sess = None
        if ":" in t:
            head, _, tail = t.partition(":")
            # A drive letter is not a session scope. Windows paths reach here
            # via @file lists and bare item ids.
            if head and not (len(head) == 1 and head.isalpha()):
                sess, t = head, tail.strip()
            if not t:
                continue
        pos, pat = groups.setdefault(sess, (set(), []))
        if RANGE_RE.fullmatch(t):
            a, b = (int(v) for v in t.split("-"))
            if b < a:
                raise SystemExit(f"ERROR: reversed frame range {t!r}")
            if a < 1:
                raise SystemExit(
                    "ERROR: frame positions are 1-based; 0 is not a frame.")
            pos.update(range(a, b + 1))
        elif t.isdigit():
            if int(t) < 1:
                raise SystemExit(
                    "ERROR: frame positions are 1-based; 0 is not a frame.")
            pos.add(int(t))
        else:
            pat.append(t)
    return groups


def select_frames(frames, include=None, exclude=None):
    """Keep only hand-verified frames. Returns (kept, dropped).

    Why this exists: `keep_empty_frames` removes frames with NO annotations,
    which does NOT cover the common case. A CVAT task pre-loaded with SAM
    proposals has annotations on EVERY frame - the ones you never reached are
    full of machine guesses with the wrong classes. Those frames are not empty,
    so nothing else filters them, and training on them actively teaches the
    model the wrong label for a correctly-shaped mask. That is worse than no
    data, because the mask geometry is right and only the class is wrong, so
    the loss is confident and consistent.

    Frames are numbered 1..n WITHIN each session, ordered by item_id - which
    matches CVAT's frame order whenever the filenames are zero-padded, as the
    extractor guarantees. An unscoped positional token is therefore ambiguous
    once more than one session is present, and is refused rather than guessed:
    silently applying `1-26` to both a weed task and an onion task would select
    the wrong frames from at least one of them."""
    inc = parse_frame_spec(include)
    exc = parse_frame_spec(exclude)

    by_session = {}
    for rec in frames:
        by_session.setdefault(getattr(rec, "session_id", "") or "", []).append(rec)
    sessions = sorted(by_session)

    named = {s for s in list(inc) + list(exc) if s is not None}
    unknown = sorted(named - set(sessions))
    if unknown:
        raise SystemExit(
            f"ERROR: frame selection names session(s) not in this export: "
            f"{unknown}\nPresent: {sessions}")

    unscoped_positions = any(inc.get(k, (set(), []))[0] or
                             exc.get(k, (set(), []))[0] for k in (None,))
    if unscoped_positions and len(sessions) > 1:
        raise SystemExit(
            f"ERROR: this build merges {len(sessions)} sessions "
            f"({', '.join(sessions)}), so a bare frame position is ambiguous - "
            f"positions are numbered WITHIN a session.\n"
            f"Scope each range with its session id, e.g.\n"
            f"    {sessions[0]}:1-26,{sessions[0]}:28-36\n"
            f"and use  <session>:*  to keep all of a session.")

    def matches(i, rec, groups):
        for key in (None, getattr(rec, "session_id", "") or ""):
            pos, pat = groups.get(key, (set(), []))
            if i in pos or any(fnmatch.fnmatch(rec.item_id, g) for g in pat):
                return True
        return False

    def relevant(rec, groups):
        """Does any token address this frame's session at all?"""
        return (None in groups) or ((getattr(rec, "session_id", "") or "")
                                    in groups)

    for sess in sessions:
        n = len(by_session[sess])
        for groups in (inc, exc):
            for key in (None, sess):
                over = sorted(v for v in groups.get(key, (set(), []))[0] if v > n)
                if over:
                    raise SystemExit(
                        f"ERROR: frame position(s) {over} exceed the {n} "
                        f"frame(s) in session {sess!r}. Positions are 1-based "
                        f"within a session; run --list-frames to see the "
                        f"numbering.")

    kept, dropped = [], []
    for sess in sessions:
        ordered = sorted(by_session[sess], key=lambda f: f.item_id)
        for i, rec in enumerate(ordered, start=1):
            want = matches(i, rec, inc) if inc else True
            if want and exc and relevant(rec, exc):
                want = not matches(i, rec, exc)
            (kept if want else dropped).append(rec)
    return kept, dropped


def _reject_silently_dropped_sessions(all_sessions, kept_per_session,
                                      include_frames, exclude_frames):
    """Refuse a build that discards an ENTIRE session nobody asked to discard.

    You loaded that export on purpose, so contributing zero frames from it is
    almost never what you meant. The realistic way it happens is not a wrong
    range but a Python dict with a repeated key:

        "INCLUDE_FRAMES": "onion_sess:1-36",
        "INCLUDE_FRAMES": "weed_sess:1-27,weed_sess:51-60",   # silently wins

    Python keeps only the last, without a warning, and the onion export
    vanishes. Everything downstream still succeeds - the model simply never
    learns the crop class, and the first evidence is `crop safety is
    UNMEASURED` after a full training run.

    Naming a session in EITHER spec counts as asking for it, so an explicit
    `<session>:*` in --exclude-frames is honoured silently."""
    named = set()
    for spec in (include_frames, exclude_frames):
        named |= {s for s in parse_frame_spec(spec) if s is not None}

    lost = sorted(s for s, n in all_sessions.items()
                  if n and not kept_per_session.get(s, 0) and s not in named)
    if not lost:
        return
    raise SystemExit(
        f"ERROR: {len(lost)} session(s) contributed ZERO frames and were never "
        f"named in the frame selection:\n"
        f"    {', '.join(lost)}\n\n"
        f"Their export was loaded, so dropping them entirely is almost "
        f"certainly not what you meant. The usual cause is a REPEATED "
        f"'INCLUDE_FRAMES' key in the CONFIG dict - Python keeps only the "
        f"last one, silently.\n\n"
        f"Fix it by putting every session in ONE spec string:\n"
        f"    \"INCLUDE_FRAMES\": \"{lost[0]}:*,<other_session>:1-27\"\n"
        f"or, to drop them on purpose, remove their DATUMARO_ROOT / name them "
        f"explicitly in EXCLUDE_FRAMES.")


def list_frames(datumaro_root, contract=None):
    """Print the numbered frame table, then stop.

    Always run this before --include-frames. The positions you remember from
    CVAT are only as good as the assumption that its frame order matches
    item_id order, and this is the cheapest way to check that assumption
    instead of discovering it in a trained model."""
    contract = contract or AnnotationContract()
    frames = []
    for f in find_annotation_files(datumaro_root):
        got, _ = dmm.load_datumaro(f, contract)
        frames.extend(got)

    by_session = {}
    for rec in frames:
        by_session.setdefault(rec.session_id or "", []).append(rec)

    out = []
    for sess in sorted(by_session):
        ordered = sorted(by_session[sess], key=lambda f: f.item_id)
        if len(by_session) > 1:
            print(f"\n=== session {sess!r}  ({len(ordered)} frames) ===")
        print(f"{'pos':>5}  {'item_id':<44}{'n':>4}  classes")
        for i, rec in enumerate(ordered, start=1):
            c = Counter(inst.class_name for inst in rec.instances)
            summary = ", ".join(f"{k}={v}" for k, v in sorted(c.items())) \
                or "-EMPTY-"
            print(f"{i:>5}  {rec.item_id:<44}{len(rec.instances):>4}  {summary}")
        out.extend(ordered)

    total = len(out)
    if len(by_session) > 1:
        print(f"\n{total} frame(s) across {len(by_session)} sessions. "
              f"Positions restart at 1 in EACH session, so scope every range "
              f"with its session id:")
        for sess in sorted(by_session):
            print(f"    {sess}:1-10      or  {sess}:*  for all of it")
    else:
        print(f"\n{total} frame(s). Positions are 1-based and are what "
              f"--include-frames / --exclude-frames use.")
    return out


def build(datumaro_root, images_root, out_root, *, contract=None,
          val_fraction=0.2, test_fraction=0.2, seed=1234,
          holdout_val=(), holdout_test=(), strict=True, gap_frames=2,
          drop_classes=(), keep_classes=None, label_provenance="hand_corrected",
          verify_images=False,
          keep_empty_frames=False, require_lep="auto",
          include_frames=None, exclude_frames=None, stratify_by_scene=True,
          split_mode="auto", blocks_per_session=1,
          split_granularity="auto"):
    """Everything after out_root is keyword-only on purpose: a positional
    fraction silently landing in the `contract` slot produced a confusing
    AttributeError deep inside validation rather than an error at the call.

    drop_classes: exclude these ontology classes from THIS dataset build.

        Use this instead of editing common/ontology.py. The ontology is the
        stable, project-wide source of truth: its order fixes the COCO category
        ids, the CVAT label schema, and every file already exported. Deleting a
        class from it would renumber everything and make a future dataset that
        DOES contain that class impossible to merge with this one. Dropping per
        build is reversible and local - re-run without the flag once you have
        examples.

        A `class_mapping.json` records ontology name -> training index, so a
        model trained on the reduced set can still be interpreted against the
        full ontology.

    label_provenance: where these labels came from. Travels with the manifest,
        because it decides what every metric downstream MEANS.

        "hand_corrected"      a person reviewed and fixed them. The default,
                              and the only value under which val/test AP
                              estimates real performance.
        "prelabel_unreviewed" model output that made a round trip through CVAT
                              without correction. Training on this is
                              DISTILLATION: the student learns the teacher's
                              misses as if they were correct and cannot exceed
                              it, and - the part that is easy to miss - the
                              metrics measure AGREEMENT WITH THE TEACHER, not
                              with reality. A high AP here says "faithfully
                              reproduces the prelabeler", which on a
                              crop-safety system is exactly the number not to
                              mistake for "finds the crop".
        "mixed"               some of each; treat as unreviewed until the split
                              is known.

        Recorded rather than inferred: nothing in an export distinguishes a
        polygon a person fixed from one they only looked at, so this is the
        annotator's declaration and the only place the distinction can live.
        preflight raises it as a finding so it is restated at train time, when
        it matters, instead of only here.

    keep_classes: the inverse - name what this build IS, and everything else in
        the ontology is dropped. None (default) means "no allow-list", which is
        not the same as an empty one; an empty list is refused rather than
        silently building nothing.

        Prefer this over drop_classes whenever the build is defined by what it
        contains rather than by what it lacks - a crop-only detector is
        "onion_plant", not "not these five weeds". The two spellings pick the
        same classes TODAY and diverge the moment a class is appended to the
        ontology: the deny-list silently admits it into a dataset that was never
        meant to have it, while the allow-list keeps meaning what it said. Since
        appending classes is the documented way the ontology grows, that is a
        question of when, not if.

        Combining both is allowed and drop_classes still applies on top, so
        keep=[a, b] with drop=[b] yields [a] - useful for narrowing an
        allow-list without rewriting it.

    require_lep: "auto" (default) | True | False. Whether a visible, targetable
        weed must carry a grouped LEP point.

        "auto" decides from the export itself, because the two situations need
        opposite treatment and are trivially distinguishable:

          * ZERO LEPs in the whole export -> you annotated masks only. That is
            a legitimate SEGMENTATION-ONLY dataset (Stage A needs no LEPs at
            all), so the requirement is lifted and no errors are raised. Stage B
            simply cannot be trained from it.
          * SOME but not all -> you started placing LEPs and stopped. That IS a
            real gap in the contract and every missing one is reported.

    keep_empty_frames: by default a frame with ZERO annotations is EXCLUDED.
        In an export from a task you annotated by hand, an empty frame is
        almost always one you did not get to - and it is indistinguishable from
        genuinely bare ground. Training on it teaches the model that a frame
        full of weeds is background, which is far more damaging than the frame
        being missing. Pass True only if your empty frames are deliberately
        empty ground.

    include_frames / exclude_frames: keep only the frames you actually
        hand-verified. Needed whenever the CVAT task was pre-loaded with SAM
        proposals, because then the frames you never reached are NOT empty -
        they carry machine guesses with the wrong classes, and no other filter
        removes them. See select_frames(). Run --list-frames first to get the
        numbering.

    images_root: a single sessions folder, or a LIST of them. Needed whenever
        the datasets being merged were not captured under one common parent -
        a weed capture set and a separately-recorded onion set, for instance.
        Every frame is resolved by trying each root in turn (seg_dataset.
        resolve_image), so a session need only exist under ONE of them, not
        all - it does not matter which root a given export's frames live
        under, only that at least one root has them."""
    contract = contract or AnnotationContract()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    # NOT validated for existence here: build() never reads an image file, only
    # records where to find one later, so it stays testable without a real
    # dataset on disk. The config-block runners (make_dataset.py) and the CLI
    # check existence up front, with an error naming which root is missing.
    # Validated first, before any export is parsed: a typo here should cost a
    # second, not a full load of every annotation file.
    PROVENANCE = ("hand_corrected", "prelabel_unreviewed", "mixed")
    if label_provenance not in PROVENANCE:
        raise SystemExit(
            f"ERROR: label_provenance must be one of {list(PROVENANCE)}; got "
            f"{label_provenance!r}")

    split_mode_wanted = str(split_mode or "auto").lower()
    if split_mode_wanted not in ("auto", "session", "frame_block"):
        raise SystemExit(
            f"ERROR: split_mode must be 'auto', 'session' or 'frame_block'; "
            f"got {split_mode!r}")

    img_roots = images_root if isinstance(images_root, (list, tuple)) \
        else [images_root]
    img_roots = [Path(r) for r in img_roots]

    drop = {c for c in (drop_classes or ())}
    unknown_drop = sorted(drop - set(CLASSES))
    if unknown_drop:
        raise SystemExit(
            f"ERROR: --drop-classes names classes not in the ontology: "
            f"{unknown_drop}\nKnown: {CLASSES}")

    # An allow-list is turned into the equivalent deny-list here, so everything
    # downstream - active_classes, class_mapping.json, the per-class report -
    # has exactly one notion of what is in this build. `is not None` and not a
    # truth test: an EMPTY keep list is a mistake worth naming, not a silent
    # "keep everything".
    if keep_classes is not None:
        keep = {c for c in keep_classes}
        unknown_keep = sorted(keep - set(CLASSES))
        if unknown_keep:
            raise SystemExit(
                f"ERROR: --keep-classes names classes not in the ontology: "
                f"{unknown_keep}\nKnown: {CLASSES}")
        if not keep:
            raise SystemExit(
                "ERROR: --keep-classes is empty. Name the classes this build "
                "is FOR, or omit it entirely to keep every class.")
        drop |= set(CLASSES) - keep

    active_classes = [c for c in CLASSES if c not in drop]
    if not active_classes:
        raise SystemExit("ERROR: every class was dropped; nothing to train on.")

    ann_files = find_annotation_files(datumaro_root)
    frames, report = [], dmm.MultitaskDatasetReport()
    origin = {}
    for f in ann_files:
        got, report = dmm.load_datumaro(f, contract, report=report)
        for rec in got:
            origin.setdefault(rec.item_id, []).append(str(f))
        frames.extend(got)

    # -- keep only hand-verified frames --------------------------------------
    # FIRST, before duplicate detection and before validation. A frame you never
    # reached in CVAT is full of SAM guesses with the wrong classes; reporting
    # errors on annotations that are about to be discarded would fill
    # annotations_needing_correction.json with noise and, under strict mode,
    # block the build over frames that are not part of the dataset.
    if include_frames or exclude_frames:
        before = len(frames)
        all_sessions = Counter(f.session_id or "" for f in frames)
        frames, discarded = select_frames(frames, include_frames, exclude_frames)
        if not frames:
            raise SystemExit(
                "ERROR: the frame selection kept 0 frames. Run --list-frames "
                "to check the numbering before selecting.")
        keep_ids = {f.item_id for f in frames}
        origin = {k: v for k, v in origin.items() if k in keep_ids}
        report = dmm.MultitaskDatasetReport()     # errors from discarded frames
        for f in ann_files:                       # are not yours to fix
            _, report = dmm.load_datumaro(f, contract, report=report,
                                          only_items=keep_ids)
        kept_ids = sorted(keep_ids)
        print(f"  SELECTED {len(frames)} of {before} frame(s); discarded "
              f"{len(discarded)}.")

        # Per session, not just a total. A single "discarded 399" line reads as
        # "the SAM-only frames I meant to drop" even when it silently includes
        # every frame of a whole export.
        kept_per_session = Counter(f.session_id or "" for f in frames)
        for sess in sorted(all_sessions):
            print(f"      {sess}: kept {kept_per_session.get(sess, 0)} "
                  f"of {all_sessions[sess]}")
        print(f"      kept: {kept_ids[0]} .. {kept_ids[-1]}")

        _reject_silently_dropped_sessions(all_sessions, kept_per_session,
                                          include_frames, exclude_frames)

    # The same frame annotated in two CVAT tasks would be counted twice and
    # could land in two splits, which is exactly the leakage this pipeline
    # exists to prevent. Detected here rather than discovered as an
    # inexplicably good validation score.
    dupes = {k: v for k, v in origin.items() if len(v) > 1}
    if dupes:
        for item_id, sources in sorted(dupes.items())[:20]:
            report.add_error(item_id, "duplicate_frame_across_exports",
                             f"annotated in {len(sources)} exports: "
                             f"{', '.join(sources)}. Keep exactly one, or the "
                             f"frame is trained on twice and may span splits.")

    # -- segmentation-only vs multitask -------------------------------------
    n_lep = sum(1 for f in frames for i in f.instances if i.lep is not None)
    n_weeds = sum(1 for f in frames for i in f.instances if not i.is_crop)
    seg_only = False
    if require_lep == "auto":
        seg_only = (n_lep == 0 and n_weeds > 0)
    elif require_lep is False:
        seg_only = True
    if seg_only:
        # Lift the requirement rather than emit one identical error per weed.
        contract = dataclasses.replace(contract, visibility_requiring_lep=())

    report = dmm.validate_frames(frames, contract, report)
    print(f"  merged {len(ann_files)} annotation file(s) -> {len(frames)} frames")
    if seg_only:
        print(f"\n  [i] SEGMENTATION-ONLY dataset: {n_weeds} weed instance(s) "
              f"and {n_lep} LEP point(s).")
        print(f"      Stage A (crop-vs-weed segmentation) trains from this "
              f"normally - it needs no LEPs.")
        print(f"      Stage B (LEPRoiNet) CANNOT be trained from it. Until you "
              f"annotate LEPs, the pipeline uses the hand-engineered estimator "
              f"in perception/lep.py, which needs no training.\n")
    elif n_lep and n_lep < n_weeds:
        print(f"  [!] {n_lep} LEP(s) for {n_weeds} weed instance(s) - PARTIAL. "
              f"Missing ones are reported as errors; finish them or set "
              f"--no-require-lep to build segmentation-only.")

    # -- drop classes for THIS build (ontology untouched) --------------------
    if drop:
        removed = 0
        for f in frames:
            before = len(f.instances)
            f.instances = [i for i in f.instances if i.class_name not in drop]
            removed += before - len(f.instances)
        print(f"  dropped {removed} instance(s) of {sorted(drop)} from this "
              f"build. common/ontology.py is UNCHANGED - re-run without "
              f"--drop-classes once you have examples.")
        # Recount, so the report describes the dataset that was actually built
        # rather than the export it was read from. A per_class still listing a
        # dropped class would send you looking for it in the trained model.
        per_class, per_session = Counter(), {}
        for f in frames:
            for i in f.instances:
                per_class[i.class_name] += 1
                per_session.setdefault(f.session_id, Counter())[i.class_name] += 1
        report.per_class = dict(per_class)
        report.per_session = {k: dict(v) for k, v in per_session.items()}
        report.n_instances = int(sum(per_class.values()))

    # -- exclude un-annotated frames ----------------------------------------
    # An empty frame in a hand-annotated export is almost always one you did
    # not reach, and it is indistinguishable from genuinely bare ground.
    # Training on it teaches the model that a frame full of weeds is
    # background, which is far worse than the frame simply being absent.
    empty = [f for f in frames if not f.instances]
    if empty and not keep_empty_frames:
        frames = [f for f in frames if f.instances]
        print(f"  EXCLUDED {len(empty)} frame(s) with no annotations "
              f"({', '.join(f.item_id for f in empty[:5])}"
              f"{' ...' if len(empty) > 5 else ''}).")
        print(f"      If any of those are genuinely bare ground you WANT to "
              f"train on, pass --keep-empty-frames.")
    elif empty:
        print(f"  [!] KEEPING {len(empty)} frame(s) with no annotations as "
              f"negative examples, because --keep-empty-frames was given.")

    # -- exclude frames whose IMAGE is not on disk ---------------------------
    # An export can outlive the frames it describes: pool frames deleted after
    # extraction, a session pruned, an rgb/ folder half-copied. The manifest
    # records where to find an image rather than reading it, so without this
    # the mismatch surfaces later, one frame at a time, as a FileNotFoundError
    # from COCO export - which names a single file and tells you nothing about
    # whether one frame is missing or nine hundred.
    #
    # Off by default so build() stays runnable with no images on disk, which is
    # what keeps it testable; the config runners turn it on.
    if verify_images:
        home = {}
        for f in frames:
            for src in origin.get(f.item_id, []):
                cand = Path(src).parent.parent
                if (cand / "rgb").is_dir():
                    home[f.item_id] = cand
                    break
        missing, kept = [], []
        for f in frames:
            try:
                sd.resolve_image(f.image_path, img_roots, f.session_id,
                                 home.get(f.item_id))
                kept.append(f)
            except FileNotFoundError:
                missing.append(f)
        if missing:
            by_session = Counter(f.session_id for f in missing)
            print(f"  EXCLUDED {len(missing)} frame(s) whose image is not on "
                  f"disk:")
            for s, n in sorted(by_session.items()):
                print(f"      {s}: {n} frame(s)")
            print(f"      e.g. {', '.join(Path(f.image_path).name for f in missing[:5])}"
                  f"{' ...' if len(missing) > 5 else ''}")
            print(f"      The annotations outlived the images - frames deleted "
                  f"after extraction, or an rgb/ folder that did not finish "
                  f"copying. The build continues WITHOUT them, so the dataset "
                  f"describes what actually exists.")
            frames = kept
            # Recount: the report must describe the dataset that was built.
            per_class, per_session = Counter(), {}
            for f in frames:
                for i in f.instances:
                    per_class[i.class_name] += 1
                    per_session.setdefault(f.session_id,
                                           Counter())[i.class_name] += 1
            report.per_class = dict(per_class)
            report.per_session = {k: dict(v) for k, v in per_session.items()}
            report.n_instances = int(sum(per_class.values()))

    if not frames:
        raise SystemExit(
            "ERROR: no annotated frames left after filtering. Check that the "
            "CVAT task actually contains saved annotations.")

    # -- splits -------------------------------------------------------------
    per_session = Counter(f.session_id for f in frames)
    # Read what each session actually IS from its own meta/session.json, so the
    # allocator can keep same-morning drives together and give every split its
    # share of onion-only, weed-only and mixed scenes. Without this the `scene`
    # and `date` fields are empty and every session looks interchangeable - and
    # with a handful of sessions it is entirely ordinary for every mixed drive
    # to land in train, leaving a validation set that never once exercises the
    # crop-vs-weed decision.
    from common.session_meta import find_session_meta, unknown_scene_report
    metas = {s: find_session_meta(img_roots, s)
             for s in sorted(per_session) if s}
    infos = [sp.SessionInfo(session_id=s, n_frames=n,
                            class_counts=report.per_session.get(s, {}),
                            scene=metas.get(s, {}).get("scene", "unknown"),
                            date=metas.get(s, {}).get("date", ""),
                            field_id=metas.get(s, {}).get("field_id", ""),
                            camera=metas.get(s, {}).get("camera", ""))
             for s, n in sorted(per_session.items()) if s]

    gaps = unknown_scene_report(list(metas.values()))
    if gaps["no_session_json"]:
        print(f"\n  [!] no meta/session.json for "
              f"{', '.join(gaps['no_session_json'])} - these sessions are "
              f"allocated without scene information.")
        print(f"      Looked under: "
              f"{', '.join(r.as_posix() for r in img_roots)}")
    elif gaps["unknown_scene"]:
        print(f"\n  [!] scene_hint missing or unrecognised for "
              f"{', '.join(gaps['unknown_scene'])}. Set it in "
              f"extract_sessions.py's INPUT_ROOTS (onion_only | weed_only | "
              f"mixed) so splits can be balanced by scene.")

    # A session-level split is the ONLY one that measures generalisation, so it
    # is tried first - but it is not always possible or even safe, and both
    # failure modes are silent unless checked. `reason` stays None while the
    # session split remains usable.
    split_mode, frame_split, split_map, reason = "session", None, None, None
    split_info = None
    if split_mode_wanted == "frame_block":
        # Chosen deliberately, not fallen back to. The reason is stated in the
        # same words the fallback uses, so the output means the same thing
        # either way and nobody reads a deliberate block split as a session one.
        reason = ("frame blocks were requested (SPLIT_MODE='frame_block') - "
                  "every session contributes to every split")
    elif len(infos) < 2:
        reason = (f"only one session "
                  f"({infos[0].session_id if infos else '?'}) - there is "
                  f"nothing to hold out")
    else:
        split_map, split_info = sp.plan_splits(
            infos, val_fraction, test_fraction, seed,
            holdout_val=holdout_val, holdout_test=holdout_test,
            stratify_by_scene=stratify_by_scene,
            granularity=split_granularity)
        # Whole sessions are indivisible, so a fraction that rounds below one
        # session yields an EMPTY val: training then runs blind, saves no
        # checkpoint and reports no metric, hours later.
        empty = [s for s, frac in (("val", val_fraction),
                                   ("test", test_fraction))
                 if frac > 0 and not split_map.get(s)]
        if empty:
            reason = (f"{len(infos)} sessions cannot fill {', '.join(empty)} at "
                      f"val={val_fraction}/test={test_fraction} - whole "
                      f"sessions are indivisible, so the fraction rounds to "
                      f"zero sessions and training would run with no "
                      f"validation set at all")
        else:
            # The subtler failure: the split is non-empty but a class lives
            # entirely in a held-out session. Training then never sees it. When
            # that class is the crop, the model reports an empty crop mask -
            # downstream, indistinguishable from "looked and found no crop".
            lost = sp.missing_from_train(split_map, report.per_session)
            if lost:
                reason = (f"holding out whole sessions would remove "
                          f"{', '.join(lost)} from TRAINING entirely - "
                          f"that class exists only in a held-out session")

    if reason is None:
        sp.check_no_leakage(split_map, {f.item_id: f.session_id for f in frames
                                        if f.session_id})
        session_where = {s: k for k, v in split_map.items() for s in v}
        where = {f.item_id: session_where.get(f.session_id) for f in frames}

        if split_info and split_info.get("fell_back"):
            # Printed BEFORE the numbers, because it changes what they mean.
            print(f"\n  [!] Split granularity: SESSION, not day.")
            print(f"      {split_info['reason']}")
        elif split_info:
            print(f"\n  Split granularity: {split_info['granularity']} "
                  f"({split_info['n_units']} independent unit(s)) - val and "
                  f"test share no day with training.")

        rep = sp.scene_representation(split_map, infos)
        print("\n  Scene representation (sessions per split):")
        for split in ("train", "val", "test"):
            counts = rep["counts"].get(split) or {}
            body = ", ".join(f"{k}={v}" for k, v in counts.items()) or "-"
            print(f"      {split:<5} {body}")
        # A split missing a scene is not an error - with few sessions it can be
        # unavoidable - but it bounds what its numbers mean, and that bound is
        # invisible in any metric the training run will print.
        for split in ("val", "test"):
            for scene in rep["missing"].get(split) or []:
                cost = {"mixed": "the crop-vs-weed decision is never exercised "
                                 "there",
                        "weed_only": "weed recall is never measured there",
                        "onion_only": "crop segmentation is never measured "
                                      "there"}.get(scene, "")
                print(f"  [!] {split} contains no {scene} session"
                      f"{' - ' + cost if cost else ''}.")
    else:
        # Blocks WITHIN each session, so every session - and therefore every
        # class - is represented in train and in val.
        split_mode = "frame_block"
        by_session = {}
        for f in sorted(frames, key=lambda f: f.item_id):
            by_session.setdefault(f.session_id, []).append(f.item_id)
        frame_split = sp.assign_frame_blocks_per_session(
            by_session, val_fraction, test_fraction, gap_frames=gap_frames,
            n_blocks=blocks_per_session, seed=seed)
        split_map = {"train": [i.session_id for i in infos], "val": [],
                     "test": []}
        where = {}
        for split in ("train", "val", "test"):
            for item in frame_split[split]:
                where[item] = split
        chosen = split_mode_wanted == "frame_block"
        head = ("SPLIT BY CONTIGUOUS FRAME BLOCKS (requested)" if chosen
                else f"SESSION-LEVEL SPLIT NOT USED: {reason}")
        print(f"\n  [!] {head}.")
        print(f"      {blocks_per_session} block(s) per session, "
              f"{gap_frames}-frame gap at each seam:")
        print(f"        train={len(frame_split['train'])} "
              f"val={len(frame_split['val'])} test={len(frame_split['test'])} "
              f"(dropped as buffer: {len(frame_split['_dropped_gap'])})")
        if frame_split.get("_train_only_sessions"):
            print(f"      Too short to block-split, so wholly in train: "
                  f"{', '.join(frame_split['_train_only_sessions'])}")

        # The gap is counted in POOL frames, and the pool is usually strided -
        # so the same number can mean 2 video frames or 50 depending on how
        # curation ran. That distinction decides whether this split measures
        # anything, and it is invisible in the configuration. So measure it.
        by_split = {s: list(frame_split[s]) for s in ("train", "val", "test")}
        sep = sp.seam_separation(by_split)
        rows = [(k, s, d) for k in ("val", "test")
                for s, d in sorted((sep.get(k) or {}).items())]
        if rows:
            print(f"      Nearest TRAIN frame, in VIDEO frames:")
            for split, sess, dist in rows:
                print(f"        {split:<5} {sess:<28} {dist:>6}")
        too_close = sp.seam_separation_problems(sep)
        if too_close:
            print(f"  [!] {len(too_close)} split/session pair(s) sit closer "
                  f"than {sp.MIN_SEAM_SEPARATION} video frames to training "
                  f"data. At walking pace that is the same ground from "
                  f"centimetres away, so those frames measure memorisation "
                  f"rather than performance.")
            print(f"      Raise GAP_FRAMES until this clears - the cost is a "
                  f"handful of annotated frames, which is far cheaper than a "
                  f"test score that is wrong in the optimistic direction.")

        print(f"      Every session contributes to every split, so no class is "
              f"missing from training and the test set is drawn from ALL of "
              f"your data.")
        print(f"      What it CANNOT tell you: val and test share each "
              f"session's lighting, soil, growth stage and field. A model that "
              f"scores well here has been shown to work on ground it has seen "
              f"the rest of - not on a new drive.")
        print(f"      Treat these numbers as an upper bound on field "
              f"performance, and record a fresh session before quoting them "
              f"as generalisation.\n")

    frames_by_session = {}
    for f in frames:
        frames_by_session.setdefault(f.session_id, []).append(f.image_path)
    summary = sp.write_splits(out / "splits", split_map, frames_by_session, infos)
    summary["split_mode"] = split_mode
    if split_info:
        # Travels with the dataset: a run against a session-granularity split
        # is not comparable with one against a held-out day, and months later
        # the manifest is the only record of which it was.
        summary["split_granularity"] = split_info["granularity"]
        summary["split_granularity_fell_back"] = bool(split_info["fell_back"])
        if split_info.get("reason"):
            summary["split_granularity_note"] = split_info["reason"]
    if frame_split:
        summary["frame_blocks"] = {k: v for k, v in frame_split.items()}
        summary["warning"] = (
            "frame_block split: val/test come from the SAME recording as train. "
            "Scores are a sanity check, not evidence of generalisation.")
        (out / "splits" / "splits_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    # -- Ultralytics labels -------------------------------------------------
    n_labels = 0
    for f in frames:
        split = where.get(f.item_id)
        if split is None:
            continue
        d = out / "labels" / split
        d.mkdir(parents=True, exist_ok=True)
        body = dmm.to_yolo_segmentation(f, active_classes)
        (d / f"{Path(f.image_path).stem}.txt").write_text(body + "\n",
                                                          encoding="utf-8")
        n_labels += 1

    if len(img_roots) > 1:
        print(f"  [!] {len(img_roots)} --images-root paths given; data.yaml "
              f"(Ultralytics backend only) can hold one 'path:', so it uses "
              f"the first: {img_roots[0]}. The permissive backends "
              f"(seg_manifest.json) are unaffected - they search every root.")
    data_yaml = (
        "# Ultralytics dataset config, generated by prepare_dataset.py.\n"
        "# Class order comes from seeweed3d/common/ontology.py and MUST NOT be\n"
        "# reordered - the indices are baked into every label file.\n"
        f"path: {img_roots[0].as_posix()}\n"
        "train: ../training/splits/train_images.txt\n"
        "val: ../training/splits/val_images.txt\n"
        "test: ../training/splits/test_images.txt\n"
        f"nc: {len(active_classes)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n"
                             for i, n in enumerate(active_classes)))
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # -- Segmentation manifest (permissive backends) ------------------------
    # The BSD-3 Mask R-CNN and Apache-2.0 RF-DETR paths train from this, so the
    # permissive default needs no YOLO-format label tree. Derived from the same
    # FrameRecords as the LEP manifest, so the two stages cannot disagree about
    # what was annotated.
    seg_frames = []
    for f in frames:
        split = where.get(f.item_id)
        if split is None or not f.instances:
            continue
        # Where this frame's export lived. An unzipped CVAT export sits at
        # <session>/annotations/default.json with the frames beside it in
        # <session>/rgb/, so the export's grandparent is the one root KNOWN to
        # hold this frame - as opposed to the flat images_root list, which has
        # lost which source each frame came from by the time it is written.
        #
        # That distinction is not academic: a session folder renamed after
        # extraction keeps the ORIGINAL prefix in its export's item ids, so the
        # frame's session id matches neither the folder nor the files on disk,
        # and every root looks equally (im)plausible. Resolving by frame index
        # instead would be ambiguous across roots and could silently pair one
        # session's annotation with another session's image.
        home = None
        for src in origin.get(f.item_id, []):
            cand = Path(src).parent.parent
            if (cand / "rgb").is_dir():
                home = cand.as_posix()
                break
        seg_frames.append({
            "session_id": f.session_id, "item_id": f.item_id,
            "image_path": Path(f.image_path).as_posix(),
            "export_dir": home,
            "width": f.width, "height": f.height, "split": split,
            "instances": [{"class_name": i.class_name,
                           "class_index": active_classes.index(i.class_name),
                           "polygons": [[round(float(v), 2) for v in p]
                                        for p in i.polygons]}
                          for i in f.instances],
            "ignore_regions": [[round(float(v), 2) for v in p]
                               for p in f.ignore_regions]})
    # dataset_kind and split_strategy travel WITH the manifest, not only in
    # dataset_report.json: the trainer logs them as run parameters, and a run
    # against a frame_block split is not comparable with one against a held-out
    # session. Months later the experiment table is the only record of which
    # was which, and "unknown" there is worse than useless.
    (out / "seg_manifest.json").write_text(
        json.dumps({"images_root": [r.as_posix() for r in img_roots],
                    "classes": list(active_classes),
                    "dataset_kind": ("segmentation_only" if seg_only
                                     else "multitask"),
                    "split_strategy": split_mode,
                    "label_provenance": label_provenance,
                    "sessions": sorted({f.session_id for f in frames}),
                    "n_frames": len(seg_frames), "frames": seg_frames}, indent=2),
        encoding="utf-8")

    # -- LEP manifest, split-aware -----------------------------------------
    rows = dmm.to_lep_manifest(frames)
    for r in rows:
        r["split"] = where.get(r["item_id"], "unassigned")
    (out / "lep_manifest.json").write_text(
        json.dumps({"images_root": [r.as_posix() for r in img_roots],
                    "dataset_kind": ("segmentation_only" if seg_only
                                     else "multitask"),
                    "split_strategy": split_mode,
                    "label_provenance": label_provenance,
                    "n_rows": len(rows), "rows": rows}, indent=2),
        encoding="utf-8")

    (out / "class_mapping.json").write_text(json.dumps({
        "ontology": list(CLASSES),
        "active_classes": list(active_classes),
        "dropped": sorted(drop),
        # What was ASKED for, alongside what it resolved to. An allow-list build
        # records the allow-list, so a later reader can tell "onion only, by
        # intent" from "these five happened to be dropped that day" - which is
        # the difference between a dataset that should gain a newly-added class
        # on rebuild and one that should not.
        "selection_mode": "keep" if keep_classes is not None else "drop",
        "kept_requested": (sorted(keep_classes)
                           if keep_classes is not None else None),
        "train_index_to_ontology_name": {i: n for i, n in
                                         enumerate(active_classes)},
        "note": ("Training indices are into active_classes, which is contiguous. "
                 "common/ontology.py is unchanged, so a future dataset "
                 "containing the dropped classes merges with this one.")},
        indent=2), encoding="utf-8")

    # -- reports ------------------------------------------------------------
    rep = report.to_dict()
    rep["splits"] = summary
    rep["n_yolo_label_files"] = n_labels
    rep["n_lep_rows"] = len(rows)
    rep["dataset_kind"] = "segmentation_only" if seg_only else "multitask"
    rep["label_provenance"] = label_provenance
    rep["lep_rows_per_split"] = dict(Counter(r["split"] for r in rows))
    (out / "dataset_report.json").write_text(json.dumps(rep, indent=2),
                                             encoding="utf-8")
    (out / "annotations_needing_correction.json").write_text(
        json.dumps(report.needs_correction, indent=2), encoding="utf-8")

    xcheck = dmm.cross_check_with_datumaro(ann_files[0], frames)
    if xcheck:
        (out / "datumaro_cross_check.json").write_text(json.dumps(xcheck, indent=2),
                                                       encoding="utf-8")

    print(f"\n{report.summary()}")
    print(f"  classes : {report.per_class}")
    # FRAMES per split, which is what "is this trainable" depends on. The
    # session counts in split_map are meaningless under the frame_block
    # fallback - every session is listed under train there, so this line read
    # "val=0" while sixteen val frames existed, which is exactly the wrong
    # thing to tell someone about to start a run.
    frames_per_split = Counter(s for s in where.values() if s)
    print(f"  splits  : " + " ".join(
        f"{k}={frames_per_split.get(k, 0)}"
        for k in ("train", "val", "test")) + " frames"
        + (f"  (by session: " + " ".join(f"{k}={len(v)}"
                                         for k, v in split_map.items()) + ")"
           if split_mode == "session" else
           f"  [{split_mode} split within each of "
           f"{len({f.session_id for f in frames})} session(s)]"))
    print(f"  LEP rows: {len(rows)}  ({rep['lep_rows_per_split']})")
    print(f"  -> {out}")

    if report.errors:
        print(f"\n  {len(report.errors)} ERROR(S). See "
              f"annotations_needing_correction.json. Fix them in CVAT, re-export, "
              f"and re-run - training on a broken contract silently corrupts the "
              f"target.")
        if strict:
            raise SystemExit(1)
    return report, split_map, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datumaro-root", required=True, nargs="+",
                   help="one or more unzipped CVAT 'Datumaro 1.0' exports, or "
                        "a single parent folder containing several of them. "
                        "Annotating one session per CVAT task and merging here "
                        "is the normal workflow.")
    p.add_argument("--images-root", required=True, nargs="+",
                   help="dataset sessions root(s) holding the RGB frames. "
                        "Pass more than one when the datasets being merged "
                        "were not captured under one common parent - each "
                        "frame is resolved by trying every root in turn, so a "
                        "session need only exist under ONE of them.")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-require-lep", action="store_true",
                   help="build a SEGMENTATION-ONLY dataset even if some LEPs "
                        "exist. Not normally needed: an export with no LEPs at "
                        "all is detected automatically.")
    p.add_argument("--drop-classes", nargs="*", default=[],
                   help="exclude these ontology classes from THIS build "
                        "(e.g. --drop-classes wild_radish weed_cluster). "
                        "common/ontology.py is NOT modified.")
    p.add_argument("--split-granularity", default="auto",
                   choices=["auto", "group", "session"],
                   help="what unit is held out. 'group' = a whole "
                        "date+field+camera, the only split that estimates "
                        "generalisation. 'session' = a whole recording from a "
                        "date training also saw - no frame leakage, but an "
                        "upper bound. 'auto' (default) tries group and falls "
                        "back, saying so.")
    p.add_argument("--keep-classes", nargs="*", default=None,
                   help="the inverse of --drop-classes: keep ONLY these and "
                        "drop the rest of the ontology (e.g. --keep-classes "
                        "onion_plant for a crop-only detector). Prefer this "
                        "when the build is defined by what it contains - a "
                        "deny-list silently admits any class appended to the "
                        "ontology later, an allow-list does not.")
    p.add_argument("--label-provenance", default="hand_corrected",
                   choices=["hand_corrected", "prelabel_unreviewed", "mixed"],
                   help="where these labels came from. 'prelabel_unreviewed' "
                        "means training is DISTILLATION from the prelabeler "
                        "and every metric measures agreement with IT, not with "
                        "reality. Recorded in the manifest and restated by "
                        "preflight at train time.")
    p.add_argument("--verify-images", action="store_true",
                   help="check every frame's image is on disk and exclude the "
                        "ones that are not, reporting the count per session. "
                        "Without it a missing image surfaces later as a "
                        "one-file error from COCO export.")
    p.add_argument("--keep-empty-frames", action="store_true",
                   help="keep frames with no annotations as negative examples. "
                        "Off by default: an empty frame is usually one you did "
                        "not annotate, and training on it teaches the model "
                        "that plants are background.")
    p.add_argument("--list-frames", action="store_true",
                   help="print the numbered frame table and exit. RUN THIS "
                        "FIRST if you intend to use --include-frames.")
    p.add_argument("--include-frames", default=None,
                   help="keep ONLY these frames, e.g. '1-26,28-36,50-59' "
                        "(1-based positions in item_id order), an item id, an "
                        "fnmatch glob, or '@list.txt'. REQUIRED when the CVAT "
                        "task was pre-loaded with SAM proposals: frames you "
                        "never reached are not empty, they hold machine "
                        "guesses with the wrong classes, and no other filter "
                        "removes them.")
    p.add_argument("--exclude-frames", default=None,
                   help="drop these frames; same syntax as --include-frames. "
                        "Applied after it.")
    p.add_argument("--gap-frames", type=int, default=2,
                   help="single-session only: frames discarded at each block "
                        "boundary to reduce temporal leakage")
    p.add_argument("--holdout-val", nargs="*", default=[])
    p.add_argument("--holdout-test", nargs="*", default=[])
    p.add_argument("--allow-errors", action="store_true",
                   help="write outputs even when the contract is violated "
                        "(for triage only - do NOT train on the result)")
    a = p.parse_args(argv)
    if a.list_frames:
        list_frames(a.datumaro_root)
        return
    build(a.datumaro_root, a.images_root, a.out,
          val_fraction=a.val_fraction, test_fraction=a.test_fraction,
          seed=a.seed, holdout_val=a.holdout_val, holdout_test=a.holdout_test,
          strict=not a.allow_errors, gap_frames=a.gap_frames,
          drop_classes=a.drop_classes,
          keep_classes=a.keep_classes,
          split_granularity=a.split_granularity,
          label_provenance=a.label_provenance,
          verify_images=a.verify_images,
          keep_empty_frames=a.keep_empty_frames,
          require_lep=False if a.no_require_lep else "auto",
          include_frames=a.include_frames, exclude_frames=a.exclude_frames)


if __name__ == "__main__":
    main()
