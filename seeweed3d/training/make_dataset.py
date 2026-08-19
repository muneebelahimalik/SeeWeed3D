#!/usr/bin/env python3
"""
SeeWeed3D - build a training dataset from a CVAT export (EDIT THE CONFIG BELOW)
==============================================================================
Same job as `prepare_dataset.py`, driven by the config block in this file
instead of a long command line - the convention the extraction and prelabel
stages already use. Edit, save, run:

    python seeweed3d/training/make_dataset.py

Works identically in cmd.exe, PowerShell and VS Code's Run button. No
backticks, no backslash-escaping, no quoting rules. (In cmd.exe a trailing
backtick is NOT a line continuation - that is `^` - which is why a
PowerShell-style multi-line command pastes as a series of broken commands.)

TWO PASSES, ON PURPOSE
----------------------
LIST_FRAMES = True first. It prints every frame in the export with its position
and class counts and writes NOTHING. Use it to confirm which positions are your
hand-corrected frames before you commit to a range - the numbering is only as
good as the assumption that CVAT's frame order matches filename order, and
checking costs seconds while discovering an off-by-one after training costs a
day.

THE ONE THAT WILL BITE YOU
--------------------------
If the CVAT task was pre-loaded with SAM 3 prelabels, EVERY frame in the export
has annotations - including the ones you never opened. They are not empty; they
hold machine guesses with the wrong classes. Training on them is worse than
having no data at all, because the mask geometry is broadly right and only the
label is wrong, so the loss is confident and consistent and the model reliably
learns the wrong class. Nothing else in the pipeline can filter them. That is
what INCLUDE_FRAMES is for, and why it is not optional here.

MULTIPLE DATASETS, ONE BUILD
-----------------------------
SOURCES below is a LIST - one entry per CVAT export you are merging (a weed
session, an onion session, another weed session, ...). Each entry names its own
CVAT export AND its own sessions folder, because they are not always the same
recording campaign under the same parent directory - an onion-only capture done
separately from the weed sessions is the ordinary case, not a special one.

A session's frames are found by trying every SOURCE's IMAGES_ROOT in turn, so a
session need only exist under ONE of them - it is fine, and normal, for two
sources to share the same IMAGES_ROOT.

Frame positions in INCLUDE_FRAMES/EXCLUDE_FRAMES restart at 1 IN EACH SESSION
(not in each source), so once SOURCES has more than one entry, or a source's
export itself spans more than one session, scope every range with its session
id - see INCLUDE_FRAMES below. LIST_FRAMES prints every session from every
source, grouped, with the numbering to use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import prepare_dataset as pdz  # noqa: E402

# #############################################################################
# ##  EDIT EVERYTHING BETWEEN THE HASH LINES                                 ##
# #############################################################################

CONFIG = {
    # -- Where things are ------------------------------------------------------
    # One entry per CVAT export being merged. Add or remove entries freely; a
    # single-source build is just a list of one.
    #
    # DATUMARO_ROOT: the UNZIPPED CVAT 'Datumaro 1.0' export - the folder
    #   containing `annotations/default.json`. May also be one PARENT folder
    #   holding several unzipped exports; they are merged. NOT the SAM 3 output
    #   folder (auto_labels_weeds/<session>/ or auto_labels_onion/<session>/) -
    #   that holds COCO, the format you IMPORT into CVAT, not the one you
    #   export.
    #
    # IMAGES_ROOT: the SESSIONS folder from extract_sessions.py FOR THIS
    #   SOURCE - the one whose CHILDREN are session ids, not a session itself
    #   and not its rgb/ subfolder:
    #
    #     <IMAGES_ROOT>\
    #       vid2_20260108_122731\
    #         rgb\    vid2_20260108_122731_000123.png   <- training images
    #         depth\  vid2_20260108_122731_000123.png   <- same name, NOT one
    #         meta\   pool.csv, frames_index.csv, session.json, calibration.json
    #
    #   Two sources may share the same IMAGES_ROOT (the normal case when they
    #   are sessions from the same capture campaign) or point at entirely
    #   different folders (an onion set recorded separately from the weed
    #   sessions). Images are never copied; manifests point at these files.
    "SOURCES": [
        # ONE SESSION FOLDER PER ENTRY. Each holds annotations/, rgb/, depth/
        # and meta/, so the same path serves as both roots: DATUMARO_ROOT finds
        # its annotations/default.json, and IMAGES_ROOT resolves
        # <root>/rgb/<name> directly.
        #
        # Note Visit1_20260108_132749's frames are named vid3_20260108_132749_*
        # - the folder was renamed after extraction and the files kept their
        # original prefix. Session id comes from the FILENAME, so that session
        # reports as vid3_20260108_132749 in every split and report.
        #
        # Visit2 is from a Mix campaign but only its ONION stretch was
        # annotated; its weed-only frames carry no annotations and are excluded
        # as empty. Being a month later than the Visit1 sessions, it is the only
        # real date variety here - which is why it is the pinned test session.
        {
            "DATUMARO_ROOT": r"E:\Dataset_Vidalia\onions_20260108_1\sessions\Visit1_20260108_132749",
            "IMAGES_ROOT":   r"E:\Dataset_Vidalia\onions_20260108_1\sessions\Visit1_20260108_132749",
        },
        {
            "DATUMARO_ROOT": r"E:\Dataset_Vidalia\onions_20260108_1\sessions\Visit1_20260108_133306",
            "IMAGES_ROOT":   r"E:\Dataset_Vidalia\onions_20260108_1\sessions\Visit1_20260108_133306",
        },
        {
            "DATUMARO_ROOT": r"E:\Dataset_Vidalia\onions_20260108_1\sessions\Visit1_20260108_134015",
            "IMAGES_ROOT":   r"E:\Dataset_Vidalia\onions_20260108_1\sessions\Visit1_20260108_134015",
        },
        {
            "DATUMARO_ROOT": r"E:\Dataset_Vidalia\Mix_2_Visit_2_2026_\sessions\Visit2_20260210_164149",
            "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Mix_2_Visit_2_2026_\sessions\Visit2_20260210_164149",
        },
        {
            "DATUMARO_ROOT": r"E:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions\Visit2_20260210_164614",
            "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions\Visit2_20260210_164614",
        },
        {
            "DATUMARO_ROOT": r"E:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions\Visit2_20260210_164812",
            "IMAGES_ROOT":   r"E:\Dataset_Vidalia\Mix_2_Visit_2_20260210_\sessions\Visit2_20260210_164812",
        },
    ],
    # The UNZIPPED CVAT 'Datumaro 1.0' export - the folder containing
    # `annotations/default.json`. May also be one PARENT folder holding several
    # unzipped exports; they are merged.
    #
    # NOT the SAM 3 output folder (auto_labels_weeds/<session>/). That holds
    # COCO, which is the format you IMPORT into CVAT, not the one you export.
    #"DATUMARO_ROOT": r"E:\Dataset_Vidalia\Weeds_3_good\auto_labels_weeds\vid2_20260108_122731",

    # The sessions root from extract_sessions.py - the folder whose children are
    # session ids. Images are never copied; manifests point at these files.
    #"IMAGES_ROOT": r"E:\Dataset_Vidalia\Weeds_3_good\sessions",

    # Where the dataset manifests are written. Safe to delete and rebuild.
    "OUT_DIR": r"E:\Dataset_Vidalia\training_onion_only_whole_dataset",
    "OUT_DIR": r"E:\Dataset_Vidalia\training_onion_only_whole_dataset",

    # -- Pass 1: look before you select ---------------------------------------
    # True  = print the numbered frame table and STOP. Writes nothing.
    # False = build the dataset.
    "LIST_FRAMES": False,

    # -- Which frames are actually yours --------------------------------------
    # 1-based POSITIONS in the table LIST_FRAMES prints. Also accepts a literal
    # item id, an fnmatch glob, or "@C:/path/keep.txt".
    #   "1-26,28-36,50-59"   ranges, inclusive, comma separated
    #   ""                   keep every frame (ONLY correct if you annotated
    #                        every frame in the task)
    #
    # POSITIONS RESTART AT 1 IN EACH SESSION, not in each SOURCE. Once SOURCES
    # has more than one entry (or even one source whose export spans more than
    # one session), scope every range with its session id - a bare position is
    # then ambiguous and is refused rather than guessed:
    #   "vid2_20260108_122731:1-27,vid2_20260108_122731:29-36,
    #    vid2_20260108_122731:51-60,onion1_20260115_090000:*"
    # `<session>:*` keeps all of that session. With exactly one session across
    # every source, a bare "1-27,29-36,51-60" is still fine.
    "INCLUDE_FRAMES": "",

    # Applied after INCLUDE_FRAMES. Same syntax. Usually left empty.
    "EXCLUDE_FRAMES": "",

    # -- Classes ---------------------------------------------------------------
    # Removed from THIS build only; common/ontology.py is not modified, so a
    # class returns automatically once you have real examples of it. A class
    # with 0-2 instances across the whole set is not learnable and only
    # pollutes the metrics.
    "DROP_CLASSES": [],

    # The inverse, and the better spelling when the build is defined by what it
    # IS rather than by what it lacks. None = no allow-list (use DROP_CLASSES).
    #
    #   "KEEP_CLASSES": ["onion_plant"]     a crop-only detector
    #
    # Why not just list the five weeds in DROP_CLASSES? Because the two pick the
    # same classes today and diverge the moment a class is appended to the
    # ontology - which is the documented way it grows. The deny-list would
    # silently admit that new class into an onion-only dataset; the allow-list
    # keeps meaning what it said. DROP_CLASSES still applies on top, so you can
    # narrow an allow-list without rewriting it.
    "KEEP_CLASSES": ["onion_plant"],

    # -- What these labels ARE -------------------------------------------------
    # Decides what every metric this dataset ever produces MEANS. Recorded in
    # the manifest and restated by preflight at train time, because the build
    # that set it may be months and several rounds behind the run reading it.
    #
    #   "hand_corrected"      a person reviewed and fixed them. The only value
    #                         under which val/test AP estimates real
    #                         performance.
    #   "prelabel_unreviewed" SAM 3 output round-tripped through CVAT without
    #                         correction. Training on it is DISTILLATION - the
    #                         model learns the prelabeler's misses as if they
    #                         were correct, cannot exceed it, and every AP
    #                         measures agreement WITH THE PRELABELER rather
    #                         than with reality. A high score there says
    #                         "faithfully reproduces the teacher", which on a
    #                         crop-safety system is exactly the number not to
    #                         mistake for "finds the crop".
    #   "mixed"               some of each.
    #
    # Nothing in an export can tell these apart - a polygon a person fixed and
    # one they only looked at are identical on disk - so this is a declaration,
    # and the only place the distinction can live.
    "LABEL_PROVENANCE": "prelabel_unreviewed",

    # Check every frame's image is on disk, and exclude + report the ones that
    # are not. An export outlives the frames it describes whenever pool frames
    # were deleted after extraction. False skips the check, which is only
    # useful when the images are on a drive that is not mounted right now.
    "VERIFY_IMAGES": True,

    # -- Splits ----------------------------------------------------------------
    # With ONE session these become contiguous FRAME BLOCKS separated by a
    # discarded gap, not random frames - adjacent video frames are near
    # duplicates and a random split would measure memorisation.
    #
    # At ~45 frames, 0.0 for test is the honest choice: a 9-frame test set has
    # error bars wider than the number it reports. The cost is that val does
    # double duty and its scores are optimistic. Cut a real test set from a
    # HELD-OUT SESSION once you have two or three annotated.
    "VAL_FRACTION": 0.15,
    "TEST_FRACTION": 0.15,

    # PIN a session to test or val by name. This is how a real held-out test
    # set is created, and it is the single most valuable thing you can do for
    # the credibility of every number this project reports.
    #
    # A fraction gives you a test set that changes whenever the dataset grows,
    # so two rounds' scores are not comparable and no improvement can be
    # attributed to anything. A pinned session gives you a FIXED ruler.
    #
    # Pin it BEFORE you annotate it into training - mine_pool.py's
    # HOLDOUT_SESSIONS is where that is enforced - and never mine a pinned
    # session, because active learning selects the frames the model finds
    # hardest, which are exactly the frames it would most benefit from having
    # seen.
    #
    # Choose one that is representative rather than convenient: a different
    # day from the training sessions, and ideally a mixed scene, since mixed is
    # the only scene where the crop-vs-weed decision is actually exercised.
    "HOLDOUT_TEST_SESSIONS": [],
    "HOLDOUT_VAL_SESSIONS": [],

    # Allocate the remaining sessions SEPARATELY WITHIN each scene, so
    # onion_only, weed_only and mixed are each represented in train, val and
    # test in proportion. The scene comes from each session's own
    # meta/session.json (`scene_hint`, set in extract_sessions.py).
    #
    # Without this the allocator is blind to what a session contains. Measured
    # on three onion and three mixed sessions across 40 seeds, validation ended
    # up holding a single scene on 10 of them - a one-in-four chance of a val
    # set that never exercises the crop-vs-weed decision, and nothing in any
    # printed metric would tell you.
    "STRATIFY_BY_SCENE": True,

    # -- How the split is made -------------------------------------------------
    # "auto"        try whole sessions; fall back to frame blocks when that is
    #               impossible (one session, or a class living only in a
    #               held-out one). The default.
    # "session"     same as auto - kept for symmetry and to say so explicitly.
    # "frame_block" ALWAYS split within each session, so the test set is drawn
    #               from every session and every scene.
    #
    # CHOOSE frame_block WHEN YOU HAVE NO SESSION TO SPARE, and understand what
    # you are buying. It gives a test set that is representative of the data you
    # have: every session, every scene, every lighting condition contributes in
    # proportion. It cannot tell you the model works on a NEW drive, because
    # every test frame shares its session's light, soil, growth stage and field
    # with training frames. The resulting score is an upper bound on field
    # performance, not an estimate of it.
    #
    # A held-out session remains strictly better and nothing here replaces it.
    # Record one when you can, and rebuild with HOLDOUT_TEST_SESSIONS.
    "SPLIT_MODE": "frame_block",

    # WHAT UNIT IS HELD OUT - this decides what val/test numbers can mean.
    #
    #   "group"    a whole date+field+camera. Val/test share no day, light or
    #              growth stage with training: the only split that estimates
    #              GENERALISATION.
    #   "session"  a whole recording, from a date training also saw. No frame
    #              leaks, but the conditions are shared, so the score is an
    #              upper bound on a new drive.
    #   "auto"     try group, fall back to session, and SAY which it used.
    #
    # Six sessions recorded on two days are only TWO groups, and two indivisible
    # units cannot fill three splits - test comes out empty and training sees
    # one date. "auto" notices that and drops to session granularity, which is
    # still strictly better than frame blocks: those put val and test inside the
    # same recording as train, where adjacent frames are near-duplicates.
    "SPLIT_GRANULARITY": "auto",

    # Cut each session into this many stretches and split WITHIN each one, so
    # val and test are sampled from several places along every drive.
    #
    # With 1 block, test is always the LAST stretch of every recording - which
    # on this data is systematically different: later in the day, further along
    # the bed, often the headland where the rig turns. A test set made only of
    # drive-ends measures drive-ends. 4 is a reasonable choice; more blocks cost
    # more gap frames, one seam per boundary.
    "BLOCKS_PER_SESSION": 3,

    # Frames discarded at each seam. Adjacent pool frames still overlap
    # heavily, and this buffer is the only thing separating a test frame from
    # its near-duplicate in training.
    #
    # COUNTED IN POOL FRAMES, NOT VIDEO FRAMES. If curation kept every 5th
    # frame, GAP_FRAMES=2 buys 10 video frames of separation - about a third of
    # a second at 30 fps, which is nothing. The build MEASURES the real
    # separation in video frames and warns when it is under 60; raise this
    # until that warning clears. The cost is a handful of annotated frames,
    # which is far cheaper than a test score wrong in the optimistic direction.
    "GAP_FRAMES": 12,

    # -- Safety ----------------------------------------------------------------
    # False (default) refuses to write when the annotation contract is violated.
    # True writes anyway for triage - read annotations_needing_correction.json
    # and do NOT train on the result.
    "ALLOW_ERRORS": False,

    # Frames with zero annotations are excluded by default: an empty frame is
    # almost always one you did not reach, and training on it teaches the model
    # that a frame full of weeds is background. Set True only if your empty
    # frames are deliberately bare ground.
    "KEEP_EMPTY_FRAMES": False,

    "SEED": 1234,
}

# #############################################################################
# ##  Nothing below here needs editing                                       ##
# #############################################################################


def _resolve_sources(c):
    """Validate SOURCES and split it into (datumaro_roots, images_roots).

    Every error names WHICH source is wrong (by its 1-based position in the
    list), because 'IMAGES_ROOT does not exist' is not actionable once there
    are several of them."""
    sources = c.get("SOURCES")
    if not sources:
        raise SystemExit(
            f"ERROR: CONFIG['SOURCES'] is empty. Edit {Path(__file__).name} "
            f"and add at least one {{'DATUMARO_ROOT': ..., 'IMAGES_ROOT': "
            f"...}} entry.")

    datumaro_roots, images_roots = [], []
    for i, src in enumerate(sources, start=1):
        for key in ("DATUMARO_ROOT", "IMAGES_ROOT"):
            if not str(src.get(key, "")).strip():
                raise SystemExit(
                    f"ERROR: SOURCES[{i}]['{key}'] is empty. Edit "
                    f"{Path(__file__).name} and set it.")
        dr = Path(src["DATUMARO_ROOT"])
        if not dr.exists():
            raise SystemExit(
                f"ERROR: SOURCES[{i}]['DATUMARO_ROOT'] does not exist:\n"
                f"    {dr}\n"
                f"Point it at the UNZIPPED CVAT 'Datumaro 1.0' export - the "
                f"folder containing annotations/default.json.")
        ir = Path(src["IMAGES_ROOT"])
        if not ir.exists():
            raise SystemExit(
                f"ERROR: SOURCES[{i}]['IMAGES_ROOT'] does not exist:\n"
                f"    {ir}\n"
                f"This is the sessions root from extract_sessions.py, whose "
                f"children are session id folders.")
        datumaro_roots.append(dr)
        if ir not in images_roots:      # sharing one root across sources is normal
            images_roots.append(ir)
    return datumaro_roots, images_roots


def main(cfg=None):
    c = dict(CONFIG if cfg is None else cfg)

    if not str(c.get("OUT_DIR", "")).strip():
        raise SystemExit(f"ERROR: CONFIG['OUT_DIR'] is empty. Edit "
                         f"{Path(__file__).name} and set it.")

    datumaro_roots, images_roots = _resolve_sources(c)

    if c["LIST_FRAMES"]:
        label = (f"{len(datumaro_roots)} source(s)" if len(datumaro_roots) > 1
                else str(datumaro_roots[0]))
        print(f"\nListing frames in {label}\n")
        pdz.list_frames(datumaro_roots)
        print("\n" + "=" * 74)
        print("NOTHING WAS WRITTEN. This was the listing pass.")
        print("  1. Find the positions of the frames you hand-corrected.")
        print("     Frames you never opened still have SAM's annotations, so")
        print("     'has annotations' does NOT mean 'verified'. A weed frame")
        print("     containing cutleaf_evening_primrose or wild_radish is")
        print("     certainly yours - SAM only ever proposes grass_weed,")
        print("     weed_cluster and other_weed. An ONION session has no such")
        print("     tell (only one class exists), so track which onion")
        print("     positions you corrected as you go through the CVAT task.")
        print("  2. Set INCLUDE_FRAMES to those positions.")
        print("  3. Set LIST_FRAMES = False and run again.")
        print("=" * 74 + "\n")
        return

    if not str(c["INCLUDE_FRAMES"]).strip():
        print("\n[!] INCLUDE_FRAMES is empty - EVERY frame in the export will "
              "be used.\n    That is correct only if you hand-corrected all of "
              "them. If the task\n    was pre-loaded with SAM 3 prelabels, the "
              "frames you never opened carry\n    machine guesses with the "
              "wrong classes and will be trained on.\n")

    pdz.build(
        datumaro_roots, images_roots, c["OUT_DIR"],
        include_frames=c["INCLUDE_FRAMES"] or None,
        exclude_frames=c["EXCLUDE_FRAMES"] or None,
        drop_classes=c["DROP_CLASSES"],
        keep_classes=c.get("KEEP_CLASSES"),
        label_provenance=c.get("LABEL_PROVENANCE", "hand_corrected"),
        # The runner has real paths, so it checks the images are actually there
        # and says how many are not - rather than letting COCO export die on
        # the first one, hours later, naming a single file.
        verify_images=c.get("VERIFY_IMAGES", True),
        val_fraction=c["VAL_FRACTION"], test_fraction=c["TEST_FRACTION"],
        gap_frames=c["GAP_FRAMES"], seed=c["SEED"],
        keep_empty_frames=c["KEEP_EMPTY_FRAMES"],
        strict=not c["ALLOW_ERRORS"],
        split_mode=c.get("SPLIT_MODE", "auto"),
        split_granularity=c.get("SPLIT_GRANULARITY", "auto"),
        blocks_per_session=c.get("BLOCKS_PER_SESSION", 1),
        holdout_test=c.get("HOLDOUT_TEST_SESSIONS") or (),
        holdout_val=c.get("HOLDOUT_VAL_SESSIONS") or (),
        stratify_by_scene=c.get("STRATIFY_BY_SCENE", True),
    )
    print("\nNext: edit seeweed3d/training/train_model.py and run it.\n")


if __name__ == "__main__":
    main()
