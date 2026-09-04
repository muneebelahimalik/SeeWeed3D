"""Plants nobody labelled: the audit that decides whether a drive can be used
as WHOLE FRAMES or only as a source of instance cut-outs.

The distinction the whole module rests on: a frame whose masks all sit a
pixel inside their leaves has a large unclaimed FRACTION and nothing missing,
while a frame with one unlabelled seedling among forty labelled plants has a
tiny fraction and a real hole in it. Counting blobs separates those; a fraction
cannot.
"""
import numpy as np
import pytest

from conftest import load_script

mp = load_script("annotation/missed_plants.py")

from common.vegetation import unclaimed_blobs  # noqa: E402

H = W = 240


def _blob(y, x, s, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[y:y + s, x:x + s] = True
    return m


# --------------------------------------------------------------------------
# blobs, not fractions - the reason this is worth measuring at all


def test_one_unlabelled_plant_is_found():
    veg = _blob(20, 20, 40) | _blob(120, 120, 40)
    n, px, mask = unclaimed_blobs(veg, _blob(20, 20, 40))
    assert n == 1 and px > 1000
    assert mask[130, 130] and not mask[30, 30]


def test_a_thin_rim_of_annotation_slop_is_not_a_missed_plant():
    """A hand-drawn polygon sits a pixel or two inside the leaf it traces. A
    check that fired on that would reject every frame in the project."""
    veg = _blob(50, 50, 60)
    claimed = _blob(52, 52, 56)
    assert unclaimed_blobs(veg, claimed)[0] == 0


def test_a_big_rim_is_still_not_a_plant_but_a_real_gap_is():
    """The dilation absorbs slop; it must not absorb a plant sitting next to a
    mask."""
    veg = _blob(50, 50, 60) | _blob(50, 130, 40)
    assert unclaimed_blobs(veg, _blob(50, 50, 60))[0] == 1


def test_specks_are_discarded():
    """A report full of leaf tips and green debris is one nobody reads."""
    veg = _blob(20, 20, 40) | _blob(150, 150, 6)
    assert unclaimed_blobs(veg, _blob(20, 20, 40))[0] == 0


def test_a_tiny_fraction_can_still_be_a_real_hole():
    """Forty labelled plants and one missed seedling: the fraction is noise and
    the blob count is 1. This is the case a fraction-based check cannot see."""
    veg = np.zeros((H, W), bool)
    claimed = np.zeros((H, W), bool)
    for i in range(5):
        for j in range(5):
            b = _blob(10 + i * 38, 10 + j * 38, 30)   # spans 10..192
            veg |= b
            claimed |= b
    veg |= _blob(205, 205, 20)                 # the missed one, clear of them
    n, px, _ = unclaimed_blobs(veg, claimed)
    assert n == 1
    assert px / veg.sum() < 0.02, "a fraction check would call this clean"


def test_nothing_claimed_means_everything_is_unclaimed():
    veg = _blob(20, 20, 40) | _blob(120, 120, 40)
    assert unclaimed_blobs(veg, np.zeros((H, W), bool))[0] == 2


def test_no_vegetation_means_nothing_missed():
    assert unclaimed_blobs(np.zeros((H, W), bool), _blob(0, 0, 10))[0] == 0


# --------------------------------------------------------------------------
# what the audit says to DO


def _frames(counts):
    return {f"f{i:03d}": {"n_missed": n, "missed_px": n * 900, "veg_px": 90000,
                          "missed_frac": n * 0.01}
            for i, n in enumerate(counts)}


def test_the_verdict_counts_clean_frames_in_its_denominator():
    """The share has to be over EVERY audited frame. Counting only the frames
    that already have a patch answers 'of the frames with a problem, how many
    have a big one' - which is near 100% by construction and says nothing about
    the drive. It read 18 of 89 (20%, NOT SAFE) on a run where 200 frames were
    audited and the honest number was 9% (MOSTLY CLEAN), and the composites it
    condemned were about to be pulled from training on the strength of it."""
    dirty = _frames([3] * 18)
    assert mp.verdict(dirty).startswith("NOT SAFE")
    with_clean = _frames([3] * 18 + [0] * 182)
    assert mp.verdict(with_clean).startswith("MOSTLY CLEAN"), (
        "182 clean frames did not move the share, so they are not being "
        "counted")


def test_a_clean_frame_is_counted_even_though_it_is_not_listed():
    """Where the bug actually lived. verdict() was always right given complete
    input; the audit loop never gave it one, dropping every clean frame on the
    way in. So the invariant is pinned at the point the frame is filed."""
    per_frame, records = {}, {}
    mp.record_frame(per_frame, records, "s/clean", "clean",
                    {"n_missed": 0}, "clean.png", (4, 4))
    mp.record_frame(per_frame, records, "s/dirty", "dirty",
                    {"n_missed": 5}, "dirty.png", (4, 4))
    assert set(per_frame) == {"clean", "dirty"}, (
        "a clean frame left the denominator, which inflates every share this "
        "tool reports")
    assert set(records) == {"s/dirty"}, "clean frames should not get overlays"


def test_listing_fewer_frames_does_not_change_the_measurement():
    """MIN_BLOBS_TO_REPORT is about attention. If raising it also moved the
    verdict, the share would be a property of the threshold, not the drive."""
    per_frame, records = {}, {}
    for i, n in enumerate([0, 1, 2, 5, 5]):
        mp.record_frame(per_frame, records, f"s/f{i}", f"f{i}",
                        {"n_missed": n}, f"f{i}.png", (4, 4), min_blobs=4)
    assert len(per_frame) == 5 and len(records) == 2


def test_a_drive_with_nothing_missed_is_clean_not_unaudited():
    """The empty-audit message and a perfect drive are different facts, and
    dropping clean frames made them the same one."""
    assert mp.verdict(_frames([0] * 40)).startswith("CLEAN")
    assert "No frames" in mp.verdict({})


def test_a_clean_drive_is_told_to_train_on_whole_frames():
    """Whole frames keep real weed-beside-weed context and real lighting - all
    things a cut-out loses - so 'clean' must recommend keeping them."""
    v = mp.verdict(_frames([0, 0, 0, 0]))
    assert v.startswith("CLEAN") and "WHOLE" in v
    assert "misses dark" in v, "the prior's blind spot belongs in the verdict"


def test_a_dirty_drive_is_told_to_become_a_cutout_source():
    """The actionable half: a cut-out carries the labelled pixels and leaves
    the missed ones behind."""
    v = mp.verdict(_frames([4, 5, 3, 6, 4]))
    assert "NOT SAFE AS WHOLE FRAMES" in v
    assert "compose_mixed" in v and "CUT-OUT SOURCE" in v


def test_a_few_bad_frames_do_not_condemn_the_drive():
    v = mp.verdict(_frames([0] * 40 + [5]))
    assert v.startswith("MOSTLY CLEAN")
    assert "exclude those few" in v


def test_an_empty_audit_says_so_rather_than_passing():
    assert "No frames" in mp.verdict({})


def test_the_verdict_threshold_is_the_configured_one():
    frames = _frames([2, 2, 2, 2])
    assert mp.verdict(frames, unsafe=2).startswith("NOT SAFE")
    assert not mp.verdict(frames, unsafe=3).startswith("NOT SAFE")


# --------------------------------------------------------------------------
# A settled drive. This audit is a heuristic over a colour prior, and it
# re-runs its opinion every time - so it has to be able to say "somebody
# already decided this, here is what they chose" instead of arguing again.
# --------------------------------------------------------------------------
def test_a_settled_drive_leads_with_the_decision_not_the_recommendation():
    v = mp.verdict(_frames([9] * 7), decided=(mp.WHOLE, "mixed.py, by hand"))
    assert v.startswith("SETTLED"), (
        "the decision has to come first - a reader who stops at the first "
        "words must not come away thinking the drive is unused")
    assert "OVERRULED" in v and "mixed.py, by hand" in v


def test_a_settled_drive_still_shows_what_the_audit_would_have_said():
    """Suppressing the recommendation would hide the disagreement, and the
    disagreement is the information - it is what a reader needs to judge
    whether the decision still holds."""
    v = mp.verdict(_frames([9] * 7), decided=(mp.WHOLE, "because"))
    assert "NOT SAFE AS WHOLE FRAMES" in v and "7 of 7" in v


def test_a_decision_the_audit_agrees_with_is_not_called_an_overrule():
    v = mp.verdict(_frames([9] * 7), decided=(mp.CUTOUT, "because"))
    assert "agrees" in v and "OVERRULED" not in v


def test_a_settled_drive_that_got_worse_can_still_say_so():
    """The counts are never suppressed, so re-running this on a decided drive
    is still how you learn the decision needs revisiting."""
    clean = mp.verdict(_frames([0, 0]), decided=(mp.WHOLE, "because"))
    dirty = mp.verdict(_frames([9] * 7), decided=(mp.WHOLE, "because"))
    assert "agrees" in clean and "OVERRULED" in dirty


def test_a_decision_reaches_the_drive_however_its_name_is_written():
    """The folder is `Mix_raj_Batch 01` and the session id is
    `Mix_raj_Batch_01`. A decision recorded against one that silently matches
    nothing is worse than no decision at all."""
    assert mp.decision_for("Mix_raj_Batch 01", {"Mix_raj_Batch_01": (mp.WHOLE, "x")})
    assert mp.decision_for("Mix_raj_Batch_01", {"Mix_raj_Batch 01": (mp.WHOLE, "x")})
    assert not mp.decision_for("vid2_20260108_122731",
                               {"Mix_raj_Batch_01": (mp.WHOLE, "x")})


def test_a_decision_about_a_drive_nobody_audits_is_reported_as_stale():
    stale = mp.stale_decisions(["Mix_raj_Batch 01"],
                               {"Mix_raj_Batch_01": (mp.WHOLE, "x"),
                                "vid9_gone": (mp.CUTOUT, "x")})
    assert stale == ["vid9_gone"]


def test_every_decision_matches_what_the_mixed_build_actually_does():
    """DECIDED records a decision; mixed.py MAKES it. If they drift, this file
    starts printing a settled use that no build honours - which is exactly the
    confident-and-wrong line it was added to remove."""
    import ntpath
    from training.datasets import mixed
    for sess, (use, why) in mp.DECIDED.items():
        cutout = any(mp._norm(sess) == mp._norm(ntpath.basename(p.rstrip("\\/")))
                     for p in mixed.CUTOUT_ONLY_SESSIONS)
        assert use == (mp.CUTOUT if cutout else mp.WHOLE), (
            f"{sess} is recorded as {use!r} here but mixed.py "
            f"{'excludes' if cutout else 'includes'} it as whole frames")
        assert "mixed.py" in why, (
            f"{sess}'s reason does not say where the decision lives, so "
            f"nobody can go and change it")


# --------------------------------------------------------------------------
# the frame audit and the report


def test_audit_frame_counts_against_a_supplied_vegetation_mask():
    """The prior is supplied so the test does not depend on ExG thresholds
    against a synthetic image - what is under test is the accounting."""
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(20, 20, 40) | _blob(120, 120, 40)
    rec, mask = mp.audit_frame(bgr, _blob(20, 20, 40), veg=veg)
    assert rec["n_missed"] == 1
    assert rec["veg_px"] == int(veg.sum())
    assert 0 < rec["missed_frac"] < 1
    assert mask.any()


def test_worst_ranks_by_patches_then_area():
    per = {"a": {"n_missed": 1, "missed_px": 5000},
           "b": {"n_missed": 4, "missed_px": 100},
           "c": {"n_missed": 0, "missed_px": 0}}
    assert mp.worst(per, 3) == ["b", "a"], "a clean frame is not 'worst'"


def test_the_report_carries_the_caveat_that_a_patch_is_not_a_proven_weed():
    """The prior calls moss and debris vegetation and misses dark seedlings, so
    the number cannot settle it and the report must not imply it does."""
    txt = mp.format_report({"vid3": _frames([2, 0, 3])})
    assert "PLACE TO LOOK" in txt
    assert "not proof of a clean frame either" in txt


def test_the_report_names_the_worst_frames():
    txt = mp.format_report({"vid3": _frames([0, 7, 0])})
    assert "f001" in txt and "7 patch" in txt


def test_summarise_counts_frames_and_patches():
    s = mp.summarise(_frames([0, 2, 3]))
    assert s["frames"] == 3 and s["missed_blobs"] == 5
    assert s["frames_with_missed"] == 2


def test_summarise_of_nothing_does_not_divide_by_zero():
    assert mp.summarise({})["mean_blobs_per_frame"] == 0.0


def test_draw_marks_missed_vegetation_and_outlines_what_was_claimed():
    bgr = np.zeros((H, W, 3), np.uint8)
    vis = mp.draw(bgr, _blob(20, 20, 40), _blob(120, 120, 40))
    assert vis.shape == bgr.shape
    assert vis[140, 140, 2] > vis[140, 140, 0], "missed vegetation drawn red"


# --------------------------------------------------------------------------
# A rim is told from a plant by WHERE it sits, not by how big it is. Getting
# that wrong biases the check against contact - the case that matters most.
# --------------------------------------------------------------------------
def test_a_rim_can_be_bigger_than_a_seedling_and_still_not_be_one():
    """A 2px rim around a 40px plant is 304 px - larger than many real
    seedlings. Size alone cannot separate them, which is why the test is
    positional."""
    veg = _blob(50, 50, 40)
    claimed = _blob(52, 52, 36)
    rim_px = int((veg & ~claimed).sum())
    assert rim_px > 300, "the rim really is bigger than the size floor"
    assert unclaimed_blobs(veg, claimed)[0] == 0


def test_a_small_unlabelled_plant_TOUCHING_a_labelled_one_is_still_found():
    """The bias worth avoiding. Thresholding on area after subtracting the
    slop band eats the near edge of anything adjacent, so an 18x18 seedling
    against a labelled plant drops under the floor and vanishes - and a weed
    touching a crop is the exact case this project exists to get right."""
    labelled = _blob(50, 50, 60)
    veg = labelled | _blob(50, 110, 18)          # 324 px, flush against it
    n, _, mask = unclaimed_blobs(veg, labelled)
    assert n == 1
    assert mask[58, 118], "the touching seedling is what got marked"


def test_the_same_seedling_is_found_touching_or_apart():
    """Adjacency must not change whether a plant counts - otherwise the audit
    under-reports exactly where contact scenes are densest."""
    labelled = _blob(50, 50, 60)
    touching = unclaimed_blobs(labelled | _blob(50, 110, 18), labelled)[0]
    apart = unclaimed_blobs(labelled | _blob(50, 140, 18), labelled)[0]
    assert touching == apart == 1


def test_a_mask_falling_well_short_of_its_plant_is_reported():
    """A rim wider than the slop band is not slop - it is a mask that missed
    a third of its own plant, and that is worth seeing."""
    veg = _blob(50, 50, 60)
    claimed = _blob(62, 62, 36)                  # 12px short on two sides
    assert unclaimed_blobs(veg, claimed)[0] >= 1


# --------------------------------------------------------------------------
# The other direction: a mask that reaches PAST its plant. Invisible from the
# union, because covering soil takes nothing away from it.
# --------------------------------------------------------------------------
def test_a_mask_covering_soil_is_invisible_to_the_union_check():
    """The reason it needed its own measurement. A mask twice the size of its
    plant leaves nothing unclaimed - the union check reports a perfect frame."""
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(50, 50, 20)                       # a small plant
    fat = _blob(40, 40, 60)                       # a mask 9x its area
    rec, _ = mp.audit_frame(bgr, fat, veg=veg, instances=[fat])
    assert rec["n_missed"] == 0, "nothing is unclaimed - the union looks clean"
    assert rec["n_soil_masks"] == 1, "but the mask is mostly soil"


def test_a_tight_mask_is_not_reported_as_claiming_soil():
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(50, 50, 40)
    rec, _ = mp.audit_frame(bgr, veg, veg=veg, instances=[_blob(50, 50, 40)])
    assert rec["n_soil_masks"] == 0
    assert rec["median_instance_veg"] == pytest.approx(1.0)


def test_the_soil_threshold_is_forgiving_by_default():
    """A polygon around a thin curved leaf encloses soil however carefully it
    is drawn, and erring large on crop masks is project policy. Only a badly
    blobby mask should trip this."""
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(50, 50, 30)                       # 900 px of plant
    mask = _blob(48, 48, 44)                      # 1936 px -> 46% vegetation
    rec, _ = mp.audit_frame(bgr, mask, veg=veg, instances=[mask])
    assert rec["n_soil_masks"] == 0, "46% vegetation is normal for thin foliage"


def test_both_failure_modes_are_counted_in_one_frame():
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(20, 20, 20) | _blob(150, 150, 40)
    fat = _blob(10, 10, 60)                       # swallows the first plant
    rec, _ = mp.audit_frame(bgr, fat, veg=veg, instances=[fat])
    assert rec["n_soil_masks"] == 1               # too big here
    assert rec["n_missed"] == 1                   # and missed the other plant


def test_an_empty_mask_is_not_counted_as_claiming_soil():
    bgr = np.zeros((H, W, 3), np.uint8)
    veg = _blob(50, 50, 30)
    rec, _ = mp.audit_frame(bgr, veg, veg=veg,
                            instances=[np.zeros((H, W), bool)])
    assert rec["n_soil_masks"] == 0


def test_the_report_shows_both_directions():
    per = {"f0": {"n_missed": 2, "missed_px": 900, "veg_px": 9000,
                  "missed_frac": 0.1, "n_instances": 12, "n_soil_masks": 3,
                  "median_instance_veg": 0.7}}
    txt = mp.format_report({"vid3": per})
    assert "MASK TOO SMALL" in txt and "MASK TOO BIG" in txt
    assert "swallowed an adjacent weed" in txt
    assert "trains that weed as CROP" in txt


def test_draw_marks_a_soil_claiming_mask_differently():
    bgr = np.zeros((H, W, 3), np.uint8)
    tight, fat = _blob(20, 20, 30), _blob(120, 120, 40)
    vis = mp.draw(bgr, tight | fat, np.zeros((H, W), bool),
                  instances=[tight, fat], soil_idx=[1])
    assert vis.shape == bgr.shape
