#!/usr/bin/env python3
"""
SeeWeed3D - the active-learning ledger: what went out, and what came back.

`mine_pool.py` ranks the unlabelled pool and exports a batch. Run once, that is
a useful tool. Run repeatedly - which is the whole point of active learning -
and two failures appear that nothing in a single pass can see.

RE-SENDING FRAMES THAT ARE ALREADY OUT
--------------------------------------
Mining skips frames that are already ANNOTATED, by reading the built dataset.
But a frame exported last Tuesday and sitting in a CVAT task nobody has
finished is not annotated yet, so the next round scores it exactly as before -
same model, same score, same diversity - and sends it again. The annotator gets
a batch they have half-done, and the round buys less than it should. The ledger
records what was exported so a later round can exclude it.

MEASURING WHETHER THE ROUND WAS WORTH IT
----------------------------------------
Active learning is a claim: that THESE frames teach more than random ones. The
claim is testable, and the test is the metric before the round against the
metric after, on a test set that never changes. Without a ledger nobody
computes it, and the loop turns into ritual - annotate 60, retrain, feel
progress. A round that moved nothing is worth knowing about: it usually means
the bottleneck moved somewhere else, most often to a class with too few
instances to learn at all.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not select from, score, or export any HOLDOUT session. A test set that
receives annotation effort chosen BY the model is no longer a measurement of
that model - the frames it finds hardest are exactly the ones it would most
benefit from having seen. `mine_pool` already refuses holdout sessions; the
ledger records the holdout with each round so a set silently redefined between
rounds is visible in the history rather than only in the numbers.

It also does not pseudo-label. Every exported prediction goes to a human. See
docs/dataset_growth.md for why that line is where it is.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER_NAME = "al_rounds.json"

#: Round lifecycle. `exported` frames are out with an annotator; `merged` ones
#: have come back and are in the dataset. Only `exported` blocks re-selection -
#: a merged round is already excluded by the dataset's own item ids.
STATES = ("exported", "merged", "abandoned")


def ledger_path(dataset_dir):
    return Path(dataset_dir) / LEDGER_NAME


def load_ledger(dataset_dir):
    """The ledger, or an empty one. A missing ledger is the normal state of a
    project that has not run a round yet, not an error."""
    path = ledger_path(dataset_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"rounds": []}
    if not isinstance(raw, dict) or not isinstance(raw.get("rounds"), list):
        return {"rounds": []}
    return raw


def save_ledger(dataset_dir, ledger):
    path = ledger_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return path


def next_round_number(ledger):
    return 1 + max([int(r.get("round", 0)) for r in ledger.get("rounds", [])]
                   or [0])


def frames_in_flight(ledger):
    """Item ids exported but not yet merged back.

    These are the frames a new round must not select again. An `abandoned`
    round releases its frames deliberately - that is the escape hatch for a
    CVAT task that was never going to be finished."""
    out = set()
    for r in ledger.get("rounds", []):
        if r.get("state") == "exported":
            out |= set(r.get("item_ids") or [])
    return out


def record_export(dataset_dir, *, item_ids, checkpoint, conf, batch_size,
                  out_dir, sessions_root, holdout_sessions=(),
                  scores=None, notes=""):
    """Append a round in the `exported` state and return it."""
    ledger = load_ledger(dataset_dir)
    rnd = {
        "round": next_round_number(ledger),
        "state": "exported",
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "merged_utc": None,
        "n_frames": len(item_ids),
        "item_ids": sorted(item_ids),
        "checkpoint": str(checkpoint),
        "conf": conf,
        "batch_size": batch_size,
        "export_dir": str(out_dir),
        "sessions_root": str(sessions_root),
        # Recorded per round on purpose: a holdout quietly redefined between
        # rounds turns a test score into a training score, and the diff between
        # two rounds' holdouts is the only place that is visible.
        "holdout_sessions": sorted(holdout_sessions or []),
        "score_summary": scores or {},
        "metrics_before": None,
        "metrics_after": None,
        "notes": notes,
    }
    ledger.setdefault("rounds", []).append(rnd)
    save_ledger(dataset_dir, ledger)
    return rnd


def mark_merged(dataset_dir, round_number=None, metrics_after=None):
    """Mark a round merged, optionally attaching the metrics measured after it.

    With no round number the newest exported round is used, which is what a
    linear annotate-merge-retrain loop always means."""
    ledger = load_ledger(dataset_dir)
    rounds = ledger.get("rounds", [])
    if round_number is None:
        candidates = [r for r in rounds if r.get("state") == "exported"]
        if not candidates:
            raise ValueError("no exported round to merge - every round is "
                             "already merged or abandoned.")
        target = max(candidates, key=lambda r: int(r.get("round", 0)))
    else:
        matches = [r for r in rounds if int(r.get("round", 0)) == int(round_number)]
        if not matches:
            raise ValueError(f"round {round_number} is not in the ledger.")
        target = matches[0]
    target["state"] = "merged"
    target["merged_utc"] = datetime.now(timezone.utc).isoformat()
    if metrics_after is not None:
        target["metrics_after"] = metrics_after
    save_ledger(dataset_dir, ledger)
    return target


def abandon(dataset_dir, round_number, reason=""):
    """Release a round's frames back into the pool.

    For a CVAT task that will not be finished. Deliberate and recorded, so the
    frames reappearing in a later batch is explicable rather than a bug."""
    ledger = load_ledger(dataset_dir)
    for r in ledger.get("rounds", []):
        if int(r.get("round", 0)) == int(round_number):
            r["state"] = "abandoned"
            r["notes"] = (r.get("notes", "") + f" abandoned: {reason}").strip()
            save_ledger(dataset_dir, ledger)
            return r
    raise ValueError(f"round {round_number} is not in the ledger.")


def attach_metrics(dataset_dir, round_number, *, before=None, after=None):
    """Record the measurement either side of a round.

    Kept separate from mark_merged because the two happen at different times:
    you merge, then retrain, then evaluate, and only then is `after` knowable."""
    ledger = load_ledger(dataset_dir)
    for r in ledger.get("rounds", []):
        if int(r.get("round", 0)) == int(round_number):
            if before is not None:
                r["metrics_before"] = before
            if after is not None:
                r["metrics_after"] = after
            save_ledger(dataset_dir, ledger)
            return r
    raise ValueError(f"round {round_number} is not in the ledger.")


def _delta(before, after, key):
    try:
        return round(float(after[key]) - float(before[key]), 4)
    except (TypeError, KeyError, ValueError):
        return None


def history(dataset_dir, metric_keys=("mAP", "mAP50", "weed_recall")):
    """One row per round: size, state, and what the metrics did.

    The `delta` column is the only honest answer to 'is the active learning
    working'. A round whose deltas are all zero or negative is information: the
    frames were not the bottleneck, and the next round should change what it
    optimises rather than repeat it larger."""
    rows = []
    for r in sorted(load_ledger(dataset_dir).get("rounds", []),
                    key=lambda r: int(r.get("round", 0))):
        before, after = r.get("metrics_before"), r.get("metrics_after")
        deltas = {}
        if isinstance(before, dict) and isinstance(after, dict):
            for k in metric_keys:
                d = _delta(before, after, k)
                if d is not None:
                    deltas[k] = d
        rows.append({"round": int(r.get("round", 0)),
                     "state": r.get("state", "?"),
                     "n_frames": int(r.get("n_frames", 0)),
                     "checkpoint": r.get("checkpoint", ""),
                     "exported_utc": r.get("exported_utc", ""),
                     "deltas": deltas})
    return rows


def format_history(rows):
    if not rows:
        return "  no active-learning rounds recorded yet."
    out = ["  round  state      frames  metric change",
           "  -----  ---------  ------  -------------"]
    for r in rows:
        d = ", ".join(f"{k} {v:+.4f}" for k, v in (r["deltas"] or {}).items())
        out.append(f"  {r['round']:>5}  {r['state']:<9}  {r['n_frames']:>6}  "
                   f"{d or '-'}")
    return "\n".join(out)
