#!/usr/bin/env python3
"""
SeeWeed3D - drive the active-learning loop from the command line.

    python -m seeweed3d.training.al_round status  --dataset E:\\Dataset_Vidalia\\training1
    python -m seeweed3d.training.al_round merge   --dataset ...
    python -m seeweed3d.training.al_round metrics --dataset ... --round 3 \\
            --after E:\\Dataset_Vidalia\\training1\\run5\\eval\\metrics.json
    python -m seeweed3d.training.al_round abandon --dataset ... --round 2 \\
            --reason "task never finished"

THE LOOP THIS SERVES
--------------------
    1. mine_pool.py          rank the pool, export a batch, record round N
    2. CVAT                  a human corrects it
    3. make_dataset.py       merge the corrected export into the dataset
    4. al_round merge        mark round N as returned
    5. train                 retrain on the larger dataset
    6. eval_seg / analyze_run   measure on the UNCHANGED test set
    7. al_round metrics      attach the measurement to round N
    8. back to 1

Steps 4 and 7 are the ones people skip, and skipping them is what turns active
learning into ritual. Without 4, the next round re-selects frames that are
still out with an annotator. Without 7, nobody ever finds out whether the last
60 frames taught the model anything - and a round that moved nothing is
information, not a failure: it means the bottleneck is somewhere the frame
selection cannot reach, most often a class with too few instances to learn.

The test set must not change between rounds. A metric delta across two
different test sets measures the test sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import al_rounds  # noqa: E402


def _load_metrics(path):
    """Metrics from an eval JSON, flattened to the scalar keys we compare.

    Accepts either a flat {metric: value} file or an eval report with a nested
    "overall" block, because both shapes exist in this repo's history and a
    hand-edited file is a legitimate input too."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"ERROR: {path} is not a JSON object.")
    flat = dict(raw.get("overall") or {}) if isinstance(
        raw.get("overall"), dict) else {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            flat.setdefault(k, v)
    if not flat:
        raise SystemExit(
            f"ERROR: no scalar metrics found in {path}. Expected a flat "
            f"{{name: number}} object or one with an 'overall' block.")
    return flat


def cmd_status(a):
    rows = al_rounds.history(a.dataset)
    print(al_rounds.format_history(rows))
    in_flight = al_rounds.frames_in_flight(al_rounds.load_ledger(a.dataset))
    if in_flight:
        print(f"\n  {len(in_flight)} frame(s) are out with an annotator and "
              f"will be skipped by the next mine_pool run.")
    unmeasured = [r for r in rows if r["state"] == "merged" and not r["deltas"]]
    if unmeasured:
        print(f"\n  [!] round(s) "
              f"{', '.join(str(r['round']) for r in unmeasured)} were merged "
              f"but never measured. Until a round has metrics either side of "
              f"it, there is no evidence the selection is buying anything.")
    return 0


def cmd_merge(a):
    rnd = al_rounds.mark_merged(a.dataset, a.round)
    print(f"  round {rnd['round']} marked merged "
          f"({rnd['n_frames']} frames released from in-flight).")
    print(f"  Now retrain, evaluate on the UNCHANGED test set, and attach the "
          f"result:")
    print(f"      python -m seeweed3d.training.al_round metrics "
          f"--dataset \"{a.dataset}\" --round {rnd['round']} --after <metrics.json>")
    return 0


def cmd_metrics(a):
    before = _load_metrics(a.before) if a.before else None
    after = _load_metrics(a.after) if a.after else None
    if before is None and after is None:
        raise SystemExit("ERROR: pass --before and/or --after.")
    rnd = al_rounds.attach_metrics(a.dataset, a.round, before=before,
                                   after=after)
    print(f"  round {rnd['round']} metrics updated.")
    print(al_rounds.format_history(al_rounds.history(a.dataset)))
    return 0


def cmd_abandon(a):
    rnd = al_rounds.abandon(a.dataset, a.round, a.reason)
    print(f"  round {rnd['round']} abandoned; its {rnd['n_frames']} frame(s) "
          f"return to the pool and may be selected again.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", required=True,
                        help="OUT_DIR from make_dataset.py (holds al_rounds.json)")
        return sp

    common(sub.add_parser("status", help="round history and metric deltas")
           ).set_defaults(fn=cmd_status)

    sp = common(sub.add_parser("merge", help="mark a round's frames returned"))
    sp.add_argument("--round", type=int, default=None,
                    help="default: the newest exported round")
    sp.set_defaults(fn=cmd_merge)

    sp = common(sub.add_parser("metrics", help="attach measurements to a round"))
    sp.add_argument("--round", type=int, required=True)
    sp.add_argument("--before", help="metrics JSON from before the round")
    sp.add_argument("--after", help="metrics JSON from after retraining")
    sp.set_defaults(fn=cmd_metrics)

    sp = common(sub.add_parser("abandon", help="release a stale round's frames"))
    sp.add_argument("--round", type=int, required=True)
    sp.add_argument("--reason", default="")
    sp.set_defaults(fn=cmd_abandon)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
