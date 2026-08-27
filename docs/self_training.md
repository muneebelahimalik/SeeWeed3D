# The self-training loop

> Score the model's own predictions, keep the ones safe to train on, and send
> the ones it got wrong to a human.
>
> `python -m seeweed3d.training.datasets.weeds_selftrain`

---

## The failure this is designed against

Keeping the frames the model scored **highest** and training on them is
self-confirmation. Where the model is already right the loss is near zero, so
there is no gradient and nothing is learned — all that changes is that the model
grows more confident about what it already knew, while its blind spots stay
exactly as blind. Its confident errors, meanwhile, become training truth and
compound.

Confidence is the wrong instrument *specifically because it is the model's own
opinion*. A model is most confident where the data looks like its training set,
which is precisely where a new frame teaches least — and softmax confidence is
badly calibrated off-distribution, so a confident wrong mask on unfamiliar
ground looks exactly like a confident right one.

## What works instead: an independent witness

The **ExG vegetation prior** is a per-pixel colour computation. It knows nothing
about the model, was not trained, and cannot be talked into agreeing.

| term | weight | measures |
|---|---|---|
| `veg_precision` | 0.35 | of the masked pixels, how many are green |
| `veg_recall` | 0.35 | of the green pixels, how many were masked |
| `stability` | 0.20 | `n / (n + duplicates)` — how much the model disagreed with itself |
| `confidence` | 0.10 | mean detection score |

`veg_recall` is why this is not a confidence filter with extra steps. A frame
where the model found three obvious weeds and silently skipped nine is a frame
where **every detection is correct** and the pseudo-label is catastrophic: the
nine become BACKGROUND in the training target, and the model is taught that
plants like those are soil.

Confidence is present and deliberately the smallest term.

## The three buckets

```python
if empty:                          → skip     # no vegetation: bare soil
if gates_failed or score < 0.45:   → review
if score >= 0.70:                  → accept
else:                              → skip
```

Hard gates, checked before the score:

```python
GATES = {"veg_precision": 0.55, "veg_recall": 0.80}
```

A gate failure sends a frame to **review, never to skip**. A frame the model got
badly wrong is the most valuable thing a human can annotate; discarding it would
throw away the loop's best signal.

---

## Look at every frame, export a separated subset

Sampling and separation are different decisions, and one stride does the worse
half of each.

- `INFER_STRIDE = 1` — a GPU pass over a drive is minutes, and a frame the model
  never saw cannot be chosen even if it was the best one in its stretch.
- `MIN_FRAME_GAP = 60` — separation at **selection** time, keeping the
  best-scoring frame in each window.

Unlike a split, where frames sitting too close merely flatter a score,
pseudo-labels get **weighted**: an error on one plant enters the training set
once per near-copy. It applies to `review/` too, where each near-copy is a
person correcting the same plant a second time.

> **The yield is arithmetic, not quality.** A 391-frame drive at gap 60 has
> 6–7 windows, so 292 frames passing the gates still becomes ~6 files. To get
> more, lower `MIN_FRAME_GAP` or add drives — 391 frames is about thirteen
> seconds of driving, and no sampling scheme extracts more independent ground
> than a drive contains.

Windows rather than greedy-by-score: taking the highest-scoring frame first lets
one early pick push the next past the following window, turning four frames into
three. The observed score spread was p10 0.83 to p90 0.91, so that trade buys a
hundredth of a point and costs a whole frame of ground.

---

## What lands in each folder

```
selftrain_<stamp>/
  <session>/accept/      pseudo-labels — cheap, safe, teach little
  <session>/review/      the frames it got WRONG — these move the model
  <session>/spot_check/  10 overlays sampled from accept/
  selftrain_report.json
  NEXT_STEPS.txt
```

Accepted frames pass three more filters, in order: **separation** → **appearance
diversity** (if over budget) → **class balance** (no class over 60%). Without the
last one the loop concentrates: the model predicts its dominant class most
confidently, those frames score highest, they get added, the class grows more
dominant. Three rounds and the rare classes are noise.

Budget: `2 × hand-corrected frames`. Computed against *hand* frames, not dataset
size, so a mostly-pseudo dataset cannot cite its own bulk to justify more. It is
enforced per session, so pooling several can pass it — the run totals them and
says so.

**Counts differ, on purpose.** `summary` is what the classifier decided;
`written` is what is in the folders. On a real drive the gap was 292 → 4.

---

## Reading the report before you open CVAT

Two minutes, and it decides whether the batch is worth an afternoon.

1. **`spot_check/`** — sweep each frame for green with no outline on it. 9–10
   clean means the gate is catching what it needs to. Two or more with an obvious
   missed plant means `veg_recall` at 0.80 is too loose; raise it and re-run
   (seconds, not a GPU pass — predictions are reused).
2. **A flat sweep** is called out:
   ```
   [!] THE THRESHOLD IS NOT SELECTING. Every cut from 0.50 to 0.80 accepts the
       same 68 frame(s) ...
   ```
   `ACCEPT` isn't deciding anything — the gates are — and turning that knob will
   not help.
3. **A gate that never fired** says so. A floor nothing has ever failed is not
   protecting anything, and "0 failures" is otherwise indistinguishable from
   "not being applied".

---

## The round trip

`NEXT_STEPS.txt` carries this with your real paths in it. **One CVAT task per
session, `review/` first.**

1. New task → upload `<session>/review/cvat_ready/`
2. **Before importing**: paste `weed_cvat_labels.json` into the Raw label editor
3. Import `instances_default.json` as **COCO 1.0**
4. Correct
5. Export as **Datumaro 1.0** → `<pool>/sessions/<session>/annotations/default.json`

Step 2 before step 3 is not stylistic: CVAT matches **by name**, and importing
into a task with no matching label silently creates a duplicate instead of
filling the one you meant.

Step 5 is why batches are per session rather than pooled — the build takes one
session folder per source, and the split logic computes seam separation from
session identity.

### Correct in this order

1. **What's missing.** A pre-labelled frame biases you toward accepting what is
   drawn rather than noticing what is absent.
2. **Species.** The model proposes only what it was trained on.
3. **Clusters.** Split one if separating is merely tedious, not impossible.
4. **Boundaries last** — and the crown matters more than the leaf margin.

### Then

Add the corrected sessions to `WEED_SESSIONS`, set `LABEL_PROVENANCE = "mixed"`,
rebuild, bump `ROUND`, retrain. Every other runner reads `ROUND` from
`weeds_train.py`.

---

## The guardrails, and why none is optional

- **Pseudo-labels are never called hand-corrected.** A dataset that cannot say
  which labels came from a model cannot be audited, and this project has already
  been bitten by unreviewed labels being read as verified.
- **Holdout sessions are never pseudo-labelled.** That would put the model's own
  output into its own test set.
- **Sessions already in training are skipped**, read from the build's manifest —
  they score at ceiling, read as a great batch, and teach nothing.
- **Regenerated each round** from the newest checkpoint, never accumulated. A
  stale pseudo-label is a mistake the current model would no longer make, kept
  alive as ground truth.
- **Reused predictions are checked** against the checkpoint's date *and* against
  the frame count the current settings select — changing `INFER_STRIDE` and
  silently rescoring the old frames is a real thing that happened.

> **Merging `accept/` alone is a model talking to itself.** It will look like it
> is working right up until it stops.

---

**See also:** [system readiness](system_readiness.md) ·
[weed active learning](weed_active_learning.md) ·
[dataset assembly](dataset_assembly.md) · [runbook](RUNBOOK.md)
