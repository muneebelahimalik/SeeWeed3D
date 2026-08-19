# Improving Stage A instance segmentation

What will actually move the numbers, what only looks like an upgrade, and the
order to try them in.

**Related:** [mixed-scene dataset strategy](mixed_dataset_strategy.md) ·
[runbook](RUNBOOK.md) · [supervised baseline](supervised_perception_baseline.md) ·
[edge model research](edge_model_research.md)

---

## Start here: the current limit is not the architecture

Stage A is trained on masks that are **unreviewed SAM 3 output** — see
`label_provenance` in [dataset assembly](dataset_assembly.md). Swapping
RF-DETR-Seg for a newer model changes how faithfully the network reproduces
those masks. It cannot make them better than the teacher.

So any architecture comparison run today is measuring **agreement with SAM**,
not segmentation quality. It will produce a number that looks like progress and
is not.

> Techniques that raise the **ceiling** come before models that improve the
> **fit**. The ordering below follows from that, not from expected AP.

---

## Raising the ceiling

### 1. Two-stage: trained model for identity, SAM for boundaries

The single biggest lever, and the only route to output *better* than the labels
you trained on.

The detector supplies **instance identity** — which pixels belong to which plant
— learned from your data, which is the part a general prior cannot supply. SAM
supplies **boundaries** from those prompts, at a quality your training masks do
not contain.

This also sidesteps the label-quality ceiling entirely: the boundary at
inference comes from SAM directly rather than from a network's imitation of SAM.

### 2. Point/crown supervision

Instance identity from clicks is cheap and exact, and the crown click *is* the
instance definition this project uses. The annotation contract already binds a
point to a mask by group id. See
[the dataset strategy](mixed_dataset_strategy.md#make-separation-cheaper-not-optional).

### 3. Copy-paste compositing

Perfect instance-separation ground truth for touching plants — the case that
fails most and is rarest in real data. Cut-outs from the manually annotated weed
frames are true masks, and their LEPs travel with them, so composites train
Stage A and Stage B together.

### 4. Depth-gated watershed barrier

Narrow and safe: a confidence-gated depth-gradient term discourages flooding
across a clear height transition between touching plants. It never deletes
tissue — it only decides where a cut falls inside vegetation already committed
by colour, which is what separates it from the
[height veto that was reverted](depth_assisted_masking.md).

---

## Architectural changes, in the order worth trying

### Resolution before architecture

`RESOLUTION` is the setting that has historically bought the most on this data —
the resolution era is its own CHANGELOG section for that reason. On 2208×1242
frames with cotyledons and thin onion leaves, capacity is rarely the binding
constraint and pixels are.

Current default is 1008. Try 1248 and 1344 against the benchmark before touching
the model. **Raise `GRAD_ACCUM` rather than trading resolution for batch size** —
that trade is what threw away small weeds before.

### Check the query budget against real instance density

DETR-family models allocate a fixed number of queries and match one-to-one via
Hungarian assignment. That converges slowly on crowded scenes and imposes a hard
cap: **a frame cannot produce more instances than there are queries.**

30–60 plants per frame is comfortably inside the usual 300. But this is worth
*measuring* rather than assuming, because if dense frames approach the limit no
amount of training fixes it, and the failure looks like "the model misses plants
in busy frames" rather than like a configuration limit.

### Alternative backends

| Option | Case for | Case against |
|---|---|---|
| **Mask2Former / MaskDINO** | masked attention; strong on dense, small instances | heavy; no real-time claim on Jetson |
| **YOLO-family seg** | fast, well-tooled | **AGPL** — kept opt-in here for licensing reasons |
| **SAM as the permanent mask head** | boundary quality above the training labels | two models at inference; latency to measure |

The third is the two-stage design promoted to architecture, and it needs no new
model — both components already exist in this repo.

---

## The order

1. **Build the benchmark** — without it none of the below is measurable, and
   "it looks better in the previews" is how a worse boundary pipeline shipped
   once already
2. **Resolution sweep** on the current architecture — cheapest real test
3. **Two-stage** detector → SAM at inference
4. **Then** revisit architecture

Steps 1–3 change what the model can achieve. Step 4 changes how well it fits
what it already has. Doing 4 first measures the teacher.

---

## Running the benchmark

```powershell
python -m seeweed3d.evaluation.bench_mixed `
    --truth E:/Dataset_Vidalia/mixed_gt `
    --pred  E:/Dataset_Vidalia/auto_labels_mixed/<session> `
    --out   E:/Dataset_Vidalia/bench/zeroshot.json
```

Both sides accept a `make_dataset` OUT_DIR or a prelabeler's COCO output, and
frames are paired by image file name. Re-run it per strategy and compare the
JSON files — that is the comparison the whole plan turns on.

It reports three groups that are never combined:

| Group | Reports |
|---|---|
| **crop** | onion-called-weed (a laser at the crop) apart from weed-called-onion (a missed weed), each also restricted to the onion/weed contact band |
| **identity** | merges and fragments, separately — they cancel in any instance-count difference |
| **cluster** | `weed_cluster` predicted over separable ground truth |

A mean IoU is deliberately not among them: it averages the catastrophic
direction into the harmless one, and a model can score well on it while
labelling crop as weed.

## Choosing what to annotate next

```powershell
python -m seeweed3d.annotation.rank_by_contact `
    --pred E:/Dataset_Vidalia/auto_labels_mixed/<session> `
    --top 40 --out E:/Dataset_Vidalia/annotate_next.txt
```

Ranks frames by how much onion/weed **contact** they contain — where the
dangerous decision is actually exercised. Runs on whatever prelabels exist, so
it is available before anything is trained.

`--per-session` (default 8) caps how many frames one drive may contribute.
Leave it on: ranking by a single signal and taking the top N otherwise returns
near-duplicate frames from one stretch of one recording.
