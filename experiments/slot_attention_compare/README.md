# MLP+attention vs Slot Attention, on Slot Attention's own objective

## Question

Can plain attention (the MLP+attention idea) do what Slot Attention does —
unsupervised object discovery — or is Slot Attention's **competition** trick
(softmax over slots) actually load-bearing?

## Setup (a controlled ablation)

Identical encoder + spatial-broadcast decoder + slot count + dim + iterations.
The ONLY thing that changes is the bottleneck's softmax axis:

| model   | bottleneck                                            |
|---------|-------------------------------------------------------|
| `slot`  | softmax over **slots** -> competition (real Slot Attn)|
| `plain` | softmax over **inputs** -> ordinary attention (the idea)|

Both have the same parameter count, so any difference is the mechanism, not size.

Objective: unsupervised reconstruction (MSE) of synthetic multi-object images
(`data.py`). Scored by:
- `recon_mse` : reconstruction quality (lower better)
- `fg_ari`    : foreground Adjusted Rand Index = object discovery quality
               (1.0 = perfect object separation, ~0 = slots don't separate objects)

## Hypothesis

Without competition, nothing stops all K slots from attending to the whole
image identically -> they collapse into one blob -> low FG-ARI even if recon is
okay. Competition forces slots to partition the scene -> high FG-ARI. If `plain`
matches `slot`, that's a surprising and interesting result.

## Run (Kaggle, GPU + Internet)

```python
!rm -rf multihop-retrieval
!git clone -b cross-layer-attention https://github.com/haaaaaaayrewugrhyw/multihop-retrieval.git
%cd multihop-retrieval/experiments/slot_attention_compare
!python train_slot.py --compare --epochs 60
```

Outputs the result table and saves `results/slot_masks.png` (per-slot masks for
both models — look at it: `slot` should show one object per slot, `plain` should
show smeared/duplicated masks).

## Files
- `data.py`        — synthetic multi-object images + masks
- `slot_models.py` — encoder, decoder, SlotBottleneck (competition flag), SlotAE
- `train_slot.py`  — train, FG-ARI eval, `--compare` runner, mask viz
