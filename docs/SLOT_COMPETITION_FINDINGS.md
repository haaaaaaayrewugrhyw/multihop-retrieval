# Slot Attention's Competition Is a Double-Edged Inductive Bias

**Status:** confirmed (hard-regime crossover replicated over 3 seeds, tight error bars)
**Code:** `experiments/slot_attention_compare/`
**Date:** 2026-06

---

## The question

Slot Attention's defining ingredient is **competition**: its attention softmax is
taken over the *slots* (not the inputs), so slots compete to "own" each input
feature. This is what's supposed to make slots bind to distinct objects.

**Does that competition actually help — and when?** We tested it as a controlled
ablation against ordinary (non-competitive) attention on the *same* objective.

## Setup (a clean ablation)

Unsupervised object discovery on synthetic multi-object images. The autoencoder
(encoder + spatial-broadcast decoder + slot count + dim + iterations) is held
**identical** across the two models; the **only** thing that changes is the
bottleneck's softmax axis:

| model   | bottleneck attention                                  |
|---------|-------------------------------------------------------|
| `slot`  | softmax over **slots** → competition (real Slot Attn) |
| `plain` | softmax over **inputs** → ordinary attention          |

Both have **identical parameter counts** (685,572), so any difference is the
*mechanism*, not capacity. Scored by:
- `FG-ARI` — foreground Adjusted Rand Index = object-discovery quality (1.0 =
  perfect object separation, ~0 = slots don't separate objects).
- `recon_mse` — reconstruction quality.

Two data regimes:
- **distinct** — each object a different color (color can shortcut grouping).
- **same-color** (`--hard`) — all objects in an image share one color, so they
  can only be separated by shape/position. This stresses the binding mechanism.

## The result — a crossover

```
regime          slot FG-ARI       plain FG-ARI      winner
distinct        0.880   [1 seed]  0.663   [1 seed]  slot  (+0.22)
same-color      0.122 ± 0.015     0.657 ± 0.005     plain (+0.54)   [3 seeds]
```

The ranking **flips** between regimes. And the deeper pattern:

- **`plain` is regime-robust:** ~0.66 FG-ARI on *both* regimes. It binds by
  position/shape and is essentially color-invariant.
- **`slot` is regime-fragile:** 0.88 on distinct objects → **collapses to ~0.12**
  when objects share color. Its competition relies on between-object feature
  distinctness; remove the color cue and it largely fails to separate objects.

The same-color collapse is replicated across 3 seeds with very tight error bars
(`plain` 0.657 ± 0.005, `slot` 0.122 ± 0.015) — a ~0.54 FG-ARI gap, far outside
noise.

## Interpretation

> **Slot Attention's competition is a double-edged inductive bias.** It is
> *superior* when objects are feature-distinct (it cleanly partitions them), but
> it *depends on* between-object feature distinctness and **collapses** when
> objects are feature-homogeneous. Ordinary attention is weaker at its peak but
> robust to feature homogeneity.

This is a controlled, in-vitro reproduction of a known open problem: Slot
Attention assumes *within-object features are homogeneous and between-object
features are distinct* — an assumption that breaks on real, visually diverse
objects (see the object-centric-learning "homogeneity" limitation). Here we make
that failure mode happen on demand (same-color objects) and quantify ordinary
attention's robustness to it.

## How the finding was de-risked (honest trail)

1. **First run (mis-tuned):** with a short warmup and no LR decay, `slot` scored
   only 0.33 on distinct objects and *lost* to `plain` (0.63). The diffuse,
   uncoverged masks flagged this as an artifact, not a real win.
2. **Fair schedule:** with proper warmup + cosine decay, `slot` jumped to 0.88 on
   distinct objects (textbook range) and clearly beat `plain` there — the initial
   "plain wins" was an under-training artifact. Correction logged.
3. **Same-color stress + replication:** `slot` collapses (0.12–0.45 depending on
   training length); `plain` holds at 0.66. Confirmed over 3 seeds.

## Honest caveats

- **`slot`'s same-color value is itself unstable** — ~0.45 at 100 epochs (1 seed),
  ~0.12 at 80 epochs (3 seeds), because its FG-ARI on same-color data *declines*
  with longer training (peaks ~0.25 around epoch 5, then drifts down). The exact
  number wobbles; the **crossover direction does not** (`slot` is always far below
  `plain` on same-color).
- **Distinct-regime is single-seed** (slot 0.88 > plain 0.66). The direction
  matches established Slot Attention results, but symmetric error bars would need
  a 3-seed run there too.
- **Decoder-competition confound:** the spatial-broadcast decoder applies a
  softmax-over-slots on its alpha masks, so *both* models have competition at the
  decoder. The cleaner statement is: "*bottleneck* competition (Slot Attention)
  helps on distinct objects and hurts on same-color ones, vs. relying on the
  decoder's competition alone."
- **One synthetic dataset, one kind of "hard"** (same color). Other stresses
  (occlusion, clutter, texture) are untested.

## Reproduce

```bash
cd experiments/slot_attention_compare
python train_slot.py --compare --epochs 80              # distinct regime
python train_slot.py --compare --hard --seeds 3 --epochs 80   # same-color, 3 seeds
```

`results/slot_masks.png` (per-slot masks) shows the qualitative story: on
same-color data `slot`'s masks smear across the scene while `plain`'s stay
localized to objects.
