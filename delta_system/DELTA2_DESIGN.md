# Delta-2: Multi-Task Objective — Design Decision Record

**Date:** 2026-06-15
**Status:** design locked, building. Supersedes the *objective* of the original delta system,
NOT the generator.

---

## 0. TL;DR of the big change (READ THIS FIRST)

We are **setting aside the token-level reconstruction decoder** (`D_recon`) as the training
objective and replacing it with a **multi-task objective** (light embedding-anchor + a swappable
semantic-novelty aux task) on a **pooled, fixed-size delta** produced by the *same* generator.

**Nothing is deleted.** `reconstruct()` / `D_recon` / `vib` / `n_slots` / `d0_aware` all stay in
[model.py](model.py) exactly as they are. The new objective lives in **separate `delta2_*`
modules** that import `DeltaSystem`, call `generate_delta()`, and never call `reconstruct()`.

**Revert path:** to go back to the original system, just train via `forward()` /
`reconstruct()` as before — no code was removed. This document + the git history are the record.

---

## 1. What the current architecture is (as of this record)

From [model.py](model.py) (re-read 2026-06-15):

### Generator — `generate_delta()` ([model.py:115](model.py#L115)) — **KEEP, reuse as-is**
- Frozen BERT (shared for A and B).
- Two-level multi-head cross-attention (self vs cross over H_A/H_B).
- `delta [b,T,768] = tanh(W_out · (norm(O_self2) − norm(O_cross2)))`
  → **token-level** novelty signal: large where B diverges from A, ~0 where B copies A.
- `delta_0 [b,768] = bottleneck(mean(H_B))`, or `bottleneck(mean(H_B) − mean(H_A))` if `d0_aware`.
  (Note: `d0_aware` delta_0 ≈ the **mean-difference baseline** — free baseline, reuse it.)
- Optional `vib` (variational info bottleneck → `self.last_kl`), `n_slots` (hard slot bottleneck).

### Reconstructor — `reconstruct()` ([model.py:166](model.py#L166)) — **SET ASIDE (kept in code)**
- Causal seq2seq decoder; memory `[delta_0 | H_A | enc(delta) or K slots]`; regenerates B
  token-by-token; LM head tied to word embeddings.
- Ablation flags `ablate_delta / drop_delta / drop_d0` for attribution.

### Why we are setting the decoder aside (the diagnosis)
The token decoder reconstructs the **full B sequence** from a **T-length token delta**. That lets
the decoder use delta as a near-complete **B template**, so reconstruction **never needs A** →
delta collapses toward `encode(B)` (the "complement collapse"). `n_slots` and `vib` were patches
against exactly this. A regeneration objective's optimum *is* "delta best for regenerating B" =
encode(B). To get **meaningful novelty** we must change the objective, not keep patching it.

---

## 2. The new design (Delta-2)

```
            frozen BERT (shared)
                 │
        generate_delta()  ← REUSED UNCHANGED
                 │  token delta [b,T,768]
        masked mean-pool (weighted by delta token-norm)
                 │  delta_vec [b,768]   ← fixed-size, probe-friendly
        ┌────────┴─────────────┐
   embedding anchor          swappable AUX  (one at a time)
   MLP(pool(H_A),delta_vec)    A: paraphrase-contrastive
     → predict encode(B)       B: NLI-teacher distillation
   (cosine/MSE; NO token       C: generate-novel-content (only if needed)
    decoder)
```

- **Anchor** keeps delta content-rich (so it isn't degenerate) WITHOUT the collapse-inducing
  token decoder. It targets the *pooled* `encode(B)`, not the token sequence.
- **Aux** forces the property reconstruction alone won't: *semantic* novelty.
  - **A (paraphrase-contrastive, CLINE-style):** `delta(A, paraphrase(A)) → ~0`;
    `delta(A, real-edit B) →` pair-specific & far from paraphrase deltas. Forces
    surface-invariance + real-change sensitivity.
  - **B (NLI-teacher):** distill a frozen NLI label/logits on (A,B) into delta via a small head.
    Forces delta to carry the entail/neutral/contradict *relationship*.
- We **train A and B separately**, run the full battery on each, and compare profiles.

**Why pooled (not token):** the rigorous battery (MDL / LEACE / DCI) needs a fixed-size vector;
the existing token delta is debugged, so we pool it rather than rebuild.

---

## 3. Data foundation (`delta2_data.py`) — STEP 1
- **Edits (A,B)** from IteraTeR (meaning-changed) + VitaminC (factual flips), with difflib gold
  spans + **change-type stratification**: insertion / entity / relational / numeric.
- **Paraphrase pairs (A, A_para)** from MRPC, **validated** by bidirectional NLI entailment +
  lexical-overlap filter (the researched standard) → negatives for task A / surface-invariance.
- **NLI-teacher labels** on (A,B) from `roberta-large-mnli` → supervision for task B + meaning probe.
- **Group split** by source/claim → no paraphrase/mirror leakage.
- `__main__` reports go/no-go: stratification counts, paraphrase pass-rate, NLI distribution.

## 4. Measurement battery (best methodology, from the start) — STEP 3
Run on each trained delta + baselines `encode(B)` / `mean(B)−mean(A)` / fixed-op delta + oracle + chance:
1. **Non-triviality:** variance / effective rank.
2. **Content (stratified):** non-circular cross-pair retrieval top-1/MRR per change-type.
3. **Isolation:** DCI (disentanglement/completeness/informativeness) or MIG, factor = change-type.
4. **A-dependence:** LEACE/amnesic causal erasure of the change-direction + shuffled-delta control.
5. **A-leakage / purity:** linear probe of A-content + control-task **selectivity** (should be low).
6. **Surface-invariance:** ‖delta(A,A_para)‖ vs ‖delta(A,B)‖ on validated paraphrases.
7. **Meaning / relationship:** linear probe → NLI 3-way with **MDL (online codelength)** +
   control-task selectivity (Hewitt-Liang / Voita-Titov), group-split.

## 5. Build order (each validated before the next)
1. **Data foundation** (`delta2_data.py`) — go/no-go stats. ← current
2. **Scaffold** (`delta2_model.py`): pool + anchor + swappable aux head; sanity-trains.
3. **Battery** (`delta2_eval.py`): validate metrics behave on baselines first.
4. **Task A** train + profile.
5. **Task B** train + profile → compare → pick → (add full recon / scale winner).

## 6. Decision gate (which aux wins)
High stratified content (esp. entity/relational) · high meaning-probe (MDL + selectivity) ·
surface-invariant · A-dependent (causal erasure hurts) · low A-leakage · isolated (DCI).
Honest ceiling: BERT discards precise numerics → **numeric flips remain the hard bound** for any aux.
