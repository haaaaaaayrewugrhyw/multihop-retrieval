# Delta-2: "What B Adds Beyond A" — Results & Findings

**Date:** 2026-06-15
**Question:** Given two documents A and B, produce a faithful, usable representation of *what B
adds beyond A* (the novelty / delta), across change types.

**One-line result:** "What B adds" is not one object — it decomposes into **detect / extract /
relation**, and the working solution is a **two-stage `gate → extract` pipeline** (NLI-gate +
zero-training token-complement) scoring **0.818 end-to-end** on same-domain edits. A single learned
"delta vector" is a **dead end** for content; the encoder (BERT), not the objective, is the wall.

---

## 1. The problem, decomposed

| sub-question | meaning |
|---|---|
| **Q1 Detect** | *where* is the new content? (a mask over B) |
| **Q2 Extract** | *what* is the new content, as a usable representation? |
| **Q3 Relation** | *how* does B relate to A? (add / entail / neutral / contradict) |

Pipeline: Q1 locates → Q2 represents → Q3 classifies.

---

## 2. What we tried, and the verdict on each (honest)

| approach | outcome | why |
|---|---|---|
| **Reconstruct B from `[A, delta]`** (original system) | ❌ collapse | A≈B, so the decoder reaches the target using A alone → delta collapses to encode(B). Under-identified objective (Locatello 2019; posterior collapse). |
| **Pooled multi-task delta** (anchor + paraphrase/NLI aux; v1 cosine, v2 contrastive+VICReg, v3 direct novelty supervision) | ❌ for content, ✅ for relation | A single pooled vector is rank-1 (collapsed); content is token-distributed and gets averaged away. Even *direct supervision* (v3) hit content top1 0.088 < trivial `B−A` (0.21). |
| **Token-level fixed complement** (`H_B − match·attended_A`, your subtraction idea) | ✅ content (direction) | Per-token subtraction preserves the residual; pooling the *complement* keeps content. Zero training. |
| **`gate → extract` pipeline** | ✅ system | NLI/relation gate filters rewordings; token complement supplies content. |

---

## 3. Key results (all held-out, non-circular)

**Q1 — Detect:** `bert_maxsim` (1 − max cosine of each B-token to any A-token) → **AUC 0.948**, zero
training (matches a 70B LLM).

**Q2 — Extract, pooled (DEAD):** held-out battery, learned pooled delta vs baselines:

| rep | eff-rank | content top1 |
|---|---|---|
| pooled delta (v1/v2/v3) | 1.0–1.4 | 0.02–0.09 (≈chance) |
| `encB` (mean B) | 10.2 | 0.67 |
| `meandiff` (B−A) | 23.3 | 0.21 |

→ the more you isolate novelty into one pooled vector, the more content you destroy.

**Q2 — Extract, token-level (WORKS):** zero-training complement, held-out, stratified:

| construction | top1 | insertion | entity | relational | numeric |
|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **gate_sub (complement)** | **0.758** | 1.00 | 0.97 | 1.00 | 0.53 |
| enc_B | 0.593 | 1.00 | 0.80 | 0.89 | 0.38 |

→ near-perfect on additive/entity/relational; numeric is the weak spot (learnable: oracle 1.0).

**Q3 — Relation:** pooled NLI-aux delta → NLI probe **MDL 58** (compact, extractable), and
surface-invariance ratio **64×** (paraphrase → ~0).

**Surface-robustness of the complement (claim tested → FAILS alone):** edit-vs-paraphrase AUC by
magnitude = **0.43** (norm); per-type relational 0.37, numeric 0.30 — the op fires *more* on rewords
than on similar-surface factual flips. So the extractor needs a gate; it can't decide *whether*
meaning changed on its own. (Its retrieval still works because retrieval uses *direction*, not
magnitude.)

**The gate (tested, domain-confound ruled out):**
- cross-domain (MRPC paraphrases vs wiki edits): ‖delta‖ AUC **0.990** — inflated by domain.
- **same-domain (IteraTeR meaning vs non-meaning): AUC 0.806** (entity 0.86, numeric 0.86,
  insertion 0.73, relational 0.63). Real, modest; ~0.18 of the 0.99 was domain.

---

## 4. The system: `gate → extract`, end-to-end

Same-domain IteraTeR (meaning-changed = real edit; non-meaning = fluency/clarity = negative):

- **Stage 1 gate** (best of ‖delta‖ vs NLI): **NLI wins, AUC 0.892** (learned ‖delta‖ gate 0.826).
  recall 0.85, precision 0.82, FPR 0.19.
- **Stage 2 extract** (zero-training token complement): top1 **0.909**.
- **END-TO-END (gated AS changed AND content top-1 correct): 0.818**

| change-type | end-to-end | gate | extract |
|---|---|---|---|
| insertion | 0.92 | 0.92 | 1.00 |
| entity | 0.85 | 0.85 | 1.00 |
| relational | 0.79 | 0.86 | 0.86 |
| numeric | 0.81 | 0.90 | 0.84 |

Balanced 0.79–0.92 across types.

---

## 5. Honest limits & scope

1. **The deployable system needs no learned delta.** Best gate = off-the-shelf **NLI (frozen
   roberta)**; extractor = **zero-training** complement. The learned ‖delta‖ gate (0.826) is only a
   *lighter* near-match (BERT-base, no 1.3 GB roberta). The learned-delta study's value is the
   rigorous mapping of what works vs what's dead.
2. **Scope = same-domain IteraTeR.** Extraction 0.91 here is higher than the 0.758 IteraTeR+VitaminC
   mix because **VitaminC's subtle factual flips are not in this eval.** The genuinely hard case —
   VitaminC numeric/substitution (extract ~0.53, gate_sub ~0.18 full-pool) — remains the open frontier.
3. **The wall is the encoder, not the objective.** BERT collapses "similar surface / different
   meaning" (30↔35, antonyms), so distance-based novelty can't separate fine substitutions. No delta
   loss on frozen BERT fixes this; only a number/char-aware encoder or a generative-reasoning paradigm
   would.
4. **Gate is imperfect:** misses ~15% of real edits, false-flags ~19% of fluency edits.
5. Small per-type pools in places (relational/insertion n≈13–14); BERT-base throughout.

---

## 6. Root causes (why the naive approaches failed)

- **Identifiability, not capability.** Regeneration is under-identified (∞ deltas regenerate B); the
  optimizer copies B. Translation works because its target is unique — multi-head capability can't fix
  an under-specified objective.
- **Shape.** Content is token-distributed; pooling to one vector is information-theoretically wrong
  for content (right for the low-dimensional relation).
- **A≈B.** Any "reconstruct B from [A,delta]" lets A short-circuit delta.

**Fix principles (validated):** keep content **token-level**; target **only what A lacks**; gate the
**delta side** (A-dropout backfires); for "did meaning change," use a **surface-invariant** signal
(NLI / pooled ‖delta‖), not surface distance.

---

## 7. Methodological note

Every circular *training* metric looked like success and hid the truth: anchor cosine 0.95,
nli train-acc 1.0, separation 230 — all collapsed/memorized. Only the **held-out battery** (stratified
retrieval, MDL, surface-invariance, A-shuffle, domain-confound check) revealed what was real. Build
the held-out battery early.

---

## 8. Reproducibility

| result | script | runner |
|---|---|---|
| data foundation | `delta2_data.py` | `run_delta2_data_check.ipynb` |
| scaffold sanity | `delta2_model.py` | `run_delta2_fresh.ipynb` |
| pooled battery (v1/v2/v3) | `delta2_eval.py` | `run_delta2_battery.ipynb` |
| token complement battery | `delta2_token_battery.py` | `run_delta2_token.ipynb` |
| surface-robustness | `delta2_surface_robust.py` | `run_delta2_surface.ipynb` |
| gate (cross/same-domain) | `delta2_gate_test.py`, `delta2_gate_samedomain.py` | `run_delta2_gate*.ipynb` |
| **integrated system** | `delta2_pipeline.py` | `run_delta2_pipeline.ipynb` |

Design record: `DELTA2_DESIGN.md`. All runs frozen BERT-base, held-out group-split, HF token via
Kaggle secret, data cached to `/kaggle/working`.
