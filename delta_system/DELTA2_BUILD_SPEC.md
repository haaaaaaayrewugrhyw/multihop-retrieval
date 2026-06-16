# Delta-2 Build Spec: token-level multi-objective "what B adds beyond A"

**Date:** 2026-06-15. Build-ready spec derived from `DELTA2_RESEARCH.md` (grounding) +
`DELTA2_RESULTS.md` (what's measured). This is the user's idea — *decompose into supervised
sub-objectives, don't compress, force info, no cheating* — corrected by the literature and made
concrete.

---

## 0. Goal & success bar
**Goal:** ONE token-level model that captures "what B adds beyond A" for **additive / entity /
relational** changes, self-supervised + weakly-supervised, doing **both** gate ("did meaning change?")
and extract ("what changed?") from a **single representation**.

**Bar to beat:** the current two-stage baseline = **0.82 end-to-end** (off-the-shelf NLI gate +
zero-training complement). The learned single model must **match/beat** that to justify itself.

**Success = all of:** content retrieval ≥ complement (0.76+); **surface-robust** (edit-vs-paraphrase
AUC ≫ 0.5 — fixing the 0.43 failure); **A-dependent** (shuffled-A drop > 0); effective rank ≫ 1
(no collapse); single-model end-to-end ≈ 0.82; **stable across ≥3 seeds**.

---

## 1. Architecture (uncompressed, token-level)
```
frozen BERT-base (A, B)
        │
generate_delta  (two-level cross-attn SUBTRACTION)  → delta [b, T, d]   ← TOKEN-LEVEL, NOT pooled
        │
   ┌────┼───────────────┬───────────────┐
 RTD head            content (routed)   gate head
 per-token p(novel)  pool delta over    pooled delta →
                     novel positions    meaning-change score
        │
PUSH = routing (delta drives ONLY novel positions; shared reconstructed from A)
       + LEACE (closed-form linear A-erasure on pooled delta) + measure residual A-leak
```
- Backbone = your subtraction architecture (already proven 0.76 zero-training).
- **No pooling of delta** for content; pooling only for the gate (where it's correct).

## 2. Data & supervision (self-supervised + weak; same-domain)
- **Real edits (positives):** IteraTeR meaning-changed; difflib novel-token mask = the novel target.
- **CLINE-style generated pairs from OUR OWN sentences** (WordNet+spaCy):
  - `x_syn` (synonyms, ~40% content words) = meaning-PRESERVING,
  - `x_ant` (antonyms/random, ~20%) = meaning-CHANGING.
  - → same-domain, self-supervised, clean labels (fixes the domain-confound + difflib noise).
- **Meaning-preserving negatives (for gate/invariance):** `x_syn` + IteraTeR non-meaning (fluency).
- **Group-split, no leakage.**

## 3. Objectives (~5; the user's "sub-objectives", research-validated)
| # | sub-objective | loss | role |
|---|---|---|---|
| 1 | **Anchor (routed recon)** | reconstruct B from `[A, delta]`, shared←A / novel←delta (CE or embedding-cos) | keep content-rich; routing = PUSH |
| 2 | **PULL-RTD** | per-token BCE: delta→p(novel) vs difflib mask | localize novelty (token-level) |
| 3 | **PULL-content** | routed-pool(delta over novel) → match gold novelty (cos / InfoNCE pair-specific) | force the new content INTO delta |
| 4 | **Invariance/gate (CLINE)** | InfoNCE on pooled delta: `ori~syn` close, `ori~ant`/real-edit far | meaning-not-surface; the gate |
| 5 | **Anti-collapse (VICReg)** | variance(≥γ) + covariance decorrelation on token delta | uncompressed → prevent degeneracy |
| + | **PUSH-LEACE** | closed-form linear A-erasure on pooled delta + A-leak probe | keep A out; measure residual |

**Balancing:** **uncertainty weighting** (learnable σ per loss) + **PCGrad** on the shared generator
gradients. **Diagnose pairwise gradient cosines** (watch PULL vs PUSH conflict — the tragic triad).
Keep the set lean (do NOT add more losses).

## 4. Training
- Frozen BERT; train generator + heads only. **≥3 seeds** (Locatello warning).
- Cache encodings to disk. Kaggle: HF token in kernel, `python -u`, restart kernel between runs
  (zombie/RAM gotcha). fp16 NLI/teacher where used.

## 5. Evaluation (clean, held-out — judge by TASK)
- **Content:** retrieval of gold novelty (direction, cosine), stratified by change-type, vs
  complement 0.76 / encB / oracle.
- **Surface-robustness:** edit-vs-paraphrase **AUC by the gate** (must fix the 0.43 failure).
- **A-dependence:** shuffled-A content drop + **LEACE residual A-leak** probe (don't assume purity).
- **End-to-end:** from ONE model, gate→extract, vs the **0.82** two-stage baseline.
- **Non-collapse:** effective rank; **multi-seed variance** reported.

## 6. Scope & non-goals
- **In scope:** additive / entity / relational.
- **OUT of scope:** numeric / subtle factual flips — that's the **encoder** (BERT sub-word), fixable
  only by a **char/number-aware encoder + augmentation** (Wallace). Deferred, separate workstream.
- **Dropped (research):** adversarial / discriminator PUSH (Elazar & Goldberg — unreliable);
  A-dropout (backfires); single *pooled* delta for content (proven dead).

## 7. Decision gates
- **GO:** content ≥ complement, surface-robust AUC ≫ 0.5, A-drop > 0, e2e ≈ 0.82, rank ≫ 1, seed-stable.
- **STOP/FALLBACK:** if the gradient-cosine diagnostic shows PULL vs PUSH strongly anti-aligned AND
  content collapses → the trade-off is binding; **fall back to the two-stage pipeline** (already 0.82,
  documented) rather than forcing a worse single model.

## 8. Build order (incremental, each validated before the next)
1. **Data/supervision module** — CLINE-style syn/ant generation + edits + difflib + non-meaning
   negatives, same-domain, group-split. Validate stats (pass-rates, label sanity).
2. **Architecture** — reuse `generate_delta` (token-level) + routing + heads (RTD, gate, anchor).
   Sanity-train: does it run, losses move?
3. **Objectives incrementally** — anchor → +RTD → +content → +invariance → +VICReg, adding
   uncertainty-weighting + PCGrad; log gradient cosines at each step.
4. **PUSH** — add LEACE cleanup + A-leak measurement.
5. **Full eval** — battery + end-to-end vs the 0.82 baseline, **multi-seed**.

---
**One-line:** your idea, built right — uncompressed token-delta on the subtraction backbone, forced
by PULL (RTD + content) and guarded by PUSH (routing + LEACE, not adversarial), made meaning-aware by
CLINE syn/ant self-supervision, kept stable by VICReg + uncertainty/PCGrad — targeting
additive/entity/relational, with numeric deferred to an encoder swap, and a hard go/fallback gate
against the 0.82 baseline.
