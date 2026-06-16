# Delta-2: Research Grounding (literature for the multi-objective token-level design)

**Date:** 2026-06-15. Purpose: ground every aspect of the "decompose + don't-compress + force-info"
idea in the actual literature, record what it means for our design, and the **corrections** it forced
to earlier claims. Companion to `DELTA2_DESIGN.md` (design) and `DELTA2_RESULTS.md` (measured results).

---

## 1. Identifiability — why naive regeneration was doomed
**Locatello et al., ICML 2019** ([paper](https://proceedings.mlr.press/v97/locatello19a.html)) —
*unsupervised disentanglement is provably impossible without inductive bias OR supervision.*
- **Theorem/intuition:** with a factorized (rotationally-symmetric) prior, you can rotate the latent
  into an infinite family of **entangled** latents that yield the **same data distribution**. An
  unsupervised objective only sees `p(x)`, so it **cannot prefer** the disentangled solution.
- **Empirical (12k models):** random seed/hyperparameters matter **more than method**; "successes"
  come from **hidden supervision/leakage**; disentanglement **does not** correlate with downstream use.
- **For us:** our collapses ARE this theorem. From regeneration alone, `delta=encode(B)`,
  `delta=novelty`, and infinitely many others minimize the loss **identically** → optimizer copies B.
  Naive self-supervised delta is unidentifiable. **Escape = inductive bias (subtraction architecture)
  + weak supervision (difflib-novel, paraphrase, A-suppression).** Warnings to obey: **multi-seed**,
  **clean held-out eval (no leakage)**, **judge by the task** not by "is delta clean."

## 2. "Don't compress" reopens the copy escape
**Posterior collapse** ([VAE](https://en.wikipedia.org/wiki/Variational_autoencoder)) — when a
powerful path can satisfy the objective without the latent, the latent is ignored; symmetrically a
high-capacity latent can absorb everything.
- **For us:** the bottleneck was what *blocked* `delta=encode(B)`. Removing compression (your
  "don't compress") removes that block → **PUSH becomes mandatory**, else uncompressed delta copies B.

## 3. PUSH — keep A/shared OUT of delta (CORRECTION: not adversarial)
**Elazar & Goldberg, EMNLP 2018** ([paper](https://aclanthology.org/D18-1002/)) — adversarial removal
**fails**: even after the adversary hits chance, a **post-hoc classifier recovers the attribute**.
Their verdict: *"do not rely on adversarial training to achieve invariant representation."*
**LEACE, 2023** ([paper](https://arxiv.org/abs/2306.03819)) — **perfect *linear* concept erasure**,
closed-form, **minimal change** (Frobenius-optimal), instant; **INLP** over-erases.
- **CORRECTION:** our old "leak-free complement / discriminator" was the adversarial kind → **drop it.**
- **PUSH design:** (1) **structural routing** — delta feeds ONLY the novel positions; shared B
  reconstructed from A (can't leak); (2) **LEACE** closed-form cleanup of *linear* A-leak; NOT
  adversarial. **Caveat:** LEACE removes only *linear* info → **measure residual (nonlinear) A-leak**,
  don't assume purity. Routing depends on the (noisy) difflib novel-mask.

## 4. PULL + meaning-perception — CLINE is the blueprint
**CLINE, ACL 2021** ([paper](https://aclanthology.org/2021.acl-long.181/)) — directly targets OUR
surface-robustness failure: models can't tell a meaning-preserving rewrite from a meaning-changing
edit, and *"adversarial training is useless or even harmful"* for detecting semantic change.
- **Method (unsupervised):** generate **x^syn** (synonyms, meaning-preserving) and **x^ant**
  (antonyms/random, meaning-changing) via WordNet. Losses: **MLM** (anchor) + **RTD** (per-token
  replaced-token-detection = *which* tokens changed) + **contrastive on [CLS]** (pull `ori~syn`,
  push `ori~ant`; antonym is the key hard negative). Uses **both shapes in one model**: token-level
  RTD + pooled contrast.
- **For us — big implications:**
  - **Blueprint for a single multi-objective architecture** carrying both token-level change and
    holistic meaning (encouraging for the single-architecture goal).
  - **Cleaner supervision:** generate syn/ant perturbations of **our own sentences** → **same-domain,
    self-supervised, controllable** → fixes the **domain confound** (MRPC-vs-wiki) and **difflib noise**.
  - Maps to our stack: RTD↔PULL, syn/ant contrastive↔invariance/gate, MLM↔anchor.
  - **Limit:** CLINE **detects & relates**, it does NOT **extract content as a vector** → still need
    the **subtraction complement** for "what changed." WordNet syn/ant is semantically noisy → combine
    with real IteraTeR edits.

## 5. Encoder / numeric ceiling (CORRECTION: not a permanent wall)
**Wallace et al., EMNLP 2019** ([paper](https://aclanthology.org/D19-1534/)) — numeracy IS present in
embeddings (GloVe 0.90 list-max on [0,99]), but **BERT sub-word is the WORST** (0.52 on [0,9999],
decoding RMSE **431**); **character-level is best** (char-CNN 0.88, RMSE **11.57**). Cause: sub-word
tokenization **mangles magnitude** ("75" and "750" split differently). **Extrapolation fails for all**
(beyond training range).
- **CORRECTION:** I said BERT has "no number sense / unfixable wall." Truth: it's a **tokenization
  limit**, fixable by a **character-level / number-aware encoder** (+ data augmentation for range).
  Explains our split: oracle 1.0 cross-pair (numeracy present) vs weak within-pair 30→35 (sub-word).
- **For us:** numeric is a **separable ENCODER workstream** (char/number-aware encoder + augmentation),
  **orthogonal to the delta objective** — no loss fixes it. Keep numeric **out of scope** for the
  multi-objective delta experiment; defer to an encoder swap.

## 6. Multi-objective balancing (CORRECTION: I under-weighted this risk)
**PCGrad — Gradient Surgery, NeurIPS 2020** ([paper](https://arxiv.org/abs/2001.06782)) — the
**"tragic triad"** (conflicting gradients `cos<0` + magnitude dominance + high curvature) stalls
multi-task training. Fix: when gradients conflict, **project one onto the normal plane of the other**
(`gᵢ ← gᵢ − (gᵢ·gⱼ/‖gⱼ‖²)gⱼ`); leave non-conflicting alone. Model-agnostic, no new hyperparameters.
**Uncertainty weighting — Kendall 2018** ([paper](https://arxiv.org/abs/1705.07115)) — learn per-task
`σ`: `L = Σ [ Lᵢ/(2σᵢ²) + log σᵢ ]`; hard tasks auto-down-weighted, `log σ` prevents blow-up.
- **For us:** **PULL vs PUSH genuinely conflict** (carry-novel vs drop-shared on the same vector) →
  textbook tragic triad. Use **uncertainty weighting (scale) + PCGrad (direction)** on the shared
  generator; **diagnose via pairwise gradient cosines**; keep the set **lean (~5 losses)**.
  **Caveat:** PULL/PUSH conflict is partly **fundamental** (a real trade-off frontier) — balancing
  reduces destructive interference, doesn't erase the trade-off.

---

## Corrections to earlier claims (summary)
1. **Adversarial PUSH is unreliable** (Elazar&Goldberg) → use **routing + LEACE** instead; drop the
   old discriminator/leak-free idea.
2. **Numeric is NOT a permanent wall** (Wallace) → sub-word tokenization limit, fixable by a
   **char/number-aware encoder** + augmentation; separable workstream.
3. **Multi-objective needs PCGrad/uncertainty weighting** — naive fixed weights will fight.
4. **The collapses were the identifiability theorem** (Locatello), inevitable for pure regeneration;
   the escape is **bias + weak supervision** (which our design supplies).

## Updated, research-grounded design (the converged spec)
- **Backbone:** subtraction generator (`generate_delta`), **token-level / uncompressed**.
- **PULL:** RTD-style per-token novelty detection + novel-token cloze/contrastive (carry the new content).
- **PUSH:** **structural routing** (delta → novel positions only; shared from A) **+ LEACE** linear
  cleanup. **NOT adversarial.** Measure residual A-leak.
- **Invariance / gate:** CLINE-style contrastive on **syn (preserve) / ant (change) perturbations of
  our own sentences** — same-domain, self-supervised.
- **Anti-collapse:** VICReg (variance + covariance) since uncompressed.
- **Anchor:** light reconstruction (MLM-ish / embedding) to keep delta content-rich.
- **Balancing:** uncertainty weighting + PCGrad; diagnose gradient cosines; ≤ ~5 losses; **multi-seed**.
- **Eval:** held-out group-split, **clean (no leakage)**, judged by the **task** (retrieval / end-to-end),
  stratified by change-type, multi-seed.
- **Scope:** additive / entity / relational (where it can win). **Numeric deferred** to an
  encoder-swap workstream (char/number-aware), per Wallace.
- **Keep** the existing two-stage `gate→extract` result as the current honest baseline to beat.

## Sources
Locatello 2019 · Elazar & Goldberg 2018 · LEACE 2023 · INLP 2020 · CLINE 2021 · Wallace 2019 ·
PCGrad (Yu 2020) · Uncertainty Weighting (Kendall 2018) · VICReg 2022 · Shortcut Learning (Geirhos 2020).
