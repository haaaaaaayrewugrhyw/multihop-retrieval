# Delta-2: Progress, Results & Ideas — Master Summary (2026-06-15)

Capstone summary of the whole effort. Companions: `DELTA2_DESIGN.md` (design), `DELTA2_RESULTS.md`
(results), `DELTA2_RESEARCH.md` (literature), `DELTA2_BUILD_SPEC.md` (spec). This file ties it all
together + records the final verdicts and the next direction.

---

## 1. The question
Extract "what B adds beyond A" (the **delta**) as a representation — detect it, extract its content,
and characterize the relation — and learn *how, by what means, with what limits*.

## 2. What we built (experiments)
data foundation (`delta2_data`, `delta2b_data`) · pooled multi-task scaffold + battery
(`delta2_model`, `delta2_eval`) · zero-training token complement battery (`delta2_token_battery`) ·
surface-robustness (`delta2_surface_robust`) · gate tests cross/same-domain (`delta2_gate_test`,
`delta2_gate_samedomain`) · integrated pipeline (`delta2_pipeline`) · token-level multi-objective
model + held-out eval (`delta2b_model`, `delta2b_eval`) · full end-to-end trained fair test
(`delta2c_e2e`). All held-out, group-split, baselined, multi-seed where it mattered.

## 3. Key results (held-out, non-circular)

| capability | result |
|---|---|
| detect/locate novelty | ✅ bert_maxsim AUC **0.948** |
| extract content — additive/clear | ✅ zero-training subtraction complement **0.84** held-out (insertion ~1.0) |
| extract content — numeric/fine | ❌ encoder wall (BERT sub-word) |
| relation / gate (did meaning change) | ✅ learnable, AUC **0.77–0.81** |
| **end-to-end system (gate → extract)** | ✅ **0.82** |
| single *learned* model for content (pooled / token / e2e) | ❌ collapses / overfits / ignored |

## 4. Final verdicts
- **A learned model does NOT extract the content delta.** Pooled → collapse (rank-1, chance);
  token-level → overfit (train cos 0.83 → held-out 0.18–0.20); full end-to-end → delta ignored
  (generation lift ≈ 0). Confirmed across **two independent metric families** (representation-match +
  generation) and multi-seed → robust, not a metric artifact.
- **The gate IS learnable** (0.77) — the one learned component that generalizes.
- **The content is best COMPUTED, not learned** — the parameter-free subtraction (your idea 8) wins.
- **Deliverable system = (learned/NLI gate) + zero-training complement = 0.82 end-to-end.**

## 5. The core insight (the real contribution)
**"What B adds" content is *instance-specific* (a per-pair comparison: the specific tokens that differ
in THIS pair, sharing nothing across pairs) → it must be *computed* (subtract A from B), not *learned*
(a trained net captures population-level features, so it memorizes train specifics but can't generalize
the per-instance content). The *gate* ("did meaning change / what kind") is *population-level* →
learnable.** Corroborated by: Locatello 2019 (regeneration unidentifiable → collapse inevitable),
posterior collapse, and the change-detection literature (learned difference functions generalize poorly
with limited samples; fixed metrics are robust). The encoder (BERT sub-word) is the numeric wall —
not the objective (Wallace 2019).

Your **subtraction architecture instinct was correct** — but as a **fixed operation**, not a trained
model. "Learn it end-to-end" fails *because* the correct operation is parameter-free (nothing to
improve, everything to overfit).

## 6. Honest evaluation of this work
- **As rigorous analysis + a working baseline + an explained negative result:** strong (B+/A−).
- **As a novel learned method:** weak — the win is a fixed op + off-the-shelf gate, not a new model.
- **Scale is small** (BERT-base, ~400–800 edits, 2 datasets, small per-type pools); numerics unsolved;
  in-domain only. Best framed as an **analysis / negative-results contribution**, not a method paper.

## 7. NEXT DIRECTION (user's idea, 2026-06-15) — self-supervised for scarce/expensive-data problems
**Premise (grounded):** self-supervised learning's gains are largest in the **low-data / expensive-label
regime** (medical, scientific, etc.). So the architecture/SSL idea may be valuable where **data
generation is hard/costly**.

**Plan (good practice):** first apply the self-supervised idea to **standard ML/DL benchmarks** (which
have established results) to get **comparison points** and see how it actually performs, *before* a
novel scarce-data application.

**Honest framing to carry in:**
1. **Define the method crisply first** — what exactly is the transferable "self-supervised architecture"
   (the subtraction/contrast residual? the multi-objective decompose-into-subtasks SSL training?).
   We must name it before benchmarking.
2. **Expectation:** on standard *population-level* tasks it will likely **match** standard SSL, not beat
   it — that's success (it confirms the architecture is sound and delta was the misfit regime).
3. **Where it could genuinely shine:** *instance-specific / comparison / low-data* problems where fixed
   metrics + light SSL beat data-hungry learned models — that's the regime our results actually point to.

## 8. Open frontiers
1. **Numerics** → char/number-aware encoder + augmentation (known recipe, separate workstream).
2. **Scale** (bigger encoder, more data/datasets, significance) to firm up the study.
3. **The new SSL-on-benchmarks direction** (define method → benchmark → then scarce-data application).

**One line:** we set out to *learn* a delta; we *proved* the content can't be learned-better-than-computed
(and *why* — instance-specific), shipped the compute-content + learn-gate system (0.82), bounded the hard
case to the encoder, and identified the regime where the SSL idea could actually pay off (scarce-data).
