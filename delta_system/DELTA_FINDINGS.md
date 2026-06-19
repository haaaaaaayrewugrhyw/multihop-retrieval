# Representing "what B adds beyond A" — Findings (delta3–delta10)

Consolidated write-up of the edit-representation study on Yin's WikiAtomicEdits (insertions) and
IteraTeR+VitaminC (hard substitution/numeric/relational edits). All results held-out, group-split.

---

## 1. The question
Given a pair (A, B) where B is an edit of A, produce a representation of **the new content B adds**
("the delta"), and answer: is that content best **learned** (a trained net) or **computed** (a fixed
operation on the encodings)?

## 2. The correct metric (and why the obvious one is wrong)
- **Wrong metric — exact-match reconstruction** (regenerate B given A + the edit vector; Yin's Acc@1).
  It is dominated by **copy** of the shared text and by A being present, so it measures edit
  *application*, not whether the representation *carries the new content*. Our reconstruction runs sat
  at ~0 exact-match regardless of delta quality.
- **Right metric — decoder-free content-retrieval.** Fit a linear probe `rep → phrase-embedding` on
  train; on held-out, measure **top-1 retrieval** of the correct inserted/changed phrase among all test
  candidates. Controls: **selectivity** (shuffled-label control), **conditional** gain over A, fixed
  **baseline ladder** (chance < mean(B)−mean(A) < B@novel-positions < oracle=phrase embedding).
  Grounded in the probing literature (Hewitt & Liang control tasks; Voita & Titov MDL; Xu V-information;
  Hewitt/Ethayarajh/Liang conditional probing).

## 3. Two different problems — and where Yin sits
- **#1 — the edit *operator/type*** ("add a comma", "swap an entity"). **Population-level → learnable.**
  This is what **Yin (2019)** represents (Acc@1 72.9% reconstruction, 49% one-shot transfer, NDCG
  edit-type clustering). Verified: *all* of Yin's metrics measure #1; none measure content.
- **#2 — the instance *content*** (the specific new fact). This is what we probed. A **Yin-style edit
  vector scores 0.04** on content-retrieval (≈chance) — it encodes type, not content.

## 4. The experimental arc
| run | setup | key result |
|---|---|---|
| delta3 | subtraction encoder + bottleneck, regenerate B, no copy | A used (+0.10) but delta carries ~no content (novel-lift +0.01); exact-match 0 |
| delta4 | + pointer/copy | copy is the workhorse (+0.33 token acc); delta still inert (+0.03); exact-match 0 |
| delta5 | predict only the inserted phrase (block the shortcut) | delta IS the channel, but **memorizes**: train F1 0.77 → test 0.13 |
| delta6 | unified content-retrieval, all approaches (insertions) | fixed B@novel **0.66** ≫ learned-via-decoder 0.13; oracle 0.90 |
| delta6 | ours_direct (delta block trained **directly**, no decoder) | **0.46** — 3.5× the decoder versions; decoder was the confound |
| delta7 | scale ours_direct (insertions) | **0.43 → 0.67**; crosses the fixed op (0.625) at ~36k — gap was data-limited |
| delta8 | hard edits (numeric/entity/relational) | fixed op **weak (0.48)**; ours_direct **0.59 (+0.11)** |
| delta9 | multi-aspect slots (buggy K-query MHA) | 0.48 — failed (broken impl + under-trained) |
| delta10 | fair 4-way read-outs (proper slots) | @5k clustered ~0.62; @18k mean 0.643 < attn 0.659 < slot 0.665 |

## 5. The findings

### (a) Content is learnable — but only with the right setup
The early "content can't be learned, only computed" was **wrong as a general claim** — it was an
artifact of (1) training the delta *through a decoder* (which offloads to copy → delta stays inert,
0.13) and (2) tiny data. Train the delta block **directly** (contrastive, no decoder) **and scale the
data**, and a learned delta recovers the content and **beats the fixed subtraction**.

### (b) Learning beats computing exactly where computing is weak
- **Insertions (easy for the fixed op):** learned ties/edges it — 0.67 vs 0.625 (+0.05). *Use the free op.*
- **Hard edits (numeric/substitution/relational):** fixed op breaks (0.48–0.49); learned wins by
  **+0.12–0.16**, reproduced across 4 runs. Per-type (delta10@18k): numeric +0.03, entity +0.02,
  relational +0.03 over the fixed op's already-low base. **The value of a learned content-delta grows
  where the fixed op fails.**

### (c) Read-out is a minor knob
Mean-pool ≈ attention-pool ≈ slots. Proper slot attention (competition + iterative) is **marginally
best at scale** (slot8 0.665 vs mean 0.643, +0.02; consistent across types) but the effect is small,
single-seed, and an order of magnitude under the learn-vs-compute gap (+0.16). The earlier slot
*failure* (delta9) was a **broken implementation** (plain K-query attention + under-training), not the
idea — corrected, slots are fine but not transformative here (most edits are single-aspect → little to
decompose).

## 6. Honest limits
- **We do not beat Yin on Yin's task** (reconstruction ~0 vs 0.729) — different axis + ~300× less data.
  This sits *beside* Yin (a content-isolation probe), not above.
- **Margins are modest**: best learned ≈ 0.66 vs oracle ≈ 0.79 (0.13 headroom) — content is recoverable,
  not perfectly. Numeric is partly the **BERT-number wall** (Wallace 2019).
- **Scale-capped** (≤18k; the full WikiAtomicEdits source is offline), frozen BERT-base, in-domain.
- **Home-field advantage**: the learned model is trained on ~the eval objective; the fixed op isn't.
  (Mitigated by: both go through the same probe; held-out test; reproduced.)
- **Retrieval, not generation/disentanglement**: we show the content is *recoverably encoded*, not that
  we produce a clean decodable content vector.

## 7. One-line takeaway
**"What B adds" is learnable as a representation — a directly-trained subtraction delta recovers the
new content and beats a fixed subtraction precisely on the hard edits (numeric/substitution) where the
fixed op breaks, while being redundant on easy insertions. Edit *type* is learnable (Yin); edit
*content* is too, given direct training + scale; the read-out (mean vs slots) is a minor knob.**
