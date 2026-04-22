# Multi-Hop Retrieval Architecture Design
**Project:** BLACK&GREEN_ASSIGNMENT — MuSiQue Multi-Hop Retrieval  
**Date:** 2026-04-22  
**Status:** Approved for implementation

---

## Problem Statement

Multi-hop retrieval over MuSiQue (2–4 hop compositional QA) requires two distinct capabilities:

1. **What does passage B contribute beyond passage A?** (query-agnostic, pre-computable)
2. **Is the hop A→B relevant for answering query Q?** (query-conditional, computed at retrieval time)

The original design used sentence-level pooled embeddings (e5-small-v2) for both models. **The core flaw:** pooling aggregates all token information into one vector before any interaction happens — the arithmetic `B_vec - A_vec` operates on compressed representations that have already lost token-level structure.

---

## Why Pooled Embeddings Fail Here

```
Pooled approach (WRONG):
  A_text → e5-small-v2 → A_vec [384]   ← all of A collapsed to one point
  B_text → e5-small-v2 → B_vec [384]   ← all of B collapsed to one point
  edge   = MLP(concat[A, B, B-A, A*B]) ← arithmetic on compressed vectors

Problem: B_vec - A_vec is NOT "what B says beyond A"
         It is "the difference between two lossy compressions of A and B"
         Token-level novelty is irretrievably lost before any comparison happens.
```

Same problem for Model 2:
```
query_vec [384] ← all of Q collapsed to one point
MLP(query_vec, edge_vec) ← can't recover which query aspects need which B content
```

---

## New Architecture

### Core Principle
**No pooling until necessary. Token matrices interact with token matrices.**

---

### Model 1 — Complement Encoder (Query-Agnostic)

**Goal:** Capture "what B contributes beyond A" without any knowledge of the query.

**Architecture: Joint BERT Encoding**
```
Input:  text_A, text_B
        ↓
Tokenize as: [CLS] A_tokens [SEP] B_tokens [SEP]
        ↓
BERT forward pass (all 12 layers attend across BOTH A and B)
        ↓
Take ONLY the B-side token representations → complement_tokens [n × 768]

Why joint encoding?
  - BERT self-attention computes A↔B interaction at every layer
  - B token outputs naturally represent "B's meaning in context of A"
  - This is the 12-layer deep version of the ESIM residual (B - aligned_A)
  - More powerful than manual cross-attention residual (single shallow layer)
```

**Why this is correct:**
- B token representations from BERT([A; B]) are conditioned on A through full self-attention
- Nothing about Q enters Model 1 — fully query-agnostic
- Output is a token MATRIX, not a pooled vector → structure preserved
- When A and B cover different topics: nothing in A aligns with B → B tokens are unchanged → complement = all of B (correct: B is fully novel)

**Optional projection (for storage efficiency):**
```
complement_tokens [n × 768]
        ↓
Linear(768 → 128) + L2-norm
        ↓
compressed_complement [n × 128]   ← ColBERT-style 128-dim token projection
```

**Training Signal — MuSiQue Chain Supervision:**
```
MuSiQue provides: Q → A → B → C (explicit compositional chains)

Positive pair:  (A, B_supporting)   ← B is the correct next hop
Negative pair:  (A, B_distractor)   ← B is a distractor from same question

Loss: Ranking loss
      complement(A, B_pos) should "contain" what C needs
      complement(A, B_neg) should NOT

Specifically: contrastive on the complement representations
  - complement(A, B_pos) closer to C token content than complement(A, B_neg)
  - This trains WITHOUT any query — uses chain structure as supervision
```

**Alternative simpler loss (also valid):**
```
Direct supervision: for each (A, B_pos, B_neg) triple from MuSiQue:
  cos_sim(compress(complement(A, B_pos)), C_tokens) > cos_sim(compress(complement(A, B_neg)), C_tokens)
  → Margin ranking loss
```

---

### Model 2 — ColBERT MaxSim Scorer (Query-Conditional)

**Goal:** Score "is the hop A→B relevant for query Q?" using token-level query representation.

**Architecture: ColBERT-style Late Interaction**
```
Input:  text_Q, complement_tokens from Model 1
        ↓
Encode Q: BERT(Q) → Q_tokens [k × 768] → Linear(768→128) → Q_compressed [k × 128]
        ↓
MaxSim interaction:
  For each query token q_i in Q_compressed:
      best_match_i = max_j  cos(q_i, complement_token_j)
        ↓
  score = Σ_i  best_match_i          ← scalar, how well Q is answered by complement
```

**Why this is correct:**
- Each query token independently finds its best match in B's novel content
- "Who founded" → finds founder tokens in complement
- "Born in" → finds birthplace tokens in complement
- Sum aggregates how many aspects of Q are covered by B's novel content
- Q only appears at scoring time → no pre-computation problem
- I(Q; complement_tokens) = 0 (correct: Model 1 is query-agnostic)
- I(Q; score) is high (correct: Model 2 is fully query-conditional)

**Training Signal — MuSiQue Hard Labels:**
```
For each (Q, A, B_pos, B_neg_1, B_neg_2, ...) from MuSiQue:
  score(Q, complement(A, B_pos)) > score(Q, complement(A, B_neg_i)) for all i

Loss: ListNet ranking loss (listwise cross-entropy)
      OR: margin ranking: score_pos - score_neg > margin
```

---

## Full Pipeline (Inference)

```
Step 1 — Seed Retrieval (BM25 + Dense, unchanged)
  Q → top-N initial candidate chunks {C_1, ..., C_N}

Step 2 — Beam Search with Model 1 + Model 2
  For each beam step (max 3 hops):
    Current beam: {A_1, ..., A_beam_width}
    
    For each beam node A_i:
      Get candidate neighbors B from graph (BM25/cosine top-K)
      
      For each candidate B:
        [Model 1]  complement_tokens = BERT_joint([A_i; B]) → take B tokens → project to 128-dim
        [Model 2]  score = ColBERT_MaxSim(Q_tokens, complement_tokens)
        
      Select top-K B by score → expand beam
    
Step 3 — Return top-10 chunks across all beam paths
```

---

## Logical Separation of Responsibilities

| | Model 1 | Model 2 |
|---|---|---|
| **Input** | text_A, text_B | text_Q, complement_tokens |
| **Knows about Q?** | NO | YES |
| **Output** | complement_tokens [n × 128] | scalar score [0, ∞) |
| **When computed** | On-the-fly per (A, B) pair during beam search | Immediately after Model 1 |
| **Training** | Chain-based contrastive (MuSiQue A→B→C) | Ranking on hard MuSiQue labels |
| **Captures** | What B says that A doesn't | Whether that new content answers Q |

---

## Comparison with Previous Design

| Aspect | Old Design | New Design |
|---|---|---|
| **A, B representation** | Pooled 384-dim vectors | Token matrices [n × 128] |
| **Model 1 interaction** | MLP on concatenated vectors | BERT joint encoding (12-layer attention) |
| **Model 1 loss** | IBDirectLoss (B_perp_A on pooled) | Chain-contrastive (complement predicts C) |
| **Query representation** | Pooled 384-dim vector | Token matrix [k × 128] |
| **Model 2 mechanism** | MLP(query_vec, edge_vec, embed_a) | ColBERT MaxSim(Q_tokens, complement_tokens) |
| **Query conditioning** | Pre-computed edge, Q arrives late | Q arrives fresh at MaxSim step |
| **I(Q; score)** | Near 0 (architectural ceiling) | High (token-level interaction) |

---

## Why Previous Model 2 Failed (Root Cause Summary)

The old Model 2 did:
```
edge_vec = f(embed_A, embed_B)    ← computed without Q
score    = MLP(query_vec, edge_vec, embed_a)
```

`edge_vec` was committed to a query-agnostic summary before Q was known. The MLP received Q separately, but the information Q needed to condition on was already compressed into `edge_vec`. Empirical confirmation: val_loss = 1.3071 (2K examples) vs 1.3055 (5K examples) — zero improvement from 2.5× more data, characteristic of architectural ceiling not data shortage.

---

## Related Literature

| Paper | Venue | Relevance |
|---|---|---|
| ESIM (Chen et al.) | ACL 2017 | Cross-attention residual for passage interaction |
| ColBERT (Khattab & Zaharia) | SIGIR 2020 | MaxSim late interaction for token-level scoring |
| MDR (Xiong et al.) | ICLR 2021 | Multi-hop dense retrieval, query extension |
| BeamRetriever | NAACL 2024 | Beam search for multi-hop, 78% R@10 on MuSiQue |
| Novelty Goes Deep | COLING 2018 | Neural complement encoding between passages |
| PLAID (Santhanam et al.) | NeurIPS 2022 | Compressed token matrix storage |
| Poly-encoder (Humeau et al.) | ICLR 2020 | Multi-vector document representation |

---

## Hardware Notes

- **Training:** Google Colab (T4 16GB or V100 32GB) — hardware is NOT a constraint
- **BERT-base:** 110M params, ~440MB float32; fine-tune both models end-to-end
- **Token matrix storage at inference:** 23K chunks × 100 tokens × 128-dim × 4 bytes ≈ 1.18GB — fits in Colab RAM
- **ColBERT MaxSim cost:** O(|Q| × |complement|) matmul — negligible on GPU
- **Model 1 inference cost:** One BERT forward pass per (A, B) candidate pair during beam search (~20ms on T4)

---

## Dataset

- **Training:** MuSiQue train split (`musique_ans_v1.0_train.jsonl`) — 17.5K questions, explicit compositional chains, no shortcut-solvable questions
- **Evaluation:** MuSiQue dev split (`musique_ans_v1.0_dev.jsonl`) — 2,417 questions
- **Metric:** Recall@10 (fraction of questions where all supporting chunks appear in top-10)
- **Baseline to beat:** MDR = 74.5% R@10 on MuSiQue dev (300-example eval); BeamRetriever = 78%

---

## Implementation Plan

1. **model1_train.py** — Rewrite with:
   - `ComplementEncoder`: BERT-base joint encoding of [A; B], extract B tokens, project to 128-dim
   - `ChainContrastiveLoss`: rank complement(A, B_pos) above complement(A, B_neg) using C content
   - Data: MuSiQue train chains, build (A, B_pos, B_neg, C) quadruples

2. **model2_train.py** — Rewrite with:
   - `ColBERTScorer`: BERT encode Q → Q_tokens [k × 128]; MaxSim against complement_tokens
   - `RankingLoss`: ListNet or margin ranking on MuSiQue hard labels
   - Data: MuSiQue (Q, A, B_pos, B_neg_1..3) quintuples

3. **run_full_system.py** — Update:
   - Replace `model2(q_t, e_t, a_t)` call with `colbert_score(Q_tokens, complement_tokens)`
   - Replace pre-computed edge_vectors with on-the-fly complement computation
