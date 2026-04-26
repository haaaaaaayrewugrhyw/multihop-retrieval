"""
Full system evaluation: graph traversal with trained Model 1 + Model 2.

Architecture:
  Model 1 — ComplementEncoder: BERT([A; B]) → B-side token matrix [n × 128]
  Model 2 — QueryEncoder: BERT(Q) → mean-pooled query vector [128]
  Scoring: mean_pool(Q) · mean_pool(complement(A, B))

Compares:
  - MDR dense (iterative query extension baseline)
  - Graph traversal with direct cosine similarity only
  - FULL: Graph + M1 complement + M2 dot-product scoring

Usage:
    python run_full_system.py --max_examples 300
    python run_full_system.py                      # full 2417 dev examples
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_musique
from evaluate import evaluate_retriever, compare_systems
from baselines import DenseRetriever, BM25Retriever, reciprocal_rank_fusion
from graph_builder import build_graph
from mdr_baseline import MDRBaseline
from model1_train import ComplementEncoder, mean_pool
from model2_train import QueryEncoder

CACHE_DIR   = Path(__file__).parent / "cache"
MODEL_DIR   = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA        = 0.5      # weight for M2 score vs direct cosine similarity
BEAM_WIDTH   = 3
MAX_HOPS     = 3
N_SEEDS      = 5
STOP_THRESH  = 0.05
QR_MIX       = 0.1      # MDR-style query reformulation weight
MAX_LEN_Q    = 64
MAX_LEN_AB   = 256


# ── Shared scoring helper ─────────────────────────────────────────────────────

class LiveScorer:
    """
    Wraps complement encoder + query encoder for on-the-fly scoring during beam search.
    Encodes query once per question; encodes (A, B) complement per candidate pair.
    Score = mean_pool(Q) · mean_pool(complement(A, B))  — dot product in shared 128-dim space.
    """

    def __init__(
        self,
        comp_enc:     ComplementEncoder,
        query_enc:    QueryEncoder,
        ab_tokenizer: BertTokenizerFast,
        q_tokenizer:  BertTokenizerFast,
        device:       str = DEVICE,
    ):
        self.comp_enc     = comp_enc
        self.query_enc    = query_enc
        self.ab_tokenizer = ab_tokenizer
        self.q_tokenizer  = q_tokenizer
        self.device       = device
        self.sep_id       = ab_tokenizer.sep_token_id

        comp_enc.eval()
        query_enc.eval()

    @torch.no_grad()
    def encode_query(self, question: str) -> torch.Tensor:
        """Encode query → q_vec [1, 128] L2-normalised."""
        enc = self.q_tokenizer(
            text=question, max_length=MAX_LEN_Q,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        q_vec = self.query_enc(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )
        return q_vec   # [1, 128]

    @torch.no_grad()
    def complement_score(
        self,
        q_vec:  torch.Tensor,   # [1, 128]
        text_a: str,
        text_b: str,
    ) -> float:
        """
        Compute M2 score for hop A→B given query Q.
        Score = dot( q_vec, mean_pool(complement(A, B)) ).
        Returns scalar float.
        """
        enc = self.ab_tokenizer(
            text=text_a, text_pair=text_b,
            max_length=MAX_LEN_AB, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        b_mask = (
            (enc["token_type_ids"] == 1) & (enc["input_ids"] != self.sep_id)
        ).to(self.device)

        comp_tokens, pad_mask = self.comp_enc(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            enc["token_type_ids"].to(self.device),
            b_mask,
        )
        c_vec = mean_pool(comp_tokens, pad_mask)   # [1, 128]
        return float((q_vec * c_vec).sum(-1).item())


# ── Full Graph Traversal ──────────────────────────────────────────────────────

class FullGraphTraversal:
    """
    Beam search over passage graph, scoring each hop with:
        alpha * M2_score(Q, complement(A, B))
        + (1 - alpha) * cos_sim(query_emb, B_emb)

    Model 1 (ComplementEncoder) and Model 2 (QueryEncoder) are run
    on-the-fly per candidate pair — no pre-computed edge vectors needed.
    """

    def __init__(
        self,
        scorer:      LiveScorer,
        graph:       Dict,
        corpus:      List[Dict],
        embeddings:  np.ndarray,
        dense:       DenseRetriever,
        bm25:        BM25Retriever,
        alpha:       float = ALPHA,
        beam_width:  int   = BEAM_WIDTH,
        max_hops:    int   = MAX_HOPS,
        n_seeds:     int   = N_SEEDS,
        stop_thresh: float = STOP_THRESH,
    ):
        self.scorer      = scorer
        self.graph       = graph
        self.embeddings  = embeddings
        self.dense       = dense
        self.bm25        = bm25
        self.alpha       = alpha
        self.beam_width  = beam_width
        self.max_hops    = max_hops
        self.n_seeds     = n_seeds
        self.stop_thresh = stop_thresh

        self.id_to_idx  = {c["chunk_id"]: i for i, c in enumerate(corpus)}
        self.id_to_text = {c["chunk_id"]: c["text"] for c in corpus}

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        # Seed retrieval
        dense_list = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_list  = self.bm25.retrieve(question,  top_k=self.n_seeds * 3)
        seeds      = reciprocal_rank_fusion([dense_list, bm25_list])[:self.n_seeds]

        # Encode query once (used for all M2 scoring calls)
        q_vec = self.scorer.encode_query(question)

        # Pooled query embedding for direct cosine similarity fallback
        query_emb = self.dense.embed_query(question)

        retrieved:     List[str] = list(seeds)
        retrieved_set: Set[str]  = set(seeds)
        beam:          List[str] = list(seeds)

        for hop in range(self.max_hops):
            # MDR-style query reformulation: mix in beam document embeddings
            beam_idxs = [self.id_to_idx[c] for c in beam if c in self.id_to_idx]
            if beam_idxs:
                beam_emb  = self.embeddings[beam_idxs].mean(axis=0)
                query_emb = query_emb + QR_MIX * beam_emb
                query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

            candidates: Dict[str, float] = {}

            for chunk_id in beam:
                text_a   = self.id_to_text.get(chunk_id)
                chunk_idx = self.id_to_idx.get(chunk_id)
                if text_a is None or chunk_idx is None:
                    continue

                for (nbr_id, _, _) in self.graph.get(chunk_id, []):
                    if nbr_id in retrieved_set:
                        continue
                    nbr_idx = self.id_to_idx.get(nbr_id)
                    text_b  = self.id_to_text.get(nbr_id)
                    if nbr_idx is None or text_b is None:
                        continue

                    # M2 score: query-conditional hop relevance
                    colbert_score = self.scorer.complement_score(
                        q_vec, text_a, text_b
                    )
                    # Direct similarity: query vs neighbor embedding
                    direct_sim = float(np.dot(query_emb, self.embeddings[nbr_idx]))

                    final = self.alpha * colbert_score + (1 - self.alpha) * direct_sim
                    if nbr_id not in candidates or final > candidates[nbr_id]:
                        candidates[nbr_id] = final

            if not candidates or max(candidates.values()) < self.stop_thresh:
                break

            top_nbrs = sorted(candidates, key=candidates.__getitem__, reverse=True)[:self.beam_width]
            beam = top_nbrs
            retrieved_set.update(top_nbrs)
            retrieved.extend(top_nbrs)

        return retrieved[:top_k]


# ── Graph traversal with direct cosine only (ablation — no model scoring) ─────

class GraphDirectCosine:
    """
    Graph traversal using only cos_sim(query_emb, B_emb).
    Ablation baseline: shows benefit of Model 1 + Model 2 scoring.
    """

    def __init__(
        self,
        graph:      Dict,
        corpus:     List[Dict],
        embeddings: np.ndarray,
        dense:      DenseRetriever,
        bm25:       BM25Retriever,
        beam_width: int   = BEAM_WIDTH,
        max_hops:   int   = MAX_HOPS,
        n_seeds:    int   = N_SEEDS,
        stop_thresh: float = STOP_THRESH,
    ):
        self.graph       = graph
        self.embeddings  = embeddings
        self.dense       = dense
        self.bm25        = bm25
        self.beam_width  = beam_width
        self.max_hops    = max_hops
        self.n_seeds     = n_seeds
        self.stop_thresh = stop_thresh
        self.id_to_idx   = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        dense_list = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_list  = self.bm25.retrieve(question,  top_k=self.n_seeds * 3)
        seeds      = reciprocal_rank_fusion([dense_list, bm25_list])[:self.n_seeds]
        query_emb  = self.dense.embed_query(question)

        retrieved:     List[str] = list(seeds)
        retrieved_set: Set[str]  = set(seeds)
        beam:          List[str] = list(seeds)

        for hop in range(self.max_hops):
            beam_idxs = [self.id_to_idx[c] for c in beam if c in self.id_to_idx]
            if beam_idxs:
                beam_emb  = self.embeddings[beam_idxs].mean(axis=0)
                query_emb = query_emb + QR_MIX * beam_emb
                query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

            candidates: Dict[str, float] = {}
            for chunk_id in beam:
                for (nbr_id, _, _) in self.graph.get(chunk_id, []):
                    if nbr_id in retrieved_set:
                        continue
                    nbr_idx = self.id_to_idx.get(nbr_id)
                    if nbr_idx is None:
                        continue
                    score = float(np.dot(query_emb, self.embeddings[nbr_idx]))
                    if nbr_id not in candidates or score > candidates[nbr_id]:
                        candidates[nbr_id] = score

            if not candidates or max(candidates.values()) < self.stop_thresh:
                break
            top_nbrs = sorted(candidates, key=candidates.__getitem__, reverse=True)[:self.beam_width]
            beam = top_nbrs
            retrieved_set.update(top_nbrs)
            retrieved.extend(top_nbrs)

        return retrieved[:top_k]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_retriever(name: str, retriever, queries: List[Dict], top_k: int = 10) -> Dict:
    print(f"\n[runner] Evaluating: {name} ...")
    t0      = time.time()
    results = []
    for q in tqdm(queries, desc=name, leave=False):
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        results.append({
            "query_id":  q["query_id"],
            "retrieved": retrieved,
            "relevant":  q["relevant_chunk_ids"],
        })
    elapsed = time.time() - t0
    metrics = evaluate_retriever(results, ks=[2, 5, 10])
    metrics["latency_ms"] = round(elapsed / len(queries) * 1000, 1)
    print(f"  Done {elapsed:.1f}s | R@10={metrics.get('recall@10', 0):.4f}"
          f" | {metrics['latency_ms']:.1f}ms/query")
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap MuSiQue dev examples (None = all 2417)")
    parser.add_argument("--top_k",        type=int, default=10)
    parser.add_argument("--alpha",        type=float, default=ALPHA,
                        help="Weight for M2 score vs direct cosine")
    args = parser.parse_args()

    # ── Data ────────────────────────────────────────────────────────────────
    corpus, queries = load_musique(
        split="validation", max_examples=args.max_examples, cache=True
    )
    cache_name = f"musique_val_{args.max_examples}"
    print(f"[runner] {len(corpus):,} chunks | {len(queries):,} queries | device: {DEVICE}")

    # ── Indexes ──────────────────────────────────────────────────────────────
    bm25 = BM25Retriever()
    bm25.build(corpus, cache_name=f"bm25_{cache_name}")

    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{cache_name}")

    n, dim = dense.index.ntotal, dense.index.d
    embeddings = np.zeros((n, dim), dtype=np.float32)
    dense.index.reconstruct_n(0, n, embeddings)

    # ── Graph ────────────────────────────────────────────────────────────────
    graph = build_graph(corpus, embeddings=embeddings, cache_name=cache_name)

    # ── Load trained models ──────────────────────────────────────────────────
    m1_ckpt = MODEL_DIR / "model1_complement.pt"
    m2_ckpt = MODEL_DIR / "model2_scorer.pt"
    has_models = m1_ckpt.exists() and m2_ckpt.exists()

    if has_models:
        comp_enc  = ComplementEncoder().to(DEVICE)
        comp_enc.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
        comp_enc.eval()

        query_enc = QueryEncoder().to(DEVICE)
        query_enc.load_state_dict(torch.load(m2_ckpt, map_location=DEVICE))
        query_enc.eval()

        ab_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        q_tokenizer  = BertTokenizerFast.from_pretrained("bert-base-uncased")

        scorer = LiveScorer(comp_enc, query_enc, ab_tokenizer, q_tokenizer, DEVICE)
        print("[runner] Loaded Model 1 (ComplementEncoder) + Model 2 (QueryScorer)")
    else:
        missing = []
        if not m1_ckpt.exists(): missing.append(str(m1_ckpt))
        if not m2_ckpt.exists(): missing.append(str(m2_ckpt))
        print(f"[runner] Missing checkpoints: {missing}")
        print("[runner] Will evaluate baselines only (no full system)")
        scorer = None

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_systems: Dict[str, Dict] = {}

    # MDR baseline
    mdr             = MDRBaseline(max_hops=3, seeds_per_hop=5)
    mdr.dense       = dense
    mdr.id_to_chunk = {c["chunk_id"]: c for c in corpus}
    all_systems["MDR (dense, iterative)"] = run_retriever(
        "MDR-dense", mdr, queries, args.top_k
    )

    # Graph traversal, direct cosine only (ablation)
    g_cos = GraphDirectCosine(graph, corpus, embeddings, dense, bm25)
    all_systems["Graph + direct cosine"] = run_retriever(
        "Graph+cos", g_cos, queries, args.top_k
    )

    # Full system: complement encoder + ColBERT MaxSim
    if scorer is not None:
        full = FullGraphTraversal(
            scorer, graph, corpus, embeddings, dense, bm25, alpha=args.alpha
        )
        all_systems["FULL: Graph + M1 + M2"] = run_retriever(
            "Full-M2", full, queries, args.top_k
        )

    # ── Results ──────────────────────────────────────────────────────────────
    compare_systems(all_systems)

    if scorer is not None and "FULL: Graph + M1 + M2" in all_systems:
        mdr_r10  = all_systems["MDR (dense, iterative)"]["recall@10"]
        full_r10 = all_systems["FULL: Graph + M1 + M2"]["recall@10"]
        gap      = full_r10 - mdr_r10
        print(f"\n[DECISION GATE] Full system R@10: {full_r10:.4f}")
        print(f"[DECISION GATE] MDR baseline R@10: {mdr_r10:.4f}")
        print(f"[DECISION GATE] Gap: {gap:+.4f} ({gap*100:+.2f}%)")
        if gap > 0.03:
            print("[DECISION GATE] FULL SYSTEM WINS by >3% ✓")
        elif gap > 0.0:
            print("[DECISION GATE] FULL SYSTEM WINS but gap ≤3% — marginal")
        else:
            print("[DECISION GATE] MDR wins or ties — consider simpler approach")

    out_file = RESULTS_DIR / f"full_system_musique_{args.max_examples}.json"
    with open(out_file, "w") as f:
        json.dump({
            "max_examples": args.max_examples,
            "alpha":        args.alpha,
            "systems":      all_systems,
        }, f, indent=2)
    print(f"\n[runner] Saved → {out_file}")


if __name__ == "__main__":
    main()
