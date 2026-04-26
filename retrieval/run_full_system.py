"""
Full system evaluation: graph traversal with trained Model 1 + Model 2.

End-to-End Architecture
=======================

OFFLINE (runs once, results cached to disk):
  1. Generic dense (sentence-transformer) FAISS + BM25
       → used ONLY for initial seed retrieval
  2. Model 1 encode_passage(P) on every chunk → M1 embeddings [N, 128]
       L_chain training: consecutive hops (B,C) are closer in this space
       than distractor pairs — reasoning-chain proximity, not topic sim.
  3. Graph built with M1 embeddings for semantic edges
       Same graph_builder.py; 128-dim M1 vecs replace 384-dim generic vecs.
       Sequential + BM25 within-example edges unchanged.
  4. Model 1 complement(A, B) on EVERY graph edge → edge_vectors dict
       edge_vectors[(A,B)] = mean_pool(complement(A,B))  [128-dim]
       Stored offline. At query time: instant dict lookup, no BERT call.
  5. M1 FAISS index (same M1 embeddings as step 2)
       Used in hop 2+ for complement-directed candidate retrieval.

ONLINE (per query):
  Seed   : generic FAISS + BM25 → RRF → top-5 seeds
  q_vec  : QueryEncoder(Q) → [128] in M1 space (initialised from M1 weights)

  Hop 1  : for each seed A, walk graph edges:
               score(A→B) = dot(q_vec, edge_vectors[(A,B)])   ← dict lookup
             select top beam_width=3 B

  Hop 2+ : for each (prev=A, curr=B) in beam:
               nav_vec = edge_vectors[(A,B)]   ← pre-computed, instant
               C_candidates = M1_FAISS.search(nav_vec, top_k=20)
                   # nav_vec ≈ encode_passage(C) by L_chain → finds C directly
               for each C:
                   if edge (B,C) in edge_vectors:
                       score = dot(q_vec, edge_vectors[(B,C)])
                   else:
                       score = dot(q_vec, m1_embeddings[C])   # fallback
             select top beam_width=3 C

  Stop   : no candidates  OR  best_score < 0.05  OR  max_hops reached

Ablations:
  MDR (dense, iterative)          — published baseline, no graph
  Graph + direct cosine (M1)      — M1 graph, dot(q_vec, M1_emb[B]), no complement
  FULL: M1 graph + complement     — full system above

Usage:
    python run_full_system.py --max_examples 300   # smoke test
    python run_full_system.py                       # full 2417 dev eval
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
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
BEAM_WIDTH   = 3
MAX_HOPS     = 3
N_SEEDS      = 5
STOP_THRESH  = 0.05
MAX_LEN_Q    = 64
MAX_LEN_AB   = 256
M1_MAX_LEN   = 128    # matches model1_train MAX_LEN_C
M1_BATCH     = 64     # passages per batch for encode_passage
EDGE_BATCH   = 8      # (A,B) pairs per batch for complement edge vectors
FAISS_TOP_K  = 20     # FAISS candidates per (prev,curr) pair in hop 2+


# ── Offline Step 1: M1 passage embeddings ────────────────────────────────────

@torch.no_grad()
def compute_m1_embeddings(
    comp_enc:   ComplementEncoder,
    corpus:     List[Dict],
    tokenizer:  BertTokenizerFast,
    device:     str,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    encode_passage(P) for every corpus chunk → [N, 128] float32.

    These vectors are in the reasoning-chain space shaped by L_chain:
    consecutive hop pairs (B, C) are closer together than (B_neg, C).
    Used for both semantic graph edges and the FAISS candidate index.
    """
    if cache_path and cache_path.exists():
        print(f"[m1_emb] Loading cached M1 embeddings from {cache_path}")
        return np.load(str(cache_path))

    print(f"[m1_emb] Computing encode_passage for {len(corpus):,} passages ...")
    comp_enc.eval()
    vecs  = []
    texts = [c["text"] for c in corpus]

    for start in tqdm(range(0, len(texts), M1_BATCH), desc="M1 encode_passage"):
        batch = texts[start : start + M1_BATCH]
        enc   = tokenizer(
            text=batch, max_length=M1_MAX_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        v = comp_enc.encode_passage(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )                                        # [B, 128] L2-normalised
        vecs.append(v.cpu().numpy())

    m1_emb = np.vstack(vecs).astype(np.float32)
    print(f"[m1_emb] Done — shape {m1_emb.shape}")
    if cache_path:
        np.save(str(cache_path), m1_emb)
        print(f"[m1_emb] Cached → {cache_path}")
    return m1_emb


# ── Offline Step 2: edge vectors for every graph edge ────────────────────────

@torch.no_grad()
def compute_edge_vectors(
    comp_enc:   ComplementEncoder,
    corpus:     List[Dict],
    graph:      Dict,
    tokenizer:  BertTokenizerFast,
    device:     str,
    cache_path: Optional[Path] = None,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    For every directed edge (A→B) in the graph run complement(A,B) offline
    and store the mean-pooled 128-dim vector.

    At query time this becomes an instant dict lookup — no BERT call needed.
    With ~250K graph edges and EDGE_BATCH=8, runs in ~6 min on T4 (one-time).

    Returns:
        edge_vecs: {(chunk_a_id, chunk_b_id): np.array([128])}
    """
    if cache_path and cache_path.exists():
        print(f"[edge_vecs] Loading cached edge vectors from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    id_to_text = {c["chunk_id"]: c["text"] for c in corpus}
    sep_id     = tokenizer.sep_token_id

    # Collect all directed edges
    all_edges: List[Tuple[str, str]] = [
        (a_id, b_id)
        for a_id, neighbors in graph.items()
        for (b_id, _, _) in neighbors
    ]
    print(f"[edge_vecs] Pre-computing complement vectors for {len(all_edges):,} edges ...")
    comp_enc.eval()
    edge_vecs: Dict[Tuple[str, str], np.ndarray] = {}

    for start in tqdm(range(0, len(all_edges), EDGE_BATCH), desc="Edge vectors"):
        batch  = all_edges[start : start + EDGE_BATCH]
        valid  = [(a, b) for (a, b) in batch
                  if a in id_to_text and b in id_to_text]
        if not valid:
            continue

        texts_a = [id_to_text[a] for a, b in valid]
        texts_b = [id_to_text[b] for a, b in valid]

        enc = tokenizer(
            text=texts_a, text_pair=texts_b,
            max_length=MAX_LEN_AB, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        b_mask = (
            (enc["token_type_ids"] == 1) & (enc["input_ids"] != sep_id)
        ).to(device)

        comp_tokens, pad_mask = comp_enc(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc["token_type_ids"].to(device),
            b_mask,
        )
        vecs_np = mean_pool(comp_tokens, pad_mask).cpu().numpy()  # [B, 128]

        for i, (a_id, b_id) in enumerate(valid):
            edge_vecs[(a_id, b_id)] = vecs_np[i]

    print(f"[edge_vecs] Done — {len(edge_vecs):,} edge vectors stored")
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(edge_vecs, f)
        print(f"[edge_vecs] Cached → {cache_path}")
    return edge_vecs


# ── Online query encoder ──────────────────────────────────────────────────────

class LiveScorer:
    """Encodes a query → q_vec [1, 128] in the M1 projection space."""

    def __init__(
        self,
        query_enc: QueryEncoder,
        tokenizer: BertTokenizerFast,
        device:    str = DEVICE,
    ):
        self.query_enc = query_enc
        self.tokenizer = tokenizer
        self.device    = device
        query_enc.eval()

    @torch.no_grad()
    def encode_query(self, question: str) -> torch.Tensor:
        enc = self.tokenizer(
            text=question, max_length=MAX_LEN_Q,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return self.query_enc(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )   # [1, 128] L2-normalised


# ── Full system ───────────────────────────────────────────────────────────────

class FullGraphTraversal:
    """
    Hop 1  : graph edges from each seed, scored with pre-computed edge_vectors.
               score(A→B) = dot(q_vec, edge_vectors[(A,B)])   ← dict lookup

    Hop 2+ : pre-computed edge_vectors[(prev,curr)] used as FAISS nav query.
               L_chain: edge_vectors[(A,B)] ≈ encode_passage(C)
               → M1 FAISS search finds C directly, no graph edge (B→C) needed.
               score(B→C) = edge_vectors[(B,C)] if exists, else M1_emb[C].

    Everything is in the 128-dim M1 space. No mixed-space arithmetic.
    No on-the-fly BERT calls after offline pre-computation is done.
    """

    def __init__(
        self,
        scorer:       LiveScorer,
        graph:        Dict,
        corpus:       List[Dict],
        edge_vecs:    Dict[Tuple[str, str], np.ndarray],
        m1_embeddings: np.ndarray,
        m1_index:     faiss.IndexFlatIP,
        bm25:         BM25Retriever,
        dense:        DenseRetriever,
        beam_width:   int   = BEAM_WIDTH,
        max_hops:     int   = MAX_HOPS,
        n_seeds:      int   = N_SEEDS,
        stop_thresh:  float = STOP_THRESH,
        faiss_top_k:  int   = FAISS_TOP_K,
    ):
        self.scorer        = scorer
        self.graph         = graph
        self.edge_vecs     = edge_vecs
        self.m1_embeddings = m1_embeddings
        self.m1_index      = m1_index
        self.bm25          = bm25
        self.dense         = dense
        self.beam_width    = beam_width
        self.max_hops      = max_hops
        self.n_seeds       = n_seeds
        self.stop_thresh   = stop_thresh
        self.faiss_top_k   = faiss_top_k

        self.corpus_ids = [c["chunk_id"] for c in corpus]
        self.id_to_idx  = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        # ── Seeds ────────────────────────────────────────────────────────────
        dense_list = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_list  = self.bm25.retrieve(question,  top_k=self.n_seeds * 3)
        seeds      = reciprocal_rank_fusion([dense_list, bm25_list])[:self.n_seeds]

        q_vec = self.scorer.encode_query(question)   # [1, 128]
        q_np  = q_vec.cpu().numpy()[0]                # [128]

        retrieved:     List[str] = list(seeds)
        retrieved_set: Set[str]  = set(seeds)

        # beam: (prev_id_or_None, curr_id)
        # prev_id=None for seeds — no edge_vector available yet for navigation
        beam: List[Tuple[Optional[str], str]] = [(None, s) for s in seeds]

        for _ in range(self.max_hops):
            # candidates[nbr_id] = (score, parent_curr_id)
            candidates: Dict[str, Tuple[float, str]] = {}

            for (prev_id, curr_id) in beam:

                if prev_id is None:
                    # ── Hop 1: graph edges + pre-computed edge vectors ────────
                    for (nbr_id, _, _) in self.graph.get(curr_id, []):
                        if nbr_id in retrieved_set:
                            continue
                        ev = self.edge_vecs.get((curr_id, nbr_id))
                        if ev is None:
                            continue
                        score = float(np.dot(q_np, ev))
                        if nbr_id not in candidates or score > candidates[nbr_id][0]:
                            candidates[nbr_id] = (score, curr_id)

                else:
                    # ── Hop 2+: complement-directed FAISS ────────────────────
                    # edge_vectors[(prev,curr)] ≈ encode_passage(next-hop)
                    # → use as FAISS query to find next-hop candidates directly
                    nav_vec = self.edge_vecs.get((prev_id, curr_id))
                    if nav_vec is None:
                        continue

                    nav_np  = nav_vec.reshape(1, -1).astype(np.float32)
                    _, idxs = self.m1_index.search(nav_np, self.faiss_top_k * 3)

                    nbr_pool = [
                        self.corpus_ids[i]
                        for i in idxs[0]
                        if 0 <= i < len(self.corpus_ids)
                        and self.corpus_ids[i] not in retrieved_set
                        and self.corpus_ids[i] != curr_id
                    ][:self.faiss_top_k]

                    for nbr_id in nbr_pool:
                        ev = self.edge_vecs.get((curr_id, nbr_id))
                        if ev is not None:
                            # edge (curr→nbr) is in graph: use pre-computed vec
                            score = float(np.dot(q_np, ev))
                        else:
                            # edge not in graph (FAISS-only candidate):
                            # fall back to direct M1 embedding similarity
                            nbr_idx = self.id_to_idx.get(nbr_id)
                            if nbr_idx is None:
                                continue
                            score = float(np.dot(q_np, self.m1_embeddings[nbr_idx]))

                        if nbr_id not in candidates or score > candidates[nbr_id][0]:
                            candidates[nbr_id] = (score, curr_id)

            if not candidates:
                break
            if max(s for s, _ in candidates.values()) < self.stop_thresh:
                break

            top_nbrs = sorted(
                candidates, key=lambda k: candidates[k][0], reverse=True
            )[:self.beam_width]

            # track (curr, nbr) pairs for next hop's complement-directed nav
            beam = [(candidates[nbr][1], nbr) for nbr in top_nbrs]
            retrieved_set.update(top_nbrs)
            retrieved.extend(top_nbrs)

        return retrieved[:top_k]


# ── Ablation: M1 graph + direct M1 cosine, no complement ─────────────────────

class GraphDirectCosine:
    """
    Graph traversal with dot(q_vec, M1_emb[B]).
    Uses the same M1 graph and M1 space as FullGraphTraversal.
    Ablation: isolates the value of complement edge vectors vs. direct passage sim.
    """

    def __init__(
        self,
        graph:         Dict,
        corpus:        List[Dict],
        m1_embeddings: np.ndarray,
        scorer:        LiveScorer,
        bm25:          BM25Retriever,
        dense:         DenseRetriever,
        beam_width:    int   = BEAM_WIDTH,
        max_hops:      int   = MAX_HOPS,
        n_seeds:       int   = N_SEEDS,
        stop_thresh:   float = STOP_THRESH,
    ):
        self.graph         = graph
        self.m1_embeddings = m1_embeddings
        self.scorer        = scorer
        self.bm25          = bm25
        self.dense         = dense
        self.beam_width    = beam_width
        self.max_hops      = max_hops
        self.n_seeds       = n_seeds
        self.stop_thresh   = stop_thresh
        self.id_to_idx     = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        dense_list = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_list  = self.bm25.retrieve(question,  top_k=self.n_seeds * 3)
        seeds      = reciprocal_rank_fusion([dense_list, bm25_list])[:self.n_seeds]

        q_vec = self.scorer.encode_query(question)
        q_np  = q_vec.cpu().numpy()[0]   # [128]

        retrieved:     List[str] = list(seeds)
        retrieved_set: Set[str]  = set(seeds)
        beam:          List[str] = list(seeds)

        for _ in range(self.max_hops):
            candidates: Dict[str, float] = {}

            for chunk_id in beam:
                for (nbr_id, _, _) in self.graph.get(chunk_id, []):
                    if nbr_id in retrieved_set:
                        continue
                    nbr_idx = self.id_to_idx.get(nbr_id)
                    if nbr_idx is None:
                        continue
                    score = float(np.dot(q_np, self.m1_embeddings[nbr_idx]))
                    if nbr_id not in candidates or score > candidates[nbr_id]:
                        candidates[nbr_id] = score

            if not candidates or max(candidates.values()) < self.stop_thresh:
                break

            top_nbrs = sorted(
                candidates, key=candidates.__getitem__, reverse=True
            )[:self.beam_width]
            beam = top_nbrs
            retrieved_set.update(top_nbrs)
            retrieved.extend(top_nbrs)

        return retrieved[:top_k]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_retriever(
    name:    str,
    ret,
    queries: List[Dict],
    top_k:   int = 10,
) -> Dict:
    print(f"\n[runner] Evaluating: {name} ...")
    t0, results = time.time(), []
    for q in tqdm(queries, desc=name, leave=False):
        results.append({
            "query_id":  q["query_id"],
            "retrieved": ret.retrieve(q["question"], top_k=top_k),
            "relevant":  q["relevant_chunk_ids"],
        })
    elapsed = time.time() - t0
    metrics = evaluate_retriever(results, ks=[2, 5, 10])
    metrics["latency_ms"] = round(elapsed / len(queries) * 1000, 1)
    print(f"  R@10={metrics.get('recall@10', 0):.4f} | {elapsed:.1f}s"
          f" | {metrics['latency_ms']:.1f} ms/query")
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap MuSiQue dev examples (None = all 2417)")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # ── Data ─────────────────────────────────────────────────────────────────
    corpus, queries = load_musique(
        split="validation", max_examples=args.max_examples, cache=True
    )
    cache_name = f"musique_val_{args.max_examples}"
    print(f"[runner] {len(corpus):,} chunks | {len(queries):,} queries | {DEVICE}")

    # ── Seed retrieval indexes (generic, unchanged) ───────────────────────────
    bm25 = BM25Retriever()
    bm25.build(corpus, cache_name=f"bm25_{cache_name}")

    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{cache_name}")

    # ── Load trained models ───────────────────────────────────────────────────
    m1_ckpt    = MODEL_DIR / "model1_complement.pt"
    m2_ckpt    = MODEL_DIR / "model2_scorer.pt"
    has_models = m1_ckpt.exists() and m2_ckpt.exists()

    scorer    = None
    m1_emb    = None
    m1_index  = None
    edge_vecs = None
    graph     = None

    if has_models:
        comp_enc = ComplementEncoder().to(DEVICE)
        comp_enc.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
        comp_enc.eval()

        query_enc = QueryEncoder().to(DEVICE)
        query_enc.load_state_dict(torch.load(m2_ckpt, map_location=DEVICE))
        query_enc.eval()

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        scorer    = LiveScorer(query_enc, tokenizer, DEVICE)
        print("[runner] Model 1 + Model 2 loaded")

        # ── Step 1: M1 passage embeddings ────────────────────────────────────
        m1_emb = compute_m1_embeddings(
            comp_enc, corpus, tokenizer, DEVICE,
            cache_path=CACHE_DIR / f"m1_emb_{cache_name}.npy",
        )

        # ── Step 2: M1 FAISS index ───────────────────────────────────────────
        m1_index = faiss.IndexFlatIP(m1_emb.shape[1])   # 128-dim
        m1_index.add(m1_emb)
        print(f"[runner] M1 FAISS: {m1_index.ntotal:,} vectors @ {m1_emb.shape[1]}-dim")

        # ── Step 3: Graph built with M1 embeddings ────────────────────────────
        graph = build_graph(
            corpus,
            embeddings=m1_emb,
            cache_name=cache_name + "_m1",
        )

        # ── Step 4: Pre-compute complement vector for every graph edge ────────
        edge_vecs = compute_edge_vectors(
            comp_enc, corpus, graph, tokenizer, DEVICE,
            cache_path=CACHE_DIR / f"edge_vecs_{cache_name}_m1.pkl",
        )
        print(f"[runner] Edge vectors: {len(edge_vecs):,} edges pre-computed")

    else:
        missing = [str(p) for p in [m1_ckpt, m2_ckpt] if not p.exists()]
        print(f"[runner] Missing checkpoints: {missing} — running MDR only")
        n, dim    = dense.index.ntotal, dense.index.d
        dense_emb = np.zeros((n, dim), dtype=np.float32)
        dense.index.reconstruct_n(0, n, dense_emb)
        graph = build_graph(corpus, embeddings=dense_emb, cache_name=cache_name)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_systems: Dict[str, Dict] = {}

    # MDR baseline
    mdr             = MDRBaseline(max_hops=3, seeds_per_hop=5)
    mdr.dense       = dense
    mdr.id_to_chunk = {c["chunk_id"]: c for c in corpus}
    all_systems["MDR (dense, iterative)"] = run_retriever(
        "MDR", mdr, queries, args.top_k
    )

    if scorer is not None:
        # Ablation: M1 graph + direct cosine (no complement)
        g_cos = GraphDirectCosine(
            graph, corpus, m1_emb, scorer, bm25, dense
        )
        all_systems["Graph + direct cosine (M1)"] = run_retriever(
            "Graph+M1cos", g_cos, queries, args.top_k
        )

        # Full system
        full = FullGraphTraversal(
            scorer, graph, corpus,
            edge_vecs, m1_emb, m1_index,
            bm25, dense,
        )
        all_systems["FULL: M1 graph + complement + FAISS"] = run_retriever(
            "Full", full, queries, args.top_k
        )

    # ── Results ───────────────────────────────────────────────────────────────
    compare_systems(all_systems)

    if scorer is not None and "FULL: M1 graph + complement + FAISS" in all_systems:
        mdr_r10  = all_systems["MDR (dense, iterative)"]["recall@10"]
        full_r10 = all_systems["FULL: M1 graph + complement + FAISS"]["recall@10"]
        gap      = full_r10 - mdr_r10
        print(f"\n[DECISION GATE] Full system R@10 : {full_r10:.4f}")
        print(f"[DECISION GATE] MDR baseline R@10 : {mdr_r10:.4f}")
        print(f"[DECISION GATE] Gap               : {gap:+.4f} ({gap*100:+.2f}%)")
        if gap > 0.03:
            print("[DECISION GATE] FULL SYSTEM WINS by >3% ✓")
        elif gap > 0.0:
            print("[DECISION GATE] FULL SYSTEM WINS but gap ≤3% — marginal")
        else:
            print("[DECISION GATE] MDR wins or ties — consider simpler approach")

    out = RESULTS_DIR / f"full_system_musique_{args.max_examples}.json"
    with open(out, "w") as f:
        json.dump({"max_examples": args.max_examples, "systems": all_systems}, f, indent=2)
    print(f"\n[runner] Results saved → {out}")


if __name__ == "__main__":
    main()
