"""
Pipeline Diagnostic — where are we losing points?

Measures loss at each stage independently:

  Stage 1 — Seed retrieval   : Does initial BM25+dense find any relevant passage?
  Stage 2 — First-hop seed   : Is the CORRECT first-hop passage in the seeds?
  Stage 3 — Graph edge cover : For each chain A→B, does the graph have that edge?
  Stage 4 — Oracle seed      : If we GIVE the correct first hop, what is R@10?
  Stage 5 — Oracle graph     : If the graph had PERFECT edges, what is R@10?
  Stage 6 — Hop breakdown    : How does performance drop from 2-hop to 3/4-hop?

Usage:
    python diagnose.py                   # full 2417 queries
    python diagnose.py --max_examples 300
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_musique
from baselines import DenseRetriever, BM25Retriever, reciprocal_rank_fusion
from graph_builder import build_graph
from evaluate import recall_at_k

CACHE_DIR = Path(__file__).parent / "cache"


# ── helpers ────────────────────────────────────────────────────────────────────

def seeds_for_query(question: str, dense: DenseRetriever,
                    bm25: BM25Retriever, n_seeds: int = 5) -> List[str]:
    d = dense.retrieve(question, top_k=n_seeds * 3)
    b = bm25.retrieve(question,  top_k=n_seeds * 3)
    return reciprocal_rank_fusion([d, b])[:n_seeds]


def graph_traverse(seeds: List[str], graph: Dict, embeddings: np.ndarray,
                   id_to_idx: Dict, query_emb: np.ndarray,
                   max_hops: int = 3, beam_width: int = 5,
                   top_k: int = 10) -> List[str]:
    QR_MIX     = 0.3
    STOP_THRESH = 0.0

    retrieved     = list(seeds)
    retrieved_set = set(seeds)
    beam          = list(seeds)

    for _ in range(max_hops):
        beam_idxs = [id_to_idx[c] for c in beam if c in id_to_idx]
        if beam_idxs:
            beam_emb  = embeddings[beam_idxs].mean(axis=0)
            q         = query_emb + QR_MIX * beam_emb
            query_emb = q / (np.linalg.norm(q) + 1e-9)

        candidates: Dict[str, float] = {}
        for cid in beam:
            for (nbr_id, _, _) in graph.get(cid, []):
                if nbr_id in retrieved_set:
                    continue
                idx = id_to_idx.get(nbr_id)
                if idx is None:
                    continue
                score = float(np.dot(query_emb, embeddings[idx]))
                if nbr_id not in candidates or score > candidates[nbr_id]:
                    candidates[nbr_id] = score

        if not candidates or max(candidates.values()) < STOP_THRESH:
            break
        top_nbrs = sorted(candidates, key=candidates.__getitem__, reverse=True)[:beam_width]
        beam = top_nbrs
        retrieved_set.update(top_nbrs)
        retrieved.extend(top_nbrs)

    return retrieved[:top_k]


# ── main diagnostic ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--n_seeds",      type=int, default=5)
    parser.add_argument("--top_k",        type=int, default=10)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    corpus, queries = load_musique(
        split="validation", max_examples=args.max_examples, cache=True
    )
    cache_name = f"musique_val_{args.max_examples}"
    print(f"\nLoaded: {len(corpus):,} chunks | {len(queries):,} queries")

    # ── Build / load indexes ───────────────────────────────────────────────────
    bm25 = BM25Retriever()
    bm25.build(corpus, cache_name=f"bm25_{cache_name}")

    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{cache_name}")

    n, dim = dense.index.ntotal, dense.index.d
    embeddings = np.zeros((n, dim), dtype=np.float32)
    dense.index.reconstruct_n(0, n, embeddings)
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    graph_cache = CACHE_DIR / f"graph_{cache_name}.pkl"
    if graph_cache.exists():
        with open(graph_cache, "rb") as f:
            graph = pickle.load(f)
    else:
        graph = build_graph(corpus, embeddings=embeddings, cache_name=cache_name)

    # ── Build adjacency set for fast edge lookup ───────────────────────────────
    edge_set: Set[Tuple[str, str]] = set()
    for src, nbrs in graph.items():
        for (dst, _, _) in nbrs:
            edge_set.add((src, dst))

    print(f"Graph: {len(graph):,} nodes | {len(edge_set):,} directed edges")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Seed retrieval: does initial BM25+dense find ANY relevant doc?
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("STAGE 1 — Seed Retrieval Quality")
    print("="*65)

    seed_any_hit   = 0   # ≥1 relevant doc in seeds
    seed_all_hit   = 0   # ALL relevant docs in seeds
    seed_r_total   = 0.0 # avg recall of seed set

    for q in tqdm(queries, desc="Stage 1", leave=False):
        seeds     = seeds_for_query(q["question"], dense, bm25, args.n_seeds)
        relevant  = set(q["relevant_chunk_ids"])
        hits      = sum(1 for s in seeds if s in relevant)
        seed_r_total += hits / len(relevant) if relevant else 0
        if hits >= 1: seed_any_hit += 1
        if hits == len(relevant): seed_all_hit += 1

    n = len(queries)
    print(f"  Queries where ≥1 relevant doc in seeds  : {seed_any_hit:,}/{n:,}  "
          f"({seed_any_hit/n*100:.1f}%)")
    print(f"  Queries where ALL relevant docs in seeds : {seed_all_hit:,}/{n:,}  "
          f"({seed_all_hit/n*100:.1f}%)")
    print(f"  Avg seed recall (hits/relevant)          : {seed_r_total/n*100:.1f}%")
    print(f"  → If seeds fail, traversal cannot recover anything.")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — First-hop seed: is chain[0] (the correct 1st hop) in seeds?
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("STAGE 2 — First-Hop Passage in Seeds")
    print("="*65)

    hop1_in_seed  = 0
    hop1_queries  = 0

    for q in tqdm(queries, desc="Stage 2", leave=False):
        chain = q.get("chain_chunk_ids", [])
        if not chain:
            continue
        hop1_queries += 1
        seeds = seeds_for_query(q["question"], dense, bm25, args.n_seeds)
        if chain[0] in set(seeds):
            hop1_in_seed += 1

    print(f"  Queries with chain info       : {hop1_queries:,}")
    print(f"  Correct 1st-hop in seeds      : {hop1_in_seed:,}/{hop1_queries:,}  "
          f"({hop1_in_seed/hop1_queries*100:.1f}%)")
    print(f"  → If 1st hop is missed, the entire chain fails.")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Graph edge coverage: A→B exists for each hop in chain?
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("STAGE 3 — Graph Edge Coverage per Hop Pair")
    print("="*65)

    total_hops  = 0
    covered_hops = 0
    chains_full  = 0    # ALL edges in chain are covered
    chains_any   = 0    # at least ONE edge is covered
    chains_total = 0

    hop_coverage_by_pos = defaultdict(lambda: [0, 0])  # hop position → [covered, total]

    for q in tqdm(queries, desc="Stage 3", leave=False):
        chain = q.get("chain_chunk_ids", [])
        if len(chain) < 2:
            continue
        chains_total += 1
        chain_covered = True
        any_covered   = False

        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            has_edge = (a, b) in edge_set or (b, a) in edge_set
            total_hops += 1
            hop_coverage_by_pos[i][1] += 1
            if has_edge:
                covered_hops += 1
                hop_coverage_by_pos[i][0] += 1
                any_covered = True
            else:
                chain_covered = False

        if chain_covered: chains_full += 1
        if any_covered:   chains_any  += 1

    print(f"  Total hop pairs in chains     : {total_hops:,}")
    print(f"  Hop pairs with graph edge     : {covered_hops:,}/{total_hops:,}  "
          f"({covered_hops/total_hops*100:.1f}%)")
    print(f"  Chains fully covered (all edges exist)   : "
          f"{chains_full:,}/{chains_total:,}  ({chains_full/chains_total*100:.1f}%)")
    print(f"  Chains partially covered (≥1 edge exists): "
          f"{chains_any:,}/{chains_total:,}  ({chains_any/chains_total*100:.1f}%)")
    print(f"\n  Coverage by hop position:")
    for pos in sorted(hop_coverage_by_pos.keys()):
        cov, tot = hop_coverage_by_pos[pos]
        print(f"    Hop {pos}→{pos+1}: {cov:,}/{tot:,}  ({cov/tot*100:.1f}%)")
    print(f"  → Missing edge = that hop is unreachable no matter how good the scorer.")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Oracle seed: give correct first hop, measure R@10
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("STAGE 4 — Oracle Seed (correct 1st hop given as seed)")
    print("="*65)

    oracle_seed_recalls = []

    for q in tqdm(queries, desc="Stage 4", leave=False):
        chain    = q.get("chain_chunk_ids", [])
        relevant = q["relevant_chunk_ids"]
        if not chain or not relevant:
            continue

        # Use correct first hop as seed instead of retrieval
        oracle_seeds = [chain[0]]
        query_emb    = dense.embed_query(q["question"])
        retrieved    = graph_traverse(
            oracle_seeds, graph, embeddings, id_to_idx, query_emb,
            top_k=args.top_k
        )
        oracle_seed_recalls.append(recall_at_k(retrieved, relevant, args.top_k))

    oracle_seed_r10 = np.mean(oracle_seed_recalls) * 100
    print(f"  R@{args.top_k} with oracle seed (correct 1st hop)  : {oracle_seed_r10:.1f}%")
    print(f"  R@{args.top_k} with actual seed retrieval          : ~38.5%  (from eval)")
    print(f"  Gap closed by fixing seed retrieval        : "
          f"{oracle_seed_r10 - 38.5:.1f} points")
    print(f"  → Shows the ceiling of fixing seed retrieval alone.")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Oracle graph: if ALL chain edges existed, what is R@10?
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("STAGE 5 — Oracle Graph (all chain edges injected)")
    print("="*65)

    # Build augmented graph with all chain edges added
    aug_graph: Dict = {k: list(v) for k, v in graph.items()}
    for q in queries:
        chain = q.get("chain_chunk_ids", [])
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i+1]
            # Add both directions if not already present
            existing_a = {n for (n, _, _) in aug_graph.get(a, [])}
            if b not in existing_a:
                if a not in aug_graph: aug_graph[a] = []
                aug_graph[a].append((b, 1.0, "oracle"))
            existing_b = {n for (n, _, _) in aug_graph.get(b, [])}
            if a not in existing_b:
                if b not in aug_graph: aug_graph[b] = []
                aug_graph[b].append((a, 1.0, "oracle"))

    oracle_graph_recalls = []
    for q in tqdm(queries, desc="Stage 5", leave=False):
        relevant  = q["relevant_chunk_ids"]
        if not relevant:
            continue
        seeds     = seeds_for_query(q["question"], dense, bm25, args.n_seeds)
        query_emb = dense.embed_query(q["question"])
        retrieved = graph_traverse(
            seeds, aug_graph, embeddings, id_to_idx, query_emb,
            top_k=args.top_k
        )
        oracle_graph_recalls.append(recall_at_k(retrieved, relevant, args.top_k))

    oracle_graph_r10 = np.mean(oracle_graph_recalls) * 100
    print(f"  R@{args.top_k} with oracle graph edges    : {oracle_graph_r10:.1f}%")
    print(f"  R@{args.top_k} with actual graph          : ~38.5%  (from eval)")
    print(f"  Gap closed by fixing graph coverage : "
          f"{oracle_graph_r10 - 38.5:.1f} points")
    print(f"  → Shows the ceiling of fixing graph edges alone.")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6 — Breakdown by hop count
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("STAGE 6 — Performance Breakdown by Hop Count")
    print("="*65)

    hop_buckets = defaultdict(list)  # hop_count → list of R@10 per query

    for q in tqdm(queries, desc="Stage 6", leave=False):
        hop_count = q.get("hop_count", 0)
        relevant  = q["relevant_chunk_ids"]
        if not relevant:
            continue
        seeds     = seeds_for_query(q["question"], dense, bm25, args.n_seeds)
        query_emb = dense.embed_query(q["question"])
        retrieved = graph_traverse(
            seeds, graph, embeddings, id_to_idx, query_emb,
            top_k=args.top_k
        )
        hop_buckets[hop_count].append(recall_at_k(retrieved, relevant, args.top_k))

    print(f"  {'Hop Count':<12} {'Queries':>8} {'R@10':>10} {'vs 2-hop':>10}")
    print(f"  {'-'*44}")
    base_r10 = None
    for hops in sorted(hop_buckets.keys()):
        vals  = hop_buckets[hops]
        r10   = np.mean(vals) * 100
        delta = f"{r10 - base_r10:+.1f}" if base_r10 is not None else "baseline"
        if base_r10 is None: base_r10 = r10
        print(f"  {hops}-hop        {len(vals):>8,} {r10:>9.1f}% {delta:>10}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("SUMMARY — Where Points Are Lost")
    print("="*65)
    print(f"  Current R@10 (Graph+cosine)    : ~38.5%")
    print(f"  Oracle seed  (fix retrieval)   : {oracle_seed_r10:.1f}%  "
          f"(+{oracle_seed_r10-38.5:.1f} pts)")
    print(f"  Oracle graph (fix edges)       : {oracle_graph_r10:.1f}%  "
          f"(+{oracle_graph_r10-38.5:.1f} pts)")
    print(f"  Graph edge coverage            : {covered_hops/total_hops*100:.1f}%")
    print(f"  First-hop in seeds             : {hop1_in_seed/hop1_queries*100:.1f}%")
    print()
    if oracle_seed_r10 > oracle_graph_r10:
        print("  VERDICT: Seed retrieval is the bigger bottleneck.")
        print("  Fix: Train the dense retriever on MuSiQue.")
    else:
        print("  VERDICT: Graph coverage is the bigger bottleneck.")
        print("  Fix: Add entity-overlap edges, lower cosine threshold.")


if __name__ == "__main__":
    main()
