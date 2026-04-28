"""
Pipeline diagnostic: find exactly where the chain breaks.

Checks (all on cached data — no recomputation):
  1. SEED HIT RATE     — does any gold passage appear in top-5 seeds?
  2. GRAPH COVERAGE    — for each consecutive gold pair (A→B), is the edge in the graph?
  3. CHAIN COMPLETION  — by hop count (2/3/4), what fraction of gold passages are retrieved?
  4. COMPLEMENT DRIFT  — avg cosine(edge_vecs[(A,B)], m1_emb[B])
                         (if ≈1.0 → complement ≈ passage embedding → scoring = plain cosine)

Usage:
    python diagnose_pipeline.py
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from baselines import BM25Retriever, DenseRetriever, reciprocal_rank_fusion
from data_loader import load_musique

CACHE_DIR = Path(__file__).parent / "cache"

N_SEEDS    = 5
CACHE_NAME = "musique_val_None"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def print_bar(label: str, value: float, width: int = 30) -> None:
    filled = int(round(value * width))
    bar    = "█" * filled + "░" * (width - filled)
    print(f"  {label:45s} [{bar}]  {value*100:5.1f}%")


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────────
    print("Loading MuSiQue dev ...")
    corpus, queries = load_musique(split="validation", cache=True)
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(corpus)}
    print(f"  {len(corpus):,} chunks | {len(queries):,} queries")

    # ── Load cached graph ────────────────────────────────────────────────────────
    graph_path = CACHE_DIR / f"graph_v2_{CACHE_NAME}_m1.pkl"
    if not graph_path.exists():
        print(f"ERROR: graph cache not found at {graph_path}")
        print("  Run run_full_system.py first to build all caches.")
        sys.exit(1)
    print(f"Loading graph from {graph_path} ...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    graph_edges: set = set()
    for a_id, neighbors in graph.items():
        for (b_id, _, _) in neighbors:
            graph_edges.add((a_id, b_id))
    print(f"  {len(graph):,} nodes | {len(graph_edges):,} directed edges")

    # ── Load cached edge vectors ─────────────────────────────────────────────────
    ev_path = CACHE_DIR / f"edge_vecs_{CACHE_NAME}_m1.pkl"
    if not ev_path.exists():
        print(f"ERROR: edge_vecs cache not found at {ev_path}")
        sys.exit(1)
    print(f"Loading edge vectors from {ev_path} ...")
    with open(ev_path, "rb") as f:
        edge_vecs = pickle.load(f)
    print(f"  {len(edge_vecs):,} edge vectors loaded")

    # ── Load cached M1 embeddings ────────────────────────────────────────────────
    m1_path = CACHE_DIR / f"m1_emb_{CACHE_NAME}.npy"
    if not m1_path.exists():
        print(f"ERROR: m1_emb cache not found at {m1_path}")
        sys.exit(1)
    print(f"Loading M1 embeddings from {m1_path} ...")
    m1_emb = np.load(str(m1_path))
    print(f"  shape {m1_emb.shape}")

    # ── Build seed retriever ─────────────────────────────────────────────────────
    print("\nBuilding seed retrievers ...")
    bm25  = BM25Retriever()
    bm25.build(corpus, cache_name=f"bm25_{CACHE_NAME}")
    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{CACHE_NAME}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1 — SEED HIT RATE
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: SEED HIT RATE (top-5 seeds contain a gold passage?)")
    print("="*70)

    seed_hit_any      = 0   # ≥1 gold passage in seeds
    seed_hit_all      = 0   # ALL gold passages in seeds
    seed_first_hop    = 0   # first chain passage in seeds
    total             = 0

    per_hop_seed: dict = defaultdict(lambda: {"hit": 0, "total": 0})

    for q in tqdm(queries, desc="Seed check"):
        gold    = set(q["relevant_chunk_ids"])
        chain   = q.get("chain_chunk_ids", [])
        n_hops  = q.get("hop_count", len(chain))

        dense_list = dense.retrieve(q["question"], top_k=N_SEEDS * 3)
        bm25_list  = bm25.retrieve(q["question"],  top_k=N_SEEDS * 3)
        seeds      = set(reciprocal_rank_fusion([dense_list, bm25_list])[:N_SEEDS])

        hit_any = bool(gold & seeds)
        hit_all = gold.issubset(seeds)
        hit_first = bool(chain) and chain[0] in seeds

        seed_hit_any   += int(hit_any)
        seed_hit_all   += int(hit_all)
        seed_first_hop += int(hit_first)
        total          += 1

        per_hop_seed[n_hops]["total"] += 1
        per_hop_seed[n_hops]["hit"]   += int(hit_any)

    print(f"\n  Total queries: {total:,}")
    print_bar("Any gold in seeds",        seed_hit_any  / total)
    print_bar("First chain passage in seeds", seed_first_hop / total)
    print_bar("ALL gold in seeds",        seed_hit_all  / total)
    print("\n  By hop count:")
    for hop in sorted(per_hop_seed):
        d = per_hop_seed[hop]
        print_bar(f"  {hop}-hop: any gold in seeds", d["hit"] / max(d["total"], 1))

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2 — GRAPH EDGE COVERAGE
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: GRAPH EDGE COVERAGE (gold chain edges in graph?)")
    print("="*70)

    edge_total    = 0
    edge_in_graph = 0
    edge_has_vec  = 0

    full_chain_covered   = 0   # all consecutive pairs have graph edges
    full_chain_total     = 0

    per_hop_edge: dict = defaultdict(lambda: {"covered": 0, "total": 0})

    for q in queries:
        chain  = q.get("chain_chunk_ids", [])
        n_hops = q.get("hop_count", len(chain))
        if len(chain) < 2:
            continue

        all_covered = True
        for i in range(len(chain) - 1):
            a_id, b_id = chain[i], chain[i + 1]
            in_graph   = (a_id, b_id) in graph_edges
            has_vec    = (a_id, b_id) in edge_vecs
            edge_total    += 1
            edge_in_graph += int(in_graph)
            edge_has_vec  += int(has_vec)
            if not in_graph:
                all_covered = False

        full_chain_covered += int(all_covered)
        full_chain_total   += 1
        per_hop_edge[n_hops]["total"]   += 1
        per_hop_edge[n_hops]["covered"] += int(all_covered)

    print(f"\n  Total gold consecutive pairs: {edge_total:,}")
    print_bar("Gold pair (A→B) has graph edge", edge_in_graph / max(edge_total, 1))
    print_bar("Gold pair (A→B) has edge vector", edge_has_vec  / max(edge_total, 1))
    print_bar("Full chain: ALL edges covered",   full_chain_covered / max(full_chain_total, 1))
    print("\n  By hop count (full chain coverage):")
    for hop in sorted(per_hop_edge):
        d = per_hop_edge[hop]
        print_bar(f"  {hop}-hop: full chain in graph", d["covered"] / max(d["total"], 1))

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 3 — CHAIN COMPLETION (simulate retrieval, count gold found)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: CHAIN COMPLETION (how many gold passages found per hop?)")
    print("="*70)

    # For each query: simulate the graph traversal (no model scoring — just coverage)
    # Checks: if we had perfect scoring, could the graph even reach all gold passages?
    per_hop_found: dict = defaultdict(lambda: {"found": 0, "gold": 0, "queries": 0, "all_found": 0})

    for q in queries:
        gold   = set(q["relevant_chunk_ids"])
        chain  = q.get("chain_chunk_ids", [])
        n_hops = q.get("hop_count", len(chain))

        # seeds
        dense_list = dense.retrieve(q["question"], top_k=N_SEEDS * 3)
        bm25_list  = bm25.retrieve(q["question"],  top_k=N_SEEDS * 3)
        seeds      = set(reciprocal_rank_fusion([dense_list, bm25_list])[:N_SEEDS])

        # BFS over graph from seeds — what gold passages are reachable within 3 hops?
        reachable = set(seeds)
        frontier  = set(seeds)
        for _ in range(3):
            next_frontier = set()
            for node in frontier:
                for (nbr, _, _) in graph.get(node, []):
                    if nbr not in reachable:
                        reachable.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier

        found_gold = gold & reachable
        all_found  = gold.issubset(reachable)

        per_hop_found[n_hops]["found"]     += len(found_gold)
        per_hop_found[n_hops]["gold"]      += len(gold)
        per_hop_found[n_hops]["queries"]   += 1
        per_hop_found[n_hops]["all_found"] += int(all_found)

    print("\n  Upper-bound recall (BFS from seeds, depth 3, perfect scoring):")
    total_found = sum(d["found"] for d in per_hop_found.values())
    total_gold  = sum(d["gold"]  for d in per_hop_found.values())
    print_bar("Overall upper-bound recall", total_found / max(total_gold, 1))
    print()
    for hop in sorted(per_hop_found):
        d = per_hop_found[hop]
        frac = d["found"] / max(d["gold"], 1)
        all_frac = d["all_found"] / max(d["queries"], 1)
        print(f"  {hop}-hop  ({d['queries']:4d} queries):")
        print_bar(f"    passage recall (BFS upper bound)", frac)
        print_bar(f"    all gold reachable (full coverage)", all_frac)

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 4 — COMPLEMENT DRIFT
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: COMPLEMENT DRIFT  (complement ≈ passage embedding?)")
    print("="*70)

    # Sample gold edges only (more meaningful than random graph edges)
    cosines = []
    sampled = 0
    for q in queries:
        chain = q.get("chain_chunk_ids", [])
        for i in range(len(chain) - 1):
            a_id, b_id = chain[i], chain[i + 1]
            if (a_id, b_id) in edge_vecs and b_id in id_to_idx:
                ev  = edge_vecs[(a_id, b_id)]
                m1v = m1_emb[id_to_idx[b_id]]
                cosines.append(cosine(ev, m1v))
                sampled += 1
        if sampled >= 2000:
            break

    # Also sample 500 random non-gold graph edges for comparison
    random_cosines = []
    count = 0
    for (a_id, b_id), ev in edge_vecs.items():
        if b_id in id_to_idx:
            m1v = m1_emb[id_to_idx[b_id]]
            random_cosines.append(cosine(ev, m1v))
            count += 1
        if count >= 500:
            break

    print(f"\n  Sampled {len(cosines)} gold chain edges, {len(random_cosines)} random edges")
    print(f"\n  cosine(complement(A,B),  m1_emb[B]):")
    print(f"    Gold edges  — mean: {np.mean(cosines):.4f}  "
          f"std: {np.std(cosines):.4f}  "
          f"median: {np.median(cosines):.4f}")
    if random_cosines:
        print(f"    Random edges— mean: {np.mean(random_cosines):.4f}  "
              f"std: {np.std(random_cosines):.4f}  "
              f"median: {np.median(random_cosines):.4f}")

    print()
    thresholds = [0.95, 0.90, 0.80, 0.70]
    for t in thresholds:
        pct = np.mean(np.array(cosines) > t)
        print(f"    Gold edges with cosine > {t:.2f}: {pct*100:.1f}%")

    print("\n  Interpretation:")
    mean_cos = np.mean(cosines)
    if mean_cos > 0.92:
        print("  ⚠  HIGH DRIFT: complement ≈ passage embedding")
        print("     L_content dominates — complement loses its directional signal")
        print("     Scoring with edge_vec ≈ scoring with m1_emb[B] (plain cosine)")
    elif mean_cos > 0.75:
        print("  ⚠  MODERATE DRIFT: complement has some passage-encoding bias")
        print("     L_chain is having partial effect but L_content still dominates")
    else:
        print("  ✓  LOW DRIFT: complement is directionally distinct from passage embedding")
        print("     If Model 2 still underperforms, problem is training data, not loss weights")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  1. Seeds catch ≥1 gold passage:  {seed_hit_any/total*100:.1f}%")
    print(f"  2. Full gold chain in graph:      {full_chain_covered/max(full_chain_total,1)*100:.1f}%")
    print(f"  3. BFS upper-bound recall:        {total_found/max(total_gold,1)*100:.1f}%")
    print(f"  4. Complement↔passage cosine:     {np.mean(cosines):.3f}")
    print()
    print("  Bottleneck is where the biggest gap appears relative to a 100% target.")


if __name__ == "__main__":
    main()
