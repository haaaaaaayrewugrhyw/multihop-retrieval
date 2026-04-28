"""
Pipeline diagnostic: find exactly where the chain breaks.

Checks:
  1. SEED HIT RATE     — does any gold passage appear in top-5 seeds?
  2. GRAPH COVERAGE    — for each consecutive gold pair (A→B), is the edge in the graph?
  3. CHAIN COMPLETION  — BFS upper-bound: with perfect scoring, what can the graph reach?
  4. COMPLEMENT DRIFT  — avg cosine(edge_vecs[(A,B)], m1_emb[B])
                         (if ≈1.0 → complement ≈ passage embedding → no extra signal)

Caches are rebuilt automatically if missing (~25 min on T4).
If eval_kaggle.ipynb was already run in this session, caches exist and this runs in ~5 min.

Usage:
    python diagnose_pipeline.py
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from baselines import BM25Retriever, DenseRetriever, reciprocal_rank_fusion
from data_loader import load_musique
from graph_builder import build_graph
from model1_train import ComplementEncoder
from model2_train import QueryEncoder
from run_full_system import compute_m1_embeddings, compute_edge_vectors
from transformers import BertTokenizerFast

CACHE_DIR  = Path(__file__).parent / "cache"
MODEL_DIR  = Path(__file__).parent / "models"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
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


def build_all_caches(corpus):
    """Load models and build all caches if missing. Returns (m1_emb, graph, edge_vecs)."""
    CACHE_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    m1_path   = CACHE_DIR / f"m1_emb_{CACHE_NAME}.npy"
    graph_path = CACHE_DIR / f"graph_v2_{CACHE_NAME}_m1.pkl"
    ev_path   = CACHE_DIR / f"edge_vecs_{CACHE_NAME}_m1.pkl"

    # Check if we need models
    need_models = not m1_path.exists() or not ev_path.exists()

    comp_enc  = None
    tokenizer = None

    if need_models:
        m1_ckpt = MODEL_DIR / "model1_complement.pt"
        if not m1_ckpt.exists():
            print(f"ERROR: model1_complement.pt not found at {m1_ckpt}")
            print("  Set up file paths (cell 4) before running.")
            sys.exit(1)

        print(f"[diag] Loading Model 1 from {m1_ckpt} ...")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        comp_enc  = ComplementEncoder().to(DEVICE)
        comp_enc.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
        comp_enc.eval()
        print("[diag] Model 1 loaded")

    # M1 embeddings
    if m1_path.exists():
        print(f"[diag] Loading cached M1 embeddings ...")
        m1_emb = np.load(str(m1_path))
    else:
        print("[diag] Building M1 embeddings (first time, ~6 min on T4) ...")
        m1_emb = compute_m1_embeddings(
            comp_enc, corpus, tokenizer, DEVICE,
            cache_path=m1_path,
        )
    print(f"[diag] M1 embeddings: {m1_emb.shape}")

    # Graph
    if graph_path.exists():
        print(f"[diag] Loading cached graph ...")
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
    else:
        print("[diag] Building graph with M1 embeddings (first time, ~2 min on T4) ...")
        graph = build_graph(corpus, embeddings=m1_emb, cache_name=CACHE_NAME + "_m1")
    total_edges = sum(len(v) for v in graph.values())
    print(f"[diag] Graph: {len(graph):,} nodes | {total_edges:,} directed edges")

    # Edge vectors
    if ev_path.exists():
        print(f"[diag] Loading cached edge vectors ...")
        with open(ev_path, "rb") as f:
            edge_vecs = pickle.load(f)
    else:
        print("[diag] Pre-computing edge vectors (first time, ~3.5 hours on T4) ...")
        print("[diag] This is the slow step — grab a coffee.")
        edge_vecs = compute_edge_vectors(
            comp_enc, corpus, graph, tokenizer, DEVICE,
            cache_path=ev_path,
        )
    print(f"[diag] Edge vectors: {len(edge_vecs):,}")

    return m1_emb, graph, edge_vecs


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────────
    print("Loading MuSiQue dev ...")
    corpus, queries = load_musique(split="validation", cache=True)
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(corpus)}
    print(f"  {len(corpus):,} chunks | {len(queries):,} queries")

    # ── Build / load all caches ──────────────────────────────────────────────────
    m1_emb, graph, edge_vecs = build_all_caches(corpus)

    graph_edges: set = set()
    for a_id, neighbors in graph.items():
        for (b_id, _, _) in neighbors:
            graph_edges.add((a_id, b_id))

    # ── Build seed retrievers ────────────────────────────────────────────────────
    print("\nBuilding seed retrievers ...")
    bm25  = BM25Retriever()
    bm25.build(corpus,  cache_name=f"bm25_{CACHE_NAME}")
    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{CACHE_NAME}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1 — SEED HIT RATE
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: SEED HIT RATE (top-5 seeds contain a gold passage?)")
    print("="*70)

    seed_hit_any   = 0
    seed_hit_all   = 0
    seed_first_hop = 0
    total          = 0
    per_hop_seed: dict = defaultdict(lambda: {"hit": 0, "total": 0})

    for q in tqdm(queries, desc="Seed check"):
        gold   = set(q["relevant_chunk_ids"])
        chain  = q.get("chain_chunk_ids", [])
        n_hops = q.get("hop_count", len(chain))

        dense_list = dense.retrieve(q["question"], top_k=N_SEEDS * 3)
        bm25_list  = bm25.retrieve(q["question"],  top_k=N_SEEDS * 3)
        seeds      = set(reciprocal_rank_fusion([dense_list, bm25_list])[:N_SEEDS])

        hit_any   = bool(gold & seeds)
        hit_all   = gold.issubset(seeds)
        hit_first = bool(chain) and chain[0] in seeds

        seed_hit_any   += int(hit_any)
        seed_hit_all   += int(hit_all)
        seed_first_hop += int(hit_first)
        total          += 1

        per_hop_seed[n_hops]["total"] += 1
        per_hop_seed[n_hops]["hit"]   += int(hit_any)

    print(f"\n  Total queries: {total:,}")
    print_bar("Any gold passage in seeds",       seed_hit_any   / total)
    print_bar("First chain passage in seeds",    seed_first_hop / total)
    print_bar("ALL gold passages in seeds",      seed_hit_all   / total)
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
    full_chain_covered = 0
    full_chain_total   = 0
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
    print_bar("Gold pair (A→B) has graph edge",  edge_in_graph      / max(edge_total, 1))
    print_bar("Gold pair (A→B) has edge vector", edge_has_vec       / max(edge_total, 1))
    print_bar("Full chain: ALL edges covered",   full_chain_covered / max(full_chain_total, 1))
    print("\n  By hop count (full chain coverage):")
    for hop in sorted(per_hop_edge):
        d = per_hop_edge[hop]
        print_bar(f"  {hop}-hop: full chain in graph", d["covered"] / max(d["total"], 1))

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 3 — BFS UPPER-BOUND RECALL
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: BFS UPPER-BOUND RECALL (max recall with perfect scoring)")
    print("="*70)

    per_hop_found: dict = defaultdict(lambda: {"found": 0, "gold": 0, "queries": 0, "all_found": 0})

    for q in tqdm(queries, desc="BFS check"):
        gold   = set(q["relevant_chunk_ids"])
        chain  = q.get("chain_chunk_ids", [])
        n_hops = q.get("hop_count", len(chain))

        dense_list = dense.retrieve(q["question"], top_k=N_SEEDS * 3)
        bm25_list  = bm25.retrieve(q["question"],  top_k=N_SEEDS * 3)
        seeds      = set(reciprocal_rank_fusion([dense_list, bm25_list])[:N_SEEDS])

        # BFS: what gold passages are reachable from seeds within 3 hops?
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
        per_hop_found[n_hops]["found"]     += len(found_gold)
        per_hop_found[n_hops]["gold"]      += len(gold)
        per_hop_found[n_hops]["queries"]   += 1
        per_hop_found[n_hops]["all_found"] += int(gold.issubset(reachable))

    total_found = sum(d["found"] for d in per_hop_found.values())
    total_gold  = sum(d["gold"]  for d in per_hop_found.values())

    print(f"\n  Upper-bound recall (BFS from seeds, depth 3, perfect scoring):")
    print_bar("Overall upper-bound passage recall", total_found / max(total_gold, 1))
    print()
    for hop in sorted(per_hop_found):
        d = per_hop_found[hop]
        print(f"  {hop}-hop  ({d['queries']:4d} queries):")
        print_bar(f"    passage recall (BFS upper bound)",  d["found"]     / max(d["gold"],    1))
        print_bar(f"    all gold reachable",                d["all_found"] / max(d["queries"], 1))

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 4 — COMPLEMENT DRIFT
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: COMPLEMENT DRIFT  (complement ≈ passage embedding?)")
    print("="*70)

    cosines = []
    for q in queries:
        chain = q.get("chain_chunk_ids", [])
        for i in range(len(chain) - 1):
            a_id, b_id = chain[i], chain[i + 1]
            if (a_id, b_id) in edge_vecs and b_id in id_to_idx:
                ev  = edge_vecs[(a_id, b_id)]
                m1v = m1_emb[id_to_idx[b_id]]
                cosines.append(cosine(ev, m1v))
        if len(cosines) >= 2000:
            break

    random_cosines = []
    for (a_id, b_id), ev in edge_vecs.items():
        if b_id in id_to_idx:
            random_cosines.append(cosine(ev, m1_emb[id_to_idx[b_id]]))
        if len(random_cosines) >= 500:
            break

    print(f"\n  Sampled {len(cosines)} gold edges, {len(random_cosines)} random edges")
    print(f"\n  cosine(complement(A,B),  m1_emb[B]):")
    print(f"    Gold edges   — mean: {np.mean(cosines):.4f}  "
          f"std: {np.std(cosines):.4f}  median: {np.median(cosines):.4f}")
    if random_cosines:
        print(f"    Random edges — mean: {np.mean(random_cosines):.4f}  "
              f"std: {np.std(random_cosines):.4f}  median: {np.median(random_cosines):.4f}")

    print()
    for t in [0.95, 0.90, 0.80, 0.70]:
        pct = np.mean(np.array(cosines) > t)
        print(f"    Gold edges with cosine > {t:.2f}: {pct*100:.1f}%")

    mean_cos = np.mean(cosines)
    print("\n  Interpretation:")
    if mean_cos > 0.92:
        print("  HIGH DRIFT: complement ≈ passage embedding")
        print("  L_content dominates — complement loses directional signal toward next hop")
        print("  Scoring with edge_vec ≈ scoring with m1_emb[B] (plain cosine)")
    elif mean_cos > 0.75:
        print("  MODERATE DRIFT: partial directional signal but L_content still dominates")
    else:
        print("  LOW DRIFT: complement is directionally distinct from passage embedding")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  1. Seeds catch any gold passage : {seed_hit_any/total*100:.1f}%")
    print(f"  2. First chain passage in seeds : {seed_first_hop/total*100:.1f}%")
    print(f"  3. Full gold chain in graph     : {full_chain_covered/max(full_chain_total,1)*100:.1f}%")
    print(f"  4. BFS upper-bound recall       : {total_found/max(total_gold,1)*100:.1f}%")
    print(f"  5. Complement-passage cosine    : {np.mean(cosines):.3f}")
    print()

    # Identify the biggest bottleneck
    bottlenecks = []
    if seed_hit_any / total < 0.60:
        bottlenecks.append(f"SEED RETRIEVAL too weak ({seed_hit_any/total*100:.0f}% hit rate)")
    if full_chain_covered / max(full_chain_total, 1) < 0.50:
        bottlenecks.append(f"GRAPH COVERAGE missing gold edges ({full_chain_covered/max(full_chain_total,1)*100:.0f}% full chains)")
    if total_found / max(total_gold, 1) < 0.70:
        bottlenecks.append(f"BFS UPPER BOUND low ({total_found/max(total_gold,1)*100:.0f}%) — ceiling on what any scorer can achieve")
    if np.mean(cosines) > 0.92:
        bottlenecks.append(f"COMPLEMENT DRIFT high ({np.mean(cosines):.3f}) — complement ≈ plain embedding, Model 2 adds no signal")

    if bottlenecks:
        print("  Bottlenecks found:")
        for b in bottlenecks:
            print(f"    • {b}")
    else:
        print("  No single obvious bottleneck — problem may be in beam width or scoring margin")


if __name__ == "__main__":
    main()
