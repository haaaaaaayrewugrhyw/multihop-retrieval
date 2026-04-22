"""
Chunk Graph Builder (Phase 3).

Builds two types of edges offline:
  - Sequential: chunk_i → chunk_{i+1} within same document (always)
  - Semantic:   top-10 cosine-similar neighbors with sim > 0.70,
                skipping adjacent chunks (already have sequential)

Storage:
    graph[chunk_id] = [(neighbor_id, cosine_sim, edge_type), ...]

edge_type: "sequential" | "semantic"
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SEM_THRESHOLD = 0.70
SEM_TOP_K = 10      # neighbors to check before threshold filtering


def build_graph(
    corpus: List[Dict],
    dense_retriever=None,    # pre-built DenseRetriever with embeddings
    embeddings: np.ndarray = None,
    sem_threshold: float = SEM_THRESHOLD,
    sem_top_k: int = SEM_TOP_K,
    cache_name: str = None,
) -> Dict[str, List[Tuple]]:
    """
    Build graph from corpus.

    Either pass a pre-built dense_retriever (reuse embeddings) or raw embeddings array.
    corpus must be in the SAME ORDER as rows in embeddings.

    Returns:
        graph: {chunk_id: [(neighbor_id, sim, edge_type), ...]}
    """
    cache_file = CACHE_DIR / f"graph_{cache_name}.pkl" if cache_name else None
    if cache_file and cache_file.exists():
        print(f"[graph_builder] Loading from cache {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    graph: Dict[str, List[Tuple]] = defaultdict(list)

    # ── 1. Sequential edges ──────────────────────────────────────────────────
    print("[graph_builder] Building sequential edges...")
    # Group chunks by (example_id, para_idx) to find within-para sub-chunk chains
    # Also group by example_id + para order for sequential para-to-para

    # Build ordered index: example → list of (para_idx, sub_idx, chunk_id)
    from collections import defaultdict as dd
    ex_chunks = dd(list)
    for chunk in corpus:
        ex_chunks[chunk["example_id"]].append(chunk)

    seq_count = 0
    for ex_id, chunks in ex_chunks.items():
        # Sort by para_idx then sub_idx to get document order
        ordered = sorted(chunks, key=lambda c: (c["para_idx"], c.get("sub_idx", 0)))
        for i in range(len(ordered) - 1):
            a = ordered[i]["chunk_id"]
            b = ordered[i + 1]["chunk_id"]
            graph[a].append((b, 1.0, "sequential"))
            graph[b].append((a, 1.0, "sequential"))  # bidirectional
            seq_count += 1

    print(f"[graph_builder]   {seq_count:,} sequential edges")

    # ── 2. Semantic edges ────────────────────────────────────────────────────
    if embeddings is None and dense_retriever is not None:
        # Pull embeddings from FAISS index (reconstruct)
        n = dense_retriever.index.ntotal
        dim = dense_retriever.index.d
        embeddings = np.zeros((n, dim), dtype=np.float32)
        dense_retriever.index.reconstruct_n(0, n, embeddings)
        print(f"[graph_builder] Recovered {n:,} embeddings from FAISS index")

    if embeddings is not None:
        print(f"[graph_builder] Building semantic edges (threshold={sem_threshold})...")
        chunk_ids = [c["chunk_id"] for c in corpus]
        id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

        # Build a fresh flat index for neighbor search
        dim = embeddings.shape[1]
        sem_index = faiss.IndexFlatIP(dim)
        sem_index.add(embeddings.astype(np.float32))

        # Build sequential neighbor set for quick adjacency check
        seq_neighbors: Dict[str, set] = defaultdict(set)
        for cid, neighbors in graph.items():
            for (ncid, _, etype) in neighbors:
                if etype == "sequential":
                    seq_neighbors[cid].add(ncid)

        sem_count = 0
        batch_size = 512
        for start in tqdm(range(0, len(corpus), batch_size), desc="Semantic edges"):
            end = min(start + batch_size, len(corpus))
            batch_embs = embeddings[start:end].astype(np.float32)
            # Fetch sem_top_k + 1 (the +1 is the chunk itself at rank 0)
            sims, indices = sem_index.search(batch_embs, sem_top_k + 1)

            for local_i, (sim_row, idx_row) in enumerate(zip(sims, indices)):
                global_i = start + local_i
                src_id = chunk_ids[global_i]

                for sim, neighbor_idx in zip(sim_row, idx_row):
                    if neighbor_idx < 0 or neighbor_idx == global_i:
                        continue  # skip self
                    if sim < sem_threshold:
                        break    # sorted descending, no point continuing

                    neighbor_id = chunk_ids[neighbor_idx]
                    if neighbor_id in seq_neighbors[src_id]:
                        continue  # already a sequential edge

                    graph[src_id].append((neighbor_id, float(sim), "semantic"))
                    sem_count += 1

        print(f"[graph_builder]   {sem_count:,} semantic edges")
    else:
        print("[graph_builder] No embeddings provided — semantic edges skipped")

    graph = dict(graph)
    print(f"[graph_builder] Total graph: {len(graph):,} nodes, "
          f"{sum(len(v) for v in graph.values()):,} directed edges")

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(graph, f)
        print(f"[graph_builder] Cached to {cache_file}")

    return graph


def graph_stats(graph: Dict[str, List[Tuple]]) -> None:
    """Print diagnostic stats about the graph."""
    degrees = [len(v) for v in graph.values()]
    seq_edges = sum(1 for nbrs in graph.values() for (_, _, t) in nbrs if t == "sequential")
    sem_edges = sum(1 for nbrs in graph.values() for (_, _, t) in nbrs if t == "semantic")

    print(f"\n[graph_stats] Nodes:          {len(graph):>10,}")
    print(f"[graph_stats] Sequential edges: {seq_edges:>10,}")
    print(f"[graph_stats] Semantic edges:   {sem_edges:>10,}")
    print(f"[graph_stats] Avg degree:       {sum(degrees)/len(degrees):>10.2f}")
    print(f"[graph_stats] Max degree:       {max(degrees):>10,}")
    print(f"[graph_stats] Min degree:       {min(degrees):>10,}")
    isolated = sum(1 for d in degrees if d == 0)
    print(f"[graph_stats] Isolated nodes:   {isolated:>10,}")
