"""
MDR-style iterative dense retrieval baseline (Phase 2 — the decision gate).

This is the critical comparison: does our graph traversal beat MDR?
If MDR ties graph traversal within 3% Recall@10, we stop and use MDR.

MDR logic (simplified, no MDR-specific model):
  1. Embed query → retrieve top-K seeds via FAISS
  2. For each retrieved passage, form extended query:
       extended_query = original_query + " " + passage_text (first 100 words)
  3. Embed extended query → retrieve more passages
  4. Repeat for max_hops iterations
  5. Deduplicate and return union of all retrieved passages

This is the key idea from:
  "Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval"
  Xiong et al. (2021) — MDR achieves 81% Recall@10 on HotpotQA full-wiki

We implement this WITHOUT the MDR-specific trained model
(that would require downloading their checkpoints).
Instead we use e5-small-v2 + iterative query expansion.
This gives us a fair baseline of the IDEA without their training advantage.
"""

from typing import List, Dict, Set

import numpy as np
from sentence_transformers import SentenceTransformer

from baselines import DenseRetriever, BM25Retriever, reciprocal_rank_fusion, QUERY_PREFIX, PASSAGE_PREFIX

MAX_WORDS_FOR_EXPANSION = 80   # how much of retrieved passage to append to query


class MDRBaseline:
    """
    Iterative dense retrieval without a trained MDR model.
    Uses e5-small-v2 for all embeddings.
    """

    def __init__(self, max_hops: int = 3, seeds_per_hop: int = 5):
        self.max_hops = max_hops
        self.seeds_per_hop = seeds_per_hop
        self.dense = DenseRetriever()
        self.id_to_chunk: Dict[str, Dict] = {}

    def build(self, corpus: List[Dict], cache_name: str = None) -> None:
        self.dense.build(corpus, cache_name=f"dense_{cache_name}" if cache_name else None)
        self.id_to_chunk = {c["chunk_id"]: c for c in corpus}

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        """
        Iterative dense retrieval.
        Returns ordered list of chunk_ids (most relevant first).
        """
        retrieved: Dict[str, float] = {}   # chunk_id → best score seen
        query = question

        for hop in range(self.max_hops):
            # Dense retrieval with current (extended) query
            hop_results = self.dense.retrieve_with_scores(query, top_k=self.seeds_per_hop * 2)

            new_chunks = []
            for chunk_id, score in hop_results:
                if chunk_id not in retrieved:
                    retrieved[chunk_id] = score
                    new_chunks.append(chunk_id)
                else:
                    retrieved[chunk_id] = max(retrieved[chunk_id], score)

            if not new_chunks:
                break

            # Extend query with top new passage (MDR key idea)
            best_new = max(new_chunks, key=lambda cid: retrieved[cid])
            passage_text = self.id_to_chunk.get(best_new, {}).get("text", "")
            expansion = " ".join(passage_text.split()[:MAX_WORDS_FOR_EXPANSION])
            query = question + " " + expansion

        # Return top_k by score
        sorted_ids = sorted(retrieved, key=retrieved.__getitem__, reverse=True)
        return sorted_ids[:top_k]


class MDRHybridBaseline:
    """
    MDR variant with BM25 + Dense fusion at each hop.
    Often stronger than pure dense.
    """

    def __init__(self, max_hops: int = 3, seeds_per_hop: int = 5):
        self.max_hops = max_hops
        self.seeds_per_hop = seeds_per_hop
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.id_to_chunk: Dict[str, Dict] = {}

    def build(self, corpus: List[Dict], cache_name: str = None) -> None:
        self.dense.build(corpus, cache_name=f"dense_{cache_name}" if cache_name else None)
        self.bm25.build(corpus, cache_name=f"bm25_{cache_name}" if cache_name else None)
        self.id_to_chunk = {c["chunk_id"]: c for c in corpus}

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        retrieved: Set[str] = set()
        all_ranked: List[str] = []
        query = question

        for hop in range(self.max_hops):
            dense_list = self.dense.retrieve(query, top_k=self.seeds_per_hop * 3)
            bm25_list = self.bm25.retrieve(query, top_k=self.seeds_per_hop * 3)
            fused = reciprocal_rank_fusion([dense_list, bm25_list])

            new_chunks = [cid for cid in fused if cid not in retrieved][:self.seeds_per_hop]
            if not new_chunks:
                break

            for cid in new_chunks:
                if cid not in retrieved:
                    retrieved.add(cid)
                    all_ranked.append(cid)

            # Extend query with best new passage
            best_new = new_chunks[0]
            passage_text = self.id_to_chunk.get(best_new, {}).get("text", "")
            expansion = " ".join(passage_text.split()[:MAX_WORDS_FOR_EXPANSION])
            query = question + " " + expansion

        return all_ranked[:top_k]


class GraphTraversalBaseline:
    """
    Our graph traversal without trained models (Phase 3 gate check).
    Uses hybrid retrieval for seeds + simple cosine-sim for traversal scoring.
    No Model 1 or Model 2 yet — just the graph structure.

    If this beats MDR → worth training models.
    If this ties MDR → use MDR, skip model training.
    """

    def __init__(
        self,
        graph: Dict = None,
        max_hops: int = 3,
        beam_width: int = 3,
        n_seeds: int = 5,
        stop_threshold: float = 0.3,
        alpha: float = 0.6,
    ):
        self.graph = graph or {}
        self.max_hops = max_hops
        self.beam_width = beam_width
        self.n_seeds = n_seeds
        self.stop_threshold = stop_threshold
        self.alpha = alpha

        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.id_to_chunk: Dict[str, Dict] = {}
        self.embeddings: np.ndarray = None
        self.chunk_ids: List[str] = None

    def build(
        self,
        corpus: List[Dict],
        embeddings: np.ndarray,
        graph: Dict,
        cache_name: str = None,
    ) -> None:
        self.dense.build(corpus, cache_name=f"dense_{cache_name}" if cache_name else None)
        self.bm25.build(corpus, cache_name=f"bm25_{cache_name}" if cache_name else None)
        self.id_to_chunk = {c["chunk_id"]: c for c in corpus}
        self.embeddings = embeddings
        self.chunk_ids = [c["chunk_id"] for c in corpus]
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.graph = graph

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        # Hybrid seeds
        dense_list = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_list = self.bm25.retrieve(question, top_k=self.n_seeds * 3)
        seeds = reciprocal_rank_fusion([dense_list, bm25_list])[:self.n_seeds]

        query_emb = self.dense.embed_query(question)

        retrieved: Set[str] = set(seeds)
        beam = list(seeds)
        result_order = list(seeds)

        for hop in range(self.max_hops):
            candidates: Dict[str, float] = {}

            for chunk_id in beam:
                neighbors = self.graph.get(chunk_id, [])
                for (neighbor_id, edge_sim, _) in neighbors:
                    if neighbor_id in retrieved:
                        continue

                    # Direct query-chunk similarity
                    n_idx = self.chunk_id_to_idx.get(neighbor_id)
                    if n_idx is None:
                        continue
                    n_emb = self.embeddings[n_idx]
                    direct_sim = float(np.dot(query_emb, n_emb))  # both L2-normalized

                    # Combined: edge_sim (structural) + direct_sim (semantic)
                    score = self.alpha * edge_sim + (1 - self.alpha) * direct_sim
                    if neighbor_id not in candidates or score > candidates[neighbor_id]:
                        candidates[neighbor_id] = score

            if not candidates or max(candidates.values()) < self.stop_threshold:
                break

            top_neighbors = sorted(candidates, key=candidates.__getitem__, reverse=True)[:self.beam_width]
            beam = top_neighbors
            retrieved.update(top_neighbors)
            result_order.extend(top_neighbors)

        # Return in order: seeds first, then graph-discovered
        return result_order[:top_k]
