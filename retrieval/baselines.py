"""
Retrieval baselines:
  1. BM25Retriever       — sparse
  2. DenseRetriever      — FAISS + e5-small-v2
  3. HybridRetriever     — BM25 + FAISS fused with RRF
  4. HybridReranker      — HybridRetriever + cross-encoder reranker

All retrievers share the same interface:
    retriever.build(corpus)  → builds index
    retriever.retrieve(question, top_k) → List[chunk_id]
"""

import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "intfloat/e5-small-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


# ─────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.corpus = None

    def build(self, corpus: List[Dict], cache_name: str = None) -> None:
        cache_file = CACHE_DIR / f"bm25_{cache_name}.pkl" if cache_name else None
        if cache_file and cache_file.exists():
            print(f"[BM25] Loading from cache {cache_file}")
            with open(cache_file, "rb") as f:
                self.bm25, self.corpus = pickle.load(f)
            return

        print(f"[BM25] Building index over {len(corpus):,} chunks...")
        self.corpus = corpus
        tokenized = [c["text"].lower().split() for c in corpus]
        self.bm25 = BM25Okapi(tokenized)

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump((self.bm25, self.corpus), f)

    def retrieve(self, question: str, top_k: int = 20) -> List[str]:
        tokens = question.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i]["chunk_id"] for i in top_indices]

    def retrieve_with_scores(self, question: str, top_k: int = 20) -> List[tuple]:
        tokens = question.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus[i]["chunk_id"], float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────
# Dense (FAISS + e5-small-v2)
# ─────────────────────────────────────────────

class DenseRetriever:
    def __init__(self, embed_model: str = EMBED_MODEL, batch_size: int = 256):
        self.embed_model_name = embed_model
        self.batch_size = batch_size
        self.model = None
        self.index = None
        self.corpus = None

    def _get_model(self):
        if self.model is None:
            print(f"[Dense] Loading embedding model: {self.embed_model_name}")
            self.model = SentenceTransformer(self.embed_model_name)
        return self.model

    def build(self, corpus: List[Dict], cache_name: str = None) -> None:
        cache_file = CACHE_DIR / f"dense_{cache_name}.pkl" if cache_name else None
        if cache_file and cache_file.exists():
            print(f"[Dense] Loading from cache {cache_file}")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.corpus = data["corpus"]
            embeddings = data["embeddings"]
            self._build_faiss(embeddings)
            return

        print(f"[Dense] Embedding {len(corpus):,} chunks with {self.embed_model_name}...")
        self.corpus = corpus
        model = self._get_model()

        texts = [PASSAGE_PREFIX + c["text"] for c in corpus]
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump({"corpus": corpus, "embeddings": embeddings}, f)

        self._build_faiss(embeddings)

    def _build_faiss(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"[Dense] FAISS index ready: {self.index.ntotal:,} vectors, dim={dim}")

    def embed_query(self, question: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(
            [QUERY_PREFIX + question],
            normalize_embeddings=True,
        )[0]

    def retrieve(self, question: str, top_k: int = 20) -> List[str]:
        q_emb = self.embed_query(question).reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(q_emb, top_k)
        return [self.corpus[i]["chunk_id"] for i in indices[0] if i >= 0]

    def retrieve_with_scores(self, question: str, top_k: int = 20) -> List[tuple]:
        q_emb = self.embed_query(question).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        return [
            (self.corpus[i]["chunk_id"], float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]


# ─────────────────────────────────────────────
# RRF fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[str]:
    """
    Combine multiple ranked lists via RRF.
    k=60 is the standard constant from the original paper.
    Returns merged list of doc_ids ordered by fused score.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


# ─────────────────────────────────────────────
# Hybrid = BM25 + FAISS + RRF
# ─────────────────────────────────────────────

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

    def build(self, corpus: List[Dict], cache_name: str = None) -> None:
        self.bm25.build(corpus, cache_name=f"bm25_{cache_name}" if cache_name else None)
        self.dense.build(corpus, cache_name=f"dense_{cache_name}" if cache_name else None)

    def retrieve(self, question: str, top_k: int = 20, candidate_k: int = 20) -> List[str]:
        bm25_list = self.bm25.retrieve(question, top_k=candidate_k)
        dense_list = self.dense.retrieve(question, top_k=candidate_k)
        fused = reciprocal_rank_fusion([bm25_list, dense_list])
        return fused[:top_k]


# ─────────────────────────────────────────────
# Hybrid + Cross-Encoder Reranker
# ─────────────────────────────────────────────

class HybridReranker:
    def __init__(self, reranker_model: str = RERANKER_MODEL):
        self.hybrid = HybridRetriever()
        self.reranker_model_name = reranker_model
        self.reranker = None
        self.id_to_chunk: Dict[str, Dict] = {}

    def _get_reranker(self):
        if self.reranker is None:
            print(f"[Reranker] Loading cross-encoder: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name)
        return self.reranker

    def build(self, corpus: List[Dict], cache_name: str = None) -> None:
        self.hybrid.build(corpus, cache_name=cache_name)
        self.id_to_chunk = {c["chunk_id"]: c for c in corpus}

    def retrieve(self, question: str, top_k: int = 10, candidate_k: int = 50) -> List[str]:
        candidates = self.hybrid.retrieve(question, top_k=candidate_k)
        reranker = self._get_reranker()

        pairs = [(question, self.id_to_chunk[cid]["text"]) for cid in candidates if cid in self.id_to_chunk]
        scores = reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(candidates[:len(pairs)], scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]
