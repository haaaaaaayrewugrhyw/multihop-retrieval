"""
FakeEncoder Full System Evaluation — retrieval_v2
==================================================

Runs the same three-way comparison as retrieval/run_full_system.py
but uses FakeEncoderModel instead of ComplementEncoder.

Systems compared:
  1. MDR (dense, iterative)              — baseline, no model needed
  2. Graph + direct cosine (FE)          — FE passage embeddings, no complement
  3. FULL: FE graph + complement + FAISS — complement edge vectors, full system

Key differences from the old pipeline:
  - passage encoding  : model.encode_query(text)  [CLS → 128-dim]
  - edge complement   : model.extract_complement(A, B)  [FakeEncoder → 128-dim]
  - checkpoint file   : fakencoder_best.pt  (not model1_complement.pt)

All offline embeddings are cached to retrieval_v2/cache/ after first run.

Usage:
  python run_eval.py                         # full 2417 dev queries
  python run_eval.py --max_examples 300      # quick 300-query smoke test
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import faiss
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "faiss-cpu"], check=True)
    import faiss

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizerFast

# ── Shared utilities from the sibling retrieval/ directory ───────────────────
RETRIEVAL_DIR = Path(__file__).parent.parent / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_musique
from evaluate import evaluate_retriever, compare_systems          # retrieval/
from baselines import DenseRetriever, BM25Retriever, reciprocal_rank_fusion  # retrieval/
from graph_builder import build_graph                              # retrieval/
from mdr_baseline import MDRBaseline                              # retrieval/
from fakencoder_train import FakeEncoderModel, MAX_A_LEN, MAX_B_LEN, D_PROJ

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_WIDTH   = 3
MAX_HOPS     = 3
N_SEEDS      = 15
STOP_THRESH  = 0.05
FAISS_TOP_K  = 20
ENC_BATCH    = 64    # passages per batch for encode_query
COMP_BATCH   = 8     # (A,B) pairs per batch for extract_complement

MODEL_DIR   = Path(__file__).parent / "models"
CACHE_DIR   = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Offline: passage embeddings ───────────────────────────────────────────────

@torch.no_grad()
def compute_passage_embeddings(
    model:      FakeEncoderModel,
    corpus:     List[Dict],
    tokenizer:  BertTokenizerFast,
    device:     str,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """encode_query(P) for every corpus chunk → [N, 128] float32."""
    if cache_path and cache_path.exists():
        print(f"[pass_emb] Loading cache: {cache_path}")
        return np.load(str(cache_path))

    print(f"[pass_emb] Computing passage embeddings for {len(corpus):,} chunks ...")
    model.eval()
    vecs  = []
    texts = [c["text"] for c in corpus]

    for start in tqdm(range(0, len(texts), ENC_BATCH), desc="Passage embeddings"):
        batch = texts[start : start + ENC_BATCH]
        enc   = tokenizer(
            batch, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        v = model.encode_query(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )   # [B, 128] L2-normalised
        vecs.append(v.cpu().numpy())

    emb = np.vstack(vecs).astype(np.float32)
    print(f"[pass_emb] Done — shape {emb.shape}")
    if cache_path:
        np.save(str(cache_path), emb)
        print(f"[pass_emb] Cached → {cache_path}")
    return emb


# ── Offline: complement edge vectors ─────────────────────────────────────────

@torch.no_grad()
def compute_edge_vectors(
    model:      FakeEncoderModel,
    corpus:     List[Dict],
    graph:      Dict,
    tokenizer:  BertTokenizerFast,
    device:     str,
    cache_path: Optional[Path] = None,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    extract_complement(A, B) for every directed graph edge → 128-dim vector.
    Stored as {(a_id, b_id): np.array([128])} for instant dict lookup at query time.
    """
    if cache_path and cache_path.exists():
        print(f"[edge_vecs] Loading cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    id_to_text = {c["chunk_id"]: c["text"] for c in corpus}

    all_edges: List[Tuple[str, str]] = [
        (a_id, b_id)
        for a_id, neighbors in graph.items()
        for (b_id, _, _) in neighbors
    ]
    print(f"[edge_vecs] Computing complements for {len(all_edges):,} edges ...")
    model.eval()
    edge_vecs: Dict[Tuple[str, str], np.ndarray] = {}

    for start in tqdm(range(0, len(all_edges), COMP_BATCH), desc="Edge complements"):
        batch = all_edges[start : start + COMP_BATCH]
        valid = [(a, b) for a, b in batch if a in id_to_text and b in id_to_text]
        if not valid:
            continue

        a_texts = [id_to_text[a] for a, b in valid]
        b_texts = [id_to_text[b] for a, b in valid]

        enc_a = tokenizer(
            a_texts, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_b = tokenizer(
            b_texts, max_length=MAX_B_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        comps = model.extract_complement(
            enc_a["input_ids"].to(device),
            enc_a["attention_mask"].to(device),
            enc_b["input_ids"].to(device),
        )   # [B, 128] L2-normalised

        for i, (a_id, b_id) in enumerate(valid):
            edge_vecs[(a_id, b_id)] = comps[i].cpu().numpy()

    print(f"[edge_vecs] Done — {len(edge_vecs):,} edge vectors")
    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump(edge_vecs, fh)
        print(f"[edge_vecs] Cached → {cache_path}")
    return edge_vecs


# ── Ablation: graph + direct passage cosine (no complement) ──────────────────

class GraphDirectCosine:
    """dot(q_vec, passage_emb[B]) — complement removed, direct sim only."""

    def __init__(
        self,
        model:        FakeEncoderModel,
        graph:        Dict,
        corpus:       List[Dict],
        pass_emb:     np.ndarray,
        tokenizer:    BertTokenizerFast,
        bm25:         BM25Retriever,
        dense:        DenseRetriever,
        beam_width:   int   = BEAM_WIDTH,
        max_hops:     int   = MAX_HOPS,
        n_seeds:      int   = N_SEEDS,
        stop_thresh:  float = STOP_THRESH,
    ):
        self.model      = model
        self.graph      = graph
        self.pass_emb   = pass_emb
        self.tokenizer  = tokenizer
        self.bm25       = bm25
        self.dense      = dense
        self.beam_width = beam_width
        self.max_hops   = max_hops
        self.n_seeds    = n_seeds
        self.stop_thresh = stop_thresh
        self.id_to_idx  = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    @torch.no_grad()
    def _encode_query(self, question: str) -> np.ndarray:
        enc = self.tokenizer(
            question, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        v = self.model.encode_query(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
        )
        return v.cpu().numpy()[0]   # [128]

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        dense_r = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_r  = self.bm25.retrieve(question,  top_k=self.n_seeds * 3)
        seeds   = reciprocal_rank_fusion([dense_r, bm25_r])[:self.n_seeds]

        q_np          = self._encode_query(question)
        retrieved     = list(seeds)
        retrieved_set = set(seeds)
        beam          = list(seeds)

        for _ in range(self.max_hops):
            candidates: Dict[str, float] = {}
            for cid in beam:
                for (nbr, _, _) in self.graph.get(cid, []):
                    if nbr in retrieved_set:
                        continue
                    idx = self.id_to_idx.get(nbr)
                    if idx is None:
                        continue
                    score = float(np.dot(q_np, self.pass_emb[idx]))
                    if nbr not in candidates or score > candidates[nbr]:
                        candidates[nbr] = score

            if not candidates or max(candidates.values()) < self.stop_thresh:
                break

            top_nbrs = sorted(candidates, key=candidates.__getitem__, reverse=True)[:self.beam_width]
            beam = top_nbrs
            retrieved_set.update(top_nbrs)
            retrieved.extend(top_nbrs)

        return retrieved[:top_k]


# ── Full system: complement-directed beam search ──────────────────────────────

class FullFETraversal:
    """
    Same structure as FullGraphTraversal in retrieval/run_full_system.py
    but uses FakeEncoder complements as edge vectors and encode_query for q_vec.
    """

    def __init__(
        self,
        model:        FakeEncoderModel,
        graph:        Dict,
        corpus:       List[Dict],
        edge_vecs:    Dict[Tuple[str, str], np.ndarray],
        pass_emb:     np.ndarray,
        faiss_index:  faiss.IndexFlatIP,
        tokenizer:    BertTokenizerFast,
        bm25:         BM25Retriever,
        dense:        DenseRetriever,
        beam_width:   int   = BEAM_WIDTH,
        max_hops:     int   = MAX_HOPS,
        n_seeds:      int   = N_SEEDS,
        stop_thresh:  float = STOP_THRESH,
        faiss_top_k:  int   = FAISS_TOP_K,
    ):
        self.model       = model
        self.graph       = graph
        self.edge_vecs   = edge_vecs
        self.pass_emb    = pass_emb
        self.faiss_index = faiss_index
        self.tokenizer   = tokenizer
        self.bm25        = bm25
        self.dense       = dense
        self.beam_width  = beam_width
        self.max_hops    = max_hops
        self.n_seeds     = n_seeds
        self.stop_thresh = stop_thresh
        self.faiss_top_k = faiss_top_k

        self.corpus_ids = [c["chunk_id"] for c in corpus]
        self.id_to_idx  = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    @torch.no_grad()
    def _encode_query(self, question: str) -> np.ndarray:
        enc = self.tokenizer(
            question, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        v = self.model.encode_query(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
        )
        return v.cpu().numpy()[0]

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        dense_r = self.dense.retrieve(question, top_k=self.n_seeds * 3)
        bm25_r  = self.bm25.retrieve(question,  top_k=self.n_seeds * 3)
        seeds   = reciprocal_rank_fusion([dense_r, bm25_r])[:self.n_seeds]

        q_np          = self._encode_query(question)
        retrieved     = list(seeds)
        retrieved_set = set(seeds)
        # beam: (prev_id or None, curr_id)
        beam: List[Tuple[Optional[str], str]] = [(None, s) for s in seeds]

        for _ in range(self.max_hops):
            candidates: Dict[str, Tuple[float, str]] = {}

            for (prev_id, curr_id) in beam:
                if prev_id is None:
                    # Hop 1: walk graph edges, score with edge complement
                    for (nbr, _, _) in self.graph.get(curr_id, []):
                        if nbr in retrieved_set:
                            continue
                        ev = self.edge_vecs.get((curr_id, nbr))
                        if ev is None:
                            continue
                        score = float(np.dot(q_np, ev))
                        if nbr not in candidates or score > candidates[nbr][0]:
                            candidates[nbr] = (score, curr_id)
                else:
                    # Hop 2+: graph neighbors + FAISS with q_np
                    for (nbr, _, _) in self.graph.get(curr_id, []):
                        if nbr in retrieved_set or nbr == curr_id:
                            continue
                        ev = self.edge_vecs.get((curr_id, nbr))
                        if ev is None:
                            continue
                        score = float(np.dot(q_np, ev))
                        if nbr not in candidates or score > candidates[nbr][0]:
                            candidates[nbr] = (score, curr_id)

                    # FAISS fallback with q_np
                    _, idxs = self.faiss_index.search(
                        q_np.reshape(1, -1).astype(np.float32),
                        self.faiss_top_k * 3,
                    )
                    for i in idxs[0]:
                        if not (0 <= i < len(self.corpus_ids)):
                            continue
                        nbr = self.corpus_ids[i]
                        if nbr in retrieved_set or nbr == curr_id:
                            continue
                        ev = self.edge_vecs.get((curr_id, nbr))
                        score = float(np.dot(q_np, ev)) if ev is not None else float(
                            np.dot(q_np, self.pass_emb[i])
                        )
                        if nbr not in candidates or score > candidates[nbr][0]:
                            candidates[nbr] = (score, curr_id)

            if not candidates:
                break
            if max(s for s, _ in candidates.values()) < self.stop_thresh:
                break

            top_nbrs = sorted(candidates, key=lambda k: candidates[k][0], reverse=True)[:self.beam_width]
            beam = [(candidates[nbr][1], nbr) for nbr in top_nbrs]
            retrieved_set.update(top_nbrs)
            retrieved.extend(top_nbrs)

        return retrieved[:top_k]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_retriever(name: str, ret, queries: List[Dict], top_k: int = 10) -> Dict:
    print(f"\n[eval] Running: {name}")
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
    print(f"  R@10={metrics.get('recall@10', 0):.4f} | {elapsed:.1f}s | {metrics['latency_ms']:.1f} ms/q")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap MuSiQue dev examples (None = all 2417)")
    parser.add_argument("--top_k",        type=int, default=10)
    args = parser.parse_args()

    # ── Data ─────────────────────────────────────────────────────────────────
    corpus, queries = load_musique(
        split="validation", max_examples=args.max_examples, cache=True,
    )
    tag = f"fe_val_{args.max_examples}"
    print(f"[eval] {len(corpus):,} chunks | {len(queries):,} queries | {DEVICE}")

    # ── Seed retrievers (generic, unchanged from original pipeline) ───────────
    bm25 = BM25Retriever()
    bm25.build(corpus, cache_name=f"bm25_{tag}")

    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{tag}")

    # ── Load FakeEncoder checkpoint ───────────────────────────────────────────
    ckpt_path = MODEL_DIR / "fakencoder_best.pt"
    if not ckpt_path.exists():
        print(f"[eval] Checkpoint not found: {ckpt_path}")
        print("[eval] Running MDR-only evaluation (no FakeEncoder)")
        has_model = False
    else:
        has_model = True
        model     = FakeEncoderModel().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        print(f"[eval] FakeEncoderModel loaded from {ckpt_path}")

    # ── Offline: passage embeddings + graph + edge vectors ───────────────────
    if has_model:
        pass_emb = compute_passage_embeddings(
            model, corpus, tokenizer, DEVICE,
            cache_path=CACHE_DIR / f"fe_pass_emb_{tag}.npy",
        )
        graph = build_graph(corpus, embeddings=pass_emb, cache_name=f"{tag}_fe")

        faiss_index = faiss.IndexFlatIP(pass_emb.shape[1])
        faiss_index.add(pass_emb)
        print(f"[eval] FAISS: {faiss_index.ntotal:,} vectors @ {pass_emb.shape[1]}-dim")

        edge_vecs = compute_edge_vectors(
            model, corpus, graph, tokenizer, DEVICE,
            cache_path=CACHE_DIR / f"fe_edge_vecs_{tag}.pkl",
        )
    else:
        n, dim  = dense.index.ntotal, dense.index.d
        emb_tmp = np.zeros((n, dim), dtype=np.float32)
        dense.index.reconstruct_n(0, n, emb_tmp)
        graph   = build_graph(corpus, embeddings=emb_tmp, cache_name=tag)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_systems: Dict[str, Dict] = {}

    mdr               = MDRBaseline(max_hops=3, seeds_per_hop=5)
    mdr.dense         = dense
    mdr.id_to_chunk   = {c["chunk_id"]: c for c in corpus}
    all_systems["MDR (dense, iterative)"] = run_retriever("MDR", mdr, queries, args.top_k)

    if has_model:
        g_cos = GraphDirectCosine(
            model, graph, corpus, pass_emb, tokenizer, bm25, dense,
        )
        all_systems["Graph + direct cosine (FE)"] = run_retriever(
            "Graph+FEcos", g_cos, queries, args.top_k,
        )

        full = FullFETraversal(
            model, graph, corpus, edge_vecs, pass_emb,
            faiss_index, tokenizer, bm25, dense,
        )
        all_systems["FULL: FE graph + complement + FAISS"] = run_retriever(
            "Full", full, queries, args.top_k,
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    compare_systems(all_systems)

    if has_model and "FULL: FE graph + complement + FAISS" in all_systems:
        mdr_r10  = all_systems["MDR (dense, iterative)"]["recall@10"]
        cos_r10  = all_systems["Graph + direct cosine (FE)"]["recall@10"]
        full_r10 = all_systems["FULL: FE graph + complement + FAISS"]["recall@10"]

        print("\n[DECISION GATE]")
        print(f"  MDR          R@10 : {mdr_r10:.4f}")
        print(f"  Graph+cos    R@10 : {cos_r10:.4f}")
        print(f"  FULL         R@10 : {full_r10:.4f}")
        print(f"  FULL vs cos  gap  : {full_r10 - cos_r10:+.4f}")

        if full_r10 > cos_r10 + 0.01:
            print("  COMPLEMENT HELPS — FakeEncoder is NOT collapsed")
        elif full_r10 >= cos_r10:
            print("  COMPLEMENT MARGINAL — FakeEncoder adds little over direct cosine")
        else:
            print("  COMPLEMENT HURTS — FakeEncoder may be collapsed (complement ≈ encode_B)")

    out_path = RESULTS_DIR / f"eval_fe_{args.max_examples}.json"
    with open(out_path, "w") as fh:
        json.dump({"max_examples": args.max_examples, "systems": all_systems}, fh, indent=2)
    print(f"\n[eval] Results saved → {out_path}")


if __name__ == "__main__":
    main()
