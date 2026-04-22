"""
Dataset loaders for retrieval experiments.

Primary dataset: MuSiQue-Ans (JSONL from GitHub release)
  - Shortcut-free 2–4 hop compositional QA
  - Has explicit chain order via question_decomposition[i]['paragraph_support_idx']
  - 17.5K train / 2.4K dev questions

MuSiQue download:
  1. Visit https://github.com/StonyBrookNLP/musique/releases/tag/v1.0
  2. Download data.zip → extract to retrieval/data/musique/
  Expected files:
    retrieval/data/musique/musique_ans_v1.0_dev.jsonl
    retrieval/data/musique/musique_ans_v1.0_train.jsonl

Chunk format:
    {"chunk_id", "text", "title", "example_id", "para_idx", "sub_idx"}

Query format:
    {"query_id", "question", "relevant_chunk_ids", "chain_chunk_ids", "hop_count"}
    relevant_chunk_ids : unordered set of supporting chunk IDs  (used for eval)
    chain_chunk_ids    : ordered list following question_decomposition  (used for training)

Training record formats:
    Chain quadruple  (Model 1):
        {"chunk_a_id", "chunk_b_pos_id", "chunk_b_neg_ids", "chunk_c_id"}
        chunk_c_id is None for the last hop in a chain.

    Scoring quintuple  (Model 2):
        {"question", "chunk_a_id", "chunk_b_pos_id", "chunk_b_neg_ids"}
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR  = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CACHE_VERSION = "v2"          # bump when query/corpus format changes
CHUNK_WORDS   = 300
CHUNK_OVERLAP = 50
NUM_HARD_NEGS = 3             # in-example distractors per positive pair

MUSIQUE_ANS_DEV   = DATA_DIR / "musique" / "musique_ans_v1.0_dev.jsonl"
MUSIQUE_ANS_TRAIN = DATA_DIR / "musique" / "musique_ans_v1.0_train.jsonl"


# ── Text utilities ─────────────────────────────────────────────────────────────

def _word_chunk(text: str, max_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks


# ── MuSiQue loader ─────────────────────────────────────────────────────────────

def load_musique(
    split: str = "validation",
    max_examples: Optional[int] = None,
    cache: bool = True,
    shuffle: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load MuSiQue from local JSONL files.

    Returns (corpus, queries).

    Each query now contains:
      relevant_chunk_ids : unordered supporting chunk IDs  → used for Recall@K eval
      chain_chunk_ids    : ordered by question_decomposition → used for training
    """
    jsonl_file = MUSIQUE_ANS_DEV if split in ("validation", "dev") else MUSIQUE_ANS_TRAIN

    if not jsonl_file.exists():
        raise FileNotFoundError(
            f"\nMuSiQue JSONL not found at {jsonl_file}\n"
            "Download:\n"
            "  1. https://github.com/StonyBrookNLP/musique/releases/tag/v1.0\n"
            "  2. Extract data.zip to retrieval/data/musique/"
        )

    shuffle_tag  = "_shuf" if shuffle else ""
    cache_file   = CACHE_DIR / f"musique_{CACHE_VERSION}_{split}_{max_examples}{shuffle_tag}.pkl"
    if cache and cache_file.exists():
        print(f"[data_loader] Loading cached MuSiQue from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[data_loader] Loading MuSiQue ({split}) from {jsonl_file} ...")
    corpus: List[Dict] = []
    queries: List[Dict] = []
    seen_chunk_ids: set = set()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if shuffle:
        import random
        random.seed(42)
        random.shuffle(lines)

    if max_examples:
        lines = lines[:max_examples]

    for line in tqdm(lines, desc="Processing MuSiQue"):
        ex = json.loads(line.strip())
        ex_id    = ex["id"]
        question = ex["question"]

        supporting_idxs = {
            p["idx"] for p in ex.get("paragraphs", [])
            if p.get("is_supporting", False)
        }

        # Ordered hop chain from question_decomposition
        # paragraph_support_idx maps directly to para["idx"]
        chain_para_idxs: List[int] = [
            d["paragraph_support_idx"]
            for d in ex.get("question_decomposition", [])
        ]

        relevant_chunk_ids: List[str] = []
        para_idx_to_first_chunk: Dict[int, str] = {}

        for para in ex.get("paragraphs", []):
            para_idx = para["idx"]
            title    = para.get("title", "")
            body     = para.get("paragraph_text", "")
            text     = f"{title}. {body}" if title else body

            for sub_idx, chunk_text in enumerate(_word_chunk(text)):
                chunk_id = f"msq_{ex_id}_{para_idx}_{sub_idx}"
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    corpus.append({
                        "chunk_id":   chunk_id,
                        "text":       chunk_text,
                        "title":      title,
                        "example_id": ex_id,
                        "para_idx":   para_idx,
                        "sub_idx":    sub_idx,
                    })
                if sub_idx == 0:
                    para_idx_to_first_chunk[para_idx] = chunk_id
                    if para_idx in supporting_idxs:
                        relevant_chunk_ids.append(chunk_id)

        # Ordered chain: use decomposition order, not paragraph list order
        chain_chunk_ids: List[str] = [
            para_idx_to_first_chunk[idx]
            for idx in chain_para_idxs
            if idx in para_idx_to_first_chunk
        ]

        queries.append({
            "query_id":          ex_id,
            "question":          question,
            "relevant_chunk_ids": relevant_chunk_ids,   # unordered, for eval
            "chain_chunk_ids":   chain_chunk_ids,        # ordered, for training
            "hop_count":         len(supporting_idxs),
        })

    print(f"[data_loader] MuSiQue: {len(corpus):,} chunks | {len(queries):,} queries")

    if cache:
        with open(cache_file, "wb") as f:
            pickle.dump((corpus, queries), f)
    return corpus, queries


# ── Training record builders ───────────────────────────────────────────────────

def build_chain_quadruples(
    corpus: List[Dict],
    queries: List[Dict],
    num_negs: int = NUM_HARD_NEGS,
) -> List[Dict]:
    """
    Build directed consecutive-hop quadruples for Model 1 training.

    Each quadruple:
        chunk_a_id      : paragraph A  (hop k)
        chunk_b_pos_id  : paragraph B  (hop k+1, correct next hop)
        chunk_b_neg_ids : list of distractor chunk IDs (in-example hard negatives)
        chunk_c_id      : paragraph C  (hop k+2) used as next-anchor; None if last hop

    Only consecutive pairs from chain_chunk_ids are used — skip pairs (A→C) are excluded
    because they span multiple hops and would confuse the complement encoder.

    Hard negatives: in-example distractors (non-supporting paragraphs from the same
    MuSiQue question). These are already challenging because they were curated to be
    topically related to the question.
    """
    id_to_chunk: Dict[str, Dict] = {c["chunk_id"]: c for c in corpus}

    # Group all chunk IDs by example_id for fast distractor lookup
    example_to_chunks: Dict[str, List[str]] = defaultdict(list)
    for c in corpus:
        example_to_chunks[c["example_id"]].append(c["chunk_id"])

    quadruples: List[Dict] = []

    for q in queries:
        chain = q.get("chain_chunk_ids", [])
        if len(chain) < 2:
            continue

        relevant_set = set(chain)
        # Use example_id from the first chain chunk
        example_id   = id_to_chunk[chain[0]]["example_id"]
        distractors  = [
            cid for cid in example_to_chunks[example_id]
            if cid not in relevant_set
        ][:num_negs]

        if len(distractors) == 0:
            continue

        # Consecutive pairs only: (A→B), (B→C), (C→D)
        for i in range(len(chain) - 1):
            chunk_a     = chain[i]
            chunk_b_pos = chain[i + 1]
            chunk_c     = chain[i + 2] if i + 2 < len(chain) else None

            quadruples.append({
                "chunk_a_id":     chunk_a,
                "chunk_b_pos_id": chunk_b_pos,
                "chunk_b_neg_ids": distractors,
                "chunk_c_id":     chunk_c,
            })

    pos = len(quadruples)
    print(f"[data_loader] Chain quadruples: {pos:,} directed hop pairs")
    return quadruples


def build_scoring_quintuples(
    corpus: List[Dict],
    queries: List[Dict],
    num_negs: int = NUM_HARD_NEGS,
) -> List[Dict]:
    """
    Build (Q, A, B_pos, B_neg_1..k) quintuples for Model 2 training.

    Each quintuple:
        question        : the full multi-hop question
        chunk_a_id      : paragraph A  (hop k, already retrieved)
        chunk_b_pos_id  : paragraph B  (correct next hop)
        chunk_b_neg_ids : list of distractor chunk IDs

    Same directed consecutive pairs as build_chain_quadruples but with Q attached.
    Hard negatives: same in-example distractors.
    """
    id_to_chunk: Dict[str, Dict] = {c["chunk_id"]: c for c in corpus}

    example_to_chunks: Dict[str, List[str]] = defaultdict(list)
    for c in corpus:
        example_to_chunks[c["example_id"]].append(c["chunk_id"])

    quintuples: List[Dict] = []

    for q in queries:
        chain = q.get("chain_chunk_ids", [])
        if len(chain) < 2:
            continue

        relevant_set = set(chain)
        example_id   = id_to_chunk[chain[0]]["example_id"]
        distractors  = [
            cid for cid in example_to_chunks[example_id]
            if cid not in relevant_set
        ][:num_negs]

        if len(distractors) == 0:
            continue

        for i in range(len(chain) - 1):
            quintuples.append({
                "question":        q["question"],
                "chunk_a_id":      chain[i],
                "chunk_b_pos_id":  chain[i + 1],
                "chunk_b_neg_ids": distractors,
            })

    print(f"[data_loader] Scoring quintuples: {len(quintuples):,} directed hop pairs")
    return quintuples


def build_training_triples(corpus: List[Dict], queries: List[Dict]) -> List[Dict]:
    """
    Legacy function — kept for backward compatibility with eval scripts.

    Builds (question, chunk_a_id, chunk_b_id, label) pairs using all combinations
    of supporting chunks (unordered). New code should use build_chain_quadruples()
    or build_scoring_quintuples() instead.
    """
    id_to_chunk = {c["chunk_id"]: c for c in corpus}
    triples = []

    for q in queries:
        relevant = q["relevant_chunk_ids"]
        if len(relevant) < 2:
            continue

        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                triples.append({
                    "question":   q["question"],
                    "chunk_a_id": relevant[i],
                    "chunk_b_id": relevant[j],
                    "label":      1,
                })

        example_id   = id_to_chunk[relevant[0]]["example_id"]
        all_ex       = [c["chunk_id"] for c in corpus if c["example_id"] == example_id]
        distractors  = [cid for cid in all_ex if cid not in set(relevant)][:3]

        for cid_a in relevant:
            for cid_d in distractors:
                triples.append({
                    "question":   q["question"],
                    "chunk_a_id": cid_a,
                    "chunk_b_id": cid_d,
                    "label":      0,
                })

    return triples


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    corpus, queries = load_musique(split="validation", max_examples=200, cache=False)

    q0 = queries[0]
    print(f"\nSample question : {q0['question']}")
    print(f"  hop_count     : {q0['hop_count']}")
    print(f"  relevant_ids  : {q0['relevant_chunk_ids']}")
    print(f"  chain_ids     : {q0['chain_chunk_ids']}")

    chain_match = set(q0["chain_chunk_ids"]) == set(q0["relevant_chunk_ids"])
    print(f"  chain == relevant (set): {chain_match}")

    quads = build_chain_quadruples(corpus, queries)
    print(f"\nSample quadruple:")
    print(f"  A      : {quads[0]['chunk_a_id']}")
    print(f"  B_pos  : {quads[0]['chunk_b_pos_id']}")
    print(f"  B_negs : {quads[0]['chunk_b_neg_ids']}")
    print(f"  C      : {quads[0]['chunk_c_id']}")

    quints = build_scoring_quintuples(corpus, queries)
    print(f"\nSample quintuple:")
    print(f"  Q      : {quints[0]['question'][:80]}...")
    print(f"  A      : {quints[0]['chunk_a_id']}")
    print(f"  B_pos  : {quints[0]['chunk_b_pos_id']}")
    print(f"  B_negs : {quints[0]['chunk_b_neg_ids']}")
