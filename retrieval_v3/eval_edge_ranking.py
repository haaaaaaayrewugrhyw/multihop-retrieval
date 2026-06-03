"""
eval_edge_ranking.py -- clean edge-ranking eval (retrieval_v3)
==============================================================

Measures the complement on its NATIVE task, with no graph / seeds / FAISS to
dilute it:

  Given (Q, A, B_pos, [B_neg x k]), rank the candidate next-hop docs B.
  Compare two scorers at putting B_pos first:

    complement (the idea) : dot( M2(Q), G(A, B) )          -- conditions on A
    cosine baseline       : dot( encode(Q), encode(B) )    -- no A conditioning
    (optional) QA-cosine  : dot( encode("Q [SEP] A"), encode(B) )  -- A-aware baseline

  Reports ranking accuracy (B_pos is #1) and MRR over the full dev set.

This isolates the contribution: does a label-free complement that conditions on A
beat plain query-passage similarity at choosing the correct next hop?

Usage:
    python eval_edge_ranking.py                 # full dev set
    python eval_edge_ranking.py --max_examples 300
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizerFast

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))   # data_loader
sys.path.insert(0, str(_HERE))                            # generator_train, model2_train

from data_loader import build_scoring_quintuples, load_musique
from generator_train import ComplementGenerator, MAX_A_LEN, MAX_B_LEN
try:
    from model2_train import QueryEncoder
except ImportError:
    QueryEncoder = None

MAX_Q_LEN = 64
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = _HERE / "models"


def _tok(tokenizer, texts, max_len):
    return tokenizer(texts, max_length=max_len, truncation=True,
                     padding="max_length", return_tensors="pt")


@torch.no_grad()
def main(max_examples=None, batch_size=16, gen_ckpt="generator_best.pt"):
    print(f"Device: {DEVICE}")

    # ── Data ─────────────────────────────────────────────────────────────────
    val_corpus, val_queries = load_musique(split="validation",
                                           max_examples=max_examples, cache=True)
    quints = build_scoring_quintuples(val_corpus, val_queries)
    id_to_text = {c["chunk_id"]: c["text"] for c in val_corpus}
    quints = [q for q in quints
              if q["chunk_a_id"] in id_to_text
              and q["chunk_b_pos_id"] in id_to_text
              and all(n in id_to_text for n in q["chunk_b_neg_ids"])]
    print(f"Scoring quintuples: {len(quints):,}")

    # ── Models ───────────────────────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    gen = ComplementGenerator().to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_DIR / gen_ckpt, map_location=DEVICE), strict=False)
    gen.eval()

    m2 = None
    m2_path = MODEL_DIR / "model2_best.pt"
    if QueryEncoder is not None and m2_path.exists():
        m2 = QueryEncoder().to(DEVICE)
        m2.load_state_dict(torch.load(m2_path, map_location=DEVICE))
        m2.eval()
        print("Model 2 (QueryEncoder) loaded")
    else:
        print("Model 2 not found -- complement query uses generator.encode_query")

    # ── Metrics accumulators ─────────────────────────────────────────────────
    # complement: dot(M2(Q), G(A,B))   |  cosine: dot(enc(Q), enc(B))   |  qa: dot(enc(Q[SEP]A), enc(B))
    stats = {"complement": [0, 0.0], "cosine": [0, 0.0], "qa_cosine": [0, 0.0]}  # [hits@1, mrr_sum]

    def rank_metrics(score_pos: float, score_negs: List[float]):
        # rank of pos among [pos, *negs] (1 = best)
        rank = 1 + sum(1 for s in score_negs if s >= score_pos)
        return (1 if rank == 1 else 0), 1.0 / rank

    for i in tqdm(range(0, len(quints), batch_size), desc="edge-rank"):
        chunk = quints[i:i + batch_size]
        Q   = [q["question"] for q in chunk]
        A   = [id_to_text[q["chunk_a_id"]] for q in chunk]
        Bp  = [id_to_text[q["chunk_b_pos_id"]] for q in chunk]
        n_neg = len(chunk[0]["chunk_b_neg_ids"])
        Bn  = [[id_to_text[q["chunk_b_neg_ids"][k]] for q in chunk] for k in range(n_neg)]
        QA  = [f"{q} [SEP] {a}" for q, a in zip(Q, A)]

        eq = _tok(tokenizer, Q,  MAX_Q_LEN)
        ea = _tok(tokenizer, A,  MAX_A_LEN)
        eqa = _tok(tokenizer, QA, MAX_A_LEN)
        ebp = _tok(tokenizer, Bp, MAX_B_LEN)

        ids_a, msk_a = ea["input_ids"].to(DEVICE), ea["attention_mask"].to(DEVICE)

        # query vectors
        q_comp = (m2.encode_query(eq["input_ids"].to(DEVICE), eq["attention_mask"].to(DEVICE))
                  if m2 is not None else
                  gen.encode_query(eq["input_ids"].to(DEVICE), eq["attention_mask"].to(DEVICE)))
        q_enc  = gen.encode_query(eq["input_ids"].to(DEVICE), eq["attention_mask"].to(DEVICE))
        q_qa   = gen.encode_query(eqa["input_ids"].to(DEVICE), eqa["attention_mask"].to(DEVICE))

        # positive candidate
        comp_pos = gen.extract_complement(ids_a, msk_a, ebp["input_ids"].to(DEVICE))
        benc_pos = gen.encode_query(ebp["input_ids"].to(DEVICE), ebp["attention_mask"].to(DEVICE))

        comp_negs, benc_negs = [], []
        for k in range(n_neg):
            ebn = _tok(tokenizer, Bn[k], MAX_B_LEN)
            comp_negs.append(gen.extract_complement(ids_a, msk_a, ebn["input_ids"].to(DEVICE)))
            benc_negs.append(gen.encode_query(ebn["input_ids"].to(DEVICE), ebn["attention_mask"].to(DEVICE)))

        # per-example scoring
        for b in range(len(chunk)):
            # complement
            sp = float((q_comp[b] * comp_pos[b]).sum())
            sn = [float((q_comp[b] * comp_negs[k][b]).sum()) for k in range(n_neg)]
            h, rr = rank_metrics(sp, sn); stats["complement"][0] += h; stats["complement"][1] += rr
            # cosine (no A)
            sp = float((q_enc[b] * benc_pos[b]).sum())
            sn = [float((q_enc[b] * benc_negs[k][b]).sum()) for k in range(n_neg)]
            h, rr = rank_metrics(sp, sn); stats["cosine"][0] += h; stats["cosine"][1] += rr
            # QA-cosine (A-aware baseline, no complement)
            sp = float((q_qa[b] * benc_pos[b]).sum())
            sn = [float((q_qa[b] * benc_negs[k][b]).sum()) for k in range(n_neg)]
            h, rr = rank_metrics(sp, sn); stats["qa_cosine"][0] += h; stats["qa_cosine"][1] += rr

    n = len(quints)
    W = 60
    print("\n" + "=" * W)
    print(f"  EDGE-RANKING EVAL  (n={n:,}, {n_neg} hard negatives each)")
    print("=" * W)
    print(f"  {'scorer':<14}{'acc@1':>10}{'MRR':>10}")
    print("  " + "-" * (W - 4))
    for name in ["cosine", "qa_cosine", "complement"]:
        hits, mrr = stats[name]
        print(f"  {name:<14}{hits/n:>10.4f}{mrr/n:>10.4f}")
    print("=" * W)
    comp_acc = stats["complement"][0] / n
    cos_acc  = stats["cosine"][0] / n
    qa_acc   = stats["qa_cosine"][0] / n
    print(f"  complement vs cosine     : {comp_acc - cos_acc:+.4f}")
    print(f"  complement vs QA-cosine  : {comp_acc - qa_acc:+.4f}")
    print("=" * W)
    print("  Interpretation:")
    print("   - complement >> cosine  : conditioning on A via complement helps pick next hop")
    print("   - complement vs QA-cos  : does the LEARNED complement beat naive [Q;A] concat?")
    print("=" * W)

    return {k: {"acc": v[0] / n, "mrr": v[1] / n} for k, v in stats.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--gen_ckpt",     type=str, default="generator_best.pt")
    a = p.parse_args()
    main(max_examples=a.max_examples, batch_size=a.batch_size, gen_ckpt=a.gen_ckpt)
