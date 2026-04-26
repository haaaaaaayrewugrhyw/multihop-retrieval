"""
diagnose_models.py — Component-level diagnostics for the multi-hop retrieval pipeline.

Pinpoints WHERE the pipeline loses points across 5 targeted tests:

  1. M1 Complement Discriminability  — does complement(A,B_pos) point toward C?
  2. Edge Coverage                    — what fraction of true hops have a graph edge?
  3. M2 Ranking Position              — where does B_pos rank among A's graph neighbors?
  4. ColBERT vs Cosine Agreement      — does M2 help or hurt over pure cosine?
  5. Beam Reach (Oracle Seed)         — given correct 1st hop, does beam find 2nd?

Usage (run from retrieval/ directory):
    python diagnose_models.py --max_examples 200    # fast (~10 min on T4)
    python diagnose_models.py                        # full val set (~40 min)
    python diagnose_models.py --skip_beam            # skip slowest test
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_musique, build_chain_quadruples, build_scoring_quintuples
from graph_builder import build_graph
from baselines import DenseRetriever, BM25Retriever, reciprocal_rank_fusion
from model1_train import ComplementEncoder, mean_pool
from model2_train import QueryEncoder

CACHE_DIR = Path(__file__).parent / "cache"
MODEL_DIR = Path(__file__).parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN_AB  = 256
MAX_LEN_C   = 128
MAX_LEN_Q   = 64
BEAM_WIDTH  = 3
MAX_HOPS    = 3
STOP_THRESH = 0.05
ALPHA       = 0.5
QR_MIX      = 0.1


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models():
    """Returns (comp_enc, query_enc, ab_tok, q_tok) or None if checkpoints missing."""
    m1_ckpt = MODEL_DIR / "model1_complement.pt"
    m2_ckpt = MODEL_DIR / "model2_scorer.pt"

    if not m1_ckpt.exists():
        print(f"[diagnose] ERROR: Model 1 not found at {m1_ckpt}")
        return None
    if not m2_ckpt.exists():
        print(f"[diagnose] ERROR: Model 2 not found at {m2_ckpt}")
        return None

    comp_enc = ComplementEncoder().to(DEVICE)
    comp_enc.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
    comp_enc.eval()
    for p in comp_enc.parameters():
        p.requires_grad_(False)

    query_enc = QueryEncoder().to(DEVICE)
    query_enc.load_state_dict(torch.load(m2_ckpt, map_location=DEVICE))
    query_enc.eval()
    for p in query_enc.parameters():
        p.requires_grad_(False)

    ab_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    q_tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print(f"[diagnose] Loaded Model 1 from {m1_ckpt}")
    print(f"[diagnose] Loaded Model 2 from {m2_ckpt}")
    return comp_enc, query_enc, ab_tok, q_tok


# ── Low-level inference helpers ────────────────────────────────────────────────

@torch.no_grad()
def complement_pool(comp_enc, ab_tok, text_a: str, text_b: str) -> torch.Tensor:
    """Mean-pooled complement representation of (A, B). Returns [128]."""
    enc = ab_tok(
        text=text_a, text_pair=text_b,
        max_length=MAX_LEN_AB, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    sep_id = ab_tok.sep_token_id
    b_mask = (
        (enc["token_type_ids"] == 1) & (enc["input_ids"] != sep_id)
    ).to(DEVICE)
    tokens, pad_mask = comp_enc(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
        enc["token_type_ids"].to(DEVICE),
        b_mask,
    )
    return mean_pool(tokens, pad_mask).squeeze(0)   # [128]


@torch.no_grad()
def passage_pool(comp_enc, ab_tok, text: str) -> torch.Tensor:
    """Standalone passage encoding (used for C in M1 loss). Returns [128]."""
    enc = ab_tok(
        text=text,
        max_length=MAX_LEN_C, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    return comp_enc.encode_passage(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
    ).squeeze(0)  # [128]


@torch.no_grad()
def complement_tokens(comp_enc, ab_tok, text_a: str, text_b: str):
    """B-side token matrix of (A,B) for ColBERT MaxSim. Returns (tokens, pad_mask)."""
    enc = ab_tok(
        text=text_a, text_pair=text_b,
        max_length=MAX_LEN_AB, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    sep_id = ab_tok.sep_token_id
    b_mask = (
        (enc["token_type_ids"] == 1) & (enc["input_ids"] != sep_id)
    ).to(DEVICE)
    return comp_enc(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
        enc["token_type_ids"].to(DEVICE),
        b_mask,
    )  # (tokens [1,n,128], pad_mask [1,n])


@torch.no_grad()
def query_vec(query_enc, q_tok, question: str) -> torch.Tensor:
    """Encode query → q_vec [1, 128] L2-normalised mean pool."""
    enc = q_tok(
        text=question, max_length=MAX_LEN_Q,
        truncation=True, padding="max_length", return_tensors="pt",
    )
    return query_enc(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
    )  # [1, 128]


def m2_score(q_vec_t: torch.Tensor, comp_tokens: torch.Tensor, pad_mask: torch.Tensor) -> float:
    """M2 score: dot(q_vec, mean_pool(complement_tokens)). Returns float."""
    c_vec = mean_pool(comp_tokens, pad_mask)   # [1, 128]
    return float((q_vec_t * c_vec).sum(-1).item())


# ── DIAGNOSTIC 1 — M1 Complement Discriminability ────────────────────────────

def diag_m1(comp_enc, ab_tok, quads, id_to_text, max_samples=500):
    """
    C-anchor test: does complement(A,B_pos) have higher cosine-to-C than complement(A,B_neg)?
    Positive mean delta = M1 is encoding direction toward C.
    """
    print("\n[DIAG 1] M1 Complement Discriminability" + "─"*33)
    samples = [q for q in quads if q.get("chunk_c_id")][:max_samples]
    print(f"  Quadruples with known C : {len(samples)}")
    if not samples:
        print("  SKIP — no 3/4-hop quadruples found")
        return {}

    pos_sims, neg_sims, deltas = [], [], []

    for item in tqdm(samples, desc="  M1 discrim", ncols=80):
        t_a   = id_to_text.get(item["chunk_a_id"])
        t_bpos= id_to_text.get(item["chunk_b_pos_id"])
        t_c   = id_to_text.get(item["chunk_c_id"])
        if not all([t_a, t_bpos, t_c]):
            continue

        pool_c   = passage_pool(comp_enc, ab_tok, t_c)
        pool_pos = complement_pool(comp_enc, ab_tok, t_a, t_bpos)
        sim_pos  = float(torch.dot(pool_pos, pool_c))
        pos_sims.append(sim_pos)

        for neg_id in item["chunk_b_neg_ids"]:
            t_bneg = id_to_text.get(neg_id)
            if not t_bneg:
                continue
            pool_neg = complement_pool(comp_enc, ab_tok, t_a, t_bneg)
            sim_neg  = float(torch.dot(pool_neg, pool_c))
            neg_sims.append(sim_neg)
            deltas.append(sim_pos - sim_neg)

    r = {
        "n":           len(samples),
        "mean_pos":    float(np.mean(pos_sims))  if pos_sims  else 0.0,
        "mean_neg":    float(np.mean(neg_sims))  if neg_sims  else 0.0,
        "mean_delta":  float(np.mean(deltas))    if deltas    else 0.0,
        "pct_pos_wins":float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
    }

    print(f"  mean_pool(complement(A,B_pos))·pool(C)   : {r['mean_pos']:+.4f}")
    print(f"  mean_pool(complement(A,B_neg))·pool(C)   : {r['mean_neg']:+.4f}")
    print(f"  mean delta (pos − neg)                   : {r['mean_delta']:+.4f}")
    print(f"  % cases pos beats neg                    : {r['pct_pos_wins']*100:.1f}%")

    if r["mean_delta"] < 0.01:
        print("  VERDICT: *** M1 NOT LEARNING — complement vectors are near-random ***")
    elif r["mean_delta"] < 0.05:
        print("  VERDICT:  ~  M1 WEAKLY discriminating (small C-direction signal)")
    else:
        print("  VERDICT:  ✓  M1 has clear C-direction signal")
    return r


# ── DIAGNOSTIC 2 — Edge Coverage ─────────────────────────────────────────────

def diag_coverage(quints, graph):
    """What fraction of ground-truth (A→B_pos) pairs are connected in the graph?"""
    print("\n[DIAG 2] Graph Edge Coverage" + "─"*44)
    total, covered = 0, 0
    by_type = defaultdict(int)

    for item in quints:
        a_id  = item["chunk_a_id"]
        b_pos = item["chunk_b_pos_id"]
        nbr_map = {nid: et for (nid, _, et) in graph.get(a_id, [])}
        total += 1
        if b_pos in nbr_map:
            covered += 1
            by_type[nbr_map[b_pos]] += 1
        else:
            by_type["missing"] += 1

    r = {
        "total":      total,
        "covered":    covered,
        "coverage":   covered / total * 100 if total else 0.0,
        "sequential": by_type["sequential"],
        "semantic":   by_type["semantic"],
        "missing":    by_type["missing"],
    }
    print(f"  True (A→B) hop pairs total      : {total}")
    print(f"  Connected by ANY edge            : {covered}  ({r['coverage']:.1f}%)")
    print(f"    via sequential edge            : {r['sequential']}")
    print(f"    via semantic  edge (cos≥0.70)  : {r['semantic']}")
    print(f"  MISSING from graph               : {r['missing']}  ({r['missing']/total*100:.1f}%)")

    if r["coverage"] < 50:
        print("  VERDICT: *** CRITICAL — graph misses >50% of true hops (biggest bottleneck) ***")
    elif r["coverage"] < 75:
        print("  VERDICT:  ~  Graph covers ~" + f"{r['coverage']:.0f}% of hops — significant gaps")
    else:
        print(f"  VERDICT:  ✓  Good edge coverage at {r['coverage']:.0f}%")
    return r


# ── DIAGNOSTIC 3 — M2 Ranking Position ───────────────────────────────────────

def diag_m2_rank(comp_enc, query_enc, ab_tok, q_tok,
                 quints, graph, id_to_text, embeddings, id_to_idx, dense,
                 max_samples=300):
    """
    Among all graph neighbors of A, rank B_pos by:
      (a) M2 score: mean_pool(Q) · mean_pool(complement(A, nbr))
      (b) cosine(e5-query-emb, e5-passage-emb)   ← mirrors actual system
      (c) alpha-blend of (a) and (b)
    Report mean rank of B_pos and top-1/top-3 hit rates.
    """
    print("\n[DIAG 3] M2 Ranking Position" + "─"*44)

    covered = [(item, [nid for (nid,_,_) in graph.get(item["chunk_a_id"],[])],
                item["chunk_b_pos_id"])
               for item in quints
               if item["chunk_b_pos_id"] in
                  {nid for (nid,_,_) in graph.get(item["chunk_a_id"],[])}]
    print(f"  Quints where B_pos ∈ graph[A]   : {len(covered)} / {len(quints)}")
    if not covered:
        print("  SKIP — edge coverage is 0%, cannot evaluate ranking")
        return {}

    samples = covered[:max_samples]
    colbert_ranks, cosine_ranks, blended_ranks = [], [], []

    for item, nbr_ids, b_pos_id in tqdm(samples, desc="  M2 rank", ncols=80):
        question = item["question"]
        a_id     = item["chunk_a_id"]
        text_a   = id_to_text.get(a_id, "")

        q_vec_t = query_vec(query_enc, q_tok, question)
        q_emb   = dense.embed_query(question)   # 384-dim e5

        cb_scores  = {}
        cos_scores = {}

        for nid in nbr_ids:
            text_b  = id_to_text.get(nid, "")
            nid_idx = id_to_idx.get(nid)
            if not text_b or nid_idx is None:
                continue

            c_tok, c_pad    = complement_tokens(comp_enc, ab_tok, text_a, text_b)
            cb_scores[nid]  = m2_score(q_vec_t, c_tok, c_pad)
            cos_scores[nid] = float(np.dot(q_emb, embeddings[nid_idx]))

        if b_pos_id not in cb_scores:
            continue

        def rank_of(d, key):
            return sorted(d, key=d.__getitem__, reverse=True).index(key) + 1

        all_ids  = list(cb_scores)
        blended  = {nid: ALPHA * cb_scores[nid] + (1-ALPHA) * cos_scores.get(nid,0)
                    for nid in all_ids}

        colbert_ranks.append(rank_of(cb_scores,  b_pos_id))
        cosine_ranks.append( rank_of(cos_scores, b_pos_id))
        blended_ranks.append(rank_of(blended,    b_pos_id))

    def stats(ranks):
        if not ranks:
            return 0.0, 0.0, 0.0
        mr = float(np.mean(ranks))
        t1 = sum(1 for r in ranks if r == 1) / len(ranks) * 100
        t3 = sum(1 for r in ranks if r <= 3) / len(ranks) * 100
        return mr, t1, t3

    r = {}
    if colbert_ranks:
        mr_cb,  t1_cb,  t3_cb  = stats(colbert_ranks)
        mr_cos, t1_cos, t3_cos = stats(cosine_ranks)
        mr_bl,  t1_bl,  t3_bl  = stats(blended_ranks)

        r = {
            "n": len(colbert_ranks),
            "colbert_mean_rank": mr_cb,  "colbert_top1": t1_cb,  "colbert_top3": t3_cb,
            "cosine_mean_rank":  mr_cos, "cosine_top1":  t1_cos, "cosine_top3":  t3_cos,
            "blended_mean_rank": mr_bl,  "blended_top1": t1_bl,  "blended_top3": t3_bl,
        }

        w = 22
        print(f"  {'Method':<{w}}  {'MeanRank':>8}  {'Top-1%':>7}  {'Top-3%':>7}")
        print(f"  {'─'*w}  {'─'*8}  {'─'*7}  {'─'*7}")
        print(f"  {'M2 (mean-pool dot)':<{w}}  {mr_cb:>8.2f}  {t1_cb:>6.1f}%  {t3_cb:>6.1f}%")
        print(f"  {'Cosine (e5-small-v2)':<{w}}  {mr_cos:>8.2f}  {t1_cos:>6.1f}%  {t3_cos:>6.1f}%")
        print(f"  {'Alpha-blend (0.5/0.5)':<{w}}  {mr_bl:>8.2f}  {t1_bl:>6.1f}%  {t3_bl:>6.1f}%")

        gain = mr_cos - mr_cb
        if gain < -0.1:
            print(f"  VERDICT: *** M2 HURTS ranking by {-gain:.2f} positions vs cosine ***")
        elif gain < 0.1:
            print("  VERDICT:  ~  M2 adds no benefit over cosine (< 0.1 position improvement)")
        else:
            print(f"  VERDICT:  ✓  M2 improves ranking by {gain:.2f} positions over cosine")
    return r


# ── DIAGNOSTIC 4 — ColBERT vs Cosine Agreement ───────────────────────────────

def diag_agreement(comp_enc, query_enc, ab_tok, q_tok,
                   quints, graph, id_to_text, embeddings, id_to_idx, dense,
                   max_samples=300):
    """
    For each (Q, A, B_pos, B_negs) where both B_pos and at least one B_neg are in graph[A]:
    Classify each (B_pos vs B_neg) comparison as:
      both_correct / m2_only / cosine_only / both_wrong
    """
    print("\n[DIAG 4] M2 vs Cosine Agreement" + "─"*41)

    nbr_sets = {
        item["chunk_a_id"]: {nid for (nid,_,_) in graph.get(item["chunk_a_id"],[])}
        for item in quints
    }

    both_correct = colbert_only = cosine_only = both_wrong = n = 0

    for item in tqdm(quints[:max_samples], desc="  Agreement", ncols=80):
        a_id     = item["chunk_a_id"]
        b_pos_id = item["chunk_b_pos_id"]
        neg_ids  = item["chunk_b_neg_ids"]
        nbrs     = nbr_sets.get(a_id, set())

        if b_pos_id not in nbrs:
            continue
        valid_negs = [nid for nid in neg_ids if nid in nbrs and id_to_idx.get(nid) is not None]
        if not valid_negs:
            continue

        text_a   = id_to_text.get(a_id, "")
        t_bpos   = id_to_text.get(b_pos_id, "")
        q_vec_t  = query_vec(query_enc, q_tok, item["question"])
        q_emb    = dense.embed_query(item["question"])

        c_pos_tok, c_pos_pad = complement_tokens(comp_enc, ab_tok, text_a, t_bpos)
        cb_pos   = m2_score(q_vec_t, c_pos_tok, c_pos_pad)
        bpos_idx = id_to_idx.get(b_pos_id)
        cos_pos  = float(np.dot(q_emb, embeddings[bpos_idx])) if bpos_idx is not None else 0.0

        for neg_id in valid_negs:
            t_bneg   = id_to_text.get(neg_id, "")
            neg_idx  = id_to_idx[neg_id]

            c_neg_tok, c_neg_pad = complement_tokens(comp_enc, ab_tok, text_a, t_bneg)
            cb_neg   = m2_score(q_vec_t, c_neg_tok, c_neg_pad)
            cos_neg  = float(np.dot(q_emb, embeddings[neg_idx]))

            cb_ok  = cb_pos  > cb_neg
            cos_ok = cos_pos > cos_neg
            n += 1

            if cb_ok and cos_ok:
                both_correct += 1
            elif cb_ok:
                colbert_only += 1
            elif cos_ok:
                cosine_only  += 1
            else:
                both_wrong   += 1

    r = {"n": n, "both_correct": both_correct, "m2_only": colbert_only,
         "cosine_only": cosine_only, "both_wrong": both_wrong}

    if n:
        print(f"  Comparisons (B_pos vs B_neg)     : {n}")
        print(f"  Both correct   : {both_correct:>5}  ({both_correct/n*100:5.1f}%)")
        print(f"  M2 only        : {colbert_only:>5}  ({colbert_only/n*100:5.1f}%)")
        print(f"  Cosine  only   : {cosine_only:>5}  ({cosine_only/n*100:5.1f}%)")
        print(f"  Both wrong     : {both_wrong:>5}  ({both_wrong/n*100:5.1f}%)")

        net = (colbert_only - cosine_only) / n * 100
        print(f"  M2 net gain vs cosine             : {net:+.1f}%")

        if net < -1:
            print("  VERDICT: *** M2 HURTS more than it helps (negative net) ***")
        elif abs(net) < 1:
            print("  VERDICT:  ~  M2 and cosine agree almost everywhere — M2 adds nothing")
        else:
            print(f"  VERDICT:  ✓  M2 corrects cosine in +{net:.1f}% of extra cases")
    return r


# ── DIAGNOSTIC 5 — Beam Reach (Oracle Seed) ──────────────────────────────────

def diag_beam_reach(comp_enc, query_enc, ab_tok, q_tok,
                    queries, graph, id_to_text, embeddings, id_to_idx, dense,
                    max_samples=100):
    """
    Oracle experiment: give the beam the correct 1st-hop chunk as seed.
    Measure: does beam search find the 2nd hop within MAX_HOPS steps?
    This isolates traversal quality from seed retrieval quality.
    """
    print("\n[DIAG 5] Beam Reach (Oracle Seed)" + "─"*39)

    multi = [q for q in queries if len(q.get("chain_chunk_ids", [])) >= 2][:max_samples]
    print(f"  Multi-hop queries sampled        : {len(multi)}")
    if not multi:
        print("  SKIP")
        return {}

    found_2nd = stopped_early = 0
    hops_taken = []

    for q in tqdm(multi, desc="  Beam reach", ncols=80):
        chain    = q["chain_chunk_ids"]
        relevant = set(q["relevant_chunk_ids"])
        seed     = chain[0]
        target   = chain[1]

        q_vec_t = query_vec(query_enc, q_tok, q["question"])
        q_emb   = dense.embed_query(q["question"])  # e5-small-v2, same as live system

        retrieved_set = {seed}
        beam          = [seed]
        hops          = 0
        found         = False

        for hop in range(MAX_HOPS):
            beam_idxs = [id_to_idx[c] for c in beam if c in id_to_idx]
            if beam_idxs:
                beam_emb = embeddings[beam_idxs].mean(axis=0)
                q_emb    = q_emb + QR_MIX * beam_emb
                q_emb    = q_emb / (np.linalg.norm(q_emb) + 1e-9)

            candidates: Dict[str, float] = {}
            for cid in beam:
                text_a  = id_to_text.get(cid, "")
                c_idx   = id_to_idx.get(cid)
                if not text_a or c_idx is None:
                    continue
                for (nbr_id, _, _) in graph.get(cid, []):
                    if nbr_id in retrieved_set:
                        continue
                    nbr_idx = id_to_idx.get(nbr_id)
                    text_b  = id_to_text.get(nbr_id, "")
                    if nbr_idx is None or not text_b:
                        continue

                    c_tok, c_pad = complement_tokens(comp_enc, ab_tok, text_a, text_b)
                    cb    = m2_score(q_vec_t, c_tok, c_pad)
                    cos   = float(np.dot(q_emb, embeddings[nbr_idx]))
                    final = ALPHA * cb + (1-ALPHA) * cos
                    if nbr_id not in candidates or final > candidates[nbr_id]:
                        candidates[nbr_id] = final

            if not candidates or max(candidates.values()) < STOP_THRESH:
                stopped_early += 1
                break

            top = sorted(candidates, key=candidates.__getitem__, reverse=True)[:BEAM_WIDTH]
            beam = top
            retrieved_set.update(top)
            hops += 1

            if target in retrieved_set:
                found = True
                break

        found_2nd  += int(found)
        hops_taken.append(hops)

    n = len(multi)
    r = {
        "n":               n,
        "found_2nd":       found_2nd,
        "stopped_early":   stopped_early,
        "pct_found":       found_2nd / n * 100 if n else 0.0,
        "pct_stopped":     stopped_early / n * 100 if n else 0.0,
        "mean_hops":       float(np.mean(hops_taken)) if hops_taken else 0.0,
    }

    print(f"  2nd hop found (oracle seed)      : {found_2nd}/{n}  = {r['pct_found']:.1f}%")
    print(f"  Beam stopped early (<threshold)  : {stopped_early}  ({r['pct_stopped']:.1f}%)")
    print(f"  Mean hops taken                  : {r['mean_hops']:.2f}")

    if r["pct_found"] < 40:
        print("  VERDICT: *** Traversal FAILS even with oracle seed — scoring or graph is killing it ***")
    elif r["pct_found"] < 60:
        print("  VERDICT:  ~  Traversal finds 2nd hop only half the time — scoring is weak")
    else:
        print(f"  VERDICT:  ✓  Traversal reaches 2nd hop {r['pct_found']:.0f}% of time with correct seed")
    return r


# ── Summary report ─────────────────────────────────────────────────────────────

def print_summary(r1, r2, r3, r4, r5):
    print("\n" + "="*72)
    print("  PIPELINE BOTTLENECK SUMMARY")
    print("="*72)

    issues = []

    if r1:
        d = r1.get("mean_delta", 1.0)
        if d < 0.01:
            issues.append(("CRITICAL", "M1 NOT learning — complement vectors have no C-direction signal"))
        elif d < 0.05:
            issues.append(("MODERATE", f"M1 weakly discriminates (delta={d:.3f}) — may need better negatives"))

    if r2:
        cov = r2.get("coverage", 100.0)
        if cov < 50:
            issues.append(("CRITICAL",
                f"Graph edge coverage only {cov:.0f}% — most true hops have no path (fix graph first)"))
        elif cov < 75:
            issues.append(("MODERATE",
                f"Graph edge coverage {cov:.0f}% — add BM25/entity edges to close the gap"))

    if r3:
        cb_mr  = r3.get("colbert_mean_rank", 0)
        cos_mr = r3.get("cosine_mean_rank",  0)
        if cb_mr > cos_mr + 0.1:
            issues.append(("CRITICAL",
                f"M2 HURTS ranking ({cb_mr:.2f} vs cosine {cos_mr:.2f}) — scoring makes things worse"))
        elif abs(cb_mr - cos_mr) < 0.1:
            issues.append(("MODERATE", "M2 adds no ranking improvement over cosine (< 0.1 position gain)"))

    if r4 and r4.get("n", 0) > 0:
        net = (r4["colbert_only"] - r4["cosine_only"]) / r4["n"] * 100
        if net < -1:
            issues.append(("CRITICAL", f"ColBERT agreement net is {net:.1f}% (hurts more than helps)"))

    if r5:
        pf  = r5.get("pct_found",   100.0)
        ps  = r5.get("pct_stopped",   0.0)
        if pf < 40:
            issues.append(("CRITICAL",
                f"Traversal finds only {pf:.0f}% of 2nd hops even with oracle seed"))
        if ps > 30:
            issues.append(("MODERATE",
                f"Beam stops early {ps:.0f}% of time — lower STOP_THRESH 0.05→0.01"))

    if not issues:
        print("  No critical issues — pipeline appears healthy at this scale")
    else:
        for sev, msg in issues:
            tag = "[!!!]" if sev == "CRITICAL" else "[ ~ ]"
            print(f"  {tag} {msg}")

    print("\n  RECOMMENDED NEXT STEPS (in order of expected impact):")
    if r2 and r2.get("coverage", 100) < 70:
        print("  1. Fix graph: add BM25-overlap edges + entity-mention links between passages")
    if r1 and r1.get("mean_delta", 1) < 0.03:
        print("  2. Retrain M1 with cross-example hard negatives (BM25 top-K from full corpus)")
    if r3 and r3.get("colbert_mean_rank", 0) > r3.get("cosine_mean_rank", 0):
        print("  3. Try ALPHA=0 (cosine only) or jointly train M1+M2 end-to-end")
    if r5 and r5.get("pct_stopped", 0) > 30:
        print("  4. Lower STOP_THRESH from 0.05 to 0.01 in run_full_system.py")
    print("="*72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline component diagnostics")
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Val examples to load (default=200; None=all 2417)")
    parser.add_argument("--max_samples",  type=int, default=300,
                        help="Max samples per diagnostic (default=300)")
    parser.add_argument("--skip_m1",   action="store_true", help="Skip M1 discriminability")
    parser.add_argument("--skip_m2",   action="store_true", help="Skip M2 ranking + agreement")
    parser.add_argument("--skip_beam", action="store_true", help="Skip beam reach (slowest)")
    args = parser.parse_args()

    print(f"[diagnose] Device      : {DEVICE}")
    print(f"[diagnose] Max examples: {args.max_examples}")

    # ── Load data ──────────────────────────────────────────────────────────────
    corpus, queries = load_musique(
        split="validation", max_examples=args.max_examples, cache=True
    )
    id_to_text = {c["chunk_id"]: c["text"] for c in corpus}

    quads  = build_chain_quadruples(corpus, queries)
    quints = build_scoring_quintuples(corpus, queries)

    # ── Build indexes ──────────────────────────────────────────────────────────
    cache_name = f"musique_val_{args.max_examples}"

    print("[diagnose] Building/loading dense index ...")
    dense = DenseRetriever()
    dense.build(corpus, cache_name=f"dense_{cache_name}")

    n_vecs, dim = dense.index.ntotal, dense.index.d
    embeddings  = np.zeros((n_vecs, dim), dtype=np.float32)
    dense.index.reconstruct_n(0, n_vecs, embeddings)
    id_to_idx   = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    # ── Build / load graph ─────────────────────────────────────────────────────
    graph = build_graph(corpus, embeddings=embeddings, cache_name=cache_name)

    # ── Load models ────────────────────────────────────────────────────────────
    models = load_models()

    r1 = r2 = r3 = r4 = r5 = {}

    # Diag 2 needs no models
    r2 = diag_coverage(quints, graph)

    if models is None:
        print("[diagnose] Model checkpoints not found — edge coverage is all we can measure.")
        print_summary(r1, r2, r3, r4, r5)
        return

    comp_enc, query_enc, ab_tok, q_tok = models

    if not args.skip_m1:
        r1 = diag_m1(comp_enc, ab_tok, quads, id_to_text,
                     max_samples=args.max_samples)

    if not args.skip_m2:
        r3 = diag_m2_rank(comp_enc, query_enc, ab_tok, q_tok,
                          quints, graph, id_to_text, embeddings, id_to_idx, dense,
                          max_samples=args.max_samples)
        r4 = diag_agreement(comp_enc, query_enc, ab_tok, q_tok,
                             quints, graph, id_to_text, embeddings, id_to_idx, dense,
                             max_samples=args.max_samples)

    if not args.skip_beam:
        beam_n = min(args.max_samples, 100)
        r5 = diag_beam_reach(comp_enc, query_enc, ab_tok, q_tok,
                              queries, graph, id_to_text, embeddings, id_to_idx, dense,
                              max_samples=beam_n)

    print_summary(r1, r2, r3, r4, r5)


if __name__ == "__main__":
    main()
