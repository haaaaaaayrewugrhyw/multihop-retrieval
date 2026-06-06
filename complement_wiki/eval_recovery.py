"""
eval_recovery.py -- does the learned complement recover "what B adds"?
======================================================================

The decisive, intrinsic test of the complement idea, on WikiAtomicEdits where
the inserted phrase IS the ground-truth complement.

For held-out (A, B, inserted_phrase) triples, each scorer produces a vector; we
test how well it points at the gold inserted phrase:

  edge    = G.extract_complement(A, B)        # the idea
  enc_B   = encode_query(B)                    # B alone (degeneracy baseline)
  enc_A   = encode_query(A)                    # sanity floor (must be worst)
  enc_AB  = encode_query("A [SEP] B")          # joint baseline
  arith   = normalize(enc_B - enc_A)           # hand-crafted complement (the bar to beat)
  gold    = encode_query(inserted_phrase)      # target

Metric 1  mean cos(scorer, gold)
Metric 2  retrieval: among gold + K distractor phrases, rank by
          cos(scorer, encode_query(candidate)); report acc@1 and MRR.

DECISION: the idea is validated iff `edge` beats `arith` AND `enc_B`
(and enc_A is near the floor).

Usage:
    python eval_recovery.py --ckpt complement_wiki_best.pt --n 2000 --n_distract 19
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v3"))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))

from generator_train import ComplementGenerator, MAX_A_LEN, MAX_B_LEN
from data_wikiedits import load_triples

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = _HERE / "models"
MAX_P_LEN = 32   # inserted-phrase / candidate tokenization length


def _enc(model, tokenizer, texts, max_len, batch=64):
    """encode_query over a list of texts -> [N, D_PROJ] L2-normalized."""
    outs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        e = tokenizer(chunk, max_length=max_len, truncation=True,
                      padding="max_length", return_tensors="pt")
        v = model.encode_query(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE))
        outs.append(v.cpu())
    return torch.cat(outs, 0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       type=str, default="complement_wiki_best.pt")
    ap.add_argument("--n",          type=int, default=2000, help="eval triples")
    ap.add_argument("--skip",       type=int, default=124000,
                    help="skip the first SKIP triples (= train+val used by training) so "
                         "eval is HELD-OUT. Must match train's max_examples+val_examples.")
    ap.add_argument("--n_distract", type=int, default=19,   help="distractors per query")
    ap.add_argument("--batch",      type=int, default=32)
    args = ap.parse_args()

    print(f"Device: {DEVICE} | ckpt: {args.ckpt}")
    # Load skip+n triples (same fixed-seed shuffle as training), then take the
    # held-out tail [skip : skip+n] so eval never overlaps training data.
    pool = load_triples(max_examples=args.skip + args.n, cache=True)
    triples = pool[args.skip:args.skip + args.n]
    if len(triples) < args.n:
        print(f"WARNING: only {len(triples)} held-out triples after skip={args.skip}; "
              "reduce --skip or --n, or filter yields fewer rows than expected.")
    print(f"Eval triples (held-out): {len(triples):,}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = ComplementGenerator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / args.ckpt, map_location=DEVICE), strict=False)
    model.eval()

    A   = [t["base"]     for t in triples]
    B   = [t["edited"]   for t in triples]
    P   = [t["inserted"] for t in triples]      # gold phrases
    AB  = [f"{a} [SEP] {b}" for a, b in zip(A, B)]

    # ── scorer vectors (one per example) ─────────────────────────────────────
    print("Encoding scorers ...")
    enc_A  = _enc(model, tokenizer, A,  MAX_A_LEN, args.batch)
    enc_B  = _enc(model, tokenizer, B,  MAX_B_LEN, args.batch)
    enc_AB = _enc(model, tokenizer, AB, MAX_A_LEN, args.batch)
    gold   = _enc(model, tokenizer, P,  MAX_P_LEN, args.batch)     # candidate space = phrases
    arith  = F.normalize(enc_B - enc_A, dim=-1)

    # edge = extract_complement(A,B), batched
    edges = []
    for i in range(0, len(triples), args.batch):
        ea = tokenizer(A[i:i+args.batch], max_length=MAX_A_LEN, truncation=True,
                       padding="max_length", return_tensors="pt")
        eb = tokenizer(B[i:i+args.batch], max_length=MAX_B_LEN, truncation=True,
                       padding="max_length", return_tensors="pt")
        e = model.extract_complement(ea["input_ids"].to(DEVICE),
                                     ea["attention_mask"].to(DEVICE),
                                     eb["input_ids"].to(DEVICE))
        edges.append(e.cpu())
    edge = torch.cat(edges, 0)

    scorers = {
        "edge (idea)":          edge,
        "arith B-A":            arith,
        "enc_B":                enc_B,
        "enc_AB":               enc_AB,
        "enc_A (floor)":        enc_A,
    }

    # ── Metric 1: direct recovery cos(scorer, gold) ──────────────────────────
    n = len(triples)
    # ── Metric 2: retrieval among gold + K distractor phrases ────────────────
    # candidate phrase bank = all gold phrase vectors; for query i, candidates are
    # gold[i] plus K random others.
    rng = np.random.default_rng(0)
    K = args.n_distract

    print("\n" + "=" * 70)
    print(f"  RECOVERY EVAL  (n={n}, {K} distractors per query)")
    print("=" * 70)
    print(f"  {'scorer':<18}{'mean_cos':>10}{'acc@1':>10}{'MRR':>10}")
    print("  " + "-" * 52)

    results = {}
    for name, S in scorers.items():
        # metric 1
        mean_cos = float((S * gold).sum(-1).mean())
        # metric 2: retrieval
        hits, rr = 0, 0.0
        for i in range(n):
            negs = rng.choice([j for j in range(n) if j != i], size=min(K, n - 1), replace=False)
            cand_idx = np.concatenate([[i], negs])
            cand = gold[cand_idx]                       # [1+K, D]
            sims = (S[i:i+1] * cand).sum(-1)            # cos to each candidate
            rank = 1 + int((sims[1:] >= sims[0]).sum()) # rank of gold (index 0)
            hits += int(rank == 1)
            rr   += 1.0 / rank
        acc, mrr = hits / n, rr / n
        results[name] = (mean_cos, acc, mrr)
        print(f"  {name:<18}{mean_cos:>10.4f}{acc:>10.4f}{mrr:>10.4f}")
    print("=" * 70)

    edge_acc = results["edge (idea)"][1]
    arith_acc = results["arith B-A"][1]
    encB_acc  = results["enc_B"][1]
    print(f"  edge vs arith(B-A) : {edge_acc - arith_acc:+.4f}")
    print(f"  edge vs enc_B      : {edge_acc - encB_acc:+.4f}")
    print("=" * 70)
    if edge_acc > arith_acc + 0.01 and edge_acc > encB_acc + 0.01:
        print("  VERDICT: IDEA VALIDATED. The learned complement recovers 'what B adds'")
        print("  better than plain vector subtraction and better than B alone.")
    elif edge_acc >= arith_acc - 0.01:
        print("  VERDICT: PARTIAL. Edge ~ arith(B-A): the complement concept holds, but")
        print("  the LEARNED model doesn't beat hand-crafted subtraction. (honest result)")
    else:
        print("  VERDICT: edge underperforms even arith(B-A) -- operation still off.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
