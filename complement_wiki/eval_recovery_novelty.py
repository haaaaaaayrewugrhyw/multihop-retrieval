"""
eval_recovery_novelty.py -- same-space recovery test for the novelty autoencoder
================================================================================

Now FAIR (unlike the prefix model): the edge and encode_text share edge_proj+LN and
both run on raw BERT outputs -> one comparable space.

Can a representation pick the GOLD inserted phrase out of distractors, via MaxSim?
  edge_novel = edge tokens where the gate alpha is OPEN (the learned "what B adds")
  edge_all   = all edge tokens (no gating)
  enc_B      = encode_text(B)   CEILING (contains the phrase verbatim)
  enc_A      = encode_text(A)   FAIR FLOOR (A lacks the phrase)

Success (thesis): edge_novel >> enc_A, approaching enc_B -> the gated edge captured
the added content, learned with NO labels.

Usage:  python eval_recovery_novelty.py --n 1000 --pool 124000 --n_distract 19
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))
from novelty_ae import NoveltyAutoencoder, MAX_A_LEN, MAX_B_LEN
from data_wikiedits import load_triples

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = _HERE / "models"
MAX_P_LEN = 32


def maxsim(rep, rep_mask, phrase, phrase_mask):
    sims = phrase @ rep.T
    sims = sims.masked_fill(~rep_mask.bool().unsqueeze(0), -1e4)
    best = sims.max(dim=1).values[phrase_mask.bool()]
    return best.mean().item() if best.numel() else -1e4


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="novelty_ae_best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=124000)
    ap.add_argument("--n_distract", type=int, default=19)
    ap.add_argument("--j_dec", type=int, default=2)
    args = ap.parse_args()

    print(f"Device: {DEVICE} | ckpt: {args.ckpt}")
    pool = load_triples(max_examples=args.pool, cache=True)
    triples = pool[:args.n]
    print(f"Eval triples (held-out): {len(triples)}")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = NoveltyAutoencoder(j_dec_layers=args.j_dec).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / args.ckpt, map_location=DEVICE), strict=False)
    model.eval()

    A = [t["base"] for t in triples]; B = [t["edited"] for t in triples]; P = [t["inserted"] for t in triples]

    def enc(texts, ml):
        e = tok(texts, max_length=ml, truncation=True, padding="max_length", return_tensors="pt")
        v = model.encode_text(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE))
        return v.cpu(), e["attention_mask"]

    encA, mA = enc(A, MAX_A_LEN)
    encB, mB = enc(B, MAX_B_LEN)
    phr, mP  = enc(P, MAX_P_LEN)

    edges, mEall, mEnov = [], [], []
    bs = 32
    for i in range(0, len(triples), bs):
        ea = tok(A[i:i+bs], max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eb = tok(B[i:i+bs], max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")
        E, alpha, _, _ = model.generate_edge(ea["input_ids"].to(DEVICE), ea["attention_mask"].to(DEVICE),
                                             eb["input_ids"].to(DEVICE), eb["attention_mask"].to(DEVICE))
        E = E.cpu(); alpha = alpha.cpu(); real = eb["attention_mask"]
        edges.append(E); mEall.append(real)
        # novel mask = gate open above this example's own mean alpha
        nov = torch.zeros_like(real)
        for j in range(E.size(0)):
            rmask = real[j].bool()
            thr = alpha[j][rmask].mean() if rmask.any() else 0.0
            nov[j] = (real[j].bool() & (alpha[j] > thr)).long()
        mEnov.append(nov)
    edge = torch.cat(edges); mEall = torch.cat(mEall); mEnov = torch.cat(mEnov)

    scorers = {
        "edge_novel (idea)": (edge, mEnov),
        "edge_all":          (edge, mEall),
        "enc_B (ceiling)":   (encB, mB),
        "enc_A (floor)":     (encA, mA),
    }
    rng = np.random.default_rng(0)
    n = len(triples); K = args.n_distract
    print("\n" + "=" * 56)
    print(f"  RECOVERY (novelty AE, same space)  n={n}, {K} distractors")
    print("=" * 56)
    print(f"  {'scorer':<20}{'acc@1':>9}{'MRR':>9}")
    print("  " + "-" * 38)
    res = {}
    for name, (R, M) in scorers.items():
        hits, rr = 0, 0.0
        for i in range(n):
            negs = rng.choice([j for j in range(n) if j != i], size=min(K, n-1), replace=False)
            cand = np.concatenate([[i], negs])
            sc = [maxsim(R[i], M[i], phr[c], mP[c]) for c in cand]
            rank = 1 + sum(1 for s in sc[1:] if s >= sc[0])
            hits += int(rank == 1); rr += 1.0 / rank
        res[name] = (hits/n, rr/n)
        print(f"  {name:<20}{hits/n:>9.4f}{rr/n:>9.4f}")
    print("=" * 56)
    e = res["edge_novel (idea)"][0]; a = res["enc_A (floor)"][0]; b = res["enc_B (ceiling)"][0]
    eall = res["edge_all"][0]
    print(f"  edge_novel vs enc_A(floor)  : {e-a:+.4f}")
    print(f"  edge_novel vs enc_B(ceiling): {e-b:+.4f}")
    print(f"  edge_all   vs enc_B         : {eall-b:+.4f}  (edge_all == encode(B); ~0 expected)")
    print("=" * 56)
    # Honest bar: beating the A-floor is trivial (any chunk of B does). The gate only
    # WORKS if the sparse novel subset approaches the full-B ceiling -> it kept the
    # phrase tokens and dropped the rest.
    if e > a + 0.10 and e > 0.85 * b:
        print("  THESIS SUPPORTED: the SPARSE gated edge nearly matches the full-B ceiling")
        print("  -> the gate kept the added content and dropped the copied -> 'what B adds', no labels.")
    elif e > a + 0.05:
        print("  WEAK: gated edge beats the A floor but is well below the ceiling -> the gate")
        print("  is picking a generic chunk of B, NOT specifically the novel tokens. (raise --beta / check alpha-localization)")
    else:
        print("  NEGATIVE: gated edge ~ A floor.")
    print("=" * 56)
    return res


if __name__ == "__main__":
    main()
