"""
eval_recovery_prefix.py -- THE thesis test for the prefix-complement model
==========================================================================

Question: did the leak-free, self-supervised edge actually capture "what B adds"
(the inserted phrase) -- with NO labels used in training?

For held-out (A, B, inserted_phrase): can a representation identify the GOLD
inserted phrase among distractor phrases, via MaxSim (multi-vector late interaction)?

  edge   = G.extract_edge(A,B)      [TB,128]   the learned complement (the idea)
  enc_A  = encode_text(A)           [TA,128]   FAIR floor: A has NO phrase -> should be worst
  enc_B  = encode_text(B)           [TB,128]   CEILING: B contains the phrase verbatim

  score(rep, phrase) = MaxSim = mean_p  max_e ( phrase_p · rep_e )

Success (thesis): edge >> enc_A  (the edge captured phrase content A lacks),
approaching enc_B (the verbatim ceiling). enc_A near floor = sanity.

Usage:  python eval_recovery_prefix.py --n 1000 --pool 124000 --n_distract 19
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
from prefix_complement import PrefixComplementLM, MAX_A_LEN, MAX_B_LEN
from data_wikiedits import load_triples

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = _HERE / "models"
MAX_P_LEN = 32


def maxsim(rep, rep_mask, phrase, phrase_mask):
    """rep [Tr,128], phrase [Tp,128]; mean over real phrase tokens of max over real rep tokens."""
    sims = phrase @ rep.T                       # [Tp, Tr]
    sims = sims.masked_fill(~rep_mask.bool().unsqueeze(0), -1e4)
    best = sims.max(dim=1).values               # [Tp]
    best = best[phrase_mask.bool()]
    return best.mean().item() if best.numel() else -1e4


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="prefix_complement_best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=124000)
    ap.add_argument("--n_distract", type=int, default=19)
    ap.add_argument("--k_edge", type=int, default=2)
    ap.add_argument("--j_dec", type=int, default=1)
    args = ap.parse_args()

    print(f"Device: {DEVICE} | ckpt: {args.ckpt}")
    pool = load_triples(max_examples=args.pool, cache=True)
    triples = pool[:args.n]                      # held-out val slice (never gradient-trained)
    print(f"Eval triples (held-out): {len(triples)}")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = PrefixComplementLM(k_edge_layers=args.k_edge, j_dec_layers=args.j_dec).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / args.ckpt, map_location=DEVICE), strict=False)
    model.eval()

    def enc(texts, max_len):
        e = tok(texts, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
        v = model.encode_text(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE))
        return v.cpu(), e["attention_mask"]

    A = [t["base"] for t in triples]
    B = [t["edited"] for t in triples]
    P = [t["inserted"] for t in triples]

    # per-example reps
    encA, mA = enc(A, MAX_A_LEN)
    encB, mB = enc(B, MAX_B_LEN)
    phr, mP  = enc(P, MAX_P_LEN)                 # candidate phrase bank

    # edge (multi-vector) per example
    edges, mE = [], []
    bs = 32
    for i in range(0, len(triples), bs):
        ea = tok(A[i:i+bs], max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eb = tok(B[i:i+bs], max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")
        E = model.extract_edge(ea["input_ids"].to(DEVICE), ea["attention_mask"].to(DEVICE),
                               eb["input_ids"].to(DEVICE), eb["attention_mask"].to(DEVICE)).cpu()
        edges.append(E); mE.append(eb["attention_mask"])
    edge = torch.cat(edges); mE = torch.cat(mE)

    scorers = {"edge (idea)": (edge, mE), "enc_B (ceiling)": (encB, mB), "enc_A (floor)": (encA, mA)}
    rng = np.random.default_rng(0)
    n = len(triples); K = args.n_distract

    print("\n" + "=" * 56)
    print(f"  RECOVERY EVAL (prefix model)  n={n}, {K} distractors")
    print("=" * 56)
    print(f"  {'scorer':<18}{'acc@1':>10}{'MRR':>10}")
    print("  " + "-" * 40)
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
        print(f"  {name:<18}{hits/n:>10.4f}{rr/n:>10.4f}")
    print("=" * 56)
    e_acc = res["edge (idea)"][0]; a_acc = res["enc_A (floor)"][0]; b_acc = res["enc_B (ceiling)"][0]
    print(f"  edge vs enc_A(floor)   : {e_acc - a_acc:+.4f}")
    print(f"  edge vs enc_B(ceiling) : {e_acc - b_acc:+.4f}")
    print("=" * 56)
    if e_acc > a_acc + 0.10:
        print("  THESIS SUPPORTED: the self-supervised edge identifies 'what B adds'")
        print("  far better than A (which lacks the phrase) -- learned with NO labels.")
    elif e_acc > a_acc + 0.02:
        print("  PARTIAL: edge beats A floor but modestly.")
    else:
        print("  edge ~ A floor: the edge did NOT capture the added phrase.")
    print("=" * 56)
    return res


if __name__ == "__main__":
    main()
