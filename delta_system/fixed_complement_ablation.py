"""
fixed_complement_ablation.py -- the HONEST test: does the complement OPERATION add
anything over just "pool the bert_maxsim-novel tokens", on a NON-circular metric?

The earlier content cosine was near-circular (delta is ~a pool of novel tokens, so it
trivially aligns with the mean of novel tokens). This script fixes two things:

1. ABLATION — compare delta constructions, all zero-training, frozen BERT:
     enc_B       : mean of H_B            (plain encode(B) baseline)
     gate_only   : sum(g*H_B)/sum(g)      g = 1-match (bert_maxsim gate), NO subtraction
     gate_sub    : sum(g*comp)/sum(g)     full fixed op (comp = H_B - match*B_in_A)
     sub_only    : mean(comp)             subtraction, NO gate
     gold_novel  : mean of H_B over GOLD novel tokens   (ORACLE ceiling)
   If gate_only ~= gate_sub -> the subtraction adds nothing; it's just bert_maxsim pooling.

2. NON-CIRCULAR METRIC — retrieval: can delta_i identify ITS OWN novel content among all
   pairs' novel contents? For each i, rank {gold_novel_j} by cos(delta_i, gold_novel_j);
   report top-1 accuracy + MRR. This is discriminative across pairs (not an average of the
   same tokens), and is a real proxy for usefulness. Higher = delta captures the SPECIFIC
   novelty, not just "something novel-ish".
   (Also report cos(delta, A): lower = less A-contaminated.)

Reading:
  - If gate_sub > gate_only on retrieval -> the operation earns its keep.
  - If gate_sub ~= gate_only -> "complement op" = bert_maxsim pooling (no extra value).
  - If gate_* >> enc_B -> the novelty selection helps over plain encode(B).
  - gold_novel is the ceiling (uses the gold span).

Run: python fixed_complement_ablation.py --n 500 --tau 0.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from insertion_cloze_eval import load_pairs

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
CONSTRS = ["enc_B", "gate_only", "gate_sub", "sub_only", "gold_novel"]


def _retrieval(deltas, golds):
    """top-1 acc + MRR of retrieving gold_novel_i from all golds using delta_i."""
    d = F.normalize(torch.tensor(np.array(deltas)), dim=-1)
    g = F.normalize(torch.tensor(np.array(golds)), dim=-1)
    sim = d @ g.T                       # [N, N]
    N = sim.size(0)
    ranks = []
    top1 = 0
    for i in range(N):
        order = torch.argsort(sim[i], descending=True)
        rank = (order == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
    mrr = float(np.mean([1.0 / r for r in ranks]))
    return top1 / N, mrr


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int,   default=500)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--bs",  type=int,   default=16)
    args = ap.parse_args()

    print("=" * 76)
    print("FIXED complement ABLATION + NON-CIRCULAR retrieval test (zero training)")
    print(f"Device {DEVICE} | tau={args.tau} | pairs={args.n}")
    print("=" * 76)

    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()
    pairs = load_pairs(args.n, tok)
    print(f"Loaded {len(pairs)} pairs\n")

    store = {c: [] for c in CONSTRS}
    cos_a = {c: [] for c in CONSTRS}
    golds = []

    for i in range(0, len(pairs), args.bs):
        batch = pairs[i:i + args.bs]
        eA = tok([p["A"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        H_A = bert(input_ids=A_ids, attention_mask=A_m).last_hidden_state
        H_B = bert(input_ids=B_ids, attention_mask=B_m).last_hidden_state

        H_A_n = F.normalize(H_A, dim=-1); H_B_n = F.normalize(H_B, dim=-1)
        cos = torch.bmm(H_B_n, H_A_n.transpose(1, 2))
        A_valid = A_m.unsqueeze(1).float()
        cos_m = cos * A_valid + (1 - A_valid) * (-1e4)
        attn = torch.softmax(cos_m / args.tau, dim=-1) * A_valid
        attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-9)
        match = attn.max(dim=-1).values
        B_in_A = torch.bmm(attn, H_A)
        comp = H_B - match.unsqueeze(-1) * B_in_A
        g = (1.0 - match).clamp(min=0)

        HA_np, HB_np, comp_np, g_np = (H_A.cpu().numpy(), H_B.cpu().numpy(),
                                       comp.cpu().numpy(), g.cpu().numpy())
        Am_np = A_m.cpu().numpy()

        for j, p in enumerate(batch):
            rl  = min(p["real_len"], MAX_LEN)
            nov = p["novel_mask"][:rl]
            tokm = np.ones(rl, dtype=bool)
            if rl > 2:
                tokm[0] = False; tokm[rl - 1] = False
            novel_idx = (nov == 1) & tokm
            if novel_idx.sum() == 0 or (nov == 0)[tokm].sum() == 0:
                continue

            HBp = HB_np[j, :rl]; cp = comp_np[j, :rl]; gg = g_np[j, :rl] * tokm
            enc_A = (HA_np[j] * Am_np[j][:, None]).sum(0) / Am_np[j].sum().clip(min=1)
            vecs = {
                "enc_B":      HBp[tokm].mean(0),
                "gate_only":  (HBp * gg[:, None]).sum(0) / gg.sum().clip(min=1e-9),
                "gate_sub":   (cp  * gg[:, None]).sum(0) / gg.sum().clip(min=1e-9),
                "sub_only":   cp[tokm].mean(0),
                "gold_novel": HBp[novel_idx].mean(0),
            }
            gold = HBp[novel_idx].mean(0)
            golds.append(gold)
            for c in CONSTRS:
                store[c].append(vecs[c])
                v, a = vecs[c], enc_A
                cos_a[c].append(float(v @ a / (np.linalg.norm(v) * np.linalg.norm(a) + 1e-9)))

    n = len(golds)
    print(f"Valid pairs: {n}\n")
    print(f"  {'construction':<12}{'top1':>8}{'MRR':>8}{'cos(.,A)':>10}   note")
    print("  " + "-" * 64)
    notes = {
        "enc_B":      "plain encode(B) baseline",
        "gate_only":  "bert_maxsim pool, NO subtraction",
        "gate_sub":   "FULL fixed op (gate + subtraction)",
        "sub_only":   "subtraction, NO gate",
        "gold_novel": "ORACLE (uses gold span)",
    }
    res = {}
    for c in CONSTRS:
        top1, mrr = _retrieval(store[c], golds)
        res[c] = (top1, mrr)
        print(f"  {c:<12}{top1:>8.3f}{mrr:>8.3f}{np.mean(cos_a[c]):>10.3f}   {notes[c]}")
    print("=" * 76)
    print("READS")
    print(f"  subtraction value : gate_sub vs gate_only  "
          f"-> top1 {res['gate_sub'][0]:.3f} vs {res['gate_only'][0]:.3f}")
    print(f"  selection value   : gate_only vs enc_B     "
          f"-> top1 {res['gate_only'][0]:.3f} vs {res['enc_B'][0]:.3f}")
    print(f"  ceiling           : gold_novel top1 = {res['gold_novel'][0]:.3f}")
    print("  - gate_sub ~= gate_only  => subtraction adds nothing (it's bert_maxsim pooling)")
    print("  - gate_* >> enc_B        => novelty selection helps over plain encode(B)")
    print("=" * 76)


if __name__ == "__main__":
    main()
