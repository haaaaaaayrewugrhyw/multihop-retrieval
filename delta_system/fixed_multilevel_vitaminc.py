"""
fixed_multilevel_vitaminc.py -- does going MULTI-LEVEL (the user's hierarchical idea) help
on the HARD case (VitaminC subtle factual flips), where last-layer-only got top1=0.179?

Mechanism we're testing: a factual flip "30"->"35" is weakly flagged at the LAST layer
(contextual embeddings of 30 and 35 are similar: "a number here") but strongly flagged at
SURFACE/early layers (different tokens). So combining levels may capture the change better.

For each BERT layer L in {1,4,8,12} we compute the SAME fixed match-scaled complement
ENTIRELY within layer L (cos_L, attn_L, match_L, comp_L, gate_L), pool to delta_L, and the
gold change gold_L = mean of H_B^L over the difflib-changed tokens. Then non-circular
retrieval: does delta_L retrieve its OWN gold_L among all pairs? Plus a COMBINED delta =
concat of L2-normalized per-layer deltas.

Reading:
  per-layer oracle (gold_L retrieving gold_L) = the achievable ceiling at that layer.
  - a single early/surface layer or the COMBINED clearly beats last-layer 0.179 -> the
    hierarchical idea helps on subtle factual change (semantic win, your idea validated).
  - nothing beats ~0.179 -> token-mismatch can't do subtle flips at any level (honest bound).

Run: python fixed_multilevel_vitaminc.py --n 2000 --tau 0.1
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
from insertion_cloze_eval      import _novel_mask_difflib
from vitaminc_probe_eval       import load_vitaminc_pairs
from fixed_complement_ablation import _retrieval

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
LAYERS  = [1, 4, 8, 12]


def _op(H_A, H_B, A_m, tau):
    """fixed match-scaled complement within one layer -> comp [b,T,D], g [b,T]."""
    HAn = F.normalize(H_A, dim=-1); HBn = F.normalize(H_B, dim=-1)
    cos = torch.bmm(HBn, HAn.transpose(1, 2))
    Av  = A_m.unsqueeze(1).float()
    cm  = cos * Av + (1 - Av) * (-1e4)
    attn = torch.softmax(cm / tau, dim=-1) * Av
    attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-9)
    match = attn.max(-1).values
    comp = H_B - match.unsqueeze(-1) * torch.bmm(attn, H_A)
    g = (1.0 - match).clamp(min=0)
    return comp, g


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int,   default=2000)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--bs",  type=int,   default=16)
    args = ap.parse_args()

    print("=" * 72)
    print("MULTI-LEVEL fixed complement on VitaminC factual flips (zero training)")
    print(f"Device {DEVICE} | tau={args.tau} | layers={LAYERS} | pairs={args.n}")
    print("=" * 72)

    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()

    raw = load_vitaminc_pairs(args.n)
    pairs = []
    for p in raw:
        mask, real_len = _novel_mask_difflib(p["A"], p["B"], tok)
        if mask is not None:
            pairs.append({"A": p["A"], "B": p["B"], "novel_mask": mask, "real_len": real_len})
    print(f"Usable pairs: {len(pairs)}\n")
    if len(pairs) < 50:
        print("ERROR: too few usable pairs."); sys.exit(1)

    deltas = {L: [] for L in LAYERS}
    golds  = {L: [] for L in LAYERS}

    for i in range(0, len(pairs), args.bs):
        batch = pairs[i:i + args.bs]
        eA = tok([p["A"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        hsA = bert(input_ids=A_ids, attention_mask=A_m, output_hidden_states=True).hidden_states
        hsB = bert(input_ids=B_ids, attention_mask=B_m, output_hidden_states=True).hidden_states

        # Decide inclusion ONCE per pair (layer-independent), so all layers stay aligned.
        valid = []
        for j, p in enumerate(batch):
            rl  = min(p["real_len"], MAX_LEN)
            nov = p["novel_mask"][:rl]
            tokm = np.ones(rl, dtype=bool)
            if rl > 2:
                tokm[0] = False; tokm[rl - 1] = False
            novel_idx = (nov == 1) & tokm
            if novel_idx.sum() > 0 and (nov == 0)[tokm].sum() > 0:
                valid.append((j, rl, tokm, novel_idx))

        for L in LAYERS:
            comp, g = _op(hsA[L], hsB[L], A_m, tau=args.tau)
            comp_np = comp.cpu().numpy(); g_np = g.cpu().numpy(); HB_np = hsB[L].cpu().numpy()
            for (j, rl, tokm, novel_idx) in valid:
                gg = g_np[j, :rl] * tokm
                d  = (comp_np[j, :rl] * gg[:, None]).sum(0) / max(gg.sum(), 1e-9)
                gold = HB_np[j, :rl][novel_idx].mean(0)
                deltas[L].append(d / (np.linalg.norm(d) + 1e-9))
                golds[L].append(gold / (np.linalg.norm(gold) + 1e-9))

    m = len(deltas[LAYERS[0]])
    print(f"Aligned pairs scored: {m}\n")

    print(f"  {'layer':<10}{'gate_sub top1':>15}{'oracle top1':>14}")
    print("  " + "-" * 40)
    best = (None, -1)
    for L in LAYERS:
        t1, _   = _retrieval(deltas[L], golds[L])
        o1, _   = _retrieval(golds[L],  golds[L])
        print(f"  layer {L:<4}{t1:>15.3f}{o1:>14.3f}")
        if t1 > best[1]:
            best = (f"layer{L}", t1)

    # combined = concat L2-normalized per-layer vectors
    comb_d = [np.concatenate([deltas[L][k] for L in LAYERS]) for k in range(m)]
    comb_g = [np.concatenate([golds[L][k]  for L in LAYERS]) for k in range(m)]
    ct1, _ = _retrieval(comb_d, comb_g)
    co1, _ = _retrieval(comb_g, comb_g)
    print(f"  {'concat':<10}{ct1:>15.3f}{co1:>14.3f}")
    if ct1 > best[1]:
        best = ("concat", ct1)

    print("=" * 72)
    print(f"  last-layer-only reference (layer 12, prior run): 0.179")
    print(f"  BEST here: {best[0]} = {best[1]:.3f}")
    print("  - clearly beats 0.179 -> multi-level/surface helps on subtle flips (hierarchy")
    print("    validated where it matters). - else token-mismatch can't do subtle flips.")
    print("=" * 72)


if __name__ == "__main__":
    main()
