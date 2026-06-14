"""
fixed_ablation_vitaminc.py -- the CORRECT semantic test: retrieval of the FACTUAL CHANGE
on VitaminC, where edits are subtle REPLACES (fact flips) with little lexical change.

Why not the SUPPORTS/REFUTES probe: delta = complement(A,B) is computed from the two
evidence passages ONLY and never sees the claim, so it cannot legitimately predict a
claim-relative label. The valid test uses a target delta CAN encode: the changed content.

Test (same non-circular retrieval as the IteraTeR ablation, now on factual flips):
  For each VitaminC contrast pair (A=evidence1, B=evidence2), gold change = difflib span
  of tokens changed/added in B vs A (the flipped fact). Build delta constructions, then ask:
  can delta_i retrieve ITS OWN changed content among all pairs' changes? (top-1 / MRR)

  enc_B / gate_only / gate_sub / sub_only / gold_novel  (same as IteraTeR ablation)

Reading vs the IteraTeR result (gate_sub top1 = 0.92 on insertions):
  - gate_sub stays high on these REPLACES too  -> delta captures SEMANTIC factual change,
    not just locating inserted text. The win generalizes.
  - gate_sub collapses toward enc_B / low       -> IteraTeR win was largely lexical
    (insertions are easy); subtle factual flips are not captured.

Run: python fixed_ablation_vitaminc.py --n 2000 --tau 0.1
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
from insertion_cloze_eval     import _novel_mask_difflib
from vitaminc_probe_eval      import load_vitaminc_pairs
from fixed_complement_ablation import _retrieval, CONSTRS

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int,   default=2000)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--bs",  type=int,   default=16)
    args = ap.parse_args()

    print("=" * 76)
    print("FIXED complement ABLATION on VitaminC FACTUAL FLIPS (retrieval, zero training)")
    print(f"Device {DEVICE} | tau={args.tau} | pairs={args.n}")
    print("=" * 76)

    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()

    raw = load_vitaminc_pairs(args.n)
    # Attach difflib change-mask (gold = changed/added tokens in B vs A)
    pairs = []
    for p in raw:
        mask, real_len = _novel_mask_difflib(p["A"], p["B"], tok)
        if mask is None:
            continue
        pairs.append({"A": p["A"], "B": p["B"], "novel_mask": mask, "real_len": real_len})
    print(f"Usable pairs with a detectable change: {len(pairs)}\n")
    if len(pairs) < 50:
        print("ERROR: too few usable pairs.")
        sys.exit(1)

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

        HAn = F.normalize(H_A, dim=-1); HBn = F.normalize(H_B, dim=-1)
        cos = torch.bmm(HBn, HAn.transpose(1, 2))
        Av  = A_m.unsqueeze(1).float()
        cm  = cos * Av + (1 - Av) * (-1e4)
        attn = torch.softmax(cm / args.tau, dim=-1) * Av
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
            if gg.sum() < 1e-9:
                continue
            enc_A = (HA_np[j] * Am_np[j][:, None]).sum(0) / Am_np[j].sum().clip(min=1)
            vecs = {
                "enc_B":      HBp[tokm].mean(0),
                "gate_only":  (HBp * gg[:, None]).sum(0) / gg.sum(),
                "gate_sub":   (cp  * gg[:, None]).sum(0) / gg.sum(),
                "sub_only":   cp[tokm].mean(0),
                "gold_novel": HBp[novel_idx].mean(0),
            }
            golds.append(vecs["gold_novel"])
            for c in CONSTRS:
                store[c].append(vecs[c])
                v = vecs[c]
                cos_a[c].append(float(v @ enc_A /
                                      (np.linalg.norm(v) * np.linalg.norm(enc_A) + 1e-9)))

    n = len(golds)
    print(f"Valid pairs scored: {n}\n")
    print(f"  {'construction':<12}{'top1':>8}{'MRR':>8}{'cos(.,A)':>10}")
    print("  " + "-" * 50)
    res = {}
    for c in CONSTRS:
        top1, mrr = _retrieval(store[c], golds)
        res[c] = (top1, mrr)
        print(f"  {c:<12}{top1:>8.3f}{mrr:>8.3f}{np.mean(cos_a[c]):>10.3f}")
    print("=" * 76)
    print(f"  IteraTeR reference (insertions): gate_sub top1 = 0.920")
    print(f"  THIS (VitaminC factual flips)  : gate_sub top1 = {res['gate_sub'][0]:.3f}")
    print("  - stays high  -> delta captures SEMANTIC factual change (not just inserted text)")
    print("  - collapses   -> IteraTeR win was largely lexical")
    print(f"  subtraction value here: gate_sub {res['gate_sub'][0]:.3f} vs gate_only {res['gate_only'][0]:.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
