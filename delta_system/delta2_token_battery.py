"""
delta2_token_battery.py -- STEP-1 of "what next": baseline the ZERO-TRAINING token-level
match-scaled complement on the SAME held-out edits as the learned battery, STRATIFIED by
change-type, with the oracle per type.

Why this is the decisive-cheap next step:
  - confirms the 0.92 token-level retrieval in our exact held-out setup (the bar any learned
    delta must beat),
  - per-type breakdown (insertion / entity / relational / numeric) sizes the LEARNABLE slice,
  - the ORACLE per type (gold tokens retrieving gold tokens) directly exposes the ENCODER WALL:
    if even gold can't be told apart for numerics, no method on this encoder will.

Representations (all token-level, pooled only at the very end for retrieval):
  enc_B     : mean of B tokens                         [baseline: just B]
  gate_sub  : gate-weighted match-scaled complement    [the token-level delta]
  gold      : mean of B's difflib-novel tokens         [ORACLE = the changed content]

Run: python delta2_token_battery.py --n_edit 400 --tau 0.1
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
from delta2_data           import load_edits, change_type, group_split
from insertion_cloze_eval  import _novel_mask_difflib
from fixed_complement_ablation import _retrieval

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
TYPES   = ["insertion", "entity", "relational", "numeric"]


def op(H_A, H_B, A_m, tau):
    """fixed match-scaled complement: comp = H_B - match * (B attended into A); gate g = 1-match."""
    HAn = F.normalize(H_A, dim=-1); HBn = F.normalize(H_B, dim=-1)
    cos = torch.bmm(HBn, HAn.transpose(1, 2))
    Av  = A_m.unsqueeze(1).float()
    cm  = cos * Av + (1 - Av) * (-1e4)
    attn = torch.softmax(cm / tau, dim=-1) * Av
    attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-9)
    match = attn.max(-1).values
    comp = H_B - match.unsqueeze(-1) * torch.bmm(attn, H_A)
    return comp, (1.0 - match).clamp(min=0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_edit", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    print("=" * 78)
    print(f"TOKEN-LEVEL COMPLEMENT BATTERY (zero-training, held-out)  device={DEVICE}  tau={args.tau}")
    print("=" * 78)
    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased", low_cpu_mem_usage=True).to(DEVICE).eval()

    edits = load_edits(args.n_edit, tok)
    for e in edits:
        e["type"] = change_type(e["A"], e["B"])
    _, te = group_split(edits, test_frac=0.25)            # SAME held-out split as the learned battery
    print(f"held-out edits: {len(te)}")

    store = {"enc_B": [], "gate_sub": [], "gold": []}
    types = []
    for i in range(0, len(te), args.bs):
        batch = te[i:i + args.bs]
        eA = tok([p["A"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        H_A = bert(input_ids=A_ids, attention_mask=A_m).last_hidden_state
        H_B = bert(input_ids=B_ids, attention_mask=B_m).last_hidden_state
        comp, g = op(H_A, H_B, A_m, args.tau)
        HB_np, comp_np, g_np = H_B.cpu().numpy(), comp.cpu().numpy(), g.cpu().numpy()

        for j, p in enumerate(batch):
            mask, rl = _novel_mask_difflib(p["A"], p["B"], tok)
            if mask is None:
                continue
            rl = min(rl, MAX_LEN); nov = mask[:rl]
            tokm = np.ones(rl, dtype=bool)
            if rl > 2:
                tokm[0] = False; tokm[rl - 1] = False
            novel_idx = (nov == 1) & tokm
            if novel_idx.sum() == 0 or (nov == 0)[tokm].sum() == 0:
                continue
            gg = g_np[j, :rl] * tokm
            if gg.sum() < 1e-9:
                continue
            store["enc_B"].append(HB_np[j, :rl][tokm].mean(0))
            store["gate_sub"].append((comp_np[j, :rl] * gg[:, None]).sum(0) / gg.sum())
            store["gold"].append(HB_np[j, :rl][novel_idx].mean(0))
            types.append(p["type"])

    n = len(types)
    types = np.array(types)
    print(f"scored pairs: {n}\n")
    if n < 20:
        print("too few scored pairs"); return

    gold = np.stack(store["gold"])
    print(f"{'construction':<12}{'top1':>8}{'MRR':>8}   per-type top1 (n):")
    print("-" * 78)
    # oracle first (gold retrieving gold) -> the encoder ceiling
    for name in ["gold", "gate_sub", "enc_B"]:
        V = np.stack(store[name])
        t1, mr = _retrieval(V, gold)
        line = f"{('oracle' if name=='gold' else name):<12}{t1:>8.3f}{mr:>8.3f}   "
        for ct in TYPES:
            m = types == ct
            if m.sum() >= 5:
                tt, _ = _retrieval(V[m], gold[m])
                line += f"{ct[:4]}={tt:.2f}(n{int(m.sum())}) "
        print(line)
    print(f"\nchance top1 (full pool) = {1/n:.3f}")

    print("\n" + "=" * 78)
    print("READ: gate_sub top1 high on insertion = token-level extraction works (the bar).")
    print("ORACLE per type = the encoder ceiling: if oracle is LOW on numeric/relational, even")
    print("perfect gold tokens are indistinguishable there -> that slice is an ENCODER wall,")
    print("not a method failure. A learned delta is only worth it where oracle is HIGH but gate_sub LOW.")
    print("=" * 78)


if __name__ == "__main__":
    main()
