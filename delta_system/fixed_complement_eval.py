"""
fixed_complement_eval.py -- ZERO-TRAINING test of the FIXED match-scaled complement
operation on REAL BERT, REAL IteraTeR edits.

This tests the user's "fixed subtraction" idea in its cheapest form (single level, frozen
BERT, no learned weights at all). It is the real-data version of synthetic_complement_test.

Per B-token t (all frozen BERT, NO learned parameters):
  cos[t,j]  = cos(H_B[t], H_A[j])
  attn[t,j] = softmax(cos[t,:] / tau)              (sharp temperature)
  match[t]  = max_j attn[t,j]                       (how strongly B[t] matches some A token)
  B_in_A[t] = sum_j attn[t,j] * H_A[j]             (A's content aligned to B[t])
  comp[t]   = H_B[t] - match[t] * B_in_A[t]        (MATCH-SCALED subtraction: novel tokens,
                                                    with no A-match, are NOT corrupted)
  g[t]      = relu(1 - match[t])                    (gate: ~1 at novel, ~0 at shared)
  delta     = sum_t g[t]*comp[t] / sum_t g[t]      (gated-pooled complement vector)

TESTS (label-free except the gold inserted span, used ONLY for scoring):
  [A] Localization  -- does the op flag the inserted tokens?
        AUC( g[t] vs novel_mask )            (the operation's own gate)
        AUC( 1-maxcos[t] vs novel_mask )     (bert_maxsim reference, known ~0.948)
  [B] Content / specificity -- does the delta VECTOR point at what B added, not at A?
        enc_novel = mean H_B over gold-novel tokens   (the added content)
        enc_A     = mean H_A                           (the old content)
        enc_B     = mean H_B                           (plain encode(B) baseline)
        delta_spec = cos(delta, enc_novel) - cos(delta, enc_A)
        encB_spec  = cos(enc_B, enc_novel) - cos(enc_B, enc_A)
        ADVANTAGE  = delta_spec - encB_spec   (does the fixed op beat plain encode(B)
                                               at pointing novel-over-A?)

VERDICT: delta_spec > 0 AND ADVANTAGE > 0 (win-rate > 0.6) => the fixed op recovers
"what B adds beyond A" on real data, zero training. Then it's worth scaling to
multi-level (frozen BERT layers) + training a decoder on it.

Run:  python fixed_complement_eval.py --n 500 --tau 0.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from insertion_cloze_eval import load_pairs   # IteraTeR meaning-changed pairs + gold novel mask

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int,   default=500)
    ap.add_argument("--tau", type=float, default=0.1, help="attention temperature (sharp)")
    ap.add_argument("--bs",  type=int,   default=16)
    args = ap.parse_args()

    print("=" * 74)
    print("FIXED match-scaled complement on REAL BERT  (zero training)")
    print(f"Device {DEVICE} | tau={args.tau} | pairs={args.n}")
    print("=" * 74)

    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()

    pairs = load_pairs(args.n, tok)
    print(f"Loaded {len(pairs)} pairs\n")

    loc_auc_g, loc_auc_bms = [], []
    delta_spec, encB_spec  = [], []
    cos_d_nov, cos_d_a     = [], []
    win = 0

    for i in range(0, len(pairs), args.bs):
        batch = pairs[i:i + args.bs]
        eA = tok([p["A"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)

        H_A = bert(input_ids=A_ids, attention_mask=A_m).last_hidden_state   # [b,T,768]
        H_B = bert(input_ids=B_ids, attention_mask=B_m).last_hidden_state

        H_A_n = F.normalize(H_A, dim=-1)
        H_B_n = F.normalize(H_B, dim=-1)
        cos   = torch.bmm(H_B_n, H_A_n.transpose(1, 2))      # [b,Tb,Ta] cosine
        A_valid = A_m.unsqueeze(1).float()                   # [b,1,Ta]
        cos_m = cos * A_valid + (1 - A_valid) * (-1e4)

        max_cos = cos_m.max(dim=-1).values                   # [b,Tb] (bert_maxsim uses 1-this)
        attn    = torch.softmax(cos_m / args.tau, dim=-1)
        attn    = attn * A_valid
        attn    = attn / attn.sum(-1, keepdim=True).clamp(min=1e-9)
        match   = attn.max(dim=-1).values                    # [b,Tb] max attention weight
        B_in_A  = torch.bmm(attn, H_A)                       # [b,Tb,768]
        comp    = H_B - match.unsqueeze(-1) * B_in_A         # [b,Tb,768] match-scaled diff
        g       = (1.0 - match).clamp(min=0)                 # [b,Tb] gate

        comp_np = comp.cpu().numpy()
        g_np    = g.cpu().numpy()
        bms_np  = (1.0 - max_cos).cpu().numpy()
        HA_np   = H_A.cpu().numpy()
        HB_np   = H_B.cpu().numpy()
        Am_np   = A_m.cpu().numpy()

        for j, p in enumerate(batch):
            rl  = min(p["real_len"], MAX_LEN)
            nov = p["novel_mask"][:rl]
            tokm = np.ones(rl, dtype=bool)
            if rl > 2:
                tokm[0] = False; tokm[rl - 1] = False
            novel_idx  = (nov == 1) & tokm
            shared_idx = (nov == 0) & tokm
            if novel_idx.sum() == 0 or shared_idx.sum() == 0:
                continue

            lbl = nov[tokm].astype(int)
            try:
                loc_auc_g.append(roc_auc_score(lbl, g_np[j, :rl][tokm]))
                loc_auc_bms.append(roc_auc_score(lbl, bms_np[j, :rl][tokm]))
            except Exception:
                pass

            gg = g_np[j, :rl] * tokm
            if gg.sum() < 1e-9:
                continue
            delta     = (comp_np[j, :rl] * gg[:, None]).sum(0) / gg.sum()
            enc_A     = (HA_np[j] * Am_np[j][:, None]).sum(0) / Am_np[j].sum().clip(min=1)
            enc_novel = HB_np[j, :rl][novel_idx].mean(0)
            enc_B     = HB_np[j, :rl][tokm].mean(0)

            dn, da = _cos(delta, enc_novel), _cos(delta, enc_A)
            d_spec = dn - da
            b_spec = _cos(enc_B, enc_novel) - _cos(enc_B, enc_A)
            cos_d_nov.append(dn); cos_d_a.append(da)
            delta_spec.append(d_spec); encB_spec.append(b_spec)
            if d_spec > b_spec:
                win += 1

    n = len(delta_spec)
    print(f"Valid pairs scored: {n}\n")
    print("=" * 74)
    print("[A] LOCALIZATION  (AUC vs gold inserted span; higher = flags novel tokens)")
    print(f"  fixed-op gate  g = 1-match : AUC = {np.mean(loc_auc_g):.4f}")
    print(f"  bert_maxsim (1 - max cos)  : AUC = {np.mean(loc_auc_bms):.4f}   (reference ~0.95)")
    print()
    print("[B] CONTENT / SPECIFICITY  (does the delta VECTOR point at NOVEL, not A?)")
    print(f"  cos(delta, novel)          : {np.mean(cos_d_nov):+.4f}")
    print(f"  cos(delta, A)              : {np.mean(cos_d_a):+.4f}")
    print(f"  delta_spec = cos(d,nov)-cos(d,A) : {np.mean(delta_spec):+.4f}")
    print(f"  encB_spec  (plain encode B)      : {np.mean(encB_spec):+.4f}")
    adv = np.mean(np.array(delta_spec) - np.array(encB_spec))
    print(f"  ADVANTAGE  = delta_spec - encB_spec : {adv:+.4f}")
    print(f"  win-rate (delta_spec > encB_spec)   : {win / max(n,1):.3f}")
    print("=" * 74)

    d_mean = float(np.mean(delta_spec))
    print("VERDICT")
    if d_mean > 0.02 and adv > 0.02 and win / max(n, 1) > 0.6:
        print("  POSITIVE — the fixed op's delta points at what B ADDED, beats encode(B).")
        print("  -> Your fixed-difference idea works on real data. Scale to multi-level +")
        print("     train a decoder on it (specialization gap test).")
    elif d_mean > 0.02 and adv > 0:
        print("  PARTIAL — delta leans toward novel and beats encode(B), but modestly.")
        print("  -> Worth pursuing; try multi-level (BERT layers) / tune tau.")
    else:
        print("  NEGATIVE — the fixed op's delta does not point at the added content more")
        print("     than plain encode(B). Fixed single-level op is not enough on real text.")
    print("=" * 74)


if __name__ == "__main__":
    main()
