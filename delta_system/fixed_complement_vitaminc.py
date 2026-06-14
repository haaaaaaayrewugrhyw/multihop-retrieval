"""
fixed_complement_vitaminc.py -- the SEMANTIC test: does the FIXED complement op capture
the DIRECTION of a factual change (not just locate inserted text)?

VitaminC pairs differ by a single factual flip (SUPPORTS <-> REFUTES) with little lexical
change. Locating "the inserted span" doesn't help here — you must capture what the change
MEANS. So a linear probe predicting SUPPORTS/REFUTES from the delta tests semantic content.

We compute, ZERO training, frozen BERT, the same fixed match-scaled complement, pooled to
one vector per pair, and probe it — against baselines:
    gate_sub   : FULL fixed op (gate + match-scaled subtraction)   [the candidate]
    gate_only  : bert_maxsim-gated pool of H_B (NO subtraction)
    enc_B      : plain mean-pool of B
    diff       : mean(B) - mean(A)
Reference: the LEARNED delta_system probe scored 0.7975 here (Experiment 10).

Reading:
  - gate_sub clearly > chance, selectivity > 0.05, and >= enc_B  -> the fixed op encodes
    SEMANTIC change direction (not just lexical insertion-finding). Real win.
  - gate_sub ~= chance / ~= enc_B  -> the IteraTeR result was largely lexical; the fixed op
    does not capture subtle semantic novelty.

Run: python fixed_complement_vitaminc.py --n 2000 --tau 0.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vitaminc_probe_eval import load_vitaminc_pairs, run_probe   # reuse data + probe

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def _pool(x, w):                       # x [b,T,D], w [b,T]
    return (x * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-9)


@torch.no_grad()
def compute_features(bert, tok, pairs, tau, bs=16):
    feats = {"gate_sub": [], "gate_only": [], "enc_B": [], "diff": []}
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
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
        attn = torch.softmax(cm / tau, dim=-1) * Av
        attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-9)
        match = attn.max(-1).values
        B_in_A = torch.bmm(attn, H_A)
        comp = H_B - match.unsqueeze(-1) * B_in_A
        g = (1.0 - match).clamp(min=0)

        Bmf = B_m.float()
        gB  = g * Bmf
        gate_sub  = _pool(comp, gB)
        gate_only = _pool(H_B,  gB)
        enc_B     = _pool(H_B,  Bmf)
        enc_A     = _pool(H_A,  A_m.float())
        diff      = enc_B - enc_A

        for k, v in [("gate_sub", gate_sub), ("gate_only", gate_only),
                     ("enc_B", enc_B), ("diff", diff)]:
            feats[k].append(v.cpu().numpy())

        if (i // bs) % 15 == 0:
            print(f"  encoding {min(i+bs, len(pairs))}/{len(pairs)}")

    return {k: np.vstack(v) for k, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",         type=int,   default=2000)
    ap.add_argument("--tau",       type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    print("=" * 72)
    print("FIXED complement on VitaminC — semantic change-direction probe (zero training)")
    print(f"Device {DEVICE} | tau={args.tau} | pairs={args.n}")
    print("=" * 72)

    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()

    pairs = load_vitaminc_pairs(args.n)
    if len(pairs) < 50:
        print(f"ERROR: only {len(pairs)} pairs.")
        sys.exit(1)
    labels = np.array([p["label"] for p in pairs])
    chance = float(max((labels == 0).mean(), (labels == 1).mean()))
    print(f"pairs={len(pairs)} | chance={chance:.4f}\n")

    feats = compute_features(bert, tok, pairs, args.tau)

    idx = np.arange(len(pairs))
    idx_tr, idx_te = train_test_split(idx, test_size=args.test_size,
                                      random_state=42, stratify=labels)
    y_tr, y_te = labels[idx_tr], labels[idx_te]
    rng = np.random.default_rng(99); y_sh = y_tr.copy(); rng.shuffle(y_sh)

    print("\n" + "=" * 72)
    print("PROBE — predict SUPPORTS/REFUTES from the (frozen) representation")
    print(f"  Train {len(idx_tr)} | Test {len(idx_te)} | chance {chance:.4f}")
    print(f"  {'Method':<40} {'Acc':>8} {'F1':>8} {'Sel':>8}")
    print("  " + "-" * 68)
    names = {
        "gate_sub":  "FIXED gate_sub delta  [candidate]",
        "gate_only": "gate_only  (bert_maxsim pool, no sub)",
        "enc_B":     "encode(B) pool  [baseline]",
        "diff":      "mean(B) - mean(A)  [baseline]",
    }
    res = {}
    for k in ["gate_sub", "gate_only", "enc_B", "diff"]:
        res[k] = run_probe(feats[k][idx_tr], feats[k][idx_te], y_tr, y_te, names[k], y_sh)
    print(f"  {'chance (majority)':<40} {chance:>8.4f}")
    print(f"  {'reference: LEARNED delta probe':<40} {0.7975:>8.4f}")
    print("=" * 72)

    gs = res["gate_sub"]
    print("VERDICT")
    if gs["acc"] > chance + 0.05 and gs["selectivity"] > 0.05 and gs["acc"] >= res["enc_B"]["acc"] - 0.01:
        print("  SEMANTIC — the fixed op encodes factual-change DIRECTION, not just inserted text.")
        print("  -> the IteraTeR win was real semantic novelty extraction. Scale it.")
    elif gs["acc"] > chance + 0.05:
        print("  PARTIAL — some semantic signal, but not clearly above encode(B). Inconclusive.")
    else:
        print("  LEXICAL — fixed op ~ chance/encode(B) here -> IteraTeR win was largely lexical")
        print("     (locating inserted text), not semantic understanding of the change.")
    print("=" * 72)


if __name__ == "__main__":
    main()
