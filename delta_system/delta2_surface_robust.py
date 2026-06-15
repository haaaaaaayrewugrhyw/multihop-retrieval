"""
delta2_surface_robust.py -- TEST claim #3: is the token-level complement SURFACE-ROBUST?

The complement keys on surface mismatch (gate = 1 - match). A paraphrase is a meaning-PRESERVING
surface change -- so a synonym/reword may FALSE-fire as "novelty." This test quantifies that risk.

For real edits (A,B) and validated paraphrases (A,A_para) we compute the complement's novelty
magnitude per pair:
  mean_gate = mean over content tokens of (1 - match)        [how "new" the op thinks the pair is]
  comp_norm = || gate-weighted complement ||                 [magnitude of the extracted delta]

Robust  => paraphrases score ~0, edits score high => high AUC separating edit-vs-paraphrase.
Over-fires => paraphrases look as "novel" as edits => AUC ~ 0.5 (surface-driven, not meaning-aware).
Stratified by edit change-type: shows WHERE it over-fires (insertions easy; substitutions risky).

Run: python delta2_surface_robust.py --n_edit 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import BertModel, BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from delta2_data import load_edits, change_type, load_validated_paraphrases

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
TAU     = 0.1
TYPES   = ["insertion", "entity", "relational", "numeric"]


def op(H_A, H_B, A_m, tau=TAU):
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
def scores(pairs, ka, kb, bert, tok, bs=16):
    gates, norms = [], []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        eA = tok([p[ka] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p[kb] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        H_A = bert(input_ids=A_ids, attention_mask=A_m).last_hidden_state
        H_B = bert(input_ids=B_ids, attention_mask=B_m).last_hidden_state
        comp, g = op(H_A, H_B, A_m)
        g_np, comp_np, m_np = g.cpu().numpy(), comp.cpu().numpy(), B_m.cpu().numpy()
        for j in range(len(batch)):
            rl = int(m_np[j].sum())
            tm = np.ones(rl, dtype=bool)
            if rl > 2:
                tm[0] = False; tm[rl - 1] = False
            gg = g_np[j, :rl] * tm
            if tm.sum() == 0:
                gates.append(0.0); norms.append(0.0); continue
            gates.append(float(gg.sum() / tm.sum()))
            pooled = (comp_np[j, :rl] * gg[:, None]).sum(0) / max(gg.sum(), 1e-9)
            norms.append(float(np.linalg.norm(pooled)))
    return np.array(gates), np.array(norms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_edit", type=int, default=400)
    args = ap.parse_args()
    tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased", low_cpu_mem_usage=True).to(DEVICE).eval()

    print("=" * 80)
    print(f"SURFACE-ROBUSTNESS of the token-level complement  device={DEVICE}")
    print("=" * 80)
    edits = load_edits(args.n_edit, tok)
    for e in edits:
        e["type"] = change_type(e["A"], e["B"])
    paras = load_validated_paraphrases(600)
    print(f"edits {len(edits)} | validated paraphrases {len(paras)}")

    eg, en = scores(edits, "A", "B", bert, tok)
    pg, pn = scores(paras, "A", "A_para", bert, tok)
    etypes = np.array([e["type"] for e in edits])

    print("\nnovelty magnitude (want edits >> paraphrases):")
    print(f"  mean_gate : edits {eg.mean():.3f}  paraphrases {pg.mean():.3f}  ratio {eg.mean()/max(pg.mean(),1e-6):.2f}")
    print(f"  comp_norm : edits {en.mean():.3f}  paraphrases {pn.mean():.3f}  ratio {en.mean()/max(pn.mean(),1e-6):.2f}")

    lab = np.r_[np.ones(len(eg)), np.zeros(len(pg))]
    auc_g = roc_auc_score(lab, np.r_[eg, pg])
    auc_n = roc_auc_score(lab, np.r_[en, pn])
    print("\nseparability AUC (edit=1 vs paraphrase=0; 0.5=can't tell change from reword):")
    print(f"  by mean_gate {auc_g:.3f} | by comp_norm {auc_n:.3f}")

    print("\nper change-type AUC vs paraphrases (where does it over-fire?):")
    for ct in TYPES:
        m = etypes == ct
        if m.sum() >= 5:
            l = np.r_[np.ones(int(m.sum())), np.zeros(len(pg))]
            a_g = roc_auc_score(l, np.r_[eg[m], pg])
            a_n = roc_auc_score(l, np.r_[en[m], pn])
            print(f"  {ct:<11} n={int(m.sum()):<4} AUC gate {a_g:.3f} | norm {a_n:.3f}")

    print("\n" + "=" * 80)
    print("READ: high AUC (esp. norm) = complement separates real change from reword (surface-ROBUST).")
    print("AUC ~0.5 on substitution/numeric = it fires on paraphrases too (surface-driven, NOT meaning-")
    print("aware) -> claim #3 fails there. Insertions likely high (new tokens) regardless.")
    print("=" * 80)


if __name__ == "__main__":
    main()
