"""
delta2_gate_samedomain.py -- rule out the DOMAIN CONFOUND in the gate result.

The earlier gate AUC 0.99 used MRPC (news) paraphrases vs wiki edits -> could be detecting DOMAIN,
not "did meaning change." This re-runs the gate with SAME-DOMAIN negatives:

  positives (edit=1)   : IteraTeR meaning-changed pairs            (real meaning change)
  negatives (edit=0)   : IteraTeR NON-meaning-changed pairs        (fluency/clarity/coherence =
                          same source, same domain, meaning-PRESERVING rewrites)

Train the pooled paraphrase-aux delta (||delta||->0 on negatives, large on positives), then on
HELD-OUT pairs measure AUC(edit vs non-edit) by ||delta||.

  high AUC (still ~0.9+) -> the gate genuinely detects MEANING CHANGE (not domain) -> gate->extract
                            validated for real.
  collapses toward 0.5   -> the earlier 0.99 was a domain artifact; the gate doesn't really work.

Run: python delta2_gate_samedomain.py --n 400 --steps 600
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model        import DeltaSystem  # noqa: F401
from delta2_model import Delta2, encode_all, DEVICE
from delta2_eval  import train, rep_delta
from delta2_data  import load_iterater_by_meaning, change_type, group_split
from transformers import BertTokenizerFast

TYPES = ["insertion", "entity", "relational", "numeric"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 80)
    print(f"SAME-DOMAIN GATE TEST (IteraTeR meaning vs non-meaning)  device={DEVICE}")
    print("=" * 80)
    pos = load_iterater_by_meaning(args.n, meaning=True)     # real meaning change (edit=1)
    neg = load_iterater_by_meaning(args.n, meaning=False)    # same-domain meaning-preserving (edit=0)
    for p in pos:
        p["A"], p["B"], p["type"] = p["before"], p["after"], change_type(p["before"], p["after"])
    for p in neg:
        p["A"], p["A_para"] = p["before"], p["after"]
    print(f"positives (meaning-changed) {len(pos)} | negatives (non-meaning) {len(neg)}")

    tr_e, te_e = group_split(pos, test_frac=0.25)
    tr_p, te_p = group_split(neg, test_frac=0.25)
    print(f"train edit {len(tr_e)} / test edit {len(te_e)} | train non {len(tr_p)} / test non {len(te_p)}")

    base = Delta2("paraphrase").to(DEVICE).eval()
    E_tr = encode_all(base, tr_e, "A", "B", tok); E_te = encode_all(base, te_e, "A", "B", tok)
    P_tr = encode_all(base, tr_p, "A", "A_para", tok); P_te = encode_all(base, te_p, "A", "A_para", tok)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nTraining pooled paraphrase-aux delta (the gate), same-domain negatives...")
    model = train("paraphrase", E_tr, P_tr, None, args.steps, "v2")

    edit_norm = np.linalg.norm(rep_delta(model, E_te), axis=1)
    non_norm  = np.linalg.norm(rep_delta(model, P_te), axis=1)
    types_te = np.array([e["type"] for e in te_e])

    print("\n||delta|| (want meaning-changed >> non-meaning):")
    print(f"  meaning-changed {edit_norm.mean():.3f}  non-meaning {non_norm.mean():.3f}  "
          f"ratio {edit_norm.mean()/max(non_norm.mean(),1e-6):.2f}")

    lab = np.r_[np.ones(len(edit_norm)), np.zeros(len(non_norm))]
    auc = roc_auc_score(lab, np.r_[edit_norm, non_norm])
    print(f"\nSAME-DOMAIN GATE AUC = {auc:.3f}   (cross-domain was 0.990; chance 0.5)")

    print("\nper change-type AUC vs non-meaning:")
    for ct in TYPES:
        m = types_te == ct
        if m.sum() >= 5:
            l = np.r_[np.ones(int(m.sum())), np.zeros(len(non_norm))]
            a = roc_auc_score(l, np.r_[edit_norm[m], non_norm])
            print(f"  {ct:<11} n={int(m.sum()):<4} AUC {a:.3f}")

    print("\n" + "=" * 80)
    print("READ: AUC stays high (~0.9+) => gate detects MEANING CHANGE, not domain => gate->extract")
    print("validated for real. AUC collapses toward 0.5 => the 0.99 was a domain artifact.")
    print("=" * 80)


if __name__ == "__main__":
    main()
