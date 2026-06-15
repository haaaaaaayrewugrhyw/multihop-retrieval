"""
delta2_gate_test.py -- the GATE half of gate->extract: can the pooled surface-invariant delta
SEPARATE real edits from paraphrases (the thing the token extractor's magnitude could NOT, AUC 0.43)?

We train the pooled paraphrase-aux delta on the TRAIN split (it learns ||delta||->0 for
meaning-preserving rewrites, large for real change), then on HELD-OUT data score each pair by
||delta|| and measure AUC(edit=1 vs paraphrase=0). Non-circular: the gate signal is the LEARNED
||delta|| (not NLI, which selected the paraphrases), evaluated on unseen pairs.

  high AUC (>0.85)  -> the gate works -> gate->extract validated end-to-end (gate filters rewordings,
                       token complement supplies content).
  AUC ~0.5          -> even detecting "did meaning change" is hard -> rethink the framing.

Run: python delta2_gate_test.py --n_edit 400 --steps 600
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
from delta2_data  import load_edits, change_type, group_split, load_validated_paraphrases
from transformers import BertTokenizerFast

TYPES = ["insertion", "entity", "relational", "numeric"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_edit", type=int, default=400)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 80)
    print(f"GATE TEST: ||delta|| separates edit vs paraphrase (held-out)  device={DEVICE}")
    print("=" * 80)
    edits = load_edits(args.n_edit, tok)
    for e in edits:
        e["type"] = change_type(e["A"], e["B"])
    tr_e, te_e = group_split(edits, test_frac=0.25)
    paras = load_validated_paraphrases(600); half = len(paras) // 2
    tr_p, te_p = paras[:half], paras[half:]
    print(f"edits train {len(tr_e)} / test {len(te_e)} | paraphrases train {len(tr_p)} / test {len(te_p)}")

    base = Delta2("paraphrase").to(DEVICE).eval()
    E_tr = encode_all(base, tr_e, "A", "B", tok); E_te = encode_all(base, te_e, "A", "B", tok)
    P_tr = encode_all(base, tr_p, "A", "A_para", tok); P_te = encode_all(base, te_p, "A", "A_para", tok)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nTraining pooled paraphrase-aux delta (the gate)...")
    model = train("paraphrase", E_tr, P_tr, None, args.steps, "v2")

    edit_norm = np.linalg.norm(rep_delta(model, E_te), axis=1)
    para_norm = np.linalg.norm(rep_delta(model, P_te), axis=1)
    types_te = np.array([e["type"] for e in te_e])

    print("\n||delta|| (want edits >> paraphrases):")
    print(f"  edits {edit_norm.mean():.3f}  paraphrases {para_norm.mean():.3f}  "
          f"ratio {edit_norm.mean()/max(para_norm.mean(),1e-6):.2f}")

    lab = np.r_[np.ones(len(edit_norm)), np.zeros(len(para_norm))]
    auc = roc_auc_score(lab, np.r_[edit_norm, para_norm])
    print(f"\nGATE AUC (edit=1 vs paraphrase=0) by ||delta|| = {auc:.3f}")
    print("  (compare: token-extractor magnitude AUC was 0.43 -> the gate is the part that must work)")

    print("\nper change-type gate AUC vs paraphrases:")
    for ct in TYPES:
        m = types_te == ct
        if m.sum() >= 5:
            l = np.r_[np.ones(int(m.sum())), np.zeros(len(para_norm))]
            a = roc_auc_score(l, np.r_[edit_norm[m], para_norm])
            print(f"  {ct:<11} n={int(m.sum()):<4} AUC {a:.3f}")

    print("\n" + "=" * 80)
    print("READ: high overall AUC (esp. on relational/numeric where the extractor FAILED) => the")
    print("pooled delta is a working surface-invariant GATE -> gate->extract validated. AUC ~0.5 =>")
    print("detecting 'did meaning change' is itself hard -> rethink, don't build the pipeline yet.")
    print("=" * 80)


if __name__ == "__main__":
    main()
