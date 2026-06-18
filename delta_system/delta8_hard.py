"""
delta8_hard.py -- THE DECISIVE TEST: content-retrieval on HARD edits (substitution / numeric /
relational), where the fixed op is WEAK. On insertions the learned delta only ties the free fixed op
(+0.05). The real question: on edits where the content is NOT just new tokens sitting in B, does the
directly-trained learned delta beat the fixed op by a LARGE margin (=> learning does what computing
can't), or also tie/lose (=> just use the free op)?

Data: load_edits (IteraTeR meaning-changed + VitaminC factual/numeric flips), kept to the hard types
{numeric, entity, relational} (drop insertion). "what B adds" phrase = words in B not in A.
Metric: same as delta7 -- decoder-free content-retrieval on a FIXED held-out test set, with the
constant baseline ladder (chance / meandiff / encB / fixed_novelB / oracle) and ours_direct (the delta
block trained directly, contrastive, no decoder) at a few data sizes.

Run: python delta8_hard.py --n 20000 --test_n 800
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model as M
M.A_DROP_P = 0.0
from model import DeltaSystem, D_MODEL
from delta2_model import DEVICE
from delta2_data  import load_edits, group_split
from delta7_scale import tokenize_all, precompute_aux, delta_reps, base_reps, train_direct, metric
from transformers import BertTokenizerFast

HARD = {"numeric", "entity", "relational"}
SIZES = [2000, 6000, 12000]


def changed_phrase(A, B):
    aset = set(A.split())
    return " ".join(w for w in B.split() if w not in aset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--test_n", type=int, default=800)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--probe_n", type=int, default=2000)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 96)
    print("DELTA-8 HARD EDITS: content-retrieval (learned-direct vs fixed op) on substitution/numeric")
    print(f"device={DEVICE}")
    print("=" * 96)
    edits = load_edits(args.n, tok)
    hard = []
    for e in edits:
        if e.get("type") not in HARD:
            continue
        ph = changed_phrase(e["A"], e["B"])
        if len(e["A"]) < 20 or len(e["B"]) < 20 or not ph.strip():
            continue
        hard.append({"A": e["A"], "B": e["B"], "phrase": ph, "group": e["group"], "type": e["type"]})
    from collections import Counter
    print(f"loaded {len(edits)} edits -> {len(hard)} hard pairs  | types {dict(Counter(h['type'] for h in hard))}")
    if len(hard) < 1500:
        print("WARNING: few hard pairs; results will be noisy / scaling limited.")

    tr_pool, te = group_split(hard, test_frac=min(0.4, args.test_n / max(len(hard), 1)))
    print(f"train_pool {len(tr_pool)} | fixed test {len(te)}")
    sizes = [s for s in SIZES if s <= len(tr_pool)]
    if not sizes or len(tr_pool) not in sizes:
        sizes.append(len(tr_pool))
    print(f"data sizes: {sizes}")

    ds0 = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    T_tr, T_te = tokenize_all(tr_pool, tok), tokenize_all(te, tok)
    nov_tr, eph_tr = precompute_aux(ds0, T_tr, tr_pool, tok)
    nov_te, eph_te = precompute_aux(ds0, T_te, te, tok)
    te_idx = np.arange(len(te))
    E_te = eph_te[te_idx].to(DEVICE)

    pn = min(args.probe_n, len(tr_pool)); base_idx = np.arange(pn)
    Btr = base_reps(ds0, T_tr, nov_tr, eph_tr, base_idx)
    Bte = base_reps(ds0, T_te, nov_te, eph_te, te_idx)
    E_btr = eph_tr[base_idx].to(DEVICE)
    refs = {name: metric(Btr[name], Bte[name], E_btr, E_te)
            for name in ["chance", "meandiff", "encB", "fixed_novelB", "oracle"]}

    rows = {}
    for size in sizes:
        st = max(args.steps, size // 6)
        print(f"\ntraining ours_direct on {size} hard pairs ({st} steps) ...")
        ds = train_direct(T_tr, nov_tr, eph_tr, np.arange(size), st)
        psamp = np.arange(min(args.probe_n, size))
        R_tr = delta_reps(ds, T_tr, nov_tr, psamp)
        R_te = delta_reps(ds, T_te, nov_te, te_idx)
        rows[size] = metric(R_tr, R_te, eph_tr[psamp].to(DEVICE), E_te)
        del ds
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print(f"HARD edits | FIXED test = {len(te)} candidates | metric = held-out content-retrieval top-1")
    print("-" * 96)
    print(f"{'approach':<22}{'test_top1':>10}{'MRR':>8}{'train_top1':>12}{'gen_gap':>9}{'selectiv':>10}")
    print("-" * 96)
    for name in ["chance", "meandiff", "encB", "fixed_novelB", "oracle"]:
        r = refs[name]
        print(f"{name:<22}{r[0]:>10.3f}{r[1]:>8.3f}{r[2]:>12.3f}{r[3]:>+9.3f}{r[4]:>+10.3f}")
    print("-" * 96)
    for size in sizes:
        r = rows[size]
        print(f"{('ours_direct n=' + str(size)):<22}{r[0]:>10.3f}{r[1]:>8.3f}{r[2]:>12.3f}"
              f"{r[3]:>+9.3f}{r[4]:>+10.3f}")
    print("=" * 96)
    print("READ: on HARD edits the fixed op should be WEAK. If ours_direct beats fixed_novelB by a LARGE")
    print("      margin -> learning does what computing can't (real positive). If it ties/loses -> use the op.")
    print("=" * 96)


if __name__ == "__main__":
    main()
