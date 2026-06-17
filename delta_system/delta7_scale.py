"""
delta7_scale.py -- DATA-SCALING curve for ours_direct (the delta block trained DIRECTLY, no decoder).

Question opened by delta6: ours_direct hit content-retrieval test_top1 0.464 (train 0.846, gap +0.38).
The gap is mostly memorization -> more data should shrink it. So: does test_top1 climb toward the
fixed-op line (fixed_novelB 0.66) as we add data, or plateau (irreducible instance-specific content)?

Design (for a FAIR curve):
  - ONE fixed held-out test set (same candidates) across all sizes -- retrieval difficulty constant.
  - vary ONLY the training-pool size; train ours_direct fresh at each size.
  - on-the-fly per-batch BERT encoding (token-level encodings for 36k pairs would be ~28GB on GPU).
  - constant reference lines: chance / meandiff / fixed_novelB / oracle on the same fixed test set.
Constraint: the WikiAtomicEdits mirror has ~49k pure insertions total -> max train ~36k (~11x of 3.2k).

Run: python delta7_scale.py --n 45000 --test_n 1500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model as M
M.A_DROP_P = 0.0
from model import DeltaSystem, D_MODEL
from delta2_model import DEVICE, MAX_LEN
from delta2_data  import load_wikiatomic_insertions, group_split
from insertion_cloze_eval import _novel_mask_difflib
from delta6_metric import _pool, _std, _ridge, _retr
from transformers import BertTokenizerFast

CLS_ID, SEP_ID = 101, 102
SIZES = [3000, 9000, 18000, 36000]


def tokenize_all(pairs, tok, bs=128):
    out = {k: [] for k in ["A_ids", "A_m", "B_ids", "B_m", "P_ids", "P_m"]}
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        for key, fld in [("A", "A"), ("B", "B"), ("P", "phrase")]:
            e = tok([p[fld] for p in ch], max_length=MAX_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
            out[f"{key}_ids"].append(e["input_ids"]); out[f"{key}_m"].append(e["attention_mask"])
    return {k: torch.cat(v) for k, v in out.items()}                  # CPU int tensors


@torch.no_grad()
def precompute_aux(ds, T, pairs, tok, bs=64):
    """nov mask [N,128] (CPU) and phrase embedding e_phrase [N,768] (CPU)."""
    nov = []
    for p in pairs:
        mask, rl = _novel_mask_difflib(p["A"], p["B"], tok)
        nv = np.zeros(MAX_LEN, np.float32)
        if mask is not None:
            rl = min(rl, MAX_LEN); tm = np.ones(rl, bool)
            if rl > 2: tm[0] = False; tm[rl - 1] = False
            nv[:rl] = (mask[:rl] == 1).astype(np.float32) * tm
        nov.append(nv)
    nov = torch.tensor(np.stack(nov))
    eph = []
    for i in range(0, len(pairs), bs):
        ids = T["P_ids"][i:i + bs].to(DEVICE); m = T["P_m"][i:i + bs].to(DEVICE)
        H = ds._enc(ids, m)
        content = (m.bool() & (ids != CLS_ID) & (ids != SEP_ID)).float()
        eph.append(_pool(H, content, m.float()).cpu())
    return nov, torch.cat(eph)


@torch.no_grad()
def delta_reps(ds, T, nov, idx, bs=64):
    """pooled-delta reps [len(idx),768] for the given row indices (on-the-fly encode)."""
    ds.eval(); reps = []; idx = torch.as_tensor(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        A_ids = T["A_ids"][b].to(DEVICE); A_m = T["A_m"][b].to(DEVICE)
        B_ids = T["B_ids"][b].to(DEVICE); B_m = T["B_m"][b].to(DEVICE)
        H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        delta, _d0, _ = ds.generate_delta(H_A, A_m, H_B, B_m)
        reps.append(_pool(delta, nov[b].to(DEVICE), B_m.float()))
    return torch.cat(reps)


@torch.no_grad()
def base_reps(ds, T, nov, eph, idx, bs=64):
    """zero-training reference reps for the same indices."""
    novB, mdiff, encB = [], [], []; idx = torch.as_tensor(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        A_ids = T["A_ids"][b].to(DEVICE); A_m = T["A_m"][b].to(DEVICE)
        B_ids = T["B_ids"][b].to(DEVICE); B_m = T["B_m"][b].to(DEVICE)
        H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        novB.append(_pool(H_B, nov[b].to(DEVICE), B_m.float()))
        mdiff.append(_pool(H_B, B_m.float(), B_m.float()) - _pool(H_A, A_m.float(), A_m.float()))
        encB.append(_pool(H_B, B_m.float(), B_m.float()))
    return {"fixed_novelB": torch.cat(novB), "meandiff": torch.cat(mdiff), "encB": torch.cat(encB),
            "oracle": eph[idx].to(DEVICE), "chance": torch.randn(len(idx), D_MODEL, device=DEVICE)}


def train_direct(T, nov, eph, train_idx, steps, bs=64, temp=0.07):
    ds = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL)).to(DEVICE)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0)
    ds.train(); head.train()
    for step in range(1, steps + 1):
        sel = torch.as_tensor(rng.choice(train_idx, bs, replace=True))
        A_ids = T["A_ids"][sel].to(DEVICE); A_m = T["A_m"][sel].to(DEVICE)
        B_ids = T["B_ids"][sel].to(DEVICE); B_m = T["B_m"][sel].to(DEVICE)
        with torch.no_grad():
            H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        delta, _d0, _ = ds.generate_delta(H_A, A_m, H_B, B_m)
        dvec = _pool(delta, nov[sel].to(DEVICE), B_m.float())
        z = F.normalize(head(dvec), dim=-1)
        e = F.normalize(eph[sel].to(DEVICE), dim=-1)
        logits = z @ e.T / temp
        loss = F.cross_entropy(logits, torch.arange(bs, device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"      step {step:>5} | loss {loss.item():.3f}")
    return ds


def metric(R_tr, R_te, E_tr, E_te):
    Xtr, Xte = _std(R_tr, R_te)
    W = _ridge(Xtr, E_tr)
    te_top1, mrr = _retr(Xte, W, E_te)
    tr_top1, _ = _retr(Xtr, W, E_tr)
    perm = torch.randperm(E_tr.size(0), device=E_tr.device)
    Wc = _ridge(Xtr, E_tr[perm]); ctrl, _ = _retr(Xte, Wc, E_te)
    return te_top1, mrr, tr_top1, tr_top1 - te_top1, te_top1 - ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45000)
    ap.add_argument("--test_n", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--probe_n", type=int, default=3000)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 96)
    print("DELTA-7 SCALING: ours_direct (delta block, no decoder) vs training-data size, fixed test set")
    print(f"device={DEVICE}")
    print("=" * 96)
    pairs = load_wikiatomic_insertions(args.n)
    tr_pool, te = group_split(pairs, test_frac=min(0.4, args.test_n / max(len(pairs), 1)))
    print(f"loaded {len(pairs)} | train_pool {len(tr_pool)} | fixed test {len(te)}")
    sizes = [s for s in SIZES if s <= len(tr_pool)]
    if len(tr_pool) not in sizes:
        sizes.append(len(tr_pool))
    print(f"data sizes: {sizes}")

    ds0 = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)     # for BERT encodings/baselines
    T_tr, T_te = tokenize_all(tr_pool, tok), tokenize_all(te, tok)
    nov_tr, eph_tr = precompute_aux(ds0, T_tr, tr_pool, tok)
    nov_te, eph_te = precompute_aux(ds0, T_te, te, tok)
    te_idx = np.arange(len(te))
    E_te = eph_te[te_idx].to(DEVICE)

    # constant reference lines (zero-training), probe fit on a train-pool sample
    pn = min(args.probe_n, len(tr_pool)); base_idx = np.arange(pn)
    Btr = base_reps(ds0, T_tr, nov_tr, eph_tr, base_idx)
    Bte = base_reps(ds0, T_te, nov_te, eph_te, te_idx)
    E_btr = eph_tr[base_idx].to(DEVICE)
    refs = {}
    for name in ["chance", "meandiff", "encB", "fixed_novelB", "oracle"]:
        refs[name] = metric(Btr[name], Bte[name], E_btr, E_te)

    # ours_direct at each data size (fixed test)
    rows = {}
    for size in sizes:
        st = max(args.steps, size // 6)                                    # bigger pool -> a few more steps
        print(f"\ntraining ours_direct on {size} pairs ({st} steps) ...")
        ds = train_direct(T_tr, nov_tr, eph_tr, np.arange(size), st)
        psamp = np.arange(min(args.probe_n, size))                         # probe-fit sample (model's train)
        R_tr = delta_reps(ds, T_tr, nov_tr, psamp)
        R_te = delta_reps(ds, T_te, nov_te, te_idx)
        E_str = eph_tr[psamp].to(DEVICE)
        rows[size] = metric(R_tr, R_te, E_str, E_te)
        del ds
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print(f"FIXED test set = {len(te)} candidates  |  metric = held-out content-retrieval top-1")
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
    print("READ: does ours_direct test_top1 climb toward fixed_novelB as n grows (data-limited memorization)")
    print("      or plateau below it (irreducible instance-specific content)? Watch gen_gap shrink too.")
    print("=" * 96)


if __name__ == "__main__":
    main()
