"""
delta9_slots.py -- MULTI-ASPECT DELTA via K slots, on HARD edits, at a SINGLE data size (no sweep).

User's idea: instead of collapsing the difference into ONE mean-pooled vector, read it with K learned
slots (K separate aspect-vectors), so different facets of "what B adds" (numeric / entity / relational)
can be captured separately. Slots = K read-out queries over the SAME difference field (NOT K full
architectures); each slot attends with multi-head attention internally.

This run (one data size = full hard-edit pool):
  ours_direct  : single mean-pool of the delta over novel positions          (the 0.59 baseline)
  ours_slots   : K slots attend the delta field -> concat K aspect vectors    (the new thing)
  refs         : chance / meandiff / fixed_novelB / oracle  (zero-training)
Reports overall content-retrieval top-1 on a fixed test, PLUS a per-type breakdown
(numeric / entity / relational) so we see where any gain concentrates.

Run: python delta9_slots.py --n 20000 --test_n 800 --K 4
"""

import argparse
import sys
from collections import Counter
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
from delta2_model import DEVICE
from delta2_data  import load_edits, group_split
from delta7_scale import tokenize_all, precompute_aux, base_reps, train_direct, delta_reps
from delta6_metric import _std, _ridge, _pool
from transformers import BertTokenizerFast

HARD = {"numeric", "entity", "relational"}


def changed_phrase(A, B):
    aset = set(A.split())
    return " ".join(w for w in B.split() if w not in aset)


def _key_mask(nov_b, B_m):
    """slot-attention key_padding_mask: attend over novel positions (fall back to all-B if none)."""
    kpm = nov_b > 0.5
    empty = kpm.sum(1) == 0
    if empty.any():
        kpm = kpm.clone(); kpm[empty] = B_m[empty].bool()
    return ~kpm


def train_slots(T, nov, eph, train_idx, steps, K, bs=64, temp=0.07):
    ds = DeltaSystem(n_slots=K, vib=False, d0_aware=False).to(DEVICE)
    head = nn.Sequential(nn.Linear(K * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL)).to(DEVICE)
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
        denc = ds.dr_delta_enc(delta, src_key_padding_mask=~B_m.bool())
        kpm = _key_mask(nov[sel].to(DEVICE), B_m)
        slots = ds.dr_slots.expand(bs, -1, -1)
        slot_out, _ = ds.dr_slot_attn(slots, denc, denc, key_padding_mask=kpm)   # [bs,K,d]
        rep = slot_out.reshape(bs, K * D_MODEL)
        z = F.normalize(head(rep), dim=-1)
        e = F.normalize(eph[sel].to(DEVICE), dim=-1)
        loss = F.cross_entropy(z @ e.T / temp, torch.arange(bs, device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"      slots step {step:>5} | loss {loss.item():.3f}")
    return ds


@torch.no_grad()
def slot_reps(ds, T, nov, idx, K, bs=64):
    ds.eval(); reps = []; idx = torch.as_tensor(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        A_ids = T["A_ids"][b].to(DEVICE); A_m = T["A_m"][b].to(DEVICE)
        B_ids = T["B_ids"][b].to(DEVICE); B_m = T["B_m"][b].to(DEVICE)
        H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        delta, _d0, _ = ds.generate_delta(H_A, A_m, H_B, B_m)
        denc = ds.dr_delta_enc(delta, src_key_padding_mask=~B_m.bool())
        kpm = _key_mask(nov[b].to(DEVICE), B_m)
        slots = ds.dr_slots.expand(len(b), -1, -1)
        slot_out, _ = ds.dr_slot_attn(slots, denc, denc, key_padding_mask=kpm)
        reps.append(slot_out.reshape(len(b), K * D_MODEL))
    return torch.cat(reps)


def evaluate(R_tr, R_te, E_tr, E_te, te_types):
    """overall + per-type held-out retrieval top-1 (one probe, per-type retrieval pools)."""
    Xtr, Xte = _std(R_tr, R_te)
    W = _ridge(Xtr, E_tr)
    pred = torch.cat([Xte, torch.ones(Xte.size(0), 1, device=Xte.device)], 1) @ W

    def top1(mask):
        p = F.normalize(pred[mask], dim=-1); t = F.normalize(E_te[mask], dim=-1)
        sim = p @ t.T
        rank = (sim >= sim.diag().unsqueeze(1)).sum(1).float()
        return (rank == 1).float().mean().item(), int(mask.sum())

    dev = E_te.device
    out = {"overall": top1(torch.ones(E_te.size(0), dtype=torch.bool, device=dev))}
    for t in ["numeric", "entity", "relational"]:
        m = torch.tensor([x == t for x in te_types], device=dev)
        if int(m.sum()) > 5:
            out[t] = top1(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--test_n", type=int, default=800)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--probe_n", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 100)
    print(f"DELTA-9 SLOTS (K={args.K}) on HARD edits, single data size: multi-aspect delta vs single mean-pool")
    print(f"device={DEVICE}")
    print("=" * 100)
    edits = load_edits(args.n, tok)
    hard = []
    for e in edits:
        if e.get("type") not in HARD:
            continue
        ph = changed_phrase(e["A"], e["B"])
        if len(e["A"]) < 20 or len(e["B"]) < 20 or not ph.strip():
            continue
        hard.append({"A": e["A"], "B": e["B"], "phrase": ph, "group": e["group"], "type": e["type"]})
    print(f"loaded {len(edits)} -> {len(hard)} hard pairs | types {dict(Counter(h['type'] for h in hard))}")

    tr_pool, te = group_split(hard, test_frac=min(0.4, args.test_n / max(len(hard), 1)))
    te_types = [p["type"] for p in te]
    print(f"train_pool {len(tr_pool)} | fixed test {len(te)} | test types {dict(Counter(te_types))}")

    ds0 = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    T_tr, T_te = tokenize_all(tr_pool, tok), tokenize_all(te, tok)
    nov_tr, eph_tr = precompute_aux(ds0, T_tr, tr_pool, tok)
    nov_te, eph_te = precompute_aux(ds0, T_te, te, tok)
    te_idx = np.arange(len(te))
    E_te = eph_te[te_idx].to(DEVICE)
    pn = min(args.probe_n, len(tr_pool)); psamp = np.arange(pn)
    E_str = eph_tr[psamp].to(DEVICE)

    results = {}
    # zero-training references
    Btr = base_reps(ds0, T_tr, nov_tr, eph_tr, psamp)
    Bte = base_reps(ds0, T_te, nov_te, eph_te, te_idx)
    for name in ["chance", "meandiff", "fixed_novelB", "oracle"]:
        results[name] = evaluate(Btr[name], Bte[name], E_str, E_te, te_types)

    # ours_direct (single mean-pool), full data
    print(f"\ntraining ours_direct on {len(tr_pool)} hard pairs ...")
    ds_d = train_direct(T_tr, nov_tr, eph_tr, np.arange(len(tr_pool)), args.steps)
    results["ours_direct"] = evaluate(delta_reps(ds_d, T_tr, nov_tr, psamp),
                                      delta_reps(ds_d, T_te, nov_te, te_idx), E_str, E_te, te_types)
    del ds_d
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ours_slots (K aspect read-outs), full data
    print(f"training ours_slots (K={args.K}) on {len(tr_pool)} hard pairs ...")
    ds_s = train_slots(T_tr, nov_tr, eph_tr, np.arange(len(tr_pool)), args.steps, args.K)
    results["ours_slots"] = evaluate(slot_reps(ds_s, T_tr, nov_tr, psamp, args.K),
                                     slot_reps(ds_s, T_te, nov_te, te_idx, args.K), E_str, E_te, te_types)
    del ds_s; gc.collect(); torch.cuda.empty_cache()

    cols = ["overall", "numeric", "entity", "relational"]
    print("\n" + "=" * 100)
    print(f"HARD edits | held-out content-retrieval top-1 | per-type (pool restricted to each type)")
    print("-" * 100)
    print(f"{'approach':<16}" + "".join(f"{c:>13}" for c in cols))
    print("-" * 100)
    for name in ["chance", "meandiff", "fixed_novelB", "oracle", "ours_direct", "ours_slots"]:
        r = results[name]
        cells = []
        for c in cols:
            cells.append(f"{r[c][0]:.3f}" if c in r else "   -")
        print(f"{name:<16}" + "".join(f"{x:>13}" for x in cells))
    print("=" * 100)
    print(f"READ: does ours_slots (K={args.K}) beat ours_direct overall and especially on the compound/")
    print("      relational types? If ~equal -> single mean-pool already captures it; slots don't help here.")
    print("=" * 100)


if __name__ == "__main__":
    main()
