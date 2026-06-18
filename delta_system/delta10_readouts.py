"""
delta10_readouts.py -- fair 4-way comparison of READ-OUTs over the SAME delta field, on HARD edits.

All four share: frozen BERT -> generate_delta (8-head subtraction) -> token-level delta field over the
NOVEL region. They differ ONLY in how that field is read into the representation:

  mean_pool   (#2 current direct)    : uniform mean over novel positions
  attn_pool   (#3 standard MHA)      : ONE learned query, standard multi-head attention (softmax over
                                       KEYS), 8 heads -> 1 vector
  slot1       (#1 proper slot)       : K slots, slot attention (softmax over SLOTS = competition) +
                                       iterative GRU update, single-head
  slot8       (#4 proper slot + MHA) : same proper slot attention, multi-head (8)

Fairness: same input field for all; each read-out -> a head -> 768-d; we extract the POST-HEAD 768-d
embedding for ALL four (equal dims, no "slots win from more dims"); matched training budget with final
losses printed (so we can see if any under-fits). Trained with contrastive InfoNCE to the phrase
embedding. Refs: chance/meandiff/fixed_novelB/oracle. Per-type breakdown (numeric/entity/relational).

Run: python delta10_readouts.py --n 6000 --test_n 700 --K 4
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
from delta7_scale import tokenize_all, precompute_aux, base_reps
from delta6_metric import _std, _ridge, _pool
from transformers import BertTokenizerFast

HARD = {"numeric", "entity", "relational"}
ARCHS = ["mean_pool", "attn_pool", "slot1", "slot8"]


def changed_phrase(A, B):
    aset = set(A.split())
    return " ".join(w for w in B.split() if w not in aset)


def _key_mask(nov_b, B_m):
    """key_padding_mask (True=ignore): attend over novel positions, fall back to all-B if none."""
    kpm = nov_b > 0.5
    empty = kpm.sum(1) == 0
    if empty.any():
        kpm = kpm.clone(); kpm[empty] = B_m[empty].bool()
    return ~kpm


class AttnPool(nn.Module):
    """#3 standard multi-head attention pooling: one learned query, softmax over keys."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, key_padding_mask):
        out, _ = self.attn(self.q.expand(x.size(0), -1, -1), x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)                                         # [b, d]


class SlotAttention(nn.Module):
    """#1/#4 canonical slot attention (Locatello 2020): softmax over SLOTS (competition) + iterative."""
    def __init__(self, K, dim, iters=3, heads=1, eps=1e-8):
        super().__init__()
        self.K, self.dim, self.iters, self.heads, self.eps = K, dim, iters, heads, eps
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.xavier_uniform_(self.slots_logsigma)
        self.to_q = nn.Linear(dim, dim); self.to_k = nn.Linear(dim, dim); self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm_in = nn.LayerNorm(dim); self.norm_slots = nn.LayerNorm(dim); self.norm_mlp = nn.LayerNorm(dim)

    def _split(self, x, n, b):
        return x.view(b, n, self.heads, self.dh).transpose(1, 2)      # [b,H,n,dh]

    def forward(self, inputs, key_padding_mask):
        b, N, d = inputs.shape
        inputs = self.norm_in(inputs)
        k = self._split(self.to_k(inputs), N, b); v = self._split(self.to_v(inputs), N, b)
        mu = self.slots_mu.expand(b, self.K, d)
        slots = mu + self.slots_logsigma.exp().expand(b, self.K, d) * torch.randn_like(mu)
        for _ in range(self.iters):
            prev = slots
            q = self._split(self.to_q(self.norm_slots(slots)), self.K, b)   # [b,H,K,dh]
            logits = torch.einsum('bhkd,bhnd->bhkn', q, k) * self.scale     # [b,H,K,N]
            attn = logits.softmax(dim=2)                                    # softmax over SLOTS (K)
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], 0.0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)       # normalize over inputs
            updates = torch.einsum('bhkn,bhnd->bhkd', attn, v)             # [b,H,K,dh]
            updates = updates.transpose(1, 2).reshape(b, self.K, d)
            slots = self.gru(updates.reshape(-1, d), prev.reshape(-1, d)).view(b, self.K, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots                                                       # [b,K,d]


def build_readout(kind, K, dim=D_MODEL):
    if kind == "mean_pool": return None, dim
    if kind == "attn_pool": return AttnPool(dim, 8).to(DEVICE), dim
    if kind == "slot1":     return SlotAttention(K, dim, 3, 1).to(DEVICE), K * dim
    if kind == "slot8":     return SlotAttention(K, dim, 3, 8).to(DEVICE), K * dim
    raise ValueError(kind)


def apply_readout(kind, readout, delta, nov_b, B_m, K):
    if kind == "mean_pool":
        return _pool(delta, nov_b, B_m.float())
    kpm = _key_mask(nov_b, B_m)
    if kind == "attn_pool":
        return readout(delta, kpm)
    slots = readout(delta, kpm)                                            # [b,K,d]
    return slots.reshape(slots.size(0), K * delta.size(-1))


def train_arch(kind, T, nov, eph, train_idx, steps, K, bs=64, temp=0.07):
    ds = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    readout, in_dim = build_readout(kind, K)
    head = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL)).to(DEVICE)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    if readout is not None: params += list(readout.parameters())
    params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0)
    ds.train(); head.train()
    if readout is not None: readout.train()
    last = 0.0
    for step in range(1, steps + 1):
        sel = torch.as_tensor(rng.choice(train_idx, bs, replace=True))
        A_ids = T["A_ids"][sel].to(DEVICE); A_m = T["A_m"][sel].to(DEVICE)
        B_ids = T["B_ids"][sel].to(DEVICE); B_m = T["B_m"][sel].to(DEVICE)
        with torch.no_grad():
            H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        delta = ds.generate_delta(H_A, A_m, H_B, B_m)[0]
        rep = apply_readout(kind, readout, delta, nov[sel].to(DEVICE), B_m, K)
        z = F.normalize(head(rep), dim=-1)
        e = F.normalize(eph[sel].to(DEVICE), dim=-1)
        loss = F.cross_entropy(z @ e.T / temp, torch.arange(bs, device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        if step % 1000 == 0:
            print(f"      {kind} step {step:>5} | loss {last:.3f}")
    return ds, readout, head, last


@torch.no_grad()
def extract(kind, ds, readout, head, T, nov, idx, K, bs=64):
    ds.eval(); head.eval()
    if readout is not None: readout.eval()
    idx = torch.as_tensor(idx); reps = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        A_ids = T["A_ids"][b].to(DEVICE); A_m = T["A_m"][b].to(DEVICE)
        B_ids = T["B_ids"][b].to(DEVICE); B_m = T["B_m"][b].to(DEVICE)
        H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        delta = ds.generate_delta(H_A, A_m, H_B, B_m)[0]
        rep = apply_readout(kind, readout, delta, nov[b].to(DEVICE), B_m, K)
        reps.append(head(rep))                                              # POST-head 768-d
    return torch.cat(reps)


def evaluate(R_tr, R_te, E_tr, E_te, te_types):
    Xtr, Xte = _std(R_tr, R_te)
    W = _ridge(Xtr, E_tr)
    pred = torch.cat([Xte, torch.ones(Xte.size(0), 1, device=Xte.device)], 1) @ W

    def top1(mask):
        p = F.normalize(pred[mask], dim=-1); t = F.normalize(E_te[mask], dim=-1)
        sim = p @ t.T
        rank = (sim >= sim.diag().unsqueeze(1)).sum(1).float()
        return (rank == 1).float().mean().item()

    dev = E_te.device
    out = {"overall": top1(torch.ones(E_te.size(0), dtype=torch.bool, device=dev))}
    for t in ["numeric", "entity", "relational"]:
        m = torch.tensor([x == t for x in te_types], device=dev)
        if int(m.sum()) > 5:
            out[t] = top1(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--test_n", type=int, default=700)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--probe_n", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 100)
    print(f"DELTA-10 READ-OUTS (K={args.K}) on HARD edits: mean-pool vs std-MHA-pool vs slot(1h) vs slot(8h)")
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
    te_idx = np.arange(len(te)); E_te = eph_te[te_idx].to(DEVICE)
    pn = min(args.probe_n, len(tr_pool)); psamp = np.arange(pn); E_str = eph_tr[psamp].to(DEVICE)

    results, losses = {}, {}
    Btr = base_reps(ds0, T_tr, nov_tr, eph_tr, psamp)
    Bte = base_reps(ds0, T_te, nov_te, eph_te, te_idx)
    for name in ["chance", "meandiff", "fixed_novelB", "oracle"]:
        results[name] = evaluate(Btr[name], Bte[name], E_str, E_te, te_types)

    import gc
    full = np.arange(len(tr_pool))
    for kind in ARCHS:
        print(f"\ntraining {kind} on {len(tr_pool)} hard pairs ...")
        ds, readout, head, last = train_arch(kind, T_tr, nov_tr, eph_tr, full, args.steps, args.K)
        losses[kind] = last
        results[kind] = evaluate(extract(kind, ds, readout, head, T_tr, nov_tr, psamp, args.K),
                                 extract(kind, ds, readout, head, T_te, nov_te, te_idx, args.K),
                                 E_str, E_te, te_types)
        del ds, readout, head; gc.collect(); torch.cuda.empty_cache()

    cols = ["overall", "numeric", "entity", "relational"]
    print("\n" + "=" * 100)
    print("HARD edits | held-out content-retrieval top-1 | post-head 768-d for all learned read-outs")
    print("-" * 100)
    print(f"{'approach':<22}" + "".join(f"{c:>12}" for c in cols) + f"{'train_loss':>12}")
    print("-" * 100)
    for name in ["chance", "meandiff", "fixed_novelB", "oracle"]:
        r = results[name]
        cells = "".join(f"{(f'{r[c]:.3f}' if c in r else '-'):>12}" for c in cols)
        print(f"{name:<22}{cells}{'':>12}")
    print("-" * 100)
    for kind in ARCHS:
        r = results[kind]
        cells = "".join(f"{(f'{r[c]:.3f}' if c in r else '-'):>12}" for c in cols)
        print(f"{kind:<22}{cells}{losses[kind]:>12.3f}")
    print("=" * 100)
    print("READ: compare the 4 read-outs (overall + per-type). attn_pool>mean_pool => learned pooling helps;")
    print("      slot8>=slot1>mean => competing aspects help. Check train_loss parity (no under-fitting).")
    print("=" * 100)


if __name__ == "__main__":
    main()
