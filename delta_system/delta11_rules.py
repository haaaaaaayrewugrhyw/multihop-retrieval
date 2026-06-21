"""
delta11_rules.py -- PHASE 1: rule-incorporated, role-vs-content delta (B = common (+) delta).

Design (signed off):
  delta = G(A,B) [subtraction block] -> VIB latent z (light, free-bits)   # minimality (beyond-A)
  ROLE rules:   gap-infill decoder (z fills the masked span of B)         # structural anti-bypass
                VICReg on z (variance + covariance)                       # anti-collapse
                light VIB KL with free-bits                               # minimality
  CONTENT objectives (decide what z carries):
    obj1 = gap-infill generation : D(common, z) -> the changed span       # common = B with span [MASK]ed
    obj2 = contrastive retrieval : H(z) -> phrase embedding (InfoNCE)
  4 runs: obj1@768, obj2@768, joint@768, joint@256.

Battery (the real test):
  (B) no-collapse + z-IS-USED : infill token-acc with z vs z-ablated (decoder runs)
  (C) content captured        : content-retrieval top-1 (frozen z -> ridge probe -> phrase), per-type
  (E) REUSABILITY (headline)  : does an obj1-trained z retrieve content ~as well as an obj2-trained z?
                                (same metric for all runs -> shared role?)
  + z std / effective rank (collapse), mean KL (VIB activity), delta-size comparison (256 vs 768).

Run: python delta11_rules.py --n 6000 --test_n 700
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
from model import DeltaSystem, D_MODEL, BOS_ID
from delta2_model import DEVICE, MAX_LEN
from delta2_data  import load_edits, group_split
from insertion_cloze_eval import _novel_mask_difflib
from delta6_metric import _std, _ridge, _pool
from transformers import BertTokenizerFast

HARD = {"numeric", "entity", "relational"}
MASK_ID, SEP_ID, CLS_ID = 103, 102, 101
PHR_LEN = 32
RUNS = [("obj1", 768), ("obj2", 768), ("joint", 768), ("joint", 256)]


def changed_phrase(A, B):
    aset = set(A.split())
    return " ".join(w for w in B.split() if w not in aset)


# ───────────────────────── precompute (frozen BERT, once, reused by all runs) ─────────────────────────
@torch.no_grad()
def precompute(ds, pairs, tok, bs=32):
    out = {k: [] for k in ["A_m", "B_m", "P_ids", "P_m", "nov", "eph", "H_A", "H_B", "H_common"]}
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        eA = tok([p["A"] for p in ch], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in ch], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eP = tok([p["phrase"] for p in ch], max_length=PHR_LEN, truncation=True, padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        P_ids, P_m = eP["input_ids"].to(DEVICE), eP["attention_mask"].to(DEVICE)
        nv = np.zeros((len(ch), MAX_LEN), np.float32)
        for k, p in enumerate(ch):
            mask, rl = _novel_mask_difflib(p["A"], p["B"], tok)
            if mask is not None:
                rl = min(rl, MAX_LEN); tm = np.ones(rl, bool)
                if rl > 2: tm[0] = False; tm[rl - 1] = False
                nv[k, :rl] = (mask[:rl] == 1).astype(np.float32) * tm
        nov = torch.tensor(nv, device=DEVICE)
        common_ids = B_ids.clone(); common_ids[nov > 0.5] = MASK_ID            # B with the changed span masked
        H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        H_common = ds._enc(common_ids, B_m)
        H_P = ds._enc(P_ids, P_m)
        pc = (P_m.bool() & (P_ids != CLS_ID) & (P_ids != SEP_ID)).float()
        out["eph"].append(_pool(H_P, pc, P_m.float()).cpu())
        out["A_m"].append(A_m.cpu()); out["B_m"].append(B_m.cpu())
        out["P_ids"].append(P_ids.cpu()); out["P_m"].append(P_m.cpu()); out["nov"].append(nov.cpu())
        out["H_A"].append(H_A.half().cpu()); out["H_B"].append(H_B.half().cpu()); out["H_common"].append(H_common.half().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def take(E, idx):
    return {k: v[idx] for k, v in E.items()}


# ───────────────────────── modules ─────────────────────────
class VIB(nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.mu = nn.Linear(D_MODEL, d_z); self.lv = nn.Linear(D_MODEL, d_z)

    def forward(self, pooled, sample=True):
        mu, lv = self.mu(pooled), self.lv(pooled).clamp(-8, 8)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if sample else mu
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp())                            # [b, d_z]
        return z, kl


def vicreg(z, gamma=1.0):
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(gamma - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, off.pow(2).sum() / z.size(1)


def kl_freebits(kl, C):
    return torch.clamp(kl.mean(0), min=C).sum()                                # free-bits: no penalty below C/dim


def infill_logits(ds, z2mem, H_common, B_m, z, P_ids, P_m):
    b, T = P_ids.shape; dev = P_ids.device
    memory = torch.cat([H_common, z2mem(z).unsqueeze(1)], 1)
    mem_pad = ~torch.cat([B_m, torch.ones(b, 1, device=dev, dtype=B_m.dtype)], 1).bool()
    shifted = torch.full((b, T), BOS_ID, dtype=P_ids.dtype, device=dev); shifted[:, 1:] = P_ids[:, :-1]
    tgt = ds.dr_word_emb(shifted) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
    causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
    dec = ds.dr_decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~P_m.bool(), memory_key_padding_mask=mem_pad)
    return ds.dr_lm(dec)


def gpu(E_b):
    return {k: (v.to(DEVICE).float() if v.dtype == torch.float16 else v.to(DEVICE)) for k, v in E_b.items()}


# ───────────────────────── train one run ─────────────────────────
def train_run(mode, d_z, E_tr, steps, bs, beta, freebits, temp=0.07):
    ds = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    vib = VIB(d_z).to(DEVICE)
    z2mem = nn.Linear(d_z, D_MODEL).to(DEVICE)
    aux = nn.Sequential(nn.Linear(d_z, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL)).to(DEVICE)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    params += list(vib.parameters()) + list(z2mem.parameters()) + list(aux.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0); N = E_tr["H_A"].size(0)
    ds.train(); vib.train(); z2mem.train(); aux.train()
    last = {}
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, N, bs))
        b = gpu(take(E_tr, idx))
        delta = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])[0]
        pooled = _pool(delta, b["nov"], b["B_m"].float())
        z, kl = vib(pooled, sample=True)
        vr, cov = vicreg(z)
        loss = beta * kl_freebits(kl, freebits) + 1.0 * vr + 0.04 * cov
        if mode in ("obj1", "joint"):
            lg = infill_logits(ds, z2mem, b["H_common"], b["B_m"], z, b["P_ids"], b["P_m"])
            loss = loss + F.cross_entropy(lg.reshape(-1, lg.size(-1)), b["P_ids"].reshape(-1), ignore_index=0)
        if mode in ("obj2", "joint"):
            zc = F.normalize(aux(z), dim=-1); e = F.normalize(b["eph"], dim=-1)
            loss = loss + F.cross_entropy(zc @ e.T / temp, torch.arange(bs, device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        last = dict(loss=loss.item(), kl=kl.mean(0).sum().item(), var=vr.item())
        if step % 1000 == 0:
            print(f"      {mode}@{d_z} step {step:>5} | loss {last['loss']:.3f} kl {last['kl']:.2f}")
    return ds, vib, z2mem, aux


@torch.no_grad()
def extract_z(ds, vib, E, idx, bs=64):
    ds.eval(); vib.eval(); idx = torch.as_tensor(idx); reps = []
    for i in range(0, len(idx), bs):
        b = gpu(take(E, idx[i:i + bs]))
        delta = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])[0]
        z, _ = vib(_pool(delta, b["nov"], b["B_m"].float()), sample=False)
        reps.append(z)
    return torch.cat(reps)


@torch.no_grad()
def infill_acc(ds, vib, z2mem, E, idx, ablate, bs=64):
    """teacher-forced span token-acc; ablate=True zeros z (the z-is-used test)."""
    ds.eval(); vib.eval(); z2mem.eval(); idx = torch.as_tensor(idx); cor = tot = 0.0
    for i in range(0, len(idx), bs):
        b = gpu(take(E, idx[i:i + bs]))
        delta = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])[0]
        z, _ = vib(_pool(delta, b["nov"], b["B_m"].float()), sample=False)
        if ablate: z = torch.zeros_like(z)
        lg = infill_logits(ds, z2mem, b["H_common"], b["B_m"], z, b["P_ids"], b["P_m"])
        pred = lg.argmax(-1); m = b["P_m"].bool() & (b["P_ids"] != CLS_ID) & (b["P_ids"] != SEP_ID)
        cor += ((pred == b["P_ids"]) & m).sum().item(); tot += m.sum().item()
    return cor / max(tot, 1)


def evaluate(R_tr, R_te, E_tr, E_te, te_types):
    Xtr, Xte = _std(R_tr, R_te); W = _ridge(Xtr, E_tr)
    pred = torch.cat([Xte, torch.ones(Xte.size(0), 1, device=Xte.device)], 1) @ W

    def top1(mask):
        p = F.normalize(pred[mask], dim=-1); t = F.normalize(E_te[mask], dim=-1)
        sim = p @ t.T
        return ((sim >= sim.diag().unsqueeze(1)).sum(1) == 1).float().mean().item()
    dev = E_te.device
    out = {"overall": top1(torch.ones(E_te.size(0), dtype=torch.bool, device=dev))}
    for t in ["numeric", "entity", "relational"]:
        m = torch.tensor([x == t for x in te_types], device=dev)
        if int(m.sum()) > 5: out[t] = top1(m)
    return out


def eff_rank(z):
    z = z - z.mean(0); s = torch.linalg.svdvals(z); p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))                    # entropy-based effective rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--test_n", type=int, default=700)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--probe_n", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--freebits", type=float, default=0.5)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 100)
    print("DELTA-11 RULES (Phase 1): role-vs-content delta | VIB+VICReg+gap-infill+contrastive | hard edits")
    print(f"device={DEVICE} beta={args.beta} freebits={args.freebits}")
    print("=" * 100)
    edits = load_edits(args.n, tok)
    hard = []
    for e in edits:
        if e.get("type") not in HARD: continue
        ph = changed_phrase(e["A"], e["B"])
        if len(e["A"]) < 20 or len(e["B"]) < 20 or not ph.strip(): continue
        hard.append({"A": e["A"], "B": e["B"], "phrase": ph, "group": e["group"], "type": e["type"]})
    print(f"loaded {len(edits)} -> {len(hard)} hard pairs | types {dict(Counter(h['type'] for h in hard))}")
    tr_pool, te = group_split(hard, test_frac=min(0.4, args.test_n / max(len(hard), 1)))
    te_types = [p["type"] for p in te]
    print(f"train_pool {len(tr_pool)} | fixed test {len(te)} | types {dict(Counter(te_types))}")

    ds0 = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    print("precomputing frozen encodings (once, reused by all runs)...")
    E_tr = precompute(ds0, tr_pool, tok); E_te = precompute(ds0, te, tok)
    del ds0
    import gc; gc.collect(); torch.cuda.empty_cache()
    te_idx = np.arange(len(te)); psamp = np.arange(min(args.probe_n, len(tr_pool)))
    E_str = E_tr["eph"][psamp].to(DEVICE); E_tte = E_te["eph"][te_idx].to(DEVICE)

    res = {}
    for mode, d_z in RUNS:
        tag = f"{mode}@{d_z}"
        print(f"\ntraining {tag} ...")
        ds, vib, z2mem, aux = train_run(mode, d_z, E_tr, args.steps, args.bs, args.beta, args.freebits)
        z_tr = extract_z(ds, vib, E_tr, psamp); z_te = extract_z(ds, vib, E_te, te_idx)
        r = evaluate(z_tr, z_te, E_str, E_tte, te_types)
        r["rank"] = eff_rank(z_te); r["std"] = float(torch.sqrt(z_te.var(0) + 1e-4).mean())
        if mode in ("obj1", "joint"):
            r["infill_z"] = infill_acc(ds, vib, z2mem, E_te, te_idx, ablate=False)
            r["infill_0"] = infill_acc(ds, vib, z2mem, E_te, te_idx, ablate=True)
        res[tag] = r
        del ds, vib, z2mem, aux; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("CONTENT-RETRIEVAL (frozen z, fresh probe) -- same metric for all runs => REUSABILITY test")
    print("-" * 100)
    print(f"{'run':<14}{'overall':>9}{'numeric':>9}{'entity':>9}{'relational':>12}{'z_rank':>8}{'z_std':>8}"
          f"{'infill_z':>10}{'infill_0':>10}")
    print("-" * 100)
    for tag in [f"{m}@{d}" for m, d in RUNS]:
        r = res[tag]
        def c(k): return f"{r[k]:.3f}" if k in r else "-"
        print(f"{tag:<14}{c('overall'):>9}{c('numeric'):>9}{c('entity'):>9}{c('relational'):>12}"
              f"{r['rank']:>8.1f}{r['std']:>8.2f}{c('infill_z'):>10}{c('infill_0'):>10}")
    print("=" * 100)
    print("READ: (E) reusability -> is obj1@768 retrieval ~ obj2@768 retrieval? (shared role)")
    print("      (B) z-is-used   -> infill_z >> infill_0 (decoder needs z; no bypass)")
    print("      collapse check  -> z_rank > 1, z_std healthy. delta-size -> joint@256 vs joint@768.")
    print("=" * 100)


if __name__ == "__main__":
    main()
