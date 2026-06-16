"""
delta2b_model.py -- STEP 2 of the build spec: the token-level multi-objective architecture.
See DELTA2_BUILD_SPEC.md.

ARCHITECTURE (uncompressed, token-level):
  frozen BERT -> generate_delta (subtraction backbone) -> delta [b,T,768]   (NOT pooled)
  heads: RTD (per-token novelty logit) ; anchor (MLP[pool(A), content] -> predict pool(B))
  content = ROUTED pool of delta over novel positions (PUSH by routing: delta only drives novelty)

OBJECTIVES (the user's sub-objectives; balanced by learned uncertainty weights):
  1 anchor   : 1 - cos(anchor([pool A, content]), pool B)        (content-rich; routed)
  2 rtd      : BCE(per-token novelty, difflib mask)              (PULL: localize novelty)
  3 content  : 1 - cos(routed-content(true mask), gold novelty)  (PULL: carry the new content)
  4 invar    : ||content(syn)|| small & ||content(edit)|| >=1    (gate/meaning; under PREDICTED routing)
  5 vicreg   : variance+covariance on content across batch       (anti-collapse, uncompressed)

This file SANITY-TRAINS: confirms losses move, RTD learns, content matches gold, invariance separates,
and prints the PULL-vs-PUSH gradient-cosine diagnostic (the tragic-triad check).

Run: python delta2b_model.py --n 400 --steps 300
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
from model        import DeltaSystem, D_MODEL
from delta2_model import encode_all, take, masked_mean, vicreg_loss, DEVICE, MAX_LEN
from delta2b_data import build
from transformers import BertTokenizerFast


class Delta2B(nn.Module):
    def __init__(self):
        super().__init__()
        self.core   = DeltaSystem()                       # frozen BERT + subtraction generator
        self.rtd    = nn.Linear(D_MODEL, 1)               # per-token novelty logit
        self.anchor = nn.Sequential(nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(),
                                    nn.Linear(D_MODEL, D_MODEL))

    @torch.no_grad()
    def encode(self, ids, m):
        return self.core._enc(ids, m)

    def delta(self, H_A, A_m, H_B, B_m):
        d, _d0, _a = self.core.generate_delta(H_A, A_m, H_B, B_m)
        return d                                           # [b,T,768] token-level, uncompressed

    def gen_params(self):
        ps = [p for n, p in self.core.named_parameters() if n.startswith("g_")]
        return ps + list(self.rtd.parameters()) + list(self.anchor.parameters())


def routed_content(delta, gate, mask):
    """pool delta over positions weighted by `gate` (true novel mask or predicted prob)."""
    w = gate * mask.float()
    w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
    return (delta * w.unsqueeze(-1)).sum(1)                # [b,768]


def pad_masks(edits):
    M = np.zeros((len(edits), MAX_LEN), np.float32)
    for i, e in enumerate(edits):
        rl = min(e["real_len"], MAX_LEN); nov = e["novel_mask"][:rl]
        tm = np.ones(rl, dtype=bool)
        if rl > 2:
            tm[0] = False; tm[rl - 1] = False
        M[i, :rl] = (nov == 1).astype(np.float32) * tm
    return torch.tensor(M, device=DEVICE)


def cosloss(a, b):
    return (1 - F.cosine_similarity(a, b, dim=-1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=24)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 80)
    print(f"DELTA-2B ARCHITECTURE SANITY-TRAIN  device={DEVICE}")
    print("=" * 80)
    d = build(args.n)
    edits = [e for e in d["edits"]]
    syn = d["syn"]
    # keep edits that have at least one novel token after CLS/SEP trim
    M_all = pad_masks(edits)
    keep = (M_all.sum(1) > 0).cpu().numpy()
    edits = [e for e, k in zip(edits, keep) if k]
    print(f"edits with novel target {len(edits)} | syn {len(syn)}")

    model = Delta2B().to(DEVICE).train()
    print("encoding (frozen BERT, one-time)...")
    E = encode_all(model, edits, "A", "B", tok)
    S = encode_all(model, syn, "A", "A_syn", tok)
    M = pad_masks(edits)
    Bw = M / M.sum(1, keepdim=True).clamp(min=1e-6)
    gold = (E["H_B"] * Bw.unsqueeze(-1)).sum(1)            # gold novelty per edit [N,768]

    log_vars = nn.Parameter(torch.zeros(5, device=DEVICE))  # uncertainty weighting (Kendall)
    opt = torch.optim.Adam(model.gen_params() + [log_vars], lr=1e-4)
    rng = np.random.default_rng(0)
    Ne, Ns = E["H_A"].size(0), S["H_A"].size(0)

    def batch_losses(idx, sidx):
        H_A, A_m, H_B, B_m = take(E, idx)
        de = model.delta(H_A, A_m, H_B, B_m)
        rtd_e = model.rtd(de).squeeze(-1)
        m = M[idx]
        # 1 anchor (content from TRUE mask)
        c_true = routed_content(de, m, B_m)
        pred = model.anchor(torch.cat([masked_mean(H_A, A_m), c_true], -1))
        L_anchor = cosloss(pred, masked_mean(H_B, B_m))
        # 2 rtd
        bm = B_m.float()
        L_rtd = (F.binary_cross_entropy_with_logits(rtd_e, m, reduction="none") * bm).sum() / bm.sum()
        # 3 content
        L_content = cosloss(c_true, gold[idx])
        # 5 vicreg (anti-collapse on content across batch)
        v, cov = vicreg_loss(c_true)
        L_vic = v + 0.04 * cov
        # 4 invariance (PREDICTED routing): edit large, syn small
        c_edit_pred = routed_content(de, torch.sigmoid(rtd_e), B_m)
        sH_A, sA_m, sH_B, sB_m = take(S, sidx)
        ds = model.delta(sH_A, sA_m, sH_B, sB_m)
        gate_s = torch.sigmoid(model.rtd(ds).squeeze(-1))
        c_syn = routed_content(ds, gate_s, sB_m)
        L_inv = c_syn.norm(dim=-1).mean() + F.relu(1.0 - c_edit_pred.norm(dim=-1)).mean()
        return [L_anchor, L_rtd, L_content, L_inv, L_vic]

    @torch.no_grad()
    def report(tag):
        model.eval()
        idx = torch.arange(min(96, Ne), device=DEVICE)
        sidx = torch.arange(min(96, Ns), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        de = model.delta(H_A, A_m, H_B, B_m); rtd_e = model.rtd(de).squeeze(-1)
        m = M[idx]; bm = B_m.bool()
        rtd_acc = (((rtd_e > 0) == (m > 0.5)) & bm).float().sum() / bm.float().sum()
        c_true = routed_content(de, m, B_m)
        ccos = F.cosine_similarity(c_true, gold[idx], dim=-1).mean()
        edit_norm = routed_content(de, torch.sigmoid(rtd_e), B_m).norm(dim=-1).mean()
        ds = model.delta(*take(S, sidx)); gate_s = torch.sigmoid(model.rtd(ds).squeeze(-1))
        syn_norm = routed_content(ds, gate_s, take(S, sidx)[3]).norm(dim=-1).mean()
        model.train()
        print(f"  {tag}: rtd_acc {rtd_acc:.3f} | content_cos {ccos:.3f} | "
              f"edit||c|| {edit_norm:.3f} syn||c|| {syn_norm:.3f} sep {edit_norm/max(syn_norm,1e-6):.2f}")

    report("BEFORE")
    names = ["anchor", "rtd", "content", "inv", "vic"]
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(rng.integers(0, Ne, args.bs), device=DEVICE)
        sidx = torch.as_tensor(rng.integers(0, Ns, args.bs), device=DEVICE)
        Ls = batch_losses(idx, sidx)
        total = sum(torch.exp(-log_vars[i]) * Ls[i] + log_vars[i] for i in range(5))
        opt.zero_grad(); total.backward(); opt.step()
        if step % 50 == 0:
            print(f"  step {step:>4} | " + " ".join(f"{names[i]} {Ls[i].item():.3f}" for i in range(5)))
    report("AFTER ")

    # PULL-vs-PUSH gradient-conflict diagnostic (tragic triad check)
    idx = torch.as_tensor(rng.integers(0, Ne, args.bs), device=DEVICE)
    sidx = torch.as_tensor(rng.integers(0, Ns, args.bs), device=DEVICE)
    Ls = batch_losses(idx, sidx)
    gp = model.gen_params()

    def flat_grad(L):
        gs = torch.autograd.grad(L, gp, retain_graph=True, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                          for g, p in zip(gs, gp)])

    cos_ci = F.cosine_similarity(flat_grad(Ls[2]), flat_grad(Ls[3]), dim=0).item()

    print("\n" + "=" * 80)
    print(f"PULL(content) vs gate(inv) gradient cosine = {cos_ci:+.3f}  "
          f"(<0 = conflicting => PCGrad needed; >=0 = cooperative)")
    print("SANITY: rtd_acc up, content_cos up, sep(edit/syn)>1 => architecture + objectives learn.")
    w = torch.exp(-log_vars).detach()
    print("learned uncertainty weights exp(-log_var):",
          {names[i]: round(float(w[i]), 2) for i in range(5)})
    print("=" * 80)


if __name__ == "__main__":
    main()
