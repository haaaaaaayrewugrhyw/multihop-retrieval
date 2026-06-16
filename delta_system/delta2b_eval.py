"""
delta2b_eval.py -- STEP 5: held-out evaluation of the single token-level multi-objective model.
See DELTA2_BUILD_SPEC.md. The make-or-break: does ONE trained Delta2B match/beat the two-stage
baseline (0.82 e2e) on HELD-OUT data?

Train Delta2B on TRAIN split (5 objectives, uncertainty weighting), then on HELD-OUT test measure:
  1 content retrieval (model vs complement 0.76 / encB / oracle), stratified by change-type
  2 gate / surface-robustness: AUC(edit vs syn+non-meaning) by the model's gate (||content||)
  3 A-dependence: content retrieval drop under shuffled A
  4 A-leak: linear probe content->pool(A) cosine (lower = cleaner)
  5 END-TO-END: gate(threshold from train) AND content top-1 correct, over test edits
Multi-seed (Locatello warning): report mean +/- std of the headline numbers.

Run: python delta2b_eval.py --n 400 --steps 400 --seeds 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model         import DeltaSystem  # noqa: F401
from delta2_model  import encode_all, take, masked_mean, vicreg_loss, DEVICE, MAX_LEN
from delta2b_model import Delta2B, routed_content, pad_masks, cosloss
from delta2b_data  import build
from delta2_data   import group_split
from delta2_token_battery import op
from transformers  import BertTokenizerFast

TYPES = ["insertion", "entity", "relational", "numeric"]


def gold_of(E, M):
    w = M / M.sum(1, keepdim=True).clamp(min=1e-6)
    return (E["H_B"] * w.unsqueeze(-1)).sum(1)


@torch.no_grad()
def content_pred(model, E):                       # routed by PREDICTED gate (realistic, no true mask)
    out, n = [], E["H_A"].size(0)
    for i in range(0, n, 64):
        idx = torch.arange(i, min(i + 64, n), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        de = model.delta(H_A, A_m, H_B, B_m)
        g = torch.sigmoid(model.rtd(de).squeeze(-1))
        out.append(routed_content(de, g, B_m).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def content_truemask(model, E, M):                # routed by the TRUE difflib mask (isolates delta from gate)
    out, n = [], E["H_A"].size(0)
    for i in range(0, n, 64):
        idx = torch.arange(i, min(i + 64, n), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        de = model.delta(H_A, A_m, H_B, B_m)
        out.append(routed_content(de, M[idx], B_m).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def comp_content(E):                              # fixed complement (the 0.76 baseline)
    out, n = [], E["H_A"].size(0)
    for i in range(0, n, 64):
        idx = torch.arange(i, min(i + 64, n), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        comp, g = op(H_A, H_B, A_m, 0.1)
        out.append(routed_content(comp, g, B_m).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def enc_b(E):
    out, n = [], E["H_A"].size(0)
    for i in range(0, n, 64):
        idx = torch.arange(i, min(i + 64, n), device=DEVICE)
        out.append(masked_mean(E["H_B"][idx], E["B_m"][idx]).cpu().numpy())
    return np.concatenate(out)


def retr(reps, golds):
    R = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-9)
    G = golds / (np.linalg.norm(golds, axis=1, keepdims=True) + 1e-9)
    S = R @ G.T; n = len(R); ranks = (-S).argsort(1)
    correct = np.array([ranks[i, 0] == i for i in range(n)])
    mrr = float(np.mean([1.0 / (1 + np.where(ranks[i] == i)[0][0]) for i in range(n)]))
    return correct.mean(), mrr, correct


def train_one(E, S, M, gold, steps, seed, bs=24):
    model = Delta2B().to(DEVICE).train()
    log_vars = nn.Parameter(torch.zeros(5, device=DEVICE))
    opt = torch.optim.Adam(model.gen_params() + [log_vars], lr=1e-4)
    rng = np.random.default_rng(seed)
    Ne, Ns = E["H_A"].size(0), S["H_A"].size(0)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, Ne, bs), device=DEVICE)
        sidx = torch.as_tensor(rng.integers(0, Ns, bs), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        de = model.delta(H_A, A_m, H_B, B_m); rtd_e = model.rtd(de).squeeze(-1); m = M[idx]; bm = B_m.float()
        c_true = routed_content(de, m, B_m)
        L_anchor = cosloss(model.anchor(torch.cat([masked_mean(H_A, A_m), c_true], -1)), masked_mean(H_B, B_m))
        L_rtd = (F.binary_cross_entropy_with_logits(rtd_e, m, reduction="none") * bm).sum() / bm.sum()
        L_content = cosloss(c_true, gold[idx])
        v, cov = vicreg_loss(c_true); L_vic = v + 0.04 * cov
        c_edit = routed_content(de, torch.sigmoid(rtd_e), B_m)
        sH_A, sA_m, sH_B, sB_m = take(S, sidx); ds = model.delta(sH_A, sA_m, sH_B, sB_m)
        c_syn = routed_content(ds, torch.sigmoid(model.rtd(ds).squeeze(-1)), sB_m)
        L_inv = c_syn.norm(dim=-1).mean() + F.relu(1.0 - c_edit.norm(dim=-1)).mean()
        Ls = [L_anchor, L_rtd, L_content, L_inv, L_vic]
        total = sum(torch.exp(-log_vars[i]) * Ls[i] + log_vars[i] for i in range(5))
        opt.zero_grad(); total.backward(); opt.step()
    return model.eval()


def f1_threshold(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    best = (-1, s.mean())
    for t in np.unique(s):
        p = s >= t; tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best[0]:
            best = (f, t)
    return best[1]


def run_seed(seed, tok, d):
    edits, non, syn = d["edits"], d["non"], d["syn"]
    edits = [e for e, k in zip(edits, (pad_masks(edits).sum(1) > 0).cpu().numpy()) if k]
    tr_e, te_e = group_split(edits, test_frac=0.25, seed=seed)
    tr_s, te_s = group_split(syn,  test_frac=0.25, seed=seed)
    tr_n, te_n = group_split(non,  test_frac=0.25, seed=seed)

    base = Delta2B().to(DEVICE).eval()
    E_tr = encode_all(base, tr_e, "A", "B", tok); E_te = encode_all(base, te_e, "A", "B", tok)
    S_tr = encode_all(base, tr_s, "A", "A_syn", tok); S_te = encode_all(base, te_s, "A", "A_syn", tok)
    N_te = encode_all(base, te_n, "A", "A_para", tok)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    M_tr, M_te = pad_masks(tr_e), pad_masks(te_e)
    gold_tr, gold_te = gold_of(E_tr, M_tr), gold_of(E_te, M_te)
    types_te = np.array([e["type"] for e in te_e])

    model = train_one(E_tr, S_tr, M_tr, gold_tr, args_steps, seed)

    # 1 content retrieval
    g = gold_te.cpu().numpy()
    m_top, _, m_corr = retr(content_pred(model, E_te), g)
    mt_top, _, _ = retr(content_truemask(model, E_te, M_te), g)   # delta with TRUE mask (gate isolated)
    c_top, _, _ = retr(comp_content(E_te), g)
    e_top, _, _ = retr(enc_b(E_te), g)
    o_top, _, _ = retr(g, g)
    per = {}
    cp = content_pred(model, E_te)
    for ct in TYPES:
        mk = types_te == ct
        if mk.sum() >= 5:
            per[ct] = retr(cp[mk], g[mk])[0]

    # 2 gate AUC (edit vs syn+non) by ||content||
    sc_e = np.linalg.norm(content_pred(model, E_te), axis=1)
    sc_neg = np.r_[np.linalg.norm(content_pred(model, S_te), axis=1),
                   np.linalg.norm(content_pred(model, N_te), axis=1)]
    auc = roc_auc_score(np.r_[np.ones(len(sc_e)), np.zeros(len(sc_neg))], np.r_[sc_e, sc_neg])

    # 3 A-dependence (shuffle A)
    perm = np.random.default_rng(seed).permutation(E_te["H_A"].size(0))
    E_sh = {"H_A": E_te["H_A"][perm], "A_m": E_te["A_m"][perm], "H_B": E_te["H_B"], "B_m": E_te["B_m"]}
    sh_top, _, _ = retr(content_pred(model, E_sh), g)

    # 4 A-leak: linear probe content -> pool(A)
    cont_tr = content_pred(model, E_tr); poolA_tr = masked_mean(E_tr["H_A"], E_tr["A_m"]).cpu().numpy()
    poolA_te = masked_mean(E_te["H_A"], E_te["A_m"]).cpu().numpy()
    pred_A = Ridge(alpha=1.0).fit(cont_tr, poolA_tr).predict(cp)
    aleak = float(np.mean([np.dot(pred_A[i], poolA_te[i]) /
                           (np.linalg.norm(pred_A[i]) * np.linalg.norm(poolA_te[i]) + 1e-9)
                           for i in range(len(cp))]))

    # 5 end-to-end: gate threshold from train, then gated AND content-correct
    sc_e_tr = np.linalg.norm(cont_tr, axis=1)
    sc_neg_tr = np.linalg.norm(content_pred(model, S_tr), axis=1)   # train syn = preserve negatives
    thr = f1_threshold(sc_e_tr, sc_neg_tr)
    gated = sc_e >= thr
    e2e = (gated & m_corr).mean()

    return dict(m_top=m_top, mt_top=mt_top, c_top=c_top, e_top=e_top, o_top=o_top, per=per,
                auc=auc, sh_top=sh_top, drop=m_top - sh_top, aleak=aleak, e2e=e2e,
                n_te=len(te_e))


def main():
    global args_steps
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    args_steps = args.steps
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 82)
    print(f"DELTA-2B HELD-OUT EVAL (single model)  device={DEVICE}  steps={args.steps}  seeds={args.seeds}")
    print("=" * 82)
    d = build(args.n)
    res = [run_seed(s, tok, d) for s in range(args.seeds)]

    def ms(key):
        v = np.array([r[key] for r in res]); return v.mean(), v.std()

    print(f"\nheld-out test edits ~{res[0]['n_te']} | seeds {args.seeds}\n")
    print("[1] content retrieval top1 (vs baselines):")
    for k, label in [("m_top", "model (pred gate)"), ("mt_top", "model (TRUE mask)"),
                     ("c_top", "complement 0.76"), ("e_top", "encB"), ("o_top", "oracle")]:
        mu, sd = ms(k); print(f"    {label:<18} {mu:.3f} +/- {sd:.3f}")
    print("    per-type (model):", {ct: round(np.mean([r['per'].get(ct, np.nan) for r in res]), 2) for ct in TYPES})

    print("\n[2] gate AUC edit-vs-(syn+non), by ||content||:")
    mu, sd = ms("auc"); print(f"    model gate AUC {mu:.3f} +/- {sd:.3f}   (complement magnitude was 0.43; pooled-delta gate 0.806)")

    print("\n[3] A-dependence (content top1 drop under shuffled A):")
    mu, sd = ms("drop"); print(f"    drop {mu:+.3f} +/- {sd:.3f}   (>0 = uses A)")

    print("\n[4] A-leak (content->pool(A) cosine; lower=cleaner):")
    mu, sd = ms("aleak"); print(f"    {mu:.3f} +/- {sd:.3f}")

    print("\n[5] END-TO-END (gated AS changed AND content top-1), test edits:")
    mu, sd = ms("e2e"); print(f"    {mu:.3f} +/- {sd:.3f}   (two-stage baseline = 0.82)")

    print("\n" + "=" * 82)
    print("READ: single-model content top1 >= complement (0.76) AND gate AUC >> 0.43 AND e2e ~ 0.82")
    print("=> one architecture matches the two-stage pipeline (the goal). Else fall back to two-stage.")
    print("=" * 82)


if __name__ == "__main__":
    main()
