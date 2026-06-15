"""
delta2_eval.py -- Delta-2 STEP 3: the measurement battery (held-out, non-circular).

Trains the scaffold under each aux (paraphrase, nli) on a GROUP-SPLIT train set, then measures
every representation on the HELD-OUT test set against baselines/oracle/chance. Unlike the step-2
sanity metrics (the training objective on train data = circular), every metric here is
non-circular and generalization-focused.

Representations (fixed-size test vectors; learned deltas are 256-d, baselines 768-d):
  delta_para  : pooled delta, paraphrase-aux model
  delta_nli   : pooled delta, nli-aux model
  encB        : pool(encode(B))                 [baseline: "just B"]
  meandiff    : pool(B) - pool(A)               [baseline: linear difference]

Content is scored with a LINEAR READOUT (Ridge) rep->gold-novelty fit on train and evaluated on
test -- dimension-agnostic (works for 256-d and 768-d) and non-circular (the "informativeness" /
DCI idea). gold-novelty = mean encode(B) over the difflib-changed tokens.

Battery:
  1. effective rank        -- non-triviality (collapsed reps ~1)
  2. content readout       -- retrieve/score gold novelty via train-fit linear readout; + per type
  3. surface-invariance    -- ||delta|| on HELD-OUT paraphrases vs edits (want para << edit)
  4. A-dependence          -- content drop when A is shuffled (delta must USE A, not be A-blind)
  5. NLI meaning probe     -- rep -> entail/neutral/contradict: test acc + selectivity + MDL bits

Run: python delta2_eval.py --steps 600
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model        import DeltaSystem  # noqa: F401  (shared defs)
from delta2_model import Delta2, encode_all, take, masked_mean, vicreg_loss, DEVICE, MAX_LEN
from delta2_data  import (load_edits, load_validated_paraphrases, load_edit_nli_labels,
                          group_split, change_type, NLI_LABELS)
from insertion_cloze_eval import _novel_mask_difflib
from transformers import BertTokenizerFast


# ── training (same objective as step 2, TRAIN split only) ────────────────────────
def train(aux, E, P, labels, steps, mode="v2", novel_T=None, novel_M=None,
          temp=0.1, lr=1e-4, bs=32, seed=0):
    model = Delta2(aux).to(DEVICE).train()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    N = E["H_A"].size(0)
    lab_t = torch.tensor(labels, device=DEVICE) if labels is not None else None
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, N, bs), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        dv_e = model.delta_from_H(H_A, A_m, H_B, B_m)
        if mode in ("v2", "v3"):
            anc = model.contrastive_anchor_loss(dv_e, H_A, A_m, H_B, B_m, temp)
            var_l, cov_l = vicreg_loss(dv_e)
            anc = anc + 1.0 * var_l + 0.04 * cov_l
        else:
            anc, _, _ = model.anchor_loss(dv_e, H_A, A_m, H_B, B_m)
        if aux == "paraphrase":
            pidx = torch.as_tensor(rng.integers(0, P["H_A"].size(0), bs), device=DEVICE)
            dv_p = model.delta_from_H(*take(P, pidx))
            aloss = dv_p.norm(dim=-1).mean() + F.relu(1.0 - dv_e.norm(dim=-1)).mean()
        else:
            aloss = F.cross_entropy(model.nli_head(dv_e), lab_t[idx])
        loss = anc + aloss
        if mode == "v3" and novel_T is not None:                 # direct gold-novelty supervision
            m = novel_M[idx]
            if m.any():
                nh = model.novel_head(dv_e)[m]
                loss = loss + (1 - F.cosine_similarity(nh, novel_T[idx][m], dim=-1)).mean()
        loss.backward(); opt.step(); opt.zero_grad()
    return model.eval()


# ── representation extractors ────────────────────────────────────────────────────
@torch.no_grad()
def rep_delta(model, E):
    out, n = [], E["H_A"].size(0)
    for i in range(0, n, 64):
        idx = torch.arange(i, min(i + 64, n), device=DEVICE)
        out.append(model.delta_from_H(*take(E, idx)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def rep_pool(E):
    encB = masked_mean(E["H_B"], E["B_m"]).cpu().numpy()
    diff = (masked_mean(E["H_B"], E["B_m"]) - masked_mean(E["H_A"], E["A_m"])).cpu().numpy()
    return encB, diff


@torch.no_grad()
def golds_novel(E, edits, tok):
    """gold = mean encode(B) over difflib-novel tokens; None if no detectable change."""
    HB = E["H_B"].cpu().numpy()
    out = []
    for i, e in enumerate(edits):
        mask, rl = _novel_mask_difflib(e["A"], e["B"], tok)
        if mask is None:
            out.append(None); continue
        rl = min(rl, MAX_LEN); nov = mask[:rl]
        idx = np.where(nov == 1)[0]; idx = idx[(idx > 0) & (idx < rl - 1)]
        out.append(HB[i, idx].mean(0) if len(idx) else None)
    return out


def stack_golds(glist):
    keep = [i for i, g in enumerate(glist) if g is not None]
    return np.stack([glist[i] for i in keep]), np.array(keep, dtype=int)


# ── metrics ──────────────────────────────────────────────────────────────────────
def eff_rank(X):
    X = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    return float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))


def _cos_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a, b


def retrieval(reps, golds):
    a, b = _cos_rows(reps, golds)
    S = a @ b.T; n = len(reps); ranks = (-S).argsort(1)
    top1 = float(np.mean([ranks[i, 0] == i for i in range(n)]))
    mrr = float(np.mean([1.0 / (1 + np.where(ranks[i] == i)[0][0]) for i in range(n)]))
    return top1, mrr


def readout(rep_tr, G_tr, rep_te):
    """linear rep->gold map fit on train (dimension-agnostic content readout)."""
    return Ridge(alpha=1.0).fit(rep_tr, G_tr).predict(rep_te)


def mean_cos(P, G):
    a, b = _cos_rows(P, G)
    return float((a * b).sum(1).mean())


def mdl_codelength(X, y, n_classes, seed=0):
    """Voita & Titov online codelength (bits); lower => more extractable / less memorized."""
    rng = np.random.default_rng(seed)
    o = rng.permutation(len(X)); X, y = X[o], y[o]
    ts = [max(2, int(f * len(X))) for f in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]
    bits = ts[0] * np.log2(n_classes)
    for a, b in zip(ts[:-1], ts[1:]):
        if len(np.unique(y[:a])) < 2:
            bits += (b - a) * np.log2(n_classes); continue
        clf = LogisticRegression(max_iter=300).fit(X[:a], y[:a])
        p = clf.predict_proba(X[a:b]); cls = list(clf.classes_)
        for k, t in enumerate(y[a:b]):
            j = cls.index(t) if t in cls else None
            bits += -np.log2(max(p[k, j] if j is not None else 1.0 / n_classes, 1e-12))
    return float(bits)


def nli_probe(Xtr, ytr, Xte, yte, n_classes=3, seed=0):
    nz = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    Xtr, Xte = nz(Xtr), nz(Xte)
    acc = LogisticRegression(max_iter=500).fit(Xtr, ytr).score(Xte, yte)
    rng = np.random.default_rng(seed)
    ctrl = LogisticRegression(max_iter=500).fit(Xtr, rng.permutation(ytr)).score(Xte, rng.permutation(yte))
    return acc, acc - ctrl, mdl_codelength(Xtr, ytr, n_classes, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_edit", type=int, default=400)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 80)
    print(f"DELTA-2 BATTERY (step 3, held-out)  device={DEVICE}  steps={args.steps}")
    print("=" * 80)

    edits = load_edits(args.n_edit, tok)
    for e in edits:
        e["type"] = change_type(e["A"], e["B"])
    load_edit_nli_labels(args.n_edit, edits)
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

    y_tr = np.array([NLI_LABELS.index(e["nli"]) for e in tr_e])
    y_te = np.array([NLI_LABELS.index(e["nli"]) for e in te_e])
    types_te = np.array([e["type"] for e in te_e])

    # gold-novelty targets on TRAIN (for v3 direct supervision; reused for the content metric)
    gn_tr = golds_novel(E_tr, tr_e, tok)
    dim = next(g for g in gn_tr if g is not None).shape[0]
    novel_T = torch.zeros(len(gn_tr), dim, device=DEVICE)
    novel_M = torch.zeros(len(gn_tr), dtype=torch.bool, device=DEVICE)
    for i, g in enumerate(gn_tr):
        if g is not None:
            novel_T[i] = torch.tensor(g, device=DEVICE); novel_M[i] = True

    print("\nTraining paraphrase v2 (contrastive+VICReg)..."); m_par = train("paraphrase", E_tr, P_tr, None, args.steps, "v2")
    print("Training nli v2...");                                m_nli = train("nli", E_tr, None, list(y_tr), args.steps, "v2")
    print("Training paraphrase v3 (direct gold-novelty supervision)...")
    m_par3 = train("paraphrase", E_tr, P_tr, None, args.steps, "v3", novel_T, novel_M)

    encB_tr, diff_tr = rep_pool(E_tr); encB_te, diff_te = rep_pool(E_te)
    reps_tr = {"para_v2": rep_delta(m_par, E_tr), "nli_v2": rep_delta(m_nli, E_tr),
               "para_v3": rep_delta(m_par3, E_tr), "encB": encB_tr, "meandiff": diff_tr}
    reps_te = {"para_v2": rep_delta(m_par, E_te), "nli_v2": rep_delta(m_nli, E_te),
               "para_v3": rep_delta(m_par3, E_te), "encB": encB_te, "meandiff": diff_te}

    G_tr, k_tr = stack_golds(gn_tr)
    G_te, k_te = stack_golds(golds_novel(E_te, te_e, tok))
    types_k = types_te[k_te]

    # 1. effective rank
    print("\n[1] Effective rank (higher = richer; ~1 = collapsed):")
    for k, V in reps_te.items():
        print(f"    {k:<11} {eff_rank(V):6.1f}")

    # 2. content readout (linear rep->gold), retrieval + cosine, overall + per type
    print("\n[2] Content readout of gold novelty (test). top1 / MRR / cos. chance top1="
          f"{1/len(k_te):.3f}; oracle cos=1.00:")
    for k in reps_te:
        pred = readout(reps_tr[k][k_tr], G_tr, reps_te[k][k_te])
        t1, mr = retrieval(pred, G_te); c = mean_cos(pred, G_te)
        line = f"    {k:<11} {t1:6.3f} {mr:6.3f} {c:6.3f}   per-type cos:"
        for ct in ["insertion", "entity", "relational", "numeric"]:
            m = types_k == ct
            if m.sum() >= 5:
                line += f" {ct[:4]}={mean_cos(pred[m], G_te[m]):.2f}"
        print(line)

    # 3. surface-invariance (held-out paraphrases)
    print("\n[3] Surface-invariance: ||delta|| held-out paraphrases vs edits (want para << edit):")
    for name, mdl in [("para_v2", m_par), ("nli_v2", m_nli), ("para_v3", m_par3)]:
        de = np.linalg.norm(rep_delta(mdl, E_te), axis=1).mean()
        dp = np.linalg.norm(rep_delta(mdl, P_te), axis=1).mean()
        print(f"    {name:<11} edit={de:7.3f}  para={dp:7.3f}  ratio={de/max(dp,1e-6):8.2f}")

    # 4. A-dependence (shuffle A -> content readout should drop)
    print("\n[4] A-dependence: content cos with CORRECT vs SHUFFLED A (drop => delta uses A):")
    perm = np.random.default_rng(0).permutation(E_te["H_A"].size(0))
    E_sh = {"H_A": E_te["H_A"][perm], "A_m": E_te["A_m"][perm], "H_B": E_te["H_B"], "B_m": E_te["B_m"]}
    for name, mdl in [("para_v2", m_par), ("nli_v2", m_nli), ("para_v3", m_par3)]:
        rt = rep_delta(mdl, E_tr); ro = rep_delta(mdl, E_te); rs = rep_delta(mdl, E_sh)
        r = Ridge(alpha=1.0).fit(rt[k_tr], G_tr)
        c_ok = mean_cos(r.predict(ro[k_te]), G_te); c_sh = mean_cos(r.predict(rs[k_te]), G_te)
        print(f"    {name:<11} correctA={c_ok:.3f}  shuffledA={c_sh:.3f}  drop={c_ok - c_sh:+.3f}")

    # 5. NLI meaning probe (acc / selectivity / MDL)
    print("\n[5] NLI meaning probe: acc | selectivity(acc-control) | MDL bits (lower=better). "
          f"chance acc={max(np.bincount(y_te))/len(y_te):.3f}:")
    for k in reps_te:
        acc, sel, mdl = nli_probe(reps_tr[k], y_tr, reps_te[k], y_te)
        print(f"    {k:<11} acc={acc:.3f}  sel={sel:+.3f}  MDL={mdl:7.1f}")

    print("\n" + "=" * 80)
    print("READ: a winning aux beats encB/meandiff on content readout (esp. entity/relational),")
    print("keeps para<<edit on HELD-OUT paraphrases, DROPS under shuffled-A, and (nli) gives high")
    print("NLI selectivity + low MDL. All non-circular -- the step-2 circular metrics are gone.")
    print("=" * 80)


if __name__ == "__main__":
    main()
