"""
Test the structural-bias idea (cluster -> additive score bias, NOT in Q/K/V).

    baseline     : plain RoPE transformer (no cluster)
    struct       : structural-bias attention, TASK LOSS ONLY
    struct_aux   : structural-bias attention, TASK LOSS + unsupervised clustering objective

The open question (from the research): the bias gives clusters a channel to act,
but does the model USE it? Reasons it broke before say a task loss alone won't make
clusters meaningful -- you need pressure. struct vs struct_aux tests exactly that.

Read-out: count-acc, source-recovery ARI (does the structural bias make clusters
track the latent sources?), soft metrics, viz.

    python train_struct.py --smoke / --quick / --seeds 3 / --analyze
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from data_deint import make, max_len_for, n_classes_for, VOCAB, PAD
from model_struct import StructNet, aux_cluster_loss, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
RESFILE = os.path.join(RESULTS, "struct.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=128, n_layers=4, heads=4, K=5, L_min=4, L_max=8, S_min=2,
           train_size=20000, epochs=40, batch=128, lr=3e-4, aux_w=0.3)

# position mode x auxiliary objective:
#   baseline       - RoPE, no cluster
#   struct         - RoPE position (independent),        cluster bias, TASK only
#   struct_aux     - RoPE position (independent),        cluster bias, TASK + aux
#   struct_pv      - cluster-coupled position,           cluster bias, TASK only
#   struct_pv_aux  - cluster-coupled position,           cluster bias, TASK + aux
ALL_MODELS = ["baseline", "struct", "struct_aux", "struct_pv", "struct_pv_aux"]
POS_MODE = {"baseline": "rope", "struct": "rope", "struct_aux": "rope",
            "struct_pv": "cluster", "struct_pv_aux": "cluster"}


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(name, cfg, max_len, n_classes):
    return StructNet(VOCAB, cfg["d"], cfg["n_layers"], cfg["heads"], cfg["K"],
                     n_classes, max_len, use_cluster=(name != "baseline"),
                     pos_mode=POS_MODE[name], pad_id=PAD)


@torch.no_grad()
def accuracy(model, t, y, batch=512):
    model.eval()
    xt = torch.tensor(t).to(DEVICE)
    preds = [model(xt[i:i + batch]).argmax(1).cpu() for i in range(0, len(t), batch)]
    return float((torch.cat(preds).numpy() == y).mean())


@torch.no_grad()
def source_metrics(model, t, src, batch=512):
    model.eval()
    xt = torch.tensor(t).to(DEVICE)
    Ps = []
    for i in range(0, len(t), batch):
        _ = model(xt[i:i + batch])
        Ps.append(model._last_P.cpu().numpy())
    P = np.concatenate(Ps, 0)
    ari, eff, conc, nmi = [], [], [], []
    for i in range(len(t)):
        m = src[i] != -1
        if m.sum() < 2 or len(set(src[i][m].tolist())) < 2:
            continue
        Pi, si = P[i][m], src[i][m]
        ari.append(adjusted_rand_score(si, Pi.argmax(-1)))
        mass = Pi.mean(0); mass = mass / (mass.sum() + 1e-9)
        eff.append(float(np.exp(-(mass * np.log(mass + 1e-9)).sum())))
        srcs = sorted(set(si.tolist()))
        J = np.stack([Pi[si == c].mean(0) for c in srcs]); J = J / (J.sum(1, keepdims=True) + 1e-9)
        conc.append(float(J.max(1).mean()))
        Jj = J * np.array([(si == c).sum() for c in srcs])[:, None]; Jj = Jj / (Jj.sum() + 1e-9)
        ps, pk = Jj.sum(1, keepdims=True), Jj.sum(0, keepdims=True)
        mi = (Jj * (np.log(Jj + 1e-12) - np.log(ps + 1e-12) - np.log(pk + 1e-12))).sum()
        Hs = -(ps * np.log(ps + 1e-12)).sum(); Hk = -(pk * np.log(pk + 1e-12)).sum()
        nmi.append(float(mi / (min(Hs, Hk) + 1e-9)))
    return (round(np.mean(ari), 3), round(np.mean(eff), 3),
            round(np.mean(conc), 3), round(np.mean(nmi), 3))


def viz(model, t, src, n=5):
    model.eval()
    with torch.no_grad():
        _ = model(torch.tensor(t[:n]).to(DEVICE))
    pred = model._last_P.argmax(-1).cpu().numpy()
    print("\n  token-by-token  (true source  vs  assigned cluster):")
    for i in range(n):
        m = src[i] != -1
        print(f"    ex{i}  src   {''.join(str(x) for x in src[i][m].tolist())}")
        print(f"          clus  {''.join(str(x) for x in pred[i][m].tolist())}")


def run(name, seed, cfg):
    set_seed(seed)
    ml = max_len_for(cfg["K"], cfg["L_max"])
    ncls = n_classes_for(cfg["K"], cfg["S_min"])
    tr_t, tr_s, tr_y, _ = make(cfg["train_size"], cfg["K"], cfg["L_min"], cfg["L_max"], cfg["S_min"], seed=seed)
    te_t, te_s, te_y, _ = make(3000, cfg["K"], cfg["L_min"], cfg["L_max"], cfg["S_min"], seed=90001)
    model = build(name, cfg, ml, ncls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    tt, ty = torch.tensor(tr_t), torch.tensor(tr_y)
    use_aux = name.endswith("aux")

    t0 = time.time()
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_size"])
        for i in range(0, cfg["train_size"], cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            x, y = tt[idx].to(DEVICE), ty[idx].to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if use_aux:
                loss = loss + cfg["aux_w"] * aux_cluster_loss(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    rec = dict(model=name, seed=seed, acc=round(accuracy(model, te_t, te_y), 4),
               params=count_params(model), sec=round(time.time() - t0, 1))
    if name != "baseline":
        rec["src_ari"], rec["eff_clu"], rec["conc"], rec["nmi"] = source_metrics(model, te_t, te_s)
    with open(RESFILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    extra = (f"  src-ARI={rec['src_ari']:.3f} eff={rec['eff_clu']}/{cfg['K']} "
             f"conc={rec['conc']} nmi={rec['nmi']}") if "src_ari" in rec else ""
    print(f"  {name:11s} seed={seed} count-acc={rec['acc']:.3f}{extra}  ({rec['params']:,}p {rec['sec']}s)")
    if name == "struct_aux" and seed == 0:
        viz(model, te_t, te_s)
    return rec


def analyze():
    rows = [json.loads(l) for l in open(RESFILE) if l.strip()]
    agg = defaultdict(lambda: defaultdict(list)); params = {}
    for r in rows:
        for kk in ("acc", "src_ari", "eff_clu", "conc", "nmi"):
            if kk in r:
                agg[r["model"]][kk].append(r[kk])
        params[r["model"]] = r["params"]

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        return m, sd

    print("\n=== structural-bias attention (cluster -> score bias; de-interleaving) ===")
    print(f"{'model':<12}{'params':>10}{'count-acc':>13}{'src-ARI':>10}{'eff_clu':>9}{'conc':>7}{'nmi':>7}")
    for m in ALL_MODELS:
        if m not in agg:
            continue
        a = ms(agg[m]["acc"])
        cells = f"{a[0]*100:>8.2f}±{a[1]*100:<3.1f}"
        for kk in ("src_ari", "eff_clu", "conc", "nmi"):
            cells += f"{ms(agg[m][kk])[0]:>{10 if kk=='src_ari' else (9 if kk=='eff_clu' else 7)}.3f}" if agg[m][kk] else f"{'-':>9}"
        print(f"{m:<12}{params[m]:>10,}{cells}")
    print("\nRead: does struct_aux lift src-ARI above struct (pressure makes clusters meaningful)?"
          "\n      does either beat baseline on count-acc?")


def smoke(cfg):
    ml, ncls = max_len_for(cfg["K"], cfg["L_max"]), n_classes_for(cfg["K"], cfg["S_min"])
    t, s, y, ns = make(8, cfg["K"], cfg["L_min"], cfg["L_max"], cfg["S_min"], seed=0)
    print(f"smoke: S={sorted(set(ns.tolist()))}, max_len={ml}, n_classes={ncls}")
    x = torch.tensor(t)
    for name in ALL_MODELS:
        model = build(name, cfg, ml, ncls)
        out = model(x)
        assert out.shape == (8, ncls), out.shape
        loss = F.cross_entropy(out, torch.tensor(y))
        if name == "struct_aux":
            loss = loss + cfg["aux_w"] * aux_cluster_loss(model)
        loss.backward()
        print(f"  {name:11s} out{tuple(out.shape)} params={count_params(model):,} "
              f"nan={bool(torch.isnan(out).any())}")
    print("  SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.smoke:
        smoke(cfg); return
    if args.analyze:
        analyze(); return
    if args.quick:
        cfg = dict(cfg, epochs=12, train_size=4000)
        models, seeds = ALL_MODELS, [0]
    else:
        models, seeds = ALL_MODELS, list(range(args.seeds))
    print(f"=== device={DEVICE} | models={models} | seeds={seeds} | "
          f"d={cfg['d']} L={cfg['n_layers']} K={cfg['K']} ep={cfg['epochs']} aux_w={cfg['aux_w']} ===")
    for m in models:
        for s in seeds:
            run(m, s, cfg)
    analyze()


if __name__ == "__main__":
    main()
