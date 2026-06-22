"""
Harness for the EM-attention architecture vs the baseline, on variable-length
grouped digit-sentences.

    baseline      : standard transformer + segment embedding (from model.py)
    em_fixed      : EM-attention (cluster centers, E/M steps, bottleneck), fixed pos
    em_varpos     : EM-attention, variable position
    baseline_wide : wider baseline, param-match (--wide)

Absent-sentence logits masked via n_sentences (well-posed K-way head as S varies).

    python train_em.py --smoke
    python train_em.py --quick
    python train_em.py --seeds 3
    python train_em.py --analyze
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

from data import make, max_len_for, VOCAB, PAD
from model import FaithfulArch, count_params
from model_em import EMArch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
RESFILE = os.path.join(RESULTS, "em.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=128, n_layers=4, heads=4, K=5, L_min=2, L_max=7,
           train_size=20000, epochs=40, batch=128, lr=3e-4)

EM_MODELS = {"em_fixed": dict(variable_pos=False), "em_varpos": dict(variable_pos=True)}


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(name, cfg, max_len):
    if name == "baseline":
        return FaithfulArch(VOCAB, cfg["d"], cfg["n_layers"], cfg["heads"], cfg["K"],
                            max_len, cluster_mode="none", use_segment=True, pad_id=PAD)
    if name == "baseline_wide":
        return FaithfulArch(VOCAB, 148, cfg["n_layers"], cfg["heads"], cfg["K"],
                            max_len, cluster_mode="none", use_segment=True, pad_id=PAD)  # ~match em_fixed 1.33M
    return EMArch(VOCAB, cfg["d"], cfg["n_layers"], cfg["heads"], cfg["K"], max_len,
                  pad_id=PAD, **EM_MODELS[name])


def mask_logits(logits, n_sent):
    K = logits.size(1)
    absent = torch.arange(K, device=logits.device).unsqueeze(0) >= n_sent.unsqueeze(1)
    return logits.masked_fill(absent, float("-inf"))


@torch.no_grad()
def accuracy(model, t, s, y, ns, batch=512):
    model.eval()
    xt, st, nt = (torch.tensor(t).to(DEVICE), torch.tensor(s).to(DEVICE),
                  torch.tensor(ns).to(DEVICE))
    preds = []
    for i in range(0, len(t), batch):
        lo = mask_logits(model(xt[i:i + batch], st[i:i + batch]), nt[i:i + batch])
        preds.append(lo.argmax(1).cpu())
    return float((torch.cat(preds).numpy() == y).mean())


@torch.no_grad()
def cluster_recovery(model, t, s, batch=512):
    model.eval(); model.collect_P = True
    xt, st = torch.tensor(t).to(DEVICE), torch.tensor(s).to(DEVICE)
    preds = []
    for i in range(0, len(t), batch):
        _ = model(xt[i:i + batch], st[i:i + batch])
        preds.append(model._last_P.argmax(-1).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    aris = []
    for i in range(len(t)):
        m = s[i] != -1
        if m.sum() >= 2 and len(set(s[i][m].tolist())) >= 2:
            aris.append(adjusted_rand_score(s[i][m], pred[i][m]))
    return float(np.mean(aris))


def viz(model, t, s, n=4):
    model.eval(); model.collect_P = True
    with torch.no_grad():
        _ = model(torch.tensor(t[:n]).to(DEVICE), torch.tensor(s[:n]).to(DEVICE))
    pred = model._last_P.argmax(-1).cpu().numpy()
    print("\n  token-by-token  (true sentence  vs  assigned cluster):")
    for i in range(n):
        m = s[i] != -1
        print(f"    ex{i}  sent  {''.join(str(x) for x in s[i][m].tolist())}")
        print(f"          clus  {''.join(str(x) for x in pred[i][m].tolist())}")


def run(name, seed, cfg):
    set_seed(seed)
    K, ml = cfg["K"], max_len_for(cfg["K"], cfg["L_max"])
    tr = make(cfg["train_size"], K, cfg["L_min"], cfg["L_max"], seed=seed)
    te = make(3000, K, cfg["L_min"], cfg["L_max"], seed=90001)
    tr_t, tr_s, tr_y, tr_n, _ = tr
    te_t, te_s, te_y, te_n, _ = te
    model = build(name, cfg, ml).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    tt, ts, ty, tn = (torch.tensor(tr_t), torch.tensor(tr_s),
                      torch.tensor(tr_y), torch.tensor(tr_n))
    bs = cfg["batch"]

    t0 = time.time()
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_size"])
        for i in range(0, cfg["train_size"], bs):
            idx = perm[i:i + bs]
            x, sgi, y, ns = (tt[idx].to(DEVICE), ts[idx].to(DEVICE),
                             ty[idx].to(DEVICE), tn[idx].to(DEVICE))
            opt.zero_grad()
            F.cross_entropy(mask_logits(model(x, sgi), ns), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    rec = dict(model=name, seed=seed,
               acc=round(accuracy(model, te_t, te_s, te_y, te_n), 4),
               params=count_params(model), sec=round(time.time() - t0, 1))
    if name.startswith("em"):
        rec["ari"] = round(cluster_recovery(model, te_t, te_s), 4)
    with open(RESFILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    extra = f"  cluster-ARI={rec['ari']:.3f}" if "ari" in rec else ""
    print(f"  {name:13s} seed={seed} acc={rec['acc']:.3f}{extra}  ({rec['params']:,}p {rec['sec']}s)")
    if name == "em_fixed" and seed == 0:
        viz(model, te_t, te_s)
    return rec


def analyze():
    rows = [json.loads(l) for l in open(RESFILE) if l.strip()]
    agg = defaultdict(lambda: defaultdict(list)); params = {}
    for r in rows:
        agg[r["model"]]["acc"].append(r["acc"])
        if "ari" in r:
            agg[r["model"]]["ari"].append(r["ari"])
        params[r["model"]] = r["params"]

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        return m, sd

    print(f"\n=== EM-attention vs baseline (K={CFG['K']} max, variable lengths) ===")
    print(f"{'model':<14}{'params':>10}{'acc':>14}{'cluster-ARI':>14}")
    for m in ("baseline", "baseline_wide", "em_fixed", "em_varpos"):
        if m not in agg:
            continue
        am, asd = ms(agg[m]["acc"])
        ari = f"{ms(agg[m]['ari'])[0]*100:>11.2f}" if agg[m]["ari"] else f"{'-':>11}"
        print(f"{m:<14}{params[m]:>10,}{am*100:>9.2f}±{asd*100:<4.1f}{ari}")
    if "baseline" in agg:
        print()
        for m in ("em_fixed", "em_varpos"):
            if m in agg:
                gap = (ms(agg[m]["acc"])[0] - ms(agg["baseline"]["acc"])[0]) * 100
                print(f"  {m} - baseline:  acc {gap:+.2f} pp")
    print("\nRead: does EM beat baseline? Does cluster-ARI stay HIGH (clusters track "
          "sentences, unlike the h2c-readout's ~0.05 drift)?")


def smoke(cfg):
    ml = max_len_for(cfg["K"], cfg["L_max"])
    t, s, y, ns, _ = make(8, cfg["K"], cfg["L_min"], cfg["L_max"], seed=0)
    print(f"smoke: S={sorted(set(ns.tolist()))}, max_len={ml}")
    x, sgi, ny = torch.tensor(t), torch.tensor(s), torch.tensor(ns)
    for name in ("baseline", "em_fixed", "em_varpos"):
        model = build(name, cfg, ml)
        logits = model(x, sgi)
        masked = mask_logits(logits, ny)
        assert logits.shape == (8, cfg["K"]), logits.shape
        for b in range(8):
            S = int(ny[b])
            assert torch.isinf(masked[b, S:]).all() and torch.isfinite(masked[b, :S]).all()
        F.cross_entropy(masked, torch.tensor(y)).backward()
        print(f"  {name:13s} logits{tuple(logits.shape)} params={count_params(model):,} "
              f"nan={bool(torch.isnan(logits).any())}")
    print("  SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.smoke:
        smoke(cfg); return
    if args.analyze:
        analyze(); return
    if args.quick:
        cfg = dict(cfg, epochs=10, train_size=3000)
        models, seeds = ["baseline", "em_fixed"], [0]
    else:
        models = ["baseline", "em_fixed", "em_varpos"]
        if args.wide:
            models.append("baseline_wide")
        seeds = list(range(args.seeds))
    print(f"=== device={DEVICE} | models={models} | seeds={seeds} | "
          f"d={cfg['d']} L={cfg['n_layers']} K={cfg['K']} ep={cfg['epochs']} ===")
    for m in models:
        for s in seeds:
            run(m, s, cfg)
    analyze()


if __name__ == "__main__":
    main()
