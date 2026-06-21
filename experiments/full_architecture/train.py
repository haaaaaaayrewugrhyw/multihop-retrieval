"""
From-scratch training/eval harness for the full architecture on ListOps.

Train on shallow depths, evaluate in-distribution (same depths) AND per-depth
OOD (deeper trees = compositional generalization). Models compared:
    baseline      : vanilla transformer (cluster_mode none, fixed position)
    full_fixed    : full cluster mechanism, fixed position
    full_varpos   : full cluster mechanism, variable position
    baseline_wide : wider vanilla transformer, param-match for full_varpos (--wide)

    python train.py --smoke           # CPU shape/grad smoke
    python train.py --quick           # 1-seed learnability check
    python train.py --seeds 3         # controlled run
    python train.py --analyze
    # LRA-scale (Kaggle):
    python train.py --seeds 8 --max_len 512 --max_args 5 --train_depths 1,2,3 \
                    --test_depths 4,5,6,7 --train_size 50000 --batch 32

Chance = 10% (10-way). Honest expectation: the cluster mechanism likely ties the
baseline. The only interesting outcome is an OOD gap that GROWS with depth.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from data import make, VOCAB
from model import FullArchitecture, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
RESFILE = os.path.join(RESULTS, "full.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=128, n_layers=4, heads=4, K=8, max_len=96, max_args=3,
           train_depths=(1, 2), test_depths=(3, 4, 5),
           train_size=20000, epochs=40, batch=128, lr=3e-4)

MODELS = {
    "baseline":      dict(cluster_mode="none", variable_pos=False),
    "full_fixed":    dict(cluster_mode="full", variable_pos=False),
    "full_varpos":   dict(cluster_mode="full", variable_pos=True),
    "baseline_wide": dict(cluster_mode="none", variable_pos=False, d_override=136),
}


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(model_name, cfg):
    spec = dict(MODELS[model_name])
    d = spec.pop("d_override", cfg["d"])
    return FullArchitecture(VOCAB, d, cfg["n_layers"], cfg["heads"], 10,
                            K=cfg["K"], max_len=cfg["max_len"], **spec)


@torch.no_grad()
def accuracy(model, toks, labels, batch=512):
    model.eval()
    xt = torch.tensor(toks).to(DEVICE)
    preds = []
    for i in range(0, len(toks), batch):
        preds.append(model(xt[i:i + batch]).argmax(1).cpu())
    return float((torch.cat(preds).numpy() == labels).mean())


def run(model_name, seed, cfg):
    set_seed(seed)
    ml, ma = cfg["max_len"], cfg["max_args"]
    tr_t, tr_y = make(cfg["train_size"], cfg["train_depths"], ml, ma, seed=seed)
    ti_t, ti_y = make(3000, cfg["train_depths"], ml, ma, seed=70001)
    tests = {dp: make(3000, (dp,), ml, ma, seed=80000 + dp) for dp in cfg["test_depths"]}

    model = build(model_name, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    tr_t = torch.tensor(tr_t); tr_y = torch.tensor(tr_y)
    bs = cfg["batch"]

    t0 = time.time()
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_size"])
        for i in range(0, cfg["train_size"], bs):
            idx = perm[i:i + bs]
            x = tr_t[idx].to(DEVICE); y = tr_y[idx].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()

    rec = dict(model=model_name, seed=seed, acc_in=round(accuracy(model, ti_t, ti_y), 4),
               params=count_params(model), sec=round(time.time() - t0, 1))
    for dp in cfg["test_depths"]:
        rec[f"d{dp}"] = round(accuracy(model, tests[dp][0], tests[dp][1]), 4)
    if hasattr(model, "lam"):
        rec["lam"] = [round(v, 3) for v in model.lam.detach().cpu().tolist()]
    with open(RESFILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    line = "  ".join(f"d{dp}={rec['d'+str(dp)]:.3f}" for dp in cfg["test_depths"])
    print(f"  {model_name:13s} seed={seed} in={rec['acc_in']:.3f}  {line}  "
          f"({rec['params']:,}p {rec['sec']}s)")
    return rec


def analyze(cfg):
    rows = [json.loads(l) for l in open(RESFILE) if l.strip()]
    agg = defaultdict(lambda: defaultdict(list))
    params = {}
    cols = ["in"] + [f"d{dp}" for dp in cfg["test_depths"]]
    for r in rows:
        agg[r["model"]]["in"].append(r["acc_in"])
        for dp in cfg["test_depths"]:
            key = f"d{dp}"
            if key in r:
                agg[r["model"]][key].append(r[key])
        params[r["model"]] = r["params"]

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        return m, sd

    print(f"\n=== full architecture on ListOps "
          f"(train {cfg['train_depths']}, test {cfg['test_depths']}) ===")
    print(f"{'model':<14}{'params':>10}" + "".join(f"{c:>13}" for c in cols))
    for m in ("baseline", "baseline_wide", "full_fixed", "full_varpos"):
        if m not in agg:
            continue
        line = f"{m:<14}{params[m]:>10,}"
        for c in cols:
            if agg[m][c]:
                mu, sd = ms(agg[m][c]); line += f"{mu*100:>8.2f}±{sd*100:<4.1f}"
            else:
                line += f"{'-':>13}"
        print(line)

    if "baseline" in agg:
        print()
        for m in ("full_fixed", "full_varpos"):
            if m not in agg:
                continue
            ref = "baseline_wide" if (m == "full_varpos" and "baseline_wide" in agg) else "baseline"
            line = f"{m + ' - ' + ref:<24}"
            for c in cols:
                if agg[m][c] and agg[ref][c]:
                    line += f"{(ms(agg[m][c])[0] - ms(agg[ref][c])[0]) * 100:>+13.2f}"
                else:
                    line += f"{'-':>13}"
            print(line)
    print("\nRead: in-dist full_* should ~tie baseline; on OOD, does the gap GROW "
          "with depth (structural prior) or stay flat/negative (null)?")


def smoke(cfg):
    print("shape/grad smoke (CPU)...")
    toks, labels = make(8, (1, 2), cfg["max_len"], cfg["max_args"], seed=0)
    x, y = torch.tensor(toks), torch.tensor(labels)
    for name in ("baseline", "full_fixed", "full_varpos"):
        model = build(name, cfg)
        out = model(x)
        assert out.shape == (8, 10), out.shape
        F.cross_entropy(out, y).backward()
        if hasattr(model, "lam"):
            assert model.lam.grad is not None, "lam has no grad"
        nan = bool(torch.isnan(out).any())
        print(f"  {name:13s} out{tuple(out.shape)} params={count_params(model):>9,} "
              f"lam={hasattr(model,'lam')} p2h={hasattr(model,'p2h')} nan={nan}")
    base = build("baseline", cfg)
    assert not hasattr(base, "lam") and not hasattr(base, "p2h"), "baseline built extra modules"
    print("  baseline builds no cluster/varpos submodules: OK")
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    for k in ("max_len", "max_args", "train_size", "batch"):
        ap.add_argument(f"--{k}", type=int)
    ap.add_argument("--train_depths", type=str)
    ap.add_argument("--test_depths", type=str)
    args = ap.parse_args()

    cfg = dict(CFG)
    for k in ("max_len", "max_args", "train_size", "batch"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.train_depths:
        cfg["train_depths"] = tuple(int(x) for x in args.train_depths.split(","))
    if args.test_depths:
        cfg["test_depths"] = tuple(int(x) for x in args.test_depths.split(","))

    if args.smoke:
        smoke(cfg); return
    if args.analyze:
        analyze(cfg); return

    if args.quick:
        cfg = dict(cfg, epochs=10, train_size=2000)
        models, seeds = ["baseline", "full_fixed"], [0]
    else:
        models = ["baseline", "full_fixed", "full_varpos"]
        if args.wide:
            models.append("baseline_wide")
        seeds = list(range(args.seeds))

    print(f"=== device={DEVICE} | models={models} | seeds={seeds} | "
          f"d={cfg['d']} L={cfg['n_layers']} ml={cfg['max_len']} ep={cfg['epochs']} ===")
    for m in models:
        for s in seeds:
            run(m, s, cfg)
    analyze(cfg)


if __name__ == "__main__":
    main()
