"""
2x2 factorial screen for the cluster-transformer idea.

Models:  baseline | varpos | cluster | cluster+varpos   (segment-aware all)
Tasks:   A (grouping needed)  vs  B (grouping irrelevant, control)
Metric:  test accuracy vs train-set size (sample efficiency), over seeds.

    python train.py --sweep          # full screen, prints mean +/- std table
    python train.py --quick          # tiny: 1 size, 1 seed (smoke / sanity)

Read: cluster should beat baseline on A (most at small data) and tie on B.
The 2x2 also isolates whether variable positions help on their own.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from data import make, num_classes, VOCAB
from models import ClusterTransformer, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "baseline":        dict(use_cluster=False, variable_pos=False),
    "varpos":          dict(use_cluster=False, variable_pos=True),
    "cluster":         dict(use_cluster=True,  variable_pos=False),
    "cluster+varpos":  dict(use_cluster=True,  variable_pos=True),
}


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run(variant, model_name, size, seed, K=4, L=6, d=64, layers=3, heads=4,
        epochs=40, batch=128):
    set_seed(seed)
    ncls = num_classes(variant, K)
    tr_t, tr_s, tr_y = make(size, K, L, variant, seed=seed)
    te_t, te_s, te_y = make(2000, K, L, variant, seed=99999)
    T = tr_t.shape[1]
    model = ClusterTransformer(VOCAB, d, layers, heads, ncls, K, T,
                               **CONFIGS[model_name]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tr_t = torch.tensor(tr_t); tr_s = torch.tensor(tr_s); tr_y = torch.tensor(tr_y)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(size)
        for i in range(0, size, batch):
            idx = perm[i:i + batch]
            x = tr_t[idx].to(DEVICE); s = tr_s[idx].to(DEVICE); y = tr_y[idx].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x, s), y).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(te_t).to(DEVICE); st = torch.tensor(te_s).to(DEVICE)
        preds = []
        for i in range(0, len(te_t), 500):
            preds.append(model(xt[i:i + 500], st[i:i + 500]).argmax(1).cpu())
        acc = (torch.cat(preds).numpy() == te_y).mean()

    rec = dict(variant=variant, model=model_name, size=size, seed=seed,
               acc=round(float(acc), 4), params=count_params(model),
               sec=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "screen.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {variant} {model_name:14s} size={size:<6} seed={seed} "
          f"acc={acc:.4f} ({rec['sec']}s)")
    return rec


def sweep(sizes=(200, 500, 1000, 2000, 5000), seeds=(0, 1, 2), epochs=60):
    print(f"=== screen | device={DEVICE} | epochs={epochs} ===")
    for variant in ("A", "B"):
        for size in sizes:
            print(f"\n[variant {variant} | size {size}]")
            for m in CONFIGS:
                for seed in seeds:
                    run(variant, m, size, seed, epochs=epochs)
    analyze()


def analyze():
    rows = []
    with open(os.path.join(RESULTS, "screen.jsonl")) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    acc = defaultdict(list)
    params = {}
    for r in rows:
        acc[(r["variant"], r["model"], r["size"])].append(r["acc"])
        params[r["model"]] = r["params"]
    sizes = sorted({r["size"] for r in rows})

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        return m, sd

    for variant in ("A", "B"):
        if not any(v == variant for (v, _, _) in acc):
            continue
        print(f"\n=== variant {variant}: test acc (mean +/- std) ===")
        head = f"{'model':<15}{'params':>9}  " + "".join(f"{s:>13}" for s in sizes)
        print(head); print("-" * len(head))
        base = {}
        for m in CONFIGS:
            line = f"{m:<15}{params.get(m,0):>9,}  "
            for s in sizes:
                xs = acc.get((variant, m, s), [])
                if xs:
                    mu, sd = ms(xs)
                    if m == "baseline":
                        base[s] = mu
                    line += f"{mu*100:>7.2f}±{sd*100:<4.1f}"
                else:
                    line += f"{'-':>13}"
            print(line)
        # gap vs baseline
        print(f"  gap vs baseline (pp):")
        for m in ("varpos", "cluster", "cluster+varpos"):
            line = f"  {m:<13}  "
            for s in sizes:
                xs = acc.get((variant, m, s), [])
                if xs and s in base:
                    line += f"{(ms(xs)[0]-base[s])*100:>+13.2f}"
                else:
                    line += f"{'-':>13}"
            print(line)
    print("\nRead: on A, want cluster > baseline (esp. small size); on B, want ~tie.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    elif args.quick:
        sweep(sizes=(2000,), seeds=(0,), epochs=20)
    else:
        sweep()


if __name__ == "__main__":
    main()
