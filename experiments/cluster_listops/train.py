"""
Fair test: train on shallow ListOps (depth 1-2), test on both in-distribution
(depth 1-2) and OOD/deeper (depth 3-4). Compare a vanilla transformer against
the cluster-augmented one. Headline = OOD accuracy (compositional
generalization), where a structured/grouping prior should help if it ever does.

    python train.py            # full run (baseline vs cluster, seeds)
    python train.py --quick    # 1 seed smoke

Chance = 10% (10-way). Vanilla transformers are known to struggle to generalize
ListOps to deeper trees -> room for a structured prior to show value.
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
from models import ListOpsTransformer, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 64
TRAIN_DEPTHS = (1, 2)
OOD_DEPTHS = (3, 4)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def acc_on(model, toks, labels, batch=500):
    model.eval()
    xt = torch.tensor(toks).to(DEVICE)
    preds = []
    for i in range(0, len(toks), batch):
        preds.append(model(xt[i:i + batch]).argmax(1).cpu())
    return float((torch.cat(preds).numpy() == labels).mean())


def run(model_name, seed, train_size=20000, d=128, layers=4, heads=4,
        epochs=40, batch=128):
    set_seed(seed)
    use_cluster = (model_name == "cluster")
    tr_t, tr_y = make(train_size, TRAIN_DEPTHS, MAX_LEN, seed=seed)
    ti_t, ti_y = make(3000, TRAIN_DEPTHS, MAX_LEN, seed=70001)        # in-dist test
    to_t, to_y = make(3000, OOD_DEPTHS, MAX_LEN, seed=80002)          # OOD test
    model = ListOpsTransformer(VOCAB, d, layers, heads, 10, K=8,
                               max_len=MAX_LEN, use_cluster=use_cluster).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_t = torch.tensor(tr_t)
    tr_y = torch.tensor(tr_y)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(train_size)
        for i in range(0, train_size, batch):
            idx = perm[i:i + batch]
            x = tr_t[idx].to(DEVICE); y = tr_y[idx].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()

    rec = dict(model=model_name, seed=seed,
               acc_in=round(acc_on(model, ti_t, ti_y), 4),
               acc_ood=round(acc_on(model, to_t, to_y), 4),
               params=count_params(model), sec=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "listops.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {model_name:9s} seed={seed}  in-dist={rec['acc_in']:.4f}  "
          f"OOD={rec['acc_ood']:.4f}  ({rec['sec']}s)")
    return rec


def analyze():
    rows = []
    with open(os.path.join(RESULTS, "listops.jsonl")) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    agg = defaultdict(lambda: {"in": [], "ood": []})
    params = {}
    for r in rows:
        agg[r["model"]]["in"].append(r["acc_in"])
        agg[r["model"]]["ood"].append(r["acc_ood"])
        params[r["model"]] = r["params"]

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        return m, sd

    print(f"\n=== ListOps (train depth {TRAIN_DEPTHS}, OOD depth {OOD_DEPTHS}) ===")
    print(f"{'model':<9}{'params':>10}{'in-dist':>16}{'OOD':>16}")
    for m in ("baseline", "cluster"):
        if m not in agg:
            continue
        im, isd = ms(agg[m]["in"]); om, osd = ms(agg[m]["ood"])
        print(f"{m:<9}{params[m]:>10,}{im*100:>9.2f}+-{isd*100:<4.1f}"
              f"{om*100:>9.2f}+-{osd*100:<4.1f}")
    if "baseline" in agg and "cluster" in agg:
        d_in = (ms(agg["cluster"]["in"])[0] - ms(agg["baseline"]["in"])[0]) * 100
        d_ood = (ms(agg["cluster"]["ood"])[0] - ms(agg["baseline"]["ood"])[0]) * 100
        print(f"\ncluster - baseline:  in-dist {d_in:+.2f} pp   OOD {d_ood:+.2f} pp")
        print("Headline = OOD gap. Positive OOD gap = the structured prior helps "
              "generalize to deeper trees.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(); return
    seeds = (0,) if args.quick else (0, 1, 2)
    epochs = 15 if args.quick else 40
    print(f"=== device={DEVICE} | seeds={seeds} | epochs={epochs} ===")
    for m in ("baseline", "cluster"):
        for s in seeds:
            run(m, s, epochs=epochs)
    analyze()


if __name__ == "__main__":
    main()
