"""
CNN vs ANN vs ANN+attention.

Tests whether forcing the feature hierarchy (depth-axis attention) helps a net
that has NO built-in structural prior (a plain MLP), using a CNN as the
reference ceiling.

    cnn       : convolutional baseline (structure for free)
    mlp       : plain fully-connected net (structureless)
    mlp_wide  : wider plain MLP, param control for mlp_attn
    mlp_attn  : the idea -> each hidden layer attends over all previous hidden
                layers, then gates itself

Headline contrast is mlp_attn vs mlp / mlp_wide. CNN just marks how much the
convolutional prior is worth.

    python arch_compare.py --sweep --dataset mnist      # full
    python arch_compare.py --quick --dataset mnist      # small tiers, 2 seeds
    python arch_compare.py --analyze --dataset mnist
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from models import HierNet, MLP, MLPAttn, count_params
from train import (get_loaders, evaluate, set_seed, DEVICE, DATASET_CFG,
                   RESULTS_DIR)

IMG_SIZE = {"mnist": 28, "cifar10": 32}
MODELS = ("cnn", "mlp", "mlp_wide", "mlp_attn")


def build_model(name, dataset):
    cfg = DATASET_CFG[dataset]
    in_ch, n_classes, channels = cfg["in_ch"], cfg["n_classes"], cfg["channels"]
    in_dim = in_ch * IMG_SIZE[dataset] ** 2
    if name == "cnn":
        return HierNet(in_ch, n_classes, channels, variant="baseline")
    if name == "mlp":
        return MLP(in_dim, n_classes, width=256, depth=4)
    if name == "mlp_wide":
        return MLP(in_dim, n_classes, width=320, depth=4)
    if name == "mlp_attn":
        return MLPAttn(in_dim, n_classes, width=256, depth=4, d=32)
    raise ValueError(name)


def results_path(dataset):
    return os.path.join(RESULTS_DIR, f"arch_{dataset}_results.jsonl")


def append_result(dataset, rec):
    with open(results_path(dataset), "a") as f:
        f.write(json.dumps(rec) + "\n")


def train_one(dataset, model_name, train_size, seed, epochs, lr=1e-3, batch=128):
    set_seed(seed)
    train_loader, test_loader = get_loaders(dataset, train_size, seed, batch)
    model = build_model(model_name, dataset).to(DEVICE)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
    acc = evaluate(model, test_loader)
    dt = time.time() - t0

    rec = dict(dataset=dataset, model=model_name, train_size=train_size,
               seed=seed, epochs=epochs, test_acc=round(acc, 4),
               n_params=n_params, sec=round(dt, 1))
    print(f"  {model_name:9s} size={train_size:<6} seed={seed} "
          f"acc={acc:.4f} params={n_params:,} ({dt:.0f}s)")
    return rec


def run_sweep(dataset, seeds=(0, 1, 2), epochs=None,
              fracs=(0.01, 0.05, 0.10, 0.25, 1.0)):
    if epochs is None:
        epochs = 20 if dataset == "mnist" else 40   # MLPs need a few more epochs
    full = 60000 if dataset == "mnist" else 50000
    sizes = [int(full * f) for f in fracs]
    print(f"=== arch sweep {dataset} | {DEVICE} | epochs={epochs} | seeds={seeds} ===")
    print(f"train sizes: {sizes}")
    for size in sizes:
        print(f"\n[train_size={size}]")
        for m in MODELS:
            for seed in seeds:
                append_result(dataset, train_one(dataset, m, size, seed, epochs))


# --------------------------------------------------------------------------- #
def analyze(dataset):
    rows = []
    with open(results_path(dataset)) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    acc = defaultdict(list)
    params = {}
    for r in rows:
        acc[(r["model"], r["train_size"])].append(r["test_acc"])
        params[r["model"]] = r["n_params"]
    sizes = sorted({r["train_size"] for r in rows})
    models = [m for m in MODELS if m in params]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    print(f"\n=== {dataset}: test accuracy (mean over seeds) ===")
    head = f"{'model':<10}{'params':>11}  " + "".join(f"{s:>10}" for s in sizes)
    print(head); print("-" * len(head))
    table = {}
    for m in models:
        line = f"{m:<10}{params[m]:>11,}  "
        for s in sizes:
            mu = mean(acc.get((m, s), []))
            table[(m, s)] = mu
            line += f"{mu*100:>10.2f}" if mu is not None else f"{'-':>10}"
        print(line)

    print(f"\n=== the contrast that matters (percentage points) ===")
    for label, a, b in [("mlp_attn - mlp     ", "mlp_attn", "mlp"),
                        ("mlp_attn - mlp_wide", "mlp_attn", "mlp_wide"),
                        ("cnn      - mlp     ", "cnn", "mlp")]:
        line = f"{label}  "
        for s in sizes:
            x, y = table.get((a, s)), table.get((b, s))
            line += f"{(x-y)*100:>+10.2f}" if x and y else f"{'-':>10}"
        print(line)
    print("\nRead: if 'mlp_attn - mlp' (and - mlp_wide) is positive and largest"
          "\nat small data, forcing the hierarchy helps a structureless net.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_CFG), default="mnist")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.dataset)
    elif args.quick:
        run_sweep(args.dataset, seeds=(0, 1), fracs=(0.01, 0.05, 0.10))
    elif args.sweep:
        run_sweep(args.dataset)
    else:
        analyze(args.dataset)


if __name__ == "__main__":
    main()
