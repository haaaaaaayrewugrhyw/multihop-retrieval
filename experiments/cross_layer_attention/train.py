"""
Training / evaluation harness for the cross-layer attention experiment.

Single run:
    python train.py --dataset mnist --variant B --train-size 1000 --seed 0 --epochs 15

Full sweep (all variants x train sizes x seeds) for one dataset:
    python train.py --sweep --dataset mnist
    python train.py --sweep --dataset cifar10

Results are appended (one JSON object per line) to results/<dataset>_results.jsonl
so the sweep is resumable and the analysis script can read everything back.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import HierNet, VARIANTS, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# per-dataset config: in_ch, n_classes, channel widths (kept small for 4GB VRAM)
DATASET_CFG = {
    "mnist":   dict(in_ch=1, n_classes=10, channels=(16, 32, 64, 128)),
    "cifar10": dict(in_ch=3, n_classes=10, channels=(32, 64, 128, 256)),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(dataset, train_size, seed, batch=128):
    if dataset == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
        test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    elif dataset == "cifar10":
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
        test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
    else:
        raise ValueError(dataset)

    if train_size > 0 and train_size < len(train):
        # class-balanced random subset so small-data runs are not degenerate
        targets = train.targets
        labels = targets.numpy() if torch.is_tensor(targets) else np.array(targets)
        per_class = train_size // DATASET_CFG[dataset]["n_classes"]
        rng = np.random.RandomState(seed)
        idx = []
        for c in range(DATASET_CFG[dataset]["n_classes"]):
            cidx = np.where(labels == c)[0]
            idx.extend(rng.choice(cidx, size=per_class, replace=False).tolist())
        rng.shuffle(idx)
        train = Subset(train, idx)

    train_loader = DataLoader(train, batch_size=batch, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=(DEVICE == "cuda"))
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train_one(dataset, variant, train_size, seed, epochs, lr=1e-3, batch=128,
              verbose=True):
    set_seed(seed)
    cfg = DATASET_CFG[dataset]
    train_loader, test_loader = get_loaders(dataset, train_size, seed, batch)

    model = HierNet(cfg["in_ch"], cfg["n_classes"], cfg["channels"],
                    variant=variant).to(DEVICE)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
    acc = evaluate(model, test_loader)
    dt = time.time() - t0

    rec = dict(dataset=dataset, variant=variant, train_size=train_size,
               seed=seed, epochs=epochs, test_acc=round(acc, 4),
               n_params=n_params, sec=round(dt, 1))
    if verbose:
        print(f"  {variant:8s} size={train_size:<6} seed={seed} "
              f"acc={acc:.4f} params={n_params:,} ({dt:.0f}s)")
    return rec


def append_result(dataset, rec):
    path = os.path.join(RESULTS_DIR, f"{dataset}_results.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_sweep(dataset, seeds=(0, 1, 2), epochs=None, fracs=(0.01, 0.05, 0.10, 0.25, 1.0)):
    if epochs is None:
        epochs = 15 if dataset == "mnist" else 30
    full = 60000 if dataset == "mnist" else 50000
    sizes = [int(full * frac) for frac in fracs]
    print(f"=== sweep {dataset} | device={DEVICE} | epochs={epochs} | seeds={seeds} ===")
    print(f"train sizes: {sizes}")
    for size in sizes:
        print(f"\n[train_size={size}]")
        for variant in VARIANTS:
            for seed in seeds:
                rec = train_one(dataset, variant, size, seed, epochs)
                append_result(dataset, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_CFG), default="mnist")
    ap.add_argument("--variant", choices=list(VARIANTS), default="baseline")
    ap.add_argument("--train-size", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--sweep", action="store_true", help="full sweep (Kaggle)")
    ap.add_argument("--quick", action="store_true",
                    help="small-data tiers only, 2 seeds (laptop first signal)")
    args = ap.parse_args()

    if args.quick:
        # the interesting region for the hypothesis: small data, fast to run
        run_sweep(args.dataset, seeds=(0, 1), fracs=(0.01, 0.05, 0.10))
    elif args.sweep:
        run_sweep(args.dataset)
    else:
        rec = train_one(args.dataset, args.variant, args.train_size,
                        args.seed, args.epochs)
        append_result(args.dataset, rec)


if __name__ == "__main__":
    main()
