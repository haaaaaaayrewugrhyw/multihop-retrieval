"""
Read results/<dataset>_results.jsonl and print the sample-efficiency table:
test accuracy (mean +/- std across seeds) for each variant at each train size.

    python analyze.py --dataset mnist

The key question: does the gap between attention variants (A/B/C) and the
controls (baseline, skip) SHRINK as train size grows? If yes, "forcing the
hierarchy" mainly buys sample efficiency. If all columns are flat, forcing
adds nothing.
"""

import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
ORDER = ["baseline", "skip", "A", "B", "C"]


def load(dataset):
    path = os.path.join(RESULTS_DIR, f"{dataset}_results.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()
    rows = load(args.dataset)

    # group: (variant, size) -> list of acc ; also track params
    acc = defaultdict(list)
    params = {}
    for r in rows:
        acc[(r["variant"], r["train_size"])].append(r["test_acc"])
        params[r["variant"]] = r["n_params"]
    sizes = sorted({r["train_size"] for r in rows})
    variants = [v for v in ORDER if any(v == r["variant"] for r in rows)]

    # header
    print(f"\n=== {args.dataset}: test accuracy (mean +/- std over seeds) ===")
    head = f"{'variant':<10}{'params':>10}  " + "".join(f"{s:>14}" for s in sizes)
    print(head)
    print("-" * len(head))
    base_by_size = {}
    for v in variants:
        line = f"{v:<10}{params[v]:>10,}  "
        for s in sizes:
            xs = acc.get((v, s), [])
            if xs:
                m, sd = mean_std(xs)
                if v == "baseline":
                    base_by_size[s] = m
                line += f"{m*100:>7.2f}±{sd*100:<5.2f}"
            else:
                line += f"{'-':>14}"
        print(line)

    # gap vs baseline -> the sample-efficiency signal
    print(f"\n=== accuracy gap vs baseline (percentage points) ===")
    print(f"{'variant':<10}  " + "".join(f"{s:>10}" for s in sizes))
    for v in variants:
        if v == "baseline":
            continue
        line = f"{v:<10}  "
        for s in sizes:
            xs = acc.get((v, s), [])
            if xs and s in base_by_size:
                m, _ = mean_std(xs)
                gap = (m - base_by_size[s]) * 100
                line += f"{gap:>+10.2f}"
            else:
                line += f"{'-':>10}"
        print(line)
    print("\nRead: if the +gap for A/B/C is large at small sizes and ~0 at the"
          "\nlargest size, forcing the hierarchy buys sample efficiency.")


if __name__ == "__main__":
    main()
