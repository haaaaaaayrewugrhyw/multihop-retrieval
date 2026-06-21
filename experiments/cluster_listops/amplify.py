"""
Amplification test: does the explicit affinity-bias advantage GROW with OOD
depth? Train on depth 1-2, evaluate at depth 3, 4, 5 separately. A real
structural prior should help MORE as generalization gets harder (gap grows
with depth); a marginal offset stays flat.

    python amplify.py            # baseline vs cluster_bias, 8 seeds
    python amplify.py --analyze
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
RESFILE = os.path.join(RESULTS, "amplify.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 96                       # bigger, so depth-5 trees fit
TRAIN_DEPTHS = (1, 2)
TEST_DEPTHS = (3, 4, 5)
MODE = {"baseline": "none", "cluster_bias": "bias"}


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
    tr_t, tr_y = make(train_size, TRAIN_DEPTHS, MAX_LEN, seed=seed)
    ti = make(3000, TRAIN_DEPTHS, MAX_LEN, seed=70001)
    tests = {dp: make(3000, (dp,), MAX_LEN, seed=80000 + dp) for dp in TEST_DEPTHS}
    model = ListOpsTransformer(VOCAB, d, layers, heads, 10, K=8,
                               max_len=MAX_LEN, cluster_mode=MODE[model_name]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_t = torch.tensor(tr_t); tr_y = torch.tensor(tr_y)

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

    rec = dict(model=model_name, seed=seed, acc_in=round(acc_on(model, *ti), 4),
               params=count_params(model), sec=round(time.time() - t0, 1))
    for dp in TEST_DEPTHS:
        rec[f"d{dp}"] = round(acc_on(model, tests[dp][0], tests[dp][1]), 4)
    with open(RESFILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {model_name:13s} seed={seed}  in={rec['acc_in']:.3f}  " +
          "  ".join(f"d{dp}={rec['d'+str(dp)]:.3f}" for dp in TEST_DEPTHS) +
          f"  ({rec['sec']}s)")
    return rec


def analyze():
    rows = []
    with open(RESFILE) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]]["in"].append(r["acc_in"])
        for dp in TEST_DEPTHS:
            agg[r["model"]][f"d{dp}"].append(r[f"d{dp}"])

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        return m, sd

    cols = ["in"] + [f"d{dp}" for dp in TEST_DEPTHS]
    print(f"\n=== amplification: train depth {TRAIN_DEPTHS}, test by depth ===")
    print(f"{'model':<13}" + "".join(f"{c:>14}" for c in cols))
    for m in ("baseline", "cluster_bias"):
        if m not in agg:
            continue
        line = f"{m:<13}"
        for c in cols:
            mu, sd = ms(agg[m][c])
            line += f"{mu*100:>8.2f}+-{sd*100:<4.1f}"
        print(line)
    if "baseline" in agg and "cluster_bias" in agg:
        print(f"\n{'gap (bias-base) pp':<13}" +
              "".join(f"{(ms(agg['cluster_bias'][c])[0]-ms(agg['baseline'][c])[0])*100:>+14.2f}"
                      for c in cols))
        print("\nRead: if the gap GROWS from d3->d4->d5, the prior is a real "
              "structural effect. If flat, it's a fixed marginal offset.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    if args.analyze:
        analyze(); return
    print(f"=== device={DEVICE} | seeds={args.seeds} ===")
    for m in ("baseline", "cluster_bias"):
        for s in range(args.seeds):
            run(m, s)
    analyze()


if __name__ == "__main__":
    main()
