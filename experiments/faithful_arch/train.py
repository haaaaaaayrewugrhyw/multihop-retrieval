"""
Harness for the faithful three-stream architecture on variable-length grouped
digit-sentences.

Models (all know the sentences, so only the MECHANISM differs):
    baseline      : standard transformer + segment embedding (sentence info)
    full_fixed    : faithful cluster architecture, fixed position
    full_varpos   : faithful cluster architecture, variable position
    baseline_wide : wider baseline, param-match for full_varpos (--wide)

Absent-sentence logits (slots S..K-1) are masked to -inf before loss/argmax via
n_sentences, so the K-way head is well-posed as the sentence count varies --
applied identically to every model.

    python train.py --smoke      # CPU shape/grad + logit-mask checks (variable lengths)
    python train.py --quick      # 1-seed learnability check
    python train.py --seeds 3    # controlled run
    python train.py --analyze
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

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
RESFILE = os.path.join(RESULTS, "faithful.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=128, n_layers=4, heads=4, K=5, L_min=2, L_max=7,
           train_size=20000, epochs=40, batch=128, lr=3e-4)

MODELS = {
    "baseline":      dict(cluster_mode="none", variable_pos=False, use_segment=True),
    "full_fixed":    dict(cluster_mode="full", variable_pos=False, use_segment=False),
    "full_varpos":   dict(cluster_mode="full", variable_pos=True,  use_segment=False),
    "baseline_wide": dict(cluster_mode="none", variable_pos=False, use_segment=True,
                          d_override=132),   # param-match for full_varpos (~1.07M)
}


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(model_name, cfg, max_len):
    spec = dict(MODELS[model_name])
    d = spec.pop("d_override", cfg["d"])
    return FaithfulArch(VOCAB, d, cfg["n_layers"], cfg["heads"], cfg["K"], max_len,
                        pad_id=PAD, **spec)


def mask_logits(logits, n_sent):
    """Mask absent-sentence slots (>= n_sent) to -inf."""
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
    """Mean ARI between argmax(final cluster P) and the true sentence id (non-PAD)."""
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


def run(model_name, seed, cfg):
    set_seed(seed)
    K, ml = cfg["K"], max_len_for(cfg["K"], cfg["L_max"])
    tr_t, tr_s, tr_y, tr_n, _ = make(cfg["train_size"], K, cfg["L_min"], cfg["L_max"], seed=seed)
    te_t, te_s, te_y, te_n, _ = make(3000, K, cfg["L_min"], cfg["L_max"], seed=90001)
    model = build(model_name, cfg, ml).to(DEVICE)
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
            opt.step()
        sched.step()

    rec = dict(model=model_name, seed=seed,
               acc=round(accuracy(model, te_t, te_s, te_y, te_n), 4),
               params=count_params(model), sec=round(time.time() - t0, 1))
    if MODELS[model_name]["cluster_mode"] == "full":
        rec["ari"] = round(cluster_recovery(model, te_t, te_s), 4)
    with open(RESFILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    extra = f"  cluster-ARI={rec['ari']:.3f}" if "ari" in rec else ""
    print(f"  {model_name:13s} seed={seed} acc={rec['acc']:.3f}{extra}  "
          f"({rec['params']:,}p {rec['sec']}s)")
    if model_name == "full_fixed" and seed == 0:
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

    print(f"\n=== faithful arch on grouped sentences (K={CFG['K']} max, variable lengths) ===")
    print(f"{'model':<14}{'params':>10}{'acc':>14}{'cluster-ARI':>14}")
    for m in ("baseline", "baseline_wide", "full_fixed", "full_varpos"):
        if m not in agg:
            continue
        am, asd = ms(agg[m]["acc"])
        ari = f"{ms(agg[m]['ari'])[0]*100:>11.2f}" if agg[m]["ari"] else f"{'-':>11}"
        print(f"{m:<14}{params[m]:>10,}{am*100:>9.2f}±{asd*100:<4.1f}{ari}")
    if "baseline" in agg:
        print()
        for m in ("full_fixed", "full_varpos"):
            if m not in agg:
                continue
            ref = "baseline_wide" if (m == "full_varpos" and "baseline_wide" in agg) else "baseline"
            gap = (ms(agg[m]["acc"])[0] - ms(agg[ref]["acc"])[0]) * 100
            print(f"  {m} - {ref}:  acc {gap:+.2f} pp")
    print("\nRead: does full_* beat baseline (expected ~tie)? Does cluster-ARI stay high "
          "(clusters track sentences) or drift?")


def smoke(cfg):
    ml = max_len_for(cfg["K"], cfg["L_max"])
    t, s, y, ns, _ = make(8, cfg["K"], cfg["L_min"], cfg["L_max"], seed=0)
    print(f"smoke: variable lengths, S in test batch = {sorted(set(ns.tolist()))}, max_len={ml}")
    x, sgi, ny = torch.tensor(t), torch.tensor(s), torch.tensor(ns)
    for name in ("baseline", "full_fixed", "full_varpos"):
        model = build(name, cfg, ml)
        logits = model(x, sgi)
        masked = mask_logits(logits, ny)
        assert logits.shape == (8, cfg["K"]), logits.shape
        for b in range(8):                              # absent slots must be -inf, present finite
            S = int(ny[b])
            assert torch.isinf(masked[b, S:]).all(), f"absent not masked, ex{b}"
            assert torch.isfinite(masked[b, :S]).all(), f"present masked, ex{b}"
        F.cross_entropy(masked, torch.tensor(y)).backward()
        qkv_in = model.blocks[0].attn.qkv.in_features
        print(f"  {name:13s} logits{tuple(logits.shape)} qkv_in={qkv_in} "
              f"params={count_params(model):,} (h2c={hasattr(model,'h2c')} p2h={hasattr(model,'p2h')})")
    base = build("baseline", cfg, ml)
    assert not hasattr(base, "h2c") and not hasattr(base, "p2h")
    print("  baseline builds no h2c/p2h; logit masking applied. SMOKE OK")


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
        models, seeds = ["baseline", "full_fixed"], [0]
    else:
        models = ["baseline", "full_fixed", "full_varpos"]
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
