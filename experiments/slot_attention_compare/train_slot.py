"""
Train the slot autoencoder under the unsupervised reconstruction objective and
score object discovery with FG-ARI. Compares real Slot Attention (competition)
against the plain-attention MLP+attn idea (no competition).

    python train_slot.py --compare            # train both, print table + viz
    python train_slot.py --model slot         # train one
    python train_slot.py --model plain

Metrics:
    recon_mse : reconstruction quality (lower better)
    fg_ari    : foreground Adjusted Rand Index, object discovery (higher better,
                1.0 = perfect object separation, ~0 = slots don't separate objects)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from data import make_dataset
from slot_models import SlotAE, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fg_ari(pred, gt):
    """Mean foreground-ARI over a batch. pred,gt: numpy (B,H,W) int."""
    scores = []
    for p, g in zip(pred, gt):
        g, p = g.reshape(-1), p.reshape(-1)
        fg = g != 0
        if fg.sum() < 2 or len(np.unique(g[fg])) < 2:
            continue
        scores.append(adjusted_rand_score(g[fg], p[fg]))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate(model, imgs, masks, batch=64):
    model.eval()
    mse_tot, n, aris = 0.0, 0, []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).to(DEVICE)
        recon, alpha = model(x)
        mse_tot += F.mse_loss(recon, x, reduction="sum").item()
        n += x.numel()
        pred = alpha.squeeze(2).argmax(dim=1).cpu().numpy()       # B,H,W
        aris.append(fg_ari(pred, masks[i:i + batch]))
    return mse_tot / n, float(np.mean(aris))


def train(model_name, epochs, hard=False, n_train=5000, n_test=512, batch=32,
          lr=4e-4, num_slots=4, iters=3, seed=0, warmup=1500):
    set_seed(seed)
    competition = (model_name == "slot")
    model = SlotAE(num_slots=num_slots, iters=iters,
                   competition=competition).to(DEVICE)
    n_params = count_params(model)
    tr_imgs, tr_masks = make_dataset(n_train, seed=seed, same_color=hard)
    te_imgs, te_masks = make_dataset(n_test, seed=999, same_color=hard)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = (n_train + batch - 1) // batch
    total_steps = epochs * steps_per_epoch
    step, t0 = 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            x = torch.from_numpy(tr_imgs[idx]).to(DEVICE)
            recon, _ = model(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # linear warmup then cosine decay (slot attn is sensitive to this)
            if step < warmup:
                cur = lr * (step + 1) / warmup
            else:
                prog = (step - warmup) / max(1, total_steps - warmup)
                cur = 0.5 * lr * (1 + np.cos(np.pi * prog))
            for g in opt.param_groups:
                g["lr"] = cur
            opt.step(); step += 1
        if ep % 5 == 0 or ep == epochs - 1:
            mse, ari = evaluate(model, te_imgs, te_masks)
            print(f"  [{model_name}] ep {ep:>3} loss {loss.item():.4f} "
                  f"test_mse {mse:.5f} fg_ari {ari:.4f} ({time.time()-t0:.0f}s)")
    mse, ari = evaluate(model, te_imgs, te_masks)
    rec = dict(model=model_name, epochs=epochs, hard=hard,
               recon_mse=round(mse, 6), fg_ari=round(ari, 4),
               n_params=n_params, sec=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS_DIR, "slot_results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return model, rec


def save_viz(models, n=6, seed=999):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("skip viz:", e); return
    imgs, _ = make_dataset(n, seed=seed)
    x = torch.from_numpy(imgs).to(DEVICE)
    ncols = 1 + len(models) * (1 + 4)        # input + per-model (recon + 4 masks)
    fig, ax = plt.subplots(n, ncols, figsize=(ncols * 1.4, n * 1.4))
    for r in range(n):
        ax[r, 0].imshow(np.transpose(imgs[r], (1, 2, 0))); ax[r, 0].set_title("in" if r == 0 else "")
        col = 1
        for name, m in models.items():
            m.eval()
            with torch.no_grad():
                recon, alpha = m(x[r:r + 1])
            ax[r, col].imshow(np.transpose(recon[0].cpu().numpy(), (1, 2, 0)).clip(0, 1))
            if r == 0: ax[r, col].set_title(f"{name}\nrecon")
            col += 1
            for s in range(4):
                ax[r, col].imshow(alpha[0, s, 0].cpu().numpy(), cmap="viridis")
                if r == 0: ax[r, col].set_title(f"slot{s}")
                col += 1
    for a in ax.ravel(): a.axis("off")
    path = os.path.join(RESULTS_DIR, "slot_masks.png")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["slot", "plain"], default="slot")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hard", action="store_true",
                    help="same-color objects: color can't shortcut binding")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    print(f"device={DEVICE} hard={args.hard} epochs={args.epochs}")
    if args.compare:
        models = {}
        for name in ["slot", "plain"]:
            print(f"\n=== training {name} ===")
            m, rec = train(name, args.epochs, hard=args.hard)
            models[name] = m
            print("  ->", rec)
        print("\n=== RESULT ===")
        print(f"{'model':<8}{'hard':>6}{'fg_ari':>10}{'recon_mse':>12}{'params':>10}")
        for line in open(os.path.join(RESULTS_DIR, "slot_results.jsonl")):
            r = json.loads(line)
            print(f"{r['model']:<8}{str(r.get('hard', False)):>6}"
                  f"{r['fg_ari']:>10.4f}{r['recon_mse']:>12.6f}{r['n_params']:>10,}")
        save_viz(models)
    else:
        train(args.model, args.epochs, hard=args.hard)


if __name__ == "__main__":
    main()
