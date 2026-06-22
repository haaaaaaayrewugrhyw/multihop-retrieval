"""
Diagnostic: can we SEE the cluster mechanism separate groups?

Train baseline (none) vs full on the interleaved-group task, then for the full
model score how well the learned cluster membership P recovers the true groups
(ARI) and visualize token-cluster vs true-group. Three possible reads:
  - P recovers groups (high ARI), acc ties  -> mechanism works, just redundant
  - P does not cluster (low ARI)             -> mechanism inert (no incentive)
  - P clusters AND acc beats baseline        -> the surprising win

    python diagnose.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from data_groups import make_groups, G, L, PAD, VOCAB, N_CLASSES
from model import FullArchitecture, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN, K, D, LAYERS, HEADS = 24, 8, 128, 4, 4


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(cluster_mode):
    return FullArchitecture(VOCAB, D, LAYERS, HEADS, N_CLASSES, K, MAX_LEN,
                            cluster_mode=cluster_mode, pad_id=PAD).to(DEVICE)


def train(model, tr_t, tr_y, epochs=30, batch=128, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(tr_t); tt = torch.tensor(tr_t); ty = torch.tensor(tr_y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x = tt[idx].to(DEVICE); y = ty[idx].to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()


@torch.no_grad()
def accuracy(model, t, y, batch=512):
    model.eval()
    xt = torch.tensor(t).to(DEVICE)
    preds = [model(xt[i:i + batch]).argmax(1).cpu() for i in range(0, len(t), batch)]
    return float((torch.cat(preds).numpy() == y).mean())


@torch.no_grad()
def cluster_recovery(model, t, groups, batch=512):
    """Mean ARI between argmax(P) and the true per-token group (non-PAD tokens)."""
    model.eval(); model.collect_P = True
    preds = []
    xt = torch.tensor(t).to(DEVICE)
    for i in range(0, len(t), batch):
        _ = model(xt[i:i + batch])
        preds.append(model._last_P.argmax(-1).cpu().numpy())
    pred = np.concatenate(preds, axis=0)                        # (N,T)
    aris = []
    for i in range(len(t)):
        m = groups[i] != -1
        if m.sum() >= 2 and len(set(groups[i][m].tolist())) >= 2:
            aris.append(adjusted_rand_score(groups[i][m], pred[i][m]))
    return float(np.mean(aris))


def viz(model, t, groups, n=5):
    model.eval(); model.collect_P = True
    with torch.no_grad():
        _ = model(torch.tensor(t[:n]).to(DEVICE))
    pred = model._last_P.argmax(-1).cpu().numpy()
    print("\n  token-by-token  (true group 0-3   vs   assigned cluster 0-7):")
    for i in range(n):
        m = groups[i] != -1
        true = "".join(str(x) for x in groups[i][m].tolist())
        clus = "".join(str(x) for x in pred[i][m].tolist())
        print(f"    ex{i}  true   {true}")
        print(f"          clust  {clus}")


def main():
    tr_t, tr_y, _ = make_groups(20000, MAX_LEN, seed=0)
    te_t, te_y, te_g = make_groups(2000, MAX_LEN, seed=999)
    print(f"task: G={G} groups x L={L} tokens, interleaved | device={DEVICE} | "
          f"chance acc={1 / N_CLASSES:.2f}")
    for mode in ("none", "full"):
        set_seed(0)
        model = build(mode)
        train(model, tr_t, tr_y)
        acc = accuracy(model, te_t, te_y)
        if mode == "full":
            ari = cluster_recovery(model, te_t, te_g)
            print(f"  {mode:5s}  acc={acc:.3f}  cluster-recovery ARI={ari:.3f}  "
                  f"({count_params(model):,}p)")
            viz(model, te_t, te_g)
        else:
            print(f"  {mode:5s}  acc={acc:.3f}  ({count_params(model):,}p)")


if __name__ == "__main__":
    main()
