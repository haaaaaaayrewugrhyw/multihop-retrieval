"""
Interleaved-group diagnostic task -- "can the cluster mechanism SEE the groups?"

Each sequence has G groups of L digit tokens. A token encodes (group, digit) as
the id  group*10 + digit. Tokens are SHUFFLED (interleaved) so groups are
NON-contiguous -- the position-bucket cluster init is therefore wrong, and the
mechanism must discover the grouping from content (group = token // 10).

Label = which group has the largest digit-sum (G-way). Solving it needs
per-group aggregation, so grouping is genuinely useful.

make_groups() also returns the ground-truth per-token group id, so we can score
how well the learned cluster membership recovers the true groups (ARI) and
visualize it -- the analog of FG-ARI / slot masks for this architecture.
"""

import numpy as np

G = 4
L = 5
PAD = G * 10              # 40
VOCAB = G * 10 + 1        # 41  (ids 0..39 = group*10+digit, 40 = PAD)
N_CLASSES = G


def make_groups(n, max_len=24, seed=0):
    rng = np.random.RandomState(seed)
    T = G * L
    assert T <= max_len
    toks = np.full((n, max_len), PAD, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    groups = np.full((n, max_len), -1, dtype=np.int64)          # -1 at PAD
    for i in range(n):
        items, sums = [], np.zeros(G, dtype=int)
        for g in range(G):
            for _ in range(L):
                dgt = int(rng.randint(0, 10))
                items.append((g, dgt)); sums[g] += dgt
        rng.shuffle(items)                                      # interleave the groups
        for pos, (g, dgt) in enumerate(items):
            toks[i, pos] = g * 10 + dgt
            groups[i, pos] = g
        labels[i] = int(np.argmax(sums))
    return toks, labels, groups


if __name__ == "__main__":
    t, y, g = make_groups(2, seed=0)
    print("token ids:", t[0][t[0] != PAD].tolist())
    print("groups   :", g[0][g[0] != -1].tolist())
    print("label (largest-sum group):", y[0])
