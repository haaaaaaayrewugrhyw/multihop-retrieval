"""
De-interleaving task -- where clustering separation is NECESSARY, LATENT, and
VISIBLE (the sequence analog of N-EM bouncing balls / source separation).

Each example interleaves S latent "source" random walks into one digit stream:
  - S in [S_min, K] sources (K = max across dataset).
  - each source is a random walk over 0-9 (start random, each step +-1, clamped),
    of variable length L in [L_min, L_max].
  - the S walks are merged in random order PRESERVING each walk's internal order,
    so sources are interleaved and NON-contiguous. Each token's source is hidden.

The model must group tokens into their coherent walks (which scattered tokens
continue each other). Source is NOT a token feature -> it must be inferred from
value-continuity = a real binding/separation problem.

Label = number of sources S (permutation-invariant; forces source-identification).
We also return per-token source ids, so cluster-recovery ARI vs SOURCE is the
right separation metric -- this time we expect/want it to be high if it works.

Tokens: 0-9 digits, 10 = PAD.
"""

import numpy as np

DIGITS = 10
PAD = 10
VOCAB = 11


def max_len_for(K, L_max):
    return K * L_max


def n_classes_for(K, S_min):
    return K - S_min + 1                       # S ranges S_min..K


def _walk(rng, L):
    v = int(rng.randint(0, DIGITS))
    out = [v]
    for _ in range(L - 1):
        v = int(np.clip(v + rng.choice([-1, 1]), 0, DIGITS - 1))
        out.append(v)
    return out


def make(n, K=5, L_min=4, L_max=8, S_min=2, seed=0):
    rng = np.random.RandomState(seed)
    T = max_len_for(K, L_max)
    toks = np.full((n, T), PAD, dtype=np.int64)
    src = np.full((n, T), -1, dtype=np.int64)          # per-token source id; -1 at PAD
    labels = np.zeros(n, dtype=np.int64)
    nsrc = np.zeros(n, dtype=np.int64)
    for i in range(n):
        S = int(rng.randint(S_min, K + 1))
        walks = [_walk(rng, int(rng.randint(L_min, L_max + 1))) for _ in range(S)]
        ptr = [0] * S
        remaining = sum(len(w) for w in walks)
        pos = 0
        while remaining > 0:                            # interleave, preserving each walk's order
            avail = [s for s in range(S) if ptr[s] < len(walks[s])]
            s = int(rng.choice(avail))
            toks[i, pos] = walks[s][ptr[s]]
            src[i, pos] = s
            ptr[s] += 1; pos += 1; remaining -= 1
        labels[i] = S - S_min                           # class 0..K-S_min
        nsrc[i] = S
    return toks, src, labels, nsrc


if __name__ == "__main__":
    t, s, y, ns = make(3, seed=0)
    for i in range(3):
        m = t[i] != PAD
        print(f"  ex{i}  S={ns[i]} (label {y[i]})")
        print(f"        toks {t[i][m].tolist()}")
        print(f"        src  {s[i][m].tolist()}")
