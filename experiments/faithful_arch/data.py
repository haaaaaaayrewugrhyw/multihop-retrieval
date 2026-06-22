"""
Grouped digit-sentences with VARIABLE lengths -- from scratch.

Each example has a VARIABLE number of sentences S in [2, K] (K = max sentences
across the whole dataset, a fixed constant), and EACH sentence has a VARIABLE
number of digits L in [L_min, L_max]. A SEP token follows every sentence; the
whole sequence is padded to max_len = K*(L_max+1).

Label = which sentence has the largest digit-sum (in [0, S)). Solving it needs
per-sentence aggregation, so sentence grouping is genuinely useful.

make() returns, per token, the sentence id (for cluster init / segment
embedding), and per example the sentence count S (so the harness can mask
absent-sentence logits and keep the K-way head well-posed as S varies).

Token ids: 0-9 digits, 10 = SEP, 11 = PAD.
"""

import numpy as np

DIGITS = 10
SEP = 10
PAD = 11
VOCAB = 12


def max_len_for(K, L_max):
    return K * (L_max + 1)


def make(n, K=5, L_min=2, L_max=7, seed=0):
    rng = np.random.RandomState(seed)
    T = max_len_for(K, L_max)
    toks = np.full((n, T), PAD, dtype=np.int64)
    sent = np.full((n, T), -1, dtype=np.int64)          # per-token sentence id; -1 at PAD
    labels = np.zeros(n, dtype=np.int64)
    n_sent = np.zeros(n, dtype=np.int64)
    for i in range(n):
        S = int(rng.randint(2, K + 1))                  # variable sentence count
        pos, sums = 0, []
        for s in range(S):
            L = int(rng.randint(L_min, L_max + 1))      # variable sentence length
            tot = 0
            for _ in range(L):
                dgt = int(rng.randint(0, DIGITS))
                toks[i, pos] = dgt; sent[i, pos] = s; pos += 1; tot += dgt
            toks[i, pos] = SEP; sent[i, pos] = s; pos += 1   # SEP joins its sentence
            sums.append(tot)
        labels[i] = int(np.argmax(sums))                # among present sentences only
        n_sent[i] = S
    return toks, sent, labels, n_sent, T


if __name__ == "__main__":
    t, s, y, ns, T = make(3, seed=0)
    print("max_len:", T)
    for i in range(3):
        m = t[i] != PAD
        print(f"  ex{i}  S={ns[i]}  label={y[i]}")
        print(f"        toks {t[i][m].tolist()}")
        print(f"        sent {s[i][m].tolist()}")
