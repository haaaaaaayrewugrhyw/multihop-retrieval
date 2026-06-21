"""
Synthetic "grouped sequences" for the cluster-transformer screen.

A sequence = K sentences of L digits each, with a <sep> after every sentence.
Two label rules over the SAME inputs:

  variant A (grouping NEEDED)    : label = index of the sentence with the
                                   largest digit-sum  -> K-way classification.
                                   Requires per-sentence aggregation + compare.
  variant B (grouping IRRELEVANT): label = is the total digit-sum >= its
                                   expected midpoint -> binary. A global
                                   magnitude property; learnable, and sentence
                                   structure does not matter. (Parity was
                                   unlearnable -> a useless control.)

We also return segment ids (which sentence each token is in) so BOTH the
baseline (via segment embeddings) and the cluster model (via cluster init)
know the sentence boundaries -- the only difference under test is the evolving
cluster mechanism, not knowledge of the boundaries.
"""

import numpy as np

DIGITS = 10
SEP = 10
PAD = 11
VOCAB = 12          # digits 0-9, SEP=10, PAD=11


def make(n, K=4, L=6, variant="A", seed=0):
    rng = np.random.RandomState(seed)
    T = K * (L + 1)                       # each sentence: L digits + 1 sep
    toks = np.full((n, T), PAD, dtype=np.int64)
    segs = np.zeros((n, T), dtype=np.int64)
    labels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        pos = 0
        sums = []
        for k in range(K):
            d = rng.randint(0, DIGITS, size=L)
            sums.append(int(d.sum()))
            for j in range(L):
                toks[i, pos] = d[j]; segs[i, pos] = k; pos += 1
            toks[i, pos] = SEP; segs[i, pos] = k; pos += 1     # sep joins its sentence
        if variant == "A":
            labels[i] = int(np.argmax(sums))         # which sentence has largest sum
        else:                                        # variant B
            mid = K * L * 4.5                        # expected total sum (digits ~U[0,9])
            labels[i] = int(sum(sums) >= mid)        # global magnitude, no grouping
    return toks, segs, labels


def num_classes(variant, K):
    return K if variant == "A" else 2


if __name__ == "__main__":
    t, s, y = make(3, variant="A")
    print("tokens\n", t)
    print("segments\n", s)
    print("labels (A=which sentence has max sum)", y)
