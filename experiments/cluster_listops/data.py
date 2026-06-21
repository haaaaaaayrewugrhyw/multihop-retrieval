"""
Small ListOps-style nested expressions -- the FAIR test for the cluster idea.

Why this task:
  * grouping must be DISCOVERED (the bracket structure), not handed over as
    segment embeddings -> the cluster mechanism finally has a real job.
  * it is hierarchical/nested, the regime where flat attention struggles.
  * depth is controllable, so we can train shallow and test deep = OOD /
    compositional generalization, which is where structured priors should win.

An expression is OP( arg arg ... ) where args are digits or sub-expressions.
Operators keep the result in 0-9, so it's 10-way classification.

  MAX / MIN / MED(ian) / SMOD(sum mod 10)

Each generated tree has nesting depth == the requested depth (one branch is
forced to recurse to the bottom; siblings are shallower), so lengths stay
roughly linear in depth, not exponential.
"""

import numpy as np

DIGITS = 10
MAX, MIN, MED, SMOD = 10, 11, 12, 13
CLOSE = 14
PAD = 15
VOCAB = 16
OPS = [MAX, MIN, MED, SMOD]


def apply_op(op, vals):
    if op == MAX:
        return max(vals)
    if op == MIN:
        return min(vals)
    if op == MED:
        return sorted(vals)[(len(vals) - 1) // 2]      # lower median, stays in 0-9
    return sum(vals) % 10                               # SMOD


def gen(rng, depth, max_args=3):
    """Return (tokens, value) for a tree of nesting depth exactly `depth`."""
    if depth == 0:
        v = int(rng.randint(0, DIGITS))
        return [v], v
    op = int(rng.choice(OPS))
    n = int(rng.randint(2, max_args + 1))
    deep = int(rng.randint(0, n))                       # this arg goes full depth
    toks = [op]
    vals = []
    for a in range(n):
        d = depth - 1 if a == deep else int(rng.randint(0, depth))
        st, sv = gen(rng, d, max_args)
        toks += st
        vals.append(sv)
    toks += [CLOSE]
    return toks, apply_op(op, vals)


def make(n, depths, max_len=64, max_args=3, seed=0):
    rng = np.random.RandomState(seed)
    toks = np.full((n, max_len), PAD, dtype=np.int64)
    labels = np.zeros((n,), dtype=np.int64)
    i = 0
    guard = 0
    while i < n:
        d = int(rng.choice(depths))
        t, v = gen(rng, d, max_args)
        guard += 1
        if guard > 100 * n + 10000:
            raise RuntimeError("too many over-length rejects; raise max_len")
        if len(t) > max_len:
            continue
        toks[i, :len(t)] = t
        labels[i] = v
        i += 1
    return toks, labels


if __name__ == "__main__":
    import collections
    t, y = make(5, depths=(2,), seed=0)
    print("example tokens (depth 2):", t[0][t[0] != PAD].tolist(), "-> label", y[0])
    _, yy = make(4000, depths=(1, 2), seed=1)
    print("label distribution:", collections.Counter(yy.tolist()))
