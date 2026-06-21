"""
ListOps-style nested-expression generator -- written from scratch.

An expression is OP( arg arg ... ) where each arg is a digit or a nested
expression. Operators keep the result in 0-9, so the task is 10-way
classification of the evaluated result:
    MAX, MIN, MED (lower median), SMOD (sum mod 10).

Generation produces a tree of EXACTLY the requested nesting depth -- one branch
is forced to recurse to the bottom while siblings are shallower -- so sequence
length stays roughly linear in depth rather than exponential.

Token ids:
    0-9    digits
    10-13  MAX, MIN, MED, SMOD
    14     CLOSE  ')'
    15     PAD
"""

import numpy as np

N_DIGITS = 10
MAXOP, MINOP, MEDOP, SMODOP = 10, 11, 12, 13
CLOSE = 14
PAD = 15
VOCAB = 16
OPS = [MAXOP, MINOP, MEDOP, SMODOP]


def _eval(op, vals):
    if op == MAXOP:
        return max(vals)
    if op == MINOP:
        return min(vals)
    if op == MEDOP:
        return sorted(vals)[(len(vals) - 1) // 2]      # lower median, stays in 0-9
    return sum(vals) % 10                               # SMOD


def _gen(rng, depth, max_args):
    """Build an expression of EXACT nesting depth `depth`; return (tokens, value)."""
    if depth == 0:
        v = int(rng.randint(0, N_DIGITS))
        return [v], v
    op = int(rng.choice(OPS))
    n_args = int(rng.randint(2, max_args + 1))
    deep_arg = int(rng.randint(0, n_args))             # this arg recurses to the bottom
    toks, vals = [op], []
    for a in range(n_args):
        sub_depth = depth - 1 if a == deep_arg else int(rng.randint(0, depth))
        sub_toks, sub_val = _gen(rng, sub_depth, max_args)
        toks += sub_toks
        vals.append(sub_val)
    toks.append(CLOSE)
    return toks, _eval(op, vals)


def make(n, depths, max_len=96, max_args=3, seed=0):
    """Return (tokens[n, max_len] int64, labels[n] int64).

    depths: tuple of nesting depths to sample uniformly from.
    Over-length expressions are rejected and resampled.
    """
    rng = np.random.RandomState(seed)
    toks = np.full((n, max_len), PAD, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    i = guard = 0
    while i < n:
        guard += 1
        if guard > 200 * n + 10000:
            raise RuntimeError("too many over-length rejects; raise max_len or lower max_args/depth")
        t, v = _gen(rng, int(rng.choice(depths)), max_args)
        if len(t) > max_len:
            continue
        toks[i, :len(t)] = t
        labels[i] = v
        i += 1
    return toks, labels


if __name__ == "__main__":
    import collections
    t, y = make(5, (2,), seed=0)
    print("tokens[0]:", t[0][t[0] != PAD].tolist(), "-> label", y[0])
    _, yy = make(4000, (1, 2), seed=1)
    print("label distribution:", dict(sorted(collections.Counter(yy.tolist()).items())))
