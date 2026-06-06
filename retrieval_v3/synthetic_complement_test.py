"""
synthetic_complement_test.py -- does the complement OPERATION recover a known complement?
=========================================================================================

Pure synthetic, no BERT, no training, CPU, runs in seconds.

Setup (known ground truth by construction):
  - A vocabulary of "concept tokens", each a fixed random unit embedding (~orthogonal).
  - A = SHARED tokens  +  A-only tokens
  - B = SHARED tokens  +  NEW tokens          (NEW = the ground-truth complement)
  - overlap = |SHARED| / |B|   (swept high -> low)

We run the exact complement operation from generator_train._generate_edge
(raw dot-product cross-attention + lambda subtraction + complement gate + gated pool)
with NO learned weights (identity projections), to test the MATHEMATICAL PRINCIPLE
independent of any training.

Decisive metrics (averaged over many random trials per overlap level):
  cos(edge, true_complement)   -- does the operation recover the added info?
  cos(encode_B, true_complement) -- does plain mean-pool(B) recover it? (baseline)
  ADVANTAGE = the difference  -- how much the complement operation BEATS plain encode(B)

Interpretation:
  - high overlap: encode(B) is polluted by SHARED tokens -> bad complement;
                  the operation should subtract them -> big ADVANTAGE  (idea works)
  - low overlap (MuSiQue regime): B is mostly NEW -> encode(B) ~= complement anyway
                  -> ADVANTAGE -> 0   (this reproduces the MuSiQue degeneracy)
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


def make_vocab(n_factors: int, dim: int) -> torch.Tensor:
    """Random ~orthogonal unit embeddings for concept tokens."""
    emb = torch.randn(n_factors, dim)
    return F.normalize(emb, dim=-1)


def make_pair(vocab, n_b, overlap_frac, n_a_only, dim):
    """
    Build one (A, B) pair with controlled overlap.
    Returns A_sa [T_A, D], B_sa [T_B, D], true_complement [D].
    """
    n_factors = vocab.size(0)
    n_shared = int(round(n_b * overlap_frac))
    n_new    = n_b - n_shared

    perm = torch.randperm(n_factors)
    shared = perm[:n_shared]
    new    = perm[n_shared:n_shared + n_new]
    a_only = perm[n_shared + n_new:n_shared + n_new + n_a_only]

    A_idx = torch.cat([shared, a_only])
    B_idx = torch.cat([shared, new])

    A_sa = vocab[A_idx]                       # [T_A, D]
    B_sa = vocab[B_idx]                       # [T_B, D]
    # ground-truth complement = mean of the NEW tokens (the added info)
    true_comp = (vocab[new].mean(0) if n_new > 0 else torch.zeros(dim))
    return A_sa, B_sa, F.normalize(true_comp, dim=-1) if n_new > 0 else true_comp


def complement_op(A_sa, B_sa, lam=1.0, use_gate=True, tau=None, scale_sub=False):
    """
    Complement operation.  tau = attention temperature (None -> sqrt(D), the
    current architecture). scale_sub=True multiplies the subtraction by each
    B token's actual match strength (max attn), so NOVEL tokens (no match in A)
    are NOT corrupted by a spurious mean(A) subtraction -- the proposed fix.
    """
    D = B_sa.size(-1)
    denom  = (D ** 0.5) if tau is None else tau
    scores = (B_sa @ A_sa.T) / denom              # [T_B, T_A]
    attn   = torch.softmax(scores, dim=-1)
    B_in_A = attn @ A_sa                          # [T_B, D]
    match  = attn.max(dim=-1).values              # [T_B] how much this token is in A

    if scale_sub:
        comp_tokens = B_sa - lam * match.unsqueeze(-1) * B_in_A   # subtract only real matches
    else:
        comp_tokens = B_sa - lam * B_in_A                          # current architecture

    g = (1.0 - match) if use_gate else torch.ones(B_sa.size(0))
    g = g.clamp(min=0)
    edge_raw = (g.unsqueeze(-1) * comp_tokens).sum(0) / g.sum().clamp(min=1e-9)
    return F.normalize(edge_raw, dim=-1)


def encode_B(B_sa):
    """Plain mean-pool of B (the baseline 'encode(B)')."""
    return F.normalize(B_sa.mean(0), dim=-1)


def _sweep(vocab, variant, n_trials, n_b, n_a_only, dim, lam, use_gate):
    overlaps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    rows = []
    for ov in overlaps:
        e_true, b_true = [], []
        for _ in range(n_trials):
            A_sa, B_sa, true_comp = make_pair(vocab, n_b, ov, n_a_only, dim)
            if true_comp.norm() < 1e-6:
                continue
            edge = complement_op(A_sa, B_sa, lam=lam, use_gate=use_gate, **variant)
            encB = encode_B(B_sa)
            e_true.append(float(edge @ true_comp))
            b_true.append(float(encB @ true_comp))
        if e_true:
            ct, cb = np.mean(e_true), np.mean(b_true)
            rows.append((ov, ct, cb, ct - cb))
    return rows


def _print_rows(title, rows):
    print(f"\n### {title}")
    print(f"{'overlap':>8} | {'cos(edge,true)':>15} | {'cos(encB,true)':>15} | {'ADVANTAGE':>10}")
    print("-" * 60)
    for ov, ct, cb, adv in rows:
        print(f"{ov:>8.2f} | {ct:>15.4f} | {cb:>15.4f} | {adv:>+10.4f}")
    high = np.mean([r[3] for r in rows if r[0] >= 0.6]) if rows else 0
    low  = np.mean([r[3] for r in rows if r[0] <= 0.2]) if rows else 0
    print(f"  ADV(high overlap>=0.6)={high:+.4f}   ADV(low overlap<=0.2)={low:+.4f}")
    return high, low


def run(n_trials, n_b, n_a_only, dim, n_factors, lam, use_gate):
    vocab = make_vocab(n_factors, dim)

    variants = {
        "CURRENT  (softmax sub, tau=sqrt(D))":          dict(tau=None, scale_sub=False),
        "SHARPER  (tau=0.1, softmax sub)":              dict(tau=0.1,  scale_sub=False),
        "FIX: match-scaled sub (tau=0.1)":              dict(tau=0.1,  scale_sub=True),
    }
    results = {}
    for name, v in variants.items():
        rows = _sweep(vocab, v, n_trials, n_b, n_a_only, dim, lam, use_gate)
        results[name] = _print_rows(name, rows)

    print("\n" + "=" * 70)
    print("  SUMMARY  (want: ADV high>0, ADV low~0  =>  operation recovers complement")
    print("            only when there's overlap to subtract, clean otherwise)")
    print("=" * 70)
    print(f"  {'variant':<40}{'ADV_high':>10}{'ADV_low':>10}")
    for name, (hi, lo) in results.items():
        print(f"  {name:<40}{hi:>+10.4f}{lo:>+10.4f}")
    print("=" * 70)
    best = max(results.items(), key=lambda kv: kv[1][0] - abs(kv[1][1]))
    print(f"  BEST variant: {best[0]}")
    bh, bl = best[1]
    if bh > 0.15 and bl > -0.05:
        print("  VERDICT: IDEA IS SOUND with the right operation. The complement is")
        print("  recoverable when overlap exists; degenerates cleanly when it doesn't.")
        print("  -> MuSiQue (near-zero overlap) was the wrong testbed, AND the current")
        print("     softmax-subtraction operation has a fixable leak (use match-scaled sub).")
    else:
        print("  VERDICT: even the best variant doesn't cleanly recover the complement.")
        print("  The operation needs rethinking, not just tuning.")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",  type=int,   default=2000)
    p.add_argument("--n_b",       type=int,   default=12,  help="tokens per B doc")
    p.add_argument("--n_a_only",  type=int,   default=12,  help="A-only tokens")
    p.add_argument("--dim",       type=int,   default=64)
    p.add_argument("--n_factors", type=int,   default=500)
    p.add_argument("--lam",       type=float, default=1.0)
    p.add_argument("--no_gate",   action="store_true")
    a = p.parse_args()
    print("Synthetic complement-operation test")
    print(f"  dim={a.dim} vocab={a.n_factors} n_b={a.n_b} n_a_only={a.n_a_only} "
          f"lambda={a.lam} gate={not a.no_gate} trials={a.n_trials}")
    run(a.n_trials, a.n_b, a.n_a_only, a.dim, a.n_factors, a.lam, not a.no_gate)
