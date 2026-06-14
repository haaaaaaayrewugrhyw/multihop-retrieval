"""
losses.py -- L_recon, L_sparsity, L_specificity for delta-system training.
"""

import torch
import torch.nn.functional as F
from model import D_MODEL, VOCAB_SIZE

TARGET_SPARSITY = 0.30   # delta should use at most 30% of its capacity


def recon_loss(logits: torch.Tensor, B_ids: torch.Tensor, B_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy reconstruction loss over real B tokens only."""
    labels = B_ids.clone()
    labels[~B_mask.bool()] = -100
    return F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),
        labels.view(-1),
        ignore_index=-100,
    )


def sparsity_loss(delta: torch.Tensor, B_mask: torch.Tensor) -> torch.Tensor:
    """
    One-sided penalty: push mean(||delta[t]|| / sqrt(d)) below TARGET_SPARSITY.
    Zero gradient when already sparse enough (one-sided).
    """
    norms     = delta.norm(dim=-1) / (D_MODEL ** 0.5)   # [b, T], normalized ~[0,1]
    real      = B_mask.float()
    mean_norm = (norms * real).sum() / real.sum().clamp(min=1)
    return F.relu(mean_norm - TARGET_SPARSITY)


def gate_loss(alpha: torch.Tensor, B_mask: torch.Tensor) -> torch.Tensor:
    """
    Sparsity penalty on the per-position gate: beta * mean(alpha over real tokens).
    Forces gate closed everywhere EXCEPT where D_recon needs it (novel positions).
    beta is ramped up from 0 during training (warmup prevents gate collapse).
    """
    real = B_mask.float()
    return (alpha * real).sum() / real.sum().clamp(min=1)


def ortho_loss(delta: torch.Tensor, H_A: torch.Tensor,
               A_mask: torch.Tensor, B_mask: torch.Tensor) -> torch.Tensor:
    """
    Anti-collapse orthogonality penalty.

    L_ortho = mean over real B tokens of  max_j |cos(delta[t], H_A[j])|

    Pushes each delta token to be orthogonal to ALL of A's token representations.
    Since B = A + novel, encode(B) overlaps A on the shared content; penalizing
    delta's alignment with A squeezes delta toward the B-beyond-A (novel) component
    and forces the generator to actually USE A. Range [0, 1] (lower = more orthogonal).

    Target metric this is meant to move: the eval's A-dependence fraction (was ~6%).
    """
    d = F.normalize(delta, dim=-1)                 # [b, T_B, D]
    a = F.normalize(H_A,  dim=-1)                  # [b, T_A, D]
    cos = torch.bmm(d, a.transpose(1, 2)).abs()    # [b, T_B, T_A]
    a_valid = A_mask.unsqueeze(1).float()          # [b, 1, T_A] -- ignore padded A
    cos = cos * a_valid
    sim = cos.max(dim=-1).values                   # [b, T_B] worst-case alignment
    real = B_mask.float()
    return (sim * real).sum() / real.sum().clamp(min=1)


def kl_loss(kl_elementwise: torch.Tensor, B_mask: torch.Tensor) -> torch.Tensor:
    """
    Aggregate the VIB per-element KL (model.last_kl, [b, T, D]) over real B tokens:
    sum over the latent dim, mean over real tokens.

    Penalizing this limits the BITS delta carries. Since the decoder already has all of A
    for free, the cheapest way to keep recon low under a bit-budget is to encode only what
    A lacks -> the novelty. This is the SOFT Information Bottleneck. It is NOT the magnitude
    penalty sparsity_loss (which penalizes size, not information, and did not break collapse).
    """
    kl_tok = kl_elementwise.sum(dim=-1)            # [b, T]
    real   = B_mask.float()
    return (kl_tok * real).sum() / real.sum().clamp(min=1)


def specificity_loss(
    logits_correct: torch.Tensor,
    logits_wrong:   torch.Tensor,
    B_ids:          torch.Tensor,
    B_mask:         torch.Tensor,
    margin:         float = 0.5,
) -> torch.Tensor:
    """
    Margin ranking loss: correct delta must reconstruct B better than wrong delta.

    L_spec = ReLU( L_correct - L_wrong + margin )

    L_correct = CE( D_recon(A_i, delta_i),           B_i )   <- should be low
    L_wrong   = CE( D_recon(A_i, delta_{(i+1)%b}),   B_i )   <- should be high

    Gradient teaches G to produce pair-specific delta and D_recon to rely on it.
    Active (non-zero) only when the margin is not satisfied.
    """
    L_c = recon_loss(logits_correct, B_ids, B_mask)
    L_w = recon_loss(logits_wrong,   B_ids, B_mask)
    return F.relu(L_c - L_w + margin)
