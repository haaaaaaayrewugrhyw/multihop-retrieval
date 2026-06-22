"""
Structural-bias attention -- the user's refined idea, built faithfully.

Cluster is NOT part of Q/K/V. Q/K/V are CONTENT-ONLY. The cluster is a relational
property and enters as an additive bias on the attention SCORE -- parallel to a
positional bias, but its own form:

    score_ij = (q_i . k_j)/sqrt(dh)            # content
             [+ position term, see pos_mode]
             + lambda * P_i^T A P_j            # STRUCTURAL bias: "do i,j share a cluster?"

A (KxK, learnable, init = I) is a group-affinity matrix; A=I => lambda*<P_i,P_j>.

Bidirectional coupling: the cluster-biased attention PROPAGATES the clusters
(P_prop = attn @ P, label propagation / islands of agreement), blended with a
content readout.

Position has two modes (the user asked to test BOTH):
  pos_mode='rope'    : position via RoPE, INDEPENDENT of cluster, never updated.
  pos_mode='cluster' : position is a learned vector p that (a) enters the score as
                       its own bias lam_p*<Wp(p_i),Wp(p_j)>, and (b) is refined by
                       the cluster-biased attention (p <- attn @ p) -> same-cluster
                       tokens' positions converge. "Position depends on cluster."

use_cluster=False -> plain RoPE transformer baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 10


def build_rope(T, dh, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):                      # x: (B,H,T,dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.stack([r1, r2], dim=-1).flatten(-2)


class StructLayer(nn.Module):
    def __init__(self, d, heads, K, use_cluster, pos_mode):
        super().__init__()
        self.h, self.dh, self.K = heads, d // heads, K
        self.use_cluster, self.pos_mode = use_cluster, pos_mode
        self.scale = (d // heads) ** -0.5
        self.pscale = d ** -0.5
        self.ln1 = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d); self.Wo = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        if use_cluster:
            self.A = nn.Parameter(torch.eye(K))
            self.lam = nn.Parameter(torch.tensor(1.0))
            self.Wc = nn.Linear(d, K)
            self.gate = nn.Parameter(torch.tensor(0.0))
        if pos_mode == "cluster":
            self.lnp = nn.LayerNorm(d)
            self.Wp = nn.Linear(d, d)
            self.lam_p = nn.Parameter(torch.tensor(1.0))
            self.gate_p = nn.Parameter(torch.tensor(0.0))

    def forward(self, h, P, p, pad_mask, cos, sin):
        B, T, d = h.shape
        x = self.ln1(h)
        q = self.Wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        if self.pos_mode == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        score = (q @ k.transpose(-2, -1)) * self.scale
        if self.pos_mode == "cluster":                              # learned position as score bias
            pk = self.Wp(self.lnp(p))
            score = score + (self.lam_p * (pk @ pk.transpose(1, 2)) * self.pscale).unsqueeze(1)
        if self.use_cluster:                                        # structural bias
            score = score + (self.lam * ((P @ self.A) @ P.transpose(1, 2))).unsqueeze(1)
        score = score.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = score.softmax(-1)
        h = h + self.Wo((attn @ v).transpose(1, 2).reshape(B, T, d))
        h = h + self.ff(self.ln2(h))

        attn_avg = attn.mean(1) if (self.use_cluster or self.pos_mode == "cluster") else None
        if self.use_cluster:                                        # attention propagates clusters
            P_prop = attn_avg @ P
            P_content = F.softmax(self.Wc(h), dim=-1)
            g = torch.sigmoid(self.gate)
            P = g * P_prop + (1 - g) * P_content
            P = P * (~pad_mask).unsqueeze(-1).float()
            P = P / (P.sum(-1, keepdim=True) + 1e-9)
        if self.pos_mode == "cluster":                             # attention refines position
            gp = torch.sigmoid(self.gate_p)
            p = gp * (attn_avg @ p) + (1 - gp) * p
        return h, P, p, attn_avg


class StructNet(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, K, n_classes, max_len,
                 use_cluster=True, pos_mode="rope", pad_id=PAD):
        super().__init__()
        self.d, self.K, self.heads, self.max_len = d, K, heads, max_len
        self.pad_id, self.use_cluster, self.pos_mode = pad_id, use_cluster, pos_mode
        self.tok = nn.Embedding(vocab, d)
        if use_cluster:
            self.Wc0 = nn.Linear(d, K)
        if pos_mode == "cluster":
            self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(
            [StructLayer(d, heads, K, use_cluster, pos_mode) for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)
        self._last_P = self._last_attn = self._last_pad = None

    def forward(self, tokens):
        B, T = tokens.shape
        pad = (tokens == self.pad_id)
        not_pad = (~pad).unsqueeze(-1).float()
        posids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        cos, sin = build_rope(T, self.d // self.heads, tokens.device)
        h = self.tok(tokens)
        P = None
        if self.use_cluster:
            P = F.softmax(self.Wc0(h), dim=-1) * not_pad
            P = P / (P.sum(-1, keepdim=True) + 1e-9)
        p = self.pos(posids) if self.pos_mode == "cluster" else None
        attn = None
        for layer in self.layers:
            h, P, p, attn = layer(h, P, p, pad, cos, sin)
        self._last_P, self._last_attn, self._last_pad = P, attn, pad
        pooled = (h * not_pad).sum(1) / not_pad.sum(1).clamp(min=1.0)
        return self.head(pooled)


def aux_cluster_loss(model, attr_w=1.0, bal_w=1.0):
    """Unsupervised pressure (DCAT/modularity spirit):
       attraction  -- high-attention pairs should share a cluster
       balance     -- cluster usage spread out (anti-collapse)."""
    P, attn, pad = model._last_P, model._last_attn, model._last_pad
    if P is None:
        return torch.zeros((), device=next(model.parameters()).device)
    not_pad = (~pad).float()
    same = P @ P.transpose(1, 2)
    attr = -(attn * same).sum(-1).mean()
    meanP = (P * not_pad.unsqueeze(-1)).sum(1) / not_pad.sum(1, keepdim=True).clamp(min=1.0)
    bal = (meanP * (meanP + 1e-9).log()).sum(-1).mean()
    return attr_w * attr + bal_w * bal


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
