"""
ListOps transformer with a custom attention so we can inject a cluster
AFFINITY BIAS into the attention scores.

cluster_mode:
  'none'     : vanilla transformer (baseline).
  'implicit' : cluster vector feeds the hidden state, refined each layer
               (the version already tested -- kept for completeness).
  'bias'     : EXPLICIT affinity bias. Each token has a cluster distribution
               P (init from position buckets, refined each layer). Attention
               score_ij gets  + lambda_l * (P_i . P_j)  so tokens that share
               a cluster attend to each other more. lambda is a learnable
               per-layer scalar (init 1) so the network can strengthen or kill
               the bias as it likes -- the fairest shot for the mechanism.
No segment embeddings: the grouping must be discovered.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD


class MHA(nn.Module):
    """Multi-head attention with an optional additive score bias (B,T,T)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, kpm, bias=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # B,h,T,dh
        s = (q @ k.transpose(-2, -1)) * self.scale       # B,h,T,T
        if bias is not None:
            s = s + bias.unsqueeze(1)                    # broadcast bias over heads
        s = s.masked_fill(kpm[:, None, None, :], float("-inf"))
        a = s.softmax(dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kpm, bias=None):
        x = x + self.attn(self.ln1(x), kpm, bias)
        x = x + self.ff(self.ln2(x))
        return x


class ListOpsTransformer(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, n_classes, K, max_len,
                 cluster_mode="none"):
        super().__init__()
        self.mode = cluster_mode
        self.K = K
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([Block(d, heads) for _ in range(n_layers)])
        if cluster_mode in ("implicit", "bias"):
            self.h2c = nn.ModuleList([nn.Linear(d, K) for _ in range(n_layers)])
        if cluster_mode == "implicit":
            self.c2h = nn.ModuleList([nn.Linear(K, d) for _ in range(n_layers)])
        if cluster_mode == "bias":
            self.lam = nn.Parameter(torch.ones(n_layers))   # per-layer affinity strength
        self.head = nn.Linear(d, n_classes)

    def forward(self, toks):
        B, T = toks.shape
        kpm = (toks == PAD)
        posids = torch.arange(T, device=toks.device).unsqueeze(0).expand(B, T)
        h = self.tok(toks) + self.pos(posids)
        if self.mode in ("implicit", "bias"):
            bucket = (posids * self.K // self.max_len).clamp(max=self.K - 1)
            c = F.one_hot(bucket, self.K).float()

        for l, layer in enumerate(self.layers):
            inp, bias = h, None
            if self.mode == "implicit":
                inp = inp + self.c2h[l](torch.softmax(c, dim=-1))
            if self.mode == "bias":
                P = torch.softmax(c, dim=-1)                  # B,T,K
                bias = self.lam[l] * (P @ P.transpose(1, 2))  # B,T,T affinity
            h = layer(inp, kpm, bias)
            if self.mode in ("implicit", "bias"):
                c = c + self.h2c[l](h)

        m = (~kpm).float().unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
