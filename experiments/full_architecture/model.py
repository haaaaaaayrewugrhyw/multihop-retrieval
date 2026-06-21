"""
Full cluster-transformer architecture -- written from scratch (no reuse).

Each token carries THREE explicit streams, each with its own update route:
    content  h : the usual token representation
    cluster  c : per-token logits over K clusters (softmaxed to a distribution P)
    position p : positional information (fixed, or a refined stream)

Per layer, when cluster_mode == "full":
    (1) cluster -> content  : h_in += c2h(softmax(c))          # cluster identity informs content
    (2) cluster -> attention: scores += lam * (P @ P^T)        # same-cluster tokens attend more
    (3) attention + MLP update the content stream
    (4) cluster refine      : c <- c + h2c(h)                  # EM-flavored re-estimation
    (5) position            : fixed (added once) OR variable (added each layer + refined)

cluster_mode="none" with variable_pos=False is a plain vanilla transformer (the baseline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 15   # padding token id (kept consistent with data.py)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with an optional score bias."""

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0, "d must be divisible by heads"
        self.heads = heads
        self.dh = d // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad_mask, attn_bias=None):
        # x: (B,T,d); pad_mask: (B,T) True at PAD; attn_bias: (B,T,T) or None
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each (B,heads,T,dh)
        scores = (q @ k.transpose(-2, -1)) * self.scale        # (B,heads,T,T)
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1)           # broadcast bias over heads
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-LN block: attention sublayer (accepts a score bias) + GELU MLP sublayer."""

    def __init__(self, d, heads, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(),
                                 nn.Linear(mlp_mult * d, d))

    def forward(self, x, pad_mask, attn_bias=None):
        x = x + self.attn(self.ln1(x), pad_mask, attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class FullArchitecture(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, n_classes, K, max_len,
                 cluster_mode="none", variable_pos=False, bias_clamp=10.0):
        super().__init__()
        assert cluster_mode in ("none", "full")
        self.cluster_mode = cluster_mode
        self.variable_pos = variable_pos
        self.K = K
        self.max_len = max_len
        self.bias_clamp = bias_clamp

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, heads) for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)

        if cluster_mode == "full":
            self.c2h = nn.ModuleList([nn.Linear(K, d) for _ in range(n_layers)])
            self.h2c = nn.ModuleList([nn.Linear(d, K) for _ in range(n_layers)])
            self.lam = nn.Parameter(torch.ones(n_layers))     # per-layer affinity strength
        if variable_pos:
            self.p2h = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, tokens):
        B, T = tokens.shape
        pad_mask = (tokens == PAD)
        posids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.tok(tokens)
        p = self.pos(posids)
        if not self.variable_pos:
            h = h + p                                          # fixed: position added once

        if self.cluster_mode == "full":
            bucket = (posids * self.K // self.max_len).clamp(max=self.K - 1)
            c = F.one_hot(bucket, self.K).float()              # cluster init from position

        for l, block in enumerate(self.blocks):
            inp, attn_bias = h, None
            if self.variable_pos:
                inp = inp + p                                  # position feeds every layer
            if self.cluster_mode == "full":
                P = torch.softmax(c, dim=-1)                   # (B,T,K)
                inp = inp + self.c2h[l](P)                     # (1) cluster -> content
                attn_bias = self.lam[l] * (P @ P.transpose(1, 2))     # (2) cluster -> attention
                attn_bias = attn_bias.clamp(-self.bias_clamp, self.bias_clamp)
            h = block(inp, pad_mask, attn_bias)                # (3) attn + MLP update content
            if self.cluster_mode == "full":
                c = c + self.h2c[l](h)                         # (4) refine cluster from content
            if self.variable_pos:
                p = p + self.p2h[l](h)                         # refine position from content

        m = (~pad_mask).float().unsqueeze(-1)                  # masked mean-pool over non-PAD
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
