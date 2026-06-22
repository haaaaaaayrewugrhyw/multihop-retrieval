"""
Models for the de-interleaving task.

  BaselineDeint : plain transformer (token-token attention), pooled -> head.
  EMDeint       : the EM-attention architecture (reusing EMLayer), but with a
                  NEUTRAL position-bucket cluster init -- the source is latent,
                  so we do NOT hand it the source labels; it must discover them.

Both pool content and classify the source count. We separately read out the
final cluster probabilities P to measure source-recovery ARI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_em import EMLayer

PAD = 10


class MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh, self.scale = heads, d // heads, (d // heads) ** -0.5
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, pad_mask):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-2, -1)) * self.scale
        s = s.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        return self.o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, self.h * self.dh))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = MHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ff(self.ln2(x))
        return x


class BaselineDeint(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, n_classes, max_len, pad_id=PAD):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, tokens):
        B, T = tokens.shape
        pad = (tokens == self.pad_id)
        posids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.tok(tokens) + self.pos(posids)
        for blk in self.blocks:
            h = blk(h, pad)
        m = (~pad).float().unsqueeze(-1)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1.0))


class EMDeint(nn.Module):
    """readout='pool'   : classify pooled per-token content (tokens can bypass clusters)
       readout='centers': PURE BOTTLENECK -- classify only from the K cluster centers,
                          so token info reaches the output ONLY through the clusters.
       freeze_P=True    : ablation -- assignment stays at the neutral init (no learned E-step)."""

    def __init__(self, vocab, d, n_layers, heads, K, n_classes, max_len,
                 variable_pos=False, pad_id=PAD, readout="pool", freeze_P=False):
        super().__init__()
        self.d, self.K, self.max_len, self.pad_id = d, K, max_len, pad_id
        self.variable_pos, self.readout, self.freeze_P = variable_pos, readout, freeze_P
        self.collect_P, self._last_P = False, None
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([EMLayer(d, K, variable_pos) for _ in range(n_layers)])
        if readout == "centers":
            self.cluster_mlp = nn.Sequential(nn.Linear(d + 1, d), nn.GELU())
        self.head = nn.Linear(d, n_classes)

    def forward(self, tokens):
        B, T = tokens.shape
        pad = (tokens == self.pad_id)
        not_pad = (~pad).unsqueeze(-1).float()
        posids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.tok(tokens)
        p = self.pos(posids)
        # NEUTRAL init: position buckets (source is latent -> NOT given to the model)
        bucket = (posids * self.K // self.max_len).clamp(max=self.K - 1)
        P = F.one_hot(bucket, self.K).float() * not_pad
        counts = P.sum(1).unsqueeze(-1) + 1e-6
        mu = (P.transpose(1, 2) @ h) / counts
        mu_p = (P.transpose(1, 2) @ p) / counts if self.variable_pos else None
        for layer in self.layers:
            h, p, P, mu, mu_p = layer(h, p, P, mu, mu_p, pad, freeze_P=self.freeze_P)
        if self.collect_P:
            self._last_P = P.detach()
        if self.readout == "centers":                       # PURE BOTTLENECK: output sees only centers
            soft_counts = P.sum(1, keepdim=True).transpose(1, 2)         # (B,K,1)
            feat = self.cluster_mlp(torch.cat([mu, torch.log1p(soft_counts)], dim=-1))
            return self.head(feat.sum(1))                   # sum over clusters (count-sensitive)
        return self.head((h * not_pad).sum(1) / not_pad.sum(1).clamp(min=1.0))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
