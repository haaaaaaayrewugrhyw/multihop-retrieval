"""
Transformer for the ListOps discovery task, with an optional cluster mechanism.

No segment embeddings here -- the grouping (bracket structure) is NOT given;
the model must discover it. The cluster mechanism is the implicit version:
each token carries a cluster-membership vector that feeds each layer and is
refined each layer from the hidden state. It is INITIALIZED from position
(contiguous position buckets) -- a neutral starting partition the network can
reshape toward the true bracket groups. Variable positions are dropped (they
hurt in the previous screen).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kpm):
        z = self.ln1(x)
        a, _ = self.attn(z, z, z, key_padding_mask=kpm, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class ListOpsTransformer(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, n_classes, K, max_len,
                 use_cluster=False):
        super().__init__()
        self.use_cluster = use_cluster
        self.K = K
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([Block(d, heads) for _ in range(n_layers)])
        if use_cluster:
            self.c2h = nn.ModuleList([nn.Linear(K, d) for _ in range(n_layers)])
            self.h2c = nn.ModuleList([nn.Linear(d, K) for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, toks):
        B, T = toks.shape
        kpm = (toks == PAD)                                  # True where pad
        posids = torch.arange(T, device=toks.device).unsqueeze(0).expand(B, T)
        h = self.tok(toks) + self.pos(posids)
        if self.use_cluster:
            bucket = (posids * self.K // self.max_len).clamp(max=self.K - 1)
            c = F.one_hot(bucket, self.K).float()            # init partition from position

        for l, layer in enumerate(self.layers):
            inp = h
            if self.use_cluster:
                inp = inp + self.c2h[l](torch.softmax(c, dim=-1))
            h = layer(inp, kpm)
            if self.use_cluster:
                c = c + self.h2c[l](h)

        m = (~kpm).float().unsqueeze(-1)                     # masked mean-pool
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
