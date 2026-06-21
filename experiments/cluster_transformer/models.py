"""
A small transformer with two independent toggles, so we can run the 2x2:

    use_cluster  : carry a per-token cluster-membership vector (init = one-hot
                   sentence), feed it into each layer, and REFINE it each layer
                   from the hidden state (EM-inspired, IMPLICIT -- it only
                   touches attention through the hidden state; no hand-forced
                   affinity bias). This is option #1.
    variable_pos : treat position as a refinable stream (added + updated each
                   layer) instead of a fixed encoding added once.

baseline       = use_cluster=False, variable_pos=False
Both models ALWAYS get segment embeddings, so both know the sentence
boundaries; the only thing under test is the evolving cluster / variable-pos
mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        z = self.ln1(x)
        a, _ = self.attn(z, z, z, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class ClusterTransformer(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, n_classes, K, max_len,
                 use_cluster=False, variable_pos=False):
        super().__init__()
        self.use_cluster = use_cluster
        self.variable_pos = variable_pos
        self.K = K
        self.tok = nn.Embedding(vocab, d)
        self.seg = nn.Embedding(K, d)               # both models know sentences
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([Block(d, heads) for _ in range(n_layers)])
        if use_cluster:
            self.c2h = nn.ModuleList([nn.Linear(K, d) for _ in range(n_layers)])
            self.h2c = nn.ModuleList([nn.Linear(d, K) for _ in range(n_layers)])
        if variable_pos:
            self.p2h = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, toks, segs):
        B, T = toks.shape
        posids = torch.arange(T, device=toks.device).unsqueeze(0).expand(B, T)
        h = self.tok(toks) + self.seg(segs)
        p = self.pos(posids)
        if not self.variable_pos:
            h = h + p                                # fixed: added once
        c = F.one_hot(segs, self.K).float() if self.use_cluster else None   # init membership

        for l, layer in enumerate(self.layers):
            inp = h
            if self.variable_pos:
                inp = inp + p                        # position stream feeds each layer
            if self.use_cluster:
                inp = inp + self.c2h[l](torch.softmax(c, dim=-1))  # cluster feeds in
            h = layer(inp)
            if self.use_cluster:
                c = c + self.h2c[l](h)               # refine cluster from hidden (E/M-ish)
            if self.variable_pos:
                p = p + self.p2h[l](h)               # refine position from hidden
        return self.head(h.mean(dim=1))             # mean-pool -> classify


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
