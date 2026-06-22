"""
Faithful three-stream cluster-transformer -- from scratch, exactly to spec.

Each token has three streams:
    content  h (B,T,d)
    cluster  c (B,T,K)   logits over K = max sentences; P = softmax(c) = membership
    position p (B,T,d)

The Q/K/V input each layer is the CONCATENATION [content | cluster | position];
W_Q/W_K/W_V handle everything -- there is NO hand-designed affinity bias. The
content stream h is the residual (dim d): attention projects the wide concat
input back down to d, so h = h + attn(...) stays dimensionally clean. The
cluster stream is a separate (B,T,K) tensor, initialized from sentence
membership, refined each layer (h2c) and re-concatenated next layer. Position
is fixed (constant) or variable (refined each layer).

Baseline (cluster_mode='none', use_segment=True) = a standard transformer that
knows the sentences via a segment embedding; concat input is [content | pos].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """Q/K/V from a (wider) input vector of dim d_in; output projected back to d."""

    def __init__(self, d_in, d, heads):
        super().__init__()
        assert d % heads == 0
        self.heads, self.dh, self.scale = heads, d // heads, (d // heads) ** -0.5
        self.qkv = nn.Linear(d_in, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad_mask):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-2, -1)) * self.scale
        s = s.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        a = s.softmax(dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, T, self.heads * self.dh)
        return self.out(o)


class Block(nn.Module):
    """Pre-LN block. QKV input = concat(LN(content), side); output residual on content."""

    def __init__(self, d, side_dim, heads):
        super().__init__()
        self.ln_in = nn.LayerNorm(d)
        self.attn = MHA(d + side_dim, d, heads)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, side, pad_mask):
        qkv_in = torch.cat([self.ln_in(h), side], dim=-1)   # [content | cluster | position]
        h = h + self.attn(qkv_in, pad_mask)
        h = h + self.ff(self.ln_ff(h))
        return h


class FaithfulArch(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, K, max_len,
                 cluster_mode="full", variable_pos=False, use_segment=False, pad_id=11):
        super().__init__()
        assert cluster_mode in ("none", "full")
        self.cluster_mode = cluster_mode
        self.variable_pos = variable_pos
        self.use_segment = use_segment
        self.K, self.max_len, self.pad_id = K, max_len, pad_id
        self.collect_P, self._last_P = False, None

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        if use_segment:
            self.seg = nn.Embedding(K, d)               # sentence info for the baseline

        side_dim = d + (K if cluster_mode == "full" else 0)   # position(d) [+ cluster(K)]
        self.blocks = nn.ModuleList([Block(d, side_dim, heads) for _ in range(n_layers)])
        if cluster_mode == "full":
            self.h2c = nn.ModuleList([nn.Linear(d, K) for _ in range(n_layers)])
        if variable_pos:
            self.p2h = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.head = nn.Linear(d, K)

    def forward(self, tokens, sentence_ids):
        B, T = tokens.shape
        pad_mask = (tokens == self.pad_id)
        posids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        seg_ids = sentence_ids.clamp(min=0)             # map PAD's -1 -> 0 (masked anyway)
        not_pad = (sentence_ids >= 0).unsqueeze(-1).float()

        h = self.tok(tokens)
        if self.use_segment:
            h = h + self.seg(seg_ids)
        p = self.pos(posids)
        if self.cluster_mode == "full":
            c = F.one_hot(seg_ids, self.K).float() * not_pad   # init from sentence; PAD -> zero

        for l, block in enumerate(self.blocks):
            if self.cluster_mode == "full":
                side = torch.cat([torch.softmax(c, dim=-1), p], dim=-1)
            else:
                side = p
            h = block(h, side, pad_mask)
            if self.cluster_mode == "full":
                c = c + self.h2c[l](h)                  # refine cluster from content
            if self.variable_pos:
                p = p + self.p2h[l](h)                  # refine position from content

        if self.cluster_mode == "full" and self.collect_P:
            self._last_P = torch.softmax(c, dim=-1).detach()

        m = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)                        # (B, K) logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
