"""
Cross-layer attention models for the "force the feature hierarchy" experiment.

All models share ONE backbone (a stack of Conv-BN-ReLU blocks with stride-2
downsampling). They differ ONLY in how block i is allowed to use the features
of earlier blocks:

    baseline : nothing. F_i = Block_i(F_{i-1}).            (hierarchy implicit)
    skip     : add a resized 1x1 projection of the previous block.
               NON-attention control -> isolates "cross-layer connection"
               from "attention selection".
    A        : cross-depth CHANNEL attention. Each channel of block i attends
               over the channels of block i-1 and gets a gate.
    B        : LAYER attention. Block i is summarized to one vector and attends
               over the set of summary vectors of ALL previous blocks, then
               recalibrates its own channels. ("which abstraction level?")
    C        : SPATIAL cross-layer attention. Each spatial position in block i
               attends to (pooled) spatial positions of block i-1.

Attention is single-head and the attention dim `d` is kept small on purpose so
the variants do not win merely by being much bigger. Param counts are reported
by count_params() so any size advantage is visible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def attend(q, k, v):
    """Single-head scaled dot-product attention.
    q: (B, Nq, d)  k,v: (B, Nk, d)  ->  (B, Nq, d)
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(1, 2)) / (d ** 0.5)
    w = scores.softmax(dim=-1)
    return w @ v


class ConvBlock(nn.Module):
    """Conv-BN-ReLU, stride-2 downsample by default."""

    def __init__(self, cin, cout, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# --------------------------------------------------------------------------- #
# Cross-layer mixers. Each takes the CURRENT block feature and previous one(s)
# and returns a modified current feature of the SAME shape.
# --------------------------------------------------------------------------- #
class SkipMixer(nn.Module):
    """Non-attention control: resize + 1x1-project previous block, add it."""

    def __init__(self, cur_ch, prev_ch):
        super().__init__()
        self.proj = nn.Conv2d(prev_ch, cur_ch, 1)

    def forward(self, cur, prev):
        p = F.adaptive_avg_pool2d(prev, cur.shape[-2:])
        return cur + self.proj(p)


class ChannelAttnMixer(nn.Module):
    """Variant A: tokens = channels. Current channels (queries) attend over
    previous block's channels (keys/values). Output is a per-channel gate."""

    def __init__(self, cur_ch, prev_ch, grid=4, d=64):
        super().__init__()
        self.grid = grid
        feat = grid * grid
        self.q = nn.Linear(feat, d)
        self.k = nn.Linear(feat, d)
        self.v = nn.Linear(feat, d)
        self.gate = nn.Linear(d, 1)

    def forward(self, cur, prev):
        cq = F.adaptive_avg_pool2d(cur, self.grid).flatten(2)    # B, Ccur, feat
        pk = F.adaptive_avg_pool2d(prev, self.grid).flatten(2)   # B, Cprev, feat
        ctx = attend(self.q(cq), self.k(pk), self.v(pk))         # B, Ccur, d
        g = torch.sigmoid(self.gate(ctx)).squeeze(-1)            # B, Ccur
        return cur * g.unsqueeze(-1).unsqueeze(-1)


class LayerAttnMixer(nn.Module):
    """Variant B: tokens = layers. Block i (one summary query) attends over the
    summary vectors of ALL previous blocks, then gates its own channels."""

    def __init__(self, cur_ch, prev_chs, d=64):
        super().__init__()
        self.q = nn.Linear(cur_ch, d)
        self.k = nn.ModuleList([nn.Linear(c, d) for c in prev_chs])
        self.v = nn.ModuleList([nn.Linear(c, d) for c in prev_chs])
        self.gate = nn.Linear(d, cur_ch)

    def forward(self, cur, prevs):
        qs = F.adaptive_avg_pool2d(cur, 1).flatten(1)            # B, Ccur
        q = self.q(qs).unsqueeze(1)                              # B, 1, d
        ks, vs = [], []
        for i, p in enumerate(prevs):
            ps = F.adaptive_avg_pool2d(p, 1).flatten(1)          # B, Cp
            ks.append(self.k[i](ps))
            vs.append(self.v[i](ps))
        k = torch.stack(ks, dim=1)                               # B, L, d
        v = torch.stack(vs, dim=1)                               # B, L, d
        ctx = attend(q, k, v).squeeze(1)                         # B, d
        g = torch.sigmoid(self.gate(ctx))                        # B, Ccur
        return cur * g.unsqueeze(-1).unsqueeze(-1)


class SpatialAttnMixer(nn.Module):
    """Variant C: tokens = spatial positions. Current positions (queries) attend
    to pooled positions of previous block. Residual injection."""

    def __init__(self, cur_ch, prev_ch, key_grid=7, d=64):
        super().__init__()
        self.key_grid = key_grid
        self.q = nn.Conv2d(cur_ch, d, 1)
        self.k = nn.Conv2d(prev_ch, d, 1)
        self.v = nn.Conv2d(prev_ch, d, 1)
        self.proj = nn.Conv2d(d, cur_ch, 1)

    def forward(self, cur, prev):
        B, _, H, W = cur.shape
        q = self.q(cur).flatten(2).transpose(1, 2)               # B, H*W, d
        pk = F.adaptive_avg_pool2d(prev, self.key_grid)
        k = self.k(pk).flatten(2).transpose(1, 2)                # B, g*g, d
        v = self.v(pk).flatten(2).transpose(1, 2)
        ctx = attend(q, k, v)                                    # B, H*W, d
        ctx = ctx.transpose(1, 2).reshape(B, -1, H, W)
        return cur + self.proj(ctx)


# --------------------------------------------------------------------------- #
# The shared network.
# --------------------------------------------------------------------------- #
VARIANTS = ("baseline", "skip", "A", "B", "C")


class HierNet(nn.Module):
    def __init__(self, in_ch, n_classes, channels=(32, 64, 128, 256),
                 variant="baseline", d=64):
        super().__init__()
        assert variant in VARIANTS, f"unknown variant {variant}"
        self.variant = variant
        self.blocks = nn.ModuleList()
        prev = in_ch
        for c in channels:
            self.blocks.append(ConvBlock(prev, c, downsample=True))
            prev = c

        self.mixers = nn.ModuleList()
        for i in range(1, len(channels)):
            if variant == "baseline":
                self.mixers.append(nn.Identity())
            elif variant == "skip":
                self.mixers.append(SkipMixer(channels[i], channels[i - 1]))
            elif variant == "A":
                self.mixers.append(ChannelAttnMixer(channels[i], channels[i - 1], d=d))
            elif variant == "B":
                self.mixers.append(LayerAttnMixer(channels[i], list(channels[:i]), d=d))
            elif variant == "C":
                self.mixers.append(SpatialAttnMixer(channels[i], channels[i - 1], d=d))

        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        feats = []
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i >= 1:
                mix = self.mixers[i - 1]
                if isinstance(mix, nn.Identity):
                    pass
                elif self.variant == "B":
                    h = mix(h, feats)            # all previous blocks
                else:
                    h = mix(h, feats[-1])        # immediately previous block
            feats.append(h)
        pooled = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        return self.head(pooled)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Plain MLP and MLP + cross-layer (depth-axis) attention.
# Used by arch_compare.py for the CNN vs ANN vs ANN+attention experiment.
# An "ANN" here = fully-connected net with NO spatial/convolutional prior, so
# it is the clean canvas for testing whether forcing the hierarchy helps.
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Plain fully-connected net. Flattens the image itself."""

    def __init__(self, in_dim, n_classes, width=256, depth=4):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(d, width), nn.LayerNorm(width), nn.ReLU()))
            d = width
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        h = x.flatten(1)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class MLPAttn(nn.Module):
    """The idea applied to an MLP: each hidden layer (l>=1) attends over the
    activation vectors of ALL previous hidden layers (tokens = layers), then
    gates itself. Direct analog of CNN variant B along the MLP depth axis."""

    def __init__(self, in_dim, n_classes, width=256, depth=4, d=32):
        super().__init__()
        self.layers = nn.ModuleList()
        dd = in_dim
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(dd, width), nn.LayerNorm(width), nn.ReLU()))
            dd = width
        # all hidden activations are width-dim, so q/k/v share one shape
        self.q = nn.ModuleList([nn.Linear(width, d) for _ in range(depth)])
        self.k = nn.ModuleList([nn.Linear(width, d) for _ in range(depth)])
        self.v = nn.ModuleList([nn.Linear(width, d) for _ in range(depth)])
        self.gate = nn.ModuleList([nn.Linear(d, width) for _ in range(depth)])
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        h = x.flatten(1)
        hs = []
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i >= 1:
                q = self.q[i](h).unsqueeze(1)                       # B, 1, d
                k = torch.stack([self.k[i](p) for p in hs], dim=1)  # B, L, d
                v = torch.stack([self.v[i](p) for p in hs], dim=1)  # B, L, d
                ctx = attend(q, k, v).squeeze(1)                    # B, d
                h = h * torch.sigmoid(self.gate[i](ctx))           # B, width
            hs.append(h)
        return self.head(h)
