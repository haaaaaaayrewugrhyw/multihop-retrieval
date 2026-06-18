"""
Shared encoder + decoder, and the two bottleneck modules under test.

The ONLY difference between the two models is inside SlotBottleneck:
    competition=True  -> softmax over slots (real Slot Attention) + GRU
    competition=False -> softmax over inputs (plain attention, the MLP+attn idea)
Encoder, decoder, slot count, dim, and #iterations are identical, so the
experiment isolates the competition mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_grid(res):
    r = torch.linspace(0.0, 1.0, res)
    y, x = torch.meshgrid(r, r, indexing="ij")
    grid = torch.stack([x, y, 1 - x, 1 - y], dim=-1)      # res,res,4
    return grid.unsqueeze(0)                               # 1,res,res,4


class Encoder(nn.Module):
    """Conv stack (one stride-2 downsample) + positional embedding -> tokens."""

    def __init__(self, dim=64, enc_res=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim, 5, stride=2, padding=2), nn.ReLU(),     # 64 -> 32
            nn.Conv2d(dim, dim, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(dim, dim, 5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(dim, dim, 5, stride=1, padding=2), nn.ReLU(),
        )
        self.register_buffer("grid", build_grid(enc_res))
        self.pos = nn.Linear(4, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        f = self.conv(x)                                   # B,dim,32,32
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1)                          # B,H,W,dim
        f = f + self.pos(self.grid)
        f = f.reshape(B, H * W, C)
        return self.mlp(self.norm(f))                      # B, N, dim


class SlotBottleneck(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden=128, competition=True,
                 eps=1e-8):
        super().__init__()
        self.num_slots, self.iters, self.eps = num_slots, iters, eps
        self.competition = competition
        self.scale = dim ** -0.5
        self.mu = nn.Parameter(torch.randn(1, 1, dim))
        self.logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.logsigma)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))
        self.norm_in = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, inputs):                             # B, N, dim
        B, N, D = inputs.shape
        mu = self.mu.expand(B, self.num_slots, -1)
        sigma = self.logsigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        inputs = self.norm_in(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale   # B,slots,N
            if self.competition:
                attn = dots.softmax(dim=1) + self.eps                # over slots
                attn = attn / attn.sum(dim=2, keepdim=True)          # mean over N
            else:
                attn = dots.softmax(dim=2)                           # over inputs
            updates = torch.einsum("bjd,bij->bid", v, attn)         # B,slots,D
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D))
            slots = slots.reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_ff(slots))
        return slots


class Decoder(nn.Module):
    """Spatial broadcast decoder: each slot -> RGB + alpha; combine by alpha."""

    def __init__(self, dim=64, res=64, hidden=64):
        super().__init__()
        self.res = res
        self.register_buffer("grid", build_grid(res))
        self.pos = nn.Linear(4, dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(hidden, 4, 3, 1, 1),
        )

    def forward(self, slots):                              # B,K,dim
        B, K, D = slots.shape
        x = slots.reshape(B * K, D, 1, 1).expand(-1, -1, self.res, self.res)
        pos = self.pos(self.grid).permute(0, 3, 1, 2)      # 1,dim,res,res
        x = self.net(x + pos)                              # B*K,4,res,res
        x = x.reshape(B, K, 4, self.res, self.res)
        rgb = torch.sigmoid(x[:, :, :3])
        alpha = x[:, :, 3:4].softmax(dim=1)                # B,K,1,res,res
        recon = (rgb * alpha).sum(dim=1)                   # B,3,res,res
        return recon, alpha


class SlotAE(nn.Module):
    def __init__(self, num_slots=4, dim=64, iters=3, competition=True, res=64):
        super().__init__()
        self.encoder = Encoder(dim)
        self.bottleneck = SlotBottleneck(num_slots, dim, iters,
                                         competition=competition)
        self.decoder = Decoder(dim, res)

    def forward(self, x):
        slots = self.bottleneck(self.encoder(x))
        recon, alpha = self.decoder(slots)
        return recon, alpha


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
