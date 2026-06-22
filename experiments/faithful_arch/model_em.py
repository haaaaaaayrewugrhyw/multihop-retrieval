"""
EM-attention architecture -- the principled version.

Each layer is one EM iteration over K cluster centers mu (B,K,d):

  E-step  : tokens attend to the centers; the softmax-over-clusters scores ARE
            the new cluster probabilities P (responsibilities).
                P = softmax_K( W_q([content|pos|cluster]) . W_k(mu)^T )
  M-step  : each center = responsibility-weighted mean of token values,
            stabilized by a GRU.
                mu = GRU( (P^T V) / sum(P) , mu )
  read-back: content (and, in the variable-pos version, position) is updated by
            reading a P-weighted mix of the centers -- so tokens communicate ONLY
            through the K centers (a bottleneck), making the clusters load-bearing.
                h = h + W_o( P @ mu )

Centers init = per-sentence content means; P init = sharp one-hot(sentence).
Classification head reads the final centers + their soft counts and emits one
score per cluster (= per sentence), so the K-way label maps to clusters directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 11


class EMLayer(nn.Module):
    def __init__(self, d, K, variable_pos):
        super().__init__()
        self.d, self.K, self.variable_pos, self.scale = d, K, variable_pos, d ** -0.5
        d_x = d + d + K                                   # [content | position | cluster]
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d_x, d)
        self.kc = nn.Linear(d, d)
        self.v = nn.Linear(d_x, d)
        self.gru = nn.GRUCell(d, d)
        self.o = nn.Linear(d, d)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        if variable_pos:
            self.vp = nn.Linear(d_x, d)
            self.grup = nn.GRUCell(d, d)
            self.op = nn.Linear(d, d)

    def forward(self, h, p, P, mu, mu_p, pad_mask):
        B, T, _ = h.shape
        x = torch.cat([self.ln(h), p, P], dim=-1)         # token vector [content|pos|cluster]

        # E-step: token-to-center attention -> new cluster probabilities
        attn = (self.q(x) @ self.kc(mu).transpose(1, 2)) * self.scale     # (B,T,K)
        P_new = attn.softmax(dim=-1) * (~pad_mask).unsqueeze(-1).float()   # zero PAD tokens

        # M-step: centers = responsibility-weighted mean of token values
        counts = P_new.sum(1).unsqueeze(-1) + 1e-6                          # (B,K,1)
        mu_new = (P_new.transpose(1, 2) @ self.v(x)) / counts              # (B,K,d)
        mu = self.gru(mu_new.reshape(B * self.K, self.d),
                      mu.reshape(B * self.K, self.d)).reshape(B, self.K, self.d)

        # read-back: content updated by reading a P-weighted mix of centers
        h = h + self.o(P_new @ mu)
        h = h + self.ff(self.ln_ff(h))

        if self.variable_pos:
            mu_p_new = (P_new.transpose(1, 2) @ self.vp(x)) / counts
            mu_p = self.grup(mu_p_new.reshape(B * self.K, self.d),
                             mu_p.reshape(B * self.K, self.d)).reshape(B, self.K, self.d)
            p = p + self.op(P_new @ mu_p)

        return h, p, P_new, mu, mu_p


class EMArch(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, K, max_len,
                 variable_pos=False, pad_id=PAD):
        super().__init__()
        self.d, self.K, self.max_len, self.pad_id = d, K, max_len, pad_id
        self.variable_pos = variable_pos
        self.collect_P, self._last_P = False, None
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([EMLayer(d, K, variable_pos) for _ in range(n_layers)])
        self.head = nn.Linear(d + 1, 1)                  # per-cluster score from [center | soft-count]

    def forward(self, tokens, sentence_ids):
        B, T = tokens.shape
        pad_mask = (tokens == self.pad_id)
        not_pad = (~pad_mask).unsqueeze(-1).float()
        seg = sentence_ids.clamp(min=0)
        posids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)

        h = self.tok(tokens)
        p = self.pos(posids)
        P = F.one_hot(seg, self.K).float() * not_pad      # init: sharp sentence assignment
        counts = P.sum(1).unsqueeze(-1) + 1e-6
        mu = (P.transpose(1, 2) @ h) / counts             # centers init = per-sentence content mean
        mu_p = (P.transpose(1, 2) @ p) / counts if self.variable_pos else None

        for layer in self.layers:
            h, p, P, mu, mu_p = layer(h, p, P, mu, mu_p, pad_mask)

        if self.collect_P:
            self._last_P = P.detach()

        soft_counts = P.sum(1, keepdim=True).transpose(1, 2)            # (B,K,1) final mass per cluster
        score = self.head(torch.cat([mu, torch.log1p(soft_counts)], dim=-1)).squeeze(-1)  # (B,K)
        return score


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
