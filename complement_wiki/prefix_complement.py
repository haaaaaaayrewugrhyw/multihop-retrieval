"""
prefix_complement.py -- leak-free complement model (the redesign)
=================================================================

Architecture (locked design):

  GENERATOR G(A, B) -> sequence edge E
    - Encode A bidirectionally (BERT)                    H_A [TA, 768]
    - Causal edge stack over B (TransformerDecoder):
        causal self-attn over B  +  cross-attn to H_A    G_B [TB, 768]
        => E[t] = f(full A, B[<=t])  (causal, leak-free)
    - Project to 128-d per token, L2-norm                E   [TB, 128]   (the edge)

  DISCRIMINATOR D(A, E) -> predict B  (leak-free, NO direct B channel)
    - pred_src[t] = edge that saw only B[<t]   (bos-shifted E)
    - up-proj pred_src -> cross-attn to FULL A -> FFN -> lm_head
    - logits[t] predicts b_t  from (A + edge-of-prefix) only

  Loss: causal cross-entropy over B  (leak-free next-token reconstruction).

Key property to verify (test_leak.py): logits at position t do NOT depend on
B tokens at positions >= t. If they do -> leak.

The edge E is multi-vector (sequence), 128-d/token, and doubles as the retrieval
representation (MaxSim against a query) at eval time.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v3"))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))

D_MODEL    = 768
N_HEADS    = 8
D_EDGE     = 128
VOCAB_SIZE = 30522
MAX_A_LEN  = 128
MAX_B_LEN  = 64


class PrefixComplementLM(nn.Module):
    def __init__(self, k_edge_layers: int = 2, j_dec_layers: int = 1):
        super().__init__()
        # A-encoder (bidirectional) + shared embeddings
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # B positional embeddings for the edge stack
        self.b_pos = nn.Embedding(MAX_B_LEN + 4, D_MODEL)

        # ---- Generator: causal edge stack over B (causal self-attn + cross-attn to A)
        edge_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.edge_stack = nn.TransformerDecoder(edge_layer, num_layers=k_edge_layers)
        self.edge_ln    = nn.LayerNorm(D_MODEL)
        self.edge_proj  = nn.Linear(D_MODEL, D_EDGE)          # -> 128-d edge per token
        self.edge_bos   = nn.Parameter(torch.zeros(1, 1, D_EDGE))  # bos edge (predict b_1)

        # ---- Discriminator: edge(prefix) + full A -> predict next B token
        self.d_up   = nn.Linear(D_EDGE, D_MODEL)             # lift edge back to 768 for D
        self.d_cross = nn.ModuleList([
            nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=0.1, batch_first=True)
            for _ in range(j_dec_layers)
        ])
        self.d_ln1 = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(j_dec_layers)])
        self.d_ln2 = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(j_dec_layers)])
        self.d_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(), nn.Linear(D_MODEL * 4, D_MODEL))
            for _ in range(j_dec_layers)
        ])

        # LM head tied to BERT word embeddings
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.bert.embeddings.word_embeddings.weight

    # ---- Generator ----------------------------------------------------------
    def generate_edge(self, A_ids, A_mask, B_ids, B_mask):
        """Returns E [B, TB, 128] (causal, leak-free)."""
        H_A = self.bert(input_ids=A_ids, attention_mask=A_mask).last_hidden_state  # [B,TA,768]
        A_pad = (A_mask == 0)                                                       # True=pad

        TB = B_ids.size(1)
        device = B_ids.device
        B_emb = self.bert.embeddings.word_embeddings(B_ids)
        B_emb = B_emb + self.b_pos(torch.arange(TB, device=device).unsqueeze(0))    # [B,TB,768]

        causal = nn.Transformer.generate_square_subsequent_mask(TB, device=device)  # no future B
        B_pad  = (B_mask == 0)

        G_B = self.edge_stack(
            tgt=B_emb, memory=H_A,
            tgt_mask=causal,
            tgt_key_padding_mask=B_pad,
            memory_key_padding_mask=A_pad,
        )                                                                          # [B,TB,768]
        E = F.normalize(self.edge_proj(self.edge_ln(G_B)), dim=-1)                  # [B,TB,128]
        return E, H_A, A_pad

    # ---- Discriminator ------------------------------------------------------
    def forward(self, A_ids, A_mask, B_ids, B_mask, ablate_edge: bool = False):
        """Returns (logits [B, TB, V], E [B, TB, 128]). Predicts each b_t leak-free.
        ablate_edge=True zeros the edge so D must reconstruct B from A ALONE
        (used to measure recon-gain = how much the edge actually helps)."""
        E, H_A, A_pad = self.generate_edge(A_ids, A_mask, B_ids, B_mask)

        # pred_src[t] = edge that saw only B[<t]  (shift E right, prepend bos edge)
        bos = self.edge_bos.expand(E.size(0), 1, D_EDGE)
        pred_src = torch.cat([bos, E[:, :-1, :]], dim=1)        # [B,TB,128]
        if ablate_edge:
            pred_src = torch.zeros_like(pred_src)               # D sees only A

        x = self.d_up(pred_src)                                  # [B,TB,768]
        for cross, ln1, ln2, ffn in zip(self.d_cross, self.d_ln1, self.d_ln2, self.d_ffn):
            ctx, _ = cross(x, H_A, H_A, key_padding_mask=A_pad, need_weights=False)
            x = ln1(x + ctx)
            x = ln2(x + ffn(x))
        logits = self.lm_head(x)                                # [B,TB,V]
        return logits, E

    @torch.no_grad()
    def extract_edge(self, A_ids, A_mask, B_ids, B_mask=None):
        """Eval: the sequence edge E [B, TB, 128] for retrieval (MaxSim)."""
        if B_mask is None:
            B_mask = (B_ids != 0).long()
        E, _, _ = self.generate_edge(A_ids, A_mask, B_ids, B_mask)
        return E

    @torch.no_grad()
    def encode_text(self, ids, mask):
        """Encode any text to token vectors in the 128-d edge space (for queries)."""
        H = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        return F.normalize(self.edge_proj(self.edge_ln(H)), dim=-1)   # [B, T, 128]
