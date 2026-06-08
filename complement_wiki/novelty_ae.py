"""
novelty_ae.py -- "edge one step ahead" novelty autoencoder (the user's design)
==============================================================================

Resolves the impossibility of the leak-free prefix model:
  - Leak-free edge (sees only b_<t) carries NO info a backbone (A + b_<t) lacks
    -> redundant -> can't supervise novelty.  (proven, see prefix model result)
  - So here the edge IS allowed to see b_t (ONE STEP AHEAD). It therefore CAN carry
    the novel content. To stop it copying ALL of B (the v1/v2 collapse), we add the
    two guardrails: A is FREE to the decoder, and the edge is SQUEEZED by a sparsity
    gate. Copied tokens are cheap from (A + b_<t) -> gate closes; novel tokens need
    the edge -> gate opens. The open gates localize "what B adds", with NO labels.

Generator G(A, B):
  H_A = BERT(A)                                    [TA,768]  full A (free side info)
  H_B = BERT(B)                                    [TB,768]  each token has SEEN b_t
  E[t]   = L2norm(edge_proj(LN(H_B[t])))           [TB,128]  the edge (one-ahead)
  alpha[t] = sigmoid(gate(H_B[t]))                 [TB]      novelty gate (sparsified)

Discriminator D(A, b_<t, alpha*E) -> predict b_t:
  dec_in[t] = emb(b_{t-1}) + pos[t] + alpha[t] * up(E[t])   (edge injected ONE AHEAD)
  causal self-attn over dec positions + cross-attn to H_A (A FREE) -> lm_head
  Loss = CE(B) + beta * mean(alpha)                          (the squeeze)

Eval-time:
  extract_edge -> E [TB,128]; gate -> alpha [TB].
  encode_text shares edge_proj+LN, SAME input distribution (raw BERT) as the edge,
  so MaxSim(edge, encode_text(phrase)) is a FAIR same-space comparison (unlike the
  prefix model, where edge lived in decoder-stack space).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))

D_MODEL    = 768
N_HEADS    = 8
D_EDGE     = 128
VOCAB_SIZE = 30522
MAX_A_LEN  = 128
MAX_B_LEN  = 64
BOS_ID     = 101   # [CLS] reused as decoder BOS


class NoveltyAutoencoder(nn.Module):
    def __init__(self, j_dec_layers: int = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # edge (shared with encode_text so spaces match)
        self.edge_ln   = nn.LayerNorm(D_MODEL)
        self.edge_proj = nn.Linear(D_MODEL, D_EDGE)

        # novelty gate (the squeeze)
        self.gate = nn.Sequential(nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(),
                                  nn.Linear(D_MODEL // 2, 1))

        # decoder
        self.d_up  = nn.Linear(D_EDGE, D_MODEL)            # lift edge back to 768
        self.d_pos = nn.Embedding(MAX_B_LEN + 4, D_MODEL)
        dec_layer  = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.d_stack = nn.TransformerDecoder(dec_layer, num_layers=j_dec_layers)

        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.bert.embeddings.word_embeddings.weight

    # ---- generator ---------------------------------------------------------
    def generate_edge(self, A_ids, A_mask, B_ids, B_mask):
        H_A = self.bert(input_ids=A_ids, attention_mask=A_mask).last_hidden_state
        H_B = self.bert(input_ids=B_ids, attention_mask=B_mask).last_hidden_state
        E     = F.normalize(self.edge_proj(self.edge_ln(H_B)), dim=-1)   # [b,TB,128]
        alpha = torch.sigmoid(self.gate(H_B)).squeeze(-1)               # [b,TB]
        return E, alpha, H_A, (A_mask == 0)

    # ---- discriminator -----------------------------------------------------
    def forward(self, A_ids, A_mask, B_ids, B_mask, ablate_edge: bool = False,
                edge_dropout: float = 0.0, alpha_override: float = None):
        """Returns (logits [b,TB,V], E [b,TB,128], alpha [b,TB]). Predicts each b_t
        from b_<t (teacher) + A (free) + alpha*edge(one-ahead).

        edge_dropout (train only): with this prob, hide the edge for an example so the
        decoder is FORCED to learn the A + b_<t backbone -> the edge then only matters
        where the backbone fails (novel tokens), which is what lets the gate localize.
        Without it the edge becomes a universal cheat sheet and the gate never closes."""
        E, alpha, H_A, A_pad = self.generate_edge(A_ids, A_mask, B_ids, B_mask)
        b, TB = B_ids.shape
        device = B_ids.device

        # teacher-forced shift: dec_tok[t] = b_{t-1}, predicts b_t
        shifted = torch.full((b, TB), BOS_ID, dtype=B_ids.dtype, device=device)
        shifted[:, 1:] = B_ids[:, :-1]
        tok_emb = self.bert.embeddings.word_embeddings(shifted)
        pos_emb = self.d_pos(torch.arange(TB, device=device).unsqueeze(0))

        if ablate_edge:
            edge_inj = torch.zeros_like(tok_emb)
        else:
            # alpha_override (diagnostic): force a constant gate (e.g. 1.0) to measure
            # the edge's POTENTIAL value per token, independent of the learned gate.
            use_alpha = alpha if alpha_override is None else torch.full_like(alpha, alpha_override)
            edge_inj = use_alpha.unsqueeze(-1) * self.d_up(E)   # one-ahead: E[t] saw b_t
            if edge_dropout > 0.0 and self.training:
                keep = (torch.rand(b, 1, 1, device=device) > edge_dropout).float()
                edge_inj = edge_inj * keep                 # hide edge for some examples
        dec_in = tok_emb + pos_emb + edge_inj

        causal = nn.Transformer.generate_square_subsequent_mask(TB, device=device)
        D = self.d_stack(
            tgt=dec_in, memory=H_A,
            tgt_mask=causal,
            tgt_key_padding_mask=(B_mask == 0),
            memory_key_padding_mask=A_pad,
        )
        logits = self.lm_head(D)
        return logits, E, alpha

    @torch.no_grad()
    def extract_edge(self, A_ids, A_mask, B_ids, B_mask=None):
        if B_mask is None:
            B_mask = (B_ids != 0).long()
        E, _, _, _ = self.generate_edge(A_ids, A_mask, B_ids, B_mask)
        return E

    @torch.no_grad()
    def gate_only(self, A_ids, A_mask, B_ids, B_mask=None):
        if B_mask is None:
            B_mask = (B_ids != 0).long()
        _, alpha, _, _ = self.generate_edge(A_ids, A_mask, B_ids, B_mask)
        return alpha

    @torch.no_grad()
    def encode_text(self, ids, mask):
        """Same edge_proj+LN on raw BERT -> SAME space as the edge (for fair MaxSim)."""
        H = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        return F.normalize(self.edge_proj(self.edge_ln(H)), dim=-1)
