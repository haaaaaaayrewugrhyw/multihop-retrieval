"""
model.py -- DeltaSystem: Generator G + Reconstructor D_recon
=============================================================

G(A, B) -> (delta [B, TB, d], delta_0 [B, d])
    SharedEncoder = frozen BERT-base (identical weights for A and B)
    Two-level multi-head cross-attention:
        Level 1:  O_self1 = MHA(H_B, H_B, H_B)
                  O_cross1 = MHA(H_B, H_A, H_A)
        Bridge:   Q_L2 = LN(W1*O_self1 + W2*O_cross1)
        Level 2:  O_self2 = MHA(Q_L2, H_B, H_B)
                  O_cross2 = MHA(Q_L2, H_A, H_A)
    delta = Tanh(W_out * (L2norm(O_self2) - L2norm(O_cross2)))
    delta_0 = Bottleneck(mean_pool(H_B))   [d -> D_SMALL -> d]

D_recon(A, delta, delta_0) -> B_hat  [causal seq2seq]
    Memory = [delta_0_token | H_A | enc(delta)]
    A dropout p=0.20 during training
    When ablate_delta=True: zeros ALL delta info (delta + delta_0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# Confirmed hyperparameters
D_MODEL    = 768
N_HEADS    = 8
N_LAYERS   = 2
D_SMALL    = round(0.30 * D_MODEL)   # 230 -- bottleneck for delta_0
VOCAB_SIZE = 30522                    # bert-base-uncased
MAX_SEQ    = 256
BOS_ID     = 101                      # [CLS] reused as decoder BOS
A_DROP_P   = 0.20                     # A dropout probability in D_recon


class DeltaSystem(nn.Module):
    def __init__(self, vib: bool = False, n_slots: int = 0):
        super().__init__()
        self.vib     = vib       # soft variational information bottleneck on delta
        self.last_kl = None      # per-element KL [b,T,D], set in generate_delta when vib
        self.n_slots = n_slots   # >0: HARD bottleneck — decoder sees delta only via K slots

        # ── Shared frozen BERT encoder ─────────────────────────────────────────
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad_(False)

        # ── Generator G: two-level cross-attention ─────────────────────────────
        # Level 1
        self.g_self1  = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True, dropout=0.1)
        self.g_cross1 = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True, dropout=0.1)
        # Bridge: learned weighted combination of O_self1 and O_cross1
        self.g_bw1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.g_bw2 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.g_bln = nn.LayerNorm(D_MODEL)
        # Level 2
        self.g_self2  = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True, dropout=0.1)
        self.g_cross2 = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True, dropout=0.1)
        # Output projection
        self.g_out = nn.Linear(D_MODEL, D_MODEL)
        # VIB heads: produce mu/logvar of the delta latent (soft info bottleneck)
        if vib:
            self.g_mu     = nn.Linear(D_MODEL, D_MODEL)
            self.g_logvar = nn.Linear(D_MODEL, D_MODEL)
        # delta_0: compress mean-pooled B through a bottleneck
        self.g_bottle = nn.Sequential(
            nn.Linear(D_MODEL, D_SMALL),
            nn.Tanh(),
            nn.Linear(D_SMALL, D_MODEL),
        )

        # ── D_recon: causal seq2seq ────────────────────────────────────────────
        # Separate trainable word embeddings (copied from BERT, not shared/frozen)
        self.dr_word_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        with torch.no_grad():
            self.dr_word_emb.weight.copy_(self.bert.embeddings.word_embeddings.weight)

        self.dr_pos = nn.Embedding(MAX_SEQ, D_MODEL)
        self.dr_d0  = nn.Linear(D_MODEL, D_MODEL)          # project delta_0 into memory

        # Causal encoder for delta (Encoder_2 in the design)
        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.dr_delta_enc = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        # HARD bottleneck: K learned slots compress enc(delta) so the decoder cannot use
        # delta as a full B-template (it must spend the slots on what A lacks = novelty).
        if n_slots > 0:
            self.dr_slots     = nn.Parameter(torch.randn(1, n_slots, D_MODEL) * 0.02)
            self.dr_slot_attn = nn.MultiheadAttention(D_MODEL, N_HEADS,
                                                      batch_first=True, dropout=0.1)

        # Causal decoder: cross-attends to [delta_0 | H_A | enc(delta) or K slots]
        dec_layer = nn.TransformerDecoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.dr_decoder = nn.TransformerDecoder(dec_layer, num_layers=N_LAYERS)

        # LM head tied to word embeddings
        self.dr_lm = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.dr_lm.weight = self.dr_word_emb.weight

    # ── Shared encoder (BERT, frozen) ─────────────────────────────────────────
    def _enc(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state

    # ── Generator ─────────────────────────────────────────────────────────────
    def generate_delta(self, H_A, A_mask, H_B, B_mask):
        A_pad = ~A_mask.bool()
        B_pad = ~B_mask.bool()

        # Level 1
        Os1, _ = self.g_self1(H_B, H_B, H_B, key_padding_mask=B_pad)
        Oc1, _ = self.g_cross1(H_B, H_A, H_A, key_padding_mask=A_pad)

        # Bridge
        Q2 = self.g_bln(self.g_bw1(Os1) + self.g_bw2(Oc1))

        # Level 2
        Os2, _ = self.g_self2(Q2, H_B, H_B, key_padding_mask=B_pad)
        Oc2, _ = self.g_cross2(Q2, H_A, H_A, key_padding_mask=A_pad)

        # Subtract L2-normalized outputs -> delta latent
        # Large where B is novel (Os2 ≠ Oc2), near-zero where B copies A (Os2 ≈ Oc2)
        z = F.normalize(Os2, dim=-1) - F.normalize(Oc2, dim=-1)
        if self.vib:
            # Variational bottleneck: delta ~ N(mu, sigma); KL(N(mu,sigma)||N(0,1))
            # penalized in the loss limits the BITS delta carries -> forces novelty-only.
            mu     = self.g_mu(z)
            logvar = self.g_logvar(z).clamp(-8.0, 8.0)
            if self.training:
                std       = torch.exp(0.5 * logvar)
                delta_pre = mu + std * torch.randn_like(std)
            else:
                delta_pre = mu                       # deterministic at eval
            self.last_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [b,T,D]
            delta = torch.tanh(delta_pre)
        else:
            self.last_kl = None
            delta = torch.tanh(self.g_out(z))

        # delta_0: compress mean-pooled H_B through bottleneck
        lens    = B_mask.float().sum(1, keepdim=True).clamp(min=1)
        B_mean  = (H_B * B_mask.unsqueeze(-1).float()).sum(1) / lens
        delta_0 = self.g_bottle(B_mean)

        # Return delta_norms as a proxy "alpha" for interface compatibility with eval/train
        alpha = delta.norm(dim=-1) / (D_MODEL ** 0.5)
        return delta, delta_0, alpha

    # ── Reconstructor ─────────────────────────────────────────────────────────
    def reconstruct(self, H_A, A_mask, delta, delta_0, B_ids, B_mask,
                    ablate_delta: bool = False,
                    drop_delta: bool = False, drop_d0: bool = False):
        # Ablation flags (for 3-way attribution in eval):
        #   ablate_delta : zero ALL delta info (token-delta + delta_0) -- ORIGINAL behavior, unchanged
        #   drop_delta   : zero ONLY the token-level delta (keep delta_0)
        #   drop_d0      : zero ONLY delta_0            (keep token-level delta)
        # The two fine-grained flags let us isolate whether delta or delta_0 causes
        # the Novel DELTA_PPL harm, without changing ablate_delta semantics.
        b, T = B_ids.shape
        dev  = B_ids.device

        # A dropout: randomly silence A for some batch items (training only)
        if self.training:
            keep   = (torch.rand(b, device=dev) > A_DROP_P).float()
            H_A    = H_A * keep.view(b, 1, 1)
            A_mask = (A_mask.float() * keep.view(b, 1)).long()

        d0_mask    = torch.ones(b, 1, dtype=torch.long, device=dev)
        delta_off  = ablate_delta or drop_delta          # token-delta path off
        d0_off     = ablate_delta or drop_d0             # delta_0 path off

        # delta_0 pathway
        d0_tok = torch.zeros(b, 1, D_MODEL, device=dev) if d0_off \
                 else self.dr_d0(delta_0).unsqueeze(1)

        # token-delta pathway -> either full T-length enc(delta) (baseline) or K slots (hard bottleneck)
        if self.n_slots > 0:
            if delta_off:
                delta_mem  = torch.zeros(b, self.n_slots, D_MODEL, device=dev)
                delta_mask = torch.zeros(b, self.n_slots, dtype=torch.long, device=dev)
            else:
                delta_enc  = self.dr_delta_enc(delta, src_key_padding_mask=~B_mask.bool())
                q          = self.dr_slots.expand(b, -1, -1)            # [b, K, D]
                delta_mem, _ = self.dr_slot_attn(q, delta_enc, delta_enc,
                                                 key_padding_mask=~B_mask.bool())
                delta_mask = torch.ones(b, self.n_slots, dtype=torch.long, device=dev)
        else:
            if delta_off:
                delta_mem  = torch.zeros(b, T, D_MODEL, device=dev)
                delta_mask = torch.zeros(b, T, dtype=torch.long, device=dev)
            else:
                delta_mem  = self.dr_delta_enc(delta, src_key_padding_mask=~B_mask.bool())
                delta_mask = B_mask

        # Build memory: [delta_0_token | H_A | delta_mem]
        memory  = torch.cat([d0_tok,  H_A,    delta_mem ], dim=1)
        mem_pad = ~torch.cat([d0_mask, A_mask, delta_mask], dim=1).bool()

        # Decoder: shifted B tokens
        shifted       = torch.full((b, T), BOS_ID, dtype=B_ids.dtype, device=dev)
        shifted[:, 1:] = B_ids[:, :-1]
        tgt = (self.dr_word_emb(shifted)
               + self.dr_pos(torch.arange(T, device=dev).unsqueeze(0)))

        # Use bool causal mask to match key_padding_mask type (avoids deprecation warning)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
        out = self.dr_decoder(
            tgt=tgt, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~B_mask.bool(),
            memory_key_padding_mask=mem_pad,
        )
        return self.dr_lm(out)

    # ── Joint forward ──────────────────────────────────────────────────────────
    def forward(self, A_ids, A_mask, B_ids, B_mask, ablate_delta: bool = False):
        H_A = self._enc(A_ids, A_mask)
        H_B = self._enc(B_ids, B_mask)
        delta, delta_0, alpha = self.generate_delta(H_A, A_mask, H_B, B_mask)
        logits = self.reconstruct(H_A, A_mask, delta, delta_0,
                                  B_ids, B_mask, ablate_delta)
        return logits, delta, delta_0, H_A, alpha
