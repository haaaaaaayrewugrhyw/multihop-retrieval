"""
FakeEncoder Training — retrieval_v2
====================================

Architecture
------------
  Encoder1 (BERT-base, 12-layer):
      A_text  →  hidden_states [n_A × 768]  =  K1, V1

  FakeEncoder (3-layer Transformer):
      F = zeros [n_slots × 768] per example   (NOT a model parameter — per-example state)
      F  →  FakeEncoder  →  K2, V2  [n_slots × 768]

  Decoder (auto-regressive, teacher-forced):
      for each position t in B:
          h_t     = MaskedSelfAttn(B[0..t-1])            [decoder hidden state]
          r1_t    = CrossAttn(h_t  →  K1, V1)            [from Encoder1]
          r2_t    = CrossAttn(h_t  →  K2, V2)            [from FakeEncoder, TRACK attn_weights]
          meta_w  = softmax(linear([r1_t, r2_t]))         [2 scalars]
          merged  = meta_w[0]·r1_t + meta_w[1]·r2_t      [blended context]
          → predict B[t]

  Loss: CrossEntropy(predictions, B_tokens)   — PAD tokens ignored

  F update (reverse cross-attention + GRU):
      feedback[i] = LayerNorm( Σ_t attn2[t, i] · h_t )  [what decoder wanted from slot i]
      F[i]        = GRUCell( feedback[i], F[i] )          [gated state update]

  Complement (test-time):
      iterate F for n_iter_test=5 passes with teacher-forced B,
      then:  complement = L2_norm( proj( mean_pool( FakeEncoder(final_F) ) ) )  [128-dim]

Usage
-----
  python fakencoder_train.py                           # full 3-epoch training on T4
  python fakencoder_train.py --smoke                   # quick CPU smoke test (50 ex)
  python fakencoder_train.py --max_examples 1000       # partial run
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import build_chain_quadruples, load_musique

# ── Constants ─────────────────────────────────────────────────────────────────
D_MODEL       = 768
N_HEADS       = 8
N_SLOTS       = 8        # size of F (number of FakeEncoder tokens per example)
N_FAKE_LAYERS = 3        # FakeEncoder transformer depth
N_DEC_LAYERS  = 2        # Decoder transformer depth
N_ITER_TRAIN  = 3        # F refinement iterations during training
N_ITER_TEST   = 5        # F refinement iterations at eval/inference
D_PROJ        = 128      # complement vector dimension
VOCAB_SIZE    = 30522    # bert-base-uncased vocab size

MAX_A_LEN     = 128      # max tokens for A  (Encoder1 input)
MAX_B_LEN     = 64       # max tokens for B  (Decoder target)

BATCH_SIZE    = 16
LR            = 2e-5
N_EPOCHS      = 3
WARMUP_FRAC   = 0.06     # 6% of total steps used for warmup
MAX_GRAD_NORM = 1.0

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")
MODEL_DIR   = Path(__file__).parent / "models"
CACHE_DIR = Path(__file__).parent / "cache"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Single decoder layer:
      1. Causal masked self-attention on B prefix
      2. Cross-attention to Encoder1 output  (K1/V1)
      3. Cross-attention to FakeEncoder output (K2/V2)  ← attention weights tracked
      4. Meta-attention merge of r1 and r2
      5. FFN
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # meta: concat(r1, r2) → 2 scalars → softmax → weights for merge
        self.meta = nn.Linear(d_model * 2, 2, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x:            torch.Tensor,            # [B, seq_B, D]
        K1:           torch.Tensor,            # [B, n_A,    D]
        V1:           torch.Tensor,            # [B, n_A,    D]
        K2:           torch.Tensor,            # [B, n_slots, D]
        V2:           torch.Tensor,            # [B, n_slots, D]
        causal_mask:  torch.Tensor,            # [seq_B, seq_B] additive float mask
        key_pad_a:    Optional[torch.Tensor],  # [B, n_A] bool — True = PAD in A
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          x        [B, seq_B, D]     — output hidden states
          h_query  [B, seq_B, D]     — decoder state before cross-attn (query for F update)
          attn2    [B, seq_B, n_slots] — decoder's attention weights over FakeEncoder slots
        """
        # 1. Causal masked self-attention
        h_self, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln1(x + h_self)
        h_query = x   # decoder hidden state used as cross-attention query

        # 2. Cross-attention to Encoder1
        r1, _ = self.cross_attn1(x, K1, V1, key_padding_mask=key_pad_a, need_weights=False)

        # 3. Cross-attention to FakeEncoder (need weights for F update)
        r2, attn2 = self.cross_attn2(x, K2, V2, need_weights=True)
        # attn2: [B, seq_B, n_slots]  (averaged over heads by default)

        # 4. Meta-attention merge
        meta_w = torch.softmax(self.meta(torch.cat([r1, r2], dim=-1)), dim=-1)  # [B, seq_B, 2]
        merged = meta_w[..., :1] * r1 + meta_w[..., 1:] * r2                   # [B, seq_B, D]
        x = self.ln2(x + merged)

        # 5. FFN
        x = self.ln3(x + self.ffn(x))

        return x, h_query, attn2


# ── FakeEncoder Model ─────────────────────────────────────────────────────────

class FakeEncoderModel(nn.Module):

    def __init__(
        self,
        n_slots:       int = N_SLOTS,
        n_fake_layers: int = N_FAKE_LAYERS,
        n_dec_layers:  int = N_DEC_LAYERS,
        n_iter_train:  int = N_ITER_TRAIN,
        n_iter_test:   int = N_ITER_TEST,
        d_proj:        int = D_PROJ,
    ):
        super().__init__()
        self.n_slots      = n_slots
        self.n_iter_train = n_iter_train
        self.n_iter_test  = n_iter_test

        # ── Encoder1: BERT-base processes A ──────────────────────────────────
        self.encoder1 = BertModel.from_pretrained("bert-base-uncased")
        self.encoder1.gradient_checkpointing_enable()

        # ── FakeEncoder: 3-layer Transformer, self-attention over F slots ────
        fe_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.fake_encoder = nn.TransformerEncoder(fe_layer, num_layers=n_fake_layers)

        # ── Decoder: multi-source cross-attention ────────────────────────────
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(n_dec_layers)])

        # Token embeddings for decoder (shared with BERT's vocab embeddings)
        # Positional embeddings are separate (BERT's are segment-based)
        self.pos_emb   = nn.Embedding(MAX_B_LEN + 4, D_MODEL)
        self.ln_dec_in = nn.LayerNorm(D_MODEL)

        # ── F update: reverse cross-attention + GRU ──────────────────────────
        self.gru_cell    = nn.GRUCell(D_MODEL, D_MODEL)
        self.ln_feedback = nn.LayerNorm(D_MODEL)
        self.ln_F        = nn.LayerNorm(D_MODEL)

        # ── Output heads ─────────────────────────────────────────────────────
        # LM head for B generation (weight-tied to BERT's word embeddings)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.encoder1.embeddings.word_embeddings.weight

        # Complement projection: FakeEncoder output → 128-dim
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, d_proj),
        )

        # Query/passage encoder: BERT CLS → 128-dim  (same space as complement)
        self.query_proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, d_proj),
        )

    # ── Encoder1 ─────────────────────────────────────────────────────────────

    def _encode_a(
        self,
        input_ids_a: torch.Tensor,  # [B, n_A]
        attn_mask_a: torch.Tensor,  # [B, n_A]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden_states [B, n_A, D], key_pad_mask [B, n_A] bool)."""
        out    = self.encoder1(input_ids=input_ids_a, attention_mask=attn_mask_a)
        hidden = out.last_hidden_state                # [B, n_A, D]
        key_pad = (attn_mask_a == 0)                  # True = PAD → ignore in cross-attn
        return hidden, key_pad

    # ── FakeEncoder ──────────────────────────────────────────────────────────

    def _fake_encode(self, F_slots: torch.Tensor) -> torch.Tensor:
        """F_slots [B, n_slots, D] → refined [B, n_slots, D]."""
        return self.fake_encoder(F_slots)

    # ── Decoder ──────────────────────────────────────────────────────────────

    def _decode(
        self,
        dec_input_ids: torch.Tensor,            # [B, seq_B]
        K1:            torch.Tensor,            # [B, n_A,    D]
        V1:            torch.Tensor,            # [B, n_A,    D]
        K2:            torch.Tensor,            # [B, n_slots, D]
        V2:            torch.Tensor,            # [B, n_slots, D]
        key_pad_a:     Optional[torch.Tensor],  # [B, n_A] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forced decoder pass. All positions processed in parallel.
        Returns:
          logits   [B, seq_B, VOCAB_SIZE]
          h_query  [B, seq_B, D]           — decoder states (for F feedback, last layer)
          attn2    [B, seq_B, n_slots]     — FakeEncoder attention weights  (last layer)
        """
        seq_B  = dec_input_ids.size(1)
        device = dec_input_ids.device

        # Token embeddings (from BERT, shared with lm_head) + positional
        tok_emb = self.encoder1.embeddings.word_embeddings(dec_input_ids)  # [B, seq_B, D]
        pos_ids = torch.arange(seq_B, device=device).unsqueeze(0)          # [1, seq_B]
        pos_emb = self.pos_emb(pos_ids)                                     # [1, seq_B, D]
        x = self.ln_dec_in(tok_emb + pos_emb)                              # [B, seq_B, D]

        # Causal additive mask: -inf above diagonal
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_B, device=device)

        h_query_out, attn2_out = x, None
        for layer in self.decoder_layers:
            x, h_query_out, attn2_out = layer(x, K1, V1, K2, V2, causal_mask, key_pad_a)

        logits = self.lm_head(x)  # [B, seq_B, VOCAB_SIZE]
        return logits, h_query_out, attn2_out

    # ── F update ─────────────────────────────────────────────────────────────

    def _update_F(
        self,
        F_slots: torch.Tensor,  # [B, n_slots, D]
        h_query: torch.Tensor,  # [B, seq_B,   D]
        attn2:   torch.Tensor,  # [B, seq_B, n_slots]
    ) -> torch.Tensor:
        """
        Reverse cross-attention: aggregate decoder states weighted by how much
        each decoder position attended to each FakeEncoder slot.
        Then gate-update F with GRU for stability.

        feedback[i] = LayerNorm( Σ_t  attn2[t, i] · h_query[t] )
        F[i]        = GRUCell( feedback[i], F[i] )
        """
        # feedback: what each slot should absorb from the decoder
        feedback = torch.einsum("bts,btd->bsd", attn2, h_query)  # [B, n_slots, D]
        feedback = self.ln_feedback(feedback)

        B, n_slots, D = F_slots.shape
        F_flat  = F_slots.reshape(B * n_slots, D)
        fb_flat = feedback.reshape(B * n_slots, D)

        # GRU: input=feedback (new info), hidden=F (current state) → new state
        new_F_flat = self.gru_cell(fb_flat, F_flat)          # [B*n_slots, D]
        new_F      = self.ln_F(new_F_flat.reshape(B, n_slots, D))
        return new_F

    # ── Full forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids_a:   torch.Tensor,  # [B, n_A]
        attn_mask_a:   torch.Tensor,  # [B, n_A]
        dec_input_ids: torch.Tensor,  # [B, seq_B]  — B shifted right (BOS + B[:-1])
        training:      bool = True,
    ) -> torch.Tensor:
        """Returns logits [B, seq_B, VOCAB_SIZE]."""
        n_iter = self.n_iter_train if training else self.n_iter_test

        # Encode A once (frozen across F refinement iterations)
        K1, key_pad_a = self._encode_a(input_ids_a, attn_mask_a)
        V1 = K1

        # Initialize F to zeros per example (NOT a model parameter)
        B = input_ids_a.size(0)
        F_slots = torch.zeros(B, self.n_slots, D_MODEL, device=input_ids_a.device)

        logits = None
        for _ in range(n_iter):
            K2 = V2 = self._fake_encode(F_slots)
            logits, h_query, attn2 = self._decode(dec_input_ids, K1, V1, K2, V2, key_pad_a)
            F_slots = self._update_F(F_slots, h_query, attn2)

        return logits  # [B, seq_B, VOCAB_SIZE]

    # ── Complement extraction (test-time) ────────────────────────────────────

    @torch.no_grad()
    def extract_complement(
        self,
        input_ids_a: torch.Tensor,  # [B, n_A]
        attn_mask_a: torch.Tensor,  # [B, n_A]
        input_ids_b: torch.Tensor,  # [B, seq_B]  full B (CLS + tokens + SEP + PAD)
    ) -> torch.Tensor:
        """
        Refine F over n_iter_test passes (teacher-forced with full B),
        then extract: complement = L2_norm( proj( mean_pool( FakeEncoder(final_F) ) ) )
        Returns [B, D_PROJ] L2-normalized complement vectors.
        """
        # Decoder input: drop last token (CLS, b1, ..., b_{n-1})
        dec_input = input_ids_b[:, :-1]

        K1, key_pad_a = self._encode_a(input_ids_a, attn_mask_a)
        V1 = K1

        B = input_ids_a.size(0)
        F_slots = torch.zeros(B, self.n_slots, D_MODEL, device=input_ids_a.device)

        for _ in range(self.n_iter_test):
            K2 = V2 = self._fake_encode(F_slots)
            _, h_query, attn2 = self._decode(dec_input, K1, V1, K2, V2, key_pad_a)
            F_slots = self._update_F(F_slots, h_query, attn2)

        # Extract complement from converged FakeEncoder
        F_final = self._fake_encode(F_slots)       # [B, n_slots, D]
        comp    = F_final.mean(dim=1)              # [B, D]
        comp    = self.proj(comp)                  # [B, D_PROJ]
        comp    = F.normalize(comp, dim=-1)        # L2-normalize
        return comp

    @torch.no_grad()
    def encode_query(
        self,
        input_ids:  torch.Tensor,  # [B, seq]
        attn_mask:  torch.Tensor,  # [B, seq]
    ) -> torch.Tensor:
        """Encode any single text (query or passage) → [B, D_PROJ] L2-normalized."""
        out   = self.encoder1(input_ids=input_ids, attention_mask=attn_mask)
        cls   = out.last_hidden_state[:, 0, :]  # [CLS] token [B, D]
        q_vec = self.query_proj(cls)            # [B, D_PROJ]
        return F.normalize(q_vec, dim=-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HopPairDataset(Dataset):
    """Consecutive-hop (A_text, B_text) pairs from chain quadruples."""

    def __init__(self, quadruples: List[Dict], id_to_text: Dict[str, str]):
        self.pairs: List[Tuple[str, str]] = []
        for q in quadruples:
            a = id_to_text.get(q["chunk_a_id"], "")
            b = id_to_text.get(q["chunk_b_pos_id"], "")
            if a and b:
                self.pairs.append((a, b))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


def make_collator(tokenizer: BertTokenizerFast):
    def collate(batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        a_texts = [item[0] for item in batch]
        b_texts = [item[1] for item in batch]

        enc_a = tokenizer(
            a_texts, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_b = tokenizer(
            b_texts, max_length=MAX_B_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        # Teacher forcing: dec_input = B[:-1], dec_target = B[1:]
        # enc_b.input_ids: [CLS, b1, ..., bn, SEP, PAD, ...]
        return {
            "input_ids_a": enc_a.input_ids,
            "attn_mask_a": enc_a.attention_mask,
            "dec_input":   enc_b.input_ids[:, :-1],   # [B, MAX_B_LEN-1]
            "dec_target":  enc_b.input_ids[:, 1:],    # [B, MAX_B_LEN-1]
        }
    return collate


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model:     FakeEncoderModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    log_every: int = 200,
) -> float:
    model.train()
    total_loss, n = 0.0, 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        ia = batch["input_ids_a"].to(DEVICE)
        ma = batch["attn_mask_a"].to(DEVICE)
        di = batch["dec_input"].to(DEVICE)
        dt = batch["dec_target"].to(DEVICE)

        with autocast("cuda", enabled=AMP_ENABLED):
            logits = model(ia, ma, di, training=True)   # [B, T, V]
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                dt.reshape(B * T),
                ignore_index=0,   # PAD token id in bert-base-uncased
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()
        n += 1

        if step > 0 and step % log_every == 0:
            tqdm.write(f"  step {step} | loss {total_loss / n:.4f}")

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model: FakeEncoderModel, loader: DataLoader) -> float:
    model.eval()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc="Val", leave=False):
        ia = batch["input_ids_a"].to(DEVICE)
        ma = batch["attn_mask_a"].to(DEVICE)
        di = batch["dec_input"].to(DEVICE)
        dt = batch["dec_target"].to(DEVICE)

        with autocast("cuda", enabled=AMP_ENABLED):
            logits = model(ia, ma, di, training=False)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                dt.reshape(B * T),
                ignore_index=0,
            )

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",        action="store_true",
                        help="Quick smoke test (50 examples, 1 epoch, CPU-safe)")
    parser.add_argument("--max_examples", type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",       type=int,   default=N_EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    args = parser.parse_args()

    if args.smoke:
        args.max_examples = 50
        args.batch_size   = 2
        args.epochs       = 1
        print("[smoke] Running quick smoke test (50 examples, 1 epoch)")

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading MuSiQue ...")
    train_corpus, train_queries = load_musique(
        split="train", max_examples=args.max_examples, cache=not args.smoke,
    )
    val_max = 10 if args.smoke else 300
    val_corpus, val_queries = load_musique(
        split="validation", max_examples=val_max, cache=not args.smoke,
    )

    train_quads = build_chain_quadruples(train_corpus, train_queries)
    val_quads   = build_chain_quadruples(val_corpus,   val_queries)

    id_to_text = {c["chunk_id"]: c["text"] for c in train_corpus + val_corpus}
    train_ds   = HopPairDataset(train_quads, id_to_text)
    val_ds     = HopPairDataset(val_quads,   id_to_text)
    print(f"   Train pairs : {len(train_ds):,}  |  Val pairs : {len(val_ds):,}")

    tokenizer  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collator   = make_collator(tokenizer)
    nw         = 0 if args.smoke else 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[2] Building FakeEncoderModel ...")
    model  = FakeEncoderModel().to(DEVICE)
    scaler = GradScaler("cuda", enabled=AMP_ENABLED)

    # Epoch 1: freeze Encoder1 (BERT) — let FakeEncoder + Decoder warm up first
    for p in model.encoder1.parameters():
        p.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params     : {total_p / 1e6:.1f} M")
    print(f"   Trainable (ep 1) : {trainable_p / 1e6:.1f} M  (BERT frozen)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[3] Training {args.epochs} epoch(s) ...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        # Epoch 2+: unfreeze Encoder1 at 10× lower LR
        if epoch == 2:
            print("   Unfreezing Encoder1 (BERT) with lr * 0.1 ...")
            for p in model.encoder1.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.encoder1.parameters(),                         "lr": args.lr * 0.1},
                {"params": [p for n, p in model.named_parameters()
                            if "encoder1" not in n],                            "lr": args.lr},
            ])
            remaining = (args.epochs - 1) * len(train_loader)
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, remaining)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss   = validate(model, val_loader)

        print(
            f"   Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ppl={val_loss:.1f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "fakencoder_best.pt")
            print(f"   -> Saved best  (val_loss={val_loss:.4f})")

    torch.save(model.state_dict(), MODEL_DIR / "fakencoder_final.pt")
    print(f"\n[4] Done. Best val_loss={best_val_loss:.4f}")
    print(f"   Checkpoints:")
    print(f"     {MODEL_DIR}/fakencoder_best.pt")
    print(f"     {MODEL_DIR}/fakencoder_final.pt")

    if args.smoke:
        print("\n[smoke] PASSED - shapes and loss computed correctly")


if __name__ == "__main__":
    main()
