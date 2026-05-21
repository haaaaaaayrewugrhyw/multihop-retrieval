"""
FakeEncoder Training — retrieval_v2
====================================

Architecture
------------
  Encoder1 (BERT-base, 12-layer):
      A_text  →  hidden_states [n_A × 768]  =  K1, V1

  FakeEncoder (3-layer Transformer):
      F = A_cls expanded [n_slots × 768]   (warm start from A's CLS, NOT zeros)
      F  →  FakeEncoder  →  K2, V2  [n_slots × 768]

  Decoder (auto-regressive, teacher-forced):
      for each position t in B:
          h_t     = MaskedSelfAttn(B[0..t-1])
          r1_t    = CrossAttn(h_t → K1, V1)
          r2_t    = CrossAttn(h_t → K2, V2)   [TRACK attn_weights]
          meta_w  = softmax(linear([r1_t, r2_t]))
          merged  = meta_w[0]·r1_t + meta_w[1]·r2_t
          → predict B[t]

  Loss: L_ce  = CrossEntropy(predictions, B_tokens)
        L_delta = 1 - cosine_sim( proj(F_final_mean),
                                   normalize(proj(b_cls) - proj(a_cls)) )
        L_total = L_ce + LAMBDA_DELTA * L_delta

  F update (reverse cross-attention + GRU):
      feedback[i] = LayerNorm( Σ_t attn2[t, i] · h_t )
      F[i]        = GRUCell( feedback[i], F[i] )

  Complement (graph-build time):
      iterate F for N_ITER passes with teacher-forced B,
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
N_SLOTS       = 8
N_FAKE_LAYERS = 3
N_DEC_LAYERS  = 2
N_ITER        = 3        # same for train AND test (was train=3, test=5 — now consistent)
D_PROJ        = 128
VOCAB_SIZE    = 30522

MAX_A_LEN     = 128
MAX_B_LEN     = 64

BATCH_SIZE    = 16
LR            = 2e-5
N_EPOCHS      = 3
WARMUP_FRAC   = 0.06
MAX_GRAD_NORM = 1.0
LAMBDA_DELTA  = 0.5      # weight for complement alignment loss

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")
MODEL_DIR   = Path(__file__).parent / "models"
CACHE_DIR   = Path(__file__).parent / "cache"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
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
        x:           torch.Tensor,
        K1:          torch.Tensor,
        V1:          torch.Tensor,
        K2:          torch.Tensor,
        V2:          torch.Tensor,
        causal_mask: torch.Tensor,
        key_pad_a:   Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_self, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln1(x + h_self)
        h_query = x

        r1, _ = self.cross_attn1(x, K1, V1, key_padding_mask=key_pad_a, need_weights=False)
        r2, attn2 = self.cross_attn2(x, K2, V2, need_weights=True)

        meta_w = torch.softmax(self.meta(torch.cat([r1, r2], dim=-1)), dim=-1)
        merged = meta_w[..., :1] * r1 + meta_w[..., 1:] * r2
        x = self.ln2(x + merged)
        x = self.ln3(x + self.ffn(x))
        return x, h_query, attn2


# ── FakeEncoder Model ─────────────────────────────────────────────────────────

class FakeEncoderModel(nn.Module):

    def __init__(
        self,
        n_slots:       int = N_SLOTS,
        n_fake_layers: int = N_FAKE_LAYERS,
        n_dec_layers:  int = N_DEC_LAYERS,
        n_iter:        int = N_ITER,
        d_proj:        int = D_PROJ,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.n_iter  = n_iter

        # Encoder1: BERT processes A
        self.encoder1 = BertModel.from_pretrained("bert-base-uncased")
        self.encoder1.gradient_checkpointing_enable()

        # FakeEncoder: 3-layer Transformer over F slots
        fe_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.fake_encoder = nn.TransformerEncoder(fe_layer, num_layers=n_fake_layers)

        # Decoder
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(n_dec_layers)])
        self.pos_emb   = nn.Embedding(MAX_B_LEN + 4, D_MODEL)
        self.ln_dec_in = nn.LayerNorm(D_MODEL)

        # F update: reverse cross-attention + GRU
        self.gru_cell    = nn.GRUCell(D_MODEL, D_MODEL)
        self.ln_feedback = nn.LayerNorm(D_MODEL)
        self.ln_F        = nn.LayerNorm(D_MODEL)

        # LM head (weight-tied to BERT vocab embeddings)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.encoder1.embeddings.word_embeddings.weight

        # Single shared projection: used for complement vectors AND passage/query encoding
        # Maps both FakeEncoder output and BERT CLS to the same 128-dim scoring space
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, d_proj),
        )

    # ── Encoder1 ─────────────────────────────────────────────────────────────

    def _encode_a(self, input_ids_a, attn_mask_a):
        out = self.encoder1(input_ids=input_ids_a, attention_mask=attn_mask_a)
        hidden  = out.last_hidden_state
        key_pad = (attn_mask_a == 0)
        return hidden, key_pad

    # ── FakeEncoder ──────────────────────────────────────────────────────────

    def _fake_encode(self, F_slots):
        return self.fake_encoder(F_slots)

    # ── Decoder ──────────────────────────────────────────────────────────────

    def _decode(self, dec_input_ids, K1, V1, K2, V2, key_pad_a):
        seq_B  = dec_input_ids.size(1)
        device = dec_input_ids.device

        tok_emb = self.encoder1.embeddings.word_embeddings(dec_input_ids)
        pos_ids = torch.arange(seq_B, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_ids)
        x = self.ln_dec_in(tok_emb + pos_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_B, device=device)

        h_query_out, attn2_out = x, None
        for layer in self.decoder_layers:
            x, h_query_out, attn2_out = layer(x, K1, V1, K2, V2, causal_mask, key_pad_a)

        logits = self.lm_head(x)
        return logits, h_query_out, attn2_out

    # ── F update ─────────────────────────────────────────────────────────────

    def _update_F(self, F_slots, h_query, attn2):
        feedback = torch.einsum("bts,btd->bsd", attn2, h_query)
        feedback = self.ln_feedback(feedback)

        B, n_slots, D = F_slots.shape
        F_flat  = F_slots.reshape(B * n_slots, D)
        fb_flat = feedback.reshape(B * n_slots, D)

        new_F_flat = self.gru_cell(fb_flat, F_flat)
        new_F      = self.ln_F(new_F_flat.reshape(B, n_slots, D))
        return new_F

    # ── Full forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids_a:   torch.Tensor,
        attn_mask_a:   torch.Tensor,
        dec_input_ids: torch.Tensor,
        input_ids_b:   Optional[torch.Tensor] = None,
        attn_mask_b:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits [B, seq_B, V], L_delta scalar).
        L_delta = 0 if input_ids_b not provided (validation path).
        """
        K1, key_pad_a = self._encode_a(input_ids_a, attn_mask_a)
        V1 = K1

        # Warm start: F initialized from A's CLS embedding, not zeros
        B = input_ids_a.size(0)
        F_slots = K1[:, 0:1, :].detach().expand(-1, self.n_slots, -1).clone()

        logits = None
        for _ in range(self.n_iter):
            K2 = V2 = self._fake_encode(F_slots)
            logits, h_query, attn2 = self._decode(dec_input_ids, K1, V1, K2, V2, key_pad_a)
            F_slots = self._update_F(F_slots, h_query, attn2)

        # Complement vector — proj NOW gets gradients every step
        F_final = self._fake_encode(F_slots)
        f_mean  = F_final.mean(dim=1)                 # [B, D_MODEL]
        comp    = self.proj(f_mean)                   # [B, D_PROJ]
        comp_n  = F.normalize(comp, dim=-1)

        # L_delta: train proj to align complement with B-A direction in shared space
        L_delta = torch.tensor(0.0, device=input_ids_a.device)
        if input_ids_b is not None:
            # BERT(B) in no_grad — save ~3GB VRAM, proj still gets gradient via b_cls value
            with torch.no_grad():
                b_out = self.encoder1(input_ids=input_ids_b, attention_mask=attn_mask_b)
                b_cls = b_out.last_hidden_state[:, 0, :]   # [B, D_MODEL]
            a_cls = K1[:, 0, :].detach()                   # [B, D_MODEL]

            # Same proj maps both BERT CLS and FakeEncoder output → shared 128-dim space
            a_128 = self.proj(a_cls)
            b_128 = self.proj(b_cls)
            delta  = F.normalize(b_128 - a_128, dim=-1)   # [B, D_PROJ]

            L_delta = 1.0 - (comp_n * delta).sum(-1).mean()

        return logits, L_delta

    # ── Complement extraction (graph-build time) ──────────────────────────────

    @torch.no_grad()
    def extract_complement(
        self,
        input_ids_a: torch.Tensor,
        attn_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine F over n_iter passes (teacher-forced with B),
        extract complement = L2_norm( proj( mean_pool( FakeEncoder(F_final) ) ) )
        Returns [B, D_PROJ] L2-normalized complement vectors.
        """
        dec_input = input_ids_b[:, :-1]

        K1, key_pad_a = self._encode_a(input_ids_a, attn_mask_a)
        V1 = K1

        B = input_ids_a.size(0)
        # Warm start from A's CLS — same as training
        F_slots = K1[:, 0:1, :].expand(-1, self.n_slots, -1).clone()

        for _ in range(self.n_iter):
            K2 = V2 = self._fake_encode(F_slots)
            _, h_query, attn2 = self._decode(dec_input, K1, V1, K2, V2, key_pad_a)
            F_slots = self._update_F(F_slots, h_query, attn2)

        F_final = self._fake_encode(F_slots)
        comp    = F_final.mean(dim=1)
        comp    = self.proj(comp)
        return F.normalize(comp, dim=-1)

    @torch.no_grad()
    def encode_query(
        self,
        input_ids:  torch.Tensor,
        attn_mask:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode any text (query or passage) → [B, D_PROJ] L2-normalized.
        Uses mean pooling (better than CLS for retrieval) + shared proj.
        """
        out    = self.encoder1(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state                                   # [B, seq, D]
        mask_f = attn_mask.unsqueeze(-1).float()                        # [B, seq, 1]
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)  # [B, D]
        return F.normalize(self.proj(pooled), dim=-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HopPairDataset(Dataset):
    def __init__(self, quadruples: List[Dict], id_to_text: Dict[str, str]):
        self.pairs: List[Tuple[str, str]] = []
        for q in quadruples:
            a = id_to_text.get(q["chunk_a_id"], "")
            b = id_to_text.get(q["chunk_b_pos_id"], "")
            if a and b:
                self.pairs.append((a, b))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_collator(tokenizer: BertTokenizerFast):
    def collate(batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        a_texts = [item[0] for item in batch]
        b_texts = [item[1] for item in batch]

        enc_a = tokenizer(
            a_texts, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        # Decoder input (teacher forcing): truncated to MAX_B_LEN
        enc_b = tokenizer(
            b_texts, max_length=MAX_B_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        # Full B encoding for BERT(B) in L_delta: use MAX_A_LEN for proper representation
        enc_b_full = tokenizer(
            b_texts, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids_a": enc_a.input_ids,
            "attn_mask_a": enc_a.attention_mask,
            "dec_input":   enc_b.input_ids[:, :-1],
            "dec_target":  enc_b.input_ids[:, 1:],
            "input_ids_b": enc_b_full.input_ids,
            "attn_mask_b": enc_b_full.attention_mask,
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
) -> Tuple[float, float]:
    model.train()
    total_ce, total_delta, n = 0.0, 0.0, 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        ia = batch["input_ids_a"].to(DEVICE)
        ma = batch["attn_mask_a"].to(DEVICE)
        di = batch["dec_input"].to(DEVICE)
        dt = batch["dec_target"].to(DEVICE)
        ib = batch["input_ids_b"].to(DEVICE)
        mb = batch["attn_mask_b"].to(DEVICE)

        with autocast("cuda", enabled=AMP_ENABLED):
            logits, L_delta = model(ia, ma, di, input_ids_b=ib, attn_mask_b=mb)
            B, T, V = logits.shape
            L_ce = F.cross_entropy(
                logits.reshape(B * T, V),
                dt.reshape(B * T),
                ignore_index=0,
            )
            loss = L_ce + LAMBDA_DELTA * L_delta

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_ce    += L_ce.item()
        total_delta += L_delta.item()
        n += 1

        if step > 0 and step % log_every == 0:
            tqdm.write(
                f"  step {step} | L_ce={total_ce/n:.4f} | L_delta={total_delta/n:.4f}"
            )

    return total_ce / max(n, 1), total_delta / max(n, 1)


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
            logits, _ = model(ia, ma, di)   # no B needed for val loss
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
    parser.add_argument("--smoke",        action="store_true")
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

    id_to_text  = {c["chunk_id"]: c["text"] for c in train_corpus + val_corpus}
    train_ds    = HopPairDataset(train_quads, id_to_text)
    val_ds      = HopPairDataset(val_quads,   id_to_text)
    print(f"   Train pairs : {len(train_ds):,}  |  Val pairs : {len(val_ds):,}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collator  = make_collator(tokenizer)
    nw        = 0 if args.smoke else 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )

    print("\n[2] Building FakeEncoderModel ...")
    model  = FakeEncoderModel().to(DEVICE)
    scaler = GradScaler("cuda", enabled=AMP_ENABLED)

    # Epoch 1: freeze BERT — FakeEncoder + Decoder + proj warm up first
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

    print(f"\n[3] Training {args.epochs} epoch(s) ...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        if epoch == 2:
            print("   Unfreezing BERT with lr * 0.1 ...")
            for p in model.encoder1.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.encoder1.parameters(),                      "lr": args.lr * 0.1},
                {"params": [p for n, p in model.named_parameters()
                            if "encoder1" not in n],                         "lr": args.lr},
            ])
            remaining = (args.epochs - 1) * len(train_loader)
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, remaining)

        train_ce, train_delta = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss              = validate(model, val_loader)

        print(
            f"   Epoch {epoch}/{args.epochs} | "
            f"L_ce={train_ce:.4f} | L_delta={train_delta:.4f} | "
            f"val_loss={val_loss:.4f}"
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
