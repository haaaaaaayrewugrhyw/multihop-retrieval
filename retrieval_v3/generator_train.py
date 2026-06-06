"""
generator_train.py -- ComplementGenerator (retrieval_v3)
=========================================================

G+D Architecture
----------------
  Generator G(A, B) -> 128-dim complement edge vector
    - Shared BERT encoder for A and B (two forward passes, no iteration)
    - B->A cross-attention: B_in_A = CrossAttn(Q=B, K=A, V=A)
    - Learned lambda gate: comp_tokens = B_sa - lambda * B_in_A
    - Complement gate: g_i = 1 - max_j Attn(b_i, A)  (which B tokens are NOT in A)
    - g-weighted complement pooling -> LayerNorm -> proj -> 128-dim L2-norm

  Discriminator D(A, e) -> reconstruct B tokens
    - Context dropout: 40% of A positions hidden from D -> forces D to use edge e
    - Edge e projected to 768-dim as a single K2/V2 virtual token
    - 2-layer causal decoder (DecoderLayer reused from retrieval_v2/fakencoder_train.py)
    - L_rec = CrossEntropy(predictions, B_tokens)

  Training loss:
    L_total = L_rec + 0.1 * L_div
    L_div = InfoNCE in-batch diversity (prevents all edges collapsing to same vector)

Interface (drop-in replacement for FakeEncoderModel in run_eval.py):
    model.encode_query(input_ids, attn_mask)                -> [B, 128] L2-norm
    model.extract_complement(ids_a, mask_a, ids_b)          -> [B, 128] L2-norm

Collapse monitor: prints cos_sim(edge, encode_query(B)) every LOG_EVERY steps.
  Target: < 0.80 by epoch 2. Both v1 and v2 were stuck at > 0.95.

Usage:
    python generator_train.py               # full 3-epoch training
    python generator_train.py --smoke       # 50 examples, 1 epoch, CPU
    python generator_train.py --max_examples 2000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

# Reuse DecoderLayer from v2 -- do not copy
_V2 = Path(__file__).parent.parent / "retrieval_v2"
sys.path.insert(0, str(_V2))
from fakencoder_train import DecoderLayer
from data_loader import build_chain_quadruples, load_musique

# ---- Constants ----------------------------------------------------------------
D_MODEL    = 768
N_HEADS    = 8
D_HEAD     = D_MODEL // N_HEADS   # 96
D_PROJ     = 128
VOCAB_SIZE = 30522
MAX_A_LEN  = 128
MAX_B_LEN  = 64

BATCH_SIZE    = 8        # halved vs v2 (two BERT passes per step)
LR            = 1e-5     # conservative for joint G+D
N_EPOCHS      = 3
WARMUP_FRAC   = 0.06
MAX_GRAD_NORM = 1.0
LAMBDA_DIV    = 0.1      # InfoNCE diversity weight
CONTEXT_DROP  = 0.40     # fraction of A positions dropped in D during training
LOG_EVERY     = 200

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")
MODEL_DIR   = Path(__file__).parent / "models"
CACHE_DIR   = Path(__file__).parent / "cache"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ---- Model -------------------------------------------------------------------

class ComplementGenerator(nn.Module):
    """
    G+D complement encoder.
    Drop-in replacement for FakeEncoderModel: same encode_query() and
    extract_complement() signatures, so run_eval.py works unchanged.
    """

    def __init__(self, n_dec_layers: int = 2, d_proj: int = D_PROJ,
                 use_gate: bool = True, context_drop: float = CONTEXT_DROP):
        super().__init__()
        # Ablation switches (saved so eval-time loading reproduces them):
        #   use_gate=False     -> uniform pooling instead of complement gate g_i
        #   context_drop=0.0   -> no context dropout (expected to collapse)
        self.use_gate     = use_gate
        self.context_drop = context_drop

        # Shared BERT: encodes both A and B
        self.encoder1 = BertModel.from_pretrained("bert-base-uncased")
        self.encoder1.gradient_checkpointing_enable()

        # Match-scaled complement (validated in synthetic_complement_test.py):
        # one cosine-similarity attention (B->A) with a LEARNABLE temperature.
        # log_tau init -1.8 -> tau = softplus(-1.8)+0.01 ~= 0.16 (sharp, so real
        # matches concentrate instead of averaging into a diffuse mean(A)).
        self.log_tau = nn.Parameter(torch.tensor(-1.8))

        # Learned lambda gate (Flamingo/ReZero principle)
        # sigmoid(0) + 0.20 = 0.70 at init -- moderate subtraction strength
        self.lambda_gate = nn.Parameter(torch.zeros(1))

        # Stabilize complement representation before projection
        self.ln_comp = nn.LayerNorm(D_MODEL)

        # Project 128-dim edge up to 768-dim for decoder K2/V2
        self.e_proj = nn.Linear(d_proj, D_MODEL)

        # Decoder (DecoderLayer from retrieval_v2, reused unchanged)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer() for _ in range(n_dec_layers)]
        )
        self.pos_emb   = nn.Embedding(MAX_B_LEN + 4, D_MODEL)
        self.ln_dec_in = nn.LayerNorm(D_MODEL)

        # LM head: weight-tied to BERT embeddings
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.encoder1.embeddings.word_embeddings.weight

        # Shared projection: maps both complement and passage vectors to 128-dim
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, d_proj),
        )

    # ---- Internal helpers ---------------------------------------------------

    def _encode(
        self,
        input_ids:  torch.Tensor,
        attn_mask:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BERT forward. Returns (hidden [B, T, D], key_pad_mask [B, T])."""
        out     = self.encoder1(input_ids=input_ids, attention_mask=attn_mask)
        hidden  = out.last_hidden_state
        key_pad = (attn_mask == 0)   # True = padding position
        return hidden, key_pad

    def _generate_edge(
        self,
        input_ids_a: torch.Tensor,
        attn_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attn_mask_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generator G(A, B) -> edge vector.

        Returns:
            edge      : [B, D_PROJ] L2-normalized complement vector
            A_sa      : [B, T_A, D] A's hidden states (reused by decoder)
            key_pad_a : [B, T_A]  A's full padding mask (before context dropout)
        """
        A_sa, key_pad_a = self._encode(input_ids_a, attn_mask_a)  # [B, T_A, D]
        B_sa, _         = self._encode(input_ids_b, attn_mask_b)  # [B, T_B, D]

        # ── Unified match-scaled complement (validated by synthetic_complement_test.py)
        # Cosine similarity B->A with a learnable temperature. ONE attention drives
        # both the subtraction and the gate, so the "match" is consistent.
        A_n = F.normalize(A_sa, dim=-1)
        B_n = F.normalize(B_sa, dim=-1)
        tau = F.softplus(self.log_tau) + 1e-2
        sim = torch.matmul(B_n, A_n.transpose(-2, -1)) / tau      # [B, T_B, T_A] cosine/tau
        sim = sim.masked_fill(key_pad_a.unsqueeze(1), float("-inf"))
        attn = torch.softmax(sim, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        B_in_A = torch.matmul(attn, A_sa)                # [B, T_B, D]  real A values (same space as B_sa)
        match  = attn.max(dim=-1).values                 # [B, T_B]  how much B token i is actually in A

        # MATCH-SCALED subtraction (the fix): novel tokens (low match) keep their
        # content; only tokens that truly appear in A get their A-part removed.
        # Old leak was `B_sa - lam*B_in_A` which subtracted a diffuse mean(A) even
        # from novel tokens (softmax sums to 1 regardless of any real match).
        lam = torch.sigmoid(self.lambda_gate) + 0.20     # scalar in [0.20, 1.20]
        comp_tokens = B_sa - lam * match.unsqueeze(-1) * B_in_A   # [B, T_B, D]

        # Complement gate from the SAME match: g_i high => B token not in A
        g_i = 1.0 - match                                # [B, T_B]

        # ABLATION: use_gate=False -> uniform weights (plain mean pool of comp_tokens)
        if not self.use_gate:
            g_i = torch.ones_like(g_i)

        # Complement-weighted pooling: only real B tokens, weighted by g_i
        b_real    = attn_mask_b.float()                   # [B, T_B]
        g_masked  = g_i * b_real
        g_sum     = g_masked.sum(dim=1, keepdim=True).clamp(min=1e-9)
        edge_raw  = (g_masked.unsqueeze(-1) * comp_tokens).sum(1) / g_sum  # [B, 768]

        edge = F.normalize(self.proj(self.ln_comp(edge_raw)), dim=-1)       # [B, D_PROJ]
        return edge, A_sa, key_pad_a

    def _decode(
        self,
        dec_input_ids: torch.Tensor,
        A_sa:          torch.Tensor,
        key_pad_a_dec: torch.Tensor,   # dropout-masked version
        edge:          torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator D: reconstruct B from dropout-masked A + edge."""
        seq_B  = dec_input_ids.size(1)
        device = dec_input_ids.device

        tok_emb = self.encoder1.embeddings.word_embeddings(dec_input_ids)
        pos_ids = torch.arange(seq_B, device=device).unsqueeze(0)
        x = self.ln_dec_in(tok_emb + self.pos_emb(pos_ids))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_B, device=device)

        # Edge as a single K2/V2 "virtual token" [B, 1, 768]
        K2 = V2 = self.e_proj(edge).unsqueeze(1)

        for layer in self.decoder_layers:
            x, _, _ = layer(x, A_sa, A_sa, K2, V2, causal_mask, key_pad_a_dec)

        return self.lm_head(x)    # [B, seq_B, VOCAB]

    # ---- Training forward ---------------------------------------------------

    def forward(
        self,
        input_ids_a:   torch.Tensor,
        attn_mask_a:   torch.Tensor,
        dec_input_ids: torch.Tensor,
        input_ids_b:   torch.Tensor,
        attn_mask_b:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits [B, T_B-1, VOCAB], edge [B, D_PROJ]).
        G always sees full A. D sees context-dropout-masked A.
        """
        edge, A_sa, key_pad_a = self._generate_edge(
            input_ids_a, attn_mask_a, input_ids_b, attn_mask_b
        )

        # Context dropout: randomly hide context_drop% of A positions from D
        # (ABLATION: context_drop=0.0 -> D sees full A -> expected to collapse)
        if self.training and self.context_drop > 0.0:
            keep = torch.rand_like(attn_mask_a.float()) > self.context_drop
            attn_mask_a_dec = attn_mask_a * keep.long()
        else:
            attn_mask_a_dec = attn_mask_a

        key_pad_a_dec = (attn_mask_a_dec == 0)
        logits = self._decode(dec_input_ids, A_sa, key_pad_a_dec, edge)

        return logits, edge

    # ---- Inference interface (same as FakeEncoderModel) ---------------------

    @torch.no_grad()
    def extract_complement(
        self,
        input_ids_a: torch.Tensor,
        attn_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Graph-build time: G(A, B) -> [B, D_PROJ] L2-normalized complement.
        No B attention mask needed -- derived from input_ids_b (non-zero = real).
        """
        attn_mask_b = (input_ids_b != 0).long()
        edge, _, _  = self._generate_edge(
            input_ids_a, attn_mask_a, input_ids_b, attn_mask_b
        )
        return edge

    @torch.no_grad()
    def encode_query(
        self,
        input_ids:  torch.Tensor,
        attn_mask:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode any text (passage or query) -> [B, D_PROJ] L2-normalized.
        Same as FakeEncoderModel.encode_query -- used for passage embeddings.
        """
        out    = self.encoder1(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state
        mask_f = attn_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return F.normalize(self.proj(pooled), dim=-1)


# ---- Dataset -----------------------------------------------------------------

class HopPairDataset(Dataset):
    """(A_text, B_pos_text) consecutive hop pairs from MuSiQue chain quadruples."""

    def __init__(self, quadruples: List[Dict], id_to_text: Dict[str, str]):
        self.pairs = [
            (id_to_text[q["chunk_a_id"]], id_to_text[q["chunk_b_pos_id"]])
            for q in quadruples
            if q["chunk_a_id"] in id_to_text and q["chunk_b_pos_id"] in id_to_text
        ]

    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


def make_collator(tokenizer: BertTokenizerFast):
    def collate(batch: List[Tuple[str, str]]) -> Dict:
        a_texts = [x[0] for x in batch]
        b_texts = [x[1] for x in batch]

        enc_a = tokenizer(
            a_texts, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_b = tokenizer(
            b_texts, max_length=MAX_B_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        # Decoder: teacher-forced -- input is B[:-1], labels are B[1:]
        dec_ids = enc_b["input_ids"][:, :-1]
        labels  = enc_b["input_ids"][:, 1:].clone()
        labels[labels == tokenizer.pad_token_id] = -100   # ignore padding in loss

        return {
            "input_ids_a":   enc_a["input_ids"],
            "attn_mask_a":   enc_a["attention_mask"],
            "dec_input_ids": dec_ids,
            "input_ids_b":   enc_b["input_ids"],
            "attn_mask_b":   enc_b["attention_mask"],
            "labels":        labels,
        }
    return collate


# ---- Loss --------------------------------------------------------------------

def infonce_diversity_loss(edge: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE in-batch diversity: pushes different edges apart.
    Prevents G from mapping all (A, B) pairs to the same vector.
    """
    e = F.normalize(edge, dim=-1)
    sim = (e @ e.T) / temperature               # [B, B]
    labels = torch.arange(e.size(0), device=e.device)
    return F.cross_entropy(sim, labels)


# ---- Training ----------------------------------------------------------------

def train_epoch(
    model:     ComplementGenerator,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
) -> Tuple[float, float, float]:
    model.train()
    total_rec, total_div, n_batches = 0.0, 0.0, 0

    for step, batch in enumerate(tqdm(loader, desc="Train G+D", leave=False)):
        ids_a  = batch["input_ids_a"].to(DEVICE)
        mask_a = batch["attn_mask_a"].to(DEVICE)
        dec    = batch["dec_input_ids"].to(DEVICE)
        ids_b  = batch["input_ids_b"].to(DEVICE)
        mask_b = batch["attn_mask_b"].to(DEVICE)
        lbl    = batch["labels"].to(DEVICE)

        with autocast("cuda", enabled=AMP_ENABLED):
            logits, edge = model(ids_a, mask_a, dec, ids_b, mask_b)
            L_rec = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), lbl.reshape(-1), ignore_index=-100
            )
            L_div = infonce_diversity_loss(edge)
            loss  = L_rec + LAMBDA_DIV * L_div

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_rec += L_rec.item()
        total_div += L_div.item()
        n_batches += 1

        if step > 0 and step % LOG_EVERY == 0:
            # Collapse monitor: how similar is the edge to encode_query(B)?
            with torch.no_grad():
                b_vecs = model.encode_query(ids_b[:4], mask_b[:4])
                csim   = F.cosine_similarity(edge[:4].detach(), b_vecs).mean().item()
            tqdm.write(
                f"  step {step} | L_rec={total_rec/n_batches:.4f} "
                f"| L_div={total_div/n_batches:.4f} "
                f"| collapse_sim={csim:.4f}  (target <0.80)"
            )

    return total_rec / max(n_batches, 1), total_div / max(n_batches, 1), n_batches


@torch.no_grad()
def validate(
    model:  ComplementGenerator,
    loader: DataLoader,
    max_steps: int = 120,
) -> Tuple[float, float]:
    model.eval()
    total_rec, n_batches = 0.0, 0
    collapse_sims: List[float] = []

    for step, batch in enumerate(tqdm(loader, desc="Val G+D", leave=False)):
        if step >= max_steps:
            break

        ids_a  = batch["input_ids_a"].to(DEVICE)
        mask_a = batch["attn_mask_a"].to(DEVICE)
        dec    = batch["dec_input_ids"].to(DEVICE)
        ids_b  = batch["input_ids_b"].to(DEVICE)
        mask_b = batch["attn_mask_b"].to(DEVICE)
        lbl    = batch["labels"].to(DEVICE)

        logits, edge = model(ids_a, mask_a, dec, ids_b, mask_b)
        L_rec = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), lbl.reshape(-1), ignore_index=-100
        )
        total_rec += L_rec.item()
        n_batches += 1

        b_vecs = model.encode_query(ids_b[:4], mask_b[:4])
        collapse_sims.append(F.cosine_similarity(edge[:4], b_vecs).mean().item())

    mean_collapse = sum(collapse_sims) / max(len(collapse_sims), 1)
    return total_rec / max(n_batches, 1), mean_collapse


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",         action="store_true")
    parser.add_argument("--max_examples",  type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",        type=int,   default=N_EPOCHS)
    parser.add_argument("--lr",            type=float, default=LR)
    # Ablation flags (for the contribution's ablation study):
    parser.add_argument("--context_drop",  type=float, default=CONTEXT_DROP,
                        help="A-token dropout in D. 0.0 = ablation (expect collapse).")
    parser.add_argument("--no_gate",       action="store_true",
                        help="Disable complement gate g_i (uniform pooling ablation).")
    parser.add_argument("--tag",           type=str,   default="",
                        help="Suffix for checkpoint names, e.g. _nodrop / _nogate.")
    args = parser.parse_args()

    if args.smoke:
        args.max_examples = 50
        args.batch_size   = 2
        args.epochs       = 1
        print("[smoke] 50 examples, 1 epoch, CPU-friendly")

    print(f"Ablation: context_drop={args.context_drop}  use_gate={not args.no_gate}  tag='{args.tag}'")

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ---- Data ---------------------------------------------------------------
    print("\n[1] Loading MuSiQue ...")
    train_corpus, train_queries = load_musique(
        split="train", max_examples=args.max_examples, cache=not args.smoke,
    )
    val_max = 10 if args.smoke else 400
    val_corpus, val_queries = load_musique(
        split="validation", max_examples=val_max, cache=not args.smoke,
    )
    id_to_text = {c["chunk_id"]: c["text"] for c in train_corpus + val_corpus}

    train_quads = build_chain_quadruples(train_corpus, train_queries)
    val_quads   = build_chain_quadruples(val_corpus,   val_queries)
    print(f"   Train pairs: {len(train_quads):,} | Val pairs: {len(val_quads):,}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collator  = make_collator(tokenizer)
    nw = 0 if args.smoke else 2

    train_loader = DataLoader(
        HopPairDataset(train_quads, id_to_text),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        HopPairDataset(val_quads, id_to_text),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    # ---- Model --------------------------------------------------------------
    print("\n[2] Building ComplementGenerator ...")
    model = ComplementGenerator(
        use_gate=not args.no_gate, context_drop=args.context_drop,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {total_params:.1f}M")

    # ---- Optimizer ----------------------------------------------------------
    scaler       = GradScaler("cuda", enabled=AMP_ENABLED)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[3] Training {args.epochs} epoch(s) | {len(train_loader):,} steps/epoch")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_rec, train_div, _ = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss, val_collapse  = validate(model, val_loader)

        print(
            f"   Epoch {epoch}/{args.epochs} | "
            f"train_rec={train_rec:.4f}  train_div={train_div:.4f} | "
            f"val_loss={val_loss:.4f}  collapse_sim={val_collapse:.4f}"
        )

        if val_collapse > 0.95:
            print("   WARNING: collapse_sim > 0.95 -- complement = encode(B). Same as v2.")
            print("   Consider: increase CONTEXT_DROP or check context dropout is active.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / f"generator_best{args.tag}.pt")
            print(f"   -> Saved best (val_loss={val_loss:.4f}, collapse_sim={val_collapse:.4f})")

    torch.save(model.state_dict(), MODEL_DIR / f"generator_final{args.tag}.pt")
    print(f"\n[4] Done. Best val_loss={best_val_loss:.4f}")
    print(f"   Checkpoints: {MODEL_DIR}/generator_best{args.tag}.pt  |  generator_final{args.tag}.pt")

    if args.smoke:
        print("\n[smoke] PASSED")


if __name__ == "__main__":
    main()
