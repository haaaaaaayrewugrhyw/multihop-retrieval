"""
Model 1 — ComplementEncoder

Architecture:
    Joint BERT encoding of [CLS] A [SEP] B [SEP].
    BERT self-attention attends across A and B at every layer, so B token
    representations naturally encode "B's content conditioned on A."
    B-side token matrix is extracted and projected to 128-dim (ColBERT style).

Why joint encoding instead of pooled vectors:
    Pooling A and B to single vectors before comparison loses token-level structure.
    Joint BERT encoding computes the A↔B interaction through 12 attention layers —
    this is the deep equivalent of the ESIM residual (B - aligned_A), not a shallow
    single-layer approximation.

Loss — Chain Contrastive (C-anchor):
    Trained on 3/4-hop examples only (those with a known next-hop C).
    For each directed pair (A → B_pos) with next-hop C:
        mean_pool(complement(A, B_pos)) should be closer to mean_pool(C_standalone)
        than mean_pool(complement(A, B_neg_i)) is, for all in-example distractors.
    This is query-agnostic: C serves as the supervision target, not Q.
    2-hop examples (no C) are used by Model 2 training instead.

Training data: MuSiQue train split, 3/4-hop questions only (~7K questions → ~56K instances)
    Chain order from question_decomposition[i]['paragraph_support_idx'].
    Hard negatives: in-example distractors (already topically challenging).

Usage:
    python model1_train.py --max_examples 300    # smoke test on Colab/local
    python model1_train.py                        # full training (Colab recommended)
    python model1_train.py --eval_only            # evaluate saved checkpoint

Output:
    models/model1_complement.pt    trained ComplementEncoder weights
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))

CACHE_DIR = Path(__file__).parent / "cache"
MODEL_DIR = Path(__file__).parent / "models"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

BERT_NAME   = "bert-base-uncased"
PROJ_DIM    = 128        # ColBERT-style token projection dimension
DROPOUT     = 0.1
MARGIN      = 0.2        # cosine margin for contrastive loss
BATCH_SIZE  = 32         # increased from 8 — needs ≥8 has_c examples/batch for InfoNCE
LR          = 2e-5       # standard for BERT fine-tuning
EPOCHS      = 3
EVAL_EVERY  = 200
MAX_LEN_AB  = 256        # max tokens for [CLS] A [SEP] B [SEP]
MAX_LEN_C   = 128        # max tokens for standalone C encoding
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
# L_next (InfoNCE): complement(A,B) must predict C — applied to has_c examples only.
#   Replaces old L_content for non-final hops. Makes edge vectors point toward the
#   next hop so hop-2+ FAISS navigation actually finds C.
# L_chain:      among edges from A, the correct one predicts C better than wrong ones.
#   Discriminative counterpart to L_next. Increased weight now that L_content no longer
#   dominates.
# L_content:    weak regression toward B — applied to no_c (final-hop) examples only.
#   Keeps final-hop edge vectors from becoming random; weight reduced from 1.0 to 0.2.
# L_orthogonal: complement must NOT encode A's content — all examples.
LAMBDA_NEXT       = 1.5
LAMBDA_CHAIN      = 1.0
LAMBDA_CONTENT    = 0.2
LAMBDA_ORTHOGONAL = 0.3
TEMPERATURE       = 0.1  # InfoNCE softmax temperature


# ── Model ──────────────────────────────────────────────────────────────────────

class ComplementEncoder(nn.Module):
    """
    Encodes a (passage_A, passage_B) pair → B-side token matrix conditioned on A.

    Input  : tokenised [CLS] A [SEP] B [SEP] + a boolean B-side mask
    Process: BERT forward (all layers attend across A and B)
    Output : complement_tokens  [batch, max_b_len, PROJ_DIM]  L2-normalised
             b_pad_mask         [batch, max_b_len]  True = valid token
    """

    def __init__(
        self,
        bert_name: str  = BERT_NAME,
        proj_dim:  int  = PROJ_DIM,
        dropout:   float = DROPOUT,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, proj_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, seq_len]
        attention_mask: torch.Tensor,   # [B, seq_len]
        token_type_ids: torch.Tensor,   # [B, seq_len]
        b_mask:         torch.Tensor,   # [B, seq_len] bool — True for B tokens to keep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden = out.last_hidden_state                    # [B, seq_len, 768]
        proj   = F.normalize(self.proj(hidden), dim=-1)  # [B, seq_len, PROJ_DIM]
        return _extract_masked_tokens(proj, b_mask)

    def encode_passage(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a standalone passage (used for C in the contrastive loss).
        Returns mean-pooled, L2-normalised vector [B, PROJ_DIM].
        """
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                    # [B, seq_len, 768]
        proj   = F.normalize(self.proj(hidden), dim=-1)  # [B, seq_len, PROJ_DIM]
        # Mean pool over non-padding tokens
        mask_f = attention_mask.unsqueeze(-1).float()     # [B, seq_len, 1]
        pool   = (proj * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return F.normalize(pool, dim=-1)                  # [B, PROJ_DIM]


def _extract_masked_tokens(
    tokens: torch.Tensor,   # [B, seq_len, dim]
    mask:   torch.Tensor,   # [B, seq_len] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather masked (B-side) tokens per example and pad to the longest in the batch.
    Returns (padded_tokens [B, max_len, dim], pad_mask [B, max_len]).
    """
    B, _, dim = tokens.shape
    lens      = mask.sum(dim=-1).clamp(min=1)          # [B]
    max_len   = int(lens.max().item())

    out      = torch.zeros(B, max_len, dim,  device=tokens.device)
    pad_mask = torch.zeros(B, max_len,       device=tokens.device, dtype=torch.bool)

    for i in range(B):
        n = int(lens[i].item())
        out[i, :n]      = tokens[i][mask[i]]
        pad_mask[i, :n] = True

    return out, pad_mask


def mean_pool(tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over valid (non-padded) token positions → [B, dim]."""
    mask_f = pad_mask.unsqueeze(-1).float()
    pool   = (tokens * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
    return F.normalize(pool, dim=-1)


# ── Loss ───────────────────────────────────────────────────────────────────────

class ChainContrastiveLoss(nn.Module):
    """
    Redesigned loss for next-hop prediction.

    L_next (InfoNCE, has_c only):
        complement(A,B) must be closer to its own C than to other C's in the batch.
        Forces edge vectors to predict the specific next-hop passage.
        Replaces old L_content for non-final hops — fixing the cosine=0.99 collapse.

    L_chain (margin ranking, has_c only):
        Among edges from A, the correct one (A→B_pos) predicts C better than
        wrong ones (A→B_neg). Discriminative signal on top of L_next.

    L_content (regression, no_c only, weak):
        complement(A,B) ≈ encode_passage(B) for final-hop pairs (no C exists).
        Prevents final-hop edge vectors from being random. Weight reduced to 0.2.

    L_orthogonal (all examples):
        |cos(complement, standalone_A)| → 0.
        Prevents complement from collapsing to A (the shortcut).
    """

    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        complement_pos:  torch.Tensor,        # [B, n_pos, 128]
        b_pad_mask_pos:  torch.Tensor,        # [B, n_pos] bool
        complement_negs: List[torch.Tensor],  # each [B, n_neg, 128]
        b_pad_masks_neg: List[torch.Tensor],  # each [B, n_neg] bool
        pool_c:          torch.Tensor,        # [n_c, 128]  standalone C (has_c rows only)
        pool_a:          torch.Tensor,        # [B, 128]  standalone A
        pool_b:          torch.Tensor,        # [B, 128]  standalone B_pos
        has_c:           torch.Tensor,        # [B] bool — True when C is available
    ) -> torch.Tensor:

        pool_pos = mean_pool(complement_pos, b_pad_mask_pos)   # [B, 128]
        dev      = complement_pos.device

        # ── L_orthogonal (all examples) ───────────────────────────────────────
        l_orthogonal = (pool_pos * pool_a).sum(-1).abs().mean()

        # ── has_c branch: L_next (InfoNCE) + L_chain ─────────────────────────
        if pool_c is not None and has_c.any():
            pool_pos_c = pool_pos[has_c]          # [n_c, 128]
            n_c        = pool_pos_c.shape[0]

            # L_next — InfoNCE: complement(A,B_i) must be closest to C_i
            # logits[i,j] = dot(complement_i, C_j) / τ
            logits = (pool_pos_c @ pool_c.T) / TEMPERATURE   # [n_c, n_c]
            labels = torch.arange(n_c, device=dev)
            l_next = F.cross_entropy(logits, labels)

            # L_chain — margin ranking: correct edge predicts C better than wrong edges
            score_pos = (pool_pos_c * pool_c).sum(-1)         # [n_c]
            l_chain   = torch.zeros(1, device=dev)
            count     = 0
            for comp_neg, mask_neg in zip(complement_negs, b_pad_masks_neg):
                pool_neg_c = mean_pool(comp_neg, mask_neg)[has_c]   # [n_c, 128]
                score_neg  = (pool_neg_c * pool_c).sum(-1)
                l_chain    = l_chain + F.relu(self.margin + score_neg - score_pos).mean()
                count += 1
            l_chain = l_chain / max(count, 1)
        else:
            l_next  = torch.zeros(1, device=dev)
            l_chain = torch.zeros(1, device=dev)

        # ── no_c branch: weak L_content (final-hop stability) ────────────────
        no_c = ~has_c
        if no_c.any():
            pool_pos_nc = pool_pos[no_c]
            pool_b_nc   = pool_b[no_c]
            l_content   = (1.0 - (pool_pos_nc * pool_b_nc).sum(-1)).mean()
        else:
            l_content = torch.zeros(1, device=dev)

        return (LAMBDA_NEXT       * l_next
              + LAMBDA_CHAIN      * l_chain
              + LAMBDA_CONTENT    * l_content
              + LAMBDA_ORTHOGONAL * l_orthogonal)


# ── Tokenizer helpers ──────────────────────────────────────────────────────────

def build_b_mask(input_ids: torch.Tensor, token_type_ids: torch.Tensor, sep_id: int) -> torch.Tensor:
    """
    B-side mask: token_type_ids == 1 AND not the trailing [SEP].
    Excludes the final [SEP] because it is a structural token, not B content.
    """
    return (token_type_ids == 1) & (input_ids != sep_id)


def tokenize_ab_pairs(
    texts_a: List[str],
    texts_b: List[str],
    tokenizer: BertTokenizerFast,
    max_length: int = MAX_LEN_AB,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Tokenise (A, B) as text/text_pair → add B-side mask.
    Returns (encoding dict, b_mask [batch, seq_len]).
    """
    enc = tokenizer(
        text=texts_a, text_pair=texts_b,
        max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    b_mask = build_b_mask(enc["input_ids"], enc["token_type_ids"], tokenizer.sep_token_id)
    return enc, b_mask


def tokenize_passages(
    texts: List[str],
    tokenizer: BertTokenizerFast,
    max_length: int = MAX_LEN_C,
) -> Dict[str, torch.Tensor]:
    """Tokenise standalone passages (used for C encoding)."""
    return tokenizer(
        text=texts,
        max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChainDataset(Dataset):
    """
    Wraps chain quadruples for Model 1 training.
    Only uses examples where chunk_c_id is not None (3/4-hop chains).
    """

    def __init__(self, quadruples: List[Dict], id_to_text: Dict[str, str]):
        self.data       = quadruples          # all hops — L_chain skipped when chunk_c_id is None
        self.id_to_text = id_to_text
        with_c    = sum(1 for q in quadruples if q.get("chunk_c_id") is not None)
        without_c = len(quadruples) - with_c
        print(f"[ChainDataset] {len(self.data):,} examples "
              f"({with_c:,} with C-anchor, {without_c:,} without)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        q = self.data[idx]
        c_id = q.get("chunk_c_id")
        return {
            "text_a":     self.id_to_text[q["chunk_a_id"]],
            "text_b_pos": self.id_to_text[q["chunk_b_pos_id"]],
            "text_b_negs":[self.id_to_text[nid] for nid in q["chunk_b_neg_ids"]],
            "text_c":     self.id_to_text[c_id] if c_id else None,
        }


def make_collate_fn(tokenizer: BertTokenizerFast):
    """Return a collate_fn that tokenises a batch of ChainDataset items."""

    def collate(batch: List[Dict]) -> Dict[str, object]:
        texts_a     = [x["text_a"]     for x in batch]
        texts_b_pos = [x["text_b_pos"] for x in batch]
        n_negs      = len(batch[0]["text_b_negs"])
        texts_b_negs = [[x["text_b_negs"][k] for x in batch] for k in range(n_negs)]

        enc_pos, b_mask_pos = tokenize_ab_pairs(texts_a, texts_b_pos, tokenizer)

        enc_negs   = []
        b_masks_neg = []
        for k in range(n_negs):
            enc_k, mask_k = tokenize_ab_pairs(texts_a, texts_b_negs[k], tokenizer)
            enc_negs.append(enc_k)
            b_masks_neg.append(mask_k)

        # C is only available for non-last hops (3/4-hop chains)
        has_c_list = [x["text_c"] is not None for x in batch]
        has_c      = torch.tensor(has_c_list, dtype=torch.bool)
        if any(has_c_list):
            texts_c_valid = [x["text_c"] for x in batch if x["text_c"] is not None]
            enc_c = tokenize_passages(texts_c_valid, tokenizer)
        else:
            enc_c = None

        enc_a = tokenize_passages(texts_a,     tokenizer)  # standalone A for L_orthogonal
        enc_b = tokenize_passages(texts_b_pos, tokenizer)  # standalone B for L_content

        return {
            "enc_pos":     enc_pos,
            "b_mask_pos":  b_mask_pos,
            "enc_negs":    enc_negs,
            "b_masks_neg": b_masks_neg,
            "enc_c":       enc_c,
            "has_c":       has_c,
            "enc_a":       enc_a,
            "enc_b":       enc_b,
        }

    return collate


# ── Training & Validation ──────────────────────────────────────────────────────

def _forward_batch(model, batch, device):
    """
    Run one forward pass over a collated batch.
    Returns (complement_pos, b_pad_mask_pos, complement_negs, b_pad_masks_neg,
             pool_c, pool_a, pool_b, has_c).
    pool_c contains embeddings only for examples where has_c is True.
    """
    def to(enc):
        return {k: v.to(device) for k, v in enc.items()}

    ep  = to(batch["enc_pos"])
    bmp = batch["b_mask_pos"].to(device)

    comp_pos, pad_mask_pos = model(
        ep["input_ids"], ep["attention_mask"], ep["token_type_ids"], bmp
    )

    comp_negs     = []
    pad_masks_neg = []
    for enc_k, mask_k in zip(batch["enc_negs"], batch["b_masks_neg"]):
        ek = to(enc_k)
        mk = mask_k.to(device)
        cn, pm = model(ek["input_ids"], ek["attention_mask"], ek["token_type_ids"], mk)
        comp_negs.append(cn)
        pad_masks_neg.append(pm)

    has_c = batch["has_c"].to(device)
    if batch["enc_c"] is not None and has_c.any():
        ec = to(batch["enc_c"])
        pc = model.encode_passage(ec["input_ids"], ec["attention_mask"])  # [n_c, 128]
    else:
        pc = None

    ea = to(batch["enc_a"])
    pa = model.encode_passage(ea["input_ids"], ea["attention_mask"])

    eb = to(batch["enc_b"])
    pb = model.encode_passage(eb["input_ids"], eb["attention_mask"])

    return comp_pos, pad_mask_pos, comp_negs, pad_masks_neg, pc, pa, pb, has_c


def validate(model, val_loader, criterion, device, max_steps=100) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            cp, mp, cns, mns, pc, pa, pb, hc = _forward_batch(model, batch, device)
            loss = criterion(cp, mp, cns, mns, pc, pa, pb, hc)
            total += loss.item()
            n     += 1
            if n >= max_steps:
                break
    model.train()
    return total / max(n, 1)


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(args) -> ComplementEncoder:
    print(f"[train] Device   : {DEVICE}")
    print(f"[train] BERT     : {BERT_NAME}")
    print(f"[train] proj_dim : {PROJ_DIM}  margin : {MARGIN}")
    print(f"[train] batch    : {BATCH_SIZE}  lr : {LR}  epochs : {EPOCHS}")

    from data_loader import load_musique, build_chain_quadruples

    print("[train] Loading MuSiQue train ...")
    train_corpus, train_queries = load_musique(
        split="train", max_examples=args.max_examples, cache=True, shuffle=True
    )
    print("[train] Loading MuSiQue val (500 examples for fast eval) ...")
    val_corpus, val_queries = load_musique(
        split="validation", max_examples=500, cache=True, shuffle=True
    )

    # Build chunk_id → text lookup (shared across train + val)
    id_to_text: Dict[str, str] = {
        c["chunk_id"]: c["text"] for c in train_corpus + val_corpus
    }

    print("[train] Building chain quadruples ...")
    train_quads = build_chain_quadruples(train_corpus, train_queries)
    val_quads   = build_chain_quadruples(val_corpus,   val_queries)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)
    collate   = make_collate_fn(tokenizer)

    train_ds = ChainDataset(train_quads, id_to_text)
    val_ds   = ChainDataset(val_quads,   id_to_text)

    if len(train_ds) == 0:
        raise RuntimeError(
            "No 3/4-hop training examples found. "
            "Check that MuSiQue train data includes multi-hop chains "
            "and chain_chunk_ids are populated (run data_loader.py smoke test)."
        )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    model     = ComplementEncoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )
    criterion = ChainContrastiveLoss(margin=MARGIN)

    best_val  = float("inf")
    step      = 0
    running   = 0.0

    print(f"[train] {len(train_ds):,} 3/4-hop train examples | {len(val_ds):,} val examples")
    print(f"[train] {len(train_loader):,} steps/epoch × {EPOCHS} epochs = {total_steps:,} total steps")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            comp_pos, pad_mask_pos, comp_negs, pad_masks_neg, pool_c, pool_a, pool_b, has_c = \
                _forward_batch(model, batch, DEVICE)

            loss = criterion(comp_pos, pad_mask_pos, comp_negs, pad_masks_neg,
                             pool_c, pool_a, pool_b, has_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            step    += 1

            if step % EVAL_EVERY == 0:
                avg_train = running / EVAL_EVERY
                val_loss  = validate(model, val_loader, criterion, DEVICE)
                running   = 0.0
                print(
                    f"  step {step:>5} | train {avg_train:.4f} | val {val_loss:.4f}"
                    f" | lr {scheduler.get_last_lr()[0]:.2e}"
                )
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), MODEL_DIR / "model1_complement.pt")
                    print(f"  *** best val={val_loss:.4f} → checkpoint saved ***")

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.0f}s")

    # Final validation pass
    val_loss = validate(model, val_loader, criterion, DEVICE, max_steps=999)
    print(f"[train] Final val loss: {val_loss:.4f}")
    if val_loss < best_val:
        torch.save(model.state_dict(), MODEL_DIR / "model1_complement.pt")
        print("  *** best val checkpoint updated ***")

    torch.save(model.state_dict(), MODEL_DIR / "model1_complement_final.pt")
    print(f"[train] Saved final → {MODEL_DIR}/model1_complement_final.pt")
    return model


# ── Evaluation helper ──────────────────────────────────────────────────────────

def eval_only(args):
    """Quick sanity check: load checkpoint and print val loss."""
    from data_loader import load_musique, build_chain_quadruples

    ckpt = MODEL_DIR / "model1_complement.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt} — run training first.")

    val_corpus, val_queries = load_musique(
        split="validation", max_examples=500, cache=True
    )
    id_to_text = {c["chunk_id"]: c["text"] for c in val_corpus}
    val_quads  = build_chain_quadruples(val_corpus, val_queries)

    tokenizer  = BertTokenizerFast.from_pretrained(BERT_NAME)
    collate    = make_collate_fn(tokenizer)
    val_ds     = ChainDataset(val_quads, id_to_text)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate, num_workers=0)

    model = ComplementEncoder().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"[eval] Loaded checkpoint from {ckpt}")

    criterion = ChainContrastiveLoss()
    val_loss  = validate(model, val_loader, criterion, DEVICE, max_steps=999)
    print(f"[eval] Val loss: {val_loss:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ComplementEncoder (Model 1)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap MuSiQue training examples (None = all ~17.5K)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate saved checkpoint on val set")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
