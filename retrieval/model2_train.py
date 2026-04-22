"""
Model 2 — ColBERTScorer

Architecture:
    Query encoder  : BERT → Q tokens [k × 128], L2-normalised
    Passage scorer : ColBERT MaxSim against complement_tokens from Model 1

    score(Q, complement(A, B)) = Σ_i  max_j  cos( Q_token_i, complement_token_j )

Why MaxSim instead of pooled dot-product:
    Each query token independently finds its best match in B's novel content.
    "Who founded" finds founder tokens; "born in" finds birthplace tokens.
    The score sums how many aspects of Q are covered by B's novel content.
    Query is introduced FRESH at scoring time — no pre-computation information loss.
    I(Q; complement_tokens) = 0 (correct: Model 1 is query-agnostic).
    I(Q; score) is high (correct: Model 2 is fully query-conditional).

Initialisation:
    Query encoder initialised from colbert-ir/colbertv2.0 (pre-trained on MS MARCO).
    Fine-tuned on MuSiQue quintuples (all hop counts: 2, 3, 4-hop).

Loss — Margin Ranking:
    For each (Q, A, B_pos, B_neg_1..3):
        score_pos = MaxSim(Q_tokens, complement(A, B_pos))
        score_neg = MaxSim(Q_tokens, complement(A, B_neg_i))
        loss      = mean( relu( margin + score_neg - score_pos ) )

    Model 1 is FROZEN during Model 2 training.
    This preserves the query-agnostic property of Model 1.

Training data: MuSiQue train, ALL hop counts (uses build_scoring_quintuples)
    ~26K directed (A→B) pairs × 4 = ~104K training instances.

Usage:
    python model2_train.py --max_examples 300    # smoke test
    python model2_train.py                        # full training (Colab recommended)
    python model2_train.py --eval_only            # evaluate saved checkpoint

Output:
    models/model2_scorer.pt    trained ColBERTScorer (query encoder) weights
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# ColBERTv2 pre-trained on MS MARCO — used to initialise query encoder
COLBERT_CKPT = "colbert-ir/colbertv2.0"
BERT_FALLBACK = "bert-base-uncased"     # used if ColBERT checkpoint unavailable

PROJ_DIM    = 128        # must match Model 1's PROJ_DIM
DROPOUT     = 0.1
MARGIN      = 0.2
BATCH_SIZE  = 8          # quintuples per step
LR          = 2e-5
EPOCHS      = 3
EVAL_EVERY  = 200
MAX_LEN_Q   = 64         # max query tokens
MAX_LEN_AB  = 256        # max [CLS] A [SEP] B [SEP] tokens (for complement encoder)
MAX_LEN_C   = 128        # not used in Model 2 but kept for import compat
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Query Encoder ──────────────────────────────────────────────────────────────

class ColBERTQueryEncoder(nn.Module):
    """
    Encodes a query → token matrix [k × PROJ_DIM], L2-normalised.

    Initialised from ColBERTv2 (MS MARCO pre-trained) if available,
    otherwise falls back to bert-base-uncased.
    """

    def __init__(self, bert_name: str = COLBERT_CKPT, proj_dim: int = PROJ_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        try:
            self.bert = BertModel.from_pretrained(bert_name)
            print(f"[ColBERTQueryEncoder] Loaded from {bert_name}")
        except Exception as e:
            print(f"[ColBERTQueryEncoder] Could not load {bert_name}: {e}")
            print(f"[ColBERTQueryEncoder] Falling back to {BERT_FALLBACK}")
            self.bert = BertModel.from_pretrained(BERT_FALLBACK)
        self.proj    = nn.Sequential(nn.Linear(self.bert.config.hidden_size, proj_dim),
                                     nn.Dropout(dropout))

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, q_len]
        attention_mask: torch.Tensor,   # [B, q_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_tokens  [B, q_len, PROJ_DIM]  L2-normalised query token embeddings
            q_mask    [B, q_len]             True for non-padding tokens
        """
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                     # [B, q_len, 768]
        proj   = F.normalize(self.proj(hidden), dim=-1)   # [B, q_len, PROJ_DIM]
        q_mask = attention_mask.bool()
        return proj, q_mask


# ── ColBERT MaxSim ─────────────────────────────────────────────────────────────

def colbert_maxsim(
    q_tokens:  torch.Tensor,   # [B, q_len, dim]
    q_mask:    torch.Tensor,   # [B, q_len] bool
    p_tokens:  torch.Tensor,   # [B, p_len, dim]
    p_mask:    torch.Tensor,   # [B, p_len] bool
) -> torch.Tensor:
    """
    ColBERT MaxSim scoring:
        For each query token, find the max cosine similarity with any passage token.
        Sum over all query tokens (masked).

    Returns score [B].
    """
    # [B, q_len, p_len]  (tokens already L2-normalised → cosine = dot product)
    sim = torch.bmm(q_tokens, p_tokens.transpose(1, 2))

    # Mask padding in passage dimension: set to -inf so they never win max
    p_pad = ~p_mask.unsqueeze(1)                         # [B, 1, p_len]
    sim   = sim.masked_fill(p_pad, -1e9)

    max_sim = sim.max(dim=-1).values                     # [B, q_len]

    # Mask padding in query dimension
    max_sim = max_sim.masked_fill(~q_mask, 0.0)

    return max_sim.sum(dim=-1)                           # [B]


# ── Scoring Dataset ────────────────────────────────────────────────────────────

class ScoringDataset(Dataset):
    """Wraps scoring quintuples (all hop counts)."""

    def __init__(self, quintuples: List[Dict], id_to_text: Dict[str, str]):
        self.data       = quintuples
        self.id_to_text = id_to_text

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        q = self.data[idx]
        return {
            "question":   q["question"],
            "text_a":     self.id_to_text[q["chunk_a_id"]],
            "text_b_pos": self.id_to_text[q["chunk_b_pos_id"]],
            "text_b_negs":[self.id_to_text[nid] for nid in q["chunk_b_neg_ids"]],
        }


def make_collate_fn(
    q_tokenizer: BertTokenizerFast,
    ab_tokenizer: BertTokenizerFast,
):
    """Return collate_fn that tokenises questions and (A, B) pairs."""

    sep_id = ab_tokenizer.sep_token_id

    def _tokenize_ab(texts_a, texts_b):
        enc = ab_tokenizer(
            text=texts_a, text_pair=texts_b,
            max_length=MAX_LEN_AB, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        b_mask = (enc["token_type_ids"] == 1) & (enc["input_ids"] != sep_id)
        return enc, b_mask

    def collate(batch: List[Dict]) -> Dict:
        questions   = [x["question"]   for x in batch]
        texts_a     = [x["text_a"]     for x in batch]
        texts_b_pos = [x["text_b_pos"] for x in batch]
        n_negs      = len(batch[0]["text_b_negs"])
        texts_b_negs = [[x["text_b_negs"][k] for x in batch] for k in range(n_negs)]

        enc_q = q_tokenizer(
            text=questions,
            max_length=MAX_LEN_Q, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_pos, b_mask_pos = _tokenize_ab(texts_a, texts_b_pos)

        enc_negs, b_masks_neg = [], []
        for k in range(n_negs):
            en, bm = _tokenize_ab(texts_a, texts_b_negs[k])
            enc_negs.append(en)
            b_masks_neg.append(bm)

        return {
            "enc_q":       enc_q,
            "enc_pos":     enc_pos,
            "b_mask_pos":  b_mask_pos,
            "enc_negs":    enc_negs,
            "b_masks_neg": b_masks_neg,
        }

    return collate


# ── Forward helpers ────────────────────────────────────────────────────────────

def _complement(comp_encoder, enc, b_mask, device):
    """Run complement encoder on one (A, B) encoding. Returns (tokens, pad_mask)."""
    from model1_train import _extract_masked_tokens
    enc_d  = {k: v.to(device) for k, v in enc.items()}
    bm     = b_mask.to(device)
    out    = comp_encoder.bert(
        input_ids=enc_d["input_ids"],
        attention_mask=enc_d["attention_mask"],
        token_type_ids=enc_d["token_type_ids"],
    )
    hidden = out.last_hidden_state
    proj   = F.normalize(comp_encoder.proj(hidden), dim=-1)
    return _extract_masked_tokens(proj, bm)


def _forward_batch(query_enc, comp_enc, batch, device):
    """
    Run one batch through both encoders.
    Returns (q_tokens, q_mask, comp_pos, pad_pos, comp_negs, pad_negs).
    """
    enc_q = {k: v.to(device) for k, v in batch["enc_q"].items()}
    q_tokens, q_mask = query_enc(enc_q["input_ids"], enc_q["attention_mask"])

    comp_pos, pad_pos = _complement(comp_enc, batch["enc_pos"], batch["b_mask_pos"], device)

    comp_negs, pad_negs = [], []
    for enc_k, mask_k in zip(batch["enc_negs"], batch["b_masks_neg"]):
        cn, pm = _complement(comp_enc, enc_k, mask_k, device)
        comp_negs.append(cn)
        pad_negs.append(pm)

    return q_tokens, q_mask, comp_pos, pad_pos, comp_negs, pad_negs


# ── Loss ───────────────────────────────────────────────────────────────────────

class MarginRankingLoss(nn.Module):
    """
    score(Q, B_pos) > score(Q, B_neg_i) + margin  for all i.
    """
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_pos:  torch.Tensor,   # [B]
        score_negs: List[torch.Tensor],  # each [B]
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=score_pos.device)
        for s_neg in score_negs:
            loss = loss + F.relu(self.margin + s_neg - score_pos).mean()
        return loss / max(len(score_negs), 1)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(query_enc, comp_enc, val_loader, criterion, device, max_steps=100) -> Tuple[float, float]:
    """
    Returns (val_loss, accuracy).
    Accuracy = fraction of examples where score_pos > ALL score_negs.
    """
    query_enc.eval(); comp_enc.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            q_tok, q_mask, cp, pp, cns, pns = _forward_batch(query_enc, comp_enc, batch, device)

            score_pos  = colbert_maxsim(q_tok, q_mask, cp, pp)
            score_negs = [colbert_maxsim(q_tok, q_mask, cn, pn) for cn, pn in zip(cns, pns)]

            loss = criterion(score_pos, score_negs)
            total_loss += loss.item()

            # Accuracy: pos beats all negs
            neg_stack = torch.stack(score_negs, dim=-1)   # [B, n_negs]
            correct  += (score_pos.unsqueeze(-1) > neg_stack).all(dim=-1).float().sum().item()
            n        += score_pos.shape[0]

            if n // BATCH_SIZE >= max_steps:
                break

    query_enc.train(); comp_enc.train()
    steps = n // BATCH_SIZE or 1
    return total_loss / steps, correct / max(n, 1)


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(args):
    print(f"[train] Device : {DEVICE}")
    print(f"[train] margin : {MARGIN}  batch : {BATCH_SIZE}  lr : {LR}  epochs : {EPOCHS}")

    from data_loader import load_musique, build_scoring_quintuples
    from model1_train import ComplementEncoder

    # ── Load data ──
    print("[train] Loading MuSiQue train ...")
    train_corpus, train_queries = load_musique(
        split="train", max_examples=args.max_examples, cache=True, shuffle=True
    )
    print("[train] Loading MuSiQue val (500 examples) ...")
    val_corpus, val_queries = load_musique(
        split="validation", max_examples=500, cache=True, shuffle=True
    )

    id_to_text = {c["chunk_id"]: c["text"] for c in train_corpus + val_corpus}

    print("[train] Building scoring quintuples ...")
    train_quints = build_scoring_quintuples(train_corpus, train_queries)
    val_quints   = build_scoring_quintuples(val_corpus,   val_queries)
    print(f"[train] Train quintuples: {len(train_quints):,}  Val: {len(val_quints):,}")

    if not train_quints:
        raise RuntimeError("No training quintuples — check MuSiQue data.")

    # ── Tokenisers ──
    # Both encoders use the same BERT tokeniser family
    q_tokenizer  = BertTokenizerFast.from_pretrained(BERT_FALLBACK)
    ab_tokenizer = BertTokenizerFast.from_pretrained(BERT_FALLBACK)
    collate      = make_collate_fn(q_tokenizer, ab_tokenizer)

    train_loader = DataLoader(
        ScoringDataset(train_quints, id_to_text),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        ScoringDataset(val_quints, id_to_text),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # ── Models ──
    query_enc = ColBERTQueryEncoder().to(DEVICE)

    # Load frozen Model 1 complement encoder
    comp_enc  = ComplementEncoder().to(DEVICE)
    m1_ckpt   = MODEL_DIR / "model1_complement.pt"
    if m1_ckpt.exists():
        comp_enc.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
        print(f"[train] Loaded Model 1 from {m1_ckpt}")
    else:
        print(f"[train] WARNING: Model 1 checkpoint not found at {m1_ckpt}")
        print(f"[train]          Using untrained complement encoder (train Model 1 first).")
    comp_enc.eval()
    for p in comp_enc.parameters():
        p.requires_grad_(False)

    # ── Optimiser — only trains query encoder ──
    optimizer  = torch.optim.AdamW(query_enc.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = len(train_loader) * EPOCHS
    scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )
    criterion  = MarginRankingLoss(margin=MARGIN)

    best_val   = float("inf")
    step       = 0
    running    = 0.0

    print(f"[train] {len(train_loader):,} steps/epoch × {EPOCHS} epochs = {total_steps:,} total")

    for epoch in range(1, EPOCHS + 1):
        query_enc.train()
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            q_tok, q_mask, cp, pp, cns, pns = _forward_batch(query_enc, comp_enc, batch, DEVICE)

            score_pos  = colbert_maxsim(q_tok, q_mask, cp, pp)
            score_negs = [colbert_maxsim(q_tok, q_mask, cn, pn) for cn, pn in zip(cns, pns)]

            loss = criterion(score_pos, score_negs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_enc.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            step    += 1

            if step % EVAL_EVERY == 0:
                avg_train        = running / EVAL_EVERY
                val_loss, val_acc = validate(query_enc, comp_enc, val_loader, criterion, DEVICE)
                running          = 0.0
                print(
                    f"  step {step:>5} | train {avg_train:.4f} | val {val_loss:.4f}"
                    f" | acc {val_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(query_enc.state_dict(), MODEL_DIR / "model2_scorer.pt")
                    print(f"  *** best val={val_loss:.4f} → checkpoint saved ***")

        print(f"Epoch {epoch} done in {time.time()-t0:.0f}s")

    val_loss, val_acc = validate(query_enc, comp_enc, val_loader, criterion, DEVICE, max_steps=999)
    print(f"[train] Final val loss: {val_loss:.4f}  acc: {val_acc:.3f}")
    if val_loss < best_val:
        torch.save(query_enc.state_dict(), MODEL_DIR / "model2_scorer.pt")
    torch.save(query_enc.state_dict(), MODEL_DIR / "model2_scorer_final.pt")
    print(f"[train] Saved → {MODEL_DIR}/model2_scorer_final.pt")


# ── Eval only ──────────────────────────────────────────────────────────────────

def eval_only(args):
    from data_loader import load_musique, build_scoring_quintuples
    from model1_train import ComplementEncoder

    ckpt = MODEL_DIR / "model2_scorer.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt} — run training first.")

    val_corpus, val_queries = load_musique(
        split="validation", max_examples=500, cache=True, shuffle=True
    )
    id_to_text  = {c["chunk_id"]: c["text"] for c in val_corpus}
    val_quints  = build_scoring_quintuples(val_corpus, val_queries)

    q_tokenizer  = BertTokenizerFast.from_pretrained(BERT_FALLBACK)
    ab_tokenizer = BertTokenizerFast.from_pretrained(BERT_FALLBACK)
    collate      = make_collate_fn(q_tokenizer, ab_tokenizer)
    val_loader   = DataLoader(
        ScoringDataset(val_quints, id_to_text),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    query_enc = ColBERTQueryEncoder().to(DEVICE)
    query_enc.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"[eval] Loaded checkpoint from {ckpt}")

    comp_enc = ComplementEncoder().to(DEVICE)
    m1_ckpt  = MODEL_DIR / "model1_complement.pt"
    if m1_ckpt.exists():
        comp_enc.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
    comp_enc.eval()
    for p in comp_enc.parameters():
        p.requires_grad_(False)

    criterion         = MarginRankingLoss()
    val_loss, val_acc = validate(query_enc, comp_enc, val_loader, criterion, DEVICE, max_steps=999)
    print(f"[eval] Val loss: {val_loss:.4f}  Accuracy: {val_acc:.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ColBERTScorer (Model 2)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap MuSiQue training examples (None = all ~19.9K)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate saved checkpoint")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
