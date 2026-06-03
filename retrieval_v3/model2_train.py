"""
Model 2 — QueryEncoder (retrieval_v2)
======================================

Architecture: BERT-base → mean-pool → proj [768→128], L2-normalised.
Field names mirror FakeEncoderModel (encoder1, proj) so weights transfer cleanly
from fakencoder_best.pt.

Training objective: align query vector with complement vectors from frozen Model 1.

    score(Q, edge A→B) = dot( QueryEncoder(Q),
                               FakeEncoderModel.extract_complement(A, B) )

    Loss = MarginRankingLoss:
        score(Q, comp_pos) > score(Q, comp_neg_i) + margin   ∀ i

Model 1 is FROZEN. Only QueryEncoder trains.
Checkpoint: model2_best.pt  (best val accuracy)

Usage:
    python model2_train.py               # full 3-epoch training
    python model2_train.py --smoke       # quick CPU smoke test (50 examples)
    python model2_train.py --max_examples 1000
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval_v2"))  # data_loader
from data_loader import build_scoring_quintuples, load_musique
from generator_train import (
    ComplementGenerator as FakeEncoderModel,   # alias -> rest of file unchanged
    D_MODEL, D_PROJ, MAX_A_LEN, MAX_B_LEN,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_Q_LEN     = 64
BATCH_SIZE    = 8
LR            = 1e-5     # was 2e-5 — lower LR fights the train 0.99 / val 0.45 overfit
N_EPOCHS      = 2        # was 3 — val_acc plateaued after epoch 1; best saved anyway
MARGIN        = 0.2
WARMUP_FRAC   = 0.06
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY  = 0.05     # was 1e-2 — stronger regularization
LOG_EVERY     = 300

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")
MODEL_DIR   = Path(__file__).parent / "models"
CACHE_DIR   = Path(__file__).parent / "cache"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ── QueryEncoder ──────────────────────────────────────────────────────────────

class QueryEncoder(nn.Module):
    """
    Encodes a query → [D_PROJ] L2-normalised.
    Field names (encoder1, proj) match FakeEncoderModel for clean weight transfer.
    """

    def __init__(self):
        super().__init__()
        self.encoder1 = BertModel.from_pretrained("bert-base-uncased")
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_PROJ),
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        out    = self.encoder1(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state                                        # [B, seq, D]
        mask_f = attn_mask.unsqueeze(-1).float()                              # [B, seq, 1]
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)    # [B, D]
        return F.normalize(self.proj(pooled), dim=-1)                         # [B, D_PROJ]

    def encode_query(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self(input_ids, attn_mask)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ScoringDataset(Dataset):
    def __init__(self, quintuples: List[Dict], id_to_text: Dict[str, str]):
        self.id_to_text = id_to_text
        self.data = [
            q for q in quintuples
            if q["chunk_a_id"]     in id_to_text
            and q["chunk_b_pos_id"] in id_to_text
            and all(nid in id_to_text for nid in q["chunk_b_neg_ids"])
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        q = self.data[idx]
        return {
            "question":    q["question"],
            "text_a":      self.id_to_text[q["chunk_a_id"]],
            "text_b_pos":  self.id_to_text[q["chunk_b_pos_id"]],
            "text_b_negs": [self.id_to_text[nid] for nid in q["chunk_b_neg_ids"]],
        }


def make_collator(tokenizer: BertTokenizerFast):
    def collate(batch: List[Dict]) -> Dict:
        questions   = [x["question"]   for x in batch]
        a_texts     = [x["text_a"]     for x in batch]
        b_pos_texts = [x["text_b_pos"] for x in batch]
        n_negs      = len(batch[0]["text_b_negs"])
        b_neg_texts = [[x["text_b_negs"][k] for x in batch] for k in range(n_negs)]

        enc_q = tokenizer(
            questions, max_length=MAX_Q_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_a = tokenizer(
            a_texts, max_length=MAX_A_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_b_pos = tokenizer(
            b_pos_texts, max_length=MAX_B_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        enc_b_negs = [
            tokenizer(
                neg_texts, max_length=MAX_B_LEN, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            for neg_texts in b_neg_texts
        ]
        return {
            "enc_q":      enc_q,
            "enc_a":      enc_a,
            "enc_b_pos":  enc_b_pos,
            "enc_b_negs": enc_b_negs,
        }
    return collate


# ── Loss ──────────────────────────────────────────────────────────────────────

class MarginRankingLoss(nn.Module):
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_pos:  torch.Tensor,
        score_negs: List[torch.Tensor],
    ) -> torch.Tensor:
        parts = [F.relu(self.margin + s_neg - score_pos).mean() for s_neg in score_negs]
        return torch.stack(parts).mean() if parts else score_pos.sum() * 0.0


# ── Complement helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_comps(
    model1: FakeEncoderModel,
    enc_a:  Dict,
    enc_b:  Dict,
    device: str,
) -> torch.Tensor:
    return model1.extract_complement(
        enc_a["input_ids"].to(device),
        enc_a["attention_mask"].to(device),
        enc_b["input_ids"].to(device),
    )   # [B, D_PROJ]


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model2:    QueryEncoder,
    model1:    FakeEncoderModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    criterion: MarginRankingLoss,
) -> Tuple[float, float]:
    model2.train()
    model1.eval()
    total_loss, correct, n_batches, n_ex = 0.0, 0, 0, 0

    for step, batch in enumerate(tqdm(loader, desc="Train M2", leave=False)):
        comp_pos  = _extract_comps(model1, batch["enc_a"], batch["enc_b_pos"],  DEVICE)
        comp_negs = [
            _extract_comps(model1, batch["enc_a"], enc_b_neg, DEVICE)
            for enc_b_neg in batch["enc_b_negs"]
        ]

        with autocast("cuda", enabled=AMP_ENABLED):
            q_vec = model2(
                batch["enc_q"]["input_ids"].to(DEVICE),
                batch["enc_q"]["attention_mask"].to(DEVICE),
            )
            score_pos  = (q_vec * comp_pos).sum(-1)
            score_negs = [(q_vec * c).sum(-1) for c in comp_negs]
            loss = criterion(score_pos, score_negs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        with torch.no_grad():
            neg_stack = torch.stack(score_negs, dim=-1)
            correct  += (score_pos.unsqueeze(-1) > neg_stack).all(-1).float().sum().item()

        total_loss += loss.item()
        n_ex       += score_pos.shape[0]
        n_batches  += 1

        if step > 0 and step % LOG_EVERY == 0:
            tqdm.write(
                f"  step {step} | loss={total_loss/n_batches:.4f} | acc={correct/max(n_ex,1):.3f}"
            )

    return total_loss / max(n_batches, 1), correct / max(n_ex, 1)


@torch.no_grad()
def validate(
    model2:    QueryEncoder,
    model1:    FakeEncoderModel,
    loader:    DataLoader,
    criterion: MarginRankingLoss,
    max_steps: int = 120,
) -> Tuple[float, float]:
    model2.eval()
    model1.eval()
    total_loss, correct, n_batches, n_ex = 0.0, 0, 0, 0

    for step, batch in enumerate(tqdm(loader, desc="Val M2", leave=False)):
        if step >= max_steps:
            break

        comp_pos  = _extract_comps(model1, batch["enc_a"], batch["enc_b_pos"],  DEVICE)
        comp_negs = [
            _extract_comps(model1, batch["enc_a"], enc_b_neg, DEVICE)
            for enc_b_neg in batch["enc_b_negs"]
        ]

        q_vec = model2(
            batch["enc_q"]["input_ids"].to(DEVICE),
            batch["enc_q"]["attention_mask"].to(DEVICE),
        )
        score_pos  = (q_vec * comp_pos).sum(-1)
        score_negs = [(q_vec * c).sum(-1) for c in comp_negs]

        neg_stack = torch.stack(score_negs, dim=-1)
        correct  += (score_pos.unsqueeze(-1) > neg_stack).all(-1).float().sum().item()
        total_loss += criterion(score_pos, score_negs).item()
        n_ex       += score_pos.shape[0]
        n_batches  += 1

    return total_loss / max(n_batches, 1), correct / max(n_ex, 1)


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
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading MuSiQue ...")
    train_corpus, train_queries = load_musique(
        split="train", max_examples=args.max_examples, cache=not args.smoke,
    )
    val_max = 10 if args.smoke else 500
    val_corpus, val_queries = load_musique(
        split="validation", max_examples=val_max, cache=not args.smoke,
    )
    id_to_text = {c["chunk_id"]: c["text"] for c in train_corpus + val_corpus}

    train_quints = build_scoring_quintuples(train_corpus, train_queries)
    val_quints   = build_scoring_quintuples(val_corpus,   val_queries)
    print(f"   Train quintuples: {len(train_quints):,} | Val: {len(val_quints):,}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collator  = make_collator(tokenizer)
    nw = 0 if args.smoke else 2

    train_loader = DataLoader(
        ScoringDataset(train_quints, id_to_text),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        ScoringDataset(val_quints, id_to_text),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    # ── Model 1 (frozen) ─────────────────────────────────────────────────────
    print("\n[2] Loading Model 1 (frozen) ...")
    m1_ckpt = MODEL_DIR / "generator_best.pt"
    if not m1_ckpt.exists():
        raise FileNotFoundError(
            f"Model 1 checkpoint not found: {m1_ckpt}\n"
            "Train Model 1 first: python generator_train.py"
        )
    model1 = FakeEncoderModel().to(DEVICE)
    model1.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE))
    model1.eval()
    for p in model1.parameters():
        p.requires_grad_(False)
    print(f"   Loaded from {m1_ckpt}")

    # ── Model 2 — init from Model 1 weights ──────────────────────────────────
    print("\n[3] Building QueryEncoder (warm-start from Model 1) ...")
    model2 = QueryEncoder().to(DEVICE)

    m1_state = model1.state_dict()
    m2_state = model2.state_dict()
    transferred = 0
    for k in list(m2_state.keys()):
        if k in m1_state and m1_state[k].shape == m2_state[k].shape:
            m2_state[k] = m1_state[k].clone()
            transferred += 1
    model2.load_state_dict(m2_state)
    print(f"   Transferred {transferred}/{len(m2_state)} tensors from Model 1")
    print(f"   QueryEncoder params: {sum(p.numel() for p in model2.parameters())/1e6:.1f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    scaler       = GradScaler("cuda", enabled=AMP_ENABLED)
    optimizer    = torch.optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = MarginRankingLoss(margin=MARGIN)

    print(f"\n[4] Training {args.epochs} epoch(s) | {len(train_loader):,} steps/epoch")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model2, model1, train_loader, optimizer, scheduler, scaler, criterion,
        )
        val_loss, val_acc = validate(model2, model1, val_loader, criterion)

        print(
            f"   Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model2.state_dict(), MODEL_DIR / "model2_best.pt")
            print(f"   -> Saved best  (val_acc={val_acc:.3f})")

    torch.save(model2.state_dict(), MODEL_DIR / "model2_final.pt")
    print(f"\n[5] Done. Best val_acc={best_val_acc:.3f}")
    print(f"   Checkpoints: {MODEL_DIR}/model2_best.pt  |  model2_final.pt")

    if args.smoke:
        print("\n[smoke] PASSED")


if __name__ == "__main__":
    main()
