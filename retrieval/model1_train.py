"""
Model 1 — ComplementEncoder

Architecture:
    Joint BERT encoding of [CLS] A [SEP] B [SEP].
    BERT self-attention attends across A and B at every layer, so B token
    representations naturally encode "B's content conditioned on A."
    B-side token matrix is extracted, projected to 128-dim, mean-pooled and
    L2-normalised → complement(A,B).

What complement(A,B) represents:
    "The additional information you gain by going from passage A to passage B."
    A semantic delta vector, pre-computable offline.

Loss — DPR-style biencoder (query-aligned hop scoring):
    For each directed hop pair (Q, A → B_pos, A → B_neg_1..k):
        dot(query_vec(Q), complement(A, B_pos))
            > dot(query_vec(Q), complement(A, B_neg_k)) + margin

    Two components:
      L_inbatch: InfoNCE over in-batch positives — q_i retrieves comp_pos_i
                 among all (B-1) other complements in the batch.
                 Temperature=0.07 (DPR standard). Up to 31 negatives at batch=32.
      L_hard:    Margin ranking against 3 in-example distractors per hop.

    Uses ALL 26,675 hop pairs (not just the 6,737 with a known C-anchor).
    Query text Q provides direct training signal for what the complement should score.

Usage:
    python model1_train.py --max_examples 300    # smoke test
    python model1_train.py                        # full training on Kaggle T4
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
PROJ_DIM    = 128
DROPOUT     = 0.1
MARGIN      = 0.2
BATCH_SIZE  = 32         # gradient checkpointing keeps this within T4 memory
LR          = 2e-5
EPOCHS      = 3
EVAL_EVERY  = 200
MAX_LEN_AB  = 256        # [CLS] A [SEP] B [SEP]
MAX_LEN_Q   = 128        # standalone query
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
# L_inbatch: in-batch InfoNCE — q_i retrieves its own complement among all batch comps
# L_hard:    margin ranking against 3 in-example hard negatives
TEMPERATURE = 0.07   # DPR standard; sharper than 0.1 but not overfit-prone at this scale
LAMBDA_HARD = 0.5    # weight of hard-negative margin loss


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
        Encode a standalone passage or query → mean-pooled L2-normalised [B, PROJ_DIM].
        Used for: query encoding (Q), passage encoding for graph/FAISS index.
        """
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                    # [B, seq_len, 768]
        proj   = F.normalize(self.proj(hidden), dim=-1)  # [B, seq_len, PROJ_DIM]
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

class HopContrastiveLoss(nn.Module):
    """
    DPR-style biencoder loss for complement(A,B) ↔ query_vec(Q) alignment.

    L_inbatch (InfoNCE):
        q_i should retrieve complement(A_i, B_pos_i) among all complements in the batch.
        With batch=32 this gives up to 31 in-batch negatives per anchor.

    L_hard (margin ranking):
        dot(q, complement(A, B_pos)) > dot(q, complement(A, B_neg_k)) + margin
        for each of the 3 in-example hard negative passages.
    """

    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        q_vecs:    torch.Tensor,        # [B, 128]  query vectors
        comp_pos:  torch.Tensor,        # [B, 128]  complement(A, B_pos)
        comp_negs: List[torch.Tensor],  # each [B, 128]  complement(A, B_neg_k)
    ) -> torch.Tensor:
        B   = q_vecs.shape[0]
        dev = q_vecs.device

        # L_inbatch — InfoNCE: q_i should retrieve comp_pos_i
        logits = (q_vecs @ comp_pos.T) / TEMPERATURE    # [B, B]
        labels = torch.arange(B, device=dev)
        l_inbatch = F.cross_entropy(logits, labels)

        # L_hard — margin ranking against in-example hard negatives
        score_pos = (q_vecs * comp_pos).sum(-1)         # [B]
        l_hard    = torch.zeros(1, device=dev)
        for comp_neg in comp_negs:
            score_neg = (q_vecs * comp_neg).sum(-1)
            l_hard    = l_hard + F.relu(self.margin + score_neg - score_pos).mean()
        l_hard = l_hard / max(len(comp_negs), 1)

        return l_inbatch + LAMBDA_HARD * l_hard


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
    max_length: int = MAX_LEN_Q,
) -> Dict[str, torch.Tensor]:
    """Tokenise standalone passages or queries."""
    return tokenizer(
        text=texts,
        max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChainDataset(Dataset):
    """
    Wraps chain quadruples for Model 1 training.
    Uses ALL hop pairs (including 2-hop final hops) — every pair has a query Q.
    """

    def __init__(self, quadruples: List[Dict], id_to_text: Dict[str, str]):
        self.data       = quadruples
        self.id_to_text = id_to_text
        print(f"[ChainDataset] {len(self.data):,} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        q = self.data[idx]
        return {
            "text_q":     q["query_text"],
            "text_a":     self.id_to_text[q["chunk_a_id"]],
            "text_b_pos": self.id_to_text[q["chunk_b_pos_id"]],
            "text_b_negs":[self.id_to_text[nid] for nid in q["chunk_b_neg_ids"]],
        }


def make_collate_fn(tokenizer: BertTokenizerFast):
    """Return a collate_fn that tokenises a batch of ChainDataset items."""

    def collate(batch: List[Dict]) -> Dict[str, object]:
        texts_q      = [x["text_q"]     for x in batch]
        texts_a      = [x["text_a"]     for x in batch]
        texts_b_pos  = [x["text_b_pos"] for x in batch]
        n_negs       = len(batch[0]["text_b_negs"])
        texts_b_negs = [[x["text_b_negs"][k] for x in batch] for k in range(n_negs)]

        # Query: standalone encoding
        enc_q = tokenize_passages(texts_q, tokenizer, max_length=MAX_LEN_Q)

        # Complement pairs
        enc_pos, b_mask_pos = tokenize_ab_pairs(texts_a, texts_b_pos, tokenizer)

        enc_negs    = []
        b_masks_neg = []
        for k in range(n_negs):
            enc_k, mask_k = tokenize_ab_pairs(texts_a, texts_b_negs[k], tokenizer)
            enc_negs.append(enc_k)
            b_masks_neg.append(mask_k)

        return {
            "enc_q":       enc_q,
            "enc_pos":     enc_pos,
            "b_mask_pos":  b_mask_pos,
            "enc_negs":    enc_negs,
            "b_masks_neg": b_masks_neg,
        }

    return collate


# ── Training & Validation ──────────────────────────────────────────────────────

def _forward_batch(model, batch, device):
    """
    Five forward passes per batch:
      1. encode_passage(Q)          → q_vecs     [B, 128]
      2. complement(A, B_pos)       → comp_pos   [B, 128]
      3-5. complement(A, B_neg_k)   → comp_negs  list of [B, 128]
    """
    def to(enc):
        return {k: v.to(device) for k, v in enc.items()}

    # Query vectors
    eq     = to(batch["enc_q"])
    q_vecs = model.encode_passage(eq["input_ids"], eq["attention_mask"])  # [B, 128]

    # Positive complement
    ep  = to(batch["enc_pos"])
    bmp = batch["b_mask_pos"].to(device)
    comp_pos_tok, pad_mask_pos = model(
        ep["input_ids"], ep["attention_mask"], ep["token_type_ids"], bmp
    )
    comp_pos = mean_pool(comp_pos_tok, pad_mask_pos)  # [B, 128]

    # Negative complements
    comp_negs = []
    for enc_k, mask_k in zip(batch["enc_negs"], batch["b_masks_neg"]):
        ek = to(enc_k)
        mk = mask_k.to(device)
        cn, pm = model(ek["input_ids"], ek["attention_mask"], ek["token_type_ids"], mk)
        comp_negs.append(mean_pool(cn, pm))  # [B, 128]

    return q_vecs, comp_pos, comp_negs


def validate(model, val_loader, device, max_steps=100) -> float:
    """
    Returns ranking accuracy: fraction of examples where
    dot(q, comp_pos) > dot(q, comp_neg_k) for ALL k.
    A well-trained model should reach >65% on unseen val hops.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            q_vecs, comp_pos, comp_negs = _forward_batch(model, batch, device)
            score_pos  = (q_vecs * comp_pos).sum(-1)  # [B]
            all_correct = torch.ones(score_pos.shape[0], dtype=torch.bool, device=device)
            for comp_neg in comp_negs:
                score_neg   = (q_vecs * comp_neg).sum(-1)
                all_correct = all_correct & (score_pos > score_neg)
            correct += all_correct.sum().item()
            total   += score_pos.shape[0]
            if i + 1 >= max_steps:
                break
    model.train()
    return correct / max(total, 1)


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(args) -> ComplementEncoder:
    print(f"[train] Device   : {DEVICE}")
    print(f"[train] BERT     : {BERT_NAME}")
    print(f"[train] proj_dim : {PROJ_DIM}  margin : {MARGIN}")
    print(f"[train] batch    : {BATCH_SIZE}  lr : {LR}  epochs : {EPOCHS}")
    print(f"[train] loss     : in-batch InfoNCE (T={TEMPERATURE}) + hard-neg margin (λ={LAMBDA_HARD})")

    from data_loader import load_musique, build_chain_quadruples

    print("[train] Loading MuSiQue train ...")
    train_corpus, train_queries = load_musique(
        split="train", max_examples=args.max_examples, cache=True, shuffle=True
    )
    print("[train] Loading MuSiQue val (500 examples for fast eval) ...")
    val_corpus, val_queries = load_musique(
        split="validation", max_examples=500, cache=True, shuffle=True
    )

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
        raise RuntimeError("No training examples found. Check MuSiQue data.")

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
    model.bert.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )
    criterion = HopContrastiveLoss(margin=MARGIN)

    best_val  = 0.0   # maximise accuracy
    step      = 0
    running   = 0.0

    print(f"[train] {len(train_ds):,} hop pairs | {len(val_ds):,} val examples")
    print(f"[train] {len(train_loader):,} steps/epoch × {EPOCHS} epochs = {total_steps:,} total steps")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            q_vecs, comp_pos, comp_negs = _forward_batch(model, batch, DEVICE)
            loss = criterion(q_vecs, comp_pos, comp_negs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            step    += 1

            if step % EVAL_EVERY == 0:
                avg_train = running / EVAL_EVERY
                val_acc   = validate(model, val_loader, DEVICE)
                running   = 0.0
                print(
                    f"  step {step:>5} | train_loss {avg_train:.4f} | val_acc {val_acc:.4f}"
                    f" | lr {scheduler.get_last_lr()[0]:.2e}"
                )
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save(model.state_dict(), MODEL_DIR / "model1_complement.pt")
                    print(f"  *** best val_acc={val_acc:.4f} → checkpoint saved ***")

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.0f}s")

    # Final validation pass
    val_acc = validate(model, val_loader, DEVICE, max_steps=999)
    print(f"[train] Final val_acc: {val_acc:.4f}")
    if val_acc > best_val:
        torch.save(model.state_dict(), MODEL_DIR / "model1_complement.pt")
        print("  *** best val checkpoint updated ***")

    torch.save(model.state_dict(), MODEL_DIR / "model1_complement_final.pt")
    print(f"[train] Saved final → {MODEL_DIR}/model1_complement_final.pt")
    return model


# ── Evaluation helper ──────────────────────────────────────────────────────────

def eval_only(args):
    """Load checkpoint and print val ranking accuracy."""
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

    val_acc = validate(model, val_loader, DEVICE, max_steps=999)
    print(f"[eval] Val ranking accuracy: {val_acc:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ComplementEncoder (Model 1)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap MuSiQue training examples (None = all)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate saved checkpoint on val set")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
