"""
kaggle_train.py -- Run delta-system on NewsEdits via Kaggle.

Setup in Kaggle notebook:
    !pip install transformers scikit-learn -q
    !git clone https://github.com/haaaaaaayrewugrhyw/multihop-retrieval.git repo
    %cd repo
    !python delta_system/kaggle_train.py

NewsEdits dataset (NAACL 2022):
    HuggingFace: "wiki_edits" or loaded from Kaggle dataset
    Filter: only keep additions/expansions where after_text > before_text
    A = before_text (what was known)
    B = after_text  (what is new)

Target: DELTA_PPL > 2 and SPECIFICITY > 2 on 1000 held-out pairs
        (proves the system generalizes beyond the 500-example overfit regime)
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

# ── Add delta_system to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model  import DeltaSystem
from losses import recon_loss, sparsity_loss, specificity_loss
from eval   import evaluate

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN   = 8000    # training pairs
N_EVAL    = 1000    # held-out eval pairs (never seen during training)
STEPS     = 2000
BS        = 16      # Kaggle T4 can handle larger batch
LR        = 1e-4
LAM_S     = 1.0
LAM_SPEC  = 1.0
MARGIN    = 2.0
LOG_EVERY = 200


# ── NewsEdits data loader ─────────────────────────────────────────────────────
def load_newsedits_pairs(n_train=8000, n_eval=1000):
    """
    Load (A=before, B=after) pairs from NewsEdits via HuggingFace datasets.
    Filters to additions only (after_text longer than before_text).
    Falls back to MuSiQue if NewsEdits unavailable.
    """
    total = n_train + n_eval

    try:
        from datasets import load_dataset
        print("Loading NewsEdits from HuggingFace...")
        # Try wiki_edits dataset (closest public version of NewsEdits)
        ds = load_dataset("wiki_edits", split="train", streaming=True)
        pairs = []
        for ex in ds:
            before = (ex.get("before_sentence") or ex.get("before") or "").strip()
            after  = (ex.get("after_sentence")  or ex.get("after")  or "").strip()
            if not before or not after:
                continue
            if len(after) <= len(before):
                continue                # only additions
            novel = after[len(before):].strip()
            if len(novel) < 20:
                continue                # too short to be meaningful
            pairs.append({"A": before, "B": after, "novel": novel})
            if len(pairs) >= total:
                break
        if len(pairs) >= total:
            print(f"Loaded {len(pairs)} pairs from NewsEdits")
            return pairs[:n_train], pairs[n_train:total]

    except Exception as e:
        print(f"NewsEdits load failed ({e}), falling back to MuSiQue...")

    # Fallback: MuSiQue (if running locally or on Kaggle with data uploaded)
    musique_path = ROOT.parent / "retrieval" / "data" / "musique" / "musique_ans_v1.0_train.jsonl"
    if musique_path.exists():
        import json
        pairs = []
        with open(musique_path, encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line)
                decomp = ex.get("question_decomposition", [])
                if len(decomp) != 2:
                    continue
                paras = {p["idx"]: p["paragraph_text"].strip() for p in ex["paragraphs"]}
                idx1  = decomp[0].get("paragraph_support_idx")
                idx2  = decomp[1].get("paragraph_support_idx")
                if idx1 is None or idx2 is None:
                    continue
                A     = paras.get(idx1, "").strip()
                novel = paras.get(idx2, "").strip()
                if A and novel:
                    pairs.append({"A": A, "B": A + " " + novel, "novel": novel})
                if len(pairs) >= total:
                    break
        print(f"Loaded {len(pairs)} MuSiQue pairs (fallback)")
        return pairs[:n_train], pairs[n_train:total]

    raise RuntimeError("No data source available. Upload MuSiQue or enable NewsEdits.")


# ── Dataset / collate ─────────────────────────────────────────────────────────
class PairDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.pairs[i]["A"], self.pairs[i]["B"]


def make_collate(tok):
    def collate(batch):
        A_texts = [x[0] for x in batch]
        B_texts = [x[1] for x in batch]
        eA = tok(A_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok(B_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        return eA["input_ids"], eA["attention_mask"], eB["input_ids"], eB["attention_mask"]
    return collate


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, train_pairs, tok):
    ds = PairDS(train_pairs)
    dl = DataLoader(ds, batch_size=BS, shuffle=True,
                    collate_fn=make_collate(tok), num_workers=2, pin_memory=True)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    model.train()
    step = 0

    while step < STEPS:
        for batch in dl:
            if step >= STEPS:
                break
            A_ids, A_mask, B_ids, B_mask = [t.to(DEVICE) for t in batch]
            b = A_ids.size(0)

            logits, delta, delta_0, H_A, alpha = model(A_ids, A_mask, B_ids, B_mask)
            L_r   = recon_loss(logits, B_ids, B_mask)
            L_s   = sparsity_loss(delta, B_mask)
            L_spec = torch.tensor(0.0, device=DEVICE)
            if b > 1:
                idx_shift    = list(range(1, b)) + [0]
                logits_wrong = model.reconstruct(
                    H_A, A_mask, delta[idx_shift], delta_0[idx_shift], B_ids, B_mask
                )
                L_spec = specificity_loss(logits, logits_wrong, B_ids, B_mask,
                                          margin=MARGIN)

            loss = L_r + LAM_S * L_s + LAM_SPEC * L_spec
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            step += 1

            if step % LOG_EVERY == 0 or step == 1:
                ppl = math.exp(min(L_r.item(), 20))
                print(f"  step {step:4d}/{STEPS} | ppl={ppl:.1f} | L_spec={L_spec.item():.4f}")

    # Save checkpoint
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    trainable = {k: v for k, v in model.state_dict().items()
                 if not k.startswith("bert.")}
    torch.save(trainable, ckpt_dir / "kaggle_model.pt")
    print(f"Saved checkpoint to {ckpt_dir / 'kaggle_model.pt'}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {N_TRAIN} train / {N_EVAL} held-out | {STEPS} steps | bs={BS}")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("\nLoading data...")
    train_pairs, eval_pairs = load_newsedits_pairs(N_TRAIN, N_EVAL)
    print(f"Train: {len(train_pairs)} | Eval (held-out): {len(eval_pairs)}")

    model = DeltaSystem().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.1f}M\n")

    print("Training...")
    train(model, train_pairs, tok)

    print("\n--- Held-out evaluation (unseen pairs) ---")
    results = evaluate(model, eval_pairs, tok)

    print("\n--- In-sample check (200 training pairs) ---")
    evaluate(model, train_pairs[:200], tok)

    return results


if __name__ == "__main__":
    main()
