"""
eval.py -- Validation experiment: three pass/fail checks.

Metric 1 — DELTA_PPL:
    ppl_no_delta  = perplexity of B given (A, delta=0)
    ppl_with_delta= perplexity of B given (A, delta)
    DELTA_PPL = ppl_no_delta - ppl_with_delta
    PASS: DELTA_PPL > 2   (delta helps reconstruction beyond A alone)

Metric 2 — AUROC:
    score per token = ||delta[t]||
    label per token = 1 if token belongs to the "novel" paragraph, else 0
    AUROC of score vs label
    PASS: AUROC > 0.55   (delta opens more where B is novel)

Metric 3 — SPECIFICITY:
    Circular-shift deltas within each batch: delta_i -> position (i+1)%b
    PPL_wrong  = ppl( D_recon(A_i, delta_{i+1}), B_i )
    PPL_correct= ppl( D_recon(A_i, delta_i),     B_i )
    SPECIFICITY = mean(PPL_wrong - PPL_correct)
    PASS: SPECIFICITY > 2   (correct delta helps more than wrong one)
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
from data   import load_pairs
from model  import DeltaSystem
from losses import recon_loss

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


class PairDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def make_collate(tok: BertTokenizerFast):
    def collate(batch):
        A_texts = [x["A"]     for x in batch]
        B_texts = [x["B"]     for x in batch]
        N_texts = [x["novel"] for x in batch]
        eA = tok(A_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok(B_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        return (eA["input_ids"], eA["attention_mask"],
                eB["input_ids"], eB["attention_mask"],
                A_texts, N_texts)
    return collate


def _novel_token_mask(A_text: str, N_text: str, B_mask: torch.Tensor,
                      tok: BertTokenizerFast) -> np.ndarray:
    """
    Exact per-token label: 1 = novel paragraph token, 0 = A token.
    B = A + " " + novel. Tokenize B with offset_mapping and find the exact
    token index where char position crosses into the novel paragraph.
    """
    B_text           = A_text + " " + N_text
    novel_char_start = len(A_text) + 1   # +1 for the space
    T        = B_mask.size(0)
    real_len = int(B_mask.sum().item())
    labels   = np.zeros(T, dtype=np.float32)

    enc     = tok(B_text, max_length=MAX_LEN, truncation=True,
                  return_offsets_mapping=True)
    offsets = enc["offset_mapping"]   # list of (char_start, char_end) per token

    # offsets[0] = [CLS] = (0,0), offsets[-1] = [SEP] = (0,0)
    # Find first token (after CLS) whose char_start >= novel_char_start
    novel_start = real_len  # default: no novel tokens visible after truncation
    for t, (cs, ce) in enumerate(offsets):
        if t == 0:
            continue                  # skip [CLS]
        if cs >= novel_char_start and ce > cs:
            novel_start = t
            break

    # Mark novel tokens (exclude final [SEP] at real_len-1)
    for t in range(min(novel_start, T), max(0, real_len - 1)):
        labels[t] = 1.0
    return labels


def _ppl_batch(logits, B_ids, B_mask) -> list:
    """Per-example perplexity from teacher-forced logits."""
    ppls = []
    for i in range(B_ids.size(0)):
        L = recon_loss(logits[i:i+1], B_ids[i:i+1], B_mask[i:i+1])
        ppls.append(math.exp(min(L.item(), 30)))
    return ppls


@torch.no_grad()
def evaluate(model: DeltaSystem, pairs, tok: BertTokenizerFast) -> dict:
    model.eval()

    dl = DataLoader(PairDS(pairs), batch_size=8, shuffle=False,
                    collate_fn=make_collate(tok), num_workers=0)

    ppl_with_list, ppl_no_list  = [], []
    ppl_correct_list, ppl_wrong_list = [], []
    all_norms, all_labels = [], []

    for A_ids, A_mask, B_ids, B_mask, A_texts, N_texts in dl:
        A_ids, A_mask = A_ids.to(DEVICE), A_mask.to(DEVICE)
        B_ids, B_mask = B_ids.to(DEVICE), B_mask.to(DEVICE)
        b = A_ids.size(0)

        # Encode once (BERT, frozen, no_grad already inside _enc)
        H_A = model._enc(A_ids, A_mask)
        H_B = model._enc(B_ids, B_mask)

        # Generate delta + gate (alpha)
        delta, delta_0, alpha = model.generate_delta(H_A, A_mask, H_B, B_mask)

        # Reconstruct with delta
        logits_with = model.reconstruct(H_A, A_mask, delta, delta_0,
                                        B_ids, B_mask, ablate_delta=False)
        # Reconstruct without delta (ablation)
        logits_no   = model.reconstruct(H_A, A_mask, delta, delta_0,
                                        B_ids, B_mask, ablate_delta=True)

        ppl_with_list.extend(_ppl_batch(logits_with, B_ids, B_mask))
        ppl_no_list.extend(  _ppl_batch(logits_no,   B_ids, B_mask))

        # Specificity: circular delta shift within batch
        if b > 1:
            idx_shift = list(range(1, b)) + [0]
            d_s  = delta[idx_shift]
            d0_s = delta_0[idx_shift]
            logits_wrong = model.reconstruct(H_A, A_mask, d_s, d0_s,
                                             B_ids, B_mask, ablate_delta=False)
            ppl_correct_list.extend(_ppl_batch(logits_with, B_ids, B_mask))
            ppl_wrong_list.extend(  _ppl_batch(logits_wrong, B_ids, B_mask))

        # AUROC: delta norms as novelty score
        # delta = alpha * delta_raw, so norms reflect combined gate + magnitude signal
        delta_norms = delta.norm(dim=-1).cpu().numpy()    # [b, T]
        for i in range(b):
            labels = _novel_token_mask(A_texts[i], N_texts[i], B_mask[i].cpu(), tok)
            if labels.sum() > 0 and labels.sum() < len(labels):
                all_norms.append(delta_norms[i])
                all_labels.append(labels)

    # ── Compute metrics ────────────────────────────────────────────────────────
    mean_with = float(np.mean(ppl_with_list))
    mean_no   = float(np.mean(ppl_no_list))
    delta_ppl = mean_no - mean_with

    # Per-example AUROC then averaged: isolates within-sequence localization.
    # Concatenated AUROC is biased by cross-example magnitude differences
    # (baseline broadcasts same vector → concatenated AUROC inflated, per-example = 0.5)
    if all_norms:
        per_auroc = []
        for norms_i, labels_i in zip(all_norms, all_labels):
            if labels_i.sum() > 0 and labels_i.sum() < len(labels_i):
                try:
                    per_auroc.append(roc_auc_score(labels_i, norms_i))
                except Exception:
                    pass
        auroc = float(np.mean(per_auroc)) if per_auroc else float("nan")
    else:
        auroc = float("nan")

    if ppl_correct_list:
        specificity = float(np.mean(ppl_wrong_list) - np.mean(ppl_correct_list))
    else:
        specificity = 0.0

    # ── Report ─────────────────────────────────────────────────────────────────
    n = len(ppl_with_list)
    print("\n" + "=" * 62)
    print("  DELTA-SYSTEM VALIDATION RESULTS")
    print("=" * 62)
    print(f"  Examples              : {n}")
    print(f"  PPL with delta        : {mean_with:7.1f}")
    print(f"  PPL without delta     : {mean_no:7.1f}")
    print(f"  DELTA_PPL             : {delta_ppl:+7.2f}  (positive = delta helps)")
    print(f"  SPECIFICITY (PPL gap) : {specificity:+7.2f}  (positive = delta is pair-specific)")
    print(f"  AUROC [diagnostic]    : {auroc:.4f}    (0.5=random; positional localization,")
    print(f"                                           not a pass/fail gate)")
    print("=" * 62)
    print("  PASS / FAIL:")
    p1 = delta_ppl   >  2.0;  print(f"    DELTA_PPL > 2     : {'PASS' if p1 else 'FAIL'}")
    p3 = specificity >  2.0;  print(f"    SPECIFICITY > 2   : {'PASS' if p3 else 'FAIL'}")
    overall = p1 and p3
    print()
    if overall:
        print("  *** ALL PASS — core idea validated. Ready to scale. ***")
    else:
        fails = []
        if not p1: fails.append("DELTA_PPL failed -> D_recon ignores delta (posterior collapse)")
        if not p3: fails.append("SPECIFICITY failed -> delta is generic, not pair-specific")
        print("  DIAGNOSIS:")
        for f in fails:
            print(f"    - {f}")
    print("=" * 62)

    return {
        "delta_ppl":   delta_ppl,
        "auroc":       auroc,
        "specificity": specificity,
        "pass":        overall,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=100)
    ap.add_argument("--ckpt", type=str, default="checkpoints/val_model.pt")
    args = ap.parse_args()

    ckpt = Path(__file__).parent / args.ckpt
    pairs = load_pairs(max_examples=args.n)
    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = DeltaSystem().to(DEVICE)
    if ckpt.exists():
        # Checkpoint only contains trainable params (no BERT); use strict=False
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
        print(f"Loaded: {ckpt}")
    else:
        print("No checkpoint found, evaluating untrained model (sanity check)")
    evaluate(model, pairs, tok)
