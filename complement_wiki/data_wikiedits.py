"""
data_wikiedits.py -- clean-insertion loader for complement validation
=====================================================================

Source: IteraTeR (wanyu/IteraTeR_full_sent on HuggingFace; Apache-2.0, parquet,
no trust_remote_code). 158K+ sentence revision pairs (before_sent -> after_sent)
across Wikipedia / arXiv / news, with edit-intention labels.

(The original WikiAtomicEdits GCS bucket is 403-gated, so we derive the SAME
"B = A + one contiguous inserted span" property from IteraTeR via a token diff.)

For each revision pair we keep ONLY clean insertions: after_sent equals before_sent
with exactly one contiguous span inserted. That inserted span is the explicit,
literal ground-truth "what B adds" -- the complement target. We require the span
to have >= MIN_INSERT_TOKENS tokens and >= 2 content words so the residual matters.

  base (A)  |  inserted_phrase (gold complement)  |  edited (B)

The Dataset/collator emit the SAME keys as generator_train.HopPairDataset so the
existing train_epoch/validate loop is reused unchanged.
"""

import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

# Reuse constants from the (fixed) model definition
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v3"))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))
from generator_train import MAX_A_LEN, MAX_B_LEN  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET = "wanyu/IteraTeR_full_sent"
CACHE_DIR  = _HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MIN_INSERT_TOKENS = 5    # keep only substantive multi-word insertions

# function-word-ish tokens; an insertion that is ONLY these is trivial
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "was", "were", "are", "be", "been", "as", "by", "with", "that", "this",
    "it", "its", "his", "her", "their", "from", "which", "who", "也", ",", ".",
}


# ── Clean-insertion diff ──────────────────────────────────────────────────────

import re
_TOK = re.compile(r"\w+|[^\w\s]")   # words OR single punctuation chars (so 'home.' -> 'home','.')


def clean_insertion(before: str, after: str, min_tokens: int) -> Optional[str]:
    """
    Return the inserted span IFF `after` == `before` with exactly ONE contiguous
    span inserted (before = prefix + suffix, after = prefix + INSERT + suffix).
    Punctuation is tokenized separately so end-of-sentence inserts (where the
    period 'moves') are still detected. Requires >= min_tokens WORD tokens in the
    insert and >= 2 content words.
    """
    bt, at = _TOK.findall(before), _TOK.findall(after)
    if len(at) <= len(bt):
        return None
    p = 0
    while p < len(bt) and p < len(at) and bt[p] == at[p]:
        p += 1
    s = 0
    while s < (len(bt) - p) and s < (len(at) - p) and bt[len(bt)-1-s] == at[len(at)-1-s]:
        s += 1
    if p + s != len(bt):          # before is NOT just after-minus-one-span -> reject
        return None
    insert = at[p: len(at) - s]
    words = [t for t in insert if any(c.isalpha() or c.isdigit() for c in t)]
    if len(words) < min_tokens:
        return None
    content = [t for t in words if t.lower() not in _STOP]
    if len(content) < 2:
        return None
    return " ".join(insert)


def load_triples(
    max_examples: int = None,
    min_insert_tokens: int = MIN_INSERT_TOKENS,
    cache: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    Load IteraTeR, keep only clean contiguous insertions, return
    [{"base":A, "inserted":span, "edited":B}, ...]; shuffle + subsample + cache.
    """
    tag = f"{max_examples}_{min_insert_tokens}"
    cache_file = CACHE_DIR / f"iterater_ins_{tag}.pkl"
    if cache and cache_file.exists():
        print(f"[data] loading cached triples: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from datasets import load_dataset
    print(f"[data] loading {HF_DATASET} (HuggingFace, parquet) ...")
    over = (max_examples * 5) if max_examples else None   # early-stop for small runs
    triples: List[Dict] = []
    for split in ["train", "validation", "test"]:
        if over and len(triples) >= over:
            break
        try:
            ds = load_dataset(HF_DATASET, split=split)
        except Exception as e:
            print(f"[data] split {split} unavailable: {e}")
            continue
        for row in ds:
            before = (row.get("before_sent") or "").strip()
            after  = (row.get("after_sent")  or "").strip()
            if not before or not after:
                continue
            ins = clean_insertion(before, after, min_insert_tokens)
            if ins is None:
                continue
            triples.append({"base": before, "inserted": ins, "edited": after})
            if over and len(triples) >= over:
                break
        print(f"[data]   {split}: cumulative clean insertions = {len(triples):,}")

    random.Random(seed).shuffle(triples)
    if max_examples:
        triples = triples[:max_examples]
    print(f"[data] clean-insertion triples: {len(triples):,} "
          f"(min_insert_tokens={min_insert_tokens})")

    if cache:
        with open(cache_file, "wb") as f:
            pickle.dump(triples, f)
    return triples


# ── Dataset + collator (keys match generator_train.HopPairDataset) ───────────

class WikiEditDataset(Dataset):
    """Yields (A=base, B=edited). 'inserted' kept only for eval, not training."""

    def __init__(self, triples: List[Dict]):
        self.data = triples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        t = self.data[i]
        return (t["base"], t["edited"])


def make_collator(tokenizer: BertTokenizerFast):
    def collate(batch):
        a_texts = [x[0] for x in batch]
        b_texts = [x[1] for x in batch]
        enc_a = tokenizer(a_texts, max_length=MAX_A_LEN, truncation=True,
                          padding="max_length", return_tensors="pt")
        enc_b = tokenizer(b_texts, max_length=MAX_B_LEN, truncation=True,
                          padding="max_length", return_tensors="pt")
        dec_ids = enc_b["input_ids"][:, :-1]
        labels  = enc_b["input_ids"][:, 1:].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids_a":   enc_a["input_ids"],
            "attn_mask_a":   enc_a["attention_mask"],
            "dec_input_ids": dec_ids,
            "input_ids_b":   enc_b["input_ids"],
            "attn_mask_b":   enc_b["attention_mask"],
            "labels":        labels,
        }
    return collate


# ── Smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    trips = load_triples(max_examples=args.n, cache=False)
    print(f"\nLoaded {len(trips)} triples. Examples:")
    for t in trips[:3]:
        print(f"  A: {t['base'][:80]}")
        print(f"  +: {t['inserted']}")
        print(f"  B: {t['edited'][:80]}")
        print()
