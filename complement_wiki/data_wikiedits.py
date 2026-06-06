"""
data_wikiedits.py -- WikiAtomicEdits loader for complement validation
=====================================================================

WikiAtomicEdits (Faruqui et al. 2018): each row is a sentence pair where the
edited sentence B = base sentence A + one contiguous inserted phrase.
  base_sentence (A)  |  inserted_phrase (gold complement)  |  edited_sentence (B)

This is the IDEAL testbed for the complement idea: A and B overlap maximally,
and the inserted phrase is an explicit, literal ground-truth "what B adds."

We filter out trivial short insertions (30% are 1 word) and keep substantive
multi-word insertions so the residual is large enough to matter.

The Dataset/collator emit the SAME keys as generator_train.HopPairDataset so the
existing train_epoch/validate loop is reused unchanged.
"""

import gzip
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

# Reuse constants from the (fixed) model definition
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v3"))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))
from generator_train import MAX_A_LEN, MAX_B_LEN  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
WIKI_URL = "https://storage.googleapis.com/wiki-atomic-edits/english/insertions.tsv.gz"
DATA_DIR  = _HERE / "data"
CACHE_DIR = _HERE / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
GZ_PATH = DATA_DIR / "insertions.tsv.gz"

MIN_INSERT_TOKENS = 5    # keep only substantive multi-word insertions

# function-word-ish tokens; an insertion that is ONLY these is trivial
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "was", "were", "are", "be", "been", "as", "by", "with", "that", "this",
    "it", "its", "his", "her", "their", "from", "which", "who", "也", ",", ".",
}


# ── Download ────────────────────────────────────────────────────────────────

def download(dest: Path = GZ_PATH) -> Path:
    """Download the WikiAtomicEdits English insertions TSV.gz if missing.
    Robust: removes any zero-byte leftover (e.g. from a failed wget), downloads
    via urllib, and verifies a non-trivial size."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[data] already present: {dest} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    if dest.exists():
        dest.unlink()   # drop empty/partial file so we actually re-download
    print(f"[data] downloading {WIKI_URL} ...")
    import urllib.request
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = dest.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(
            f"download failed: {dest} is only {size} bytes. "
            f"Check connectivity / URL {WIKI_URL}"
        )
    print(f"[data] saved {dest} ({size/1e6:.0f} MB)")
    return dest


# ── Filtering ───────────────────────────────────────────────────────────────

def _ok_insertion(phrase: str, min_tokens: int) -> bool:
    toks = phrase.split()
    if len(toks) < min_tokens:
        return False
    # require at least 2 content (non-stopword, alphabetic) tokens
    content = [t for t in toks if t.lower() not in _STOP and any(c.isalpha() for c in t)]
    return len(content) >= 2


def load_triples(
    max_examples: int = None,
    min_insert_tokens: int = MIN_INSERT_TOKENS,
    cache: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    Return filtered [{"base":A, "inserted":phrase, "edited":B}, ...].
    Streams the gz, applies the multi-word filter, shuffles, subsamples.
    """
    tag = f"{max_examples}_{min_insert_tokens}"
    cache_file = CACHE_DIR / f"wikiedits_{tag}.pkl"
    if cache and cache_file.exists():
        print(f"[data] loading cached triples: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    download()
    # Read more than we need (filtering drops many), then subsample.
    over = (max_examples * 12) if max_examples else None
    triples: List[Dict] = []
    with gzip.open(GZ_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            base, inserted, edited = parts
            if not base or not edited or not inserted:
                continue
            if not _ok_insertion(inserted, min_insert_tokens):
                continue
            triples.append({"base": base, "inserted": inserted, "edited": edited})
            if over and len(triples) >= over:
                break

    random.Random(seed).shuffle(triples)
    if max_examples:
        triples = triples[:max_examples]
    print(f"[data] filtered triples: {len(triples):,} "
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
