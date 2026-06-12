"""
wiki_atomic_eval.py -- Token-level novelty extraction on WikiAtomicEdits.

Gold label: the inserted phrase in each sentence-level Wikipedia edit.
  base_sentence  = A  (original)
  edited_sentence= B  (after insertion)
  phrase         = gold delta (the exact text that was inserted)

Metrics (token-level, per example then macro-averaged):
  AUC-ROC        -- threshold-free ranking quality
  Avg Precision  -- area under precision-recall curve
  F1 @ best thr  -- best F1 over threshold sweep

Methods compared:
  1. delta_system   -- ||delta[t]|| from trained G(A,B) (zero labels)
  2. bert_maxsim    -- 1 - max_j cos(H_B[t], H_A[j]) frozen BERT, no training
  3. lexical        -- 1 if subword token not in A vocab, else 0

bert_maxsim is the key ablation: same encoder as delta_system, no learned G.
If delta_system > bert_maxsim, the cross-attention training adds real value.

Dataset: google-research-datasets/wiki_atomic_edits  (English insertions)
Checkpoint: Wikipedia-trained wiki_model.pt (same as all prior experiments)

Run:
    python wiki_atomic_eval.py --ckpt /path/to/wiki_model.pt --n 2000
"""

import argparse
import difflib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import DeltaSystem

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


# ── Gold label construction ────────────────────────────────────────────────────

def _find_phrase_char_span(text: str, phrase: str):
    """Return (start, end) character span of phrase in text, or None."""
    idx = text.find(phrase)
    if idx == -1:
        # Try case-insensitive
        idx = text.lower().find(phrase.lower())
    if idx == -1:
        return None
    return idx, idx + len(phrase)


def _gold_labels_from_span(text: str, char_start: int, char_end: int,
                            tok: BertTokenizerFast):
    """
    Return float32 array of length T (BERT tokens of text):
      1.0 if token overlaps [char_start, char_end), else 0.0.
    Includes [CLS] and [SEP] (always 0).
    """
    enc     = tok(text, max_length=MAX_LEN, truncation=True,
                  return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    T       = len(offsets)
    labels  = np.zeros(T, dtype=np.float32)

    for t, (cs, ce) in enumerate(offsets):
        if cs == ce:
            continue  # special token or padding
        # Overlap: token [cs,ce) intersects inserted span [char_start,char_end)
        if cs < char_end and ce > char_start:
            labels[t] = 1.0

    return labels, enc["attention_mask"]


def _gold_labels_difflib(base: str, edited: str, tok: BertTokenizerFast):
    """
    Fallback: derive inserted phrase via word-level diff when 'phrase' field
    is missing or not found in edited_sentence.
    Returns (labels, attention_mask) or (None, None).
    """
    base_words   = base.split()
    edited_words = edited.split()
    sm = difflib.SequenceMatcher(None, base_words, edited_words, autojunk=False)

    # Collect indices of inserted words in edited
    inserted_word_idx = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            inserted_word_idx.update(range(j1, j2))

    if not inserted_word_idx:
        return None, None

    # Map word indices to char spans in edited string
    inserted_chars = set()
    pos = 0
    for wi, word in enumerate(edited_words):
        start = edited.find(word, pos)
        if start == -1:
            pos += 1
            continue
        end = start + len(word)
        if wi in inserted_word_idx:
            inserted_chars.update(range(start, end))
        pos = end

    if not inserted_chars:
        return None, None

    enc     = tok(edited, max_length=MAX_LEN, truncation=True,
                  return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    T       = len(offsets)
    labels  = np.zeros(T, dtype=np.float32)

    for t, (cs, ce) in enumerate(offsets):
        if cs == ce:
            continue
        if any(c in inserted_chars for c in range(cs, ce)):
            labels[t] = 1.0

    return labels, enc["attention_mask"]


def make_gold_labels(example: dict, tok: BertTokenizerFast):
    """
    Return (labels [T], real_len) or (None, None) if example is unusable.
    """
    base   = example["base_sentence"].strip()
    edited = example["edited_sentence"].strip()

    # Try 'phrase' field first (direct gold span)
    phrase = example.get("phrase", "").strip()
    labels = attn = None

    if phrase:
        span = _find_phrase_char_span(edited, phrase)
        if span is not None:
            labels, attn = _gold_labels_from_span(edited, span[0], span[1], tok)

    # Fall back to difflib
    if labels is None:
        labels, attn = _gold_labels_difflib(base, edited, tok)

    if labels is None:
        return None, None

    real_len = int(sum(attn)) if attn else MAX_LEN

    # Reject degenerate cases
    n_gold = int(labels.sum())
    if n_gold == 0 or n_gold >= real_len - 2:
        return None, None

    return labels[:real_len], real_len


# ── Data loading ───────────────────────────────────────────────────────────────

def _stream_wiki_atomic():
    """
    Try WikiAtomicEdits English insertions via several loading strategies.
    Returns (dataset_iterator, field_map) or raises RuntimeError.

    field_map maps our keys to dataset field names:
      base_sentence, edited_sentence, phrase
    """
    configs_to_try = [
        ("google-research-datasets/wiki_atomic_edits", "english_insertions"),
        ("google-research-datasets/wiki_atomic_edits", None),
    ]
    for repo, cfg in configs_to_try:
        for trust in (True, False):
            kwargs = dict(split="train", streaming=True)
            if cfg:
                kwargs["name"] = cfg
            if trust:
                kwargs["trust_remote_code"] = True
            try:
                ds = load_dataset(repo, **kwargs)
                print(f"  Loaded via {repo} cfg={cfg} trust={trust}")
                return ds, {"base": "base_sentence",
                            "edited": "edited_sentence",
                            "phrase": "phrase"}
            except Exception as e:
                print(f"  Attempt failed ({repo}, cfg={cfg}, trust={trust}): "
                      f"{type(e).__name__}")

    raise RuntimeError("WikiAtomicEdits unavailable — see below for IteraTeR fallback")


def _load_iterater(n: int, min_insert: int, max_insert_ratio: float,
                   tok: BertTokenizerFast):
    """
    Fallback: IteraTeR human-annotated sentences (wanyu/IteraTeR_human_sent).
    Uses 'meaning-changed' examples only — these are genuine content insertions.
    Gold spans derived via difflib (no explicit phrase field in this dataset).
    """
    print("Falling back to wanyu/IteraTeR_human_sent (meaning-changed only)...")
    ds = load_dataset("wanyu/IteraTeR_human_sent", split="train", streaming=True)

    pairs, checked = [], 0
    for ex in ds:
        checked += 1

        # Keep only 'meaning-changed' — new factual content was inserted
        labels_field = ex.get("labels") or []
        if isinstance(labels_field, str):
            labels_field = [labels_field]
        if not any("meaning" in str(l).lower() for l in labels_field):
            continue

        base   = (ex.get("before_sent") or "").strip()
        edited = (ex.get("after_sent")  or "").strip()

        if len(base) < 20 or len(edited) < 20:
            continue
        if len(edited) <= len(base):
            continue

        insert_len = len(edited) - len(base)
        if insert_len < min_insert:
            continue
        if insert_len / max(len(edited), 1) > max_insert_ratio:
            continue

        # No phrase field — use difflib
        ex_fake = {"base_sentence": base, "edited_sentence": edited, "phrase": ""}
        gold_labels, real_len = make_gold_labels(ex_fake, tok)
        if gold_labels is None:
            continue

        pairs.append({"A": base, "B": edited,
                      "labels": gold_labels, "real_len": real_len})
        if len(pairs) >= n:
            break
        if checked % 1000 == 0:
            print(f"  checked {checked:,} | collected {len(pairs)}/{n}")

    print(f"IteraTeR: {len(pairs)} pairs from {checked:,} examples")
    return pairs


def load_pairs(n: int, min_insert: int, max_insert_ratio: float,
               tok: BertTokenizerFast):
    """
    Load insertion pairs with gold labels. Tries WikiAtomicEdits first,
    falls back to IteraTeR if WikiAtomicEdits is unavailable.
    Returns list of dicts: {A, B, labels, real_len}
    """
    # ── Try WikiAtomicEdits ────────────────────────────────────────────────────
    wiki_ok = False
    try:
        ds, fmap = _stream_wiki_atomic()
        wiki_ok = True
        print("Source: WikiAtomicEdits English insertions")
    except RuntimeError as e:
        print(f"WikiAtomicEdits unavailable: {e}")

    if wiki_ok:
        pairs, checked = [], 0
        for ex in ds:
            checked += 1
            base   = (ex.get(fmap["base"])   or "").strip()
            edited = (ex.get(fmap["edited"]) or "").strip()

            if len(base) < 20 or len(edited) < 20:
                continue
            if len(edited) <= len(base):
                continue

            insert_len = len(edited) - len(base)
            if insert_len < min_insert:
                continue
            if insert_len / max(len(edited), 1) > max_insert_ratio:
                continue

            # Inject phrase field into a dict make_gold_labels can use
            ex_norm = {"base_sentence": base, "edited_sentence": edited,
                       "phrase": (ex.get(fmap["phrase"]) or "")}
            gold_labels, real_len = make_gold_labels(ex_norm, tok)
            if gold_labels is None:
                continue

            pairs.append({"A": base, "B": edited,
                          "labels": gold_labels, "real_len": real_len})
            if len(pairs) >= n:
                break
            if checked % 10000 == 0:
                print(f"  checked {checked:,} | collected {len(pairs)}/{n}")

        print(f"Loaded {len(pairs)} pairs from {checked:,} examples "
              f"({len(pairs)/max(checked,1)*100:.1f}% pass rate)")
        return pairs

    # ── Fallback: IteraTeR ─────────────────────────────────────────────────────
    return _load_iterater(n, min_insert, max_insert_ratio, tok)


# ── Method 1: delta_system ─────────────────────────────────────────────────────

@torch.no_grad()
def score_delta_system(model: DeltaSystem, pairs: list,
                       tok: BertTokenizerFast, batch_size: int = 16):
    model.eval()
    all_scores, all_labels = [], []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        eA = tok([p["A"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")

        A_ids  = eA["input_ids"].to(DEVICE)
        A_mask = eA["attention_mask"].to(DEVICE)
        B_ids  = eB["input_ids"].to(DEVICE)
        B_mask = eB["attention_mask"].to(DEVICE)

        H_A = model._enc(A_ids, A_mask)
        H_B = model._enc(B_ids, B_mask)
        delta, _, _ = model.generate_delta(H_A, A_mask, H_B, B_mask)

        norms = delta.norm(dim=-1).cpu().numpy()  # [b, T]

        for j, p in enumerate(batch):
            rl  = min(p["real_len"], MAX_LEN)
            s   = norms[j, :rl]
            lbl = p["labels"][:rl]
            if len(s) == len(lbl) and lbl.sum() > 0:
                all_scores.append(s)
                all_labels.append(lbl)

        if (i // batch_size) % 20 == 0:
            print(f"  delta_system: {min(i+batch_size, len(pairs))}/{len(pairs)}")

    return all_scores, all_labels


# ── Method 2: BERT max-sim (no trained G) ─────────────────────────────────────

@torch.no_grad()
def score_bert_maxsim(model: DeltaSystem, pairs: list,
                      tok: BertTokenizerFast, batch_size: int = 16):
    """
    For each token t in B: score_t = 1 - max_j cos(H_B[t], H_A[j]).
    Uses the same frozen BERT encoder as delta_system, but no learned G.
    High score = token in B has no close match in A = likely inserted.
    """
    model.eval()
    all_scores, all_labels = [], []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        eA = tok([p["A"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")

        A_ids  = eA["input_ids"].to(DEVICE)
        A_mask = eA["attention_mask"].to(DEVICE)
        B_ids  = eB["input_ids"].to(DEVICE)
        B_mask = eB["attention_mask"].to(DEVICE)

        H_A = model._enc(A_ids, A_mask)   # [b, T, 768]
        H_B = model._enc(B_ids, B_mask)   # [b, T, 768]

        # Normalize
        H_A_n = F.normalize(H_A, dim=-1)  # [b, T, 768]
        H_B_n = F.normalize(H_B, dim=-1)

        # cos[b, t_B, t_A] = H_B_n[b,t_B] · H_A_n[b,t_A]
        cos = torch.bmm(H_B_n, H_A_n.transpose(1, 2))  # [b, T_B, T_A]

        # Mask padding in A
        A_mask_f = A_mask.unsqueeze(1).float()          # [b, 1, T_A]
        cos = cos * A_mask_f + (1 - A_mask_f) * (-1e9)

        max_sim = cos.max(dim=-1).values.cpu().numpy()  # [b, T_B]
        novelty = 1.0 - max_sim                         # high = novel

        for j, p in enumerate(batch):
            rl  = min(p["real_len"], MAX_LEN)
            s   = novelty[j, :rl]
            lbl = p["labels"][:rl]
            if len(s) == len(lbl) and lbl.sum() > 0:
                all_scores.append(s)
                all_labels.append(lbl)

        if (i // batch_size) % 20 == 0:
            print(f"  bert_maxsim: {min(i+batch_size, len(pairs))}/{len(pairs)}")

    return all_scores, all_labels


# ── Method 3: Lexical baseline ─────────────────────────────────────────────────

def score_lexical(pairs: list, tok: BertTokenizerFast):
    """
    score_t = 1.0 if subword token not in A's token vocabulary, else 0.0.
    No model, no embeddings — pure vocabulary lookup.
    """
    all_scores, all_labels = [], []

    for p in pairs:
        enc_A = tok(p["A"], max_length=MAX_LEN, truncation=True,
                    add_special_tokens=True)
        enc_B = tok(p["B"], max_length=MAX_LEN, truncation=True,
                    add_special_tokens=True)

        a_token_ids = set(enc_A["input_ids"])
        b_ids       = enc_B["input_ids"]
        real_len    = min(p["real_len"], len(b_ids))

        scores = np.array(
            [0.0 if bid in a_token_ids else 1.0 for bid in b_ids],
            dtype=np.float32)

        lbl = p["labels"][:real_len]
        s   = scores[:real_len]
        if len(s) == len(lbl) and lbl.sum() > 0:
            all_scores.append(s)
            all_labels.append(lbl)

    return all_scores, all_labels


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(all_scores, all_labels, name: str) -> dict:
    """Macro-average per-example metrics + concatenated AUC."""
    per_auc, per_ap, per_f1 = [], [], []

    for s, l in zip(all_scores, all_labels):
        if l.sum() == 0 or l.sum() == len(l):
            continue
        try:
            per_auc.append(roc_auc_score(l, s))
            per_ap.append(average_precision_score(l, s))
            # F1 at best threshold
            thresholds = np.unique(np.percentile(s, np.arange(5, 96, 5)))
            best = max(
                f1_score(l, (s >= t).astype(int), zero_division=0)
                for t in thresholds)
            per_f1.append(best)
        except Exception:
            pass

    auc = float(np.mean(per_auc)) if per_auc else float("nan")
    ap  = float(np.mean(per_ap))  if per_ap  else float("nan")
    f1  = float(np.mean(per_f1))  if per_f1  else float("nan")
    n   = len(per_auc)

    print(f"  {name:<20} AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  "
          f"(n={n} examples)")
    return {"name": name, "auc": auc, "ap": ap, "f1": f1, "n": n}


# ── Qualitative examples ───────────────────────────────────────────────────────

def show_examples(pairs, delta_scores, n_show=5):
    """Print examples where delta_system correctly ranks inserted tokens highest."""
    print()
    print("QUALITATIVE EXAMPLES (top-scoring delta tokens vs gold insertion)")
    print("-" * 70)
    shown = 0
    for i, (p, s) in enumerate(zip(pairs, delta_scores)):
        if shown >= n_show:
            break
        lbl  = p["labels"][:len(s)]
        if lbl.sum() == 0:
            continue
        # Check if any gold token is in top-25% by delta norm
        thr  = np.percentile(s, 75)
        hit  = ((s >= thr) & (lbl == 1)).sum()
        if hit == 0:
            continue
        print(f"  A (base)  : {p['A'][:100]}")
        print(f"  B (edited): {p['B'][:100]}")
        gold_idx = np.where(lbl == 1)[0]
        top_idx  = np.argsort(s)[-5:][::-1]
        print(f"  Gold token positions : {list(gold_idx)}")
        print(f"  Top-5 delta positions: {list(top_idx)}")
        print(f"  Overlap: {hit} gold token(s) in top-25% delta norms")
        print()
        shown += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_checkpoint(default):
    candidates = [
        default,
        "/kaggle/working/checkpoints/wiki_model.pt",
        "/kaggle/working/checkpoints/kaggle_model.pt",
        str(ROOT / "checkpoints" / "wiki_model.pt"),
        str(ROOT / "checkpoints" / "val_model.pt"),
    ]
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default="/kaggle/working/checkpoints/wiki_model.pt")
    ap.add_argument("--n",     type=int, default=2000,
                    help="Number of WikiAtomicEdits pairs to evaluate")
    ap.add_argument("--min_insert", type=int, default=3,
                    help="Min chars in inserted phrase")
    ap.add_argument("--max_insert_ratio", type=float, default=0.5,
                    help="Max insertion / edited_sentence length ratio")
    ap.add_argument("--no_bert_maxsim", action="store_true",
                    help="Skip bert_maxsim baseline (faster)")
    args = ap.parse_args()

    print("=" * 70)
    print("DELTA SYSTEM — WikiAtomicEdits Token-Level Novelty Eval")
    print("=" * 70)
    print(f"Device    : {DEVICE}")
    print(f"Pairs     : {args.n}")
    print()

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ── Load data ──────────────────────────────────────────────────────────────
    pairs = load_pairs(args.n, args.min_insert, args.max_insert_ratio, tok)
    if len(pairs) < 50:
        print(f"ERROR: only {len(pairs)} pairs. Lower --min_insert or increase --n")
        sys.exit(1)
    print()

    # ── Load model ─────────────────────────────────────────────────────────────
    model = DeltaSystem().to(DEVICE)
    ckpt  = _find_checkpoint(args.ckpt)
    if ckpt.exists():
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE), strict=False)
        print(f"Checkpoint: {ckpt}")
    else:
        print(f"WARNING: no checkpoint at {ckpt} — using random weights")
    print()

    # ── Run methods ────────────────────────────────────────────────────────────
    print("Running delta_system...")
    ds_scores, ds_labels = score_delta_system(model, pairs, tok)
    print()

    bm_scores = bm_labels = None
    if not args.no_bert_maxsim:
        print("Running bert_maxsim baseline...")
        bm_scores, bm_labels = score_bert_maxsim(model, pairs, tok)
        print()

    print("Running lexical baseline...")
    lex_scores, lex_labels = score_lexical(pairs, tok)
    print()

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("=" * 70)
    print("TOKEN-LEVEL RESULTS  (macro-avg per example)")
    print("=" * 70)
    print(f"  {'Method':<20} {'AUC':>8} {'AvgPrec':>9} {'F1':>8}  {'n':>6}")
    print("-" * 70)

    results = []
    results.append(compute_metrics(ds_scores,  ds_labels,  "delta_system"))
    if bm_scores is not None:
        results.append(compute_metrics(bm_scores, bm_labels, "bert_maxsim"))
    results.append(compute_metrics(lex_scores, lex_labels, "lexical"))

    # Random baseline estimate
    print(f"  {'random (est.)':<20} {'~0.500':>8} {'~frac':>9} {'~frac':>8}")
    print()

    # ── Summary interpretation ─────────────────────────────────────────────────
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    ds_r  = results[0]
    lex_r = results[-1]

    print(f"  delta_system AUC  : {ds_r['auc']:.4f}")
    if bm_scores is not None:
        bm_r = results[1]
        print(f"  bert_maxsim AUC  : {bm_r['auc']:.4f}  "
              f"({'delta_system wins' if ds_r['auc'] > bm_r['auc'] else 'bert_maxsim wins'})")
        gain = ds_r['auc'] - bm_r['auc']
        print(f"  G cross-attn gain: {gain:+.4f} AUC over frozen BERT alone")
    print(f"  lexical AUC       : {lex_r['auc']:.4f}")
    print()

    # Cross-domain summary
    print("=" * 70)
    print("CROSS-DATASET SUMMARY (all experiments)")
    print("=" * 70)
    print(f"  {'Dataset':<35} {'Metric':<14} {'Value':>8}  {'Domain'}")
    print("-" * 70)
    print(f"  {'Wikipedia (8000tr/1000ev)':<35} {'DELTA_PPL':>14} {'  +755':>8}  same domain")
    print(f"  {'HotpotQA (5000tr/500ev)':<35} {'DELTA_PPL':>14} {'  +480':>8}  cross-dataset")
    print(f"  {'NewsEdits AP (0tr/500ev)':<35} {'DELTA_PPL':>14} {' +1295':>8}  cross-domain")
    print(f"  {'Token eval (0tr/{args.n}ev)':<35} "
          f"{'Token AUC':>14} {ds_r['auc']:>+8.4f}  zero-shot token F1")
    print("=" * 70)

    # ── Qualitative ────────────────────────────────────────────────────────────
    show_examples(pairs, ds_scores)


if __name__ == "__main__":
    main()
