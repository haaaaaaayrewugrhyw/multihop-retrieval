"""
gpt_comparison_eval.py -- LLM vs bert_maxsim on token-level insertion localization.

Evaluates GPT-4o (or Groq Llama-3.3-70B) on the same IteraTeR meaning-changed
pairs used in wiki_atomic_eval.py. LLM is prompted to identify verbatim what
text was inserted/changed in B compared to A.

Metrics (token-level, macro-avg per example):
  Precision, Recall, F1       -- from LLM-predicted spans vs gold difflib labels
  AUC, AvgPrec, F1            -- from bert_maxsim continuous scores vs gold labels

Key question: does bert_maxsim (0 parameters trained, frozen BERT)
match GPT-4o (~175B params, API cost) on insertion localization?

LLM providers supported:
  --provider openai   : GPT-4o  (set OPENAI_API_KEY env var)
  --provider groq     : Llama-3.3-70B  (set GROQ_API_KEY env var)
  --provider anthropic: Claude claude-sonnet-4-6 (set ANTHROPIC_API_KEY env var)

Run:
    python gpt_comparison_eval.py \\
        --provider groq \\
        --n 200 \\
        --ckpt /path/to/wiki_model.pt
"""

import argparse
import difflib
import os
import sys
import time
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

# ── Prompting ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are given two sentences: A (original) and B (edited). "
    "Identify exactly what text was inserted or changed in B compared to A. "
    "Output ONLY the new or changed text from B, verbatim. "
    "Do not explain. Do not paraphrase. Output only the literal text from B."
)


def _user_prompt(base: str, edited: str) -> str:
    return (
        f"A (original): {base}\n"
        f"B (edited):   {edited}\n\n"
        "What text is new or changed in B compared to A? "
        "Output only the verbatim text from B."
    )


# ── LLM clients ───────────────────────────────────────────────────────────────

def _call_openai(prompt: str, model: str = "gpt-4o") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=128,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=128,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(prompt: str,
                    model: str = "claude-sonnet-4-6") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def call_llm(base: str, edited: str, provider: str) -> str:
    prompt = _user_prompt(base, edited)
    if provider == "openai":
        return _call_openai(prompt)
    elif provider == "groq":
        return _call_groq(prompt)
    elif provider == "anthropic":
        return _call_anthropic(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Gold label construction (same as wiki_atomic_eval.py) ─────────────────────

def _gold_labels_difflib(base: str, edited: str, tok: BertTokenizerFast):
    base_words   = base.split()
    edited_words = edited.split()
    sm = difflib.SequenceMatcher(None, base_words, edited_words, autojunk=False)

    inserted_word_idx = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            inserted_word_idx.update(range(j1, j2))

    if not inserted_word_idx:
        return None, None

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

    real_len = int(sum(enc["attention_mask"]))
    return labels, real_len


# ── LLM span → token labels ───────────────────────────────────────────────────

def _llm_pred_labels(edited: str, llm_output: str,
                     tok: BertTokenizerFast, real_len: int) -> np.ndarray:
    """
    Find LLM's predicted span in edited sentence.
    Returns binary array: 1 = LLM predicted this token as novel.

    Strategy:
      1. Exact substring search
      2. Fuzzy: find best aligned subsequence using SequenceMatcher
    """
    enc     = tok(edited, max_length=MAX_LEN, truncation=True,
                  return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    T       = len(offsets)
    preds   = np.zeros(T, dtype=np.float32)

    if not llm_output:
        return preds

    # Strategy 1: exact substring
    idx = edited.find(llm_output)
    if idx != -1:
        char_start = idx
        char_end   = idx + len(llm_output)
        for t, (cs, ce) in enumerate(offsets):
            if cs == ce:
                continue
            if cs < char_end and ce > char_start:
                preds[t] = 1.0
        return preds

    # Strategy 2: token overlap — mark B tokens whose text appears in LLM output
    llm_words = set(llm_output.lower().split())
    b_text = edited.lower()
    for t, (cs, ce) in enumerate(offsets):
        if cs == ce:
            continue
        tok_text = b_text[cs:ce].strip()
        if tok_text in llm_words and len(tok_text) > 2:
            preds[t] = 1.0

    return preds


# ── Data loading ───────────────────────────────────────────────────────────────

def load_iterater_pairs(n: int, tok: BertTokenizerFast):
    print("Loading wanyu/IteraTeR_human_sent (meaning-changed only)...")
    ds = load_dataset("wanyu/IteraTeR_human_sent",
                      split="train", streaming=True)

    pairs, checked = [], 0
    for ex in ds:
        checked += 1
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

        gold_labels, real_len = _gold_labels_difflib(base, edited, tok)
        if gold_labels is None:
            continue

        n_gold = int(gold_labels.sum())
        if n_gold == 0 or n_gold >= real_len - 2:
            continue

        pairs.append({"A": base, "B": edited,
                      "labels": gold_labels, "real_len": real_len})
        if len(pairs) >= n:
            break
        if checked % 500 == 0:
            print(f"  checked {checked:,} | collected {len(pairs)}/{n}")

    print(f"Loaded {len(pairs)} pairs from {checked:,} examples")
    return pairs


# ── bert_maxsim scoring ────────────────────────────────────────────────────────

@torch.no_grad()
def score_bert_maxsim(model: DeltaSystem, pairs: list,
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

        H_A_n = F.normalize(H_A, dim=-1)
        H_B_n = F.normalize(H_B, dim=-1)
        cos    = torch.bmm(H_B_n, H_A_n.transpose(1, 2))
        A_mask_f = A_mask.unsqueeze(1).float()
        cos = cos * A_mask_f + (1 - A_mask_f) * (-1e9)
        max_sim = cos.max(dim=-1).values.cpu().numpy()
        novelty = 1.0 - max_sim

        for j, p in enumerate(batch):
            rl  = min(p["real_len"], MAX_LEN)
            s   = novelty[j, :rl]
            lbl = p["labels"][:rl]
            if len(s) == len(lbl) and lbl.sum() > 0:
                all_scores.append(s)
                all_labels.append(lbl)

    return all_scores, all_labels


# ── Metrics ────────────────────────────────────────────────────────────────────

def _f1(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(gold, pred, zero_division=0))


def compute_continuous_metrics(all_scores, all_labels, name: str) -> dict:
    """For methods with continuous scores (bert_maxsim)."""
    per_auc, per_ap, per_f1 = [], [], []
    for s, l in zip(all_scores, all_labels):
        if l.sum() == 0 or l.sum() == len(l):
            continue
        try:
            per_auc.append(roc_auc_score(l, s))
            per_ap.append(average_precision_score(l, s))
            thresholds = np.unique(np.percentile(s, np.arange(5, 96, 5)))
            per_f1.append(max(
                f1_score(l, (s >= t).astype(int), zero_division=0)
                for t in thresholds))
        except Exception:
            pass

    auc = float(np.mean(per_auc)) if per_auc else float("nan")
    ap  = float(np.mean(per_ap))  if per_ap  else float("nan")
    f1  = float(np.mean(per_f1))  if per_f1  else float("nan")
    return {"name": name, "auc": auc, "ap": ap, "f1": f1, "n": len(per_auc)}


def compute_binary_metrics(all_preds, all_labels, name: str) -> dict:
    """For LLM predictions (binary span output, no continuous score)."""
    per_p, per_r, per_f1 = [], [], []
    for pred, gold in zip(all_preds, all_labels):
        if gold.sum() == 0:
            continue
        min_len = min(len(pred), len(gold))
        p = pred[:min_len]; g = gold[:min_len]
        tp = float((p * g).sum())
        prec = tp / max(p.sum(), 1)
        rec  = tp / max(g.sum(), 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_p.append(prec); per_r.append(rec); per_f1.append(f1)

    prec = float(np.mean(per_p))  if per_p  else float("nan")
    rec  = float(np.mean(per_r))  if per_r  else float("nan")
    f1   = float(np.mean(per_f1)) if per_f1 else float("nan")
    return {"name": name, "prec": prec, "rec": rec, "f1": f1,
            "n": len(per_f1)}


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_checkpoint(default):
    candidates = [
        default,
        "/kaggle/working/checkpoints/wiki_model.pt",
        "/kaggle/working/checkpoints/kaggle_model.pt",
        str(ROOT / "checkpoints" / "wiki_model.pt"),
    ]
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq",
                    choices=["openai", "groq", "anthropic"],
                    help="LLM provider to use")
    ap.add_argument("--n",    type=int, default=200,
                    help="Number of IteraTeR pairs to evaluate")
    ap.add_argument("--ckpt", default="/kaggle/working/checkpoints/wiki_model.pt")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="Seconds between API calls (rate limiting)")
    args = ap.parse_args()

    model_names = {
        "openai":    "gpt-4o",
        "groq":      "llama-3.3-70b-versatile",
        "anthropic": "claude-sonnet-4-6",
    }
    llm_name = model_names[args.provider]

    print("=" * 70)
    print("DELTA SYSTEM — LLM vs bert_maxsim Comparison")
    print("=" * 70)
    print(f"Device   : {DEVICE}")
    print(f"Provider : {args.provider} ({llm_name})")
    print(f"Pairs    : {args.n}")
    print()

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ── Load data ──────────────────────────────────────────────────────────────
    pairs = load_iterater_pairs(args.n, tok)
    if len(pairs) < 20:
        print(f"ERROR: only {len(pairs)} pairs loaded")
        sys.exit(1)
    print()

    # ── Load model (for bert_maxsim) ───────────────────────────────────────────
    model = DeltaSystem().to(DEVICE)
    ckpt  = _find_checkpoint(args.ckpt)
    if ckpt.exists():
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE), strict=False)
        print(f"Checkpoint: {ckpt}")
    else:
        print("WARNING: no checkpoint — bert_maxsim still runs (uses frozen BERT only)")
    print()

    # ── bert_maxsim ────────────────────────────────────────────────────────────
    print("Running bert_maxsim...")
    bm_scores, bm_labels = score_bert_maxsim(model, pairs, tok)
    bm_result = compute_continuous_metrics(bm_scores, bm_labels, "bert_maxsim")
    print()

    # ── LLM evaluation ─────────────────────────────────────────────────────────
    print(f"Running {llm_name} on {len(pairs)} pairs...")
    print(f"  Rate limit delay: {args.delay}s between calls")
    print()

    llm_preds  = []
    llm_labels = []
    llm_outputs = []
    errors = 0

    for i, p in enumerate(pairs):
        try:
            output = call_llm(p["A"], p["B"], args.provider)
            pred   = _llm_pred_labels(p["B"], output,
                                      tok, p["real_len"])
            llm_preds.append(pred)
            llm_labels.append(p["labels"])
            llm_outputs.append(output)

            if i < 3:
                print(f"  Example {i+1}:")
                print(f"    A     : {p['A'][:80]}")
                print(f"    B     : {p['B'][:80]}")
                print(f"    LLM   : {output[:80]}")
                gold_pos = np.where(p["labels"] == 1)[0]
                pred_pos = np.where(pred == 1)[0]
                print(f"    Gold tokens : {list(gold_pos[:10])}")
                print(f"    Pred tokens : {list(pred_pos[:10])}")
                print()

        except Exception as e:
            errors += 1
            llm_preds.append(np.zeros(p["real_len"], dtype=np.float32))
            llm_labels.append(p["labels"])
            llm_outputs.append("")
            if errors <= 3:
                print(f"  API error on pair {i}: {e}")

        if args.delay > 0 and i < len(pairs) - 1:
            time.sleep(args.delay)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} done  (errors: {errors})")

    llm_result = compute_binary_metrics(llm_preds, llm_labels, llm_name)
    print()

    # ── Results table ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS — Token-Level Insertion Localization (IteraTeR meaning-changed)")
    print("=" * 70)
    print(f"\n  {'Method':<30} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}  {'n':>5}")
    print("  " + "-" * 66)

    # LLM (binary — no AUC)
    print(f"  {llm_result['name']:<30} "
          f"{llm_result['prec']:>8.4f} "
          f"{llm_result['rec']:>8.4f} "
          f"{llm_result['f1']:>8.4f} "
          f"{'  N/A':>8}  "
          f"{llm_result['n']:>5}")

    # bert_maxsim (continuous — has AUC)
    # Compute P/R at best F1 threshold for bert_maxsim too
    bm_f1_scores = []
    bm_precs     = []
    bm_recs      = []
    for s, l in zip(bm_scores, bm_labels):
        if l.sum() == 0 or l.sum() == len(l):
            continue
        thresholds = np.unique(np.percentile(s, np.arange(5, 96, 5)))
        best_f1 = best_prec = best_rec = 0.0
        for t in thresholds:
            pred = (s >= t).astype(int)
            tp   = float((pred * l).sum())
            prec = tp / max(pred.sum(), 1)
            rec  = tp / max(l.sum(), 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1 = f1; best_prec = prec; best_rec = rec
        bm_f1_scores.append(best_f1)
        bm_precs.append(best_prec)
        bm_recs.append(best_rec)

    bm_prec_avg = float(np.mean(bm_precs)) if bm_precs else float("nan")
    bm_rec_avg  = float(np.mean(bm_recs))  if bm_recs  else float("nan")

    print(f"  {'bert_maxsim (frozen BERT)':<30} "
          f"{bm_prec_avg:>8.4f} "
          f"{bm_rec_avg:>8.4f} "
          f"{bm_result['f1']:>8.4f} "
          f"{bm_result['auc']:>8.4f}  "
          f"{bm_result['n']:>5}")

    # Reported from prior run
    print(f"  {'delta_system (trained G)':<30} "
          f"{'  N/A':>8} "
          f"{'  N/A':>8} "
          f"{'0.4090':>8} "
          f"{'0.5047':>8}  "
          f"{'535':>5}  (prior run)")

    print(f"  {'lexical (vocab diff)':<30} "
          f"{'  N/A':>8} "
          f"{'  N/A':>8} "
          f"{'0.6637':>8} "
          f"{'0.8334':>8}  "
          f"{'535':>5}  (prior run)")

    print(f"  {'random baseline':<30} "
          f"{'~frac':>8} {'~frac':>8} {'~frac':>8} {'~0.500':>8}")

    print()

    # ── Interpretation ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    gap = bm_result['f1'] - llm_result['f1']
    if gap > 0.05:
        print(f"  bert_maxsim BEATS {llm_name} by F1 {gap:+.4f}")
        print(f"  -> Frozen BERT cosine similarity outperforms a {llm_name}-scale LLM")
        print(f"     on token-level insertion localization, with ZERO training.")
    elif gap > -0.05:
        print(f"  bert_maxsim MATCHES {llm_name} (F1 gap = {gap:+.4f})")
        print(f"  -> Frozen BERT achieves LLM-level localization with zero training.")
    else:
        print(f"  {llm_name} BEATS bert_maxsim by F1 {-gap:+.4f}")
        print(f"  -> LLM has an advantage. bert_maxsim F1={bm_result['f1']:.4f} "
              f"is still strong for a zero-training baseline.")

    print()
    print(f"  Errors during API calls: {errors}/{len(pairs)}")
    print("=" * 70)

    # ── Paper headline ─────────────────────────────────────────────────────────
    print()
    print("PAPER HEADLINE NUMBERS:")
    print(f"  {llm_name:<35} F1 = {llm_result['f1']:.4f}")
    print(f"  bert_maxsim (frozen BERT, 0 training)  F1 = {bm_result['f1']:.4f}  "
          f"AUC = {bm_result['auc']:.4f}")
    print(f"  delta_system (reconstruction, 0 labels) AUC = 0.5047  "
          f"(localization: random)")
    print(f"  lexical (vocab diff)                    F1 = 0.6637  "
          f"AUC = 0.8334")
    print()


if __name__ == "__main__":
    main()
