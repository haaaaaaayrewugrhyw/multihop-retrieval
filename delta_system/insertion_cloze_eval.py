"""
insertion_cloze_eval.py -- Proof that delta encodes "what B adds beyond A"

Core test (Novel-only DELTA_PPL):

  For each pair (A=before, B=after, novel_mask=which tokens were inserted):
    PPL_novel_with_delta  = D_recon(A, delta=G(A,B)) perplexity on novel tokens only
    PPL_novel_no_delta    = D_recon(A, delta=0)       perplexity on novel tokens only
    PPL_shared_with_delta = D_recon(A, delta=G(A,B)) perplexity on shared tokens only
    PPL_shared_no_delta   = D_recon(A, delta=0)       perplexity on shared tokens only

  Novel  DELTA_PPL = mean(PPL_novel_no_delta   - PPL_novel_with_delta)
  Shared DELTA_PPL = mean(PPL_shared_no_delta  - PPL_shared_with_delta)

If Novel DELTA_PPL >> Shared DELTA_PPL:
  delta specifically helps predict the INSERTED tokens, not the shared ones.
  → delta encodes what B adds beyond A. Proven.

Why this is the right proof:
  D_recon(A, delta=0)       must reconstruct B from A alone — hard for novel tokens.
  D_recon(A, delta=G(A,B))  has access to delta — if delta encodes novel content,
                             it should make novel token prediction much easier.
  Shared tokens (copied from A) should benefit equally from both conditions
  because they are already in A. If delta helps NOVEL tokens MORE than SHARED tokens,
  delta is specifically encoding the novel content, not just compressing B.

Dataset : wanyu/IteraTeR_human_sent (meaning-changed, same as Exp 8/9)
Checkpoint: Wikipedia-trained wiki_model.pt (same as all prior experiments)

Run:
    python insertion_cloze_eval.py --ckpt /path/to/wiki_model.pt --n 500
"""

import argparse
import difflib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import DeltaSystem

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


# ── Gold novel-token mask (same logic as wiki_atomic_eval.py) ─────────────────

def _novel_mask_difflib(base: str, edited: str, tok: BertTokenizerFast):
    """
    Return float32 array of length real_len:
      1.0 if BERT token is in the inserted/changed content of edited vs base
      0.0 if token exists unchanged in base

    Returns (novel_mask, real_len) or (None, None) if unusable.
    """
    base_words   = base.split()
    edited_words = edited.split()
    sm = difflib.SequenceMatcher(None, base_words, edited_words, autojunk=False)

    inserted_word_idx = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            inserted_word_idx.update(range(j1, j2))

    if not inserted_word_idx:
        return None, None

    # Map inserted word indices → character spans in edited string
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
    real_len = int(sum(enc["attention_mask"]))
    T        = len(offsets)
    mask     = np.zeros(T, dtype=np.float32)

    for t, (cs, ce) in enumerate(offsets):
        if cs == ce:
            continue  # special token / padding
        if any(c in inserted_chars for c in range(cs, ce)):
            mask[t] = 1.0

    n_novel = int(mask[:real_len].sum())
    n_total = real_len - 2  # exclude CLS, SEP
    if n_novel == 0 or n_novel >= n_total:
        return None, None

    return mask[:real_len], real_len


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pairs(n: int, tok: BertTokenizerFast):
    """Load IteraTeR meaning-changed pairs with gold novel-token masks."""
    print("Loading wanyu/IteraTeR_human_sent (meaning-changed only)...")
    ds = load_dataset("wanyu/IteraTeR_human_sent", split="train", streaming=True)

    pairs, checked = [], 0
    for ex in ds:
        checked += 1
        labels = ex.get("labels") or []
        if isinstance(labels, str):
            labels = [labels]
        if not any("meaning" in str(l).lower() for l in labels):
            continue

        base   = (ex.get("before_sent") or "").strip()
        edited = (ex.get("after_sent")  or "").strip()
        if len(base) < 20 or len(edited) < 20:
            continue
        if len(edited) <= len(base):
            continue

        novel_mask, real_len = _novel_mask_difflib(base, edited, tok)
        if novel_mask is None:
            continue

        pairs.append({"A": base, "B": edited,
                      "novel_mask": novel_mask, "real_len": real_len})
        if len(pairs) >= n:
            break
        if checked % 1000 == 0:
            print(f"  checked {checked:,} | collected {len(pairs)}/{n}")

    print(f"Loaded {len(pairs)} pairs from {checked:,} examples")
    return pairs


# ── Per-token perplexity with / without delta ──────────────────────────────────

@torch.no_grad()
def eval_novel_ppl(model: DeltaSystem, pairs: list,
                   tok: BertTokenizerFast, batch_size: int = 8):
    """
    For each pair, compute per-token cross-entropy with delta and without delta.
    Split results into novel tokens and shared tokens.

    Returns four lists (one float per pair):
      ppl_novel_with, ppl_novel_no, ppl_shared_with, ppl_shared_no
    """
    model.eval()

    ppl_novel_with  = []
    ppl_novel_no    = []
    ppl_shared_with = []
    ppl_shared_no   = []

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

        # Forward with delta
        logits_with, _, _, _, _ = model(A_ids, A_mask, B_ids, B_mask,
                                        ablate_delta=False)
        # Forward without delta (ablation: zeros all delta info)
        logits_no, _, _, _, _ = model(A_ids, A_mask, B_ids, B_mask,
                                      ablate_delta=True)

        # Per-token CE: shape [b, T]
        b, T, V = logits_with.shape
        targets = B_ids  # teacher-forced targets

        ce_with = F.cross_entropy(
            logits_with.view(-1, V), targets.view(-1),
            reduction="none").view(b, T)
        ce_no = F.cross_entropy(
            logits_no.view(-1, V), targets.view(-1),
            reduction="none").view(b, T)

        ce_with = ce_with.cpu().numpy()
        ce_no   = ce_no.cpu().numpy()
        b_mask  = B_mask.cpu().numpy()

        for j, p in enumerate(batch):
            rl       = min(p["real_len"], T)
            nov_mask = p["novel_mask"][:rl]       # 1 = novel, 0 = shared
            tok_mask = b_mask[j, :rl].astype(bool) # 1 = real token
            # Exclude CLS (position 0) and SEP (last real position)
            if rl > 2:
                tok_mask[0] = False
                tok_mask[rl - 1] = False

            novel_idx  = (nov_mask == 1) & tok_mask
            shared_idx = (nov_mask == 0) & tok_mask

            if novel_idx.sum() == 0 or shared_idx.sum() == 0:
                continue

            # Mean CE → PPL per category per pair
            ppl_novel_with.append(float(np.exp(np.minimum(ce_with[j, :rl][novel_idx].mean(), 20))))
            ppl_novel_no.append(  float(np.exp(np.minimum(ce_no[j,   :rl][novel_idx].mean(), 20))))
            ppl_shared_with.append(float(np.exp(np.minimum(ce_with[j, :rl][shared_idx].mean(), 20))))
            ppl_shared_no.append(  float(np.exp(np.minimum(ce_no[j,   :rl][shared_idx].mean(), 20))))

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(pairs))
            print(f"  Evaluating: {done}/{len(pairs)}  "
                  f"(novel pairs so far: {len(ppl_novel_with)})")

    return ppl_novel_with, ppl_novel_no, ppl_shared_with, ppl_shared_no


# ── Checkpoint finder ─────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kaggle/working/checkpoints/wiki_model.pt")
    ap.add_argument("--n", type=int, default=500,
                    help="Number of IteraTeR pairs to evaluate")
    args = ap.parse_args()

    print("=" * 70)
    print("DELTA SYSTEM — Insertion Cloze Eval  (Experiment 11)")
    print("Does delta specifically help predict NOVEL tokens in B?")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    print(f"Pairs  : {args.n}")
    print()
    print("Hypothesis:")
    print("  If Novel DELTA_PPL >> Shared DELTA_PPL:")
    print("  delta specifically encodes what B adds beyond A.")
    print()

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ── Load data ──────────────────────────────────────────────────────────────
    pairs = load_pairs(args.n, tok)
    if len(pairs) < 30:
        print(f"ERROR: only {len(pairs)} pairs loaded.")
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

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("Running D_recon with delta (G(A,B)) and without delta (ablation)...")
    print("Splitting results into novel tokens vs shared tokens...")
    print()

    ppl_nov_w, ppl_nov_no, ppl_sh_w, ppl_sh_no = eval_novel_ppl(model, pairs, tok)

    n = len(ppl_nov_w)
    print(f"\n  Valid pairs: {n}")
    print()

    # ── Compute DELTA_PPL per category ─────────────────────────────────────────
    arr_nw  = np.array(ppl_nov_w)
    arr_nno = np.array(ppl_nov_no)
    arr_sw  = np.array(ppl_sh_w)
    arr_sno = np.array(ppl_sh_no)

    novel_delta  = arr_nno  - arr_nw    # positive = delta helps novel tokens
    shared_delta = arr_sno  - arr_sw    # positive = delta helps shared tokens

    mean_novel  = float(novel_delta.mean())
    mean_shared = float(shared_delta.mean())
    ratio       = mean_novel / max(abs(mean_shared), 1e-6)

    # Paired t-test: is the novel improvement significantly larger than shared?
    diff = novel_delta - shared_delta
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)

    print("=" * 70)
    print("RESULTS — Novel-only DELTA_PPL  (Experiment 11)")
    print("=" * 70)
    print()
    print(f"  {'Category':<30} {'PPL with δ':>12} {'PPL no δ':>12} {'DELTA_PPL':>12}")
    print("  " + "-" * 68)
    print(f"  {'Novel tokens (inserted)':<30} "
          f"{arr_nw.mean():>12.2f} {arr_nno.mean():>12.2f} {mean_novel:>+12.2f}")
    print(f"  {'Shared tokens (from A)':<30} "
          f"{arr_sw.mean():>12.2f} {arr_sno.mean():>12.2f} {mean_shared:>+12.2f}")
    print()
    print(f"  Novel / Shared DELTA_PPL ratio : {ratio:.2f}x")
    print(f"  Paired t-test (novel > shared) : t={t_stat:.3f}, p={p_val:.4f}")
    print()

    # ── Interpretation ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if mean_novel > 0 and mean_novel > mean_shared * 1.5 and p_val < 0.05:
        verdict = "STRONG PROOF"
        detail  = (f"Novel DELTA_PPL ({mean_novel:+.1f}) >> "
                   f"Shared DELTA_PPL ({mean_shared:+.1f}) by {ratio:.1f}x, p={p_val:.4f}.\n"
                   f"  Delta specifically reduces perplexity on INSERTED tokens.\n"
                   f"  Delta encodes what B adds beyond A. PROVEN.")
    elif mean_novel > 0 and mean_novel > mean_shared and p_val < 0.10:
        verdict = "MODERATE PROOF"
        detail  = (f"Novel DELTA_PPL ({mean_novel:+.1f}) > "
                   f"Shared DELTA_PPL ({mean_shared:+.1f}), p={p_val:.4f}.\n"
                   f"  Delta helps novel tokens more than shared. Trend is real but modest.")
    elif mean_novel > 0 and mean_shared > 0 and ratio < 1.2:
        verdict = "INCONCLUSIVE"
        detail  = (f"Novel DELTA_PPL ({mean_novel:+.1f}) ≈ "
                   f"Shared DELTA_PPL ({mean_shared:+.1f}) (ratio {ratio:.2f}x).\n"
                   f"  Delta helps reconstruction generally, not specifically novel tokens.\n"
                   f"  Does NOT prove delta specifically encodes what B adds.")
    else:
        verdict = "NEGATIVE"
        detail  = (f"Novel DELTA_PPL ({mean_novel:+.1f}), "
                   f"Shared DELTA_PPL ({mean_shared:+.1f}).\n"
                   f"  Delta does not preferentially help novel tokens.\n"
                   f"  Delta encodes general reconstruction signal, not specific novelty.")

    print(f"  Verdict: {verdict}")
    print(f"  {detail}")
    print()

    # ── Paper headline ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("PAPER HEADLINE (Experiment 11 — Insertion Cloze):")
    print("=" * 70)
    print(f"  Novel  DELTA_PPL (inserted tokens) : {mean_novel:+.2f}")
    print(f"  Shared DELTA_PPL (copied tokens)   : {mean_shared:+.2f}")
    print(f"  Ratio                               : {ratio:.2f}x")
    print(f"  p-value (novel > shared)            : {p_val:.4f}")
    print()
    print("  Full experiment table:")
    print(f"  {'Dataset':<35} {'Metric':<20} {'Value':>8}")
    print("  " + "-" * 65)
    print(f"  {'Wikipedia (same domain)':<35} {'DELTA_PPL':>20} {'  +755':>8}")
    print(f"  {'HotpotQA (cross-dataset)':<35} {'DELTA_PPL':>20} {'  +480':>8}")
    print(f"  {'NewsEdits (cross-domain)':<35} {'DELTA_PPL':>20} {' +1295':>8}")
    print(f"  {'IteraTeR bert_maxsim':<35} {'Token AUC':>20} {'0.948':>8}")
    print(f"  {'VitaminC probe':<35} {'Acc (corrected)':>20} {'~0.797':>8}")
    print(f"  {'IteraTeR novel tokens':<35} {'Novel DELTA_PPL':>20} {mean_novel:>+8.2f}")
    print(f"  {'IteraTeR shared tokens':<35} {'Shared DELTA_PPL':>20} {mean_shared:>+8.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
