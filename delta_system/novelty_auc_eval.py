"""
novelty_auc_eval.py -- Zero-shot novelty scoring evaluation.

Evaluates whether G's delta norms predict paragraph-level novelty using
vocab_novelty as an independent ground-truth signal that was NEVER seen
during training.

    vocab_novelty(A, novel) = fraction of novel-paragraph tokens that are
                              genuinely new words not present in A at all.

This is independent of reconstruction loss, specificity loss, or any
training signal used in the delta system.

Baseline: TF-IDF cosine distance (1 - cosine_sim(TF-IDF(A), TF-IDF(novel))).
Both our model and TF-IDF are lexically-grounded. If our model outperforms
TF-IDF, it demonstrates that cross-attention learns context-aware novelty
detection beyond bag-of-words overlap.

Run on Kaggle after the Wikipedia training is complete:
    python /kaggle/working/repo/delta_system/novelty_auc_eval.py \\
        --ckpt /kaggle/working/checkpoints/wiki_model.pt

Metrics:
    Spearman rho : monotonic correlation of score vs vocab_novelty
    AUC-ROC      : binary classification at median vocab_novelty split
    Quartiles    : mean(delta.norm()) per novelty quartile (should rise Q1->Q4)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import DeltaSystem

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


# ── Data loading ──────────────────────────────────────────────────────────────

def load_wikipedia_pairs(n=1000, skip=10000):
    """
    Stream Wikipedia paragraphs, skip the first `skip` pairs (training range),
    and return n fresh held-out pairs.

    The Wikipedia training used pairs 0-9000, so skip=10000 ensures no overlap.
    """
    from datasets import load_dataset
    print(f"Streaming Wikipedia (wikimedia/wikipedia 20231101.en), skipping first {skip} pairs...")
    ds    = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    pairs = []
    seen  = 0

    for ex in ds:
        sents = [s.strip() for s in ex["text"].split("\n") if len(s.strip()) > 80]
        if len(sents) < 2:
            continue
        for i in range(len(sents) - 1):
            if seen < skip:
                seen += 1
                continue
            A, novel = sents[i], sents[i + 1]
            if len(A) < 50 or len(novel) < 50:
                continue
            if len(A) > 1500 or len(novel) > 1500:
                continue
            pairs.append({"A": A, "B": A + " " + novel, "novel": novel})
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break

    print(f"Loaded {len(pairs)} fresh evaluation pairs (never used in training)")
    return pairs


# ── Independent ground-truth signal ──────────────────────────────────────────

def vocab_novelty_score(A: str, novel: str) -> float:
    """
    Fraction of novel-paragraph tokens that are genuinely new words not in A.
    Never used during training — fully independent of reconstruction objective.
    """
    a_tokens     = set(A.lower().split())
    novel_tokens = novel.lower().split()
    if not novel_tokens:
        return 0.0
    return sum(1 for t in novel_tokens if t not in a_tokens) / len(novel_tokens)


# ── Model scoring ─────────────────────────────────────────────────────────────

class _PairDS(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self):         return len(self.pairs)
    def __getitem__(self, i):  return self.pairs[i]["A"], self.pairs[i]["B"]


def _make_collate(tok):
    def collate(batch):
        eA = tok([x[0] for x in batch], max_length=MAX_LEN, truncation=True,
                  padding="max_length", return_tensors="pt")
        eB = tok([x[1] for x in batch], max_length=MAX_LEN, truncation=True,
                  padding="max_length", return_tensors="pt")
        return (eA["input_ids"], eA["attention_mask"],
                eB["input_ids"], eB["attention_mask"])
    return collate


@torch.no_grad()
def compute_delta_scores(model, pairs, tok):
    """
    For each (A, B) pair, compute mean(delta.norm()) over non-padding tokens.
    High score = model thinks B has more novel content relative to A.
    """
    dl = DataLoader(_PairDS(pairs), batch_size=16, shuffle=False,
                    collate_fn=_make_collate(tok), num_workers=0)
    model.eval()
    scores = []
    for A_ids, A_mask, B_ids, B_mask in dl:
        A_ids, A_mask = A_ids.to(DEVICE), A_mask.to(DEVICE)
        B_ids, B_mask = B_ids.to(DEVICE), B_mask.to(DEVICE)

        H_A = model._enc(A_ids, A_mask)
        H_B = model._enc(B_ids, B_mask)
        delta, _, _ = model.generate_delta(H_A, A_mask, H_B, B_mask)

        norms = delta.norm(dim=-1)  # [b, T]
        for i in range(A_ids.size(0)):
            n_real = int(B_mask[i].sum().item())
            scores.append(norms[i, :n_real].mean().item())

    return np.array(scores, dtype=np.float32)


# ── TF-IDF baseline ───────────────────────────────────────────────────────────

def compute_tfidf_scores(pairs):
    """
    Baseline: 1 - cosine_sim(TF-IDF(A), TF-IDF(novel)).
    High = more lexically different = more "novel" by surface measure.
    """
    A_texts     = [p["A"]     for p in pairs]
    novel_texts = [p["novel"] for p in pairs]

    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    vect.fit(A_texts + novel_texts)
    A_vecs     = vect.transform(A_texts)
    novel_vecs = vect.transform(novel_texts)

    scores = []
    for i in range(len(pairs)):
        sim = cosine_similarity(A_vecs[i:i+1], novel_vecs[i:i+1])[0][0]
        scores.append(1.0 - float(sim))

    return np.array(scores, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_checkpoint(default_path: str) -> Path:
    """Try several common checkpoint locations, return the first that exists."""
    candidates = [
        default_path,
        "/kaggle/working/checkpoints/wiki_model.pt",
        "/kaggle/working/checkpoints/kaggle_model.pt",
        "/kaggle/working/checkpoints/val_model.pt",
        str(ROOT / "checkpoints" / "wiki_model.pt"),
        str(ROOT / "checkpoints" / "kaggle_model.pt"),
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return Path(default_path)   # return default even if missing (will warn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kaggle/working/checkpoints/wiki_model.pt",
                    help="Path to trained G checkpoint (non-BERT weights only)")
    ap.add_argument("--n",    type=int, default=1000,
                    help="Number of evaluation pairs")
    ap.add_argument("--skip", type=int, default=10000,
                    help="Wikipedia pairs to skip (avoid training overlap)")
    args = ap.parse_args()

    print("=" * 66)
    print("DELTA SYSTEM — Zero-Shot Novelty AUC Evaluation")
    print("=" * 66)
    print(f"Device     : {DEVICE}")
    print(f"Pairs      : {args.n}  (skipping first {args.skip} to avoid training overlap)")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    pairs = load_wikipedia_pairs(args.n, args.skip)
    print()

    # ── Independent ground-truth ──────────────────────────────────────────────
    print("Computing vocab_novelty labels (independent of training signal)...")
    gt = np.array([vocab_novelty_score(p["A"], p["novel"]) for p in pairs])
    print(f"  mean = {gt.mean():.3f}  std = {gt.std():.3f}  "
          f"range = [{gt.min():.3f}, {gt.max():.3f}]")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = DeltaSystem().to(DEVICE)
    ckpt  = _find_checkpoint(args.ckpt)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
        print(f"Checkpoint loaded: {ckpt}")
    else:
        print(f"WARNING: no checkpoint found — evaluating UNTRAINED model (sanity check only)")
    print()

    # ── Compute scores ────────────────────────────────────────────────────────
    print("Running G inference (zero-shot — no training on these novelty labels)...")
    delta_scores = compute_delta_scores(model, pairs, tok)
    print(f"  delta norms : mean={delta_scores.mean():.5f}  std={delta_scores.std():.5f}")
    print()

    print("Computing TF-IDF baseline...")
    tfidf_scores = compute_tfidf_scores(pairs)
    print(f"  tfidf dist  : mean={tfidf_scores.mean():.5f}  std={tfidf_scores.std():.5f}")
    print()

    # ── Metrics ───────────────────────────────────────────────────────────────
    rho_delta, p_delta = spearmanr(delta_scores, gt)
    rho_tfidf, p_tfidf = spearmanr(tfidf_scores, gt)

    median_gt  = np.median(gt)
    labels     = (gt > median_gt).astype(int)
    auc_delta  = roc_auc_score(labels, delta_scores)
    auc_tfidf  = roc_auc_score(labels, tfidf_scores)

    # Quartile breakdown of delta norms
    q_bounds = np.percentile(gt, [25, 50, 75])
    q_labels = np.digitize(gt, q_bounds)       # 0=Q1, 1=Q2, 2=Q3, 3=Q4
    q_means  = [delta_scores[q_labels == q].mean() if (q_labels == q).any() else float("nan")
                for q in range(4)]
    q_counts = [(q_labels == q).sum() for q in range(4)]

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 66)
    print("  ZERO-SHOT NOVELTY AUC RESULTS")
    print("=" * 66)
    print()
    print("  Ground truth: vocab_novelty (new tokens in novel / all novel tokens)")
    print("  Training signal: reconstruction loss only — vocab_novelty NEVER seen")
    print()
    print(f"  {'Metric':<30} {'Our model':>12} {'TF-IDF':>12} {'Random':>8}")
    print(f"  {'-'*64}")
    print(f"  {'Spearman rho':<30} {rho_delta:>+12.4f} {rho_tfidf:>+12.4f} {'0.000':>8}")
    print(f"  {'(p-value)':<30} {p_delta:>12.2e} {p_tfidf:>12.2e} {'—':>8}")
    print(f"  {'AUC-ROC (median split)':<30} {auc_delta:>12.4f} {auc_tfidf:>12.4f} {'0.500':>8}")
    print()
    print("  Quartile breakdown (Q1=least novel -> Q4=most novel):")
    print(f"  {'Quartile':<12} {'Novelty range':>20} {'mean(delta.norm())':>20} {'n':>6}")
    print(f"  {'-'*60}")
    q_ranges = [(gt.min(), q_bounds[0]),
                (q_bounds[0], q_bounds[1]),
                (q_bounds[1], q_bounds[2]),
                (q_bounds[2], gt.max())]
    for q in range(4):
        lo, hi = q_ranges[q]
        print(f"  Q{q+1:<11} {lo:.3f} – {hi:.3f}          {q_means[q]:>20.5f} {q_counts[q]:>6}")

    trend_ok = all(q_means[i] <= q_means[i+1] for i in range(3)
                   if not (np.isnan(q_means[i]) or np.isnan(q_means[i+1])))
    print(f"  Monotone Q1->Q4 : {'YES (delta norms increase with novelty)' if trend_ok else 'NO'}")
    print()

    print("=" * 66)
    print("  INTERPRETATION")
    print("=" * 66)
    gain_rho = rho_delta - rho_tfidf
    gain_auc = auc_delta - auc_tfidf

    if rho_delta > 0.15 and gain_rho > 0.0:
        print(f"  STRONG: delta norms correlate with novelty (rho={rho_delta:+.3f})")
        print(f"          outperforms TF-IDF by rho gain={gain_rho:+.4f}")
    elif rho_delta > 0.05:
        print(f"  MODERATE: positive novelty correlation (rho={rho_delta:+.3f})")
        if gain_rho > 0:
            print(f"            matches or exceeds TF-IDF (gain={gain_rho:+.4f})")
    else:
        print(f"  WEAK: low correlation (rho={rho_delta:+.3f}) — model may need more steps")

    if auc_delta > 0.60:
        print(f"  AUC={auc_delta:.3f}: strong zero-shot novelty detection above TF-IDF baseline")
    elif auc_delta > 0.55:
        print(f"  AUC={auc_delta:.3f}: above-chance zero-shot detection (TF-IDF={auc_tfidf:.3f})")
    elif auc_delta > 0.50:
        print(f"  AUC={auc_delta:.3f}: slight above-random signal")
    else:
        print(f"  AUC={auc_delta:.3f}: near-random — low novelty signal in delta norms")

    print()
    print("  Compare to main experiments:")
    print("    Wikipedia DELTA_PPL   : +755  (held-out, PASS)")
    print("    HotpotQA  DELTA_PPL   : +480  (held-out, PASS)")
    print(f"   This eval  AUC-ROC     : {auc_delta:.3f}  (zero-shot, vs TF-IDF={auc_tfidf:.3f})")
    print("=" * 66)


if __name__ == "__main__":
    main()
