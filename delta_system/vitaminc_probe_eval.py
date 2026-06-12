"""
vitaminc_probe_eval.py -- VitaminC linear probe: does delta encode factual change direction?

VitaminC (Schuster et al., NAACL 2021, arXiv 2103.08541) contains Wikipedia revision
pairs where a SINGLE minimal factual change in the evidence flips whether a claim is
SUPPORTED or REFUTED.

Experiment (Experiment 10):
  1. Load VitaminC pairs: A = evidence that SUPPORTS claim,
                          B = evidence that REFUTES same claim
  2. Compute delta = G(A, B) using trained delta_system (no fine-tuning, no labels)
  3. Train a FROZEN linear probe (logistic regression) on delta[:, 0] (CLS token)
     to predict SUPPORTS / REFUTES
  4. Baselines: probe on BERT(B) alone, probe on BERT(B) - BERT(A) difference

Key insight: delta is computed from BOTH A and B, but the probe input is ONLY delta.
If a linear function of delta predicts the label, delta encodes the direction of
factual change — it is not just a reconstruction artifact.

Selectivity = accuracy_probe - accuracy_control (probe on shuffled labels).
Selectivity > 0.10 = meaningful signal beyond memorization.

Dataset : tals/vitaminc (HuggingFace, 488K Wikipedia revision pairs)
Checkpoint: Wikipedia-trained wiki_model.pt (same as all prior experiments)

Run:
    python vitaminc_probe_eval.py --ckpt /path/to/wiki_model.pt --n 2000
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import DeltaSystem

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

LABEL2INT = {"SUPPORTS": 0, "REFUTES": 1}
INT2LABEL = {v: k for k, v in LABEL2INT.items()}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_vitaminc_pairs(n_pairs: int):
    """
    Load VitaminC from HuggingFace and build contrast pairs.

    VitaminC has claims with multiple evidence passages (Wikipedia revisions).
    We find claims where the same claim appears with both SUPPORTS and REFUTES
    evidence — the passages differ by a single factual change.

    Pair construction:
      A = evidence that SUPPORTS the claim  (label 0)
      B = evidence that REFUTES  the claim  (label 1)
    Then we add the reverse pair (B as A, A as B) with flipped label.
    This ensures label balance and probes both directions of change.

    Returns: list of dicts {A, B, claim, label}
    """
    print("Loading tals/vitaminc (streaming)...")
    try:
        ds = load_dataset("tals/vitaminc", split="train", streaming=True)
    except Exception as e:
        print(f"ERROR: cannot load tals/vitaminc: {e}")
        sys.exit(1)

    # Peek at structure
    first = next(iter(ds))
    print(f"  Dataset features: {list(first.keys())}")
    print(f"  Example label   : {first.get('label')}")
    print(f"  Example evidence: {str(first.get('evidence', ''))[:80]!r}")
    print()

    # Re-stream after peeking
    ds = load_dataset("tals/vitaminc", split="train", streaming=True)

    by_claim = defaultdict(lambda: {"SUPPORTS": [], "REFUTES": []})
    n_scanned = 0
    n_found   = 0

    for ex in ds:
        claim    = (ex.get("claim")    or "").strip()
        evidence = (ex.get("evidence") or "").strip()
        label    = (ex.get("label")    or "").strip().upper()

        if not claim or not evidence or label not in LABEL2INT:
            n_scanned += 1
            continue
        if len(evidence) < 30:
            n_scanned += 1
            continue

        by_claim[claim][label].append(evidence)
        n_scanned += 1

        if n_scanned % 20000 == 0:
            n_found = sum(
                1 for v in by_claim.values()
                if v["SUPPORTS"] and v["REFUTES"]
            )
            print(f"  Scanned {n_scanned:,} | contrast-pair claims: {n_found:,}")
            # Each claim contributes 2 pairs (forward + reverse)
            if n_found >= n_pairs:
                break

        if n_scanned >= 300_000:
            break

    print(f"  Total scanned: {n_scanned:,}")

    pairs = []
    for claim, evs in by_claim.items():
        if not evs["SUPPORTS"] or not evs["REFUTES"]:
            continue

        A = evs["SUPPORTS"][0]
        B = evs["REFUTES"][0]

        if A == B or len(A) < 30 or len(B) < 30:
            continue

        # Forward: G(A_sup → B_ref). Delta captures A_sup's factual content
        # (the supporting fact), so probe sees a SUPPORTS signal → label 0.
        pairs.append({"A": A, "B": B, "claim": claim, "label": 0})
        # Reverse: G(A_ref → B_sup). Delta captures A_ref's factual content
        # (the refuting fact), so probe sees a REFUTES signal → label 1.
        pairs.append({"A": B, "B": A, "claim": claim, "label": 1})

        if len(pairs) >= n_pairs:
            break

    print(f"  Built {len(pairs)} pairs  "
          f"(SUPPORTS={sum(p['label']==0 for p in pairs)}, "
          f"REFUTES={sum(p['label']==1 for p in pairs)})")
    return pairs


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_pairs(model: DeltaSystem, pairs: list,
                 tok: BertTokenizerFast, batch_size: int = 16):
    """
    For each pair compute three representations:
      delta_vec : delta[:, 0] — CLS token of G(A,B) output  [d_delta]
      h_b_vec   : H_B[:, 0]  — BERT CLS of B alone          [768]
      diff_vec  : H_B[:,0] - H_A[:,0] — embedding difference [768]

    The probe takes only these vectors as input — it never sees A or B directly.
    """
    model.eval()
    delta_vecs, h_b_vecs, diff_vecs = [], [], []

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

        delta, _, _ = model.generate_delta(H_A, A_mask, H_B, B_mask)  # [b, T, d]

        delta_vecs.append(delta[:, 0, :].cpu().numpy())         # CLS of delta
        h_b_vecs.append(H_B[:, 0, :].cpu().numpy())            # CLS of B
        diff_vecs.append((H_B[:, 0, :] - H_A[:, 0, :]).cpu().numpy())  # B-A

        if (i // batch_size) % 15 == 0:
            done = min(i + batch_size, len(pairs))
            print(f"  Encoding: {done}/{len(pairs)}")

    return (np.vstack(delta_vecs),
            np.vstack(h_b_vecs),
            np.vstack(diff_vecs))


# ── Probe ─────────────────────────────────────────────────────────────────────

def run_probe(X_tr, X_te, y_tr, y_te, name: str,
              y_shuffled: np.ndarray = None) -> dict:
    """
    Fit logistic regression probe on X_tr → y_tr.
    Evaluate on X_te. Report accuracy, macro-F1, selectivity.

    Selectivity = accuracy - control_accuracy (same probe with shuffled labels).
    Selectivity > 0.10: meaningful semantic content in the representation.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    probe = LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
    )
    probe.fit(X_tr_s, y_tr)
    y_pred = probe.predict(X_te_s)

    acc = float(accuracy_score(y_te, y_pred))
    f1  = float(f1_score(y_te, y_pred, average="macro", zero_division=0))

    # Selectivity: probe on shuffled labels (same X)
    sel = float("nan")
    if y_shuffled is not None:
        ctrl = LogisticRegression(
            max_iter=500, C=1.0, random_state=42, solver="lbfgs"
        )
        ctrl.fit(X_tr_s, y_shuffled)
        ctrl_acc = float(accuracy_score(y_te, ctrl.predict(X_te_s)))
        sel = acc - ctrl_acc

    sel_str = f"{sel:+.4f}" if not np.isnan(sel) else "  N/A"
    print(f"  {name:<40} Acc={acc:.4f}  F1={f1:.4f}  Sel={sel_str}")
    return {"name": name, "acc": acc, "f1": f1, "selectivity": sel}


# ── Qualitative examples ──────────────────────────────────────────────────────

def show_examples(pairs, n_show: int = 4):
    """Print a few pairs to show what A vs B actually looks like."""
    print()
    print("EXAMPLE CONTRAST PAIRS")
    print("-" * 70)
    for i, p in enumerate(pairs[:n_show]):
        lbl = INT2LABEL[p["label"]]
        print(f"  Example {i+1} — B label: {lbl}")
        print(f"  Claim : {p['claim'][:100]}")
        print(f"  A     : {p['A'][:100]}")
        print(f"  B     : {p['B'][:100]}")
        print()


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
    ap.add_argument("--n", type=int, default=2000,
                    help="Total contrast pairs to use (balanced: N/2 each direction)")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction for probe test set")
    args = ap.parse_args()

    print("=" * 70)
    print("DELTA SYSTEM — VitaminC Linear Probe  (Experiment 10)")
    print("Can a frozen probe predict SUPPORTS/REFUTES from delta alone?")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Pairs      : {args.n}")
    print(f"Test split : {args.test_size:.0%}")
    print()

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ── Load data ──────────────────────────────────────────────────────────────
    pairs = load_vitaminc_pairs(args.n)
    if len(pairs) < 50:
        print(f"ERROR: only {len(pairs)} pairs collected. Lower --n.")
        sys.exit(1)
    print()

    show_examples(pairs)

    labels = np.array([p["label"] for p in pairs])
    chance = float(max((labels == 0).mean(), (labels == 1).mean()))
    print(f"Chance (majority class): {chance:.4f}")
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

    # ── Encode ─────────────────────────────────────────────────────────────────
    print("Encoding pairs (BERT + delta_system)...")
    delta_vecs, h_b_vecs, diff_vecs = encode_pairs(model, pairs, tok)
    print(f"  delta_vec shape : {delta_vecs.shape}")
    print(f"  h_b_vec  shape  : {h_b_vecs.shape}")
    print()

    # ── Train/test split ───────────────────────────────────────────────────────
    idx = np.arange(len(pairs))
    idx_tr, idx_te = train_test_split(
        idx, test_size=args.test_size, random_state=42, stratify=labels)
    y_tr, y_te = labels[idx_tr], labels[idx_te]

    rng = np.random.default_rng(99)
    y_shuffled = y_tr.copy()
    rng.shuffle(y_shuffled)

    # ── Probes ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PROBE RESULTS  (linear logistic regression, frozen representation)")
    print("=" * 70)
    print(f"  Train: {len(idx_tr)} | Test: {len(idx_te)}")
    print(f"  {'Method':<40} {'Acc':>8} {'F1':>8} {'Sel':>8}")
    print("  " + "-" * 70)

    results = []

    results.append(run_probe(
        delta_vecs[idx_tr], delta_vecs[idx_te], y_tr, y_te,
        "delta_system  G(A,B) CLS  [no fine-tune]",
        y_shuffled))

    results.append(run_probe(
        h_b_vecs[idx_tr], h_b_vecs[idx_te], y_tr, y_te,
        "BERT(B) alone  CLS  [baseline]",
        y_shuffled))

    results.append(run_probe(
        diff_vecs[idx_tr], diff_vecs[idx_te], y_tr, y_te,
        "BERT(B) - BERT(A)  [difference baseline]",
        y_shuffled))

    print(f"  {'chance (majority class)':<40} {chance:>8.4f}")
    print()

    # ── Confusion matrix for delta ──────────────────────────────────────────────
    scaler_d = StandardScaler()
    probe_d  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe_d.fit(scaler_d.fit_transform(delta_vecs[idx_tr]), y_tr)
    y_pred_d = probe_d.predict(scaler_d.transform(delta_vecs[idx_te]))

    cm = confusion_matrix(y_te, y_pred_d, labels=[0, 1])
    print("Confusion matrix — delta_system probe:")
    print(f"  {'':15} pred SUPPORTS  pred REFUTES")
    print(f"  {'true SUPPORTS':15}  {cm[0,0]:>10}     {cm[0,1]:>10}")
    print(f"  {'true REFUTES':15}  {cm[1,0]:>10}     {cm[1,1]:>10}")
    print()

    # ── Interpretation ─────────────────────────────────────────────────────────
    delta_r = results[0]
    b_r     = results[1]
    diff_r  = results[2]

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  delta_system  Acc={delta_r['acc']:.4f}  Sel={delta_r['selectivity']:+.4f}")
    print(f"  B alone       Acc={b_r['acc']:.4f}  Sel={b_r['selectivity']:+.4f}")
    print(f"  B-A diff      Acc={diff_r['acc']:.4f}  Sel={diff_r['selectivity']:+.4f}")
    print(f"  Chance        Acc={chance:.4f}")
    print()

    sel = delta_r["selectivity"]
    gain_over_b = delta_r["acc"] - b_r["acc"]
    gain_over_chance = delta_r["acc"] - chance

    if sel > 0.10:
        verdict = "STRONG: delta encodes factual change direction (selectivity > 0.10)"
    elif sel > 0.05:
        verdict = "MODERATE: delta has real semantic content (selectivity 0.05-0.10)"
    elif sel < -0.10:
        verdict = "INVERTED STRONG: high signal but wrong polarity — check label convention"
    else:
        verdict = "WEAK: delta CLS does not clearly encode SUPPORTS/REFUTES direction"

    print(f"  Verdict: {verdict}")
    print()

    if gain_over_b > 0.02:
        print(f"  delta BEATS B-alone by +{gain_over_b:.4f} Acc")
        print("  -> delta captures relational information that B's embedding alone misses.")
    elif gain_over_b > -0.02:
        print(f"  delta ≈ B-alone (gap {gain_over_b:+.4f})")
        print("  -> delta's semantic content is comparable to B's BERT representation.")
    else:
        print(f"  B-alone BEATS delta by {-gain_over_b:.4f} Acc")
        print("  -> B's raw embedding is more predictive than G(A,B). "
              "Delta is encoding relational change, not B's label-predictive content.")

    print()

    # ── Paper headline ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("PAPER HEADLINE (Experiment 10 — VitaminC Probe):")
    print("=" * 70)
    print(f"  delta_system probe  Acc={delta_r['acc']:.4f}  F1={delta_r['f1']:.4f}"
          f"  Selectivity={delta_r['selectivity']:+.4f}")
    print(f"  B-alone probe       Acc={b_r['acc']:.4f}  F1={b_r['f1']:.4f}"
          f"  Selectivity={b_r['selectivity']:+.4f}")
    print(f"  B-A diff probe      Acc={diff_r['acc']:.4f}  F1={diff_r['f1']:.4f}"
          f"  Selectivity={diff_r['selectivity']:+.4f}")
    print(f"  Chance (binary)     Acc={chance:.4f}")
    print()
    print("  Cross-experiment summary (all prior results):")
    print(f"  {'Wikipedia   DELTA_PPL':<30}: +755   (same domain)")
    print(f"  {'HotpotQA    DELTA_PPL':<30}: +480   (cross-dataset)")
    print(f"  {'NewsEdits   DELTA_PPL':<30}: +1295  (cross-domain, zero-shot)")
    print(f"  {'IteraTeR    bert_maxsim AUC':<30}: 0.948  (token localization)")
    print(f"  {'Llama-3.3-70B F1':<30}: 0.875  (LLM baseline)")
    print(f"  {'VitaminC    delta probe Acc':<30}: {delta_r['acc']:.4f}  (semantic content probe)")
    print("=" * 70)


if __name__ == "__main__":
    main()
