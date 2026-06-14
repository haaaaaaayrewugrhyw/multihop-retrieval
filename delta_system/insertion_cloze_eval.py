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

CONDS = ["full", "none", "no_delta", "no_d0", "shuf", "noA"]
CATS  = ["novel", "shared"]


@torch.no_grad()
def eval_conditions(model: DeltaSystem, pairs: list,
                    tok: BertTokenizerFast, batch_size: int = 8):
    """
    For each pair, compute per-token cross-entropy under FOUR ablation conditions,
    split into novel vs shared tokens:

      full     : delta ON,  delta_0 ON   (the real system)
      none     : delta OFF, delta_0 OFF  (original Exp-11 ablation)
      no_delta : delta OFF, delta_0 ON   (isolate token-delta contribution)
      no_d0    : delta ON,  delta_0 OFF  (isolate delta_0 contribution)
      shuf     : delta from a DIFFERENT pair (circular shift) -- pair-specificity control
      noA      : delta computed with the generator BLIND to A (H_A zeroed), but
                 reconstruction still uses the real A -- tests whether the "beyond A"
                 conditioning is what makes delta useful, vs delta just encoding B.
                 (Off-distribution: G was never trained with H_A=0 -- interpret directionally.)

    Returns:
      pair_ce  : {cat: {cond: [per-pair mean CE in nats]}}   -- macro stats / median / t-test
      pool_sum : {cat: {cond: sum of token CE}}              -- micro (pooled) mean
      pool_cnt : {cat: total token count}
    Per-pair lists are index-aligned across conditions and categories (same pair order),
    so paired tests between conditions/categories are valid.
    """
    model.eval()

    pair_ce  = {c: {cond: [] for cond in CONDS} for c in CATS}
    pool_sum = {c: {cond: 0.0 for cond in CONDS} for c in CATS}
    pool_cnt = {c: 0 for c in CATS}

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

        # Encode + generate delta ONCE, then reconstruct under each condition
        b = A_ids.size(0)
        H_A = model._enc(A_ids, A_mask)
        H_B = model._enc(B_ids, B_mask)
        delta, delta_0, _ = model.generate_delta(H_A, A_mask, H_B, B_mask)

        # Control 1 — shuffled delta (circular shift within batch): wrong pair's delta
        if b > 1:
            sidx = list(range(1, b)) + [0]
            delta_shuf, delta0_shuf = delta[sidx], delta_0[sidx]
        else:
            delta_shuf, delta0_shuf = delta, delta_0

        # Control 2 — generator blind to A: zero H_A only for delta generation
        # (reconstruction below still gets the REAL H_A)
        H_A_zero = torch.zeros_like(H_A)
        delta_noA, delta0_noA, _ = model.generate_delta(H_A_zero, A_mask, H_B, B_mask)

        logits = {
            "full":     model.reconstruct(H_A, A_mask, delta, delta_0, B_ids, B_mask),
            "none":     model.reconstruct(H_A, A_mask, delta, delta_0, B_ids, B_mask, ablate_delta=True),
            "no_delta": model.reconstruct(H_A, A_mask, delta, delta_0, B_ids, B_mask, drop_delta=True),
            "no_d0":    model.reconstruct(H_A, A_mask, delta, delta_0, B_ids, B_mask, drop_d0=True),
            "shuf":     model.reconstruct(H_A, A_mask, delta_shuf, delta0_shuf, B_ids, B_mask),
            "noA":      model.reconstruct(H_A, A_mask, delta_noA,  delta0_noA,  B_ids, B_mask),
        }

        b, T, V = logits["full"].shape
        ce = {}
        for cond in CONDS:
            ce[cond] = F.cross_entropy(
                logits[cond].view(-1, V), B_ids.view(-1),
                reduction="none").view(b, T).cpu().numpy()
        b_mask = B_mask.cpu().numpy()

        for j, p in enumerate(batch):
            rl       = min(p["real_len"], T)
            nov_mask = p["novel_mask"][:rl]        # 1 = novel, 0 = shared
            tok_mask = b_mask[j, :rl].astype(bool)  # 1 = real token
            if rl > 2:                               # exclude CLS + final SEP
                tok_mask[0] = False
                tok_mask[rl - 1] = False

            idx = {"novel": (nov_mask == 1) & tok_mask,
                   "shared": (nov_mask == 0) & tok_mask}
            if idx["novel"].sum() == 0 or idx["shared"].sum() == 0:
                continue

            for c in CATS:
                m = idx[c]
                pool_cnt[c] += int(m.sum())
                for cond in CONDS:
                    vals = ce[cond][j, :rl][m]
                    pair_ce[c][cond].append(float(vals.mean()))   # nats, no exp
                    pool_sum[c][cond] += float(vals.sum())

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(pairs))
            print(f"  Evaluating: {done}/{len(pairs)}  "
                  f"(valid pairs so far: {len(pair_ce['novel']['full'])})")

    return pair_ce, pool_sum, pool_cnt


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
    ap.add_argument("--vib", action="store_true",
                    help="instantiate the VIB variant (for wiki_model_vib.pt)")
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
    model = DeltaSystem(vib=args.vib).to(DEVICE)
    ckpt  = _find_checkpoint(args.ckpt)
    if ckpt.exists():
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE), strict=False)
        print(f"Checkpoint: {ckpt}")
    else:
        print(f"WARNING: no checkpoint at {ckpt} — using random weights")
    print()

    # ── Evaluate (4 ablation conditions) ────────────────────────────────────────
    print("Running 4 conditions: full | none | no_delta | no_d0 ...")
    print("Splitting results into novel tokens vs shared tokens...")
    print()

    pair_ce, pool_sum, pool_cnt = eval_conditions(model, pairs, tok)

    n = len(pair_ce["novel"]["full"])
    print(f"\n  Valid pairs: {n}")
    if n < 5:
        print("ERROR: too few valid pairs to report stats.")
        sys.exit(1)
    print()

    # Per-pair CE arrays (nats), index-aligned across conditions/categories
    pc = {c: {cond: np.array(pair_ce[c][cond]) for cond in CONDS} for c in CATS}
    # Micro (token-pooled) mean CE — robust, not outlier-inflated
    micro = {c: {cond: pool_sum[c][cond] / max(pool_cnt[c], 1) for cond in CONDS}
             for c in CATS}

    def _report_delta(name, base_cond, vs_cond, cat):
        """ΔCE = CE(vs_cond) - CE(base_cond): positive = base_cond is BETTER
        (lower CE), i.e. the removed component HELPS this category."""
        per_pair = pc[cat][vs_cond] - pc[cat][base_cond]    # per-pair ΔCE
        micro_d  = micro[cat][vs_cond] - micro[cat][base_cond]
        mean_d   = float(per_pair.mean())
        med_d    = float(np.median(per_pair))
        t, p     = stats.ttest_1samp(per_pair, 0.0)
        print(f"  {name:<46} ΔCE(micro)={micro_d:+7.4f}  "
              f"mean={mean_d:+7.4f}  median={med_d:+7.4f}  t={t:+6.2f} p={p:.4g}")
        return {"micro": micro_d, "mean": mean_d, "median": med_d, "p": float(p)}

    print("=" * 78)
    print("RESULTS — CE-space (nats). Positive ΔCE = the removed signal HELPS.")
    print("Primary signal = micro ΔCE + median + p-value. (PPL shown for continuity only.)")
    print("=" * 78)
    print()
    print("  Micro mean CE per condition (lower = better reconstruction):")
    print(f"  {'':<10}{'full':>10}{'none':>10}{'no_delta':>10}{'no_d0':>10}")
    for c in CATS:
        print(f"  {c:<10}" + "".join(f"{micro[c][cond]:>10.4f}" for cond in CONDS))
    print()
    print("  Equivalent PPL = exp(micro CE):")
    for c in CATS:
        print(f"  {c:<10}" + "".join(f"{np.exp(min(micro[c][cond],30)):>10.1f}" for cond in CONDS))
    print()

    # ── Full delta-info effect (none - full): the headline question ──────────────
    print("-" * 78)
    print("  [1] FULL delta-info effect  (none - full): does delta+delta_0 help?")
    nov_full = _report_delta("novel:  delta+delta_0 effect",  "full", "none", "novel")
    sh_full  = _report_delta("shared: delta+delta_0 effect",  "full", "none", "shared")
    print()

    # ── 3-way attribution on NOVEL tokens ────────────────────────────────────────
    print("-" * 78)
    print("  [2] Attribution on NOVEL tokens — which component helps/hurts?")
    nov_tokdelta = _report_delta("novel:  token-delta only  (no_delta - full)",
                                 "full", "no_delta", "novel")
    nov_d0       = _report_delta("novel:  delta_0 only      (no_d0 - full)",
                                 "full", "no_d0", "novel")
    print()

    # ── Paired novel-vs-shared (full effect), CE-space ───────────────────────────
    diff = (pc["novel"]["none"] - pc["novel"]["full"]) - \
           (pc["shared"]["none"] - pc["shared"]["full"])
    t_ns, p_ns = stats.ttest_1samp(diff, 0.0)
    print(f"  [3] Paired novel-vs-shared full effect: "
          f"mean Δ={float(diff.mean()):+.4f}  t={t_ns:+.2f}  p={p_ns:.4g}")
    print()

    # ── [4] Controls: is delta pair-specific, and does it need A? ─────────────────
    print("-" * 78)
    print("  [4] CONTROLS on NOVEL tokens — positive ΔCE = the control is WORSE than full")
    nov_shuf = _report_delta("novel:  shuffled-delta penalty  (shuf - full)",
                             "full", "shuf", "novel")
    nov_noA  = _report_delta("novel:  no-A-in-generator penalty (noA - full)",
                             "full", "noA", "novel")
    full_benefit = micro["novel"]["none"] - micro["novel"]["full"]   # CE recovered by full delta
    spec_frac = nov_shuf["micro"] / full_benefit if full_benefit > 1e-6 else float("nan")
    Adep_frac = nov_noA["micro"]  / full_benefit if full_benefit > 1e-6 else float("nan")
    print(f"  full-delta benefit on novel (none-full) = {full_benefit:+.4f} nats")
    print(f"  pair-specific fraction  (shuf penalty / benefit) = {spec_frac:6.1%}  "
          f"(100% = wrong-pair delta is useless -> fully pair-specific)")
    print(f"  A-dependence fraction   (noA  penalty / benefit) = {Adep_frac:6.1%}  "
          f"(100% = delta needs A -> captures B-beyond-A;  ~0% = delta is just B)")
    print()

    # ── Verdict (based on robust CE-space novel effect) ──────────────────────────
    print("=" * 78)
    print("VERDICT  (based on micro ΔCE on NOVEL tokens + significance + controls)")
    print("=" * 78)
    m, p = nov_full["micro"], nov_full["p"]
    if m > 0.02 and p < 0.05:
        v = "POSITIVE — delta-info significantly HELPS novel-token prediction."
    elif m < -0.02 and p < 0.05:
        v = "NEGATIVE — delta-info significantly HURTS novel-token prediction (collapse)."
    else:
        v = "NEUTRAL — no significant delta effect on novel tokens."
    print(f"  {v}")
    print(f"  novel full-effect micro ΔCE = {m:+.4f} (p={p:.4g});  "
          f"shared = {sh_full['micro']:+.4f}")
    print(f"  Attribution: token-delta ΔCE={nov_tokdelta['micro']:+.4f} (p={nov_tokdelta['p']:.3g}) | "
          f"delta_0 ΔCE={nov_d0['micro']:+.4f} (p={nov_d0['p']:.3g})")
    print(f"  Controls: pair-specific={spec_frac:.0%} (shuf p={nov_shuf['p']:.3g}) | "
          f"A-dependence={Adep_frac:.0%} (noA p={nov_noA['p']:.3g})")
    print()
    print("  READS:")
    print("   - pair-specific HIGH  -> delta's novel help is tied to THIS pair (not generic).")
    print("   - A-dependence  HIGH  -> delta captures B-BEYOND-A (needs A); LOW -> delta is just B.")
    print("=" * 78)


if __name__ == "__main__":
    main()
