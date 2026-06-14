"""
train_wae.py -- DATASET experiment: train the BASELINE G on SCATTERED edits.

The only change vs the baseline run is the TRAINING DATA:
  baseline : Wikipedia consecutive sentences, B = A + " " + next  (APPEND-ONLY)
  this     : sentence-level revision pairs where the edit is SCATTERED inside the
             sentence (not appended): A = before, B = after.

Hypothesis: append-only training lets the decoder reconstruct novel tokens from
left-context ("just continue A"), so delta is never forced to carry novelty. Scattered
mid-sentence edits remove that shortcut naturally, forcing delta to be the source of
novel content -> delta should SPECIALIZE (help NOVEL >> SHARED).

PURE: no novelty labels in the objective, no oracle, no architecture/objective change.
Same model, same losses, same hyperparameters as baseline. Only the data differs.

Source: WikiAtomicEdits is script-based and no longer loadable, so we use IteraTeR_full_sent
(large, sentence-level, scattered) with ParaRev as fallback. To avoid leakage with the
IteraTeR_human_sent EVAL set, we EXCLUDE the exact eval pairs from training (dedup via the
eval's own loader).

Saves to wiki_model_wae.pt (baseline wiki_model.pt untouched).

Run:
    python train_wae.py --steps 2000
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model  import DeltaSystem
from losses import recon_loss, sparsity_loss, specificity_loss
from insertion_cloze_eval import load_pairs as load_eval_pairs   # for leakage dedup

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# Scattered-edit sources (standard parquet — no loading script). Tried in order.
SOURCES = [
    "wanyu/IteraTeR_full_sent",
    "taln-ls2n/pararev",
]
# Candidate (before, after) column name pairs for auto-detection.
COL_PAIRS = [
    ("before_sent", "after_sent"), ("before", "after"),
    ("source", "target"), ("src", "tgt"), ("old", "new"),
    ("original", "revised"), ("previous_text", "new_text"),
    ("text_before", "text_after"), ("old_text", "new_text"),
]


def _detect_cols(ex: dict):
    for a, b in COL_PAIRS:
        if a in ex and b in ex and isinstance(ex[a], str) and isinstance(ex[b], str):
            return a, b
    return None, None


def _open_source():
    last_keys = None
    for repo in SOURCES:
        try:
            ds    = load_dataset(repo, split="train", streaming=True)
            first = next(iter(ds))
            a, b  = _detect_cols(first)
            if a:
                print(f"  using {repo}  cols=({a}, {b})")
                ds = load_dataset(repo, split="train", streaming=True)  # reopen
                return repo, ds, a, b
            last_keys = list(first.keys())
            print(f"  {repo}: no before/after cols; keys={last_keys}")
        except Exception as e:
            print(f"  {repo} failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"No scattered source loaded. Last keys seen: {last_keys}")


def load_scattered_pairs(n_train, exclude: set):
    repo, ds, ca, cb = _open_source()
    pairs, checked, skipped_leak = [], 0, 0
    for ex in ds:
        checked += 1
        A = (ex.get(ca) or "").strip()
        B = (ex.get(cb) or "").strip()
        if len(A) < 20 or len(B) < 20 or len(A) > 1500 or len(B) > 1500:
            continue
        if A == B:
            continue
        # meaning-changed only, if the dataset carries labels (matches the eval)
        labels = ex.get("labels")
        if labels is not None:
            labs = labels if isinstance(labels, list) else [labels]
            if not any("meaning" in str(l).lower() for l in labs):
                continue
        if B in exclude or A in exclude:        # leakage guard vs eval set
            skipped_leak += 1
            continue
        pairs.append({"A": A, "B": B})
        if len(pairs) >= n_train:
            break
        if checked % 20000 == 0:
            print(f"  checked {checked:,} | collected {len(pairs)}/{n_train} "
                  f"| leak-skipped {skipped_leak}")
    print(f"Loaded {len(pairs)} scattered pairs from {repo} "
          f"({checked:,} examples scanned, {skipped_leak} eval-overlap skipped)")
    if len(pairs) < n_train * 0.5:
        raise RuntimeError(f"Only {len(pairs)} pairs — source too short / over-filtered.")
    return pairs


class PairDS(Dataset):
    def __init__(self, p): self.p = p
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.p[i]["A"], self.p[i]["B"]


def make_col(tok):
    def col(batch):
        eA = tok([x[0] for x in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([x[1] for x in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        return eA["input_ids"], eA["attention_mask"], eB["input_ids"], eB["attention_mask"]
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train",  type=int,   default=8000)
    ap.add_argument("--steps",    type=int,   default=2000)
    ap.add_argument("--bs",       type=int,   default=16)
    ap.add_argument("--lr",       type=float, default=1e-4)
    ap.add_argument("--lam_s",    type=float, default=1.0)
    ap.add_argument("--lam_spec", type=float, default=1.0)
    ap.add_argument("--margin",   type=float, default=2.0)
    ap.add_argument("--out", default="/kaggle/working/checkpoints/wiki_model_wae.pt")
    args = ap.parse_args()

    print(f"Device: {DEVICE} | steps={args.steps} | SOURCE = scattered edits")
    print("Baseline architecture + losses; ONLY the training data differs.")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Build leakage-exclusion set = exact eval pairs (load a buffer > the 500 used)
    print("Loading eval pairs to build leakage-exclusion set...")
    eval_pairs = load_eval_pairs(700, tok)
    exclude = {p["B"].strip() for p in eval_pairs} | {p["A"].strip() for p in eval_pairs}
    print(f"Excluding {len(exclude)} eval texts from training.")

    pairs = load_scattered_pairs(args.n_train, exclude)

    model = DeltaSystem().to(DEVICE)                       # baseline: vib=False, no ortho
    dl    = DataLoader(PairDS(pairs), batch_size=args.bs, shuffle=True,
                       collate_fn=make_col(tok), num_workers=2, pin_memory=True)
    opt   = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    step = 0
    while step < args.steps:
        for A_ids, A_mask, B_ids, B_mask in dl:
            if step >= args.steps:
                break
            A_ids, A_mask, B_ids, B_mask = [t.to(DEVICE) for t in (A_ids, A_mask, B_ids, B_mask)]
            b = A_ids.size(0)

            logits, delta, delta_0, H_A, _ = model(A_ids, A_mask, B_ids, B_mask)
            L_r = recon_loss(logits, B_ids, B_mask)
            L_s = sparsity_loss(delta, B_mask)

            L_spec = torch.tensor(0.0, device=DEVICE)
            if b > 1:
                idx = list(range(1, b)) + [0]
                lw  = model.reconstruct(H_A, A_mask, delta[idx], delta_0[idx], B_ids, B_mask)
                L_spec = specificity_loss(logits, lw, B_ids, B_mask, margin=args.margin)

            loss = L_r + args.lam_s * L_s + args.lam_spec * L_spec
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1

            if step % 200 == 0 or step == 1:
                print(f"  step {step:4d}/{args.steps} | "
                      f"ppl={math.exp(min(L_r.item(), 20)):.1f} | "
                      f"spec={L_spec.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("bert.")}
    torch.save(trainable, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
