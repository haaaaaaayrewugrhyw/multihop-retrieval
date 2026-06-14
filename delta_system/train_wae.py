"""
train_wae.py -- DATASET experiment: train the BASELINE G on SCATTERED edits.

The only change vs the baseline run is the TRAINING DATA:
  baseline  : Wikipedia consecutive sentences, B = A + " " + next   (APPEND-ONLY)
  this      : WikiAtomicEdits, A = base_sentence, B = edited_sentence (insertion is
              SCATTERED inside the sentence, not appended)

Hypothesis: append-only training lets the decoder reconstruct novel tokens from
left-context ("just continue A") so delta is never forced to carry novelty. Scattered
edits remove that shortcut naturally (mid-sentence insertions have weak left-context),
forcing delta to be the source of novel content -> delta should SPECIALIZE.

PURE: no novelty labels, no oracle, no architecture/objective change. Same model, same
losses, same hyperparameters as baseline. Only the data differs. Preserves the original
idea + claim.

Saves to wiki_model_wae.pt (baseline wiki_model.pt untouched).
Success: insertion_cloze_eval.py --ckpt wiki_model_wae.pt -> specialization gap
(novel vs shared full-effect; baseline = -0.14) flips toward/positive.

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

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def _stream_wae():
    """Robustly open WikiAtomicEdits English insertions (several load strategies)."""
    configs = [
        ("google-research-datasets/wiki_atomic_edits", "english_insertions"),
        ("google-research-datasets/wiki_atomic_edits", None),
    ]
    for repo, cfg in configs:
        for trust in (True, False):
            kwargs = dict(split="train", streaming=True)
            if cfg:
                kwargs["name"] = cfg
            if trust:
                kwargs["trust_remote_code"] = True
            try:
                ds = load_dataset(repo, **kwargs)
                print(f"  opened {repo} cfg={cfg} trust={trust}")
                return ds
            except Exception as e:
                print(f"  attempt failed ({repo}, cfg={cfg}, trust={trust}): {type(e).__name__}")
    raise RuntimeError(
        "WikiAtomicEdits could not be loaded. Do NOT fall back to IteraTeR "
        "(that is the eval set -> leakage). Report this and we'll pick a disjoint "
        "scattered source.")


def load_wae_pairs(n_train, min_insert=3, max_insert_ratio=0.5):
    ds = _stream_wae()
    pairs, checked = [], 0
    for ex in ds:
        checked += 1
        base   = (ex.get("base_sentence")   or "").strip()
        edited = (ex.get("edited_sentence") or "").strip()
        if len(base) < 20 or len(edited) < 20:
            continue
        if len(edited) <= len(base):
            continue
        insert_len = len(edited) - len(base)
        if insert_len < min_insert:
            continue
        if insert_len / max(len(edited), 1) > max_insert_ratio:
            continue
        pairs.append({"A": base, "B": edited})
        if len(pairs) >= n_train:
            break
        if checked % 20000 == 0:
            print(f"  checked {checked:,} | collected {len(pairs)}/{n_train}")
    print(f"Loaded {len(pairs)} scattered-edit pairs from {checked:,} examples")
    if len(pairs) < n_train * 0.5:
        raise RuntimeError(f"Only {len(pairs)} pairs — WikiAtomicEdits stream too short.")
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

    print(f"Device: {DEVICE} | steps={args.steps} | SOURCE = WikiAtomicEdits (scattered)")
    print("Baseline architecture + losses; ONLY the training data differs.")
    pairs = load_wae_pairs(args.n_train)

    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = DeltaSystem().to(DEVICE)                      # baseline: vib=False, no ortho
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
