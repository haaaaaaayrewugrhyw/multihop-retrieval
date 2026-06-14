"""
train_ortho.py -- Anti-collapse probe: train wiki G with an orthogonality loss.

Adds L_ortho = mean_t max_j |cos(delta[t], H_A[j])| to the standard recipe to push
delta orthogonal to A -> force it to encode B-BEYOND-A instead of just copying B
(the complement collapse confirmed by the eval's A-dependence = 6%).

Saves to a SEPARATE checkpoint (wiki_model_ortho.pt) so the baseline wiki_model.pt
stays intact for a clean A/B comparison.

Success metric: re-run insertion_cloze_eval.py on this checkpoint and check whether
the A-dependence fraction rises from ~6%.

Run:
    python train_ortho.py --steps 2000 --lam_ortho 1.0
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
from losses import recon_loss, sparsity_loss, specificity_loss, ortho_loss

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def load_wiki_pairs(n_train):
    """Same source/recipe as the baseline notebook: consecutive Wikipedia sentences,
    B = A + ' ' + next sentence (append-only)."""
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    pairs = []
    for ex in ds:
        sents = [s.strip() for s in ex["text"].split("\n") if len(s.strip()) > 80]
        for i in range(len(sents) - 1):
            A, novel = sents[i], sents[i + 1]
            if len(A) > 1500 or len(novel) > 1500:
                continue
            pairs.append({"A": A, "B": A + " " + novel})
            if len(pairs) >= n_train:
                break
        if len(pairs) >= n_train:
            break
        if len(pairs) % 1000 == 0 and len(pairs) > 0:
            print(f"  loaded {len(pairs)}/{n_train}")
    return pairs[:n_train]


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
    ap.add_argument("--n_train",     type=int,   default=8000)
    ap.add_argument("--steps",       type=int,   default=2000)
    ap.add_argument("--bs",          type=int,   default=16)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--lam_s",       type=float, default=1.0)
    ap.add_argument("--lam_spec",    type=float, default=1.0)
    ap.add_argument("--margin",      type=float, default=2.0)
    ap.add_argument("--lam_ortho",   type=float, default=1.0,
                    help="weight on the orthogonality (anti-collapse) loss")
    ap.add_argument("--warmup_frac", type=float, default=0.5,
                    help="ramp lam_ortho 0 -> lam_ortho over this fraction of steps")
    ap.add_argument("--out", default="/kaggle/working/checkpoints/wiki_model_ortho.pt")
    args = ap.parse_args()

    print(f"Device: {DEVICE} | lam_ortho={args.lam_ortho} | steps={args.steps}")
    print("Loading Wikipedia pairs (append-only, same as baseline)...")
    pairs = load_wiki_pairs(args.n_train)
    print(f"Train pairs: {len(pairs)}")

    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = DeltaSystem().to(DEVICE)
    dl    = DataLoader(PairDS(pairs), batch_size=args.bs, shuffle=True,
                       collate_fn=make_col(tok), num_workers=2, pin_memory=True)
    opt   = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    warmup = max(1, int(args.steps * args.warmup_frac))
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
            L_o = ortho_loss(delta, H_A, A_mask, B_mask)

            L_spec = torch.tensor(0.0, device=DEVICE)
            if b > 1:
                idx = list(range(1, b)) + [0]
                lw  = model.reconstruct(H_A, A_mask, delta[idx], delta_0[idx], B_ids, B_mask)
                L_spec = specificity_loss(logits, lw, B_ids, B_mask, margin=args.margin)

            lam_o_t = args.lam_ortho * min(1.0, step / warmup)
            loss = L_r + args.lam_s * L_s + args.lam_spec * L_spec + lam_o_t * L_o

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1

            if step % 200 == 0 or step == 1:
                print(f"  step {step:4d}/{args.steps} | "
                      f"ppl={math.exp(min(L_r.item(), 20)):.1f} | "
                      f"ortho={L_o.item():.3f} (lam={lam_o_t:.2f}) | "
                      f"spec={L_spec.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("bert.")}
    torch.save(trainable, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
