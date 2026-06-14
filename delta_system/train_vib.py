"""
train_vib.py -- SOFT bottleneck probe: train wiki G with a Variational Information
Bottleneck (VIB) on delta.

delta becomes stochastic (delta = mu + sigma*eps) and we add a KL penalty
KL(N(mu,sigma) || N(0,1)). This penalizes the BITS delta carries (not its magnitude).
Since the decoder already has all of A for free, the cheapest way to keep reconstruction
low under a bit-budget is to encode only what A lacks -> the novelty. ("Starve delta.")

This is the architecture-PRESERVING test: the token-level [T x 768] delta stays; we only
add mu/logvar heads + the KL penalty. No novelty labels -> stays pure self-supervised.

Saves to a SEPARATE checkpoint (wiki_model_vib.pt). Baseline wiki_model.pt untouched.

Success metric: re-run insertion_cloze_eval.py --vib on this checkpoint and check the
SPECIALIZATION gap (novel vs shared full-effect; baseline = -0.14) flipping positive.

beta is the key knob. If KL barely moves -> beta too low. If ppl explodes / delta -> 0
-> beta too high. Watch the 'kl=' value in the logs.

Run:
    python train_vib.py --steps 2000 --beta 0.1
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
from losses import recon_loss, sparsity_loss, specificity_loss, kl_loss

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def load_wiki_pairs(n_train):
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
    ap.add_argument("--beta",        type=float, default=0.1,
                    help="VIB KL weight (the key knob)")
    ap.add_argument("--warmup_frac", type=float, default=0.5,
                    help="ramp beta 0 -> beta over this fraction of steps")
    ap.add_argument("--out", default="/kaggle/working/checkpoints/wiki_model_vib.pt")
    args = ap.parse_args()

    print(f"Device: {DEVICE} | VIB beta={args.beta} | steps={args.steps}")
    print("Loading Wikipedia pairs (append-only, same as baseline)...")
    pairs = load_wiki_pairs(args.n_train)
    print(f"Train pairs: {len(pairs)}")

    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = DeltaSystem(vib=True).to(DEVICE)
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
            L_r  = recon_loss(logits, B_ids, B_mask)
            L_s  = sparsity_loss(delta, B_mask)
            L_kl = kl_loss(model.last_kl, B_mask)

            L_spec = torch.tensor(0.0, device=DEVICE)
            if b > 1:
                idx = list(range(1, b)) + [0]
                lw  = model.reconstruct(H_A, A_mask, delta[idx], delta_0[idx], B_ids, B_mask)
                L_spec = specificity_loss(logits, lw, B_ids, B_mask, margin=args.margin)

            beta_t = args.beta * min(1.0, step / warmup)
            loss = L_r + args.lam_s * L_s + args.lam_spec * L_spec + beta_t * L_kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1

            if step % 200 == 0 or step == 1:
                print(f"  step {step:4d}/{args.steps} | "
                      f"ppl={math.exp(min(L_r.item(), 20)):.1f} | "
                      f"kl={L_kl.item():.2f} (beta={beta_t:.3f}) | "
                      f"spec={L_spec.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("bert.")}
    torch.save(trainable, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
