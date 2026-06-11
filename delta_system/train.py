"""
train.py -- Validation experiment: Phase 1 training only.

Losses: L_recon + lam_s * L_sparsity  (no D_gan, no specificity)
Goal  : confirm delta contributes to reconstruction before full training.

Usage:
    python train.py                        # 100 examples, 500 steps
    python train.py --n 100 --steps 500
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
from data   import load_pairs
from model  import DeltaSystem, MAX_SEQ
from losses import recon_loss, sparsity_loss, gate_loss, specificity_loss

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


class PairDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]["A"], self.pairs[i]["B"]


def make_collate(tok: BertTokenizerFast):
    def collate(batch):
        A_texts = [x[0] for x in batch]
        B_texts = [x[1] for x in batch]
        eA = tok(A_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok(B_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        return (eA["input_ids"], eA["attention_mask"],
                eB["input_ids"], eB["attention_mask"])
    return collate


def train(args, pairs=None):
    if pairs is None:
        pairs = load_pairs(max_examples=args.n)
    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ds = PairDS(pairs)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=make_collate(tok), num_workers=0)

    model = DeltaSystem().to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE} | Trainable params: {n_train/1e6:.1f}M")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    model.train()
    step = 0
    # Beta warmup: ramp gate penalty 0 -> beta_gate over first half of training
    # Decoder first learns to USE delta, THEN gate is squeezed to localize
    warmup_steps = max(1, args.steps // 2)

    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break

            A_ids, A_mask, B_ids, B_mask = [t.to(DEVICE) for t in batch]

            b = A_ids.size(0)
            logits, delta, delta_0, H_A, alpha = model(A_ids, A_mask, B_ids, B_mask)

            L_r = recon_loss(logits, B_ids, B_mask)
            L_s = sparsity_loss(delta, B_mask)

            # Gate loss: force alpha (gate) sparse — ramped up after warmup
            beta_t = args.beta_gate * min(1.0, step / warmup_steps)
            L_g = gate_loss(alpha, B_mask)

            # L_specificity: circular-shift delta within batch
            L_spec = torch.tensor(0.0, device=DEVICE)
            if args.lam_spec > 0 and b > 1:
                idx_shift = list(range(1, b)) + [0]
                d_wrong  = delta[idx_shift]
                d0_wrong = delta_0[idx_shift]
                logits_wrong = model.reconstruct(
                    H_A, A_mask, d_wrong, d0_wrong, B_ids, B_mask
                )
                L_spec = specificity_loss(logits, logits_wrong, B_ids, B_mask,
                                          margin=args.margin)

            loss = L_r + args.lam_s * L_s + beta_t * L_g + args.lam_spec * L_spec

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                ppl = math.exp(min(L_r.item(), 20))
                print(f"  step {step:4d} | "
                      f"L_recon={L_r.item():.4f}  ppl={ppl:.1f} | "
                      f"delta_norm={alpha.mean().item():.3f} | "
                      f"L_spec={L_spec.item():.4f}")

    # Save only trainable parameters (excludes frozen BERT ~110M params)
    # This reduces checkpoint from ~700MB to ~250MB
    trainable_state = {k: v for k, v in model.state_dict().items()
                       if not k.startswith("bert.")}
    ckpt_path = Path(__file__).parent / "checkpoints" / "val_model.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    try:
        torch.save(trainable_state, ckpt_path)
        print(f"\nSaved: {ckpt_path}")
    except Exception as e:
        print(f"\nWarning: checkpoint save failed ({e}). Continuing with in-memory model.")
    return model, tok, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",         type=int,   default=100)
    ap.add_argument("--steps",     type=int,   default=500)
    ap.add_argument("--bs",        type=int,   default=8)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--lam_s",     type=float, default=0.1,
                    help="sparsity loss weight")
    ap.add_argument("--lam_spec",  type=float, default=0.0,
                    help="specificity loss weight (0 = disabled)")
    ap.add_argument("--margin",    type=float, default=0.5,
                    help="margin for specificity ranking loss")
    ap.add_argument("--beta_gate", type=float, default=0.3,
                    help="gate sparsity weight (ramped up with warmup)")
    ap.add_argument("--log_every", type=int,   default=50)
    return train(ap.parse_args())


if __name__ == "__main__":
    main()
