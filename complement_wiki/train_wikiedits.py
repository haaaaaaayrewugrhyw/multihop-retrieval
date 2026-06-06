"""
train_wikiedits.py -- train the (fixed) ComplementGenerator on WikiAtomicEdits
==============================================================================

Reuses the model AND the training loop from retrieval_v3/generator_train.py.
Only the DATA changes: high-overlap (A=base, B=edited) sentence pairs where
B = A + an inserted phrase.

Purely SELF-SUPERVISED: trains reconstruction of B from A+edge (+ InfoNCE edge
diversity). The gold inserted phrase is NEVER used as a training target -- that
is the whole claim. eval_recovery.py later checks whether the edge captured it.

Watch `collapse_sim`: unlike MuSiQue (stuck ~0.89, no overlap to subtract), here
it SHOULD drop -- on high-overlap data the true complement differs from encode(B).

Usage:
    python train_wikiedits.py --smoke
    python train_wikiedits.py --max_examples 120000
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "retrieval_v3"))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))

from generator_train import (                       # the FIXED model + reusable loop
    ComplementGenerator, train_epoch, validate,
    DEVICE, AMP_ENABLED, WARMUP_FRAC,
)
from data_wikiedits import WikiEditDataset, make_collator, load_triples

# Local training constants (sentences are short -> bigger batch / lr than MuSiQue)
BATCH_SIZE = 16
LR         = 2e-5
N_EPOCHS   = 3
MODEL_DIR  = _HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",        action="store_true")
    ap.add_argument("--max_examples", type=int,   default=120_000)
    ap.add_argument("--val_examples", type=int,   default=4_000)
    ap.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    ap.add_argument("--epochs",       type=int,   default=N_EPOCHS)
    ap.add_argument("--lr",           type=float, default=LR)
    ap.add_argument("--tag",          type=str,   default="")
    args = ap.parse_args()

    if args.smoke:
        args.max_examples = 60
        args.val_examples = 20
        args.batch_size   = 4
        args.epochs       = 1
        print("[smoke] 60 train / 20 val, 1 epoch")

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading WikiAtomicEdits ...")
    all_trips = load_triples(
        max_examples=args.max_examples + args.val_examples, cache=not args.smoke,
    )
    val_trips   = all_trips[:args.val_examples]
    train_trips = all_trips[args.val_examples:]
    print(f"   Train: {len(train_trips):,} | Val: {len(val_trips):,}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collate   = make_collator(tokenizer)
    nw = 0 if args.smoke else 2

    train_loader = DataLoader(
        WikiEditDataset(train_trips), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=nw, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        WikiEditDataset(val_trips), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # ── Model (the fixed match-scaled ComplementGenerator) ───────────────────
    print("\n[2] Building ComplementGenerator (fixed) ...")
    model = ComplementGenerator().to(DEVICE)
    print(f"   Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    scaler       = GradScaler("cuda", enabled=AMP_ENABLED)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[3] Training {args.epochs} epoch(s) | {len(train_loader):,} steps/epoch")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_rec, tr_div, _ = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss, val_collapse = validate(model, val_loader)
        print(f"   Epoch {epoch}/{args.epochs} | train_rec={tr_rec:.4f} "
              f"train_div={tr_div:.4f} | val_loss={val_loss:.4f} "
              f"collapse_sim={val_collapse:.4f}")
        if val_collapse < 0.85:
            print("   note: collapse_sim < 0.85 -> complement is differentiating from encode(B)")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_DIR / f"complement_wiki_best{args.tag}.pt")
            print(f"   -> saved best (val_loss={val_loss:.4f}, collapse_sim={val_collapse:.4f})")

    torch.save(model.state_dict(), MODEL_DIR / f"complement_wiki_final{args.tag}.pt")
    print(f"\n[4] Done. best val_loss={best_val:.4f}")
    print(f"   Checkpoints: {MODEL_DIR}/complement_wiki_best{args.tag}.pt")
    if args.smoke:
        print("\n[smoke] PASSED")


if __name__ == "__main__":
    main()
