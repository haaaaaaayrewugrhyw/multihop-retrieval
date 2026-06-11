"""
run.py -- Entry point for the delta-system validation experiment.

Usage:
    python run.py                  # 100 examples, 500 steps, then eval
    python run.py --smoke          # 20 examples, 50 steps (2-minute sanity check)
    python run.py --eval-only      # eval with saved checkpoint
    python run.py --n 200 --steps 1000   # bigger run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",     action="store_true",
                    help="tiny run: 20 examples, 50 steps")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate saved checkpoint")
    ap.add_argument("--n",         type=int,   default=100,
                    help="number of MuSiQue pairs to use")
    ap.add_argument("--steps",     type=int,   default=500)
    ap.add_argument("--bs",        type=int,   default=8)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--lam_s",     type=float, default=0.1)
    ap.add_argument("--lam_spec",  type=float, default=0.0,
                    help="specificity loss weight")
    ap.add_argument("--margin",    type=float, default=0.5,
                    help="margin for specificity ranking loss")
    ap.add_argument("--beta_gate", type=float, default=0.0,
                    help="gate sparsity penalty weight (ramped via warmup)")
    ap.add_argument("--held_out",  type=int,   default=0,
                    help="held-out eval examples (loaded after training set, never seen during training)")
    ap.add_argument("--log_every", type=int,   default=50)
    args = ap.parse_args()

    if args.smoke:
        args.n, args.steps, args.bs = 20, 50, 4
        args.log_every = 10
        print("[smoke] 20 examples, 50 steps — quick sanity check")

    if not args.eval_only:
        from data  import load_pairs
        from train import train
        from eval  import evaluate

        # Load train + held-out in one shot to guarantee no overlap
        total   = args.n + args.held_out
        all_pairs = load_pairs(max_examples=total)
        train_pairs = all_pairs[:args.n]
        eval_pairs  = all_pairs[args.n:] if args.held_out > 0 else train_pairs

        if args.held_out > 0:
            print(f"[split] {len(train_pairs)} train / {len(eval_pairs)} held-out eval")
        else:
            print(f"[split] evaluating on training set (no held-out)")

        model, tok, _ = train(args, pairs=train_pairs)

        print("\n--- Training complete. Running evaluation. ---")
        if args.held_out > 0:
            print(f"*** HELD-OUT EVAL on {len(eval_pairs)} unseen pairs ***")
        results = evaluate(model, eval_pairs, tok)
        return results

    else:
        import torch
        from transformers import BertTokenizerFast
        from data  import load_pairs
        from model import DeltaSystem
        from eval  import evaluate

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        pairs  = load_pairs(max_examples=args.n)
        tok    = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model  = DeltaSystem().to(DEVICE)
        ckpt   = Path(__file__).parent / "checkpoints" / "val_model.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"Loaded: {ckpt}")
        evaluate(model, pairs, tok)


if __name__ == "__main__":
    main()
