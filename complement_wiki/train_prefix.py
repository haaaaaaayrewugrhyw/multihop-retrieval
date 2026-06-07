"""
train_prefix.py -- train the leak-free prefix-complement model + measure reconstruction
=======================================================================================

Self-supervised: reconstruct B from A + causal edge. NO labels for "what B adds".

Reports the two things that matter for the thesis:
  1. Reconstruction quality of B   -> val perplexity + token accuracy
  2. Recon-GAIN from the edge      -> perplexity(with edge) vs perplexity(A-only, edge zeroed)
        big gain => the edge carries real info about B that A lacks (leak-free)
Also re-runs a quick leak check at the end.

Usage:
    python train_prefix.py --smoke
    python train_prefix.py --max_examples 120000
"""

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))
from prefix_complement import PrefixComplementLM, MAX_A_LEN, MAX_B_LEN, VOCAB_SIZE
from data_wikiedits import load_triples

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")
PAD_ID      = 0
MODEL_DIR   = _HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)


class PairDS(Dataset):
    def __init__(self, triples): self.d = triples
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return (self.d[i]["base"], self.d[i]["edited"])


def collate(batch, tok):
    A = [x[0] for x in batch]; B = [x[1] for x in batch]
    ea = tok(A, max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
    eb = tok(B, max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return ea["input_ids"], ea["attention_mask"], eb["input_ids"], eb["attention_mask"]


def rec_loss(logits, B_ids):
    # predict each b_t (targets = B_ids), ignore pad
    tgt = B_ids.clone()
    tgt[tgt == PAD_ID] = -100
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model, loader, max_steps=200):
    model.eval()
    tot, tot_abl, n = 0.0, 0.0, 0
    correct, total_tok = 0, 0
    for i, (Aid, Am, Bid, Bm) in enumerate(loader):
        if i >= max_steps: break
        Aid, Am, Bid, Bm = Aid.to(DEVICE), Am.to(DEVICE), Bid.to(DEVICE), Bm.to(DEVICE)
        logits, _ = model(Aid, Am, Bid, Bm)
        tot += rec_loss(logits, Bid).item()
        logits_abl, _ = model(Aid, Am, Bid, Bm, ablate_edge=True)
        tot_abl += rec_loss(logits_abl, Bid).item()
        # token accuracy (with edge), over real tokens
        pred = logits.argmax(-1)
        real = (Bid != PAD_ID)
        correct += ((pred == Bid) & real).sum().item()
        total_tok += real.sum().item()
        n += 1
    L = tot / max(n, 1); L_abl = tot_abl / max(n, 1)
    return L, L_abl, correct / max(total_tok, 1)


@torch.no_grad()
def leak_check(model, tok):
    model.eval()
    A = ["The Eiffel Tower is a landmark located in the city of Paris in France ."]
    B = ["It was completed in the year eighteen eighty nine for the world fair ."]
    ea = tok(A, max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
    eb = tok(B, max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")
    Aid, Am, Bid, Bm = (ea["input_ids"].to(DEVICE), ea["attention_mask"].to(DEVICE),
                        eb["input_ids"].to(DEVICE), eb["attention_mask"].to(DEVICE))
    p = int(Bm[0].sum()) - 2
    l1, _ = model(Aid, Am, Bid, Bm)
    Bid2 = Bid.clone(); Bid2[0, p] = (Bid2[0, p].item() + 137) % 30000 + 1
    l2, _ = model(Aid, Am, Bid2, Bm)
    before = (l1[0, :p+1] - l2[0, :p+1]).abs().max().item()
    after  = (l1[0, p+1:int(Bm[0].sum())] - l2[0, p+1:int(Bm[0].sum())]).abs().max().item()
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_examples", type=int, default=120000)
    ap.add_argument("--val_examples", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--k_edge", type=int, default=2)
    ap.add_argument("--j_dec", type=int, default=1)
    args = ap.parse_args()
    if args.smoke:
        args.max_examples, args.val_examples = 60, 20
        args.batch_size, args.epochs = 4, 1
        print("[smoke] tiny run")

    print(f"Device: {DEVICE}")
    trips = load_triples(max_examples=args.max_examples + args.val_examples, cache=not args.smoke)
    val, train = trips[:args.val_examples], trips[args.val_examples:]
    print(f"Train {len(train):,} | Val {len(val):,}")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    cf = partial(collate, tok=tok)        # picklable (lambda breaks Windows DataLoader workers)
    nw = 0 if args.smoke else 2
    tl = DataLoader(PairDS(train), batch_size=args.batch_size, shuffle=True, collate_fn=cf,
                    num_workers=nw, pin_memory=(DEVICE == "cuda"))
    vl = DataLoader(PairDS(val), batch_size=args.batch_size, shuffle=False, collate_fn=cf, num_workers=0)

    model = PrefixComplementLM(k_edge_layers=args.k_edge, j_dec_layers=args.j_dec).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    scaler = GradScaler("cuda", enabled=AMP_ENABLED)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total = len(tl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0
        for step, (Aid, Am, Bid, Bm) in enumerate(tqdm(tl, desc=f"ep{ep}", leave=False)):
            Aid, Am, Bid, Bm = Aid.to(DEVICE), Am.to(DEVICE), Bid.to(DEVICE), Bm.to(DEVICE)
            with autocast("cuda", enabled=AMP_ENABLED):
                logits, _ = model(Aid, Am, Bid, Bm)
                loss = rec_loss(logits, Bid)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            run += loss.item()
            if step > 0 and step % 200 == 0:
                tqdm.write(f"  step {step} | L_rec={run/(step+1):.4f} ppl={math.exp(run/(step+1)):.1f}")
        L, L_abl, acc = evaluate(model, vl)
        print(f"  Epoch {ep} | val L_rec={L:.4f} ppl={math.exp(L):.1f} | "
              f"A-only ppl={math.exp(L_abl):.1f} | tok_acc={acc:.3f} | "
              f"recon-gain(ppl)={math.exp(L_abl)-math.exp(L):.1f}")
        if L < best:
            best = L
            torch.save(model.state_dict(), MODEL_DIR / "prefix_complement_best.pt")
            print("   -> saved best")

    torch.save(model.state_dict(), MODEL_DIR / "prefix_complement_final.pt")
    before, after = leak_check(model, tok)
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    L, L_abl, acc = evaluate(model, vl)
    print(f"  reconstruction:   val ppl = {math.exp(L):.1f}   token_acc = {acc:.3f}")
    print(f"  A-only (no edge): val ppl = {math.exp(L_abl):.1f}")
    print(f"  RECON-GAIN from edge:     = {math.exp(L_abl)-math.exp(L):.1f} ppl  "
          f"({'edge helps' if math.exp(L_abl)-math.exp(L) > 1 else 'edge adds little'})")
    print(f"  leak check: max|d| before perturb = {before:.2e}  (want ~0)")
    print(f"              max|d| after  perturb = {after:.2e}  (should be >0)")
    print(f"  -> {'LEAK-FREE' if before < 1e-4 else 'LEAK!'}")
    print("=" * 60)
    if args.smoke: print("[smoke] PASSED")


if __name__ == "__main__":
    main()
