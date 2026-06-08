"""
train_novelty.py -- train the "edge one-ahead" novelty autoencoder
==================================================================

Loss = reconstruction CE over B  +  beta * mean(alpha)   (the sparsity squeeze).
Self-supervised: the inserted phrase is NEVER a training target.

Final report = THE thesis test, the right way this time:
  ALPHA-LOCALIZATION: mean gate alpha on NOVEL (inserted phrase) tokens vs COPIED.
    alpha(novel) >> alpha(copied)  =>  the squeeze made the edge open ONLY for the
    added content -> it localized "what B adds", with NO labels.
  Also: reconstruction ppl, recon-gain (gate vs ablated), and the novel/copied
  recon-gain split (does the edge actually help the novel tokens?).

Usage:
    python train_novelty.py --smoke
    python train_novelty.py --max_examples 120000 --beta 0.10
"""

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))
from novelty_ae import NoveltyAutoencoder, MAX_A_LEN, MAX_B_LEN, VOCAB_SIZE
from data_wikiedits import load_triples

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")
PAD_ID      = 0
MODEL_DIR   = _HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)


class TripDS(Dataset):
    def __init__(self, triples): self.d = triples
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return (self.d[i]["base"], self.d[i]["edited"], self.d[i]["inserted"])


def collate(batch, tok):
    A = [x[0] for x in batch]; B = [x[1] for x in batch]; P = [x[2] for x in batch]
    ea = tok(A, max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
    eb = tok(B, max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return ea["input_ids"], ea["attention_mask"], eb["input_ids"], eb["attention_mask"], P


def rec_loss(logits, B_ids):
    tgt = B_ids.clone(); tgt[tgt == PAD_ID] = -100
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=-100)


def find_span(hay, needle):
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if hay[i:i + m] == needle:
            return i, i + m
    return None


@torch.no_grad()
def evaluate(model, loader, tok, max_steps=200):
    """ppl (gate + ablated) and the alpha/recon-gain localization on phrase spans."""
    model.eval()
    tot, tot_abl, n = 0.0, 0.0, 0
    a_nov, a_cop = [], []          # gate alpha
    d_nov, d_cop = [], []          # per-token recon-gain (nll_abl - nll_edge)
    matched = 0
    for i, (Aid, Am, Bid, Bm, P) in enumerate(loader):
        if i >= max_steps: break
        Aid, Am, Bid, Bm = Aid.to(DEVICE), Am.to(DEVICE), Bid.to(DEVICE), Bm.to(DEVICE)
        logits, _, alpha = model(Aid, Am, Bid, Bm)
        logits_abl, _, _ = model(Aid, Am, Bid, Bm, ablate_edge=True)
        tot += rec_loss(logits, Bid).item(); tot_abl += rec_loss(logits_abl, Bid).item(); n += 1

        nll_e = -F.log_softmax(logits.float(), -1).gather(-1, Bid.unsqueeze(-1)).squeeze(-1)
        nll_a = -F.log_softmax(logits_abl.float(), -1).gather(-1, Bid.unsqueeze(-1)).squeeze(-1)
        delta = (nll_a - nll_e).cpu().numpy()
        al = alpha.cpu().numpy()
        for bi, phrase in enumerate(P):
            real = int(Bm[bi].sum().item())
            b_ids = Bid[bi].cpu().tolist()
            p_ids = tok(phrase, add_special_tokens=False)["input_ids"]
            span = find_span(b_ids[:real], p_ids)
            if span is None:
                continue
            matched += 1; lo, hi = span
            for t in range(1, real - 1):
                (a_nov if lo <= t < hi else a_cop).append(float(al[bi, t]))
                (d_nov if lo <= t < hi else d_cop).append(float(delta[bi, t]))
    L = tot / max(n, 1); L_abl = tot_abl / max(n, 1)
    return {
        "ppl": math.exp(L), "ppl_abl": math.exp(L_abl),
        "a_nov": float(np.mean(a_nov)) if a_nov else 0.0,
        "a_cop": float(np.mean(a_cop)) if a_cop else 0.0,
        "d_nov": float(np.mean(d_nov)) if d_nov else 0.0,
        "d_cop": float(np.mean(d_cop)) if d_cop else 0.0,
        "matched": matched,
    }


def report(tag, m):
    print(f"\n{'='*60}\n  {tag}\n{'='*60}")
    print(f"  reconstruction ppl (gate)   = {m['ppl']:.1f}")
    print(f"  backbone ppl (edge hidden)  = {m['ppl_abl']:.1f}   "
          f"({'BACKBONE OK' if m['ppl_abl'] < 1000 else 'NO BACKBONE -> edge is a cheat sheet'})")
    print(f"  matched phrase spans        = {m['matched']}")
    print(f"  --- ALPHA (gate) localization ------------------------------")
    print(f"    alpha NOVEL  = {m['a_nov']:.4f}   alpha COPIED = {m['a_cop']:.4f}   "
          f"ratio {m['a_nov']/max(m['a_cop'],1e-6):.2f}x")
    print(f"  --- recon-gain localization --------------------------------")
    print(f"    delta NOVEL  = {m['d_nov']:+.4f}  delta COPIED = {m['d_cop']:+.4f}  "
          f"ratio {m['d_nov']/max(m['d_cop'],1e-6):.2f}x")
    print("=" * 60)
    if m['a_nov'] > 1.5 * max(m['a_cop'], 1e-6) and m['d_nov'] > 1.5 * max(m['d_cop'], 1e-6):
        print("  THESIS SUPPORTED: gate opens on the ADDED content and the edge helps")
        print("  it specifically -> 'what B adds' localized, with NO labels.")
    elif m['a_nov'] > m['a_cop']:
        print("  PARTIAL: gate leans novel but not decisively (try higher --beta).")
    else:
        print("  NEGATIVE: gate did not localize to the added content.")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_examples", type=int, default=120000)
    ap.add_argument("--val_examples", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta", type=float, default=0.30, help="sparsity squeeze on the gate")
    ap.add_argument("--edge_dropout", type=float, default=0.5,
                    help="prob of hiding the edge (forces the A+b_<t backbone)")
    ap.add_argument("--j_dec", type=int, default=2)
    args = ap.parse_args()
    if args.smoke:
        args.max_examples, args.val_examples = 60, 20
        args.batch_size, args.epochs = 4, 1
        print("[smoke] tiny run")

    print(f"Device: {DEVICE} | beta={args.beta} | edge_dropout={args.edge_dropout}")
    trips = load_triples(max_examples=args.max_examples + args.val_examples, cache=not args.smoke)
    val, train = trips[:args.val_examples], trips[args.val_examples:]
    print(f"Train {len(train):,} | Val {len(val):,}")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    cf = partial(collate, tok=tok)
    nw = 0 if args.smoke else 2
    tl = DataLoader(TripDS(train), batch_size=args.batch_size, shuffle=True, collate_fn=cf,
                    num_workers=nw, pin_memory=(DEVICE == "cuda"))
    vl = DataLoader(TripDS(val), batch_size=args.batch_size, shuffle=False, collate_fn=cf, num_workers=0)

    model = NoveltyAutoencoder(j_dec_layers=args.j_dec).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    scaler = GradScaler("cuda", enabled=AMP_ENABLED)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total = len(tl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train(); run_r, run_a = 0.0, 0.0
        for step, (Aid, Am, Bid, Bm, _) in enumerate(tqdm(tl, desc=f"ep{ep}", leave=False)):
            Aid, Am, Bid, Bm = Aid.to(DEVICE), Am.to(DEVICE), Bid.to(DEVICE), Bm.to(DEVICE)
            with autocast("cuda", enabled=AMP_ENABLED):
                logits, _, alpha = model(Aid, Am, Bid, Bm, edge_dropout=args.edge_dropout)
                L_rec = rec_loss(logits, Bid)
                real = (Bm > 0).float()
                L_spar = (alpha * real).sum() / real.sum().clamp(min=1)   # mean gate over real tokens
                loss = L_rec + args.beta * L_spar
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            run_r += L_rec.item(); run_a += L_spar.item()
            if step > 0 and step % 200 == 0:
                tqdm.write(f"  step {step} | L_rec={run_r/(step+1):.4f} "
                           f"ppl={math.exp(run_r/(step+1)):.1f} | mean_alpha={run_a/(step+1):.3f}")
        m = evaluate(model, vl, tok)
        print(f"  Epoch {ep} | ppl={m['ppl']:.1f} | ppl_abl={m['ppl_abl']:.1f} | "
              f"alpha nov/cop={m['a_nov']:.3f}/{m['a_cop']:.3f} | "
              f"recon-gain nov/cop={m['d_nov']:+.3f}/{m['d_cop']:+.3f}")
        if m['ppl'] < best:
            best = m['ppl']; torch.save(model.state_dict(), MODEL_DIR / "novelty_ae_best.pt")
            print("   -> saved best")

    torch.save(model.state_dict(), MODEL_DIR / "novelty_ae_final.pt")
    report("FINAL REPORT (alpha-localization = the thesis test)", evaluate(model, vl, tok))
    if args.smoke:
        print("[smoke] PASSED")


if __name__ == "__main__":
    main()
