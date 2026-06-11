"""
test_decoder.py -- Local smoke test for DeltaDecoder.

Trains G on 50 MuSiQue pairs (100 steps), then trains DeltaDecoder
on the same pairs (200 steps), then shows 5 qualitative examples.
Total runtime: ~3-5 minutes on CPU, ~1 min on GPU.

Run:
    cd delta_system
    python test_decoder.py
    python test_decoder.py --smoke   # even smaller: 20 pairs, 50+100 steps
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))

from data          import load_pairs
from model         import DeltaSystem, D_MODEL, VOCAB_SIZE
from losses        import recon_loss, sparsity_loss, specificity_loss
from delta_decoder import DeltaDecoder, train_decoder, show_examples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_g(model, pairs, tok, steps=100, bs=4, lr=1e-4, log_every=20):
    """Quick G training loop (mirrors kaggle_notebook Cell 6)."""
    class PairDS(Dataset):
        def __init__(self, p): self.p = p
        def __len__(self): return len(self.p)
        def __getitem__(self, i): return self.p[i]['A'], self.p[i]['B']

    def collate(batch):
        eA = tok([x[0] for x in batch], max_length=128, truncation=True,
                 padding='max_length', return_tensors='pt')
        eB = tok([x[1] for x in batch], max_length=128, truncation=True,
                 padding='max_length', return_tensors='pt')
        return eA['input_ids'], eA['attention_mask'], eB['input_ids'], eB['attention_mask']

    dl  = DataLoader(PairDS(pairs), batch_size=bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    step = 0
    while step < steps:
        for batch in dl:
            if step >= steps: break
            A_ids, A_mask, B_ids, B_mask = [t.to(DEVICE) for t in batch]
            b = A_ids.size(0)
            logits, delta, delta_0, H_A, _ = model(A_ids, A_mask, B_ids, B_mask)
            L_r = recon_loss(logits, B_ids, B_mask)
            L_s = sparsity_loss(delta, B_mask)
            L_spec = torch.tensor(0.0, device=DEVICE)
            if b > 1:
                idx = list(range(1, b)) + [0]
                lw  = model.reconstruct(H_A, A_mask, delta[idx], delta_0[idx], B_ids, B_mask)
                L_spec = specificity_loss(logits, lw, B_ids, B_mask, margin=2.0)
            loss = L_r + L_s + L_spec
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); step += 1
            if step % log_every == 0 or step == 1:
                print(f'  [G] step {step:3d}/{steps} | ppl={math.exp(min(L_r.item(),20)):.1f} | L_spec={L_spec.item():.3f}')
    return model


def check_delta0_stats(model, pairs, tok):
    """Print δ_0 statistics — confirms G produces pair-specific representations."""
    model.eval()
    sample = pairs[:8]
    with torch.no_grad():
        eA = tok([p['A'] for p in sample], max_length=128, truncation=True,
                 padding='max_length', return_tensors='pt')
        eB = tok([p['B'] for p in sample], max_length=128, truncation=True,
                 padding='max_length', return_tensors='pt')
        A_ids = eA['input_ids'].to(DEVICE); A_mask = eA['attention_mask'].to(DEVICE)
        B_ids = eB['input_ids'].to(DEVICE); B_mask = eB['attention_mask'].to(DEVICE)
        H_A = model._enc(A_ids, A_mask)
        H_B = model._enc(B_ids, B_mask)
        _, d0, _ = model.generate_delta(H_A, A_mask, H_B, B_mask)
    print(f'  δ_0 mean_abs : {d0.abs().mean().item():.4f}')
    print(f'  δ_0 std_across_examples : {d0.std(0).mean().item():.4f}  (>0.01 = pair-specific, good)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='ultra-fast: 20 pairs, 50 G steps, 100 decoder steps')
    ap.add_argument('--n',          type=int, default=50)
    ap.add_argument('--g_steps',    type=int, default=100)
    ap.add_argument('--dec_steps',  type=int, default=200)
    ap.add_argument('--bs',         type=int, default=4)
    ap.add_argument('--n_examples', type=int, default=5)
    args = ap.parse_args()

    if args.smoke:
        args.n, args.g_steps, args.dec_steps, args.bs = 20, 50, 100, 4

    print(f'Device : {DEVICE}')
    print(f'Config : {args.n} pairs | G:{args.g_steps} steps | Dec:{args.dec_steps} steps')
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading MuSiQue pairs...')
    pairs = load_pairs(max_examples=args.n)
    print(f'  Loaded {len(pairs)} pairs')
    print()

    tok   = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = DeltaSystem().to(DEVICE)
    n_g   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'G trainable params: {n_g/1e6:.1f}M')
    print()

    # ── Train G ───────────────────────────────────────────────────────────────
    print('--- Phase 1: Train G ---')
    model = train_g(model, pairs, tok,
                    steps=args.g_steps, bs=args.bs, log_every=max(1, args.g_steps//5))
    print()

    # ── Check δ_0 stats ───────────────────────────────────────────────────────
    print('--- δ_0 diagnostic (before decoder training) ---')
    check_delta0_stats(model, pairs, tok)
    print()

    # ── Freeze G, init decoder ────────────────────────────────────────────────
    print('--- Phase 2: Train DeltaDecoder ---')
    for p in model.parameters():
        p.requires_grad_(False)

    dec = DeltaDecoder().to(DEVICE)
    dec.copy_bert_embeddings(model.bert)
    n_dec = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f'Decoder params: {n_dec/1e6:.2f}M | BERT embeddings copied')
    print(f'Expect loss to START ~10 nats and DROP. If stuck at 20+ → still broken.')
    print()

    dec = train_decoder(model, dec, pairs, tok,
                        steps=args.dec_steps, bs=args.bs, lr=3e-4,
                        log_every=max(1, args.dec_steps//5),
                        device=DEVICE)
    print()

    # ── Qualitative examples ──────────────────────────────────────────────────
    print('--- Phase 3: Qualitative examples ---')
    show_examples(model, dec, pairs, tok, n=args.n_examples, device=DEVICE)


if __name__ == '__main__':
    main()
