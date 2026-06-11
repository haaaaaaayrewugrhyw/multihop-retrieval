"""
baseline.py -- Naive delta: mean(BERT(B)) - mean(BERT(A))

Replaces the entire cross-attention Generator G with a simple mean-pool subtraction.
D_recon architecture and all losses are identical to the real system.

If this passes DELTA_PPL → cross-attention adds nothing over naive subtraction.
If this fails DELTA_PPL → cross-attention is doing real work.

Usage:
    python baseline.py --n 500 --steps 500
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
from data   import load_pairs
from losses import recon_loss, sparsity_loss, specificity_loss
from eval   import _novel_token_mask, _ppl_batch

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 128
D_MODEL    = 768
N_HEADS    = 8
N_LAYERS   = 2
VOCAB_SIZE = 30522
MAX_SEQ    = 256
BOS_ID     = 101
A_DROP_P   = 0.20
D_SMALL    = round(0.30 * D_MODEL)


class BaselineSystem(nn.Module):
    """
    G = mean(BERT(B)) - mean(BERT(A))  broadcast to all token positions.
    D_recon = identical to real system.
    """
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad_(False)

        # D_recon (identical to real system)
        self.dr_word_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        with torch.no_grad():
            self.dr_word_emb.weight.copy_(self.bert.embeddings.word_embeddings.weight)
        self.dr_pos = nn.Embedding(MAX_SEQ, D_MODEL)
        self.dr_d0  = nn.Linear(D_MODEL, D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.dr_delta_enc = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        dec_layer = nn.TransformerDecoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.dr_decoder = nn.TransformerDecoder(dec_layer, num_layers=N_LAYERS)
        self.dr_lm = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.dr_lm.weight = self.dr_word_emb.weight

    def _enc(self, ids, mask):
        with torch.no_grad():
            return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state

    def _mean_pool(self, H, mask):
        lens = mask.float().sum(1, keepdim=True).clamp(min=1)
        return (H * mask.unsqueeze(-1).float()).sum(1) / lens   # [b, 768]

    def generate_delta(self, H_A, A_mask, H_B, B_mask):
        # NAIVE BASELINE: mean(B) - mean(A), broadcast to all token positions
        mean_A = self._mean_pool(H_A, A_mask)   # [b, 768]
        mean_B = self._mean_pool(H_B, B_mask)   # [b, 768]
        diff   = mean_B - mean_A                 # [b, 768]

        T     = B_mask.size(1)
        delta = diff.unsqueeze(1).expand(-1, T, -1)   # [b, T, 768] — same vector everywhere
        delta = torch.tanh(delta)

        delta_0 = diff   # same signal as global summary

        alpha = delta.norm(dim=-1) / (D_MODEL ** 0.5)  # proxy for interface compatibility
        return delta, delta_0, alpha

    def reconstruct(self, H_A, A_mask, delta, delta_0, B_ids, B_mask,
                    ablate_delta: bool = False):
        b, T = B_ids.shape
        dev  = B_ids.device

        if self.training:
            keep   = (torch.rand(b, device=dev) > A_DROP_P).float()
            H_A    = H_A * keep.view(b, 1, 1)
            A_mask = (A_mask.float() * keep.view(b, 1)).long()

        if ablate_delta:
            d0_tok    = torch.zeros(b, 1, D_MODEL, device=dev)
            delta_enc = torch.zeros(b, T, D_MODEL, device=dev)
            delta_mask = torch.zeros(b, T, dtype=torch.long, device=dev)
        else:
            d0_tok     = self.dr_d0(delta_0).unsqueeze(1)
            delta_enc  = self.dr_delta_enc(delta, src_key_padding_mask=~B_mask.bool())
            delta_mask = B_mask

        d0_mask = torch.ones(b, 1, dtype=torch.long, device=dev)
        memory  = torch.cat([d0_tok, H_A, delta_enc], dim=1)
        mem_pad = ~torch.cat([d0_mask, A_mask, delta_mask], dim=1).bool()

        shifted        = torch.full((b, T), BOS_ID, dtype=B_ids.dtype, device=dev)
        shifted[:, 1:] = B_ids[:, :-1]
        tgt = (self.dr_word_emb(shifted)
               + self.dr_pos(torch.arange(T, device=dev).unsqueeze(0)))

        causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
        out = self.dr_decoder(
            tgt=tgt, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~B_mask.bool(),
            memory_key_padding_mask=mem_pad,
        )
        return self.dr_lm(out)

    def forward(self, A_ids, A_mask, B_ids, B_mask):
        H_A = self._enc(A_ids, A_mask)
        H_B = self._enc(B_ids, B_mask)
        delta, delta_0, alpha = self.generate_delta(H_A, A_mask, H_B, B_mask)
        logits = self.reconstruct(H_A, A_mask, delta, delta_0, B_ids, B_mask)
        return logits, delta, delta_0, H_A, alpha


class PairDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.pairs[i]["A"], self.pairs[i]["B"]


def make_collate_train(tok):
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


def make_collate_eval(tok):
    def collate(batch):
        A_texts = [x["A"]     for x in batch]
        B_texts = [x["B"]     for x in batch]
        N_texts = [x["novel"] for x in batch]
        eA = tok(A_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok(B_texts, max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        return (eA["input_ids"], eA["attention_mask"],
                eB["input_ids"], eB["attention_mask"],
                A_texts, N_texts)
    return collate


def train(model, pairs, tok, args):
    ds = PairDS(pairs)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=make_collate_train(tok), num_workers=0)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    model.train()
    step = 0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            A_ids, A_mask, B_ids, B_mask = [t.to(DEVICE) for t in batch]
            b = A_ids.size(0)

            logits, delta, delta_0, H_A, alpha = model(A_ids, A_mask, B_ids, B_mask)
            L_r = recon_loss(logits, B_ids, B_mask)
            L_s = sparsity_loss(delta, B_mask)

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

            loss = L_r + args.lam_s * L_s + args.lam_spec * L_spec
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                ppl = math.exp(min(L_r.item(), 20))
                print(f"  step {step:4d} | L_recon={L_r.item():.4f}  ppl={ppl:.1f} | "
                      f"L_spec={L_spec.item():.4f}")


@torch.no_grad()
def evaluate(model, pairs, tok):
    from torch.utils.data import Dataset as DS

    class EvalDS(DS):
        def __init__(self, p): self.p = p
        def __len__(self): return len(self.p)
        def __getitem__(self, i): return self.p[i]

    model.eval()
    dl = DataLoader(EvalDS(pairs), batch_size=8, shuffle=False,
                    collate_fn=make_collate_eval(tok), num_workers=0)

    ppl_with_list, ppl_no_list = [], []
    ppl_correct_list, ppl_wrong_list = [], []
    all_norms, all_labels = [], []

    for A_ids, A_mask, B_ids, B_mask, A_texts, N_texts in dl:
        A_ids, A_mask = A_ids.to(DEVICE), A_mask.to(DEVICE)
        B_ids, B_mask = B_ids.to(DEVICE), B_mask.to(DEVICE)
        b = A_ids.size(0)

        H_A = model._enc(A_ids, A_mask)
        H_B = model._enc(B_ids, B_mask)
        delta, delta_0, alpha = model.generate_delta(H_A, A_mask, H_B, B_mask)

        logits_with = model.reconstruct(H_A, A_mask, delta, delta_0,
                                        B_ids, B_mask, ablate_delta=False)
        logits_no   = model.reconstruct(H_A, A_mask, delta, delta_0,
                                        B_ids, B_mask, ablate_delta=True)

        ppl_with_list.extend(_ppl_batch(logits_with, B_ids, B_mask))
        ppl_no_list.extend(  _ppl_batch(logits_no,   B_ids, B_mask))

        if b > 1:
            idx_shift = list(range(1, b)) + [0]
            d_s  = delta[idx_shift]
            d0_s = delta_0[idx_shift]
            logits_wrong = model.reconstruct(H_A, A_mask, d_s, d0_s,
                                             B_ids, B_mask, ablate_delta=False)
            ppl_correct_list.extend(_ppl_batch(logits_with, B_ids, B_mask))
            ppl_wrong_list.extend(  _ppl_batch(logits_wrong, B_ids, B_mask))

        # AUROC: delta norms (but delta is same vector at every position, so this
        # will be constant per-example → AUROC ≈ 0.5 by design for baseline)
        delta_norms = delta.norm(dim=-1).cpu().numpy()
        for i in range(b):
            labels = _novel_token_mask(A_texts[i], N_texts[i], B_mask[i].cpu(), tok)
            if labels.sum() > 0 and labels.sum() < len(labels):
                all_norms.append(delta_norms[i])
                all_labels.append(labels)

    mean_with = float(np.mean(ppl_with_list))
    mean_no   = float(np.mean(ppl_no_list))
    delta_ppl = mean_no - mean_with

    # Per-example AUROC averaged: within-sequence localization only
    if all_norms:
        per_auroc = []
        for norms_i, labels_i in zip(all_norms, all_labels):
            if labels_i.sum() > 0 and labels_i.sum() < len(labels_i):
                try:
                    per_auroc.append(roc_auc_score(labels_i, norms_i))
                except Exception:
                    pass
        auroc = float(np.mean(per_auroc)) if per_auroc else float("nan")
    else:
        auroc = float("nan")

    specificity = float(np.mean(ppl_wrong_list) - np.mean(ppl_correct_list)) \
                  if ppl_correct_list else 0.0

    n = len(ppl_with_list)
    print("\n" + "=" * 62)
    print("  BASELINE RESULTS  (G = mean(BERT(B)) - mean(BERT(A)))")
    print("=" * 62)
    print(f"  Examples              : {n}")
    print(f"  PPL with delta        : {mean_with:7.1f}")
    print(f"  PPL without delta     : {mean_no:7.1f}")
    print(f"  DELTA_PPL             : {delta_ppl:+7.2f}  (positive = delta helps)")
    print(f"  AUROC (delta vs novel): {auroc:.4f}    (0.5 = random, expected ~0.5)")
    print(f"  SPECIFICITY (PPL gap) : {specificity:+7.2f}")
    print("=" * 62)
    p1 = delta_ppl   >  2.0;  print(f"    DELTA_PPL > 2   : {'PASS' if p1 else 'FAIL'}")
    p2 = auroc       > 0.55;  print(f"    AUROC > 0.55    : {'PASS' if p2 else 'FAIL'} (expected FAIL — same vector everywhere)")
    p3 = specificity >  2.0;  print(f"    SPECIFICITY > 2 : {'PASS' if p3 else 'FAIL'}")
    print("=" * 62)
    if p1:
        print("\n  !! DELTA_PPL PASS on baseline means:")
        print("     D_recon can extract info from mean-pool diff alone.")
        print("     Cross-attention G may not add much over this simple signal.")
    else:
        print("\n  ✓  DELTA_PPL FAIL on baseline — cross-attention G is doing real work.")

    return {"delta_ppl": delta_ppl, "auroc": auroc, "specificity": specificity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",        type=int,   default=500)
    ap.add_argument("--steps",    type=int,   default=500)
    ap.add_argument("--bs",       type=int,   default=8)
    ap.add_argument("--lr",       type=float, default=1e-4)
    ap.add_argument("--lam_s",    type=float, default=1.0)
    ap.add_argument("--lam_spec", type=float, default=1.0)
    ap.add_argument("--margin",   type=float, default=2.0)
    ap.add_argument("--log_every",type=int,   default=50)
    args = ap.parse_args()

    pairs = load_pairs(max_examples=args.n)
    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BaselineSystem().to(DEVICE)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BASELINE — G = mean(BERT(B)) - mean(BERT(A))")
    print(f"Device: {DEVICE} | Trainable params: {n_train/1e6:.1f}M")

    train(model, pairs, tok, args)
    print("\n--- Training complete. Running evaluation. ---")
    evaluate(model, pairs, tok)


if __name__ == "__main__":
    main()
