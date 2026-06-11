"""
delta_decoder.py -- δ decoder: generate novel text from δ_0 alone.

δ_0 is the bottleneck representation from G:
    mean_pool(BERT(B)) → Linear(768, 230) → Tanh → Linear(230, 768)
It is a 768-dim vector summarising what B adds beyond A.

This decoder takes δ_0 as the ONLY input (no A, no full delta)
and generates the novel paragraph as text.

If the decoder can produce text resembling the novel content,
it proves δ_0 encodes readable novelty — not just abstract statistics.

Architecture:
    memory  = [d0_proj(δ_0)]          ← single 768-dim memory token
    decoder = 2-layer causal Transformer
    target  = novel paragraph tokens (teacher-forced during training)
    loss    = cross-entropy on novel tokens

Training:
    G is frozen (loaded from checkpoint).
    Only DeltaDecoder weights are updated.

Qualitative evaluation:
    For held-out (A, B, novel) pairs:
        1. Compute δ_0 with frozen G
        2. Greedy-decode from DeltaDecoder
        3. Print: A snippet | True novel | Decoded novel
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
from model import DeltaSystem, D_MODEL, N_HEADS, VOCAB_SIZE, MAX_SEQ, BOS_ID

MAX_LEN     = 128
N_DEC_LAYERS = 2
PAD_ID      = 0
SEP_ID      = 102   # [SEP]


# ── δ Decoder ────────────────────────────────────────────────────────────────

class DeltaDecoder(nn.Module):
    """
    Small causal decoder: δ_0 (768-dim) → novel text tokens.
    """
    def __init__(self):
        super().__init__()
        self.word_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_emb  = nn.Embedding(MAX_SEQ, D_MODEL)
        self.d0_proj  = nn.Linear(D_MODEL, D_MODEL)

        dec_layer = nn.TransformerDecoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=1024, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=N_DEC_LAYERS)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.word_emb.weight   # tie weights

    def forward(self, delta_0, novel_ids, novel_mask):
        """
        delta_0   : [b, 768]
        novel_ids : [b, T]   teacher-forced target tokens
        novel_mask: [b, T]   1 = real token, 0 = pad
        """
        b, T = novel_ids.shape
        dev  = novel_ids.device

        # Memory: single δ_0 token
        memory     = self.d0_proj(delta_0).unsqueeze(1)       # [b, 1, 768]
        mem_pad    = torch.zeros(b, 1, dtype=torch.bool, device=dev)  # no padding

        # Shifted target (teacher-forced)
        shifted        = torch.full((b, T), BOS_ID, dtype=novel_ids.dtype, device=dev)
        shifted[:, 1:] = novel_ids[:, :-1]
        tgt = (self.word_emb(shifted)
               + self.pos_emb(torch.arange(T, device=dev).unsqueeze(0)))

        causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~novel_mask.bool(),
            memory_key_padding_mask=mem_pad,
        )
        return self.lm_head(out)   # [b, T, VOCAB_SIZE]

    @torch.no_grad()
    def greedy_decode(self, delta_0, max_len=60):
        """
        Generate novel text greedily from δ_0 alone.
        Stops at [SEP] or max_len.
        """
        b   = delta_0.size(0)
        dev = delta_0.device
        memory  = self.d0_proj(delta_0).unsqueeze(1)       # [b, 1, 768]
        mem_pad = torch.zeros(b, 1, dtype=torch.bool, device=dev)

        tokens = torch.full((b, 1), BOS_ID, dtype=torch.long, device=dev)

        for _ in range(max_len - 1):
            T   = tokens.size(1)
            tgt = (self.word_emb(tokens)
                   + self.pos_emb(torch.arange(T, device=dev).unsqueeze(0)))
            causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
            out    = self.decoder(tgt=tgt, memory=memory,
                                  tgt_mask=causal,
                                  memory_key_padding_mask=mem_pad)
            next_id = self.lm_head(out[:, -1, :]).argmax(-1, keepdim=True)  # [b, 1]
            tokens  = torch.cat([tokens, next_id], dim=1)
            if (next_id == SEP_ID).all():
                break

        return tokens   # [b, T]


# ── Dataset ───────────────────────────────────────────────────────────────────

class DecoderDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        return p['A'], p['B'], p['novel']


def make_collate(tok):
    def collate(batch):
        A_texts = [x[0] for x in batch]
        B_texts = [x[1] for x in batch]
        N_texts = [x[2] for x in batch]
        eA = tok(A_texts, max_length=MAX_LEN, truncation=True,
                 padding='max_length', return_tensors='pt')
        eB = tok(B_texts, max_length=MAX_LEN, truncation=True,
                 padding='max_length', return_tensors='pt')
        eN = tok(N_texts, max_length=MAX_LEN, truncation=True,
                 padding='max_length', return_tensors='pt')
        return (eA['input_ids'], eA['attention_mask'],
                eB['input_ids'], eB['attention_mask'],
                eN['input_ids'], eN['attention_mask'],
                N_texts)
    return collate


# ── Training ─────────────────────────────────────────────────────────────────

def train_decoder(g_model, dec_model, train_pairs, tok,
                  steps=1000, bs=16, lr=1e-4, log_every=100,
                  device='cuda'):
    dl  = DataLoader(DecoderDS(train_pairs), batch_size=bs, shuffle=True,
                     collate_fn=make_collate(tok), num_workers=2, pin_memory=True)
    opt = torch.optim.Adam(dec_model.parameters(), lr=lr)

    g_model.eval()
    dec_model.train()
    step = 0

    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            A_ids, A_mask, B_ids, B_mask, N_ids, N_mask, _ = [
                t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
            ]

            # δ_0 from frozen G
            with torch.no_grad():
                H_A = g_model._enc(A_ids, A_mask)
                H_B = g_model._enc(B_ids, B_mask)
                _, delta_0, _ = g_model.generate_delta(H_A, A_mask, H_B, B_mask)

            # Decode novel text from δ_0
            logits = dec_model(delta_0, N_ids, N_mask)

            # Loss: cross-entropy on real novel tokens only
            labels = N_ids.clone()
            labels[~N_mask.bool()] = -100
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                labels.view(-1),
                ignore_index=-100,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % log_every == 0 or step == 1:
                ppl = math.exp(min(loss.item(), 20))
                print(f'  step {step:4d}/{steps} | dec_ppl={ppl:.1f}')

    return dec_model


# ── Qualitative evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def show_examples(g_model, dec_model, pairs, tok, n=10, device='cuda'):
    """
    Print side-by-side: A (first 80 chars) | True novel | Decoded novel
    """
    g_model.eval()
    dec_model.eval()

    print('\n' + '=' * 70)
    print('  δ DECODER — QUALITATIVE EXAMPLES')
    print('  Input: δ_0 only (no A, no D_recon)')
    print('=' * 70)

    for i, pair in enumerate(pairs[:n]):
        A_ids  = tok(pair['A'],     max_length=MAX_LEN, truncation=True,
                     return_tensors='pt')['input_ids'].to(device)
        A_mask = tok(pair['A'],     max_length=MAX_LEN, truncation=True,
                     return_tensors='pt')['attention_mask'].to(device)
        B_ids  = tok(pair['B'],     max_length=MAX_LEN, truncation=True,
                     return_tensors='pt')['input_ids'].to(device)
        B_mask = tok(pair['B'],     max_length=MAX_LEN, truncation=True,
                     return_tensors='pt')['attention_mask'].to(device)

        H_A = g_model._enc(A_ids, A_mask)
        H_B = g_model._enc(B_ids, B_mask)
        _, delta_0, _ = g_model.generate_delta(H_A, A_mask, H_B, B_mask)

        token_ids = dec_model.greedy_decode(delta_0, max_len=60)[0].tolist()
        # Strip BOS, stop at SEP/PAD
        token_ids = [t for t in token_ids[1:] if t not in (BOS_ID, SEP_ID, PAD_ID)]
        decoded   = tok.decode(token_ids, skip_special_tokens=True)

        print(f'\n[Example {i+1}]')
        print(f'  A (known)    : {pair["A"][:100]}...')
        print(f'  True novel   : {pair["novel"][:100]}...')
        print(f'  Decoded δ    : {decoded[:100]}')
        print(f'  ' + '-' * 66)

    print('=' * 70)
