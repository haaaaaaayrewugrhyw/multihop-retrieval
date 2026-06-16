"""
delta2c_e2e.py -- FULL end-to-end, trained, FAIR test of the learned delta.

Fixes the prior drift: instead of cosine-to-H_B-gold (which the complement ~equals by construction),
the neutral target is GENERATING THE ACTUAL NOVEL TOKENS of B from [A, delta], scored on HELD-OUT
data, with the LIFT from delta isolated (with-delta minus delta-ablated). Learned delta and the
zero-training complement go through the SAME decoder -> apples-to-apples.

  architecture: frozen BERT -> delta (learned generate_delta, OR fixed complement) ->
                TransformerDecoder generates B's novel tokens, memory = [H_A | proj(delta)]
  metric: held-out novel-token accuracy, and lift = acc(with delta) - acc(delta ablated)

  learned lift >= complement lift  => the learned delta carries the novel content (idea viable).
  learned lift <  complement lift  => complement still better, fairly (trained, neutral, held-out).

Run: python delta2c_e2e.py --n 1200 --steps 1500 --seeds 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model        import DeltaSystem, D_MODEL, VOCAB_SIZE, BOS_ID
from delta2_model import encode_all, take, DEVICE, MAX_LEN
from delta2b_data import build
from delta2_data  import group_split
from delta2_token_battery import op
from transformers import BertTokenizerFast

L_TGT = 12   # max novel-token sequence length


class E2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = DeltaSystem()                      # frozen BERT + generate_delta
        self.dproj = nn.Linear(D_MODEL, D_MODEL)       # project delta/comp into memory
        self.wemb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        with torch.no_grad():
            self.wemb.weight.copy_(self.core.bert.embeddings.word_embeddings.weight)
        self.pos = nn.Embedding(L_TGT + 1, D_MODEL)
        dl = nn.TransformerDecoderLayer(D_MODEL, 8, 2048, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, num_layers=2)
        self.lm = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False); self.lm.weight = self.wemb.weight

    @torch.no_grad()
    def encode(self, ids, m):
        return self.core._enc(ids, m)

    def learned_delta(self, H_A, A_m, H_B, B_m):
        d, _d0, _a = self.core.generate_delta(H_A, A_m, H_B, B_m)
        return d

    def decode(self, H_A, A_m, mem_delta, B_m, tgt, ablate=False):
        b, T = tgt.shape
        if ablate:
            mem_delta = torch.zeros_like(mem_delta)
        mem = torch.cat([H_A, self.dproj(mem_delta)], 1)
        mem_pad = ~torch.cat([A_m, B_m], 1).bool()
        shifted = torch.full((b, T), BOS_ID, dtype=tgt.dtype, device=tgt.device)
        shifted[:, 1:] = tgt[:, :-1]
        ti = self.wemb(shifted) + self.pos(torch.arange(T, device=tgt.device).unsqueeze(0))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device).bool()
        out = self.dec(ti, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.lm(out)

    def train_params(self, mode):
        ps = list(self.dproj.parameters()) + list(self.dec.parameters()) + list(self.pos.parameters())
        if mode == "learned":
            ps += [p for n, p in self.core.named_parameters() if n.startswith("g_")]
        return ps


def novel_targets(edits, tok):
    ids = np.zeros((len(edits), L_TGT), np.int64)
    for i, e in enumerate(edits):
        enc = tok(e["B"], max_length=MAX_LEN, truncation=True)["input_ids"]
        rl = min(e["real_len"], len(enc)); nov = e["novel_mask"][:rl]
        pos = [j for j in range(1, rl - 1) if nov[j] == 1]
        toks = [enc[j] for j in pos][:L_TGT]
        ids[i, :len(toks)] = toks
    return torch.tensor(ids, device=DEVICE)


def mem_delta(model, E, idx, mode):
    H_A, A_m, H_B, B_m = take(E, idx)
    if mode == "learned":
        return model.learned_delta(H_A, A_m, H_B, B_m)
    comp, _g = op(H_A, H_B, A_m, 0.1)                   # fixed complement
    return comp


def token_acc(logits, tgt, k=1):
    mask = (tgt != 0)
    topk = logits.topk(k, dim=-1).indices                 # [b,L,k]
    hit = (topk == tgt.unsqueeze(-1)).any(-1)
    return (hit & mask).float().sum() / mask.float().sum().clamp(min=1)


def run(mode, E_tr, T_tr, E_te, T_te, steps, seed, bs=24):
    model = E2E().to(DEVICE).train()
    opt = torch.optim.Adam(model.train_params(mode), lr=1e-4)
    rng = np.random.default_rng(seed)
    N = E_tr["H_A"].size(0)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, N, bs), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E_tr, idx)
        md = mem_delta(model, E_tr, idx, mode)
        logits = model.decode(H_A, A_m, md, B_m, T_tr[idx])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), T_tr[idx].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        accs = {}
        for ab, key in [(False, "with"), (True, "ablate")]:
            t1, t5, n = 0.0, 0.0, 0
            for i in range(0, E_te["H_A"].size(0), 64):
                idx = torch.arange(i, min(i + 64, E_te["H_A"].size(0)), device=DEVICE)
                H_A, A_m, H_B, B_m = take(E_te, idx)
                md = mem_delta(model, E_te, idx, mode)
                lg = model.decode(H_A, A_m, md, B_m, T_te[idx], ablate=ab)
                t1 += token_acc(lg, T_te[idx], 1).item() * len(idx)
                t5 += token_acc(lg, T_te[idx], 5).item() * len(idx); n += len(idx)
            accs[key] = (t1 / n, t5 / n)
    return accs["with"], accs["ablate"]                    # each = (top1, top5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 82)
    print(f"DELTA-2C FULL END-TO-END (novel-token generation, held-out)  device={DEVICE}  "
          f"steps={args.steps} seeds={args.seeds}")
    print("=" * 82)
    d = build(args.n)
    edits = [e for e in d["edits"]]
    print(f"edits {len(edits)}")

    rows = {"learned": [], "complement": []}
    for seed in range(args.seeds):
        tr, te = group_split(edits, test_frac=0.25, seed=seed)
        base = E2E().to(DEVICE).eval()
        E_tr = encode_all(base, tr, "A", "B", tok); E_te = encode_all(base, te, "A", "B", tok)
        del base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        T_tr, T_te = novel_targets(tr, tok), novel_targets(te, tok)
        for mode in ("learned", "complement"):
            w, a = run(mode, E_tr, T_tr, E_te, T_te, args.steps, seed)   # w=(top1,top5), a=(top1,top5)
            rows[mode].append((w, a))
            print(f"  seed {seed} {mode:<10} top1 with {w[0]:.3f}/ab {a[0]:.3f} LIFT {w[0]-a[0]:+.3f}  |  "
                  f"top5 with {w[1]:.3f}/ab {a[1]:.3f} LIFT {w[1]-a[1]:+.3f}")

    print("\n" + "=" * 82)
    for mode in ("learned", "complement"):
        w1 = np.mean([x[0][0] for x in rows[mode]]); a1 = np.mean([x[1][0] for x in rows[mode]])
        w5 = np.mean([x[0][1] for x in rows[mode]]); a5 = np.mean([x[1][1] for x in rows[mode]])
        print(f"  {mode:<10} top1 lift {w1-a1:+.3f} (with {w1:.3f}) | top5 lift {w5-a5:+.3f} (with {w5:.3f})")
    print("\nREAD (fair, trained, held-out, neutral metric):")
    print("  delta LIFT (with - ablate) is what the delta itself contributes. Compare learned vs complement.")
    print("  If both top5 ~ floor => generation too hard to discriminate (inconclusive, not a verdict).")
    print("=" * 82)


if __name__ == "__main__":
    main()
