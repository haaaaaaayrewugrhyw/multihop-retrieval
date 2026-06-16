"""
delta3_clean.py -- the CLEAN test of the ORIGINAL idea + ONE change (small bottleneck), A-dropout
removed, single regeneration objective, on YIN'S DATASET (WikiAtomicEdits) so we compare to the paper.

Original architecture:
  edit encoder = generate_delta (subtraction + two-level cross-attn)   [unchanged]
  decoder      = original reconstruct-B decoder (cross-attends to A)    [single CE objective]
The ONLY changes vs the original:
  - delta bottlenecked to ONE small slot (n_slots=1)        ("keep delta small")
  - A-dropout = 0 (decoder may FREELY use A -- Yin's key condition)
  - drop delta_0 (mean-B summary shortcut); no VIB / no multi-objective
NO copy mechanism (original architecture only) -- so exact-match will likely be << Yin's 0.73; that
gap is the value of copy. The A-dependence diagnostic still tells us if the collapse is fixed.

Metrics on held-out WikiAtomicEdits:
  - Yin-comparable: GREEDY exact-match generation accuracy (vs Yin Acc@1 = 0.7294)
  - collapse diagnostic: teacher-forced token acc (overall|novel) for full / no_delta / no_A
      A-dependence = full - no_A   (prior collapsed run ~+0.06)

Run: python delta3_clean.py --n 4000 --steps 3000
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
import model as M
M.A_DROP_P = 0.0                                   # remove the A-dropout sabotage
from model import DeltaSystem, D_MODEL, BOS_ID
from delta2_model import DEVICE, MAX_LEN
from delta2_data  import load_wikiatomic_insertions, group_split
from insertion_cloze_eval import _novel_mask_difflib
from transformers import BertTokenizerFast

SEP_ID = 102


@torch.no_grad()
def encode_pairs(ds, pairs, tok, bs=16):
    HA, AM, HB, BM, BID, NOV = [], [], [], [], [], []
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        eA = tok([p["A"] for p in ch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in ch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        HA.append(ds._enc(A_ids, A_m)); AM.append(A_m)
        HB.append(ds._enc(B_ids, B_m)); BM.append(B_m); BID.append(B_ids)
        for p in ch:
            mask, rl = _novel_mask_difflib(p["A"], p["B"], tok)
            nv = np.zeros(MAX_LEN, np.float32)
            if mask is not None:
                rl = min(rl, MAX_LEN); nn_ = mask[:rl]; tm = np.ones(rl, bool)
                if rl > 2:
                    tm[0] = False; tm[rl - 1] = False
                nv[:rl] = (nn_ == 1).astype(np.float32) * tm
            NOV.append(nv)
    return dict(H_A=torch.cat(HA), A_m=torch.cat(AM), H_B=torch.cat(HB), B_m=torch.cat(BM),
                B_ids=torch.cat(BID), nov=torch.tensor(np.stack(NOV), device=DEVICE))


def take(E, idx):
    return {k: v[idx] for k, v in E.items()}


def build_memory(ds, H_A, A_m, delta, B_m, drop_delta=False, no_A=False):
    """Replicates the original reconstruct memory = [zeroed-d0 | H_A | 1-slot delta], drop_d0."""
    b, dev = H_A.size(0), H_A.device
    d0_tok = torch.zeros(b, 1, D_MODEL, device=dev)
    d0_mask = torch.ones(b, 1, dtype=torch.long, device=dev)
    if drop_delta:
        delta_mem = torch.zeros(b, ds.n_slots, D_MODEL, device=dev)
        delta_mask = torch.zeros(b, ds.n_slots, dtype=torch.long, device=dev)
    else:
        denc = ds.dr_delta_enc(delta, src_key_padding_mask=~B_m.bool())
        q = ds.dr_slots.expand(b, -1, -1)
        delta_mem, _ = ds.dr_slot_attn(q, denc, denc, key_padding_mask=~B_m.bool())
        delta_mask = torch.ones(b, ds.n_slots, dtype=torch.long, device=dev)
    A_mask = torch.zeros_like(A_m) if no_A else A_m
    memory = torch.cat([d0_tok, H_A, delta_mem], 1)
    mem_pad = ~torch.cat([d0_mask, A_mask, delta_mask], 1).bool()
    return memory, mem_pad


def tf_logits(ds, memory, mem_pad, B_ids, B_m):
    b, T = B_ids.shape; dev = B_ids.device
    shifted = torch.full((b, T), BOS_ID, dtype=B_ids.dtype, device=dev)
    shifted[:, 1:] = B_ids[:, :-1]
    tgt = ds.dr_word_emb(shifted) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
    causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
    out = ds.dr_decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~B_m.bool(),
                        memory_key_padding_mask=mem_pad)
    return ds.dr_lm(out)


@torch.no_grad()
def greedy(ds, memory, mem_pad, max_len):
    b, dev = memory.size(0), memory.device
    cur = torch.full((b, 1), BOS_ID, dtype=torch.long, device=dev)
    done = torch.zeros(b, dtype=torch.bool, device=dev)
    for _ in range(max_len - 1):
        T = cur.size(1)
        tgt = ds.dr_word_emb(cur) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
        out = ds.dr_decoder(tgt, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        nxt = ds.dr_lm(out[:, -1]).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, 0), nxt)
        cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
        done = done | (nxt == SEP_ID)
        if done.all():
            break
    return cur


def trim(ids):
    out = []
    for t in ids[1:]:                                  # drop BOS
        if t in (SEP_ID, 0):
            break
        out.append(int(t))
    return out


def acc(logits, B_ids, B_m, nov=None):
    pred = logits.argmax(-1)
    m = B_m.bool() & (B_ids != BOS_ID) & (B_ids != SEP_ID)
    if nov is not None:
        m = m & (nov > 0.5)
    return ((pred == B_ids) & m).float().sum() / m.float().sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gen_n", type=int, default=200)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 82)
    print("DELTA-3 CLEAN on WikiAtomicEdits (Yin's data): original arch + bottleneck(n_slots=1) + NO A-dropout")
    print(f"device={DEVICE} steps={args.steps} A_DROP_P={M.A_DROP_P}")
    print("=" * 82)
    pairs = load_wikiatomic_insertions(args.n)
    tr, te = group_split(pairs, test_frac=0.2)
    print(f"edits train {len(tr)} / test {len(te)}")

    ds = DeltaSystem(n_slots=1, vib=False, d0_aware=False).to(DEVICE)
    E_tr = encode_pairs(ds, tr, tok); E_te = encode_pairs(ds, te, tok)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0); N = E_tr["H_A"].size(0)

    ds.train()
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(rng.integers(0, N, args.bs), device=DEVICE)
        b = take(E_tr, idx)
        delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
        mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"])
        lg = tf_logits(ds, mem, mp, b["B_ids"], b["B_m"])
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), b["B_ids"].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"  step {step:>4} | loss {loss.item():.3f}")

    ds.eval()
    # collapse diagnostic (teacher-forced token acc)
    res = {}
    with torch.no_grad():
        for cond in ["full", "no_delta", "no_A"]:
            ov, nv, n = 0.0, 0.0, 0
            for i in range(0, E_te["H_A"].size(0), 32):
                idx = torch.arange(i, min(i + 32, E_te["H_A"].size(0)), device=DEVICE)
                b = take(E_te, idx)
                delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
                mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"],
                                       drop_delta=(cond == "no_delta"), no_A=(cond == "no_A"))
                lg = tf_logits(ds, mem, mp, b["B_ids"], b["B_m"])
                ov += acc(lg, b["B_ids"], b["B_m"]).item() * len(idx)
                nv += acc(lg, b["B_ids"], b["B_m"], b["nov"]).item() * len(idx); n += len(idx)
            res[cond] = (ov / n, nv / n)

    # Yin-comparable: greedy exact-match generation (full)
    em, n = 0, 0
    with torch.no_grad():
        for i in range(0, min(args.gen_n, E_te["H_A"].size(0)), 16):
            idx = torch.arange(i, min(i + 16, min(args.gen_n, E_te["H_A"].size(0))), device=DEVICE)
            b = take(E_te, idx)
            delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
            mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"])
            gen = greedy(ds, mem, mp, MAX_LEN).cpu().numpy()
            B = b["B_ids"].cpu().numpy()
            for g, bb in zip(gen, B):
                em += int(trim(g) == trim(bb)); n += 1

    print("\nheld-out token accuracy (overall | novel):")
    for c in ["full", "no_delta", "no_A"]:
        print(f"  {c:<9} overall {res[c][0]:.3f} | novel {res[c][1]:.3f}")
    print(f"\nGREEDY exact-match generation acc (full) = {em/max(n,1):.3f}  over {n}  (Yin Acc@1 = 0.729)")
    a_dep = res["full"][0] - res["no_A"][0]
    d_lift = res["full"][1] - res["no_delta"][1]
    print("\n" + "=" * 82)
    print(f"A-dependence (full - no_A, overall) = {a_dep:+.3f}   (prior collapsed run ~+0.06)")
    print(f"delta novel-lift (full - no_delta, novel) = {d_lift:+.3f}")
    print("READ: large A-dependence => decoder USES A (collapse fixed by bottleneck+no-dropout).")
    print("      exact-match << 0.729 is expected without a COPY mechanism -> that gap = value of copy.")
    print("=" * 82)


if __name__ == "__main__":
    main()
