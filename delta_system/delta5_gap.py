"""
delta5_gap.py -- THE USER'S IDEA: replace B-regeneration with MISSING-SPAN PREDICTION so the model
can never collapse on a shortcut. Instead of regenerating all of B (where A + copy do the work and the
delta stays inert), we predict ONLY the inserted phrase, given A + the delta.

Why this is the sharpest test of "does the learned delta carry WHAT B adds":
  - the phrase is NOT in A  -> copy is impossible, A cannot shortcut: the answer can ONLY come
    through the delta;
  - the phrase IS the entire target -> all gradient pressure is on the content (in regeneration the
    inserted tokens were a tiny fraction of the loss, so the delta was never forced to carry them);
  - clean reads: full - no_delta = the delta's content contribution; train - test = memorization
    vs genuine generalization.

Architecture: original subtraction encoder (`generate_delta`, unchanged); delta given FULL token-level
capacity (NO 1-slot bottleneck) so capacity is not the confound; decoder cross-attends to [H_A | delta]
and generates the phrase. Single objective = CE on the phrase tokens.

Conditions (held-out + train): full (A + delta) | no_delta (A only) | no_A (delta only).
Metrics: phrase token-acc (teacher-forced), greedy exact-match, greedy token-F1 (order-free overlap).
KEY: full-no_delta = delta content contribution; test full vs train full = generalization gap.

Run: python delta5_gap.py --n 4000 --steps 3000
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model as M
M.A_DROP_P = 0.0
from model import DeltaSystem, D_MODEL, BOS_ID
from delta2_model import DEVICE, MAX_LEN
from delta2_data  import load_wikiatomic_insertions, group_split
from transformers import BertTokenizerFast

SEP_ID = 102
CLS_ID = 101
PHR_LEN = 32                                            # inserted phrases are short


@torch.no_grad()
def encode_pairs(ds, pairs, tok, bs=16):
    HA, AM, HB, BM, PID, PM = [], [], [], [], [], []
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        eA = tok([p["A"] for p in ch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in ch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eP = tok([p["phrase"] for p in ch], max_length=PHR_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        HA.append(ds._enc(A_ids, A_m)); AM.append(A_m)
        HB.append(ds._enc(B_ids, B_m)); BM.append(B_m)
        PID.append(eP["input_ids"].to(DEVICE)); PM.append(eP["attention_mask"].to(DEVICE))
    return dict(H_A=torch.cat(HA), A_m=torch.cat(AM), H_B=torch.cat(HB), B_m=torch.cat(BM),
                P_ids=torch.cat(PID), P_m=torch.cat(PM))


def take(E, idx):
    return {k: v[idx] for k, v in E.items()}


def build_memory(ds, H_A, A_m, delta, B_m, drop_delta=False, no_A=False):
    """memory = [H_A | full token-level delta]; ablate by masking either block."""
    delta_enc = ds.dr_delta_enc(delta, src_key_padding_mask=~B_m.bool())     # [b,T,d]
    A_mask = torch.zeros_like(A_m) if no_A else A_m
    d_mask = torch.zeros_like(B_m) if drop_delta else B_m
    memory = torch.cat([H_A, delta_enc], 1)
    mem_pad = ~torch.cat([A_mask, d_mask], 1).bool()
    return memory, mem_pad


def tf_logits(ds, memory, mem_pad, P_ids, P_m):
    b, T = P_ids.shape; dev = P_ids.device
    shifted = torch.full((b, T), BOS_ID, dtype=P_ids.dtype, device=dev)
    shifted[:, 1:] = P_ids[:, :-1]
    tgt = ds.dr_word_emb(shifted) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
    causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
    out = ds.dr_decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~P_m.bool(),
                        memory_key_padding_mask=mem_pad)
    return ds.dr_lm(out)


@torch.no_grad()
def greedy(ds, memory, mem_pad, max_len=PHR_LEN):
    b, dev = memory.size(0), memory.device
    cur = torch.full((b, 1), BOS_ID, dtype=torch.long, device=dev)
    done = torch.zeros(b, dtype=torch.bool, device=dev)
    for _ in range(max_len - 1):
        T = cur.size(1)
        tgt = ds.dr_word_emb(cur) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
        out = ds.dr_decoder(tgt, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        nxt = ds.dr_lm(out[:, -1]).argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
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


def tok_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    cp, cg = Counter(pred), Counter(gold)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(gold)
    return 2 * p * r / (p + r)


def tf_token_acc(logits, P_ids, P_m):
    pred = logits.argmax(-1)
    m = P_m.bool() & (P_ids != CLS_ID) & (P_ids != SEP_ID)
    return (((pred == P_ids) & m).float().sum() / m.float().sum().clamp(min=1)).item()


@torch.no_grad()
def gen_metrics(ds, E, n_eval, drop_delta=False, no_A=False, bs=16):
    """greedy exact-match + mean token-F1 over the first n_eval examples."""
    em = f1 = cnt = 0
    Ntot = min(n_eval, E["H_A"].size(0))
    for i in range(0, Ntot, bs):
        idx = torch.arange(i, min(i + bs, Ntot), device=DEVICE)
        b = take(E, idx)
        delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
        mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"],
                               drop_delta=drop_delta, no_A=no_A)
        gen = greedy(ds, mem, mp).cpu().numpy()
        gold = b["P_ids"].cpu().numpy()
        for g, gd in zip(gen, gold):
            pt, gt = trim(g), trim(gd)
            em += int(pt == gt); f1 += tok_f1(pt, gt); cnt += 1
    return em / max(cnt, 1), f1 / max(cnt, 1), cnt


@torch.no_grad()
def tf_acc_all(ds, E, drop_delta=False, no_A=False, bs=16):
    tot = w = 0.0
    for i in range(0, E["H_A"].size(0), bs):
        idx = torch.arange(i, min(i + bs, E["H_A"].size(0)), device=DEVICE)
        b = take(E, idx)
        delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
        mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"],
                               drop_delta=drop_delta, no_A=no_A)
        lg = tf_logits(ds, mem, mp, b["P_ids"], b["P_m"])
        tot += tf_token_acc(lg, b["P_ids"], b["P_m"]) * len(idx); w += len(idx)
    return tot / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gen_n", type=int, default=200)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 92)
    print("DELTA-5 GAP on WikiAtomicEdits (Yin's data): predict the INSERTED PHRASE from A + delta")
    print("  (user's idea: missing-span prediction blocks the regeneration shortcut; full-capacity delta)")
    print(f"device={DEVICE} steps={args.steps} bs={args.bs}")
    print("=" * 92)
    pairs = load_wikiatomic_insertions(args.n)
    tr, te = group_split(pairs, test_frac=0.2)
    print(f"edits train {len(tr)} / test {len(te)}")

    ds = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)        # n_slots=0 => full token delta
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
        lg = tf_logits(ds, mem, mp, b["P_ids"], b["P_m"])
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), b["P_ids"].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"  step {step:>4} | loss {loss.item():.3f}")

    ds.eval()
    # teacher-forced phrase token accuracy (held-out), 3 conditions
    tf = {c: tf_acc_all(ds, E_te, drop_delta=(c == "no_delta"), no_A=(c == "no_A"))
          for c in ["full", "no_delta", "no_A"]}
    # greedy exact-match + token-F1 (held-out), 3 conditions
    g_te = {c: gen_metrics(ds, E_te, args.gen_n, drop_delta=(c == "no_delta"), no_A=(c == "no_A"))
            for c in ["full", "no_delta", "no_A"]}
    # greedy on TRAIN (full) -> generalization gap
    g_tr_full = gen_metrics(ds, E_tr, args.gen_n)

    print("\nheld-out TEACHER-FORCED phrase token accuracy:")
    for c in ["full", "no_delta", "no_A"]:
        print(f"  {c:<9} {tf[c]:.3f}")
    print("\nheld-out GREEDY phrase generation (exact-match | token-F1):")
    for c in ["full", "no_delta", "no_A"]:
        em, f1, cnt = g_te[c]
        print(f"  {c:<9} EM {em:.3f} | F1 {f1:.3f}   (n={cnt})")
    print(f"\nTRAIN GREEDY (full): EM {g_tr_full[0]:.3f} | F1 {g_tr_full[1]:.3f}   (n={g_tr_full[2]})")

    d_lift_tf = tf["full"] - tf["no_delta"]
    d_lift_f1 = g_te["full"][1] - g_te["no_delta"][1]
    gap = g_tr_full[1] - g_te["full"][1]
    print("\n" + "=" * 92)
    print(f"delta content contribution (full - no_delta): TF {d_lift_tf:+.3f} | greedy-F1 {d_lift_f1:+.3f}")
    print(f"generalization gap (train F1 - test F1, full): {gap:+.3f}")
    print("READ: large full-no_delta on HELD-OUT + small train-test gap => delta GENUINELY carries content.")
    print("      high train but low test (large gap) + no_delta~test full => delta MEMORIZES, doesn't generalize")
    print("      (instance-specific, confirming the project's central finding).")
    print("=" * 92)


if __name__ == "__main__":
    main()
