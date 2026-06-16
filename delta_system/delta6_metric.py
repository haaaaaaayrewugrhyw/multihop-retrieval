"""
delta6_metric.py -- THE CORRECT METRIC, applied uniformly to ALL approaches on the SAME data/split.

Question: does an approach's representation r carry "what B adds beyond A" (the inserted content)?
Correct metric (decoder-free, confound-controlled; per the probing literature):
  - recoverability: fit a LINEAR probe r -> phrase-embedding on train, RETRIEVE the correct inserted
    phrase on held-out (top-1 / MRR). Linear probe = low memorization (Hewitt & Liang 2019).
  - selectivity: same probe on SHUFFLED phrase labels (control task); selectivity = real - control.
  - generalization: train-retrieval vs test-retrieval gap (memorization detector).
  - conditional (info BEYOND A): retrieval from [A_rep, r] minus from [A_rep] alone
    (Hewitt/Ethayarajh/Liang 2021 conditional probing -- the principled "beyond baseline").
  - baseline ladder: chance < meandiff(B-A) < fixed_novelB(B@novel) < encB(B) < oracle(phrase).

Approaches compared (same WikiAtomicEdits group-split, same probe):
  zero-train: chance | meandiff | fixed_novelB | encB | oracle
  trained:    ours_recon (subtraction encoder, regenerate B) | ours_sent (same encoder, predict the
              inserted phrase) | yin (diff-aligned edit encoder + copy decoder, regenerate B)
ours_recon vs ours_sent isolates the OBJECTIVE; ours vs yin isolates the ENCODER ARCHITECTURE.

Run: python delta6_metric.py --n 4000 --steps 3000
"""

import argparse
import difflib
import gc
import sys
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
from insertion_cloze_eval import _novel_mask_difflib
from delta4_copy import CopyHead
from transformers import BertTokenizerFast

SEP_ID = 102
CLS_ID = 101
EQ, INS, DEL, REP = 1, 2, 3, 4                          # Yin edit tags (0 = pad)


# ───────────────────────── encoding ─────────────────────────
def _align(a_ids, b_ids):
    """difflib alignment of two token-id lists -> (tokens, tags) for Yin's edit encoder."""
    sm = difflib.SequenceMatcher(a=a_ids, b=b_ids, autojunk=False)
    toks, tags = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            toks += a_ids[i1:i2]; tags += [EQ] * (i2 - i1)
        elif op == "insert":
            toks += b_ids[j1:j2]; tags += [INS] * (j2 - j1)
        elif op == "delete":
            toks += a_ids[i1:i2]; tags += [DEL] * (i2 - i1)
        else:                                          # replace
            toks += a_ids[i1:i2] + b_ids[j1:j2]; tags += [REP] * ((i2 - i1) + (j2 - j1))
    return toks[:MAX_LEN], tags[:MAX_LEN]


@torch.no_grad()
def encode_all(ds, pairs, tok, bs=16):
    out = {k: [] for k in ["H_A", "A_m", "A_ids", "H_B", "B_m", "B_ids", "P_ids", "P_m",
                            "e_phrase", "nov", "al_tok", "al_tag", "al_m"]}
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        eA = tok([p["A"] for p in ch], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in ch], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eP = tok([p["phrase"] for p in ch], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        P_ids, P_m = eP["input_ids"].to(DEVICE), eP["attention_mask"].to(DEVICE)
        H_A, H_B = ds._enc(A_ids, A_m), ds._enc(B_ids, B_m)
        H_P = ds._enc(P_ids, P_m)
        p_content = P_m.bool() & (P_ids != CLS_ID) & (P_ids != SEP_ID)
        e_phrase = _pool(H_P, p_content.float(), P_m.float())
        out["H_A"].append(H_A); out["A_m"].append(A_m); out["A_ids"].append(A_ids); out["H_B"].append(H_B)
        out["B_m"].append(B_m); out["B_ids"].append(B_ids); out["P_ids"].append(P_ids); out["P_m"].append(P_m)
        out["e_phrase"].append(e_phrase)
        # novel mask over B + Yin alignment
        for k, p in enumerate(ch):
            mask, rl = _novel_mask_difflib(p["A"], p["B"], tok)
            nv = np.zeros(MAX_LEN, np.float32)
            if mask is not None:
                rl = min(rl, MAX_LEN); tm = np.ones(rl, bool)
                if rl > 2: tm[0] = False; tm[rl - 1] = False
                nv[:rl] = (mask[:rl] == 1).astype(np.float32) * tm
            out["nov"].append(nv)
            a_raw = [t for t in A_ids[k].tolist() if t not in (0, CLS_ID, SEP_ID)]
            b_raw = [t for t in B_ids[k].tolist() if t not in (0, CLS_ID, SEP_ID)]
            tks, tgs = _align(a_raw, b_raw)
            at = np.zeros(MAX_LEN, np.int64); ag = np.zeros(MAX_LEN, np.int64); am = np.zeros(MAX_LEN, np.float32)
            at[:len(tks)] = tks; ag[:len(tgs)] = tgs; am[:len(tks)] = 1.0
            out["al_tok"].append(at); out["al_tag"].append(ag); out["al_m"].append(am)
    E = {}
    for k in ["H_A", "A_m", "A_ids", "H_B", "B_m", "B_ids", "P_ids", "P_m", "e_phrase"]:
        E[k] = torch.cat(out[k])
    E["nov"] = torch.tensor(np.stack(out["nov"]), device=DEVICE)
    E["al_tok"] = torch.tensor(np.stack(out["al_tok"]), device=DEVICE)
    E["al_tag"] = torch.tensor(np.stack(out["al_tag"]), device=DEVICE)
    E["al_m"] = torch.tensor(np.stack(out["al_m"]), device=DEVICE)
    return E


def _pool(X, primary, fallback):
    """masked mean of X[b,T,d] over `primary`; rows with empty primary fall back to `fallback`."""
    w = primary.unsqueeze(-1)
    s = w.sum(1)
    out = (X * w).sum(1) / s.clamp(min=1)
    empty = (s.squeeze(-1) == 0)
    if empty.any():
        wf = fallback.unsqueeze(-1)
        outf = (X * wf).sum(1) / wf.sum(1).clamp(min=1)
        out[empty] = outf[empty]
    return out


def take(E, idx):
    return {k: v[idx] for k, v in E.items()}


# ───────────────────────── zero-training representations ─────────────────────────
def zero_reps(E):
    novB = _pool(E["H_B"], E["nov"], E["B_m"].float())
    meanB = _pool(E["H_B"], E["B_m"].float(), E["B_m"].float())
    meanA = _pool(E["H_A"], E["A_m"].float(), E["A_m"].float())
    return {
        "chance":       torch.randn_like(novB),
        "meandiff":     meanB - meanA,
        "fixed_novelB": novB,
        "encB":         meanB,
        "oracle":       E["e_phrase"],
    }


def A_rep(E):
    return _pool(E["H_A"], E["A_m"].float(), E["A_m"].float())


# ───────────────────────── OURS: subtraction encoder + decoder ─────────────────────────
def _decode_logits(ds, memory, mem_pad, T_ids, T_m):
    b, T = T_ids.shape; dev = T_ids.device
    shifted = torch.full((b, T), BOS_ID, dtype=T_ids.dtype, device=dev)
    shifted[:, 1:] = T_ids[:, :-1]
    tgt = ds.dr_word_emb(shifted) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
    causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
    out = ds.dr_decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~T_m.bool(),
                        memory_key_padding_mask=mem_pad)
    return ds.dr_lm(out)


def get_ours_reps(target, E_tr, E_te, steps, bs):
    """target='B' (regenerate B) or 'P' (predict the inserted phrase). Returns pooled-delta reps."""
    ds = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0); N = E_tr["H_A"].size(0)
    Tid, Tm = ("B_ids", "B_m") if target == "B" else ("P_ids", "P_m")
    ds.train()
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, N, bs), device=DEVICE); b = take(E_tr, idx)
        delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
        denc = ds.dr_delta_enc(delta, src_key_padding_mask=~b["B_m"].bool())
        memory = torch.cat([b["H_A"], denc], 1)
        mem_pad = ~torch.cat([b["A_m"], b["B_m"]], 1).bool()
        lg = _decode_logits(ds, memory, mem_pad, b[Tid], b[Tm])
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), b[Tid].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 600 == 0:
            print(f"    ours_{target} step {step:>4} | loss {loss.item():.3f}")

    @torch.no_grad()
    def extract(E):
        ds.eval(); reps = []
        for i in range(0, E["H_A"].size(0), 32):
            idx = torch.arange(i, min(i + 32, E["H_A"].size(0)), device=DEVICE); b = take(E, idx)
            delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
            reps.append(_pool(delta, b["nov"], b["B_m"].float()))
        return torch.cat(reps)
    R_tr, R_te = extract(E_tr), extract(E_te)
    del ds, opt; gc.collect(); torch.cuda.empty_cache()
    return R_tr, R_te


# ───────────────────────── YIN: diff-aligned edit encoder + copy decoder ─────────────────────────
class YinEdit(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag_emb = nn.Embedding(5, D_MODEL, padding_idx=0)
        layer = nn.TransformerEncoderLayer(D_MODEL, 8, 2048, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, word_emb, al_tok, al_tag, al_m):
        x = word_emb(al_tok) + self.tag_emb(al_tag)
        h = self.enc(x, src_key_padding_mask=~al_m.bool())
        return self.proj(_pool(h, al_m, al_m))          # [b,d] edit vector f_delta


def get_yin_reps(E_tr, E_te, steps, bs):
    ds = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)
    yin = YinEdit().to(DEVICE); cph = CopyHead().to(DEVICE)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    params += list(yin.parameters()) + list(cph.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0); N = E_tr["H_A"].size(0)
    ds.train(); yin.train(); cph.train()
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, N, bs), device=DEVICE); b = take(E_tr, idx)
        f = yin(ds.dr_word_emb, b["al_tok"], b["al_tag"], b["al_m"])           # [b,d]
        memory = torch.cat([b["H_A"], f.unsqueeze(1)], 1)
        mem_pad = ~torch.cat([b["A_m"], torch.ones(b["A_m"].size(0), 1, device=DEVICE)], 1).bool()
        B_ids, B_m = b["B_ids"], b["B_m"]
        bb, T = B_ids.shape
        shifted = torch.full((bb, T), BOS_ID, dtype=B_ids.dtype, device=DEVICE); shifted[:, 1:] = B_ids[:, :-1]
        tgt = ds.dr_word_emb(shifted) + ds.dr_pos(torch.arange(T, device=DEVICE).unsqueeze(0))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=DEVICE).bool()
        dec = ds.dr_decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~B_m.bool(),
                            memory_key_padding_mask=mem_pad)
        gen_logits = ds.dr_lm(dec)
        cmask = b["A_m"].bool() & (b["A_ids"] != CLS_ID) & (b["A_ids"] != SEP_ID)   # copy over A's real tokens
        P, _ = cph(dec, gen_logits, b["H_A"], b["A_ids"], cmask)
        p_true = P.gather(-1, B_ids.unsqueeze(-1)).squeeze(-1)
        loss = (-(p_true.clamp_min(1e-12).log()) * B_m.float()).sum() / B_m.float().sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 600 == 0:
            print(f"    yin step {step:>4} | loss {loss.item():.3f}")

    @torch.no_grad()
    def extract(E):
        yin.eval(); reps = []
        for i in range(0, E["H_A"].size(0), 64):
            idx = torch.arange(i, min(i + 64, E["H_A"].size(0)), device=DEVICE); b = take(E, idx)
            reps.append(yin(ds.dr_word_emb, b["al_tok"], b["al_tag"], b["al_m"]))
        return torch.cat(reps)
    R_tr, R_te = extract(E_tr), extract(E_te)
    del ds, yin, cph, opt; gc.collect(); torch.cuda.empty_cache()
    return R_tr, R_te


# ───────────────────────── the metric: linear probe + retrieval ─────────────────────────
def _std(Xtr, Xte):
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge(X, Y, lam=10.0):
    Xb = torch.cat([X, torch.ones(X.size(0), 1, device=X.device)], 1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.size(1), device=X.device)
    return torch.linalg.solve(A, Xb.T @ Y)


def _retr(Xte, W, E_target):
    Xb = torch.cat([Xte, torch.ones(Xte.size(0), 1, device=Xte.device)], 1)
    pred = F.normalize(Xb @ W, dim=-1); tgt = F.normalize(E_target, dim=-1)
    sim = pred @ tgt.T
    true = sim.diag().unsqueeze(1)
    rank = (sim >= true).sum(1).float()                                # 1-based rank of the true item
    top1 = (rank == 1).float().mean().item()
    mrr = (1.0 / rank).mean().item()
    return top1, mrr


def evaluate(R_tr, R_te, E_tr, E_te, A_tr, A_te):
    """returns dict: test_top1, mrr, train_top1, gen_gap, selectivity, cond_gain."""
    Xtr, Xte = _std(R_tr, R_te)
    W = _ridge(Xtr, E_tr)
    te_top1, mrr = _retr(Xte, W, E_te)
    tr_top1, _ = _retr(Xtr, W, E_tr)
    # selectivity: shuffle phrase targets in train (control task)
    perm = torch.randperm(E_tr.size(0), device=E_tr.device)
    Wc = _ridge(Xtr, E_tr[perm]); ctrl_top1, _ = _retr(Xte, Wc, E_te)
    # conditional: [A] vs [A, R]
    AtrS, AteS = _std(A_tr, A_te)
    Wa = _ridge(AtrS, E_tr); a_top1, _ = _retr(AteS, Wa, E_te)
    ARtr = torch.cat([AtrS, Xtr], 1); ARte = torch.cat([AteS, Xte], 1)
    War = _ridge(ARtr, E_tr); ar_top1, _ = _retr(ARte, War, E_te)
    return dict(test=te_top1, mrr=mrr, train=tr_top1, gap=tr_top1 - te_top1,
                sel=te_top1 - ctrl_top1, cond=ar_top1 - a_top1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--skip_yin", action="store_true")
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 100)
    print("DELTA-6 UNIFIED METRIC on WikiAtomicEdits: content-retrieval (decoder-free) across all approaches")
    print(f"device={DEVICE} steps={args.steps} bs={args.bs}")
    print("=" * 100)
    pairs = load_wikiatomic_insertions(args.n)
    tr, te = group_split(pairs, test_frac=0.2)
    print(f"edits train {len(tr)} / test {len(te)}")

    ds0 = DeltaSystem(n_slots=0, vib=False, d0_aware=False).to(DEVICE)         # for BERT encodings only
    E_tr, E_te = encode_all(ds0, tr, tok), encode_all(ds0, te, tok)
    A_tr, A_te = A_rep(E_tr), A_rep(E_te)
    Etgt_tr, Etgt_te = E_tr["e_phrase"], E_te["e_phrase"]
    del ds0; gc.collect(); torch.cuda.empty_cache()

    reps = {}
    z_tr, z_te = zero_reps(E_tr), zero_reps(E_te)
    for k in z_tr:
        reps[k] = (z_tr[k], z_te[k])
    print("\ntraining ours_recon (regenerate B) ...")
    reps["ours_recon"] = get_ours_reps("B", E_tr, E_te, args.steps, args.bs)
    print("training ours_sent (predict inserted phrase) ...")
    reps["ours_sent"] = get_ours_reps("P", E_tr, E_te, args.steps, args.bs)
    if not args.skip_yin:
        try:
            print("training yin (diff-aligned edit encoder + copy decoder, regenerate B) ...")
            reps["yin"] = get_yin_reps(E_tr, E_te, args.steps, args.bs)
        except Exception as e:
            print(f"  yin failed ({e}); continuing without it")

    order = ["chance", "meandiff", "encB", "fixed_novelB", "oracle",
             "ours_recon", "ours_sent", "yin"]
    print("\n" + "=" * 100)
    print(f"{'approach':<14}{'test_top1':>10}{'MRR':>8}{'train_top1':>12}{'gen_gap':>9}"
          f"{'selectiv':>10}{'cond(>A)':>10}")
    print("-" * 100)
    for name in order:
        if name not in reps:
            continue
        r = evaluate(reps[name][0], reps[name][1], Etgt_tr, Etgt_te, A_tr, A_te)
        print(f"{name:<14}{r['test']:>10.3f}{r['mrr']:>8.3f}{r['train']:>12.3f}{r['gap']:>+9.3f}"
              f"{r['sel']:>+10.3f}{r['cond']:>+10.3f}")
    print("=" * 100)
    print("READ: test_top1 vs the ladder (chance < meandiff < fixed_novelB <= oracle). A LEARNED rep that")
    print("      carries content should beat meandiff and approach fixed_novelB, with small gen_gap and")
    print("      positive selectivity + cond(>A). High train but low test (large gap) = memorization.")
    print("=" * 100)


if __name__ == "__main__":
    main()
