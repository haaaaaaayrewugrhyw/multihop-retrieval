"""
delta4_copy.py -- delta3's CLEAN setup + a COPY/POINTER mechanism over A (See et al. 2017
pointer-generator). Goal: DECOMPOSE Yin's 0.729 into "how much is copy" vs "how much is delta".

Identical to delta3_clean.py (original arch, 1-slot bottleneck, A_DROP_P=0, drop_d0, single regen
objective, WikiAtomicEdits/Yin data). The ONLY addition is a copy head: at each step it mixes the
decoder's vocab distribution P_vocab with a copy distribution over A's tokens, gated by a learned
p_gen in [0,1]:   P(w) = p_gen * P_vocab(w) + (1 - p_gen) * sum_{i: A_i = w} copy_attn_i.
Loss = NLL over this MIXED probability (not CE over logits).

Held-out conditions:
  full     = A in memory + delta + copy
  no_delta = delta zeroed (copy on)
  no_A     = A removed from memory AND copy off  -> total value of A
  no_copy  = copy off, A still in memory          -> isolates the pointer
Decompositions:
  A-dependence     = full - no_A     (overall)
  copy value       = full - no_copy  (overall)   <-- what the pointer buys
  delta novel-lift = full - no_delta (novel)     (delta's marginal value WITH copy present)
Greedy exact-match reported for full(copy) vs no_copy (vs Yin Acc@1 = 0.729). Also mean p_gen
(overall vs novel) -- low p_gen => copying, high => generating.

Run: python delta4_copy.py --n 4000 --steps 3000   (lower --bs if OOM)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model as M
M.A_DROP_P = 0.0                                   # remove the A-dropout sabotage (as in delta3)
from model import DeltaSystem, D_MODEL, BOS_ID
from delta2_model import DEVICE, MAX_LEN
from delta2_data  import load_wikiatomic_insertions, group_split
from insertion_cloze_eval import _novel_mask_difflib
from transformers import BertTokenizerFast

SEP_ID = 102
CLS_ID = 101


@torch.no_grad()
def encode_pairs(ds, pairs, tok, bs=16):
    HA, AM, AID, HB, BM, BID, NOV = [], [], [], [], [], [], []
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        eA = tok([p["A"] for p in ch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([p["B"] for p in ch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        A_ids, A_m = eA["input_ids"].to(DEVICE), eA["attention_mask"].to(DEVICE)
        B_ids, B_m = eB["input_ids"].to(DEVICE), eB["attention_mask"].to(DEVICE)
        HA.append(ds._enc(A_ids, A_m)); AM.append(A_m); AID.append(A_ids)
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
    return dict(H_A=torch.cat(HA), A_m=torch.cat(AM), A_ids=torch.cat(AID),
                H_B=torch.cat(HB), B_m=torch.cat(BM), B_ids=torch.cat(BID),
                nov=torch.tensor(np.stack(NOV), device=DEVICE))


def take(E, idx):
    return {k: v[idx] for k, v in E.items()}


def build_memory(ds, H_A, A_m, delta, B_m, drop_delta=False, no_A=False):
    """Same memory as delta3 = [zeroed-d0 | H_A | 1-slot delta], drop_d0."""
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


class CopyHead(nn.Module):
    """Pointer-generator head: dedicated single-head attention from the decoder state to A's tokens,
    a copy distribution over A positions, and a learned p_gen mixing generate vs copy."""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.copy_attn = nn.MultiheadAttention(d, 1, batch_first=True)
        self.pgen = nn.Linear(2 * d, 1)

    def forward(self, dec_out, gen_logits, H_A, A_ids, copy_key_mask):
        # dec_out [b,T,d] ; gen_logits [b,T,V] ; H_A [b,La,d] ; A_ids [b,La] ; copy_key_mask [b,La] True=keep
        ctx, attn = self.copy_attn(dec_out, H_A, H_A,
                                   key_padding_mask=~copy_key_mask, need_weights=True)  # attn [b,T,La]
        pgen = torch.sigmoid(self.pgen(torch.cat([dec_out, ctx], -1)))                  # [b,T,1]
        Pvocab = torch.softmax(gen_logits, -1)                                          # [b,T,V]
        m = copy_key_mask.unsqueeze(1).float()                                          # [b,1,La]
        attn = attn * m
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-9)                               # renormalize over A
        final = pgen * Pvocab                                                           # [b,T,V]
        idx = A_ids.unsqueeze(1).expand(-1, dec_out.size(1), -1)                        # [b,T,La]
        final = final.scatter_add(-1, idx, (1.0 - pgen) * attn)                         # mix in copy mass
        return final.clamp_min(1e-12), pgen.squeeze(-1)


def copy_mask_of(A_ids, A_m):
    return A_m.bool() & (A_ids != CLS_ID) & (A_ids != SEP_ID)


def tf_probs(ds, cph, memory, mem_pad, b, use_copy):
    """Teacher-forced mixed probabilities [b,T,V] (+ p_gen [b,T] or None)."""
    B_ids, B_m = b["B_ids"], b["B_m"]
    bsz, T = B_ids.shape; dev = B_ids.device
    shifted = torch.full((bsz, T), BOS_ID, dtype=B_ids.dtype, device=dev)
    shifted[:, 1:] = B_ids[:, :-1]
    tgt = ds.dr_word_emb(shifted) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
    causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
    out = ds.dr_decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~B_m.bool(),
                        memory_key_padding_mask=mem_pad)
    gen_logits = ds.dr_lm(out)
    if not use_copy:
        return torch.softmax(gen_logits, -1), None
    return cph(out, gen_logits, b["H_A"], b["A_ids"], copy_mask_of(b["A_ids"], b["A_m"]))


@torch.no_grad()
def greedy(ds, cph, memory, mem_pad, b, use_copy, max_len):
    H_A, A_ids = b["H_A"], b["A_ids"]
    cmask = copy_mask_of(A_ids, b["A_m"])
    bsz, dev = memory.size(0), memory.device
    cur = torch.full((bsz, 1), BOS_ID, dtype=torch.long, device=dev)
    done = torch.zeros(bsz, dtype=torch.bool, device=dev)
    for _ in range(max_len - 1):
        T = cur.size(1)
        tgt = ds.dr_word_emb(cur) + ds.dr_pos(torch.arange(T, device=dev).unsqueeze(0))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=dev).bool()
        out = ds.dr_decoder(tgt, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        gen_logits = ds.dr_lm(out)
        if use_copy:
            P, _ = cph(out, gen_logits, H_A, A_ids, cmask)
            nxt = P[:, -1].argmax(-1)
        else:
            nxt = gen_logits[:, -1].argmax(-1)
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


def tok_acc(pred, B_ids, B_m, nov=None):
    m = B_m.bool() & (B_ids != BOS_ID) & (B_ids != SEP_ID)
    if nov is not None:
        m = m & (nov > 0.5)
    return ((pred == B_ids) & m).float().sum() / m.float().sum().clamp(min=1)


COND = {                                              # (drop_delta, no_A, use_copy)
    "full":     (False, False, True),
    "no_delta": (True,  False, True),
    "no_A":     (False, True,  False),
    "no_copy":  (False, False, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gen_n", type=int, default=200)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 90)
    print("DELTA-4 COPY on WikiAtomicEdits (Yin's data): delta3 clean + pointer-generator copy over A")
    print(f"device={DEVICE} steps={args.steps} bs={args.bs} A_DROP_P={M.A_DROP_P}")
    print("=" * 90)
    pairs = load_wikiatomic_insertions(args.n)
    tr, te = group_split(pairs, test_frac=0.2)
    print(f"edits train {len(tr)} / test {len(te)}")

    ds = DeltaSystem(n_slots=1, vib=False, d0_aware=False).to(DEVICE)
    cph = CopyHead().to(DEVICE)
    E_tr = encode_pairs(ds, tr, tok); E_te = encode_pairs(ds, te, tok)
    params = [p for n, p in ds.named_parameters() if not n.startswith("bert.") and p.requires_grad]
    params += list(cph.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = np.random.default_rng(0); N = E_tr["H_A"].size(0)

    ds.train(); cph.train()
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(rng.integers(0, N, args.bs), device=DEVICE)
        b = take(E_tr, idx)
        delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
        mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"])
        P, _ = tf_probs(ds, cph, mem, mp, b, use_copy=True)
        p_true = P.gather(-1, b["B_ids"].unsqueeze(-1)).squeeze(-1)        # [b,T]
        m = b["B_m"].float()                                              # ignore PAD only (as delta3)
        loss = (-(p_true.clamp_min(1e-12).log()) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"  step {step:>4} | loss {loss.item():.3f}")

    ds.eval(); cph.eval()
    res = {}
    pg_ov = pg_ovn = pg_nv = pg_nvn = 0.0
    Nte = E_te["H_A"].size(0)
    with torch.no_grad():
        for cond in ["full", "no_delta", "no_A", "no_copy"]:
            dd, na, uc = COND[cond]
            ov = nv = n = 0.0
            for i in range(0, Nte, 16):
                idx = torch.arange(i, min(i + 16, Nte), device=DEVICE)
                b = take(E_te, idx)
                delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
                mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"],
                                       drop_delta=dd, no_A=na)
                P, pgen = tf_probs(ds, cph, mem, mp, b, use_copy=uc)
                pred = P.argmax(-1)
                ov += tok_acc(pred, b["B_ids"], b["B_m"]).item() * len(idx)
                nv += tok_acc(pred, b["B_ids"], b["B_m"], b["nov"]).item() * len(idx); n += len(idx)
                if cond == "full" and pgen is not None:
                    vm = (b["B_m"].bool() & (b["B_ids"] != BOS_ID) & (b["B_ids"] != SEP_ID)).float()
                    nvm = vm * (b["nov"] > 0.5).float()
                    pg_ov += (pgen * vm).sum().item(); pg_ovn += vm.sum().item()
                    pg_nv += (pgen * nvm).sum().item(); pg_nvn += nvm.sum().item()
            res[cond] = (ov / n, nv / n)

    # Yin-comparable greedy exact-match: with copy vs without copy (both with full A+delta memory)
    def em_for(use_copy):
        em = n = 0
        for i in range(0, min(args.gen_n, Nte), 16):
            idx = torch.arange(i, min(i + 16, min(args.gen_n, Nte)), device=DEVICE)
            b = take(E_te, idx)
            delta, _d0, _ = ds.generate_delta(b["H_A"], b["A_m"], b["H_B"], b["B_m"])
            mem, mp = build_memory(ds, b["H_A"], b["A_m"], delta, b["B_m"])
            gen = greedy(ds, cph, mem, mp, b, use_copy, MAX_LEN).cpu().numpy()
            B = b["B_ids"].cpu().numpy()
            for g, bb in zip(gen, B):
                em += int(trim(g) == trim(bb)); n += 1
        return em / max(n, 1), n

    with torch.no_grad():
        em_full, n_em = em_for(True)
        em_nocopy, _ = em_for(False)

    print("\nheld-out token accuracy (overall | novel):")
    for c in ["full", "no_delta", "no_A", "no_copy"]:
        print(f"  {c:<9} overall {res[c][0]:.3f} | novel {res[c][1]:.3f}")
    print(f"\nmean p_gen (full): overall {pg_ov / max(pg_ovn,1):.3f} | novel {pg_nv / max(pg_nvn,1):.3f}"
          f"   (low => copying A, high => generating)")
    print(f"\nGREEDY exact-match over {n_em}:  full(copy) = {em_full:.3f}   |   no_copy = {em_nocopy:.3f}"
          f"   (Yin Acc@1 = 0.729)")

    a_dep = res["full"][0] - res["no_A"][0]
    copy_val = res["full"][0] - res["no_copy"][0]
    d_lift = res["full"][1] - res["no_delta"][1]
    print("\n" + "=" * 90)
    print(f"A-dependence    (full - no_A,    overall) = {a_dep:+.3f}")
    print(f"COPY value      (full - no_copy, overall) = {copy_val:+.3f}   <-- what the pointer buys")
    print(f"delta novel-lift(full - no_delta, novel)  = {d_lift:+.3f}   <-- delta's marginal value w/ copy")
    print("READ: em_full >> em_nocopy and copy_val large => Yin's 0.729 is mostly COPY, not delta.")
    print("      delta novel-lift still ~0 => the learned delta still doesn't carry the inserted content.")
    print("=" * 90)


if __name__ == "__main__":
    main()
