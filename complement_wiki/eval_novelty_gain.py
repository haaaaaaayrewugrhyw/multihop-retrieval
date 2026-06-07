"""
eval_novelty_gain.py -- the FAIR thesis test (native generative space)
======================================================================

The MaxSim recovery test failed at RANDOM level because the edge and encode_text
live in two different vector spaces (decoder-stack output vs raw BERT output) that
training never aligned -- so cross-space dot-products are meaningless.

This test avoids cross-space comparison entirely. It asks the thesis question in
the model's OWN generative space:

    Does the edge's reconstruction help concentrate on the NOVEL tokens
    (the inserted phrase) rather than the COPIED tokens (already in A)?

  per B-token t:
     nll_edge[t]  = -log p(b_t | A, edge-of-prefix)        (full model)
     nll_abl[t]   = -log p(b_t | A only, edge zeroed)      (ablate_edge=True)
     delta[t]     = nll_abl[t] - nll_edge[t]               (how much the edge helps)

  Split B tokens into NOVEL (inside the inserted-phrase span) vs COPIED (the rest).
  Copied tokens: A supplies them by cross-attention -> edge should help LITTLE.
  Novel tokens : A cannot supply them            -> edge should help A LOT.

Success (thesis): mean delta(novel) >> mean delta(copied). That means the edge
specifically carries "what B adds", learned with NO labels (the phrase span is
used ONLY here at eval time, never as a training target).

Usage:  python eval_novelty_gain.py --n 1000 --pool 124000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "retrieval_v2"))
from prefix_complement import PrefixComplementLM, MAX_A_LEN, MAX_B_LEN
from data_wikiedits import load_triples

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = _HERE / "models"
PAD_ID    = 0


def find_span(hay, needle):
    """Return (start, end) of the first contiguous occurrence of needle in hay, else None."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if hay[i:i + m] == needle:
            return i, i + m
    return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="prefix_complement_best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=124000)
    ap.add_argument("--k_edge", type=int, default=2)
    ap.add_argument("--j_dec", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    print(f"Device: {DEVICE} | ckpt: {args.ckpt}")
    pool = load_triples(max_examples=args.pool, cache=True)
    triples = pool[:args.n]                       # held-out val slice (never gradient-trained)
    print(f"Eval triples (held-out): {len(triples)}")

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = PrefixComplementLM(k_edge_layers=args.k_edge, j_dec_layers=args.j_dec).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / args.ckpt, map_location=DEVICE), strict=False)
    model.eval()

    novel_deltas, copied_deltas = [], []
    per_ex_novel, per_ex_copied = [], []   # per-example means (for the win-rate)
    matched = 0

    for s in range(0, len(triples), args.bs):
        chunk = triples[s:s + args.bs]
        A = [t["base"] for t in chunk]
        B = [t["edited"] for t in chunk]
        P = [t["inserted"] for t in chunk]

        ea = tok(A, max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
        eb = tok(B, max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")
        Aid, Am = ea["input_ids"].to(DEVICE), ea["attention_mask"].to(DEVICE)
        Bid, Bm = eb["input_ids"].to(DEVICE), eb["attention_mask"].to(DEVICE)

        logits_e, _ = model(Aid, Am, Bid, Bm)                       # full
        logits_a, _ = model(Aid, Am, Bid, Bm, ablate_edge=True)     # A only

        logp_e = F.log_softmax(logits_e.float(), dim=-1)
        logp_a = F.log_softmax(logits_a.float(), dim=-1)
        tgt = Bid.unsqueeze(-1)
        nll_e = -logp_e.gather(-1, tgt).squeeze(-1)                 # [b,TB]
        nll_a = -logp_a.gather(-1, tgt).squeeze(-1)
        delta = (nll_a - nll_e).cpu().numpy()                      # how much edge helps

        for bi, phrase in enumerate(P):
            b_ids = Bid[bi].cpu().tolist()
            real_len = int(Bm[bi].sum().item())
            p_ids = tok(phrase, add_special_tokens=False)["input_ids"]
            span = find_span(b_ids[:real_len], p_ids)
            if span is None:
                continue                                            # phrase truncated/retokenized -> skip
            matched += 1
            lo, hi = span
            nov, cop = [], []
            for t in range(1, real_len - 1):                        # skip [CLS]@0 and [SEP]@last
                (nov if lo <= t < hi else cop).append(float(delta[bi, t]))
            if nov and cop:
                novel_deltas.extend(nov); copied_deltas.extend(cop)
                per_ex_novel.append(np.mean(nov)); per_ex_copied.append(np.mean(cop))

    nov = np.array(novel_deltas); cop = np.array(copied_deltas)
    pn = np.array(per_ex_novel);  pc = np.array(per_ex_copied)
    win = float((pn > pc).mean()) if len(pn) else 0.0

    print("\n" + "=" * 60)
    print(f"  NOVELTY-LOCALIZED RECON-GAIN   (matched {matched}/{len(triples)} examples)")
    print("  delta = nll(A only) - nll(A + edge)   [higher = edge helps more]")
    print("=" * 60)
    print(f"  NOVEL  (inserted phrase) tokens: n={len(nov):6d}  mean delta = {nov.mean():+.4f}")
    print(f"  COPIED (already in A)    tokens: n={len(cop):6d}  mean delta = {cop.mean():+.4f}")
    print(f"  ratio  novel/copied            : {nov.mean()/ (cop.mean() if abs(cop.mean())>1e-9 else 1e-9):+.2f}x")
    print(f"  per-example win-rate (novel>copied): {win:.3f}")
    print("=" * 60)
    if nov.mean() > 2 * max(cop.mean(), 1e-6) and win > 0.6:
        print("  THESIS SUPPORTED: the edge's help concentrates on the ADDED content,")
        print("  not on copied content -- it captured 'what B adds', with NO labels.")
    elif nov.mean() > cop.mean():
        print("  PARTIAL: edge helps novel tokens more than copied, but not decisively.")
    else:
        print("  NEGATIVE: edge does not specifically help the added tokens.")
    print("=" * 60)
    return {"novel": float(nov.mean()), "copied": float(cop.mean()), "win": win, "matched": matched}


if __name__ == "__main__":
    main()
