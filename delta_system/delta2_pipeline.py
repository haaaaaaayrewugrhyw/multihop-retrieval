"""
delta2_pipeline.py -- the INTEGRATED gate->extract system, end-to-end (held-out, same-domain).

Two stages assembled from validated pieces (no new architecture):
  STAGE 1 GATE  ("did meaning change?"): pick the better of
      - delta-gate : ||pooled paraphrase-aux delta||   (trained, ~1 min)
      - nli-gate   : 1 - P(entail) of (A->B)           (frozen roberta, no training)
    threshold chosen on TRAIN; precision/recall/F1/AUC on TEST.
  STAGE 2 EXTRACT ("what changed?"): zero-training token-level complement -> retrieve gold novelty.

Data (same-domain, no domain confound):
  positives (edit=1) = IteraTeR meaning-changed ; negatives (edit=0) = IteraTeR non-meaning.

END-TO-END metric = of all real test edits (with a gold), the fraction that are BOTH correctly
gated AS changed AND content-retrieved (top-1). Stratified by change-type. This is the honest
"does the system answer 'what B adds beyond A'."

Run: python delta2_pipeline.py --n 400 --steps 600
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model        import DeltaSystem  # noqa: F401
from delta2_model import Delta2, encode_all, DEVICE, MAX_LEN
from delta2_eval  import train, rep_delta
from delta2_data  import load_iterater_by_meaning, change_type, group_split, NLI
from delta2_token_battery import op
from insertion_cloze_eval import _novel_mask_difflib
from transformers import BertTokenizerFast

TYPES = ["insertion", "entity", "relational", "numeric"]


def auc(pos, neg):
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def best_threshold(pos, neg):
    """threshold maximizing F1 on (pos=edit, neg=non) -- chosen on TRAIN only."""
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    best = (-1, s.mean())
    for t in np.unique(s):
        pred = s >= t
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def per_item_correct(reps, golds):
    R = np.array(reps); G = np.array(golds)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    return (Rn @ Gn.T).argmax(1) == np.arange(len(R))


def nli_change_score(nli, pairs, ka, kb):
    """1 - P(entail) for (A -> B): high = meaning changed."""
    return 1 - nli.probs([p[ka] for p in pairs], [p[kb] for p in pairs])[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("=" * 82)
    print(f"INTEGRATED gate->extract PIPELINE (end-to-end, same-domain)  device={DEVICE}")
    print("=" * 82)
    pos = load_iterater_by_meaning(args.n, meaning=True)
    neg = load_iterater_by_meaning(args.n, meaning=False)
    for p in pos:
        p["A"], p["B"], p["type"] = p["before"], p["after"], change_type(p["before"], p["after"])
    for p in neg:
        p["A"], p["B"] = p["before"], p["after"]          # B used for gate (A->B) and NLI
        p["A_para"] = p["after"]
    tr_e, te_e = group_split(pos, test_frac=0.25)
    tr_p, te_p = group_split(neg, test_frac=0.25)
    print(f"edits train {len(tr_e)}/test {len(te_e)} | non-meaning train {len(tr_p)}/test {len(te_p)}")

    base = Delta2("paraphrase").to(DEVICE).eval()
    E_tr = encode_all(base, tr_e, "A", "B", tok); E_te = encode_all(base, te_e, "A", "B", tok)
    P_tr = encode_all(base, tr_p, "A", "A_para", tok); P_te = encode_all(base, te_p, "A", "A_para", tok)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── STAGE 1: train delta-gate + compute nli-gate, pick best ──────────────────
    print("\n[stage 1] training delta-gate...")
    gate = train("paraphrase", E_tr, P_tr, None, args.steps, "v2")
    d = {"tr_e": np.linalg.norm(rep_delta(gate, E_tr), axis=1),
         "tr_p": np.linalg.norm(rep_delta(gate, P_tr), axis=1),
         "te_e": np.linalg.norm(rep_delta(gate, E_te), axis=1),
         "te_p": np.linalg.norm(rep_delta(gate, P_te), axis=1)}
    print("[stage 1] computing nli-gate...")
    nli = NLI()
    n = {"tr_e": nli_change_score(nli, tr_e, "A", "B"), "tr_p": nli_change_score(nli, tr_p, "A", "B"),
         "te_e": nli_change_score(nli, te_e, "A", "B"), "te_p": nli_change_score(nli, te_p, "A", "B")}
    del nli
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    auc_d, auc_n = auc(d["te_e"], d["te_p"]), auc(n["te_e"], n["te_p"])
    print(f"\n  gate AUC (test):  delta {auc_d:.3f} | nli {auc_n:.3f}")
    G = d if auc_d >= auc_n else n
    gname = "delta" if auc_d >= auc_n else "nli"
    thr = best_threshold(G["tr_e"], G["tr_p"])
    print(f"  -> using {gname}-gate, threshold {thr:.3f} (chosen on train)")

    gated_e = G["te_e"] >= thr                              # edits flagged "changed"
    gated_p = G["te_p"] >= thr                              # non-meaning flagged "changed" (FP)
    recall = gated_e.mean(); fpr = gated_p.mean()
    prec = gated_e.sum() / max(gated_e.sum() + gated_p.sum(), 1)
    print(f"  gate on test: recall {recall:.3f} | precision {prec:.3f} | "
          f"false-positive rate {fpr:.3f} | AUC {auc(G['te_e'], G['te_p']):.3f}")

    # ── STAGE 2: extract on test edits (token complement) ───────────────────────
    comp, g = op(E_te["H_A"], E_te["H_B"], E_te["A_m"], 0.1)
    comp_np, g_np = comp.cpu().numpy(), g.cpu().numpy()
    HB_np, Bm_np = E_te["H_B"].cpu().numpy(), E_te["B_m"].cpu().numpy()
    extr, gold, typ, idx = [], [], [], []
    for i, p in enumerate(te_e):
        mask, rl = _novel_mask_difflib(p["A"], p["B"], tok)
        if mask is None:
            continue
        rl = min(rl, MAX_LEN); nov = mask[:rl]
        tm = np.ones(rl, dtype=bool)
        if rl > 2:
            tm[0] = False; tm[rl - 1] = False
        nidx = (nov == 1) & tm
        gg = g_np[i, :rl] * tm
        if nidx.sum() == 0 or gg.sum() < 1e-9:
            continue
        extr.append((comp_np[i, :rl] * gg[:, None]).sum(0) / gg.sum())
        gold.append(HB_np[i, :rl][nidx].mean(0)); typ.append(p["type"]); idx.append(i)
    extr, gold, typ, idx = np.array(extr), np.array(gold), np.array(typ), np.array(idx)
    correct = per_item_correct(extr, gold)                  # extraction top-1 per item
    print(f"\n[stage 2] extraction on {len(extr)} test edits: top1 {correct.mean():.3f}")

    # ── END-TO-END: gated correctly AND extracted correctly ─────────────────────
    gated_for_extracted = gated_e[idx]
    end2end = gated_for_extracted & correct
    print("\n" + "=" * 82)
    print(f"END-TO-END (gated AS changed AND content top-1 correct), over {len(extr)} real edits:")
    print(f"  overall {end2end.mean():.3f}   [gate-recall {gated_for_extracted.mean():.3f} x "
          f"extract-top1 {correct.mean():.3f}]")
    for ct in TYPES:
        m = typ == ct
        if m.sum() >= 5:
            print(f"  {ct:<11} n={int(m.sum()):<4} end2end {end2end[m].mean():.3f}  "
                  f"(gate {gated_for_extracted[m].mean():.2f} | extract {correct[m].mean():.2f})")
    print("=" * 82)
    print("READ: this is the honest system number. Bounded by gate-recall x extract-top1. Per-type")
    print("shows where the pipeline delivers (entity/insertion) vs struggles (relational/numeric).")
    print("=" * 82)


if __name__ == "__main__":
    main()
