"""
test_leak.py -- prove the reconstruction is LEAK-FREE
=====================================================

Decisive masking test: change a FUTURE B token (position p) and verify the
model's predictions at positions t < p do NOT change. If they change -> leak.

Also reports edge-stack causality: E[t] must not change when B[>t] changes.

Run (CPU is fine):  python test_leak.py
"""

import sys
from pathlib import Path

import torch
from transformers import BertTokenizerFast

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from prefix_complement import PrefixComplementLM, MAX_A_LEN, MAX_B_LEN

DEVICE = "cpu"   # leak test is deterministic; CPU avoids any nondeterminism


def main():
    torch.manual_seed(0)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = PrefixComplementLM(k_edge_layers=2, j_dec_layers=1).to(DEVICE).eval()

    A = ["The Eiffel Tower is a landmark located in the city of Paris in France ."]
    B = ["It was completed in the year eighteen eighty nine for the world fair ."]
    ea = tok(A, max_length=MAX_A_LEN, truncation=True, padding="max_length", return_tensors="pt")
    eb = tok(B, max_length=MAX_B_LEN, truncation=True, padding="max_length", return_tensors="pt")

    A_ids, A_m = ea["input_ids"], ea["attention_mask"]
    B_ids, B_m = eb["input_ids"], eb["attention_mask"]

    n_real = int(B_m[0].sum())
    p = n_real - 2                      # perturb a near-final REAL B token (a clear "future")
    assert p >= 3, "need a few real tokens"

    with torch.no_grad():
        logits1, E1 = model(A_ids, A_m, B_ids, B_m)

        # perturb B at position p (change the token id to something else, keep it real)
        B_ids2 = B_ids.clone()
        orig = B_ids2[0, p].item()
        B_ids2[0, p] = (orig + 137) % 30000 + 1
        logits2, E2 = model(A_ids, A_m, B_ids2, B_m)

    # ---- Edge causality: E[t] for t < p must be identical ----
    edge_diff_before = (E1[0, :p] - E2[0, :p]).abs().max().item()
    edge_diff_at_after = (E1[0, p:] - E2[0, p:]).abs().max().item()

    # ---- Prediction leak: logits[t] for t <= p must be identical ----
    # (position t predicts b_t from edge that saw b_<t; changing b_p can only
    #  affect predictions at t > p)
    logit_diff_before = (logits1[0, :p + 1] - logits2[0, :p + 1]).abs().max().item()
    logit_diff_after  = (logits1[0, p + 1:n_real] - logits2[0, p + 1:n_real]).abs().max().item()

    W = 60
    print("=" * W)
    print(f"  LEAK TEST  (perturbed B token at position p={p} of {n_real} real)")
    print("=" * W)
    print(f"  edge:   max|dE| at t<p      = {edge_diff_before:.3e}   (want ~0)")
    print(f"  edge:   max|dE| at t>=p     = {edge_diff_at_after:.3e}   (should be >0)")
    print(f"  logits: max|d| at t<=p      = {logit_diff_before:.3e}   (want ~0)")
    print(f"  logits: max|d| at t>p       = {logit_diff_after:.3e}    (should be >0)")
    print("=" * W)
    tol = 1e-4
    leak_free = (edge_diff_before < tol) and (logit_diff_before < tol)
    responsive = (edge_diff_at_after > tol) and (logit_diff_after > tol)
    if leak_free and responsive:
        print("  RESULT: LEAK-FREE  (past predictions unaffected by future B,")
        print("          and the model IS responsive to the change after p)")
    elif not leak_free:
        print("  RESULT: LEAK DETECTED -- future B changed a past prediction/edge.")
    else:
        print("  RESULT: suspicious — no leak but also not responsive; check masks.")
    print("=" * W)


if __name__ == "__main__":
    main()
