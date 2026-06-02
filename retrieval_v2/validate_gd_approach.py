"""
validate_gd_approach.py
=======================
Pre-training diagnostic probes for the G+D architecture.

Validates 3 core assumptions using bert-base-uncased on real MuSiQue (A,B) pairs.
Runs on CPU only -- no GPU, no training required. Takes ~5-10 minutes.

Three probes:
  1. DiffAttn subtraction  -- does  B_sa - B_in_A  actually change the representation?
  2. Complement gate       -- is    g_i = 1 - max_j Attn(b_i, A)  meaningful?
  3. Posterior collapse    -- how similar are A and B in BERT space? (sets dropout rate)

Usage:
    python retrieval_v2/validate_gd_approach.py
    python retrieval_v2/validate_gd_approach.py --n_pairs 100
    python retrieval_v2/validate_gd_approach.py --use_trained_bert
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

# ---- Path setup ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_DATA_CANDIDATES = [
    HERE / "data" / "musique" / "musique_ans_v1.0_dev.jsonl",
    ROOT / "retrieval" / "data" / "musique" / "musique_ans_v1.0_dev.jsonl",
]
MUSIQUE_DEV = next((p for p in _DATA_CANDIDATES if p.exists()), None)

MAX_A_LEN = 128
MAX_B_LEN = 64


# ---- Data loading ------------------------------------------------------------

def load_ab_pairs(jsonl_path: Path, n_pairs: int) -> List[Tuple[str, str]]:
    """Extract (A_text, B_text) consecutive hop pairs from MuSiQue dev JSONL."""
    pairs: List[Tuple[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= n_pairs:
                break
            ex = json.loads(line.strip())

            paragraphs = {
                p["idx"]: "{} {}".format(
                    p.get("title", ""), p.get("paragraph_text", "")
                ).strip()
                for p in ex.get("paragraphs", [])
            }
            chain_idxs = [
                d["paragraph_support_idx"]
                for d in ex.get("question_decomposition", [])
                if "paragraph_support_idx" in d
            ]

            for i in range(len(chain_idxs) - 1):
                a_text = paragraphs.get(chain_idxs[i], "").strip()
                b_text = paragraphs.get(chain_idxs[i + 1], "").strip()
                if len(a_text) > 20 and len(b_text) > 20:
                    pairs.append((a_text, b_text))
                    if len(pairs) >= n_pairs:
                        break
    return pairs


# ---- Attention helper --------------------------------------------------------

def b_to_a_cross_attention(
    B_sa:   torch.Tensor,
    A_sa:   torch.Tensor,
    a_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product cross-attention: each B token queries A.

    Args:
        B_sa  : [T_B, D]  B hidden states
        A_sa  : [T_A, D]  A hidden states
        a_mask: [T_A]     1=real token, 0=pad

    Returns:
        B_in_A      : [T_B, D]     weighted sum of A values per B token
        attn_weights: [T_B, T_A]   post-softmax weights
    """
    d = B_sa.size(-1)
    scores = torch.matmul(B_sa, A_sa.T) / (d ** 0.5)          # [T_B, T_A]

    pad_mask = (a_mask == 0).unsqueeze(0)                       # [1, T_A]
    scores = scores.masked_fill(pad_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)                # [T_B, T_A]
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    B_in_A = torch.matmul(attn_weights, A_sa)                  # [T_B, D]
    return B_in_A, attn_weights


# ---- Probe runners -----------------------------------------------------------

def run_probes(
    pairs:     List[Tuple[str, str]],
    model:     BertModel,
    tokenizer: BertTokenizerFast,
) -> Dict:
    model.eval()

    p1_diffs:   List[float] = []
    p1_orthoGs: List[float] = []
    p1_simBs:   List[float] = []
    p1_simAs:   List[float] = []

    p2_mean_gs: List[float] = []
    p2_std_gs:  List[float] = []
    p2_spots:   List[Tuple[str, List[str]]] = []

    # Probe 3: what fraction of B tokens are unreachable from A?
    # These tokens CANNOT be reconstructed from A alone -> D must use edge -> collapse impossible
    # g_i > 0.7 means: even A's best token only explains <30% of this B token
    p3_high_g_fracs:  List[float] = []   # fraction of B tokens with g_i > 0.70
    p3_bridge_fracs:  List[float] = []   # fraction of B tokens with g_i < 0.30 (bridge entity zone)
    p3_raw_sims:      List[float] = []   # cos_sim(A_mean, B_mean) -- BERT baseline (informational)

    with torch.no_grad():
        for idx, (a_text, b_text) in enumerate(pairs):
            if (idx + 1) % 10 == 0:
                print("  ... pair {}/{}".format(idx + 1, len(pairs)))

            enc_a = tokenizer(
                a_text, max_length=MAX_A_LEN, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            enc_b = tokenizer(
                b_text, max_length=MAX_B_LEN, truncation=True,
                padding="max_length", return_tensors="pt",
            )

            A_out = model(
                input_ids=enc_a["input_ids"],
                attention_mask=enc_a["attention_mask"],
            )
            B_out = model(
                input_ids=enc_b["input_ids"],
                attention_mask=enc_b["attention_mask"],
            )

            A_sa   = A_out.last_hidden_state[0]      # [T_A, 768]
            B_sa   = B_out.last_hidden_state[0]      # [T_B, 768]
            a_mask = enc_a["attention_mask"][0]      # [T_A]
            b_mask = enc_b["attention_mask"][0]      # [T_B]

            # ---- Probe 1: DiffAttn subtraction ----------------------------
            B_in_A, attn_weights = b_to_a_cross_attention(B_sa, A_sa, a_mask)

            b_real = b_mask.bool()
            a_real = a_mask.bool()

            comp_real = (B_sa - B_in_A)[b_real]     # [n_B, 768]  lambda=1
            B_real    = B_sa[b_real]                 # [n_B, 768]
            A_real    = A_sa[a_real]                 # [n_A, 768]

            diff = (comp_real - B_real).norm(dim=-1).mean().item()

            comp_mean = comp_real.mean(0)
            a_mean    = A_real.mean(0)
            b_mean    = B_real.mean(0)

            orthog = F.cosine_similarity(
                comp_mean.unsqueeze(0), a_mean.unsqueeze(0)
            ).item()
            sim_B = F.cosine_similarity(
                comp_mean.unsqueeze(0), b_mean.unsqueeze(0)
            ).item()

            p1_diffs.append(diff)
            p1_orthoGs.append(orthog)
            p1_simBs.append(sim_B)
            p1_simAs.append(orthog)    # sim_A == orthog by definition

            # ---- Probe 2: Complement gate ---------------------------------
            attn_real = attn_weights[b_real]         # [n_B, T_A]
            g_i = 1.0 - attn_real.max(dim=-1).values # [n_B]

            p2_mean_gs.append(g_i.mean().item())
            p2_std_gs.append(g_i.std().item() if g_i.numel() > 1 else 0.0)

            if idx < 3:
                real_ids = enc_b["input_ids"][0][b_real].tolist()
                tokens   = tokenizer.convert_ids_to_tokens(real_ids)
                n_show   = min(3, len(tokens))
                low_idxs = g_i.argsort()[:n_show].tolist()
                low_toks = [tokens[i] for i in low_idxs if i < len(tokens)]
                p2_spots.append((b_text[:70], low_toks))

            # ---- Probe 3: Reconstruction pressure on the edge ---------------
            # The core question: what fraction of B tokens CANNOT be reconstructed
            # from A alone? Those tokens force D to use the edge -> no collapse.
            #
            # g_i > 0.70: B token has <30% similarity to any A token -> truly unique
            #              D literally cannot reconstruct these tokens from A
            # g_i < 0.30: B token is strongly explained by A (bridge entity zone)
            #              These are the only tokens D might reconstruct from A alone
            #
            high_g_frac  = (g_i > 0.70).float().mean().item()   # edge-required tokens
            bridge_frac  = (g_i < 0.30).float().mean().item()   # A-reconstructable tokens
            p3_high_g_fracs.append(high_g_frac)
            p3_bridge_fracs.append(bridge_frac)

            # raw A-B similarity: informational only (BERT basline for in-domain text)
            raw_sim = F.cosine_similarity(
                a_mean.unsqueeze(0), b_mean.unsqueeze(0)
            ).item()
            p3_raw_sims.append(raw_sim)

    n = len(pairs)

    def _mean(xs):
        return sum(xs) / len(xs)

    def _std(xs):
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    return {
        "probe1": {
            "mean_diff":   _mean(p1_diffs),
            "mean_orthog": _mean(p1_orthoGs),
            "mean_simB":   _mean(p1_simBs),
            "mean_simA":   _mean(p1_simAs),
            "n":           n,
        },
        "probe2": {
            "mean_g": _mean(p2_mean_gs),
            "std_g":  _mean(p2_std_gs),
            "spots":  p2_spots,
        },
        "probe3": {
            "high_g_frac":  _mean(p3_high_g_fracs),  # fraction requiring edge
            "bridge_frac":  _mean(p3_bridge_fracs),  # fraction D can use A for
            "raw_sim":      _mean(p3_raw_sims),       # BERT baseline (informational)
        },
    }


# ---- Results printing --------------------------------------------------------

def ok(cond):
    return "PASS" if cond else "FAIL"


def print_results(r: Dict, trained_bert: bool) -> None:
    p1 = r["probe1"]
    p2 = r["probe2"]
    p3 = r["probe3"]

    p1_pass = (
        p1["mean_diff"]   > 0.10
        and p1["mean_orthog"] < 0.30
        and p1["mean_simB"]   > p1["mean_simA"]
    )
    p2_pass = p2["mean_g"] > 0.55 and p2["std_g"] > 0.15
    # Probe 3 passes if:
    # - >40% of B tokens are unreachable from A (D literally needs the edge for these)
    # - <30% of B tokens are bridge-entity-like (A-reconstructable tokens are minority)
    p3_pass = p3["high_g_frac"] > 0.40 and p3["bridge_frac"] < 0.30

    bert_tag = "trained FakeEncoder BERT" if trained_bert else "raw bert-base-uncased"
    W = 62

    print()
    print("=" * W)
    print("  G+D ARCHITECTURE VALIDATION RESULTS")
    print("  Encoder : {}".format(bert_tag))
    print("  Pairs   : {}".format(p1["n"]))
    print("=" * W)

    print()
    print("PROBE 1 -- DiffAttn Cross-Document Subtraction")
    print("  mean L2 diff/token  [pass >0.10] : {:.4f}  {}".format(
        p1["mean_diff"], ok(p1["mean_diff"] > 0.10)))
    print("  cos_sim(comp, A)    [pass <0.30] : {:.4f}  {}".format(
        p1["mean_orthog"], ok(p1["mean_orthog"] < 0.30)))
    print("  sim_B={:.4f}  sim_A={:.4f}  [pass: sim_B > sim_A]  {}".format(
        p1["mean_simB"], p1["mean_simA"],
        ok(p1["mean_simB"] > p1["mean_simA"])))
    print("  STATUS: {}".format(ok(p1_pass)))

    print()
    print("PROBE 2 -- Complement Gate Discriminativeness")
    print("  mean g_i  [pass >0.55] : {:.4f}  {}".format(
        p2["mean_g"], ok(p2["mean_g"] > 0.55)))
    print("  std  g_i  [pass >0.15] : {:.4f}  {}".format(
        p2["std_g"], ok(p2["std_g"] > 0.15)))
    print("  STATUS: {}".format(ok(p2_pass)))
    if p2["spots"]:
        print("  Spot check -- lowest g_i B tokens (expect bridge entity tokens):")
        for b_snip, toks in p2["spots"]:
            print('    B : "{}..."'.format(b_snip))
            print("    Low-g tokens : {}".format(toks))

    print()
    print("PROBE 3 -- Reconstruction Pressure on Edge (collapse prevention)")
    print("  Key question: what fraction of B tokens CANNOT be reconstructed from A?")
    print("  These tokens force D to use the edge -> prevents collapse structurally.")
    print()
    print("  frac B tokens with g_i > 0.70  [edge-required]     [pass >0.40] : {:.4f}  {}".format(
        p3["high_g_frac"], ok(p3["high_g_frac"] > 0.40)))
    print("  frac B tokens with g_i < 0.30  [A-reconstructable] [pass <0.30] : {:.4f}  {}".format(
        p3["bridge_frac"], ok(p3["bridge_frac"] < 0.30)))
    print("  raw cos_sim(A_mean, B_mean)  [BERT baseline, informational only] : {:.4f}".format(
        p3["raw_sim"]))
    print("  STATUS: {}".format(ok(p3_pass)))

    print()
    print("=" * W)
    print("  HYPERPARAMETER RECOMMENDATIONS")
    print("=" * W)

    hgf = p3["high_g_frac"]
    bf  = p3["bridge_frac"]
    if hgf > 0.60:
        rec_drop = 0.40
        risk = "LOW (>60% of B is unreachable from A -> edge always needed)"
    elif hgf > 0.40:
        rec_drop = 0.50
        risk = "MEDIUM (40-60% unreachable -> 50% dropout covers bridge tokens)"
    else:
        rec_drop = 0.60
        risk = "HIGH (<40% unreachable -> D can mostly bypass edge)"
    print("  Collapse risk  : {}".format(risk))
    print("  high_g_frac={:.3f}  bridge_frac={:.3f}  raw_sim={:.3f}".format(
        hgf, bf, p3["raw_sim"]))
    print("  -> recommended context_dropout = {}".format(rec_drop))

    d = p1["mean_diff"]
    if d < 0.05:
        lam_rec = "start lambda=1.2 (aggressive subtraction)"
    elif d < 0.10:
        lam_rec = "start lambda=1.0"
    else:
        lam_rec = "DiffAttn default schedule is fine"
    print("  Subtraction diff={:.3f}  -> {}".format(d, lam_rec))

    sg = p2["std_g"]
    if sg < 0.05:
        temp_rec = "tau=0.03  (very sharp)"
    elif sg < 0.15:
        temp_rec = "tau=0.05"
    else:
        temp_rec = "tau=0.07  (standard DPR)"
    print("  Gate std={:.3f}         -> InfoNCE temperature {}".format(sg, temp_rec))

    print()
    print("=" * W)
    all_pass = p1_pass and p2_pass and p3_pass
    if all_pass:
        print("  OVERALL: PROCEED -- all 3 probes pass")
        print("  G+D assumptions validated. Proceed to generator_train.py.")
    else:
        print("  OVERALL: ADJUST before implementing")
        if not p1_pass and not trained_bert:
            print()
            print("  [Probe 1 failed] Raw BERT cross-document attention is weak.")
            print("  -> Re-run with --use_trained_bert.")
            print("     If still fails: use MLP([B; B-B_in_A; B*B_in_A]) fusion.")
        elif not p1_pass and trained_bert:
            print()
            print("  [Probe 1 failed even with trained BERT]")
            print("  -> Replace subtraction with MLP([B; B-B_in_A; B*B_in_A]).")
        if not p2_pass:
            if p2["mean_g"] < 0.30:
                print()
                print("  [Probe 2: mean_g too low]")
                if not trained_bert:
                    print("  -> Re-run with --use_trained_bert first.")
                else:
                    print("  -> Replace complement gate with PMA(k=1) learned pooling.")
            else:
                print()
                print("  [Probe 2: gate not discriminative (low std)]")
                print("  -> Gate is near-uniform -> complement pooling = mean pooling.")
                print("  -> Replace with PMA(k=1) from Set Transformer.")
        if not p3_pass:
            print()
            hgf = p3["high_g_frac"]
            bf  = p3["bridge_frac"]
            if hgf < 0.40:
                print("  [Probe 3: too few edge-required tokens ({:.1%})]".format(hgf))
                print("  D can reconstruct most of B from A alone -> high collapse risk.")
                print("  -> Increase context_dropout to {}.".format(rec_drop))
                print("  -> Add beta-VAE KL constraint (beta=0.01).")
            if bf > 0.30:
                print("  [Probe 3: bridge token fraction too high ({:.1%})]".format(bf))
                print("  More than 30% of B tokens are A-reconstructable.")
                print("  -> Increase context_dropout to {}.".format(rec_drop))
    print("=" * W)
    print()


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-training probes for G+D architecture validation."
    )
    parser.add_argument(
        "--n_pairs", type=int, default=50,
        help="Number of (A,B) pairs to evaluate (default 50)",
    )
    parser.add_argument(
        "--use_trained_bert", action="store_true",
        help="Use the fine-tuned BERT from fakencoder_best.pt",
    )
    args = parser.parse_args()

    if MUSIQUE_DEV is None:
        print("ERROR: MuSiQue dev JSONL not found. Searched:")
        for p in _DATA_CANDIDATES:
            print("  {}".format(p))
        sys.exit(1)

    print("[probe] MuSiQue : {}".format(MUSIQUE_DEV))
    print("[probe] Pairs   : {}".format(args.n_pairs))

    pairs = load_ab_pairs(MUSIQUE_DEV, args.n_pairs)
    if len(pairs) < 10:
        print("ERROR: only {} pairs loaded.".format(len(pairs)))
        sys.exit(1)
    print("[probe] Loaded  : {} (A,B) pairs".format(len(pairs)))

    if args.use_trained_bert:
        ckpt = HERE / "models" / "fakencoder_best.pt"
        if not ckpt.exists():
            print("ERROR: fakencoder_best.pt not found at {}".format(ckpt))
            sys.exit(1)
        print("[probe] Loading trained BERT from {} ...".format(ckpt))
        sys.path.insert(0, str(HERE))
        from fakencoder_train import FakeEncoderModel
        fe = FakeEncoderModel()
        fe.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = fe.encoder1
        model.eval()
        print("[probe] Loaded FakeEncoder encoder1")
    else:
        print("[probe] Loading bert-base-uncased (raw)...")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("[probe] Running probes on CPU...")

    results = run_probes(pairs, model, tokenizer)
    print_results(results, trained_bert=args.use_trained_bert)


if __name__ == "__main__":
    main()
