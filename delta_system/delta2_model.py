"""
delta2_model.py -- Delta-2 scaffold (STEP 2). See delta_system/DELTA2_DESIGN.md.

REUSES DeltaSystem.generate_delta() UNCHANGED (frozen BERT + two-level cross-attn generator).
Adds, with NO token decoder:
  - novelty-weighted masked pool of the token delta -> fixed delta_vec [768] -> project to 256.
    (The single bottlenecked vector cannot carry all of B -> this IS the anti-collapse mechanism.)
  - embedding ANCHOR: MLP([pool(H_A), delta_vec]) -> predict pool(encode(B)) via cosine.
    Because pool(H_A) already supplies B's shared content, the anchor forces delta_vec to carry
    what A LACKS = novelty.
  - swappable AUX:
      paraphrase: ||delta_vec(A, paraphrase)|| -> 0  &  ||delta_vec(A, real-edit B)|| kept >= 1
      nli       : nli_head(delta_vec) -> entail/neutral/contradict (teacher-distilled)

Trains ONLY the generator (core "g_*") + the new heads; BERT stays frozen; the token decoder is
never called. SPEED: BERT is frozen, so we encode every sentence through it ONCE up front and
cache the hidden states; each training step then runs only the tiny trainable generator (no
repeated BERT forwards). __main__ runs a short SANITY-TRAIN and prints whether it learns.

Run: python delta2_model.py --aux paraphrase --steps 400
     python delta2_model.py --aux nli        --steps 400
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid unauthenticated HF-Hub rate-limit stalls on model downloads: pull a token from the
# Kaggle secret 'HF_TOKEN' if one isn't already set. Safe no-op off Kaggle / if absent.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        _t = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = _t
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _t
        print("HF token: loaded from Kaggle secret")
    except Exception:
        print("HF token: none (downloads use unauthenticated rate limits)")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model       import DeltaSystem, D_MODEL
from delta2_data import (load_edits, load_validated_paraphrases,
                         load_edit_nli_labels, NLI_LABELS)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
PROJ    = 256


def masked_mean(H, m):
    w = m.float().unsqueeze(-1)
    return (H * w).sum(1) / w.sum(1).clamp(min=1e-6)


class Delta2(nn.Module):
    def __init__(self, aux="paraphrase", proj=PROJ):
        super().__init__()
        self.aux  = aux
        self.core = DeltaSystem()                       # frozen BERT + generator (+ unused decoder)
        # NO final LayerNorm: delta_vec magnitude must stay free for the paraphrase objective.
        self.delta_proj = nn.Sequential(nn.Linear(D_MODEL, proj), nn.GELU(),
                                        nn.Linear(proj, proj))
        self.anchor = nn.Sequential(nn.Linear(proj + D_MODEL, D_MODEL), nn.GELU(),
                                    nn.Linear(D_MODEL, D_MODEL))
        if aux == "nli":
            self.nli_head = nn.Sequential(nn.Linear(proj, proj), nn.GELU(),
                                          nn.Linear(proj, 3))

    def trainable_parameters(self):
        ps  = [p for n, p in self.core.named_parameters() if n.startswith("g_")]
        ps += list(self.delta_proj.parameters()) + list(self.anchor.parameters())
        if self.aux == "nli":
            ps += list(self.nli_head.parameters())
        return ps

    @torch.no_grad()
    def encode(self, ids, m):
        return self.core._enc(ids, m)                   # frozen BERT (no grad)

    def delta_from_H(self, H_A, A_m, H_B, B_m):
        """delta_vec from PRECOMPUTED frozen-BERT hidden states (no BERT forward here)."""
        delta, _delta0, alpha = self.core.generate_delta(H_A, A_m, H_B, B_m)
        w = alpha * B_m.float()                          # novelty-weighted pool
        w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
        pooled = (delta * w.unsqueeze(-1)).sum(1)        # [b,768]
        return self.delta_proj(pooled)

    def anchor_loss(self, dv, H_A, A_m, H_B, B_m):
        pred   = self.anchor(torch.cat([dv, masked_mean(H_A, A_m)], dim=-1))
        target = masked_mean(H_B, B_m)
        return (1 - F.cosine_similarity(pred, target, dim=-1)).mean(), pred, target


# ── tokenize + one-time frozen-BERT encoding ─────────────────────────────────────
def enc_ids(tok, texts):
    e = tok(texts, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE)


@torch.no_grad()
def encode_all(model, items, ka, kb, tok, bs=32):
    """Run frozen BERT over every (A,B) pair ONCE; cache hidden states on device."""
    HA, AM, HB, BM = [], [], [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        a_ids, a_m = enc_ids(tok, [x[ka] for x in chunk])
        b_ids, b_m = enc_ids(tok, [x[kb] for x in chunk])
        HA.append(model.encode(a_ids, a_m)); AM.append(a_m)
        HB.append(model.encode(b_ids, b_m)); BM.append(b_m)
    return {"H_A": torch.cat(HA), "A_m": torch.cat(AM),
            "H_B": torch.cat(HB), "B_m": torch.cat(BM)}


def take(C, idx):
    return C["H_A"][idx], C["A_m"][idx], C["H_B"][idx], C["B_m"][idx]


@torch.no_grad()
def evaluate(model, E, P, labels_t, n=64):
    model.eval()
    s = torch.arange(min(n, E["H_A"].size(0)), device=DEVICE)
    H_A, A_m, H_B, B_m = take(E, s)
    dv_e = model.delta_from_H(H_A, A_m, H_B, B_m)
    _, pred, target = model.anchor_loss(dv_e, H_A, A_m, H_B, B_m)
    out = {"anchor_cos": F.cosine_similarity(pred, target, dim=-1).mean().item(),
           "edit_norm":  dv_e.norm(dim=-1).mean().item()}
    if model.aux == "paraphrase" and P is not None:
        ps = torch.arange(min(n, P["H_A"].size(0)), device=DEVICE)
        dv_p = model.delta_from_H(*take(P, ps))
        out["para_norm"]  = dv_p.norm(dim=-1).mean().item()
        out["separation"] = out["edit_norm"] / max(out["para_norm"], 1e-6)
    if model.aux == "nli":
        out["nli_acc"] = (model.nli_head(dv_e).argmax(-1) == labels_t[s]).float().mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", choices=["paraphrase", "nli"], default="paraphrase")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_edit", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    print("=" * 74)
    print(f"DELTA-2 SCAFFOLD SANITY-TRAIN  aux={args.aux}  device={DEVICE}")
    print("=" * 74)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("\nLoading data...")
    edits = load_edits(args.n_edit, tok)
    paras, labels, labels_t = None, None, None
    if args.aux == "paraphrase":
        paras = load_validated_paraphrases(600)         # NLI loaded lazily, only on cache miss
        print(f"  edits {len(edits)} | validated paraphrases {len(paras)}")
    else:
        labels = [NLI_LABELS.index(l) for l in load_edit_nli_labels(args.n_edit, edits)]
        print(f"  edits {len(edits)} | nli labels "
              f"{ {l: labels.count(i) for i,l in enumerate(NLI_LABELS)} }")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("Building model (loading bert-base-uncased)...", flush=True)
    model = Delta2(args.aux).to(DEVICE).train()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    print("Encoding through frozen BERT (one-time)...")
    E = encode_all(model, edits, "A", "B", tok)
    P = encode_all(model, paras, "A", "A_para", tok) if args.aux == "paraphrase" else None
    if args.aux == "nli":
        labels_t = torch.tensor(labels, device=DEVICE)
    N = E["H_A"].size(0)

    print("BEFORE:", {k: round(v, 3) for k, v in evaluate(model, E, P, labels_t).items()})
    print()
    for step in range(1, args.steps + 1):
        idx = torch.as_tensor(rng.integers(0, N, args.bs), device=DEVICE)
        H_A, A_m, H_B, B_m = take(E, idx)
        dv_e = model.delta_from_H(H_A, A_m, H_B, B_m)
        anc, _, _ = model.anchor_loss(dv_e, H_A, A_m, H_B, B_m)

        if args.aux == "paraphrase":
            pidx = torch.as_tensor(rng.integers(0, P["H_A"].size(0), args.bs), device=DEVICE)
            dv_p = model.delta_from_H(*take(P, pidx))
            aux = dv_p.norm(dim=-1).mean() + F.relu(1.0 - dv_e.norm(dim=-1)).mean()
        else:
            aux = F.cross_entropy(model.nli_head(dv_e), labels_t[idx])

        loss = anc + aux
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"  step {step:>4} | loss {loss.item():.3f} | anchor {anc.item():.3f} | aux {aux.item():.3f}")

    after = evaluate(model, E, P, labels_t)
    print("\nAFTER :", {k: round(v, 3) for k, v in after.items()})

    print("\n" + "=" * 74)
    print("SANITY VERDICT (did the scaffold LEARN?)")
    ok_anchor = after["anchor_cos"] > 0.5
    print(f"  anchor reconstructs encode(B): {'OK ' if ok_anchor else 'WEAK'} (cos={after['anchor_cos']:.3f})")
    if args.aux == "paraphrase":
        ok_aux = after.get("separation", 0) > 1.5
        print(f"  surface-invariance separation: {'OK ' if ok_aux else 'WEAK'} "
              f"(edit/para norm = {after.get('separation', 0):.2f}; want >1.5)")
    else:
        ok_aux = after.get("nli_acc", 0) > 0.45
        print(f"  delta carries NLI relation   : {'OK ' if ok_aux else 'WEAK'} "
              f"(train acc={after.get('nli_acc', 0):.3f}; chance 0.33)")
    print("  GO -> scaffold trains; proceed to step 3 (battery)." if (ok_anchor and ok_aux)
          else "  Scaffold trains but signal weak at this budget; inspect before step 3.")
    print("=" * 74)


if __name__ == "__main__":
    main()
