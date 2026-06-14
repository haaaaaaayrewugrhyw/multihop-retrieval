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
                  (surface-invariance: meaning-preserving rewrite => no delta; real change => delta)
      nli       : nli_head(delta_vec) -> entail/neutral/contradict (teacher-distilled)

Trains ONLY the generator (core "g_*") + the new heads; BERT stays frozen; the token decoder is
never called. __main__ runs a short SANITY-TRAIN and prints whether it actually learns.

Run: python delta2_model.py --aux paraphrase --steps 400
     python delta2_model.py --aux nli        --steps 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model       import DeltaSystem, D_MODEL
from delta2_data import (load_edits, load_validated_paraphrases,
                         nli_label_edits, NLI, NLI_LABELS)

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

    def delta_vec(self, A_ids, A_m, B_ids, B_m):
        H_A = self.core._enc(A_ids, A_m)
        H_B = self.core._enc(B_ids, B_m)
        delta, _delta0, alpha = self.core.generate_delta(H_A, A_m, H_B, B_m)
        # novelty-weighted pool: emphasize tokens where B diverges from A
        w = alpha * B_m.float()
        w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
        pooled = (delta * w.unsqueeze(-1)).sum(1)       # [b,768]
        return self.delta_proj(pooled), H_A, H_B

    def anchor_loss(self, dv, H_A, A_m, H_B, B_m):
        pred   = self.anchor(torch.cat([dv, masked_mean(H_A, A_m)], dim=-1))
        target = masked_mean(H_B, B_m)
        return (1 - F.cosine_similarity(pred, target, dim=-1)).mean(), pred, target


# ── data tensors ────────────────────────────────────────────────────────────────
def enc(tok, texts):
    e = tok(texts, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE)


def batch(items, key_a, key_b, idx, tok):
    A = [items[i][key_a] for i in idx]
    B = [items[i][key_b] for i in idx]
    A_ids, A_m = enc(tok, A); B_ids, B_m = enc(tok, B)
    return A_ids, A_m, B_ids, B_m


@torch.no_grad()
def evaluate(model, edits, paras, labels, tok, n=64):
    model.eval()
    idx = list(range(min(n, len(edits))))
    A_ids, A_m, B_ids, B_m = batch(edits, "A", "B", idx, tok)
    dv_e, H_A, H_B = model.delta_vec(A_ids, A_m, B_ids, B_m)
    _, pred, target = model.anchor_loss(dv_e, H_A, A_m, H_B, B_m)
    anc_cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
    out = {"anchor_cos": anc_cos, "edit_norm": dv_e.norm(dim=-1).mean().item()}
    if model.aux == "paraphrase" and paras:
        pidx = list(range(min(n, len(paras))))
        pa, pam, pb, pbm = batch(paras, "A", "A_para", pidx, tok)
        dv_p, _, _ = model.delta_vec(pa, pam, pb, pbm)
        out["para_norm"] = dv_p.norm(dim=-1).mean().item()
        out["separation"] = out["edit_norm"] / max(out["para_norm"], 1e-6)
    if model.aux == "nli":
        logits = model.nli_head(dv_e)
        out["nli_acc"] = (logits.argmax(-1) == torch.tensor(labels[:len(idx)], device=DEVICE)).float().mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", choices=["paraphrase", "nli"], default="paraphrase")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_edit", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    print("=" * 74)
    print(f"DELTA-2 SCAFFOLD SANITY-TRAIN  aux={args.aux}  device={DEVICE}")
    print("=" * 74)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("\nLoading data...")
    edits = load_edits(args.n_edit, tok)
    paras, labels = None, None
    nli = NLI()
    if args.aux == "paraphrase":
        paras = load_validated_paraphrases(1200, nli)
        print(f"  edits {len(edits)} | validated paraphrases {len(paras)}")
    else:
        nli_label_edits(edits, nli)
        labels = [NLI_LABELS.index(e["nli"]) for e in edits]
        print(f"  edits {len(edits)} | nli labels "
              f"{ {l: labels.count(i) for i,l in enumerate(NLI_LABELS)} }")
    del nli
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = Delta2(args.aux).to(DEVICE).train()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    print("\nBEFORE:", {k: round(v, 3) for k, v in evaluate(model, edits, paras, labels, tok).items()})
    print()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(edits), args.bs)
        A_ids, A_m, B_ids, B_m = batch(edits, "A", "B", idx, tok)
        dv_e, H_A, H_B = model.delta_vec(A_ids, A_m, B_ids, B_m)
        anc, _, _ = model.anchor_loss(dv_e, H_A, A_m, H_B, B_m)

        if args.aux == "paraphrase":
            pidx = rng.integers(0, len(paras), args.bs)
            pa, pam, pb, pbm = batch(paras, "A", "A_para", pidx, tok)
            dv_p, _, _ = model.delta_vec(pa, pam, pb, pbm)
            aux = dv_p.norm(dim=-1).mean() + F.relu(1.0 - dv_e.norm(dim=-1)).mean()
        else:
            lab = torch.tensor([labels[i] for i in idx], device=DEVICE)
            aux = F.cross_entropy(model.nli_head(dv_e), lab)

        loss = anc + aux
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"  step {step:>4} | loss {loss.item():.3f} | anchor {anc.item():.3f} | aux {aux.item():.3f}")

    after = evaluate(model, edits, paras, labels, tok)
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
