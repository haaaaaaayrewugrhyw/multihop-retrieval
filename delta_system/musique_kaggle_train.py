"""
musique_kaggle_train.py -- MuSiQue second-dataset validation on Kaggle.

Run on Kaggle with two cells only:

    # Cell 1
    !pip install transformers scikit-learn -q
    !git clone https://github.com/haaaaaaayrewugrhyw/multihop-retrieval.git /kaggle/working/repo

    # Cell 2
    !python /kaggle/working/repo/delta_system/musique_kaggle_train.py

Downloads MuSiQue automatically. Trains G on 5000 pairs, evaluates on 500
held-out, prints cross-dataset comparison against Wikipedia results.
"""

import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model  import DeltaSystem
from losses import recon_loss, sparsity_loss, specificity_loss
from eval   import evaluate

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN   = 128
N_TRAIN   = 5000
N_EVAL    = 500
STEPS     = 2000
BS        = 16
LR        = 1e-4
LAM_S     = 1.0
LAM_SPEC  = 1.0
MARGIN    = 2.0
LOG_EVERY = 200

# ── Download MuSiQue ──────────────────────────────────────────────────────────

def download_musique():
    data_dir  = Path("/kaggle/working/musique")
    data_dir.mkdir(exist_ok=True)
    jsonl_path = data_dir / "musique_ans_v1.0_train.jsonl"

    if jsonl_path.exists():
        print(f"Data already present: {jsonl_path}")
        return jsonl_path

    print("Downloading MuSiQue from GitHub...")
    zip_path = Path("/kaggle/working/musique.zip")

    urls = [
        "https://github.com/StonyBrookNLP/musique/releases/download/v1.0/musique_v1.0.zip",
        "https://github.com/StonyBrookNLP/musique/releases/download/v0.0.1/musique_v1.0.zip",
    ]
    for url in urls:
        ret = os.system(f'wget -q "{url}" -O "{zip_path}"')
        if ret == 0 and zip_path.exists() and zip_path.stat().st_size > 1e6:
            print(f"  Downloaded from {url}")
            break
    else:
        raise RuntimeError(
            "MuSiQue download failed from all URLs.\n"
            "Manual fix: upload musique_ans_v1.0_train.jsonl to /kaggle/working/musique/"
        )

    extract_dir = Path("/kaggle/working/musique_raw")
    os.system(f'unzip -q "{zip_path}" -d "{extract_dir}"')

    found = list(extract_dir.rglob("musique_ans_v1.0_train.jsonl"))
    if not found:
        all_jsonl = list(extract_dir.rglob("*.jsonl"))
        raise RuntimeError(
            f"musique_ans_v1.0_train.jsonl not found in zip.\n"
            f"Files found: {[str(f) for f in all_jsonl]}"
        )

    os.system(f'cp "{found[0]}" "{jsonl_path}"')
    print(f"  Saved to {jsonl_path} ({jsonl_path.stat().st_size/1e6:.1f} MB)")
    return jsonl_path


# ── Load pairs ────────────────────────────────────────────────────────────────

def load_musique_pairs(jsonl_path, n_train=5000, n_eval=500):
    total = n_train + n_eval
    pairs = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            ex     = json.loads(line)
            decomp = ex.get("question_decomposition", [])
            if len(decomp) != 2:
                continue
            paras = {p["idx"]: p["paragraph_text"].strip() for p in ex["paragraphs"]}
            idx1  = decomp[0].get("paragraph_support_idx")
            idx2  = decomp[1].get("paragraph_support_idx")
            if idx1 is None or idx2 is None:
                continue
            A     = paras.get(idx1, "").strip()
            novel = paras.get(idx2, "").strip()
            if not A or not novel:
                continue
            pairs.append({"A": A, "B": A + " " + novel, "novel": novel})
            if len(pairs) >= total:
                break

    print(f"Loaded {len(pairs)} 2-hop pairs  ({n_train} train / {n_eval} eval)")
    return pairs[:n_train], pairs[n_train:total]


# ── Dataset / collate ─────────────────────────────────────────────────────────

class PairDS(Dataset):
    def __init__(self, p): self.p = p
    def __len__(self):     return len(self.p)
    def __getitem__(self, i): return self.p[i]["A"], self.p[i]["B"]


def make_collate(tok):
    def collate(batch):
        eA = tok([x[0] for x in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        eB = tok([x[1] for x in batch], max_length=MAX_LEN, truncation=True,
                 padding="max_length", return_tensors="pt")
        return eA["input_ids"], eA["attention_mask"], eB["input_ids"], eB["attention_mask"]
    return collate


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_pairs, tok):
    dl  = DataLoader(PairDS(train_pairs), batch_size=BS, shuffle=True,
                     collate_fn=make_collate(tok), num_workers=2, pin_memory=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    step = 0

    while step < STEPS:
        for batch in dl:
            if step >= STEPS: break
            A_ids, A_mask, B_ids, B_mask = [t.to(DEVICE) for t in batch]
            b = A_ids.size(0)

            logits, delta, delta_0, H_A, _ = model(A_ids, A_mask, B_ids, B_mask)
            L_r    = recon_loss(logits, B_ids, B_mask)
            L_s    = sparsity_loss(delta, B_mask)
            L_spec = torch.tensor(0.0, device=DEVICE)
            if b > 1:
                idx   = list(range(1, b)) + [0]
                lw    = model.reconstruct(H_A, A_mask, delta[idx], delta_0[idx], B_ids, B_mask)
                L_spec = specificity_loss(logits, lw, B_ids, B_mask, margin=MARGIN)

            loss = L_r + LAM_S * L_s + LAM_SPEC * L_spec
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1

            if step % LOG_EVERY == 0 or step == 1:
                ppl = math.exp(min(L_r.item(), 20))
                print(f"  step {step:4d}/{STEPS} | ppl={ppl:.1f} | L_spec={L_spec.item():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("DELTA SYSTEM — MuSiQue Validation (Second Dataset)")
    print("=" * 62)
    print(f"Device : {DEVICE}")
    print(f"Config : {N_TRAIN} train / {N_EVAL} held-out | {STEPS} steps | bs={BS}")
    print()

    # Data
    jsonl_path           = download_musique()
    train_pairs, eval_pairs = load_musique_pairs(jsonl_path, N_TRAIN, N_EVAL)
    print()

    # Model
    tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = DeltaSystem().to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params : {n_p/1e6:.1f}M  (same architecture as Wikipedia run)")
    print()

    # Train
    print("--- Training G ---")
    train(model, train_pairs, tok)
    print()

    # Save
    ckpt_dir = Path("/kaggle/working/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("bert.")}
    torch.save(trainable, ckpt_dir / "musique_model.pt")
    print(f"Checkpoint saved: {ckpt_dir / 'musique_model.pt'}")
    print()

    # Evaluate held-out
    print("=" * 62)
    print(f"HELD-OUT EVAL — {len(eval_pairs)} MuSiQue pairs (never seen during training)")
    print("=" * 62)
    held_out = evaluate(model, eval_pairs, tok)
    print()

    # In-sample
    print("=" * 62)
    print("IN-SAMPLE CHECK — 200 training pairs")
    print("=" * 62)
    train_res = evaluate(model, train_pairs[:200], tok)
    print()

    # Cross-dataset summary
    print("=" * 62)
    print("CROSS-DATASET COMPARISON")
    print("=" * 62)
    print(f"{'Dataset':<28} {'DELTA_PPL':>10} {'SPECIFICITY':>12} {'Pass?':>6}")
    print("-" * 62)
    print(f"{'Wikipedia (8000tr/1000ev)':<28} {'  +755':>10} {'   +608':>12} {'PASS':>6}")
    print(f"{'MuSiQue  (5000tr/500ev)':<28} {held_out['delta_ppl']:>+10.0f} "
          f"{held_out['specificity']:>+12.0f} {'PASS' if held_out['pass'] else 'FAIL':>6}")
    print()
    if held_out["pass"]:
        print("CONCLUSION: Both datasets PASS.")
        print("The architecture generalizes across dataset types:")
        print("  - Wikipedia: free-form encyclopedic paragraphs")
        print("  - MuSiQue  : structured 2-hop reasoning across articles")
        print("No architecture changes. Same hyperparameters. Both generalize.")
    else:
        print("CONCLUSION: MuSiQue FAIL — investigate above.")
    print("=" * 62)


if __name__ == "__main__":
    main()
