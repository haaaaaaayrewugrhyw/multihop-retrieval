"""
delta2_data.py -- Data foundation for the Delta-2 multi-task experiment (STEP 1).

See delta_system/DELTA2_DESIGN.md for the full design. This module assembles + VALIDATES the
data everything else rests on, with leakage control:

  - load_edits()        : real edit pairs (A,B) with gold change span + change-type
                          (insertion / entity / relational / numeric), source + group id.
  - load_paraphrases()  : paraphrase candidates (A, A_para) from MRPC.
  - validate_paraphrases: keep only BIDIRECTIONALLY-entailing (meaning preserved) AND
                          lexically-different (real rewrite) pairs -- the researched standard.
  - nli_label_edits()   : entail / neutral / contradict label per (A,B) from a frozen NLI
                          teacher (roberta-large-mnli) -- free supervision for task B + meaning probe.
  - group_split()       : split by source/group id so paraphrase/mirror pairs never leak.

Run as a script -> reports the go/no-go FOUNDATION stats:
  * change-type stratification (enough entity/relational edits?)
  * paraphrase validity pass-rate (negatives meaning-preserving + different?)
  * NLI label distribution over edits (real edits look like contradiction/neutral, not entail?)

Run: python delta2_data.py --n_edit 600 --n_para 600
"""

import argparse
import pickle
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from insertion_cloze_eval import load_pairs as load_iterater
from vitaminc_probe_eval  import load_vitaminc_pairs

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NLI_NAME   = "roberta-large-mnli"
NLI_LABELS = ["contradiction", "neutral", "entailment"]   # roberta-large-mnli label order


# ── change-type ────────────────────────────────────────────────────────────────
def change_type(A: str, B: str) -> str:
    aw, bw = A.split(), B.split()
    aset = set(aw)
    changed = [w for w in bw if w not in aset]
    if not changed:
        return "none"
    text = " ".join(changed)
    if re.search(r"\d", text):
        return "numeric"
    if any(w[:1].isupper() for w in changed):
        return "entity"
    if len(bw) > len(aw) + 2:          # B notably longer -> a span was inserted
        return "insertion"
    return "relational"                # same-length-ish reword / role / relation change


# ── NLI teacher ────────────────────────────────────────────────────────────────
class NLI:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(NLI_NAME)
        self.m   = AutoModelForSequenceClassification.from_pretrained(NLI_NAME).to(DEVICE).eval()

    @torch.no_grad()
    def probs(self, prem, hyp, bs=16):
        out = []
        for i in range(0, len(prem), bs):
            enc = self.tok(prem[i:i + bs], hyp[i:i + bs], truncation=True, max_length=256,
                           padding=True, return_tensors="pt").to(DEVICE)
            out.append(F.softmax(self.m(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)            # [n,3] cols: contra, neutral, entail


def lex_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


# ── disk cache (VitaminC's 20k streaming scan is the per-run bottleneck) ─────────
def _cache_dir():
    base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(tempfile.gettempdir())
    d = base / "delta2_cache"; d.mkdir(parents=True, exist_ok=True)
    return d


# ── loaders ────────────────────────────────────────────────────────────────────
def load_edits(n, tok, cache=True):
    """IteraTeR (meaning-changed) + VitaminC contrast pairs -> edits with type + group id.
    Cached to disk: persists across cells/runs in a session so we stream VitaminC only ONCE."""
    fp = _cache_dir() / f"edits_{n}.pkl"
    if cache and fp.exists():
        print(f"  [cache] edits <- {fp}")
        return pickle.load(open(fp, "rb"))
    edits = []
    for k, p in enumerate(load_iterater(n // 2, tok)):
        edits.append({"A": p["A"], "B": p["B"], "type": change_type(p["A"], p["B"]),
                      "source": "iterater", "group": f"it{k}"})
    for p in load_vitaminc_pairs(n - len(edits)):
        edits.append({"A": p["A"], "B": p["B"], "type": change_type(p["A"], p["B"]),
                      "source": "vitaminc", "group": f"vc::{p['claim'][:60]}"})   # group by claim
    if cache:
        pickle.dump(edits, open(fp, "wb"))
    return edits


def load_validated_paraphrases(n_cand, nli, ent_thr=0.6, max_overlap=0.85, cache=True):
    """Validated paraphrase pairs, cached (roberta-large over n_cand x 2 directions is slow)."""
    fp = _cache_dir() / f"paras_{n_cand}.pkl"
    if cache and fp.exists():
        print(f"  [cache] paraphrases <- {fp}")
        return pickle.load(open(fp, "rb"))
    valid, _ = validate_paraphrases(load_paraphrases(n_cand), nli, ent_thr, max_overlap)
    if cache:
        pickle.dump(valid, open(fp, "wb"))
    return valid


def load_paraphrases(n):
    """MRPC paraphrase pairs (label=1) across ALL splits as raw candidates (validated
    separately). MRPC's notion of 'equivalent' is loose (asymmetric add/drop clauses), so the
    strict bidirectional-NLI gate keeps only ~18% -- we pull the whole positive pool (~3.9k) so
    enough survive WITHOUT weakening the gate."""
    pos, seen = [], set()
    for split in ("train", "validation", "test"):
        for ex in load_dataset("glue", "mrpc", split=split):
            if ex["label"] == 1:
                key = (ex["sentence1"], ex["sentence2"])
                if key not in seen:
                    seen.add(key)
                    pos.append(key)
    return pos[:n]


def validate_paraphrases(pairs, nli, ent_thr=0.6, max_overlap=0.85):
    """Keep pairs that are BIDIRECTIONALLY entailing (meaning preserved) AND lexically
    different (real rewrite). Returns (valid_list, stats_dict)."""
    if not pairs:
        return [], {"n": 0, "bidir_entail": 0, "lex_ok": 0, "valid": 0}
    ab = nli.probs([a for a, _ in pairs], [b for _, b in pairs])[:, 2]
    ba = nli.probs([b for _, b in pairs], [a for a, _ in pairs])[:, 2]
    valid, both, lex_ok = [], 0, 0
    for (a, b), e1, e2 in zip(pairs, ab, ba):
        bidir = (e1 > ent_thr) and (e2 > ent_thr)
        ov = lex_overlap(a, b)
        both += bidir
        lex_ok += (ov <= max_overlap)
        if bidir and ov <= max_overlap:
            valid.append({"A": a, "A_para": b})
    n = len(pairs)
    return valid, {"n": n, "bidir_entail": both / n, "lex_ok": lex_ok / n, "valid": len(valid) / n}


def nli_label_edits(edits, nli):
    ab = nli.probs([e["A"] for e in edits], [e["B"] for e in edits])
    for e, p in zip(edits, ab):
        e["nli"] = NLI_LABELS[int(p.argmax())]
        e["nli_probs"] = p.tolist()
    return edits


def group_split(items, test_frac=0.2, seed=42):
    groups = list(dict.fromkeys(it["group"] for it in items))
    rng = np.random.default_rng(seed); rng.shuffle(groups)
    test = set(groups[:max(1, int(len(groups) * test_frac))])
    tr = [it for it in items if it["group"] not in test]
    te = [it for it in items if it["group"] in test]
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_edit", type=int, default=600)
    ap.add_argument("--n_para", type=int, default=600)
    args = ap.parse_args()

    print("=" * 74)
    print("DELTA-2 DATA FOUNDATION CHECK (go/no-go for the multi-task experiment)")
    print("=" * 74)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    nli = NLI()

    print("\n[1] Edits (IteraTeR + VitaminC) + change-type stratification...")
    edits = load_edits(args.n_edit, tok)
    types = Counter(e["type"] for e in edits)
    print(f"  edits: {len(edits)}")
    for t, c in types.most_common():
        print(f"    {t:<11} {c:>5}  ({c/len(edits):.0%})")
    recoverable = sum(types[t] for t in ("entity", "relational", "insertion"))
    print(f"  encoder-recoverable (entity+relational+insertion): {recoverable/len(edits):.0%} "
          f"| numeric (hard bound): {types['numeric']/max(len(edits),1):.0%}")

    print("\n[2] NLI-teacher labels on edits (should skew contradiction/neutral, not entail)...")
    edits = nli_label_edits(edits, nli)
    nli_dist = Counter(e["nli"] for e in edits)
    for l in NLI_LABELS:
        print(f"    {l:<13} {nli_dist[l]:>5}  ({nli_dist[l]/len(edits):.0%})")

    print("\n[3] Paraphrase negatives (MRPC) -> bidirectional-NLI + lexical validation...")
    paras = load_paraphrases(args.n_para)
    valid, st = validate_paraphrases(paras, nli)
    print(f"    candidates {st['n']} | bidir-entail {st['bidir_entail']:.0%} | "
          f"lexically-different {st['lex_ok']:.0%} | VALID {st['valid']:.0%} ({len(valid)} usable)")

    print("\n[4] Group split sanity...")
    tr, te = group_split(edits)
    print(f"    edits train {len(tr)} | test {len(te)} (split by source/claim group)")

    print("\n" + "=" * 74)
    print("FOUNDATION VERDICT")
    ok_strat = recoverable / max(len(edits), 1) > 0.4
    # COUNT is what matters, not pass-rate: MRPC is a loose paraphrase source and our gate is
    # strict, so a low rate is expected and irrelevant -- we only need enough CLEAN pairs.
    ok_para  = len(valid) >= 300
    ok_nli   = (nli_dist["contradiction"] + nli_dist["neutral"]) / max(len(edits), 1) > 0.5
    print(f"  enough recoverable edits  : {'OK ' if ok_strat else 'LOW'} ({recoverable/max(len(edits),1):.0%})")
    print(f"  paraphrase negatives valid: {'OK ' if ok_para else 'LOW'} (n={len(valid)} clean pairs; "
          f"rate {st['valid']:.0%} just reflects MRPC looseness, not quality)")
    print(f"  edits are real changes    : {'OK ' if ok_nli else 'LOW'} "
          f"(contra+neutral {(nli_dist['contradiction']+nli_dist['neutral'])/max(len(edits),1):.0%})")
    print("  GO -> proceed to scaffold (step 2)." if (ok_strat and ok_para and ok_nli)
          else "  FIX the flagged item before building the scaffold.")
    print("=" * 74)


if __name__ == "__main__":
    main()
