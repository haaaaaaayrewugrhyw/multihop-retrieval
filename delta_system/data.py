"""
data.py -- MuSiQue 2-hop pair loader for delta-system validation.

A     = hop-1 supporting paragraph   ("what is already known")
B     = hop-1 + hop-2 concatenated   ("the new document")
novel = hop-2 paragraph              (ground-truth novelty for AUROC eval)

Requires:
    retrieval/data/musique/musique_ans_v1.0_train.jsonl
    Download: https://github.com/StonyBrookNLP/musique/releases/tag/v1.0
"""

import json
from pathlib import Path
from typing import Dict, List

_ROOT          = Path(__file__).resolve().parent.parent
_MUSIQUE_TRAIN = _ROOT / "retrieval" / "data" / "musique" / "musique_ans_v1.0_train.jsonl"


def load_pairs(max_examples: int = 100, path: Path = _MUSIQUE_TRAIN) -> List[Dict]:
    """
    Return list of {"A": str, "B": str, "novel": str}
    where B = A + " " + novel.  Only 2-hop questions.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"MuSiQue data not found at {path}\n"
            "Download from: https://github.com/StonyBrookNLP/musique/releases/tag/v1.0\n"
            "Extract musique_ans_v1.0_train.jsonl to retrieval/data/musique/"
        )

    pairs: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ex    = json.loads(line)
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
            if max_examples and len(pairs) >= max_examples:
                break

    print(f"[data] {len(pairs)} 2-hop pairs from {path.name}")
    return pairs


if __name__ == "__main__":
    pairs = load_pairs(max_examples=5)
    for p in pairs:
        print(f"  A    : {p['A'][:80]}")
        print(f"  novel: {p['novel'][:80]}")
        print()
