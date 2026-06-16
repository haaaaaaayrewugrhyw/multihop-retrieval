"""
delta2b_data.py -- STEP 1 of the build spec: data + SELF-SUPERVISED supervision for the
token-level multi-objective model. See DELTA2_BUILD_SPEC.md.

Provides, same-domain and group-split (no leakage):
  - real EDITS (IteraTeR meaning-changed): {A, B, novel_mask, real_len, type}  (difflib novelty target)
  - NON-MEANING edits (IteraTeR fluency/clarity): {A, A_para}  (same-domain meaning-preserving)
  - CLINE-style GENERATED perturbations of OUR OWN sentences (WordNet):
        x_syn (synonyms, meaning-PRESERVING) and x_ant (antonyms/random, meaning-CHANGING)
    -> clean, same-domain, self-supervised positives/negatives for the invariance/gate objective,
       fixing the MRPC-domain-confound + difflib-noise problems.

Run as a script -> validation stats (generation success, replacement rates, and an NLI sanity
check that syn≈entailment >> ant) so we trust the generated labels before training.

Run: python delta2b_data.py --n 400
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from delta2_data          import (load_iterater_by_meaning, change_type, group_split,
                                  _cache_dir, NLI)
from insertion_cloze_eval import _novel_mask_difflib

import nltk
for _pkg in ("wordnet", "omw-1.4", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass
from nltk.corpus import wordnet as wn

SYN_FRAC = 0.40       # CLINE: ~40% content words -> synonyms (meaning preserved)
ANT_FRAC = 0.20       # CLINE: ~20% content words -> antonyms/random (meaning changed)


def _wn_pos(tag):
    if tag.startswith("N"): return wn.NOUN
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("R"): return wn.ADV
    return None


def _synonym(word, pos):
    out = set()
    for s in wn.synsets(word, pos=pos):
        for l in s.lemmas():
            n = l.name().replace("_", " ")
            if n.lower() != word.lower() and n.isalpha():
                out.add(n)
    return sorted(out)


def _antonym(word, pos):
    out = set()
    for s in wn.synsets(word, pos=pos):
        for l in s.lemmas():
            for a in l.antonyms():
                n = a.name().replace("_", " ")
                if n.isalpha():
                    out.add(n)
    return sorted(out)


def perturb(sent, mode, frac, rng, vocab):
    """Replace `frac` of content words with synonyms (mode='syn') or antonyms/random (mode='ant')."""
    words = sent.split()
    try:
        tags = nltk.pos_tag(words)
    except Exception:
        return sent, 0
    cand = [i for i, (w, t) in enumerate(tags)
            if _wn_pos(t) and w.isalpha() and len(w) > 2]
    rng.shuffle(cand)
    k = max(1, int(frac * max(len(cand), 1)))
    out, done = words[:], 0
    for i in cand:
        if done >= k:
            break
        w, t = tags[i]; pos = _wn_pos(t)
        if mode == "syn":
            opts = _synonym(w.lower(), pos)
            repl = rng.choice(opts) if opts else None
        else:
            opts = _antonym(w.lower(), pos)
            repl = rng.choice(opts) if opts else (rng.choice(vocab) if len(vocab) else None)
        if repl and repl.lower() != w.lower():
            out[i] = repl; done += 1
    return " ".join(out), done


def build(n, seed=0, cache=True):
    """Assemble edits + non-meaning + generated syn/ant. Cached."""
    fp = _cache_dir() / f"delta2b_{n}.pkl"
    if cache and fp.exists():
        print(f"  [cache] delta2b <- {fp}")
        return pickle.load(open(fp, "rb"))

    rng = np.random.default_rng(seed)
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    raw_edit = load_iterater_by_meaning(n, meaning=True)
    raw_non  = load_iterater_by_meaning(n, meaning=False)

    # vocabulary of content words (for random antonym fallback)
    vocab = []
    for p in raw_edit + raw_non:
        vocab += [w for w in p["before"].split() if w.isalpha() and len(w) > 3]
    vocab = list(set(vocab))

    edits = []
    for k, p in enumerate(raw_edit):
        mask, rl = _novel_mask_difflib(p["before"], p["after"], tok)
        if mask is None:
            continue
        edits.append({"A": p["before"], "B": p["after"], "novel_mask": mask, "real_len": rl,
                      "type": change_type(p["before"], p["after"]), "group": p["group"]})

    non = [{"A": p["before"], "A_para": p["after"], "group": p["group"]} for p in raw_non]

    # generate syn (preserve) and ant (change) for each edit's A sentence
    syn, ant = [], []
    for e in edits:
        s, ds = perturb(e["A"], "syn", SYN_FRAC, rng, vocab)
        a, da = perturb(e["A"], "ant", ANT_FRAC, rng, vocab)
        if ds > 0:
            syn.append({"A": e["A"], "A_syn": s, "n_repl": ds, "group": e["group"]})
        if da > 0:
            ant.append({"A": e["A"], "A_ant": a, "n_repl": da, "group": e["group"]})

    data = {"edits": edits, "non": non, "syn": syn, "ant": ant}
    if cache:
        pickle.dump(data, open(fp, "wb"))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--nli_sample", type=int, default=120)
    args = ap.parse_args()

    print("=" * 78)
    print("DELTA-2B DATA/SUPERVISION CHECK (step 1)")
    print("=" * 78)
    d = build(args.n)
    edits, non, syn, ant = d["edits"], d["non"], d["syn"], d["ant"]

    print(f"\nedits (meaning-changed)      : {len(edits)}")
    types = Counter(e["type"] for e in edits)
    for t, c in types.most_common():
        print(f"    {t:<11} {c:>4} ({c/max(len(edits),1):.0%})")
    print(f"non-meaning (same-domain neg): {len(non)}")
    print(f"generated syn (preserve)     : {len(syn)}  | mean repl {np.mean([s['n_repl'] for s in syn]):.1f}")
    print(f"generated ant (change)       : {len(ant)}  | mean repl {np.mean([a['n_repl'] for a in ant]):.1f}")

    # group-split sanity
    tr, te = group_split(edits, test_frac=0.25)
    print(f"\ngroup split edits: train {len(tr)} / test {len(te)}")

    # NLI sanity: syn should ENTAIL (meaning preserved) >> ant
    print("\nNLI sanity (entail prob of A->variant; want syn >> ant):")
    nli = NLI()
    m = min(args.nli_sample, len(syn), len(ant))
    e_syn = nli.probs([s["A"] for s in syn[:m]], [s["A_syn"] for s in syn[:m]])[:, 2]
    e_ant = nli.probs([a["A"] for a in ant[:m]], [a["A_ant"] for a in ant[:m]])[:, 2]
    print(f"  entail(A, syn) mean {e_syn.mean():.3f}   entail(A, ant) mean {e_ant.mean():.3f}   "
          f"gap {e_syn.mean()-e_ant.mean():+.3f}")

    print("\n" + "=" * 78)
    ok = (len(edits) > 100 and len(syn) > 100 and len(ant) > 100 and
          e_syn.mean() - e_ant.mean() > 0.1)
    print("GO -> supervision is clean (syn preserves, ant changes); proceed to step 2 (architecture)."
          if ok else "FIX flagged item before step 2.")
    print(f"  examples:\n    A   : {edits[0]['A'][:90]}")
    print(f"    syn : {syn[0]['A_syn'][:90]}")
    print(f"    ant : {ant[0]['A_ant'][:90]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
