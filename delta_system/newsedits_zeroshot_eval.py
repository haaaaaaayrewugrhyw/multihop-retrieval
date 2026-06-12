"""
newsedits_zeroshot_eval.py -- Zero-shot domain-transfer evaluation on NewsEdits.

Uses the same DELTA_PPL + SPECIFICITY metrics as Wikipedia and HotpotQA.
G is trained on Wikipedia only. NewsEdits AP revision pairs are the test set.

    A     = preserved sentences from old article version
    novel = sentences genuinely added in new version
    B     = A + novel

If DELTA_PPL > 2: delta helps reconstruct news content better than A alone.
This means the architecture generalizes across domains (Wikipedia -> news).

Cross-domain comparison:
    Wikipedia  DELTA_PPL +755  SPEC +608   (same domain)
    HotpotQA   DELTA_PPL +480  SPEC +2547  (cross-dataset)
    NewsEdits  DELTA_PPL  ???  SPEC  ???   (cross-domain zero-shot)
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import torch
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import DeltaSystem
from eval  import evaluate

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data loading from NewsEdits SQLite ───────────────────────────────────────

def _get_version_pairs(conn):
    cur = conn.execute("""
        SELECT DISTINCT entry_id, version_x, version_y
        FROM matched_sentences
        ORDER BY entry_id, version_x, version_y
    """)
    return cur.fetchall()


def _get_preserved_sentences(conn, entry_id, version_x, version_y,
                              max_dist=0.15):
    cur = conn.execute("""
        SELECT ss.sent_idx, ss.sentence
        FROM split_sentences ss
        JOIN matched_sentences ms
            ON ss.entry_id  = ms.entry_id
            AND ss.version  = ms.version_x
            AND ss.sent_idx = ms.sent_idx_x
        WHERE ms.entry_id  = ?
          AND ms.version_x = ?
          AND ms.version_y = ?
          AND ms.avg_sentence_distance_x <= ?
        ORDER BY ss.sent_idx
    """, (entry_id, version_x, version_y, max_dist))
    return cur.fetchall()


def _get_added_sentences(conn, entry_id, version_x, version_y,
                          min_sent_len=25):
    cur = conn.execute("""
        SELECT ss.sent_idx, ss.sentence
        FROM split_sentences ss
        LEFT JOIN matched_sentences ms
            ON ss.entry_id  = ms.entry_id
            AND ss.version  = ms.version_y
            AND ss.sent_idx = ms.sent_idx_y
            AND ms.version_x = ?
        WHERE ss.entry_id = ?
          AND ss.version  = ?
          AND ms.sent_idx_y IS NULL
          AND LENGTH(ss.sentence) >= ?
        ORDER BY ss.sent_idx
    """, (version_x, entry_id, version_y, min_sent_len))
    return cur.fetchall()


def load_newsedits_pairs(db_path, n_pairs=500, min_added=2,
                          max_len_chars=2000):
    print(f"Opening: {db_path}")
    conn = sqlite3.connect(db_path)

    print("Creating indexes (one-time, ~30 sec)...")
    t0 = time.time()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ms_triple "
                 "ON matched_sentences(entry_id, version_x, version_y)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ms_sent_y "
                 "ON matched_sentences(entry_id, version_y, sent_idx_y)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ss_entry_ver "
                 "ON split_sentences(entry_id, version)")
    conn.commit()
    print(f"  Indexes ready in {time.time()-t0:.0f}s")

    print("Loading revision pair index...")
    all_pairs = _get_version_pairs(conn)
    print(f"  Found {len(all_pairs):,} revision pairs total")

    pairs, checked = [], 0
    t_start = time.time()

    for entry_id, version_x, version_y in all_pairs:
        if len(pairs) >= n_pairs:
            break
        checked += 1

        if checked % 500 == 0:
            elapsed = time.time() - t_start
            rate    = checked / elapsed if elapsed > 0 else 1
            print(f"  checked {checked:,} | collected {len(pairs)}/{n_pairs} "
                  f"| {rate:.0f} rev/s")

        preserved = _get_preserved_sentences(conn, entry_id, version_x, version_y)
        added     = _get_added_sentences(conn, entry_id, version_x, version_y)

        if len(added) < min_added or len(preserved) < 2:
            continue

        A_text     = ' '.join(r[1].strip() for r in preserved)[:max_len_chars]
        novel_text = ' '.join(r[1].strip() for r in added)[:max_len_chars]

        if len(A_text) < 80 or len(novel_text) < 40:
            continue

        pairs.append({'A': A_text, 'B': A_text + ' ' + novel_text,
                      'novel': novel_text})

    conn.close()
    print(f"Loaded {len(pairs)} pairs from {checked:,} revisions "
          f"({len(pairs)/max(checked,1)*100:.1f}% pass rate)")
    return pairs


# ── Checkpoint finder ─────────────────────────────────────────────────────────

def _find_checkpoint(default):
    candidates = [
        default,
        '/kaggle/working/checkpoints/wiki_model.pt',
        '/kaggle/working/checkpoints/kaggle_model.pt',
        str(ROOT / 'checkpoints' / 'wiki_model.pt'),
        str(ROOT / 'checkpoints' / 'val_model.pt'),
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return Path(default)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db',   required=True)
    ap.add_argument('--ckpt', default='/kaggle/working/checkpoints/wiki_model.pt')
    ap.add_argument('--n',    type=int, default=500)
    ap.add_argument('--min_added', type=int, default=2)
    args = ap.parse_args()

    print('=' * 66)
    print('DELTA SYSTEM -- NewsEdits Zero-Shot Domain-Transfer Evaluation')
    print('=' * 66)
    print(f'Device     : {DEVICE}')
    print(f'Checkpoint : Wikipedia-trained G (NEVER saw news data)')
    print(f'Metric     : DELTA_PPL + SPECIFICITY (same as Wikipedia/HotpotQA)')
    print()

    # Load pairs
    pairs = load_newsedits_pairs(args.db, n_pairs=args.n,
                                  min_added=args.min_added)
    if len(pairs) < 50:
        print(f'ERROR: only {len(pairs)} pairs. Try --min_added 1')
        sys.exit(1)
    print()

    # Load model
    tok   = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = DeltaSystem().to(DEVICE)
    ckpt  = _find_checkpoint(args.ckpt)
    if ckpt.exists():
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE), strict=False)
        print(f'Checkpoint: {ckpt}  (Wikipedia-trained, zero-shot on news)')
    else:
        print(f'WARNING: no checkpoint at {ckpt} — using random weights')
    print()

    # Evaluate — identical to Wikipedia and HotpotQA evaluation
    print('=' * 66)
    print(f'HELD-OUT EVAL -- {len(pairs)} AP news revision pairs')
    print('(zero-shot: G trained on Wikipedia, never saw news)')
    print('=' * 66)
    results = evaluate(model, pairs, tok)

    # Cross-domain summary
    print()
    print('=' * 66)
    print('CROSS-DOMAIN COMPARISON')
    print('=' * 66)
    print(f"{'Dataset':<30} {'DELTA_PPL':>10} {'SPEC':>8} {'Domain':>20}")
    print('-' * 66)
    print(f"{'Wikipedia (8000tr/1000ev)':<30} {'  +755':>10} {'  +608':>8} {'same domain':>20}")
    print(f"{'HotpotQA (5000tr/500ev)':<30} {'  +480':>10} {' +2547':>8} {'cross-dataset':>20}")
    print(f"{'NewsEdits AP (0tr/500ev)':<30} "
          f"{results['delta_ppl']:>+10.0f} "
          f"{results['specificity']:>+8.0f} "
          f"{'cross-domain (news)':>20}")
    print()
    print('Key: DELTA_PPL>2 AND SPEC>2 = PASS')
    print(f"NewsEdits: {'PASS -- zero-shot domain transfer confirmed' if results['pass'] else 'FAIL'}")
    print('=' * 66)


if __name__ == '__main__':
    main()
