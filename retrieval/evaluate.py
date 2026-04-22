"""
Evaluation metrics for retrieval experiments.
Reports: Recall@k (k=2,5,10), MRR, NDCG@k
"""

import math
from typing import List, Dict


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant passages found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for r in relevant_ids if r in top_k)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank — rank of the first relevant hit."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """NDCG@k — position-weighted recall (binary relevance)."""
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant docs ranked at top positions
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(
    results: List[Dict],  # [{"query_id": ..., "retrieved": [...], "relevant": [...]}]
    ks: List[int] = (2, 5, 10),
) -> Dict:
    """
    Aggregate metrics across all queries.
    results: list of dicts with keys: query_id, retrieved (ordered list of ids), relevant (list of ids)
    Returns dict of metric -> value (averaged over all queries).
    """
    if not results:
        return {}

    totals = {f"recall@{k}": 0.0 for k in ks}
    totals["mrr"] = 0.0
    totals["ndcg@10"] = 0.0

    for item in results:
        retrieved = item["retrieved"]
        relevant = item["relevant"]
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
        totals["mrr"] += mrr(retrieved, relevant)
        totals["ndcg@10"] += ndcg_at_k(retrieved, relevant, 10)

    n = len(results)
    return {metric: round(val / n, 4) for metric, val in totals.items()}


def print_results_table(system_name: str, metrics: Dict) -> None:
    """Pretty-print a single system's results."""
    print(f"\n{'='*55}")
    print(f"  {system_name}")
    print(f"{'='*55}")
    for metric, value in metrics.items():
        print(f"  {metric:<15} {value:.4f}  ({value*100:.2f}%)")
    print(f"{'='*55}")


def compare_systems(systems: Dict[str, Dict]) -> None:
    """Print comparison table across multiple systems."""
    if not systems:
        return
    metrics = list(next(iter(systems.values())).keys())
    col_w = 12

    header = f"{'System':<30}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, mets in systems.items():
        row = f"{name:<30}" + "".join(f"{mets.get(m, 0)*100:>{col_w}.2f}" for m in metrics)
        print(row)
    print("=" * len(header))
    print("(all values in %)")
