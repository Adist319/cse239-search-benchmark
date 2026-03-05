"""
Compute retrieval-quality metrics (NDCG@k, Precision@k, Recall@k, MAP@k)
from stored benchmark_results against ground-truth qrels.

Supports graded relevance (NFCorpus: 0/1/2) and binary relevance (FiQA: 0/1).
NDCG uses actual grade values; Precision/Recall/MAP use binary (rel > 0).

Usage:
    python compute_relevance.py [--k 10] [--dataset nfcorpus] [--methods ...]
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ── Metric functions (graded relevance; binary metrics use rel > 0) ────────────

def dcg(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    """NDCG using actual relevance grades (handles both binary and graded)."""
    rels = [graded.get(doc, 0) for doc in retrieved[:k]]
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    relevant = {d for d, r in graded.items() if r > 0}
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    relevant = {d for d, r in graded.items() if r > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    relevant = {d for d, r in graded.items() if r > 0}
    if not relevant:
        return 0.0
    hits, running_sum = 0, 0.0
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            hits += 1
            running_sum += hits / (i + 1)
    return running_sum / len(relevant)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_qrels(conn) -> dict[str, dict[str, int]]:
    """Returns {query_ext_id -> {doc_ext_id -> relevance_grade}}."""
    cur = conn.cursor()
    cur.execute("SELECT query_ext_id, doc_ext_id, relevance FROM qrels WHERE relevance > 0")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qid, did, score in cur.fetchall():
        qrels[qid][did] = score
    return dict(qrels)


def load_results(conn, methods: list[str], eval_qids: set[str], dataset: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Returns {method -> {query_ext_id -> [ranked doc_ext_ids]}}."""
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(methods))
    dataset_filter = "AND run_config->>'dataset' = %s" if dataset else ""
    params = methods + [list(eval_qids)]
    if dataset:
        params.append(dataset)
    cur.execute(
        f"""
        SELECT DISTINCT ON (method, query_ext_id) method, query_ext_id, result_ids
        FROM benchmark_results
        WHERE method IN ({placeholders})
          AND query_ext_id = ANY(%s)
          {dataset_filter}
        ORDER BY method, query_ext_id, created_at DESC
        """,
        params,
    )
    results: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for method, qid, doc_ids in cur.fetchall():
        results[method][qid] = list(doc_ids) if doc_ids else []
    return dict(results)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    method_results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> dict[str, float]:
    """Compute mean metrics over all evaluated queries for one method."""
    totals = defaultdict(float)
    n = 0

    for qid, relevant in qrels.items():
        retrieved = method_results.get(qid, [])
        totals["ndcg"] += ndcg(retrieved, relevant, k)
        totals["precision"] += precision_at_k(retrieved, relevant, k)
        totals["recall"] += recall_at_k(retrieved, relevant, k)
        totals["map"] += average_precision(retrieved, relevant, k)
        n += 1

    return {metric: total / n for metric, total in totals.items()} if n else {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute retrieval quality metrics")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["vanilla_fts", "pgvector", "paradedb_bm25", "hybrid_rrf"],
    )
    parser.add_argument("--dataset", type=str, default=None, help="Filter results by dataset (e.g. fiqa, nfcorpus)")
    parser.add_argument("--csv", action="store_true", help="Also write results/relevance_metrics.csv")
    args = parser.parse_args()

    conn = get_connection()
    qrels = load_qrels(conn)
    eval_qids = set(qrels.keys())
    print(f"Evaluating on {len(eval_qids)} queries with qrels (k={args.k})\n")

    results = load_results(conn, args.methods, eval_qids, dataset=args.dataset)
    conn.close()

    metrics_rows = []
    header = f"{'Method':<18} {'NDCG@k':>8} {'P@k':>8} {'Recall@k':>10} {'MAP@k':>8}"
    print(header)
    print("─" * len(header))

    for method in args.methods:
        if method not in results:
            print(f"  {method}: no results found")
            continue
        m = evaluate(results[method], qrels, args.k)
        print(
            f"{method:<18} {m['ndcg']:>8.4f} {m['precision']:>8.4f}"
            f" {m['recall']:>10.4f} {m['map']:>8.4f}"
        )
        metrics_rows.append({"method": method, "k": args.k, **m})

    if args.csv:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "relevance_metrics.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "k", "ndcg", "precision", "recall", "map"])
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
