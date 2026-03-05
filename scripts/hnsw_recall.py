"""
Verify HNSW recall@k by comparing approximate ANN results (hnsw.ef_search)
against exact brute-force search for a sample of queries.

Usage:
    python hnsw_recall.py [--n-queries 100] [--ef-search 100] [--k 10] [--csv]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection
from config import DATA_DIR, EF_SEARCH, K, DATASET_NAME

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def _vec_str(emb: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in emb) + "]"


def load_query_embeddings(dataset: str) -> tuple[np.ndarray, list[str]]:
    emb_path = DATA_DIR / f"{dataset}_query_embeddings.npy"
    ids_path = DATA_DIR / f"{dataset}_query_ids.json"

    if not emb_path.exists() or not ids_path.exists():
        print(
            f"Missing embedding files for dataset '{dataset}'.\n"
            "Run `make embed` first to generate query embeddings."
        )
        sys.exit(1)

    embeddings = np.load(emb_path)
    query_ids = json.loads(ids_path.read_text())
    return embeddings, query_ids


def hnsw_search(cur, vec_str: str, k: int, ef_search: int) -> set[str]:
    cur.execute(f"SET hnsw.ef_search = {ef_search}")
    cur.execute(
        "SELECT id FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
        (vec_str, k),
    )
    return {str(row[0]) for row in cur.fetchall()}


def _find_node_types(node: dict) -> list[str]:
    """Recursively collect all Node Type values in a plan tree."""
    types = [node.get("Node Type", "")]
    for child in (node.get("Plans") or []):
        if isinstance(child, dict):
            types.extend(_find_node_types(child))
    return types


def verify_exact_plan(cur, vec_str: str, k: int) -> None:
    """Assert that the forced sequential scan actually uses Seq Scan (not an index)."""
    cur.execute(
        "EXPLAIN (FORMAT JSON) SELECT id FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
        (vec_str, k),
    )
    plan = cur.fetchone()[0]
    all_node_types = _find_node_types(plan[0]["Plan"])
    has_seq_scan = any("Seq Scan" in t or "Gather" in t for t in all_node_types)
    has_index = any("Index" in t or "Custom Scan" in t for t in all_node_types)
    if not has_seq_scan or has_index:
        raise RuntimeError(
            f"exact_search is NOT using Seq Scan (plan nodes: {all_node_types}). "
            "Recall results would be invalid. Check that enable_indexscan=off works for pgvector."
        )


def exact_search(cur, vec_str: str, k: int, verify: bool = False) -> set[str]:
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_bitmapscan = off")
    try:
        if verify:
            verify_exact_plan(cur, vec_str, k)
        cur.execute(
            "SELECT id FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec_str, k),
        )
        return {str(row[0]) for row in cur.fetchall()}
    finally:
        cur.execute("SET enable_indexscan = on")
        cur.execute("SET enable_bitmapscan = on")


def compute_recalls(
    embeddings: np.ndarray,
    query_ids: list[str],
    n_queries: int,
    ef_search: int,
    k: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    indices = list(range(len(query_ids)))
    sampled = rng.sample(indices, min(n_queries, len(indices)))

    conn = get_connection()
    try:
        cur = conn.cursor()
        rows = []
        first = True
        for idx in sampled:
            vec_str = _vec_str(embeddings[idx])
            hnsw_ids = hnsw_search(cur, vec_str, k, ef_search)
            # Verify seq scan plan on first query only (avoid per-query overhead)
            exact_ids = exact_search(cur, vec_str, k, verify=first)
            first = False
            recall = len(hnsw_ids & exact_ids) / max(len(exact_ids), 1)
            rows.append({"query_id": query_ids[idx], "recall": recall})
    finally:
        conn.close()
    return rows


def print_report(rows: list[dict], ef_search: int, k: int, threshold: float = 0.95) -> None:
    recalls = [r["recall"] for r in rows]
    n = len(recalls)
    mean_r = sum(recalls) / n
    median_r = sorted(recalls)[n // 2]
    min_r = min(recalls)
    max_r = max(recalls)
    above_95 = sum(1 for r in recalls if r >= 0.95)
    above_90 = sum(1 for r in recalls if r >= 0.90)

    print(f"\nHNSW Recall Verification (ef_search={ef_search}, k={k}, n={n} queries)\n")
    print(f"Per-query recall@{k} distribution:")
    print(f"  Mean:   {mean_r:.3f}")
    print(f"  Median: {median_r:.3f}")
    print(f"  Min:    {min_r:.3f}")
    print(f"  Max:    {max_r:.3f}")
    print(f"  Queries with recall >= 0.95: {above_95}/{n} ({100*above_95/n:.1f}%)")
    print(f"  Queries with recall >= 0.90: {above_90}/{n} ({100*above_90/n:.1f}%)")

    verdict = "PASS" if mean_r >= threshold else "FAIL"
    mark = "\u2713" if mean_r >= threshold else "\u2717"
    print(f"\nVerdict: HNSW recall@{k} = {mean_r:.3f} {mark} (threshold: {threshold})")
    if verdict == "FAIL":
        print("  WARNING: recall below threshold — consider increasing ef_search.")


def main():
    parser = argparse.ArgumentParser(description="Verify HNSW recall vs exact search")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=EF_SEARCH)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", action="store_true", help="Write results/hnsw_recall.csv")
    args = parser.parse_args()

    embeddings, query_ids = load_query_embeddings(args.dataset)
    rows = compute_recalls(embeddings, query_ids, args.n_queries, args.ef_search, args.k, args.seed)
    print_report(rows, args.ef_search, args.k)

    if args.csv:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "hnsw_recall.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "recall"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
