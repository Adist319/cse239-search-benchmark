"""
Main benchmark runner.

Design:
  1. WARMUP per benchmark block (not per query) — warms buffer cache + plan cache
     for the entire query set before any measurements.
  2. TIMED RUNS use normal query execution only — no EXPLAIN in the hot loop.
  3. EXPLAIN ANALYZE collected in a separate optional pass (--explain flag),
     one plan per (query, method) pair, stored in benchmark_results.plan_json.
  4. Results stored as (p50, p95, p99, mean) per query in benchmark_results table.

Usage:
    python run_benchmarks.py [options]

    # Run all methods, warm cache, k=10
    python run_benchmarks.py

    # Run only fts and pgvector, 10 timed runs per query
    python run_benchmarks.py --methods vanilla_fts pgvector --n-runs 10

    # Collect EXPLAIN ANALYZE plans after timing
    python run_benchmarks.py --explain

    # Limit to first 50 queries (dev mode)
    python run_benchmarks.py --limit 50
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR,
    DATASET_NAME,
    EF_SEARCH,
    N_RUNS,
    N_WARMUP,
    RRF_K,
    RRF_POOL,
    K,
)
from db import get_connection

# ── SQL Templates ─────────────────────────────────────────────────────────────
# Using %s-style placeholders; params built per method in _build_params()

SQL_VANILLA_FTS = """
SELECT d.id, d.ext_id, ts_rank_cd(d.body_tsv, query) AS score
FROM documents d,
     plainto_tsquery('english', %s) AS query
WHERE d.body_tsv @@ query
ORDER BY score DESC
LIMIT %s
"""

SQL_PGVECTOR = """
SELECT id, ext_id, 1 - (embedding <=> %s::vector) AS score
FROM documents
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

SQL_PARADEDB_BM25 = """
SELECT id, ext_id, pdb.score(id) AS score
FROM documents
WHERE (title @@@ pdb.match(%s) OR body @@@ pdb.match(%s))
ORDER BY pdb.score(id) DESC
LIMIT %s
"""

SQL_HYBRID_RRF = """
WITH bm25_results AS (
    SELECT id, ext_id,
           pdb.score(id) AS bm25_score,
           ROW_NUMBER() OVER (ORDER BY pdb.score(id) DESC) AS rank
    FROM documents
    WHERE (title @@@ pdb.match(%s) OR body @@@ pdb.match(%s))
    LIMIT %s
),
vector_results AS (
    SELECT id, ext_id,
           embedding <=> %s::vector AS distance,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT %s
),
rrf AS (
    SELECT id, ext_id, 1.0 / (%s + rank) AS score FROM bm25_results
    UNION ALL
    SELECT id, ext_id, 1.0 / (%s + rank) AS score FROM vector_results
)
SELECT r.id, r.ext_id, SUM(r.score) AS rrf_score
FROM rrf r
GROUP BY r.id, r.ext_id
ORDER BY rrf_score DESC
LIMIT %s
"""


# ── Parameter builders ─────────────────────────────────────────────────────────


def _vec_str(emb: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in emb) + "]"


# ParadeDB uses Tantivy's query parser — special chars must be stripped
# so natural-language queries don't get misinterpreted as operators.
_BM25_SPECIAL = str.maketrans("", "", '()[]{}:"\'^~*?\\/')

def _sanitize_bm25(text: str) -> str:
    return text.translate(_BM25_SPECIAL).strip()


def build_params_fts(q: dict, k: int, **_) -> tuple:
    return (q["text"], k)


def build_params_pgvector(q: dict, k: int, ef_search: int = EF_SEARCH, **_) -> tuple:
    v = _vec_str(q["embedding"])
    return (v, v, k)


def build_params_bm25(q: dict, k: int, **_) -> tuple:
    t = _sanitize_bm25(q["text"])
    return (t, t, k)  # title match, body match, LIMIT


def build_params_hybrid(
    q: dict,
    k: int,
    rrf_pool: int = RRF_POOL,
    rrf_k: int = RRF_K,
    ef_search: int = EF_SEARCH,
    **_,
) -> tuple:
    v = _vec_str(q["embedding"])
    t = _sanitize_bm25(q["text"])
    return (
        t, t,       # bm25_results: title match, body match
        rrf_pool,   # bm25_results: LIMIT
        v,
        v,
        v,
        rrf_pool,  # vector_results: <=> ?, ORDER BY <=> ?, LIMIT ?
        rrf_k,
        rrf_k,  # rrf: / (? + rank), / (? + rank)
        k,  # final LIMIT
    )


METHODS = {
    "vanilla_fts": (SQL_VANILLA_FTS, build_params_fts),
    "pgvector": (SQL_PGVECTOR, build_params_pgvector),
    "paradedb_bm25": (SQL_PARADEDB_BM25, build_params_bm25),
    "hybrid_rrf": (SQL_HYBRID_RRF, build_params_hybrid),
}


# ── Session setup ──────────────────────────────────────────────────────────────


def setup_session(conn, method: str, ef_search: int) -> None:
    """Set session-level parameters before a benchmark block."""
    cur = conn.cursor()
    cur.execute("SET synchronous_commit = off")  # less WAL noise during reads
    cur.execute("SET log_min_duration_statement = -1")  # silence PG logging

    if method in ("pgvector", "hybrid_rrf"):
        cur.execute(f"SET hnsw.ef_search = {ef_search}")

    conn.commit()


# ── Warmup ─────────────────────────────────────────────────────────────────────


def run_warmup(
    conn, sql: str, queries: list[dict], n_warmup: int, param_fn, method: str, **kwargs
) -> None:
    """
    Run the first n_warmup queries once each to warm the buffer cache and plan cache.
    Results are discarded.
    """
    cur = conn.cursor()
    warmup_queries = queries[:n_warmup]
    for q in warmup_queries:
        try:
            cur.execute(sql, param_fn(q, K, **kwargs))
            cur.fetchall()
        except Exception:
            conn.rollback()


# ── Timed runs ─────────────────────────────────────────────────────────────────


def run_timed(
    conn,
    sql: str,
    queries: list[dict],
    n_runs: int,
    k: int,
    param_fn,
    **kwargs,
) -> list[dict]:
    """
    Execute each query n_runs times using normal query execution (no EXPLAIN).
    Returns per-query aggregated latency stats.
    """
    cur = conn.cursor()
    results = []

    for q in tqdm(queries, desc="  timing", leave=False):
        params = param_fn(q, k, **kwargs)
        latencies_ms = []
        last_rows = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"Query failed for method: {e}") from e
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            last_rows = rows

        latencies_ms.sort()
        n = len(latencies_ms)
        p50 = latencies_ms[n // 2]
        p95 = latencies_ms[min(int(0.95 * n), n - 1)]
        p99 = latencies_ms[min(int(0.99 * n), n - 1)]
        mean = sum(latencies_ms) / n

        result_ids = [str(row[1]) for row in last_rows]  # ext_id is col index 1

        results.append(
            {
                "query_ext_id": q["id"],
                "query_text": q["text"],
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "latency_mean_ms": mean,
                "n_runs": n_runs,
                "num_results": len(last_rows),
                "result_ids": result_ids,
                "plan_json": None,
            }
        )

    return results


# ── EXPLAIN ANALYZE pass ───────────────────────────────────────────────────────


def run_explain_pass(
    conn, sql: str, queries: list[dict], param_fn, method: str, k: int, **kwargs
) -> dict[str, Any]:
    """
    Separate optional pass: run EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) for each query.
    Returns mapping of query_ext_id → plan JSON.
    """
    cur = conn.cursor()
    plans = {}
    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"

    for q in tqdm(queries, desc=f"  explain {method}", leave=False):
        params = param_fn(q, k, **kwargs)
        try:
            cur.execute(explain_sql, params)
            plan = cur.fetchone()[0]
            plans[q["id"]] = plan
        except Exception as e:
            conn.rollback()
            plans[q["id"]] = {"error": str(e)}

    return plans


# ── Persist results ────────────────────────────────────────────────────────────


def store_results(conn, method: str, rows: list[dict], run_config: dict) -> None:
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO benchmark_results
            (method, query_ext_id, query_text,
             latency_p50_ms, latency_p95_ms, latency_p99_ms, latency_mean_ms,
             n_runs, num_results, result_ids, plan_json, run_config)
        VALUES %s
        """,
        [
            (
                method,
                r["query_ext_id"],
                r["query_text"],
                r["latency_p50_ms"],
                r["latency_p95_ms"],
                r["latency_p99_ms"],
                r["latency_mean_ms"],
                r["n_runs"],
                r["num_results"],
                r["result_ids"],
                json.dumps(r["plan_json"]) if r["plan_json"] else None,
                json.dumps(run_config),
            )
            for r in rows
        ],
        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        page_size=500,
    )
    conn.commit()


# ── Query loading ──────────────────────────────────────────────────────────────


def load_queries(dataset: str, limit: int = 0) -> list[dict]:
    """Load queries with their pre-computed embeddings."""
    import json as _json

    queries_path = DATA_DIR / f"{dataset}_queries.jsonl"
    emb_path = DATA_DIR / f"{dataset}_query_embeddings.npy"
    ids_path = DATA_DIR / f"{dataset}_query_ids.json"

    if not queries_path.exists():
        print(f"Missing {queries_path} — run `make download` first")
        sys.exit(1)

    with queries_path.open() as f:
        raw = [_json.loads(line) for line in f]

    embeddings = None
    id_to_idx = {}
    if emb_path.exists() and ids_path.exists():
        embeddings = np.load(emb_path)
        q_ids = _json.loads(ids_path.read_text())
        id_to_idx = {qid: i for i, qid in enumerate(q_ids)}

    queries = []
    for row in raw:
        q = {"id": row["_id"], "text": row["text"], "embedding": None}
        if embeddings is not None and row["_id"] in id_to_idx:
            q["embedding"] = embeddings[id_to_idx[row["_id"]]]
        queries.append(q)

    if limit:
        queries = queries[:limit]

    return queries


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run search benchmarks")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
        help="Methods to benchmark",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=N_WARMUP,
        help="Warmup queries per block (default %(default)s)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help="Timed repetitions per query (default %(default)s)",
    )
    parser.add_argument(
        "--k", type=int, default=K, help="Result set size (default %(default)s)"
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=EF_SEARCH,
        help="HNSW ef_search (default %(default)s)",
    )
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--rrf-pool", type=int, default=RRF_POOL)
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of queries (0 = all)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run EXPLAIN ANALYZE pass after timed runs",
    )
    parser.add_argument(
        "--label", default="", help="Optional label added to run_config for tracking"
    )
    args = parser.parse_args()

    queries = load_queries(args.dataset, args.limit)
    print(f"Loaded {len(queries)} queries from dataset '{args.dataset}'")

    # Check which methods need embeddings
    needs_emb = {"pgvector", "hybrid_rrf"} & set(args.methods)
    if needs_emb:
        missing_emb = [q for q in queries if q["embedding"] is None]
        if missing_emb:
            print(
                f"WARNING: {len(missing_emb)} queries have no embedding — "
                f"run `make embed` first. These queries will be skipped for {needs_emb}."
            )

    conn = get_connection()

    run_config = {
        "dataset": args.dataset,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "k": args.k,
        "ef_search": args.ef_search,
        "rrf_k": args.rrf_k,
        "rrf_pool": args.rrf_pool,
        "label": args.label,
    }

    kwargs = dict(
        ef_search=args.ef_search,
        rrf_k=args.rrf_k,
        rrf_pool=args.rrf_pool,
    )

    for method in args.methods:
        sql, param_fn = METHODS[method]
        print(f"\n── {method} ──────────────────────────────────")

        # Filter out queries without embeddings for vector methods
        active_queries = queries
        if method in ("pgvector", "hybrid_rrf"):
            active_queries = [q for q in queries if q["embedding"] is not None]
            if len(active_queries) < len(queries):
                print(f"  Using {len(active_queries)} queries with embeddings")

        setup_session(conn, method, args.ef_search)

        # 1. WARMUP PHASE (per block, not per query)
        print(f"  Warming up ({args.n_warmup} queries) ...")
        run_warmup(conn, sql, active_queries, args.n_warmup, param_fn, method, **kwargs)

        # 2. TIMED PHASE (normal execute only, no EXPLAIN)
        print(f"  Timing ({len(active_queries)} queries × {args.n_runs} runs) ...")
        results = run_timed(
            conn, sql, active_queries, args.n_runs, args.k, param_fn, **kwargs
        )

        # 3. EXPLAIN ANALYZE PASS (optional, separate)
        if args.explain:
            print(f"  EXPLAIN ANALYZE pass ...")
            plans = run_explain_pass(
                conn, sql, active_queries, param_fn, method, args.k, **kwargs
            )
            for r in results:
                r["plan_json"] = plans.get(r["query_ext_id"])

        # 4. STORE
        store_results(conn, method, results, run_config)

        # Quick summary
        p50s = [r["latency_p50_ms"] for r in results]
        p95s = [r["latency_p95_ms"] for r in results]
        p99s = [r["latency_p99_ms"] for r in results]
        print(
            f"  p50={_agg_p50(p50s):.1f}ms  p95={_agg_p50(p95s):.1f}ms  p99={_agg_p50(p99s):.1f}ms  "
            f"(median of per-query medians)"
        )

    conn.close()
    print("\nBenchmark complete. Run `make export` to export results to CSV.")


def _agg_p50(vals: list[float]) -> float:
    """Median of a list (used to summarize per-query p50s)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[len(s) // 2]


if __name__ == "__main__":
    main()
