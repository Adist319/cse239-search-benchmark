"""
Analyze EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) plans stored in benchmark_results.plan_json.

For each method, aggregates across all queries:
  - Planning and execution time distributions (median, p95)
  - Index node types used
  - Buffer cache hit rate
  - Planner row-estimate accuracy

Plans are collected by running: make bench-explain

Usage:
    python analyze_plans.py [--dataset fiqa] [--methods vanilla_fts pgvector ...]
    python analyze_plans.py --csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Generator

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Node types that indicate index usage — matched by substring
INDEX_NODE_KEYWORDS = ("Index", "Bitmap", "Custom Scan")

# Cap actual/plan ratio to avoid outliers from near-zero estimates dominating
ACCURACY_CAP = 10.0


# ── Plan tree traversal ────────────────────────────────────────────────────────


def walk_plan(node: dict) -> Generator[tuple, None, None]:
    """
    Recursively walk a Postgres EXPLAIN JSON plan node tree.

    Yields tuples of:
        (node_type, plan_rows, actual_rows, shared_hit, shared_read)

    All integer fields default to 0 when absent.
    """
    node_type    = node.get("Node Type", "Unknown")
    plan_rows    = node.get("Plan Rows", 0) or 0
    actual_rows  = node.get("Actual Rows", 0) or 0
    shared_hit   = node.get("Shared Hit Blocks", 0) or 0
    shared_read  = node.get("Shared Read Blocks", 0) or 0

    yield (node_type, plan_rows, actual_rows, shared_hit, shared_read)

    for child in (node.get("Plans") or []):
        if isinstance(child, dict):
            yield from walk_plan(child)


# ── Per-plan extraction ────────────────────────────────────────────────────────


def extract_plan_stats(plan_json) -> dict | None:
    """
    Parse one benchmark_results.plan_json value.

    plan_json arrives from psycopg2 as a Python list (psycopg2 decodes JSONB
    automatically).  The Postgres FORMAT JSON output is:
        [ { "Plan": {...}, "Planning Time": X, "Execution Time": Y, ... } ]

    Returns a dict with keys:
        planning_time_ms, execution_time_ms,
        index_nodes (Counter), shared_hit, shared_read, accuracy_ratios (list)

    Returns None when the plan is missing or malformed.
    """
    if plan_json is None:
        return None

    try:
        # psycopg2 returns JSONB as a Python object; handle both list and dict
        top = plan_json[0] if isinstance(plan_json, list) else plan_json
        planning_time_ms  = float(top.get("Planning Time",  0.0))
        execution_time_ms = float(top.get("Execution Time", 0.0))
        root_node = top.get("Plan", {})
    except (IndexError, KeyError, TypeError, ValueError):
        return None

    index_nodes    = Counter()
    shared_hit     = 0
    shared_read    = 0
    accuracy_ratios: list[float] = []

    for node_type, plan_rows, actual_rows, s_hit, s_read in walk_plan(root_node):
        # Collect index node types
        if any(kw in node_type for kw in INDEX_NODE_KEYWORDS):
            index_nodes[node_type] += 1

        shared_hit  += s_hit
        shared_read += s_read

        # Planner accuracy: ratio actual/plan, capped to avoid extreme outliers
        if plan_rows > 0:
            ratio = actual_rows / plan_rows
            accuracy_ratios.append(min(ratio, ACCURACY_CAP))

    return {
        "planning_time_ms":  planning_time_ms,
        "execution_time_ms": execution_time_ms,
        "index_nodes":       index_nodes,
        "shared_hit":        shared_hit,
        "shared_read":       shared_read,
        "accuracy_ratios":   accuracy_ratios,
    }


# ── Data loading ───────────────────────────────────────────────────────────────


def load_plans(conn, methods: list[str], dataset: str) -> dict[str, list[dict]]:
    """
    Returns {method -> list of extracted plan stat dicts} for rows where
    plan_json IS NOT NULL and the run_config matches the requested dataset.
    """
    if not methods:
        return {}
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(methods))
    cur.execute(
        f"""
        SELECT method, plan_json
        FROM benchmark_results
        WHERE method IN ({placeholders})
          AND plan_json IS NOT NULL
          AND run_config->>'dataset' = %s
        ORDER BY method, id
        """,
        methods + [dataset],
    )

    method_plans: dict[str, list[dict]] = {m: [] for m in methods}
    for method, plan_json in cur.fetchall():
        stats = extract_plan_stats(plan_json)
        if stats is not None:
            method_plans[method].append(stats)

    return method_plans


# ── Aggregation ────────────────────────────────────────────────────────────────


def aggregate_method(plan_stats: list[dict]) -> dict:
    """Compute aggregate statistics across all per-query plan stat dicts."""
    if not plan_stats:
        raise ValueError("aggregate_method called with empty plan list")
    planning_times  = [s["planning_time_ms"]  for s in plan_stats]
    execution_times = [s["execution_time_ms"] for s in plan_stats]

    total_hit  = sum(s["shared_hit"]  for s in plan_stats)
    total_read = sum(s["shared_read"] for s in plan_stats)

    all_ratios: list[float] = []
    for s in plan_stats:
        all_ratios.extend(s["accuracy_ratios"])

    combined_index_nodes: Counter = Counter()
    for s in plan_stats:
        combined_index_nodes.update(s["index_nodes"])

    planning_arr  = np.array(planning_times)
    execution_arr = np.array(execution_times)

    hit_rate = (
        total_hit / (total_hit + total_read) if (total_hit + total_read) > 0 else None
    )
    mean_accuracy = float(np.mean(all_ratios)) if all_ratios else None

    return {
        "n_queries":           len(plan_stats),
        "planning_median_ms":  float(np.median(planning_arr)),
        "planning_p95_ms":     float(np.percentile(planning_arr, 95)),
        "execution_median_ms": float(np.median(execution_arr)),
        "execution_p95_ms":    float(np.percentile(execution_arr, 95)),
        "index_nodes":         combined_index_nodes,
        "buffer_hit_rate":     hit_rate,
        "mean_accuracy_ratio": mean_accuracy,
    }


# ── Formatting ─────────────────────────────────────────────────────────────────


def format_index_nodes(counter: Counter, n_queries: int) -> str:
    """Format index node counts as 'Node Type xN' strings."""
    if not counter:
        return "none"
    parts = [f"{node_type} x{count}" for node_type, count in counter.most_common(5)]
    return ", ".join(parts)


def print_method_summary(method: str, agg: dict) -> None:
    n       = agg["n_queries"]
    hit     = agg["buffer_hit_rate"]
    acc     = agg["mean_accuracy_ratio"]
    nodes   = format_index_nodes(agg["index_nodes"], n)
    hit_str = f"{hit * 100:.1f}%" if hit is not None else "n/a (no reads)"
    acc_str = f"{acc:.2f}x" if acc is not None else "n/a"

    print(f"\nMethod: {method}")
    print(f"  Planning time:    median {agg['planning_median_ms']:.2f}ms"
          f"  (p95 {agg['planning_p95_ms']:.2f}ms)")
    print(f"  Execution time:   median {agg['execution_median_ms']:.2f}ms"
          f"  (p95 {agg['execution_p95_ms']:.2f}ms)")
    print(f"  Index nodes:      {nodes}")
    print(f"  Buffer cache hit: {hit_str}")
    print(f"  Planner accuracy: mean ratio {acc_str} (actual/estimated rows)")


# ── CSV export ─────────────────────────────────────────────────────────────────


def write_csv(method_aggs: list[tuple[str, dict]], out_path: Path, dataset: str = "") -> None:
    """Write per-method aggregate stats to CSV."""
    fieldnames = [
        "dataset",
        "method",
        "n_queries",
        "planning_median_ms",
        "planning_p95_ms",
        "execution_median_ms",
        "execution_p95_ms",
        "top_index_nodes",
        "buffer_hit_rate_pct",
        "mean_accuracy_ratio",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)

    # Read existing data (from other datasets) to preserve it
    existing_rows: list[dict] = []
    if out_path.exists():
        with out_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Keep rows from other datasets; drop rows from this dataset (will be replaced)
                row_dataset = row.get("dataset", "")
                if row_dataset != dataset:
                    existing_rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write back preserved rows from other datasets
        for row in existing_rows:
            writer.writerow(row)

        # Write new rows for current dataset
        for method, agg in method_aggs:
            hit = agg["buffer_hit_rate"]
            writer.writerow({
                "dataset":              dataset,
                "method":               method,
                "n_queries":            agg["n_queries"],
                "planning_median_ms":   round(agg["planning_median_ms"], 4),
                "planning_p95_ms":      round(agg["planning_p95_ms"], 4),
                "execution_median_ms":  round(agg["execution_median_ms"], 4),
                "execution_p95_ms":     round(agg["execution_p95_ms"], 4),
                "top_index_nodes":      format_index_nodes(agg["index_nodes"], agg["n_queries"]),
                "buffer_hit_rate_pct":  round(hit * 100, 2) if hit is not None else "",
                "mean_accuracy_ratio":  round(agg["mean_accuracy_ratio"], 4)
                                        if agg["mean_accuracy_ratio"] is not None else "",
            })


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze EXPLAIN ANALYZE plans from benchmark_results")
    parser.add_argument("--dataset", default="fiqa", help="Dataset name (default: fiqa)")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["vanilla_fts", "pgvector", "paradedb_bm25", "hybrid_rrf"],
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write results/plan_analysis.csv",
    )
    args = parser.parse_args()

    conn = get_connection()
    method_plans = load_plans(conn, args.methods, args.dataset)
    conn.close()

    total_plans = sum(len(v) for v in method_plans.values())
    if total_plans == 0:
        print("No EXPLAIN plans found. Run: make bench-explain")
        sys.exit(0)

    print(f"Query Plan Analysis (dataset={args.dataset}, N={total_plans} queries with plans)")

    method_aggs: list[tuple[str, dict]] = []
    for method in args.methods:
        plans = method_plans.get(method, [])
        if not plans:
            print(f"\nMethod: {method}")
            print("  No plans found — run: make bench-explain")
            continue
        agg = aggregate_method(plans)
        print_method_summary(method, agg)
        method_aggs.append((method, agg))

    if args.csv and method_aggs:
        out = RESULTS_DIR / "plan_analysis.csv"
        write_csv(method_aggs, out, dataset=args.dataset)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
