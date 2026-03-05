"""
Export benchmark results and index build results to CSV files in results/.

Usage:
    python export_results.py [--out-dir results]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def export_table(conn, table: str, out_path: Path) -> int:
    cur = conn.cursor()
    with out_path.open("w") as f:
        cur.copy_expert(f"COPY {table} TO STDOUT WITH CSV HEADER", f)
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]


def main():
    parser = argparse.ArgumentParser(description="Export benchmark results to CSV")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    conn = get_connection()

    tables = {
        "benchmark_results":   out_dir / "benchmark_results.csv",
        "index_build_results": out_dir / "index_build_results.csv",
    }

    for table, path in tables.items():
        n = export_table(conn, table, path)
        print(f"  {table:<25} → {path}  ({n:,} rows)")

    conn.close()
    print("\nExport complete.")


if __name__ == "__main__":
    main()
