"""
Create and time all three index types:
  1. GIN on body_tsv (vanilla FTS)
  2. HNSW on embedding (pgvector)
  3. BM25 on (id, title, body) (ParadeDB pg_search)

Stores results in index_build_results table.
Verifies index usage via EXPLAIN ANALYZE on a sample query.

Usage:
    python create_indexes.py [--drop] [--skip gin|hnsw|bm25]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection, check_extensions


INDEXES = [
    {
        "name":  "idx_documents_tsv",
        "type":  "gin",
        "label": "GIN (Vanilla FTS)",
        "ddl":   "CREATE INDEX IF NOT EXISTS idx_documents_tsv ON documents USING GIN (body_tsv)",
        "verify": (
            "SELECT id FROM documents WHERE body_tsv @@ plainto_tsquery('english', 'financial advice') LIMIT 1"
        ),
    },
    {
        "name":  "idx_documents_bm25",
        "type":  "bm25",
        "label": "BM25 (ParadeDB pg_search)",
        "ddl": (
            "CREATE INDEX IF NOT EXISTS idx_documents_bm25 ON documents "
            "USING bm25 (id, title, body) WITH (key_field = 'id')"
        ),
        "verify": (
            "SELECT id FROM documents WHERE body @@@ 'financial' LIMIT 1"
        ),
    },
    {
        "name":  "idx_documents_hnsw",
        "type":  "hnsw",
        "label": "HNSW (pgvector, m=16 ef_construction=200)",
        "ddl": (
            "CREATE INDEX IF NOT EXISTS idx_documents_hnsw ON documents "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200)"
        ),
        "verify": None,  # verified separately after SET hnsw.ef_search
    },
]


def index_size(conn, index_name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT pg_relation_size(%s)", (index_name,))
    row = cur.fetchone()
    return row[0] if row else 0


def table_size(conn) -> tuple[int, int]:
    """Return (table_bytes, total_with_indexes_bytes)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT pg_relation_size('documents'), pg_total_relation_size('documents')"
    )
    return cur.fetchone()


def row_count(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    return cur.fetchone()[0]


def verify_index(conn, label: str, verify_sql: str) -> None:
    cur = conn.cursor()
    try:
        cur.execute(f"EXPLAIN (FORMAT TEXT) {verify_sql}")
        plan = "\n".join(row[0] for row in cur.fetchall())
        if "Index" in plan or "Bitmap" in plan or "Parallel" in plan:
            print(f"    Index scan confirmed in plan ✓")
        else:
            print(f"    WARNING: sequential scan in plan — check index usage")
        conn.rollback()
    except Exception as e:
        print(f"    WARNING: verify query failed: {e}")
        conn.rollback()


def drop_index(conn, name: str) -> None:
    cur = conn.cursor()
    cur.execute(f"DROP INDEX IF EXISTS {name}")
    conn.commit()
    print(f"  Dropped {name}")


def store_result(conn, name: str, idx_type: str, build_s: float,
                 idx_size: int, tbl_size: int, nrows: int) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO index_build_results
            (index_name, index_type, build_time_s, index_size_bytes, table_size_bytes, row_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (name, idx_type, build_s, idx_size, tbl_size, nrows),
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Create benchmark indexes")
    parser.add_argument("--drop",  action="store_true", help="Drop existing indexes first")
    parser.add_argument("--skip",  nargs="*", default=[], choices=["gin", "hnsw", "bm25"],
                        help="Skip specific index types")
    parser.add_argument("--only",  choices=["gin", "hnsw", "bm25"],
                        help="Build only this index type")
    args = parser.parse_args()

    conn = get_connection()
    check_extensions(conn)

    nrows = row_count(conn)
    print(f"Table 'documents' has {nrows:,} rows\n")
    if nrows == 0:
        print("No rows found — run `make load` first")
        sys.exit(1)

    tbl_size, _ = table_size(conn)

    for idx in INDEXES:
        if idx["type"] in args.skip:
            print(f"Skipping {idx['label']}")
            continue
        if args.only and idx["type"] != args.only:
            continue

        print(f"Building {idx['label']} ...")

        if args.drop:
            drop_index(conn, idx["name"])

        # Use autocommit for CREATE INDEX so we see progress in psql \watch
        conn.rollback()   # close any open transaction before switching autocommit
        conn.autocommit = True
        t0 = time.perf_counter()
        cur = conn.cursor()

        try:
            cur.execute(idx["ddl"])
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.autocommit = False
            continue

        elapsed = time.perf_counter() - t0
        conn.autocommit = False

        idx_bytes = index_size(conn, idx["name"])
        store_result(conn, idx["name"], idx["type"], elapsed, idx_bytes, tbl_size, nrows)

        print(f"  Built in {elapsed:.1f}s  |  index size: {idx_bytes / 1e6:.1f} MB")

        if idx.get("verify"):
            verify_index(conn, idx["label"], idx["verify"])

        print()

    # Post-build: VACUUM ANALYZE for fresh stats
    print("Running VACUUM ANALYZE ...")
    conn.rollback()
    conn.autocommit = True
    conn.cursor().execute("VACUUM ANALYZE documents")
    conn.autocommit = False

    # Summary
    print("\n── Index Build Summary ──────────────────────────────────")
    cur = conn.cursor()
    cur.execute(
        "SELECT index_name, index_type, build_time_s, index_size_bytes "
        "FROM index_build_results ORDER BY created_at"
    )
    for row in cur.fetchall():
        name, itype, t, sz = row
        print(f"  {name:<30}  {itype:<5}  {t:>7.1f}s  {sz/1e6:>8.1f} MB")

    conn.close()


if __name__ == "__main__":
    main()
