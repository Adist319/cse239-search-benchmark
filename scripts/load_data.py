"""
Bulk-load corpus documents + embeddings into Postgres using COPY.

Steps:
  1. COPY (ext_id, title, body) into documents — body_tsv auto-generated
  2. UPDATE documents SET embedding = ... in batches (after numpy load)
  3. COPY queries + embeddings
  4. COPY qrels

Usage:
    python load_data.py [--dataset fiqa]
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, DATASET_NAME, LOAD_CHUNK
from db import get_connection, check_extensions


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def copy_documents(conn, rows: list[dict], chunk_size: int = LOAD_CHUNK) -> None:
    """
    COPY (ext_id, title, body) into documents table.
    Uses tab-delimited CSV via psycopg2.copy_expert — 10-50x faster than INSERT.
    The body_tsv GENERATED column auto-populates server-side.
    The embedding column is left NULL and filled in a separate pass.
    """
    cur = conn.cursor()
    total = len(rows)

    for start in range(0, total, chunk_size):
        chunk = rows[start : start + chunk_size]
        buf = io.StringIO()
        for row in chunk:
            # Escape for FORMAT text: backslash must be doubled, tabs/newlines replaced
            ext_id = row["_id"].replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            title  = (row.get("title") or "").replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            body   = (row.get("text") or "").replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            buf.write(f"{ext_id}\t{title}\t{body}\n")
        buf.seek(0)

        cur.copy_expert(
            "COPY documents (ext_id, title, body) FROM STDIN WITH (FORMAT text)",
            buf,
        )
        conn.commit()
        print(f"  Loaded {min(start + chunk_size, total):>8,} / {total:,} documents")


def update_embeddings(conn, doc_ids: list[str], embeddings: np.ndarray,
                      chunk_size: int = LOAD_CHUNK) -> None:
    """
    UPDATE documents SET embedding = ... WHERE ext_id = ...
    Uses execute_values for efficient batch updates.
    """
    cur = conn.cursor()
    total = len(doc_ids)

    for start in tqdm(range(0, total, chunk_size), desc="  Updating embeddings"):
        chunk_ids  = doc_ids[start : start + chunk_size]
        chunk_embs = embeddings[start : start + chunk_size]

        # Build list of (ext_id, vector_str) tuples
        data = [
            (ext_id, "[" + ",".join(f"{v:.6f}" for v in emb) + "]")
            for ext_id, emb in zip(chunk_ids, chunk_embs)
        ]

        psycopg2.extras.execute_values(
            cur,
            "UPDATE documents SET embedding = data.emb::vector FROM (VALUES %s) AS data(ext_id, emb) "
            "WHERE documents.ext_id = data.ext_id",
            data,
            template="(%s, %s)",
            page_size=1000,
        )
        conn.commit()


def copy_queries(conn, rows: list[dict], embeddings: np.ndarray,
                 query_ids: list[str]) -> None:
    """Load queries with their embeddings."""
    cur = conn.cursor()

    # Build a mapping from ext_id → embedding index
    id_to_idx = {qid: i for i, qid in enumerate(query_ids)}

    buf = io.StringIO()
    for row in rows:
        ext_id = row["_id"].replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        text   = row["text"].replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        idx    = id_to_idx.get(row["_id"])
        if idx is None:
            emb_str = r"\N"   # NULL
        else:
            emb = embeddings[idx]
            emb_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
        buf.write(f"{ext_id}\t{text}\t{emb_str}\n")
    buf.seek(0)

    cur.copy_expert(
        "COPY queries (ext_id, query_text, embedding) FROM STDIN "
        "WITH (FORMAT text, NULL '\\N')",
        buf,
    )
    conn.commit()
    print(f"  Loaded {len(rows):,} queries")


def copy_qrels(conn, qrels_path: Path) -> None:
    """Load relevance judgments from TSV file."""
    cur = conn.cursor()
    count = 0
    buf = io.StringIO()

    with qrels_path.open() as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            q_id, d_id, score = parts[0], parts[1], parts[2]
            buf.write(f"{q_id}\t{d_id}\t{score}\n")
            count += 1

    buf.seek(0)
    cur.copy_expert(
        "COPY qrels (query_ext_id, doc_ext_id, relevance) FROM STDIN "
        "WITH (FORMAT text)",
        buf,
    )
    conn.commit()
    print(f"  Loaded {count:,} qrels")


def main():
    parser = argparse.ArgumentParser(description="Bulk load dataset into Postgres")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--skip-documents",  action="store_true",
                        help="Skip document COPY (already loaded)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding UPDATE pass")
    parser.add_argument("--skip-queries",    action="store_true")
    parser.add_argument("--skip-qrels",      action="store_true")
    args = parser.parse_args()

    corpus_path  = DATA_DIR / f"{args.dataset}_corpus.jsonl"
    queries_path = DATA_DIR / f"{args.dataset}_queries.jsonl"
    qrels_path   = DATA_DIR / f"{args.dataset}_qrels.tsv"
    emb_path     = DATA_DIR / f"{args.dataset}_embeddings.npy"
    q_emb_path   = DATA_DIR / f"{args.dataset}_query_embeddings.npy"
    doc_ids_path = DATA_DIR / f"{args.dataset}_doc_ids.json"
    q_ids_path   = DATA_DIR / f"{args.dataset}_query_ids.json"

    conn = get_connection()
    check_extensions(conn)

    # 1. Documents
    if not args.skip_documents:
        print("\nLoading documents ...")
        corpus = load_jsonl(corpus_path)
        copy_documents(conn, corpus)

    # 2. Embeddings
    if not args.skip_embeddings:
        print("\nUpdating document embeddings ...")
        if not emb_path.exists():
            print(f"  {emb_path} not found — run `make embed` first")
            sys.exit(1)
        embeddings = np.load(emb_path)
        doc_ids    = json.loads(doc_ids_path.read_text())
        update_embeddings(conn, doc_ids, embeddings)

    # 3. Queries
    if not args.skip_queries:
        print("\nLoading queries ...")
        queries = load_jsonl(queries_path)
        q_emb   = np.load(q_emb_path) if q_emb_path.exists() else None
        q_ids   = json.loads(q_ids_path.read_text()) if q_ids_path.exists() else []
        copy_queries(conn, queries, q_emb if q_emb is not None else np.array([]), q_ids)

    # 4. Qrels
    if not args.skip_qrels and qrels_path.exists():
        print("\nLoading qrels ...")
        copy_qrels(conn, qrels_path)

    conn.close()
    print("\nLoad complete.")


if __name__ == "__main__":
    main()
