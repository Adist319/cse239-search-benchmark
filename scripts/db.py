"""
Database connection helper.
Wraps psycopg2 with pgvector type registration and a simple context manager.
"""
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from config import DB_DSN


def get_connection(autocommit: bool = False) -> psycopg2.extensions.connection:
    """Return a psycopg2 connection with pgvector types registered."""
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = autocommit
    register_vector(conn)
    return conn


def dict_cursor(conn) -> psycopg2.extras.RealDictCursor:
    """Return a cursor that yields rows as dicts."""
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def check_extensions(conn) -> None:
    """Assert that pgvector and pg_search are installed."""
    cur = conn.cursor()
    cur.execute(
        "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
        (["vector", "pg_search"],)
    )
    found = {row[0] for row in cur.fetchall()}
    missing = {"vector", "pg_search"} - found
    if missing:
        raise RuntimeError(f"Missing Postgres extensions: {missing}. Did you run `make up`?")
    print("Extensions OK: vector, pg_search")


def table_row_count(conn, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT reltuples::BIGINT FROM pg_class WHERE relname = %s", (table,))
    row = cur.fetchone()
    return row[0] if row else 0
