-- ============================================================
-- CSE 239A Search Benchmark — DB initialization
-- Runs once on first container start via docker-entrypoint
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_search;   -- ParadeDB BM25
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ============================================================
-- Core document table (single table for all three approaches)
-- ============================================================
CREATE TABLE documents (
    id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ext_id    TEXT UNIQUE NOT NULL,          -- BEIR document _id
    title     TEXT DEFAULT '',
    body      TEXT,

    -- Vanilla FTS: generated, stored tsvector (auto-maintained on INSERT/UPDATE)
    body_tsv  TSVECTOR GENERATED ALWAYS AS (
                  setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(body, '')), 'B')
              ) STORED,

    -- pgvector + hybrid: embeddings filled in a separate pass after bulk load
    embedding VECTOR(384),                  -- all-MiniLM-L6-v2, 384 dims

    metadata  JSONB DEFAULT '{}'
);

-- ============================================================
-- Query table (BEIR queries with ground-truth qrels)
-- ============================================================
CREATE TABLE queries (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ext_id     TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    embedding  VECTOR(384)                  -- pre-computed query embedding
);

CREATE TABLE qrels (
    query_ext_id TEXT NOT NULL,
    doc_ext_id   TEXT NOT NULL,
    relevance    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (query_ext_id, doc_ext_id)
);

-- ============================================================
-- Benchmark output tables
-- ============================================================

-- Per-query aggregated latency results
CREATE TABLE benchmark_results (
    id              SERIAL PRIMARY KEY,
    method          TEXT NOT NULL,           -- vanilla_fts | pgvector | paradedb_bm25 | hybrid_rrf
    query_ext_id    TEXT NOT NULL,
    query_text      TEXT,
    latency_p50_ms  FLOAT NOT NULL,
    latency_p95_ms  FLOAT NOT NULL,
    latency_p99_ms  FLOAT NOT NULL,
    latency_mean_ms FLOAT NOT NULL,
    n_runs          INTEGER NOT NULL,
    num_results     INTEGER,
    result_ids      TEXT[],                  -- top-k doc ext_ids for relevance scoring
    plan_json       JSONB,                   -- from optional EXPLAIN ANALYZE pass
    run_config      JSONB,                   -- k, ef_search, rrf_k, etc.
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON benchmark_results (method);
CREATE INDEX ON benchmark_results (query_ext_id);

-- Index build metrics
CREATE TABLE index_build_results (
    id               SERIAL PRIMARY KEY,
    index_name       TEXT NOT NULL,
    index_type       TEXT NOT NULL,          -- gin | hnsw | bm25
    build_time_s     FLOAT NOT NULL,
    index_size_bytes BIGINT,
    table_size_bytes BIGINT,
    row_count        BIGINT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
