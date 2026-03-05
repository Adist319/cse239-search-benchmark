# PostgreSQL Retrieval Benchmark

Comparing four text retrieval methods inside a single PostgreSQL instance, evaluated on two BEIR datasets.

## Overview

This project measures how far retrieval quality can be pushed without leaving PostgreSQL. No Elasticsearch, no Solr, no separate vector database -- just one PostgreSQL 16 container with pg_search (ParadeDB's Tantivy-backed BM25) and pgvector (HNSW). The benchmark covers four methods: vanilla full-text search, BM25, dense vector retrieval, and hybrid RRF. Each method is evaluated on latency and retrieval quality (NDCG, MAP, Recall, Precision) across two datasets with different characteristics: FiQA for semantic queries, NFCorpus for keyword-heavy biomedical IR.

| Method | Engine | Index Type | Ranking |
|--------|--------|------------|---------|
| Vanilla FTS | PostgreSQL `tsvector` | GIN | `ts_rank_cd` (cover density) |
| BM25 | ParadeDB `pg_search` (Tantivy) | BM25 inverted index | BM25 scoring |
| Dense (pgvector) | `pgvector` HNSW | HNSW graph | Cosine similarity |
| Hybrid RRF | BM25 + pgvector | Both | Reciprocal Rank Fusion |

## Key Results

On FiQA-2018, queries are natural-language financial questions with significant vocabulary mismatch against their answers. pgvector achieves NDCG@10 = 0.367 while BM25 scores 0.233. Hybrid RRF scores 0.360, slightly below pgvector alone, because BM25 recall on these queries is low enough that including its candidates degrades the final ranking.

On NFCorpus, queries are short biomedical keyword phrases. BM25 recall is competitive with pgvector recall, and hybrid RRF wins at NDCG@10 = 0.336 vs. pgvector's 0.314. When both retrieval legs are individually strong, RRF improves coverage by combining their candidate sets.

The pattern across both datasets: hybrid search improves results when BM25 recall is close to pgvector recall. When BM25 lags significantly, fusion adds irrelevant candidates and hurts quality.

## Tech Stack

- **Database:** PostgreSQL 16 (ParadeDB base image)
- **Extensions:** pg_search 0.21.8 (Tantivy BM25), pgvector 0.8 (HNSW)
- **Embeddings:** all-MiniLM-L6-v2 (384-dim, Sentence Transformers)
- **Client:** Python 3.12, psycopg2
- **Infrastructure:** Docker (single container)
- **Evaluation:** BEIR framework (NDCG, MAP, Recall, Precision)

## Prerequisites

- Docker Desktop
- Python 3.12+
- ~2 GB disk space (embeddings + Docker image)

## Setup

```bash
pip install -r requirements.txt
make up
```

## Running the Benchmark

Full pipeline (after `make up`):

```bash
make all    # download -> embed -> load -> index -> bench -> relevance -> export
```

Step by step:

```bash
make download              # Download dataset from HuggingFace
make embed                 # Generate sentence embeddings (~10 min on CPU)
make load                  # Bulk load into PostgreSQL
make index                 # Build GIN, BM25, HNSW indexes
make bench                 # Run benchmarks (5 runs per query, 10 warmup)
make relevance             # Compute NDCG, MAP, Recall, Precision
make export                # Export results to results/*.csv
```

To run on a different dataset:

```bash
make reset-data
make all DATASET=nfcorpus
```

Development shortcuts:

```bash
make bench-dev             # 50 queries, 3 runs (fast iteration)
make bench-explain         # Collect EXPLAIN ANALYZE query plans
make show-results          # Print latency summary
make show-indexes          # Print index build times
make psql                  # Open psql shell in container
```

Override parameters:

```bash
make bench DATASET=fiqa N_RUNS=10 K=20
make download DATASET=fiqa LIMIT=5000
```

## Project Structure

```
.
├── Makefile                  Pipeline automation
├── requirements.txt          Python dependencies
├── docker/
│   ├── docker-compose.yml    Single-container PostgreSQL setup
│   └── init.sql              Schema: documents, queries, qrels, benchmark tables
├── scripts/
│   ├── config.py             Central configuration (DB, dataset, HNSW params)
│   ├── download_data.py      Download BEIR datasets from HuggingFace
│   ├── generate_embeddings.py  Encode documents/queries with all-MiniLM-L6-v2
│   ├── load_data.py          Bulk COPY into PostgreSQL
│   ├── create_indexes.py     Build and time GIN, BM25, HNSW indexes
│   ├── run_benchmarks.py     Execute queries, measure latency, store results
│   ├── compute_relevance.py  Compute NDCG, MAP, Recall, Precision vs. qrels
│   ├── analyze_plans.py      Parse EXPLAIN ANALYZE JSON plans
│   ├── hnsw_recall.py        Verify HNSW recall vs. brute-force
│   ├── export_results.py     Export benchmark_results to CSV
│   └── db.py                 Database connection helper
├── queries/
│   ├── vanilla_fts.sql       GIN tsvector query template
│   ├── paradedb_bm25.sql     ParadeDB BM25 query template
│   ├── pgvector_search.sql   HNSW cosine similarity query template
│   └── hybrid_rrf.sql        RRF fusion (BM25 + pgvector) query template
├── analysis/
│   └── benchmark_analysis.ipynb  Latency CDFs, NDCG bar charts, Pareto frontier
├── findings/
│   ├── report.tex            Full academic report (LaTeX)
│   └── results.md            Structured results per dataset
└── results/                  Generated CSV exports (git-ignored)
```

## Results

Full results, query plan analysis, and discussion are in [findings/report.tex](findings/report.tex).

### FiQA-2018 (semantic queries, 648 evaluated queries)

| Method | NDCG@10 | p50 Latency |
|--------|---------|-------------|
| vanilla_fts | 0.047 | 0.83 ms |
| paradedb_bm25 | 0.233 | 5.05 ms |
| hybrid_rrf | 0.360 | 24.61 ms |
| **pgvector** | **0.367** | 1.87 ms |

### NFCorpus (keyword queries, 323 evaluated queries)

| Method | NDCG@10 | p50 Latency |
|--------|---------|-------------|
| vanilla_fts | 0.207 | 0.30 ms |
| paradedb_bm25 | 0.299 | 55.79 ms |
| pgvector | 0.314 | 0.72 ms |
| **hybrid_rrf** | **0.336** | 68.10 ms |

## Course Context

CSE 239A -- Graduate Seminar: Storage and Retrieval Systems
UC San Diego, Winter 2026
