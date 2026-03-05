##############################################################################
# CSE 239A Search Benchmark — Makefile
#
# Typical end-to-end workflow:
#   make up       — start Postgres (ParadeDB) in Docker
#   make download — download FiQA-2018 dataset from HuggingFace
#   make embed    — generate sentence embeddings (slow on CPU, ~10 min)
#   make load     — bulk load data into Postgres
#   make index    — build GIN, BM25, HNSW indexes (records build times)
#   make bench    — run all benchmarks, store results in DB
#   make export   — export benchmark_results.csv + index_build_results.csv
#   make down     — stop and remove container (keeps volume)
#
# One-shot (after `make up` and `make download`):
#   make all
##############################################################################

DATASET     ?= fiqa
LIMIT       ?= 0           # 0 = full dataset; set e.g. LIMIT=5000 for dev
N_RUNS      ?= 5
N_WARMUP    ?= 10
K           ?= 10
METHODS     ?= vanilla_fts pgvector paradedb_bm25 hybrid_rrf
EXPLAIN     ?=             # set to --explain to collect EXPLAIN ANALYZE plans

PYTHON      := ../.venv/bin/python3
SCRIPTS     := scripts
DOCKER_DIR  := docker

.PHONY: help up down reset download embed load index bench export all clean

help:
	@echo ""
	@echo "  make up        Start ParadeDB container"
	@echo "  make down      Stop container (keeps DB volume)"
	@echo "  make reset     Stop + delete DB volume (destructive)"
	@echo "  make download  Download dataset (DATASET=$(DATASET))"
	@echo "  make embed     Generate embeddings"
	@echo "  make load      Bulk load data into Postgres"
	@echo "  make index     Create and time indexes"
	@echo "  make bench     Run benchmarks"
	@echo "  make export    Export results to CSV"
	@echo "  make all       download + embed + load + index + bench + export"
	@echo ""
	@echo "  Options (override on command line):"
	@echo "    DATASET=$(DATASET)  LIMIT=$(LIMIT)  N_RUNS=$(N_RUNS)"
	@echo "    METHODS='$(METHODS)'"
	@echo "    EXPLAIN=--explain  (collect EXPLAIN ANALYZE plans)"
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────

up:
	docker compose -f $(DOCKER_DIR)/docker-compose.yml up -d
	@echo "Waiting for Postgres to be ready..."
	@until docker exec cse239-bench pg_isready -U bench -d search_bench -q; do sleep 1; done
	@echo "Postgres is ready ✓"

down:
	docker compose -f $(DOCKER_DIR)/docker-compose.yml stop

reset:
	docker compose -f $(DOCKER_DIR)/docker-compose.yml down -v
	@echo "Container and volume removed."

# Wipe corpus/queries/qrels so a new dataset can be loaded without resetting
# the container. Benchmark results are preserved (filtered by run_config->dataset).
reset-data:
	docker exec cse239-bench psql -U $(shell docker exec cse239-bench printenv POSTGRES_USER) \
		-d $(shell docker exec cse239-bench printenv POSTGRES_DB) \
		-c "TRUNCATE documents, queries, qrels RESTART IDENTITY CASCADE;"
	@echo "Corpus tables cleared — ready to load a new dataset."

logs:
	docker compose -f $(DOCKER_DIR)/docker-compose.yml logs -f db

psql:
	docker exec -it cse239-bench psql -U bench -d search_bench

# ── Data pipeline ─────────────────────────────────────────────────────────────

download:
	cd $(SCRIPTS) && $(PYTHON) download_data.py --dataset $(DATASET) --limit $(LIMIT)

embed:
	cd $(SCRIPTS) && $(PYTHON) generate_embeddings.py --dataset $(DATASET)

load:
	cd $(SCRIPTS) && $(PYTHON) load_data.py --dataset $(DATASET)

# ── Indexes ───────────────────────────────────────────────────────────────────

index:
	cd $(SCRIPTS) && $(PYTHON) create_indexes.py

index-gin:
	cd $(SCRIPTS) && $(PYTHON) create_indexes.py --only gin

index-bm25:
	cd $(SCRIPTS) && $(PYTHON) create_indexes.py --only bm25

index-hnsw:
	cd $(SCRIPTS) && $(PYTHON) create_indexes.py --only hnsw

# ── Benchmark ─────────────────────────────────────────────────────────────────

bench:
	cd $(SCRIPTS) && $(PYTHON) run_benchmarks.py \
		--dataset $(DATASET) \
		--methods $(METHODS) \
		--n-runs $(N_RUNS) \
		--n-warmup $(N_WARMUP) \
		--k $(K) \
		$(EXPLAIN)

# Quick dev run: limit to 50 queries, 3 runs each
bench-dev:
	cd $(SCRIPTS) && $(PYTHON) run_benchmarks.py \
		--dataset $(DATASET) \
		--methods $(METHODS) \
		--n-runs 3 \
		--n-warmup 5 \
		--k 10 \
		--limit 50

# Benchmark with EXPLAIN ANALYZE plans collected
bench-explain:
	cd $(SCRIPTS) && $(PYTHON) run_benchmarks.py \
		--dataset $(DATASET) \
		--methods $(METHODS) \
		--n-runs $(N_RUNS) \
		--n-warmup $(N_WARMUP) \
		--k $(K) \
		--explain

# ── Relevance metrics ─────────────────────────────────────────────────────────

relevance:
	cd $(SCRIPTS) && $(PYTHON) compute_relevance.py --k $(K) --dataset $(DATASET) --csv

# ── Export ────────────────────────────────────────────────────────────────────

export:
	cd $(SCRIPTS) && $(PYTHON) export_results.py
	@echo ""
	@ls -lh results/*.csv 2>/dev/null || echo "No CSV files found in results/"

# ── Convenience ───────────────────────────────────────────────────────────────

all: download embed load index bench relevance export

# Install Python deps (into current env / venv)
install:
	pip install -r requirements.txt

# Show index build times from DB
show-indexes:
	docker exec -it cse239-bench psql -U bench -d search_bench \
		-c "SELECT index_name, index_type, round(build_time_s::numeric,1) AS build_s, \
		           round(index_size_bytes/1e6::numeric,1) AS mb, row_count \
		    FROM index_build_results ORDER BY created_at;"

# Show benchmark summary from DB
show-results:
	docker exec -it cse239-bench psql -U bench -d search_bench \
		-c "SELECT method, \
		           round(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_p50_ms)::numeric, 2) AS med_p50_ms, \
		           round(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_p95_ms)::numeric, 2) AS med_p95_ms, \
		           round(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_p99_ms)::numeric, 2) AS med_p99_ms, \
		           COUNT(*) AS n_queries \
		    FROM benchmark_results \
		    GROUP BY method \
		    ORDER BY med_p50_ms;"

clean:
	rm -f data/*.jsonl data/*.npy data/*.json data/*.tsv
	rm -f results/*.csv
	@echo "Cleaned data/ and results/ (Docker volume untouched)"
