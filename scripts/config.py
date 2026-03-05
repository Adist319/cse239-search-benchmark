"""
Central configuration for the CSE 239A benchmark.
All scripts import from here — change defaults in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Database ─────────────────────────────────────────────────
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB",   "search_bench")
DB_USER = os.getenv("POSTGRES_USER", "bench")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "benchpass")

DB_DSN = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASS}"

# ── Dataset ──────────────────────────────────────────────────
# Primary: FiQA-2018 (57K docs, 648 queries) — ideal class-project size
# Change DATASET_NAME to swap to another BEIR dataset (e.g. "trec-covid")
DATASET_NAME   = os.getenv("DATASET_NAME", "fiqa")
DATASET_LIMIT  = int(os.getenv("DATASET_LIMIT", "0"))  # 0 = no limit

# ── Embeddings ───────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, well-benchmarked
EMBEDDING_DIM   = 384
EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "64"))  # 64 CPU / 256 GPU
EMBEDDINGS_FILE = DATA_DIR / f"{DATASET_NAME}_embeddings.npy"
QUERY_EMBEDDINGS_FILE = DATA_DIR / f"{DATASET_NAME}_query_embeddings.npy"

# ── Benchmark defaults ────────────────────────────────────────
N_WARMUP   = int(os.getenv("N_WARMUP", "10"))    # warmup queries per block
N_RUNS     = int(os.getenv("N_RUNS",   "5"))     # timed repetitions per query
K          = int(os.getenv("K",        "10"))    # result set size (top-k)
RRF_POOL   = int(os.getenv("RRF_POOL", "20"))    # candidates per leg in RRF
RRF_K      = int(os.getenv("RRF_K",   "60"))     # RRF constant (Cormack 2009)
EF_SEARCH  = int(os.getenv("EF_SEARCH","100"))   # HNSW ef_search (default 40)

LOAD_CHUNK = 50_000   # rows per COPY batch during bulk load
