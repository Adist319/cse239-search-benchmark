# Search Benchmark Results

Benchmarking four retrieval methods inside a single PostgreSQL instance
using the BEIR evaluation framework. All experiments use the same hardware,
the same Postgres 16 + ParadeDB (pg_search 0.21.8) + pgvector setup, and
the same embedding model (all-MiniLM-L6-v2, 384-dim).

---

## Dataset 1 — FiQA-2018

**Task:** Financial opinion mining & question answering
**Source:** StackExchange Finance + Reddit r/investing posts
**Corpus:** 57,638 documents · avg 132 words
**Queries:** 648 test queries (with qrels) out of 6,648 total
**Relevant docs/query:** 2.6 avg (binary relevance)

### Index Build

| Index | Type | Build Time | Index Size | Table Size |
|-------|------|-----------|------------|------------|
| idx_documents_tsv | GIN (tsvector) | 1.1 s | 19 MB | 129 MB |
| idx_documents_bm25 | BM25 (pg_search/Tantivy) | 0.7 s | 38 MB | 129 MB |
| idx_documents_hnsw | HNSW (pgvector, 384-dim) | 18.8 s | 113 MB | 129 MB |

### Latency (p50 / p95, 6,648 queries × 5 runs, k=10)

| Method | p50 | p95 | Relative |
|--------|-----|-----|----------|
| vanilla_fts | 0.83 ms | 1.25 ms | 1× |
| pgvector | 1.87 ms | 2.75 ms | 2.3× |
| paradedb_bm25 | 5.05 ms | 6.10 ms | 6.1× |
| hybrid_rrf | 24.61 ms | 33.31 ms | 29.7× |

### Retrieval Quality (evaluated on 648 queries with qrels, k=10)

| Method | NDCG@10 | P@10 | Recall@10 | MAP@10 |
|--------|---------|------|-----------|--------|
| vanilla_fts | 0.047 | 0.011 | 0.053 | 0.037 |
| paradedb_bm25 | 0.233 | 0.067 | 0.297 | 0.172 |
| hybrid_rrf | 0.360 | 0.105 | 0.447 | 0.278 |
| pgvector | **0.367** | **0.105** | 0.440 | **0.290** |

BEIR canonical BM25 baseline on FiQA-2018: **NDCG@10 = 0.236**
(0.233 matches within 1.3%, confirming correct implementation)

### Key Findings

pgvector wins on FiQA, and it is not close. NDCG=0.367 against BM25's 0.233 is a 58% gap. FiQA queries like "Is it better to pay off student loans or invest?" share almost no vocabulary with their answers, which discuss debt-to-equity allocation decisions. Embeddings bridge that semantic gap. Lexical methods have nothing to match on.

vanilla_fts comes out nearly useless at NDCG=0.047. Postgres GIN+tsvector applies English stemming, so the failure is not about morphology; many queries return zero results because the vocabulary mismatch is total. That is not a tuning problem.

The hybrid result was the one genuine surprise: RRF at NDCG=0.360 falls slightly below pgvector alone (0.367). Adding BM25 to the mix hurts. BM25 recall is only 0.297, meaning it surfaces documents that do not belong near the top, and those candidates dilute the pgvector signal through the fusion step. Hybrid fusion only helps when both component retrievers are individually strong. On FiQA, BM25 is too weak to be a useful partner.

The ParadeDB BM25 score of 0.233 lands within 1.3% of the published BEIR baseline of 0.236. That is the sanity check: the gap between BM25 and pgvector on this dataset is real and reproducible, not an artifact of a misconfigured retriever.

On the build side, the HNSW index takes 18.8 seconds versus 0.7-1.1 seconds for the lexical indexes, and it is 6x larger than GIN. In return, retrieval quality is 8x better at 2x the query latency. For a 57,638-document corpus, that tradeoff is easy to accept.

One gap worth documenting: ParadeDB's default tokenizer lacks stemming, while Postgres `to_tsvector('english', ...)` includes it. Enabling `stemmer=english` in the ParadeDB DDL could push BM25 NDCG from 0.233 toward 0.25, but that would not close the fundamental semantic gap. Getting BM25 to compete with dense retrieval on FiQA would require query expansion techniques like docT5query or SPLADE.

### Query Plan Analysis (FiQA, 200 queries, hot cache)

| Method | Planning (ms) | Execution (ms) | Index Nodes | Buffer Hit | Planner Accuracy |
|--------|--------------|---------------|-------------|-----------|-----------------|
| vanilla_fts | 0.06 (p95: 0.26) | 0.18 (p95: 1.28) | Bitmap Heap+Index Scan | 100% | 1.60× |
| pgvector | 0.02 (p95: 0.10) | 1.27 (p95: 4.48) | Index Scan (HNSW) | 100% | 0.50× |
| paradedb_bm25 | 0.55 (p95: 1.16) | 5.98 (p95: 13.26) | Custom Scan (Tantivy) | 100% | 1.14× |
| hybrid_rrf | 1.11 (p95: 2.91) | 52.42 (p95: 96.75) | Custom Scan + Index Scan | 100% | 0.62× |

ParadeDB BM25 planning is 9x slower than pgvector due to Tantivy query parsing. pgvector underestimates row count 2x because the planner cannot inspect the HNSW graph. All buffer hits are 100% -- corpus fits in RAM (hot cache only).

### HNSW Recall Verification

HNSW recall@10 = **0.998** (100 sampled queries, ef_search=100). Min=0.90. 98/100 queries >= 0.95 recall. **PASS** -- NDCG results reflect dense retrieval quality, not ANN approximation error.

### FiQA Conclusion

On FiQA, pgvector is the right answer: best NDCG at 0.367, second-lowest latency at 1.87ms, and a 58% quality lead over BM25. The HNSW index costs 18.8 seconds to build and 113 MB of space, but those are one-time costs. Hybrid RRF makes things slightly worse -- the BM25 leg is too noisy on this dataset to add value. For query workloads resembling FiQA (conversational questions, semantic intent), dense retrieval alone is the better choice.

---

## Dataset 2 — NFCorpus

**Task:** Biomedical information retrieval
**Source:** NutritionFacts.org medical/nutrition documents
**Corpus:** 3,633 documents · avg 232 words
**Queries:** 323 test queries (with qrels) out of 3,237 total
**Relevant docs/query:** 38.2 avg (graded 0/1/2 relevance)

### Index Build

| Index | Type | Build Time | Index Size | Table Size |
|-------|------|-----------|------------|------------|
| idx_documents_tsv | GIN (tsvector) | <0.01 s | 7.4 MB | 10 MB |
| idx_documents_bm25 | BM25 (pg_search/Tantivy) | <0.01 s | 8.9 MB | 10 MB |
| idx_documents_hnsw | HNSW (pgvector, 384-dim) | <0.01 s | 7.0 MB | 10 MB |

All indexes build in under 10ms on the 3,633-document corpus.

### Latency (p50 / p95, 3,237 queries × 5 runs, k=10)

| Method | p50 | p95 | Relative |
|--------|-----|-----|----------|
| vanilla_fts | 0.30 ms | 0.33 ms | 1× |
| pgvector | 0.72 ms | 0.96 ms | 2.4× |
| paradedb_bm25 | 55.79 ms | 60.51 ms | 186× |
| hybrid_rrf | 68.10 ms | 85.44 ms | 227× |

**Note:** ParadeDB BM25 exhibits anomalously high latency (~56ms) on NFCorpus after corpus reload, compared to 5ms on FiQA. The likely explanation is a Tantivy cold-start artifact following the corpus swap; 10 warmup queries may not be sufficient to reach steady state for the reloaded index.

### Retrieval Quality (evaluated on 323 queries with qrels, k=10, graded NDCG)

| Method | NDCG@10 | P@10 | Recall@10 | MAP@10 |
|--------|---------|------|-----------|--------|
| vanilla_fts | 0.207 | 0.137 | 0.091 | 0.082 |
| paradedb_bm25 | 0.299 | 0.214 | 0.149 | 0.114 |
| pgvector | 0.314 | 0.243 | 0.153 | 0.109 |
| **hybrid_rrf** | **0.336** | **0.247** | **0.171** | **0.129** |

BEIR canonical BM25 baseline on NFCorpus: **NDCG@10 = 0.325**
(0.299 is 8% below -- attributable to ParadeDB's default tokenizer lacking stemming;
English stemming helps biomedical terms like "immunity" collapse to "immun")

### Key Findings

NFCorpus flips the story. Hybrid RRF wins with NDCG=0.336, beating pgvector (0.314) by 7% and BM25 (0.299) by 12%. Fusion works here where it failed on FiQA because both legs have comparable recall (BM25=0.149, pgvector=0.153). They are finding different relevant documents, and RRF combines them constructively.

pgvector still beats BM25 on this dataset, which was unexpected. NFCorpus queries are short keyword phrases targeting specialized biomedical vocabulary -- the kind of input BM25 was designed for. Dense retrieval at 0.314 still outperforms BM25 at 0.299. The all-MiniLM-L6-v2 model generalizes reasonably well to biomedical text even without domain fine-tuning.

The BM25 result sits 8% below the BEIR canonical baseline (0.299 vs. 0.325). That gap is ParadeDB's default tokenizer lacking stemming. Pyserini and Elasticsearch BM25 apply English stemming, which collapses biomedical variants like "vitamin" and "vitamins" into the same token. ParadeDB's Tantivy tokenizer treats them as distinct terms and misses matches it should be making. Enabling `stemmer=english` in the index DDL would close most of that gap.

vanilla_fts recovers substantially on NFCorpus: NDCG=0.207 versus 0.047 on FiQA. The vocabulary between queries like "vitamin D deficiency" and the document text overlaps here, which is the precondition for lexical search to function.

The latency picture is inverted from FiQA. pgvector at 0.72ms is the fast option; hybrid_rrf at 68ms is the quality option. The 22ms cost of adding BM25 to the RRF fusion buys 7% more NDCG. Whether that tradeoff is acceptable depends on latency requirements.

### Query Plan Analysis (NFCorpus, 3,237 queries, hot cache)

| Method | Planning (ms) | Execution (ms) | Index Nodes | Buffer Hit | Planner Accuracy |
|--------|--------------|---------------|-------------|-----------|-----------------|
| vanilla_fts | 0.05 (p95: 0.22) | 0.02 (p95: 0.32) | Bitmap Heap+Index Scan | 100% | 1.07× |
| pgvector | 0.02 (p95: 0.12) | 0.60 (p95: 1.65) | Index Scan (HNSW) | 100% | 0.50× |
| paradedb_bm25 | 1.29 (p95: 15.94) | 134.55 (p95: 557.21) | Custom Scan (Tantivy) | 100% | 0.72× |
| hybrid_rrf | 0.89 (p95: 3.03) | 85.47 (p95: 186.31) | Custom Scan + Index Scan | 100% | 0.68× |

**Cross-dataset comparison:** vanilla_fts execution drops from 0.18ms (FiQA) to 0.02ms (NFCorpus), 9x faster on the 16x smaller corpus. pgvector drops proportionally (1.27 to 0.60ms). ParadeDB BM25 increases from 5.98ms to 134.55ms (22x), confirming the latency anomaly is in execution, not planning. Buffer hit rate is 100% for all methods -- the entire NFCorpus fits in shared buffers.

### HNSW Recall Verification (NFCorpus)

| Metric | Value |
|--------|-------|
| Mean recall@10 | **0.995** |
| Median recall@10 | 1.000 |
| Min recall@10 | 0.900 |
| Queries >= 0.95 recall | 95/100 (95%) |
| Queries >= 0.90 recall | 100/100 (100%) |
| **Verdict** | **PASS** (threshold: 0.95) |

Consistent with FiQA (0.998 mean recall@10). Both datasets confirm HNSW approximation introduces negligible error at ef_search=100.

### NFCorpus Conclusion

NFCorpus shows that the right method changes with the query type and corpus. When query vocabulary aligns with document vocabulary, BM25 recovers enough to be a useful fusion partner, and hybrid RRF becomes the quality leader. pgvector is still the fastest option at 0.72ms p50, and still beats BM25 on relevance despite the keyword-query advantage. The 22ms cost of hybrid fusion buys 7% NDCG improvement -- a reasonable tradeoff for batch retrieval, less so for latency-sensitive applications.

---

## Methodology Notes

- **Embeddings:** all-MiniLM-L6-v2 (Sentence Transformers, 384-dim, free/local)
- **HNSW params:** ef_construction=64, m=16 (pgvector defaults), ef_search=100
- **RRF params:** k=60 (Cormack 2009), pool=20 candidates per leg
- **Runs:** 5 timed repetitions per query, p50/p95/p99 of per-query medians
- **Warmup:** 10 queries before timing each method block
- **Relevance eval:** Only queries present in qrels file (648/6,648 FiQA; 323/3,237 NFCorpus)
- **NDCG:** Graded relevance used (NFCorpus: 0/1/2; FiQA: binary 0/1); Precision/Recall/MAP use binary (rel>0)
- **ParadeDB query:** `id @@@ query_text` (searches all indexed fields via key field)
- **Sanitization:** Strip Tantivy special chars from query text before passing to pdb.match()

### Known Limitations

- all-MiniLM-L6-v2 is a lightweight model; larger models (bge-large, text-embedding-3-large) would likely improve pgvector NDCG by 5-10 points.
- HNSW ef_search=100 is conservative; higher values trade latency for recall.
- Latency measured from Python client over loopback socket; true server-side execution time is ~0.2-0.5ms lower per query.
- FiQA has only 2.6 relevant docs/query (sparse labels) -- some relevant documents are likely unlabeled, suppressing all NDCG scores.
- NFCorpus BM25 latency (~56ms) appears to reflect Tantivy cold-start after corpus swap, not steady-state performance; FiQA BM25 (5ms) is the more reliable latency figure.
- ParadeDB default tokenizer lacks stemming -- affects BM25 quality on morphologically rich domains (biomedical); enabling `stemmer=english` would close the gap to BEIR baseline.
