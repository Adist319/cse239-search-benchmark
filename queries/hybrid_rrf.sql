-- Hybrid Search: BM25 + pgvector via Reciprocal Rank Fusion (RRF)
--
-- RRF score = sum_over_rankers( 1 / (k + rank) )   where k=60 (Cormack 2009)
-- This is scale-independent: we combine ranked positions, not raw scores.
--
-- Parameters:
--   query_text      (str)   — text query for BM25 leg
--   rrf_pool        (int)   — candidates per leg (e.g. 20)
--   query_embedding (vec)   — pre-computed query embedding (×3 uses)
--   rrf_k           (int)   — RRF constant (typically 60)
--   k               (int)   — final result set size

WITH bm25_results AS (
    SELECT
        id,
        ext_id,
        paradedb.score(id)                                       AS bm25_score,
        ROW_NUMBER() OVER (ORDER BY paradedb.score(id) DESC)    AS rank
    FROM documents
    WHERE id @@@ %(query_text)s
    LIMIT %(rrf_pool)s
),
vector_results AS (
    SELECT
        id,
        ext_id,
        embedding <=> %(query_embedding)s::vector               AS distance,
        ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_embedding)s::vector) AS rank
    FROM documents
    ORDER BY embedding <=> %(query_embedding)s::vector
    LIMIT %(rrf_pool)s
),
rrf AS (
    SELECT id, ext_id, 1.0 / (%(rrf_k)s + rank) AS score FROM bm25_results
    UNION ALL
    SELECT id, ext_id, 1.0 / (%(rrf_k)s + rank) AS score FROM vector_results
)
SELECT
    r.id,
    r.ext_id,
    SUM(r.score) AS rrf_score
FROM rrf r
GROUP BY r.id, r.ext_id
ORDER BY rrf_score DESC
LIMIT %(k)s;
