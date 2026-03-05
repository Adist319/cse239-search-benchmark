-- pgvector Semantic Search (HNSW index, cosine distance)
-- Embeddings are normalized (L2=1), so cosine similarity = 1 - cosine_distance.
-- ef_search is set at the session level before running this query.
-- Parameters: query_embedding (vector str), k (int)

SELECT
    id,
    ext_id,
    1 - (embedding <=> %(query_embedding)s::vector) AS score
FROM
    documents
ORDER BY
    embedding <=> %(query_embedding)s::vector
LIMIT %(k)s;
