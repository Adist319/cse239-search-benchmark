-- Vanilla Postgres Full-Text Search
-- Uses pre-computed, stored tsvector column (body_tsv) with GIN index.
-- ts_rank_cd applies cover density normalization for multi-word queries.
-- Parameters: query_text (str), k (int)

SELECT
    d.id,
    d.ext_id,
    ts_rank_cd(d.body_tsv, query) AS score
FROM
    documents d,
    plainto_tsquery('english', %(query_text)s) AS query
WHERE
    d.body_tsv @@ query
ORDER BY
    score DESC
LIMIT %(k)s;
