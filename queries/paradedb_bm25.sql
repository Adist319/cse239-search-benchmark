-- ParadeDB BM25 Full-Text Search
-- Uses Tantivy-backed BM25 index via @@@  operator on the key_field (id).
-- paradedb.score(id) returns the BM25 relevance score for each matched row.
-- The index covers: id, title, body (created in create_indexes.py).
-- Parameters: query_text (str), k (int)

SELECT
    id,
    ext_id,
    paradedb.score(id) AS score
FROM
    documents
WHERE
    id @@@ %(query_text)s
ORDER BY
    paradedb.score(id) DESC
LIMIT %(k)s;
