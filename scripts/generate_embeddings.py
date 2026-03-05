"""
Generate embeddings for corpus documents and queries using sentence-transformers.

Reads from:  data/{dataset}_corpus.jsonl, data/{dataset}_queries.jsonl
Writes to:   data/{dataset}_embeddings.npy      (shape: [N_docs, 384])
             data/{dataset}_query_embeddings.npy (shape: [N_queries, 384])
             data/{dataset}_doc_ids.json         (list of ext_ids in row order)
             data/{dataset}_query_ids.json       (list of ext_ids in row order)

Usage:
    python generate_embeddings.py [--dataset fiqa] [--batch 64]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, DATASET_NAME, EMBEDDING_MODEL, EMBEDDING_BATCH, EMBEDDING_DIM


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """
    Encode texts in batches, sorting by length to minimize padding overhead.
    Returns embeddings in the original input order.
    """
    # Sort by length (shortest first) to reduce padding waste
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    embeddings_sorted = model.encode(
        sorted_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine via inner product
        convert_to_numpy=True,
    )

    # Restore original order
    restore = np.argsort(order)
    return embeddings_sorted[restore].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate sentence embeddings")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--batch",   type=int, default=EMBEDDING_BATCH,
                        help="Batch size (64 CPU / 256 GPU)")
    parser.add_argument("--model",   default=EMBEDDING_MODEL)
    args = parser.parse_args()

    corpus_path  = DATA_DIR / f"{args.dataset}_corpus.jsonl"
    queries_path = DATA_DIR / f"{args.dataset}_queries.jsonl"

    for p in (corpus_path, queries_path):
        if not p.exists():
            print(f"Missing {p} — run `make download` first")
            sys.exit(1)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    assert model.get_sentence_embedding_dimension() == EMBEDDING_DIM, (
        f"Model dim mismatch: expected {EMBEDDING_DIM}, "
        f"got {model.get_sentence_embedding_dimension()}"
    )

    # ── Corpus embeddings ────────────────────────────────────
    print("\nEncoding corpus ...")
    corpus = load_jsonl(corpus_path)
    doc_ids = [row["_id"] for row in corpus]
    # Combine title + body for embedding (mirrors how tsvector weights them)
    doc_texts = [
        (row["title"] + " " + row["text"]).strip() if row["title"] else row["text"]
        for row in corpus
    ]

    emb = encode_texts(model, doc_texts, args.batch, "corpus")
    assert emb.shape == (len(doc_texts), EMBEDDING_DIM)

    emb_path  = DATA_DIR / f"{args.dataset}_embeddings.npy"
    ids_path  = DATA_DIR / f"{args.dataset}_doc_ids.json"
    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(doc_ids))
    print(f"  Saved {emb.shape} → {emb_path}")

    # ── Query embeddings ─────────────────────────────────────
    print("\nEncoding queries ...")
    queries = load_jsonl(queries_path)
    q_ids   = [row["_id"] for row in queries]
    q_texts = [row["text"] for row in queries]

    q_emb = encode_texts(model, q_texts, args.batch, "queries")
    assert q_emb.shape == (len(q_texts), EMBEDDING_DIM)

    q_emb_path = DATA_DIR / f"{args.dataset}_query_embeddings.npy"
    q_ids_path = DATA_DIR / f"{args.dataset}_query_ids.json"
    np.save(q_emb_path, q_emb)
    q_ids_path.write_text(json.dumps(q_ids))
    print(f"  Saved {q_emb.shape} → {q_emb_path}")

    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()
