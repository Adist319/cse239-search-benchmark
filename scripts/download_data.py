"""
Download a BEIR dataset from HuggingFace and save as local JSON files.

Usage:
    python download_data.py [--dataset fiqa] [--limit 0]

Outputs (under data/):
    {dataset}_corpus.jsonl    — {_id, title, text} per line
    {dataset}_queries.jsonl   — {_id, text} per line
    {dataset}_qrels.tsv       — query_id\tdoc_id\trelevance
"""
import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, DATASET_NAME, DATASET_LIMIT


# Mapping of BEIR dataset names to their HuggingFace identifiers.
# Some datasets use a non-standard qrels source.
BEIR_HF = {
    "fiqa":       ("BeIR/fiqa",       "BeIR/fiqa-qrels"),
    "trec-covid": ("BeIR/trec-covid", "BeIR/trec-covid-qrels"),
    "nfcorpus":   ("BeIR/nfcorpus",   "BeIR/nfcorpus-qrels"),
    "scifact":    ("BeIR/scifact",    "BeIR/scifact-qrels"),
    "scidocs":    ("BeIR/scidocs",    "BeIR/scidocs-qrels"),
    "quora":      ("BeIR/quora",      "BeIR/quora-qrels"),
    "msmarco":    ("BeIR/msmarco",    "BeIR/msmarco-qrels"),
}


def download_corpus(hf_id: str, limit: int, out_path: Path) -> int:
    print(f"Downloading corpus from {hf_id} ...")
    ds = load_dataset(hf_id, "corpus", split="corpus", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    count = 0
    with out_path.open("w") as f:
        for row in tqdm(ds, desc="corpus"):
            record = {
                "_id":   str(row["_id"]),
                "title": row.get("title") or "",
                "text":  row["text"],
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"  Wrote {count:,} corpus documents → {out_path}")
    return count


def download_queries(hf_id: str, out_path: Path) -> int:
    print(f"Downloading queries from {hf_id} ...")
    ds = load_dataset(hf_id, "queries", split="queries", trust_remote_code=True)

    count = 0
    with out_path.open("w") as f:
        for row in tqdm(ds, desc="queries"):
            record = {"_id": str(row["_id"]), "text": row["text"]}
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"  Wrote {count:,} queries → {out_path}")
    return count


def download_qrels(hf_qrels_id: str, out_path: Path) -> int:
    print(f"Downloading qrels from {hf_qrels_id} ...")
    # BEIR qrels are typically in a 'test' split
    for split in ("test", "validation", "train"):
        try:
            ds = load_dataset(hf_qrels_id, split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    else:
        print(f"  WARNING: Could not download qrels from {hf_qrels_id}")
        print(f"  Writing empty qrels file. Relevance metrics will not be available.")
        out_path.write_text("query-id\tcorpus-id\tscore\n")
        return 0

    count = 0
    with out_path.open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in tqdm(ds, desc="qrels"):
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
            count += 1
    print(f"  Wrote {count:,} qrels → {out_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download BEIR dataset")
    parser.add_argument("--dataset", default=DATASET_NAME,
                        choices=list(BEIR_HF), help="BEIR dataset name")
    parser.add_argument("--limit", type=int, default=DATASET_LIMIT,
                        help="Limit corpus size (0 = no limit, for dev use 10000)")
    args = parser.parse_args()

    if args.dataset not in BEIR_HF:
        print(f"Unknown dataset '{args.dataset}'. Choose from: {list(BEIR_HF)}")
        sys.exit(1)

    hf_corpus_id, hf_qrels_id = BEIR_HF[args.dataset]

    corpus_path  = DATA_DIR / f"{args.dataset}_corpus.jsonl"
    queries_path = DATA_DIR / f"{args.dataset}_queries.jsonl"
    qrels_path   = DATA_DIR / f"{args.dataset}_qrels.tsv"

    n_docs    = download_corpus(hf_corpus_id, args.limit, corpus_path)
    n_queries = download_queries(hf_corpus_id, queries_path)
    n_qrels   = download_qrels(hf_qrels_id, qrels_path)

    print(f"\nDone. Dataset '{args.dataset}':")
    print(f"  {n_docs:>8,} corpus documents")
    print(f"  {n_queries:>8,} queries")
    print(f"  {n_qrels:>8,} qrels")


if __name__ == "__main__":
    main()
