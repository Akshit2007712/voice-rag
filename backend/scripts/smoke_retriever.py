"""Run a retrieval-only smoke test against the persistent local Qdrant store."""

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.indexing.vector_store import VectorStore  # noqa: E402
from app.rag.retrieval.retriever import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-lang")
    args = parser.parse_args()
    try:
        embedder = E5Embedder()
        store = VectorStore()
        results = Retriever(embedder, store).retrieve(args.query, args.top_k, args.target_lang)
    except Exception as error:
        parser.error(f"Retriever smoke test failed: {type(error).__name__}: {error}")

    print(f"QUERY: {args.query}")
    print(f"E5 DEVICE: {embedder.device}")
    print(f"RESULT COUNT: {len(results)}")
    for rank, result in enumerate(results, start=1):
        print(f"RANK: {rank}")
        print(f"SCORE: {result.score}")
        print(f"TEXT: {result.text[:200]}")
        print(f"METADATA: {json.dumps(result.metadata, ensure_ascii=False, default=str)}")
    relevant = any(result.metadata.get("is_selected") == 1 for result in results)
    print(f"GROUND-TRUTH RELEVANT IN TOP-K: {relevant}")
    store.close()


if __name__ == "__main__":
    main()
