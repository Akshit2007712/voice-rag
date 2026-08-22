"""Retrieve and format bounded LLM-ready context without calling an LLM."""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.generation.context_builder import ContextBuilder  # noqa: E402
from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.indexing.vector_store import VectorStore  # noqa: E402
from app.rag.retrieval.retriever import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=1_000)
    args = parser.parse_args()

    store: VectorStore | None = None
    try:
        embedder = E5Embedder()
        store = VectorStore()
        retrieved = Retriever(embedder, store).retrieve(args.query, top_k=args.top_k)
        bundle = ContextBuilder().build(retrieved, args.max_context_tokens)
    except Exception as error:
        parser.error(f"Context builder smoke test failed: {type(error).__name__}: {error}")
    finally:
        if store is not None:
            store.close()

    print(f"QUERY: {args.query}")
    print(f"RETRIEVED CHUNKS: {len(retrieved)}")
    print(f"SELECTED EVIDENCE COUNT: {bundle.evidence_count}")
    print(f"ESTIMATED CONTEXT TOKENS: {bundle.estimated_token_count}")
    print("FINAL CONTEXT:")
    print(bundle.text)


if __name__ == "__main__":
    main()
