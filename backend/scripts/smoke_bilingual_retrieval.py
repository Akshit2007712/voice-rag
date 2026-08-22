"""Read-only bilingual retrieval smoke test with language-isolation checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.bilingual_cloud_runtime import create_bilingual_cloud_runtime  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402


def _queries(language: str, count: int) -> list[str]:
    field = "Eng_Query" if language == "en" else "query"
    queries: list[str] = []
    for record in iter_msmarco_xi_records(language, "validation", batch_size=500):
        query = str(record.get(field, "")).strip()
        if query:
            queries.append(query)
            if len(queries) == count:
                return queries
    raise RuntimeError(f"Only found {len(queries)} usable {language} queries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-per-language", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if min(args.queries_per_language, args.top_k) < 1:
        parser.error("numeric arguments must be positive")
    runtime, _ = create_bilingual_cloud_runtime(BACKEND_ROOT)
    try:
        for language in ("hi", "en"):
            target_lang = get_qdrant_target_lang(language)
            for query in _queries(language, args.queries_per_language):
                results = runtime.retrieve(query, top_k=args.top_k, target_lang=target_lang)
                if not results:
                    raise RuntimeError(f"No {language} results for smoke query: {query}")
                foreign = [chunk.metadata.get("language") for chunk in results if chunk.metadata.get("language") != language]
                if foreign:
                    raise RuntimeError(f"Cross-language retrieval for {language}: {foreign}")
            print(f"SMOKE_{language.upper()}_QUERIES={args.queries_per_language} STATUS=PASS")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
