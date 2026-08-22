"""Run a bounded local ingestion → preprocessing → chunking smoke test."""

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.ingestion.chunker import chunk_retrieval_document  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record  # noqa: E402


def main() -> None:
    """Chunk documents from only the requested number of local raw records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.records < 1:
        parser.error("--records must be at least 1")

    raw_records_processed = 0
    documents = []
    chunks = []
    records = iter_msmarco_xi_records(batch_size=args.batch_size)
    while raw_records_processed < args.records:
        try:
            raw_record = next(records)
        except StopIteration:
            break
        raw_records_processed += 1
        documents.extend(preprocess_msmarco_xi_record(raw_record))

    for document in documents:
        chunks.extend(chunk_retrieval_document(document, max_tokens=args.max_tokens))

    strategies = [chunk.metadata["chunk_strategy"] for chunk in chunks]
    print(f"RAW RECORDS PROCESSED: {raw_records_processed}")
    print(f"RETRIEVAL DOCUMENTS: {len(documents)}")
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print(f"WHOLE-PASSAGE CHUNKS: {strategies.count('whole_passage')}")
    print(f"SENTENCE-OVERLAP CHUNKS: {strategies.count('sentence_overlap')}")
    print(f"TOKEN-WINDOW FALLBACK CHUNKS: {strategies.count('token_window_fallback')}")
    print(f"MAXIMUM OBSERVED TOKEN COUNT: {max((chunk.metadata['token_count'] for chunk in chunks), default=0)}")
    if chunks:
        print(f"EXAMPLE CHUNK TEXT: {chunks[0].text}")
        print(f"EXAMPLE CHUNK METADATA: {json.dumps(chunks[0].metadata, ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    main()
