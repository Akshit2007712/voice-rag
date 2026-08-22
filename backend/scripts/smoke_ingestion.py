"""Run a small local-Parquet-to-preprocessor ingestion smoke test."""

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from the backend directory.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.ingestion.dataset_loader import (  # noqa: E402
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
    iter_msmarco_xi_records,
)
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record  # noqa: E402


def main() -> None:
    """Process only the requested number of local records through preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--records", type=int, default=3)
    args = parser.parse_args()

    if args.records < 1:
        parser.error("--records must be at least 1")

    raw_records_tested = 0
    retrieval_documents = []
    records = iter_msmarco_xi_records(
        language=args.language,
        split=args.split,
        batch_size=args.batch_size,
    )

    while raw_records_tested < args.records:
        try:
            record = next(records)
        except StopIteration:
            break
        raw_records_tested += 1
        retrieval_documents.extend(preprocess_msmarco_xi_record(record))

    print(f"RAW RECORDS: {raw_records_tested}")
    print(f"RETRIEVAL DOCUMENTS: {len(retrieval_documents)}")
    if retrieval_documents:
        first_document = retrieval_documents[0]
        print(f"FIRST DOCUMENT TEXT: {first_document.text}")
        print(
            "FIRST DOCUMENT METADATA: "
            f"{json.dumps(first_document.metadata, ensure_ascii=False, default=str)}"
        )
    else:
        print("FIRST DOCUMENT TEXT: <none>")
        print("FIRST DOCUMENT METADATA: <none>")


if __name__ == "__main__":
    main()
