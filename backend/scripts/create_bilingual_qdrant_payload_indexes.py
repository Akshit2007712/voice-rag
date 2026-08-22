"""Create only the Qdrant Cloud payload indexes required by bilingual retrieval."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.bilingual_cloud_runtime import BILINGUAL_COLLECTION  # noqa: E402


REQUIRED_KEYWORD_INDEXES = ("language", "target_lang")


def _payload_schema(client: QdrantClient) -> dict[str, object]:
    info = client.get_collection(BILINGUAL_COLLECTION)
    schema = getattr(info, "payload_schema", None)
    return dict(schema or {})


def _is_keyword_index(schema: dict[str, object], field: str) -> bool:
    index = schema.get(field)
    if index is None:
        return False
    data_type = getattr(index, "data_type", index)
    return str(getattr(data_type, "value", data_type)).lower() == "keyword"


def _ensure_keyword_index(client: QdrantClient, field: str) -> None:
    schema = _payload_schema(client)
    if _is_keyword_index(schema, field):
        print(f"PAYLOAD_INDEX_{field.upper()}_EXISTS=true")
        return
    client.create_payload_index(
        collection_name=BILINGUAL_COLLECTION,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )
    schema = _payload_schema(client)
    if not _is_keyword_index(schema, field):
        raise RuntimeError(f"Qdrant did not report a keyword payload index for {field}")
    print(f"PAYLOAD_INDEX_{field.upper()}_CREATED=true")


def _verify_language_filter(client: QdrantClient, language: str, target_lang: str) -> None:
    query_filter = Filter(
        must=[
            FieldCondition(key="language", match=MatchValue(value=language)),
            FieldCondition(key="target_lang", match=MatchValue(value=target_lang)),
        ]
    )
    points, _ = client.scroll(
        collection_name=BILINGUAL_COLLECTION,
        scroll_filter=query_filter,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        raise RuntimeError(f"Filtered bilingual query returned no {language} points")
    incorrect = [
        dict(point.payload or {})
        for point in points
        if (point.payload or {}).get("language") != language
        or (point.payload or {}).get("target_lang") != target_lang
    ]
    if incorrect:
        raise RuntimeError(f"{language} filter returned cross-language payloads")
    print(f"FILTER_LANGUAGE_{language.upper()}_QUERY_SUCCEEDS=true")
    print(f"FILTER_LANGUAGE_{language.upper()}_PAYLOADS_ONLY_{language.upper()}=true")


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL is required")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY is required")
    print("DOCUMENT_EMBEDDINGS_CREATED=0")
    print("QDRANT_POINT_WRITES=0")
    print("COLLECTION_RESET=0")
    print("COLLECTION_DELETE=0")
    print("QDRANT_MODE=remote")
    print(f"QDRANT_COLLECTION={BILINGUAL_COLLECTION}")
    print("QDRANT_API_KEY_PRESENT=true")
    client = QdrantClient(url=url, api_key=api_key)
    try:
        if not client.collection_exists(BILINGUAL_COLLECTION):
            raise RuntimeError(f"Required collection does not exist: {BILINGUAL_COLLECTION}")
        print(f"QDRANT_COLLECTION_POINT_COUNT={client.count(collection_name=BILINGUAL_COLLECTION, exact=True).count}")
        for field in REQUIRED_KEYWORD_INDEXES:
            _ensure_keyword_index(client, field)
        schema = _payload_schema(client)
        for field in REQUIRED_KEYWORD_INDEXES:
            if not _is_keyword_index(schema, field):
                raise RuntimeError(f"Required keyword index missing after creation: {field}")
            print(f"PAYLOAD_INDEX_{field.upper()}_VERIFIED=keyword")
        _verify_language_filter(client, "hi", "hin_Deva")
        _verify_language_filter(client, "en", "eng_Latn")
    finally:
        client.close()


if __name__ == "__main__":
    main()
