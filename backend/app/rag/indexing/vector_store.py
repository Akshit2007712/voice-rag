"""Qdrant-backed vector store with local/remote configuration boundaries."""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.rag.language_config import get_application_language
from app.rag.ingestion.chunker import Chunk
from app.rag.indexing.qdrant_retry import QdrantOperationMetrics, QdrantRetryPolicy, call_with_qdrant_retry


DEFAULT_COLLECTION_NAME = "msmarco_xi_hindi"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PointIdBuilder = Callable[[Chunk], str]


@dataclass(frozen=True)
class QdrantSettings:
    mode: str = "local"
    path: Path = PROJECT_ROOT / "data" / "qdrant"
    url: str | None = None
    api_key: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME

    @classmethod
    def from_environment(cls) -> "QdrantSettings":
        mode = os.getenv("QDRANT_MODE", "local").lower()
        collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
        if mode == "local":
            path = Path(os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "data" / "qdrant")))
            return cls(mode=mode, path=path, collection_name=collection_name)
        if mode == "remote":
            return cls(mode=mode, url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), collection_name=collection_name)
        raise ValueError("QDRANT_MODE must be 'local' or 'remote'")


def deterministic_point_id(chunk: Chunk) -> str:
    """Create the legacy stable UUID used by the development collection."""
    metadata = chunk.metadata
    identity = ":".join(str(metadata.get(key)) for key in ("query_id", "passage_index", "chunk_index"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msmarco-xi-hindi:{identity}"))


def build_strategy_aware_point_identity(
    query_id: object,
    passage_index: object,
    chunk_index: object,
    chunk_strategy: object,
) -> str:
    """Return the canonical full-index provenance string used for UUIDv5 IDs."""
    return (
        f"query_id={query_id}|passage_index={passage_index}|"
        f"chunk_index={chunk_index}|chunk_strategy={chunk_strategy}"
    )


def build_strategy_aware_point_id(
    query_id: object,
    passage_index: object,
    chunk_index: object,
    chunk_strategy: object,
) -> str:
    """Build the full-collection UUID from unambiguous chunk provenance fields."""
    identity = build_strategy_aware_point_identity(
        query_id, passage_index, chunk_index, chunk_strategy
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msmarco-xi-hindi-full:{identity}"))


def strategy_aware_point_id(chunk: Chunk) -> str:
    """Create a full-Hindi-collection ID that cannot collide across chunk strategies."""
    metadata = chunk.metadata
    return build_strategy_aware_point_id(
        metadata.get("query_id"),
        metadata.get("passage_index"),
        metadata.get("chunk_index"),
        metadata.get("chunk_strategy"),
    )


def payload_from_chunk(chunk: Chunk) -> dict[str, Any]:
    """Keep text and metadata separate from the embedding vector."""
    return {"text": chunk.text, **chunk.metadata}


class VectorStore:
    """Stable interface over local or future remote Qdrant clients."""

    def __init__(
        self,
        settings: QdrantSettings | None = None,
        client: QdrantClient | None = None,
        point_id_builder: PointIdBuilder = deterministic_point_id,
        retry_policy: QdrantRetryPolicy | None = None,
    ) -> None:
        self.settings = settings or QdrantSettings.from_environment()
        self._point_id_builder = point_id_builder
        self._retry_policy = retry_policy or QdrantRetryPolicy()
        self.qdrant_retry_count = 0
        self.qdrant_failed_request_count = 0
        self.last_qdrant_operation_metrics: QdrantOperationMetrics | None = None
        if client is not None:
            self.client = client
        elif self.settings.mode == "local":
            self.settings.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.settings.path))
        else:
            if not self.settings.url:
                raise ValueError("QDRANT_URL is required for remote mode")
            self.client = QdrantClient(url=self.settings.url, api_key=self.settings.api_key)

    @property
    def collection_name(self) -> str:
        return self.settings.collection_name

    def ensure_collection(self, vector_size: int, reset: bool = False) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be at least 1")
        exists = self.collection_exists()
        if exists and reset:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            collection = self._remote_read("collection_info", lambda: self.client.get_collection(self.collection_name))
            configured_size = collection.config.params.vectors.size
            if configured_size != vector_size:
                raise ValueError(f"Existing collection vector size {configured_size} does not match E5 dimension {vector_size}; use --reset to recreate it")

        # Create payload indexes required by Qdrant Cloud for filtered search
        from qdrant_client.models import PayloadSchemaType
        for field in ("language", "target_lang"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass


    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match")
        points = [
            PointStruct(id=self._point_id_builder(chunk), vector=vector.tolist(), payload=payload_from_chunk(chunk))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        target_lang: str | None = None,
    ) -> list[Any]:
        if limit < 1:
            raise ValueError("top_k must be at least 1")
        query_filter = None
        if target_lang:
            language = get_application_language(target_lang)
            query_filter = Filter(
                must=[
                    FieldCondition(key="language", match=MatchValue(value=language)),
                    FieldCondition(key="target_lang", match=MatchValue(value=target_lang)),
                ]
            )
        if hasattr(self.client, "query_points"):
            return self._remote_read(
                "query_points",
                lambda: list(self.client.query_points(collection_name=self.collection_name, query=query_vector.tolist(), limit=limit, with_payload=True, query_filter=query_filter).points),
            )
        return self._remote_read(
            "search",
            lambda: list(self.client.search(collection_name=self.collection_name, query_vector=query_vector.tolist(), limit=limit, with_payload=True, query_filter=query_filter)),
        )

    def point_count(self) -> int:
        return int(self._remote_read("count", lambda: self.client.count(collection_name=self.collection_name, exact=True)).count)

    def collection_info(self) -> Any:
        """Read collection configuration through the remote retry boundary."""
        return self._remote_read("collection_info", lambda: self.client.get_collection(self.collection_name))

    def collection_exists(self) -> bool:
        """Check collection availability with remote transient-failure retry only."""
        return bool(self._remote_read("collection_exists", lambda: self.client.collection_exists(self.collection_name)))

    def scroll(self, **kwargs: Any) -> tuple[list[Any], Any]:
        """Read payload pages for startup BM25 construction through the retry path."""
        return self._remote_read("scroll", lambda: self.client.scroll(**kwargs))

    def _remote_read(self, operation: str, call: Callable[[], Any]) -> Any:
        """Reuse the one client and retry only remote read transport failures."""
        if self.settings.mode != "remote":
            return call()
        result, metrics, failed_requests = call_with_qdrant_retry(operation, call, self._retry_policy)
        self.last_qdrant_operation_metrics = metrics
        self.qdrant_retry_count += metrics.retry_count
        self.qdrant_failed_request_count += failed_requests
        return result

    def close(self) -> None:
        self.client.close()
