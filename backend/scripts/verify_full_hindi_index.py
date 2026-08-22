"""Read-only structural verification for the full Hindi MSMARCO-XI index.

This intentionally regenerates the loader -> preprocessor -> chunker -> point-ID
stream without creating embeddings or writing to Qdrant.  Its temporary SQLite
database bounds memory while comparing roughly one million deterministic IDs.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import heapq
import json
import math
import os
import pickle
import re
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.rag.indexing.embedder import E5Embedder
from app.rag.indexing.vector_store import QdrantSettings, VectorStore, strategy_aware_point_id
from app.rag.ingestion.chunker import Chunk, get_e5_tokenizer, iter_document_chunks
from app.rag.ingestion.dataset_loader import DEFAULT_LANGUAGE, DEFAULT_SPLIT, iter_msmarco_xi_records
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record
from app.rag.language_config import get_qdrant_target_lang
from app.rag.retrieval.bm25_store import tokenize_text
from app.rag.retrieval.retriever import Retriever


FULL_HINDI_COLLECTION = "msmarco_xi_hindi_full"
FAILED_QUERY_ID = "430672"
FAILED_QUERY = "वाष्प दबाव में वृद्धि या कमी"
REQUIRED_PAYLOAD_FIELDS = (
    "query_id",
    "passage_index",
    "chunk_index",
    "chunk_strategy",
    "target_lang",
    "text",
)
LOCAL_POINT_ID_PATTERN = re.compile(
    rb"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
)


@dataclass(frozen=True)
class ChunkDescriptor:
    """Compact, printable provenance for one regenerated chunk."""

    point_id: str
    row_number: int
    query_id: str
    passage_index: int
    chunk_index: int
    chunk_strategy: str
    text_hash: str
    preview: str

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (
            self.query_id,
            self.passage_index,
            self.chunk_index,
            self.chunk_strategy,
        )


@dataclass(frozen=True)
class AuditLexicalMatch:
    """A query-scoped, disk-backed BM25 diagnostic result."""

    rank: int
    score: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class AuditHybridMatch:
    """RRF output for diagnostic reporting without constructing a corpus BM25 store."""

    rank: int
    metadata: dict[str, object]
    semantic_score: float | None
    bm25_score: float | None
    fused_score: float


class QueryScopedBM25Audit:
    """Calculate BM25 for one fixed query with compact on-disk candidate state.

    The production ``BM25Store`` deliberately keeps the whole corpus in memory.
    That is unsuitable for a one-off million-point verifier, so this helper stores
    only documents containing a token from the one failed smoke query.  It uses the
    same tokenizer and BM25 constants without altering production BM25 behavior.
    """

    def __init__(self, connection: sqlite3.Connection, query: str, k1: float = 1.5, b: float = 0.75) -> None:
        self.connection = connection
        self.query_tokens = tuple(sorted(set(tokenize_text(query))))
        self.k1 = k1
        self.b = b
        self.document_count = 0
        self.total_document_length = 0
        self.document_frequency = {token: 0 for token in self.query_tokens}
        self._candidate_sequence = 0

    def observe_payload(self, point_id: str, payload: Mapping[str, Any], target_lang: str) -> None:
        """Accumulate one payload's compact contribution while Qdrant is scrolled."""
        if payload.get("target_lang") != target_lang:
            return
        text = _normalize_text(payload.get("text"))
        if not text:
            return
        tokens = tokenize_text(text)
        self.document_count += 1
        self.total_document_length += len(tokens)
        frequencies = {token: tokens.count(token) for token in self.query_tokens if token in tokens}
        for token in frequencies:
            self.document_frequency[token] += 1
        if not frequencies:
            return
        self._candidate_sequence += 1
        self.connection.execute(
            """
            INSERT OR REPLACE INTO bm25_candidates (
                point_id, sequence, query_id, passage_index, chunk_index,
                chunk_strategy, document_length, frequencies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                point_id,
                self._candidate_sequence,
                str(payload.get("query_id")),
                str(payload.get("passage_index")),
                str(payload.get("chunk_index")),
                str(payload.get("chunk_strategy")),
                len(tokens),
                json.dumps(frequencies, ensure_ascii=False, sort_keys=True),
            ),
        )

    def observe_chunk(self, point_id: str, chunk: Chunk, target_lang: str) -> None:
        """Accumulate source-chunk statistics without a corpus-wide Qdrant payload scan."""
        self.observe_payload(point_id, {"text": chunk.text, **chunk.metadata}, target_lang)

    def save_summary(self) -> None:
        """Persist the tiny corpus aggregates required by post-scan-only BM25 ranking."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO bm25_summary(
                name, query_tokens, document_count, total_document_length, document_frequency
            ) VALUES ('query_430672', ?, ?, ?, ?)
            """,
            (
                json.dumps(self.query_tokens, ensure_ascii=False),
                self.document_count,
                self.total_document_length,
                json.dumps(self.document_frequency, ensure_ascii=False, sort_keys=True),
            ),
        )

    @classmethod
    def load(cls, connection: sqlite3.Connection, query: str) -> "QueryScopedBM25Audit | None":
        """Restore the compact query statistics generated during a prior ID scan."""
        row = connection.execute(
            """
            SELECT query_tokens, document_count, total_document_length, document_frequency
            FROM bm25_summary WHERE name = 'query_430672'
            """
        ).fetchone()
        if row is None:
            return None
        audit = cls(connection, query)
        query_tokens, audit.document_count, audit.total_document_length, frequencies = row
        if tuple(json.loads(query_tokens)) != audit.query_tokens:
            raise ValueError("Verification database BM25 query does not match query 430672")
        audit.document_frequency = json.loads(frequencies)
        return audit

    def search(self, top_k: int) -> list[AuditLexicalMatch]:
        """Score only the compact on-disk candidate rows for the fixed query."""
        if top_k < 1 or not self.document_count or not self.query_tokens:
            return []
        average_length = self.total_document_length / self.document_count
        heap: list[tuple[float, int, dict[str, object]]] = []
        for row in self.connection.execute(
            """
            SELECT sequence, query_id, passage_index, chunk_index, chunk_strategy,
                   document_length, frequencies
            FROM bm25_candidates
            """
        ):
            sequence, query_id, passage_index, chunk_index, strategy, length, frequency_json = row
            frequencies = json.loads(frequency_json)
            score = 0.0
            for token, frequency in frequencies.items():
                document_frequency = self.document_frequency[token]
                inverse_frequency = math.log(
                    1 + (self.document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / average_length
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            entry = (
                score,
                int(sequence),
                {
                    "query_id": query_id,
                    "passage_index": passage_index,
                    "chunk_index": chunk_index,
                    "chunk_strategy": strategy,
                },
            )
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
        ordered = sorted(
            heap,
            key=lambda entry: (-entry[0], tuple(str(entry[2][key]) for key in ("query_id", "passage_index", "chunk_index"))),
        )
        return [AuditLexicalMatch(rank, score, metadata) for rank, (score, _, metadata) in enumerate(ordered, start=1)]


def _normalize_text(value: object) -> str:
    """Return basic whitespace-normalized text for diagnostics only."""
    return " ".join(value.split()) if isinstance(value, str) else ""


def _text_hash(text: str) -> str:
    """Return a stable content hash without retaining complete duplicate text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preview(text: str, limit: int = 80) -> str:
    """Return a one-line safe preview for collision output."""
    normalized = _normalize_text(text)
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}…"


def _descriptor(chunk: Chunk, row_number: int) -> ChunkDescriptor:
    """Build the deterministic-ID descriptor used by the full index."""
    metadata = chunk.metadata
    return ChunkDescriptor(
        point_id=strategy_aware_point_id(chunk),
        row_number=row_number,
        query_id=str(metadata.get("query_id")),
        passage_index=int(metadata.get("passage_index")),
        chunk_index=int(metadata.get("chunk_index")),
        chunk_strategy=str(metadata.get("chunk_strategy")),
        text_hash=_text_hash(chunk.text),
        preview=_preview(chunk.text),
    )


def _print_json(label: str, value: object) -> None:
    """Print readable UTF-8 structured diagnostics."""
    print(f"{label}={json.dumps(value, ensure_ascii=False, sort_keys=True)}", flush=True)


def _rss_mb() -> float | None:
    """Return current process RSS where the platform exposes it without a dependency."""
    try:
        if sys.platform == "win32":
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return counters.WorkingSetSize / (1024 * 1024)
            return None
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * 4096 / (1024 * 1024)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _print_progress(
    label: str,
    *,
    rows: int,
    chunks: int,
    unique_ids: int,
    collisions: int,
    started_at: float,
) -> None:
    """Emit bounded-scan progress and a lightweight process-memory diagnostic."""
    rss_mb = _rss_mb()
    rss = f"{rss_mb:.1f}" if rss_mb is not None else "unavailable"
    print(
        f"{label} ROWS_SCANNED={rows} CHUNKS_SCANNED={chunks} "
        f"UNIQUE_IDS_SO_FAR={unique_ids} COLLISIONS_SO_FAR={collisions} "
        f"ELAPSED_TIME_S={time.perf_counter() - started_at:.1f} RSS_MB={rss}",
        flush=True,
    )


def _stage_start(name: str) -> float:
    """Print an explicit bounded-verifier stage boundary with current RSS."""
    rss_mb = _rss_mb()
    print(f"STAGE_START name={name} RSS_MB={rss_mb:.1f}" if rss_mb is not None else f"STAGE_START name={name} RSS_MB=unavailable", flush=True)
    return time.perf_counter()


def _stage_end(name: str, started_at: float) -> None:
    """Print elapsed time and RSS after a bounded-verifier stage."""
    rss_mb = _rss_mb()
    print(
        f"STAGE_END name={name} ELAPSED_TIME_S={time.perf_counter() - started_at:.1f} "
        f"RSS_MB={rss_mb:.1f}" if rss_mb is not None else
        f"STAGE_END name={name} ELAPSED_TIME_S={time.perf_counter() - started_at:.1f} RSS_MB=unavailable",
        flush=True,
    )


def _open_verification_database(path: Path, reuse: bool = False) -> sqlite3.Connection:
    """Create or reopen the bounded local verifier store without touching Qdrant."""
    if reuse and not path.is_file():
        raise FileNotFoundError(f"Verification database not found: {path}")
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = FILE;
        PRAGMA cache_size = -8192;
        CREATE TABLE IF NOT EXISTS expected_ids (
            point_id TEXT PRIMARY KEY,
            row_number INTEGER NOT NULL,
            query_id TEXT NOT NULL,
            passage_index INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_strategy TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            preview TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collisions (
            point_id TEXT NOT NULL,
            original_row_number INTEGER NOT NULL,
            original_query_id TEXT NOT NULL,
            original_passage_index INTEGER NOT NULL,
            original_chunk_index INTEGER NOT NULL,
            original_chunk_strategy TEXT NOT NULL,
            original_text_hash TEXT NOT NULL,
            original_preview TEXT NOT NULL,
            duplicate_row_number INTEGER NOT NULL,
            duplicate_query_id TEXT NOT NULL,
            duplicate_passage_index INTEGER NOT NULL,
            duplicate_chunk_index INTEGER NOT NULL,
            duplicate_chunk_strategy TEXT NOT NULL,
            duplicate_text_hash TEXT NOT NULL,
            duplicate_preview TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_rows (
            query_id TEXT PRIMARY KEY,
            row_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS actual_ids (point_id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS bm25_candidates (
            point_id TEXT PRIMARY KEY,
            sequence INTEGER NOT NULL,
            query_id TEXT NOT NULL,
            passage_index TEXT NOT NULL,
            chunk_index TEXT NOT NULL,
            chunk_strategy TEXT NOT NULL,
            document_length INTEGER NOT NULL,
            frequencies TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_records (
            query_id TEXT PRIMARY KEY,
            record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bm25_summary (
            name TEXT PRIMARY KEY,
            query_tokens TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            total_document_length INTEGER NOT NULL,
            document_frequency TEXT NOT NULL
        );
        """
    )
    return connection


def _insert_expected_descriptor(connection: sqlite3.Connection, descriptor: ChunkDescriptor) -> bool:
    """Store an ID or record the exact pair when that ID has already occurred."""
    try:
        connection.execute(
            """
            INSERT INTO expected_ids (
                point_id, row_number, query_id, passage_index, chunk_index,
                chunk_strategy, text_hash, preview
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                descriptor.point_id,
                descriptor.row_number,
                descriptor.query_id,
                descriptor.passage_index,
                descriptor.chunk_index,
                descriptor.chunk_strategy,
                descriptor.text_hash,
                descriptor.preview,
            ),
        )
        return False
    except sqlite3.IntegrityError:
        original = connection.execute(
            """
            SELECT row_number, query_id, passage_index, chunk_index,
                   chunk_strategy, text_hash, preview
            FROM expected_ids WHERE point_id = ?
            """,
            (descriptor.point_id,),
        ).fetchone()
        if original is None:  # Defensive: an IntegrityError must have a source row.
            raise RuntimeError("Could not resolve a duplicate deterministic point ID")
        connection.execute(
            """
            INSERT INTO collisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                descriptor.point_id,
                *original,
                descriptor.row_number,
                descriptor.query_id,
                descriptor.passage_index,
                descriptor.chunk_index,
                descriptor.chunk_strategy,
                descriptor.text_hash,
                descriptor.preview,
            ),
        )
        return True


def _record_query_row(connection: sqlite3.Connection, query_id: object) -> None:
    """Count source rows per query ID to test the ID scheme's uniqueness premise."""
    connection.execute(
        """
        INSERT INTO query_rows(query_id, row_count) VALUES (?, 1)
        ON CONFLICT(query_id) DO UPDATE SET row_count = row_count + 1
        """,
        (str(query_id),),
    )


def _store_failed_query_source(connection: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    """Persist the one source row needed by post-scan-only diagnostics."""
    connection.execute(
        "INSERT OR REPLACE INTO source_records(query_id, record_json) VALUES (?, ?)",
        (FAILED_QUERY_ID, json.dumps(record, ensure_ascii=False)),
    )


def _load_failed_query_source(connection: sqlite3.Connection) -> dict[str, Any] | None:
    """Load only query 430672's persisted source record for a resumed verifier run."""
    row = connection.execute(
        "SELECT record_json FROM source_records WHERE query_id = ?", (FAILED_QUERY_ID,)
    ).fetchone()
    return json.loads(row[0]) if row else None


def _reset_post_scan_tables(connection: sqlite3.Connection) -> None:
    """Clear only prior verifier observations before a new read-only Qdrant scan."""
    connection.executescript(
        """
        DROP TABLE IF EXISTS actual_ids;
        CREATE TABLE actual_ids (point_id TEXT PRIMARY KEY);
        """
    )
    connection.commit()


def _iter_regenerated_chunks(
    language: str,
    split: str,
    batch_size: int,
    max_tokens: int,
    connection: sqlite3.Connection,
    max_rows: int | None,
    progress_every_rows: int,
    lexical_audit: QueryScopedBM25Audit,
    target_lang: str,
) -> tuple[int, int, dict[str, Any] | None]:
    """Regenerate every chunk lazily and record its deterministic identity."""
    tokenizer = get_e5_tokenizer()
    total_chunks = 0
    duplicate_occurrences = 0
    failed_record: dict[str, Any] | None = None
    started_at = time.perf_counter()
    for row_number, record in enumerate(
        iter_msmarco_xi_records(language, split, batch_size), start=1
    ):
        if max_rows is not None and row_number > max_rows:
            break
        _record_query_row(connection, record.get("query_id"))
        if str(record.get("query_id")) == FAILED_QUERY_ID:
            failed_record = record
            _store_failed_query_source(connection, record)
        documents = preprocess_msmarco_xi_record(record)
        for chunk in iter_document_chunks(documents, max_tokens=max_tokens, tokenizer=tokenizer):
            total_chunks += 1
            descriptor = _descriptor(chunk, row_number)
            duplicate_occurrences += _insert_expected_descriptor(connection, descriptor)
            lexical_audit.observe_chunk(descriptor.point_id, chunk, target_lang)
        if row_number % progress_every_rows == 0:
            connection.commit()
            unique_ids = connection.execute("SELECT COUNT(*) FROM expected_ids").fetchone()[0]
            _print_progress(
                "ID_REGENERATION_PROGRESS",
                rows=row_number,
                chunks=total_chunks,
                unique_ids=unique_ids,
                collisions=duplicate_occurrences,
                started_at=started_at,
            )
    connection.commit()
    lexical_audit.save_summary()
    connection.commit()
    return total_chunks, duplicate_occurrences, failed_record


def _collision_rows(connection: sqlite3.Connection) -> list[dict[str, object]]:
    """Return every collision with both source descriptors for human review."""
    columns = [column[1] for column in connection.execute("PRAGMA table_info(collisions)")]
    return [dict(zip(columns, row, strict=True)) for row in connection.execute("SELECT * FROM collisions")]


def _print_collision_report(connection: sqlite3.Connection) -> tuple[int, int, int]:
    """Print collision type counts and both logical chunk descriptors."""
    rows = _collision_rows(connection)
    identical_content = 0
    different_content = 0
    exact_duplicate_chunks = 0
    for collision in rows:
        same_identity = (
            collision["original_query_id"],
            collision["original_passage_index"],
            collision["original_chunk_index"],
            collision["original_chunk_strategy"],
        ) == (
            collision["duplicate_query_id"],
            collision["duplicate_passage_index"],
            collision["duplicate_chunk_index"],
            collision["duplicate_chunk_strategy"],
        )
        same_content = collision["original_text_hash"] == collision["duplicate_text_hash"]
        if same_content:
            identical_content += 1
        else:
            different_content += 1
        if same_identity and same_content:
            exact_duplicate_chunks += 1
        _print_json(
            "POINT_ID_COLLISION",
            {
                "point_id": collision["point_id"],
                "original": {
                    "row_number": collision["original_row_number"],
                    "query_id": collision["original_query_id"],
                    "passage_index": collision["original_passage_index"],
                    "chunk_index": collision["original_chunk_index"],
                    "chunk_strategy": collision["original_chunk_strategy"],
                    "text_hash": collision["original_text_hash"],
                    "preview": collision["original_preview"],
                },
                "duplicate": {
                    "row_number": collision["duplicate_row_number"],
                    "query_id": collision["duplicate_query_id"],
                    "passage_index": collision["duplicate_passage_index"],
                    "chunk_index": collision["duplicate_chunk_index"],
                    "chunk_strategy": collision["duplicate_chunk_strategy"],
                    "text_hash": collision["duplicate_text_hash"],
                    "preview": collision["duplicate_preview"],
                },
                "same_logical_identity": same_identity,
                "same_content": same_content,
            },
        )
    print(f"ID_COLLISIONS_DIFFERENT_CONTENT={different_content}", flush=True)
    print(f"ID_COLLISIONS_IDENTICAL_CONTENT={identical_content}", flush=True)
    print(f"EXACT_DUPLICATE_CHUNKS={exact_duplicate_chunks}", flush=True)
    return len(rows), different_content, exact_duplicate_chunks


def _regeneration_summary(connection: sqlite3.Connection) -> tuple[int, int, int, int]:
    """Recover persisted ID-scan totals without reloading the Parquet corpus."""
    unique_ids = connection.execute("SELECT COUNT(*) FROM expected_ids").fetchone()[0]
    collisions = connection.execute("SELECT COUNT(*) FROM collisions").fetchone()[0]
    total_chunks = unique_ids + collisions
    different_content = connection.execute(
        "SELECT COUNT(*) FROM collisions WHERE original_text_hash != duplicate_text_hash"
    ).fetchone()[0]
    return total_chunks, unique_ids, collisions, different_content


def _print_duplicate_query_report(connection: sqlite3.Connection, detail_limit: int) -> None:
    """Report duplicate raw query IDs, which are omitted by the current ID tuple."""
    duplicate_query_ids = connection.execute(
        "SELECT COUNT(*) FROM query_rows WHERE row_count > 1"
    ).fetchone()[0]
    affected_rows = connection.execute(
        "SELECT COALESCE(SUM(row_count - 1), 0) FROM query_rows WHERE row_count > 1"
    ).fetchone()[0]
    samples = connection.execute(
        "SELECT query_id, row_count FROM query_rows WHERE row_count > 1 ORDER BY query_id LIMIT ?",
        (detail_limit,),
    ).fetchall()
    print(f"DUPLICATE_QUERY_ID_COUNT={duplicate_query_ids}", flush=True)
    print(f"DUPLICATE_QUERY_ID_AFFECTED_ROWS={affected_rows}", flush=True)
    _print_json("DUPLICATE_QUERY_ID_SAMPLES", [{"query_id": row[0], "row_count": row[1]} for row in samples])


def _print_collision_root_summary(
    connection: sqlite3.Connection, different_content: int, identical_content: int
) -> None:
    """State the observed duplicate-row root cause without changing the ID scheme."""
    duplicate_ids = [
        row[0]
        for row in connection.execute(
            "SELECT query_id FROM query_rows WHERE row_count > 1 ORDER BY query_id"
        )
    ]
    if duplicate_ids == ["177416"]:
        print("COLLISION_ROOT_CAUSE=duplicate_query_id_177416", flush=True)
    else:
        _print_json("COLLISION_ROOT_CAUSE_DUPLICATE_QUERY_IDS", duplicate_ids)
    print(f"DIFFERENT_CONTENT_OVERWRITES={different_content}", flush=True)
    print(f"IDENTICAL_DUPLICATES={identical_content}", flush=True)


def _payload_errors(payload: Mapping[str, Any], target_lang: str) -> list[str]:
    """Return structural payload errors without mutating the stored payload."""
    errors = [field for field in REQUIRED_PAYLOAD_FIELDS if field not in payload]
    if "target_lang" in payload and payload.get("target_lang") != target_lang:
        errors.append(f"target_lang={payload.get('target_lang')!r}")
    if "text" in payload and not _normalize_text(payload.get("text")):
        errors.append("empty_text")
    return errors


def _qdrant_id_scroll(
    store: VectorStore,
    connection: sqlite3.Connection,
    expected_unique_ids: int,
    collision_count: int,
) -> tuple[int, int | None]:
    """Stream only point IDs into SQLite; payloads and vectors stay disabled."""
    collection = store.client.get_collection(store.collection_name)
    vectors = collection.config.params.vectors
    vector_dimension = getattr(vectors, "size", None)
    offset: Any = None
    actual_count = 0
    started_at = time.perf_counter()
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=512,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points:
            actual_count += 1
            connection.execute(
                "INSERT OR IGNORE INTO actual_ids(point_id) VALUES (?)", (str(point.id),)
            )
        if actual_count and actual_count % 10_000 == 0:
            connection.commit()
            _print_progress(
                "QDRANT_ID_SCROLL_PROGRESS",
                rows=actual_count,
                chunks=0,
                unique_ids=expected_unique_ids,
                collisions=collision_count,
                started_at=started_at,
            )
        if offset is None:
            break
    connection.commit()
    return actual_count, vector_dimension


def _local_collection_storage_path(settings: QdrantSettings) -> Path:
    """Resolve the current qdrant-client local collection SQLite file read-only."""
    return settings.path / "collection" / settings.collection_name / "storage.sqlite"


def _decode_local_storage_point_id(encoded_id: str) -> str:
    """Decode only the known UUID-string key format without unpickling point data."""
    raw = base64.b64decode(encoded_id, validate=True)
    match = LOCAL_POINT_ID_PATTERN.search(raw)
    if match is None:
        raise ValueError("Unsupported local Qdrant point-ID encoding")
    point_id = match.group().decode("ascii")
    if base64.b64encode(pickle.dumps(point_id, protocol=4)).decode("ascii") != encoded_id:
        raise ValueError("Local Qdrant point ID did not pass the safe encoding check")
    return point_id


def _local_storage_vector_dimension(settings: QdrantSettings) -> int | None:
    """Read the local Qdrant collection dimension from metadata JSON only."""
    metadata_path = settings.path / "meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vectors = metadata["collections"][settings.collection_name]["vectors"]
        return int(vectors["size"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _local_storage_id_scan(
    settings: QdrantSettings,
    connection: sqlite3.Connection,
    expected_unique_ids: int,
    collision_count: int,
) -> tuple[int, int | None]:
    """Stream local Qdrant IDs from SQLite without starting embedded QdrantLocal.

    This reads only the ``points.id`` key column. It never unpickles point blobs,
    reads vectors, or writes to Qdrant's storage.
    """
    storage_path = _local_collection_storage_path(settings)
    if not storage_path.is_file():
        raise FileNotFoundError(f"Local Qdrant collection storage not found: {storage_path}")
    uri = f"file:{storage_path.resolve().as_posix()}?mode=ro"
    local_connection = sqlite3.connect(uri, uri=True)
    try:
        cursor = local_connection.execute("SELECT id FROM points")
        actual_count = 0
        started_at = time.perf_counter()
        while batch := cursor.fetchmany(512):
            for (encoded_id,) in batch:
                actual_count += 1
                connection.execute(
                    "INSERT OR IGNORE INTO actual_ids(point_id) VALUES (?)",
                    (_decode_local_storage_point_id(encoded_id),),
                )
            if actual_count % 10_000 < len(batch):
                connection.commit()
                _print_progress(
                    "QDRANT_LOCAL_STORAGE_PROGRESS",
                    rows=actual_count,
                    chunks=0,
                    unique_ids=expected_unique_ids,
                    collisions=collision_count,
                    started_at=started_at,
                )
        connection.commit()
        return actual_count, _local_storage_vector_dimension(settings)
    finally:
        local_connection.close()


def _create_remote_verifier_store(
    settings: QdrantSettings, timeout_seconds: float
) -> VectorStore:
    """Create a bounded-time server client exclusively for verifier read operations."""
    if not settings.url:
        raise ValueError("QDRANT_URL or --qdrant-url is required for server-mode verification")
    client = QdrantClient(
        url=settings.url,
        api_key=settings.api_key,
        timeout=timeout_seconds,
        check_compatibility=False,
    )
    return VectorStore(settings, client=client)


def _resolve_verifier_settings(
    configured_settings: QdrantSettings,
    *,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    collection_name: str,
) -> QdrantSettings:
    """Apply secret-safe CLI overrides to environment-derived Qdrant settings."""
    resolved_url = (
        qdrant_url
        if qdrant_url is not None
        else configured_settings.url or os.getenv("QDRANT_URL")
    )
    resolved_api_key = (
        qdrant_api_key
        if qdrant_api_key is not None
        else configured_settings.api_key or os.getenv("QDRANT_API_KEY")
    )
    return replace(
        configured_settings,
        mode="remote" if resolved_url else configured_settings.mode,
        url=resolved_url,
        api_key=resolved_api_key,
        collection_name=collection_name,
    )


def _payload_integrity_sample(
    store: VectorStore,
    connection: sqlite3.Connection,
    target_lang: str,
    sample_size: int,
) -> tuple[list[dict[str, object]], int]:
    """Validate a small, evenly distributed payload sample without a corpus payload scan."""
    point_count = connection.execute("SELECT COUNT(*) FROM actual_ids").fetchone()[0]
    if not point_count:
        return [], 0
    offsets = sorted({round(index * (point_count - 1) / max(sample_size - 1, 1)) for index in range(sample_size)})
    sample_ids = [
        connection.execute(
            "SELECT point_id FROM actual_ids ORDER BY point_id LIMIT 1 OFFSET ?", (offset,)
        ).fetchone()[0]
        for offset in offsets
    ]
    points = store.client.retrieve(
        collection_name=store.collection_name,
        ids=sample_ids,
        with_payload=True,
        with_vectors=False,
    )
    malformed: list[dict[str, object]] = []
    for point in points:
        errors = _payload_errors(dict(point.payload or {}), target_lang)
        if errors:
            malformed.append({"point_id": str(point.id), "errors": errors})
    return malformed, len(points)


def _id_set_report(connection: sqlite3.Connection, detail_limit: int) -> tuple[int, int, int, list[dict[str, object]], list[str]]:
    """Summarize expected IDs versus actual IDs after the read-only Qdrant scan."""
    expected_unique = connection.execute("SELECT COUNT(*) FROM expected_ids").fetchone()[0]
    missing_rows = connection.execute(
        """
        SELECT point_id, row_number, query_id, passage_index, chunk_index,
               chunk_strategy, text_hash, preview
        FROM expected_ids
        WHERE NOT EXISTS (SELECT 1 FROM actual_ids WHERE actual_ids.point_id = expected_ids.point_id)
        ORDER BY point_id LIMIT ?
        """,
        (detail_limit,),
    ).fetchall()
    unexpected = [
        row[0]
        for row in connection.execute(
            """
            SELECT point_id FROM actual_ids
            WHERE NOT EXISTS (SELECT 1 FROM expected_ids WHERE expected_ids.point_id = actual_ids.point_id)
            ORDER BY point_id LIMIT ?
            """,
            (detail_limit,),
        )
    ]
    missing = [
        {
            "point_id": row[0],
            "row_number": row[1],
            "query_id": row[2],
            "passage_index": row[3],
            "chunk_index": row[4],
            "chunk_strategy": row[5],
            "text_hash": row[6],
            "preview": row[7],
        }
        for row in missing_rows
    ]
    missing_count = connection.execute(
        """
        SELECT COUNT(*) FROM expected_ids
        WHERE NOT EXISTS (SELECT 1 FROM actual_ids WHERE actual_ids.point_id = expected_ids.point_id)
        """
    ).fetchone()[0]
    unexpected_count = connection.execute(
        """
        SELECT COUNT(*) FROM actual_ids
        WHERE NOT EXISTS (SELECT 1 FROM expected_ids WHERE expected_ids.point_id = actual_ids.point_id)
        """
    ).fetchone()[0]
    print(f"EXPECTED_UNIQUE_IDS={expected_unique}", flush=True)
    print(f"MISSING_EXPECTED_IDS={missing_count}", flush=True)
    print(f"UNEXPECTED_QDRANT_IDS={unexpected_count}", flush=True)
    _print_json("MISSING_EXPECTED_ID_DETAILS", missing)
    _print_json("UNEXPECTED_QDRANT_ID_DETAILS", unexpected)
    return expected_unique, missing_count, unexpected_count, missing, unexpected


def _scroll_query_points(store: VectorStore, query_id: object) -> list[Any]:
    """Read all existing points for one query ID using a read-only payload filter."""
    points_for_query: list[Any] = []
    offset: Any = None
    query_filter = Filter(must=[FieldCondition(key="query_id", match=MatchValue(value=query_id))])
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points_for_query.extend(points)
        if offset is None:
            return points_for_query


def _is_relevant(value: object) -> bool:
    """Interpret MSMARCO-XI's relevance label without filtering any documents."""
    return value is True or value == 1 or value == "1"


def _rank_for_query_id(results: Iterable[Any], query_id: str) -> int | None:
    """Find an expected query ID inside already-requested ranked results."""
    for rank, result in enumerate(results, start=1):
        metadata = getattr(result, "metadata", {})
        if str(metadata.get("query_id")) == query_id:
            return rank
    return None


def _print_ranked(label: str, results: Iterable[Any], include_scores: bool) -> None:
    """Print the requested top-20 rank/provenance rows without raw vectors."""
    print(f"{label}_TOP_20", flush=True)
    for rank, result in enumerate(islice(results, 20), start=1):
        metadata = getattr(result, "metadata", {})
        row: dict[str, object] = {
            "rank": rank,
            "query_id": metadata.get("query_id"),
            "passage_index": metadata.get("passage_index"),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_strategy": metadata.get("chunk_strategy"),
        }
        if include_scores:
            row["semantic_score"] = getattr(result, "score", None)
        if hasattr(result, "semantic_score"):
            row.update(
                {
                    "semantic_score": result.semantic_score,
                    "bm25_score": result.bm25_score,
                    "fused_score": result.fused_score,
                }
            )
        _print_json(label, row)


def _fuse_diagnostic_rankings(
    semantic: list[Any], lexical: list[AuditLexicalMatch], top_k: int, rrf_k: int = 60
) -> list[AuditHybridMatch]:
    """Apply the production RRF formula to the two bounded diagnostic rankings."""
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    for rank, result in enumerate(semantic, start=1):
        metadata = dict(result.metadata)
        provenance = tuple(str(metadata.get(key, "")) for key in ("query_id", "passage_index", "chunk_index"))
        entry = candidates.setdefault(
            provenance,
            {"metadata": metadata, "semantic_score": None, "bm25_score": None, "fused_score": 0.0},
        )
        entry["semantic_score"] = result.score
        entry["fused_score"] = float(entry["fused_score"]) + 1 / (rrf_k + rank)
    for match in lexical:
        metadata = dict(match.metadata)
        provenance = tuple(str(metadata.get(key, "")) for key in ("query_id", "passage_index", "chunk_index"))
        entry = candidates.setdefault(
            provenance,
            {"metadata": metadata, "semantic_score": None, "bm25_score": None, "fused_score": 0.0},
        )
        entry["bm25_score"] = match.score
        entry["fused_score"] = float(entry["fused_score"]) + 1 / (rrf_k + match.rank)
    ordered = sorted(
        candidates.values(),
        key=lambda entry: (
            -float(entry["fused_score"]),
            tuple(str(entry["metadata"].get(key, "")) for key in ("query_id", "passage_index", "chunk_index")),
        ),
    )
    return [
        AuditHybridMatch(
            rank=rank,
            metadata=dict(entry["metadata"]),
            semantic_score=entry["semantic_score"],
            bm25_score=entry["bm25_score"],
            fused_score=float(entry["fused_score"]),
        )
        for rank, entry in enumerate(ordered[:top_k], start=1)
    ]


def _failed_query_report(
    failed_record: Mapping[str, Any] | None,
    store: VectorStore,
    language: str,
    top_rank_limit: int,
    lexical_audit: QueryScopedBM25Audit,
) -> str:
    """Separate absent/corrupt evidence from an ordinary retrieval-quality miss."""
    source_stage = _stage_start("query_430672_source_inspection")
    if failed_record is None:
        _stage_end("query_430672_source_inspection", source_stage)
        print("FAILED_QUERY_CLASSIFICATION=INDEXING_BUG", flush=True)
        print("FAILED_QUERY_REASON=query_id 430672 was not found in the local Parquet stream", flush=True)
        return "INDEXING_BUG"

    raw_query = _normalize_text(failed_record.get("query"))
    passages = failed_record.get("passages")
    if not isinstance(passages, Mapping):
        _stage_end("query_430672_source_inspection", source_stage)
        print("FAILED_QUERY_CLASSIFICATION=AMBIGUOUS_GROUND_TRUTH", flush=True)
        print("FAILED_QUERY_REASON=source row has no valid passages mapping", flush=True)
        return "AMBIGUOUS_GROUND_TRUTH"
    translated = passages.get("Translated_passages")
    selected = passages.get("is_selected")
    if not isinstance(translated, list) or not isinstance(selected, list) or len(translated) != len(selected):
        _stage_end("query_430672_source_inspection", source_stage)
        print("FAILED_QUERY_CLASSIFICATION=AMBIGUOUS_GROUND_TRUTH", flush=True)
        print("FAILED_QUERY_REASON=source passage/relevance lists are malformed", flush=True)
        return "AMBIGUOUS_GROUND_TRUTH"

    _print_json(
        "FAILED_QUERY_SOURCE_ROW",
        {
            "query_id": failed_record.get("query_id"),
            "query": raw_query,
            "query_type": failed_record.get("query_type"),
            "answer": failed_record.get("Answer"),
            "source_lang": failed_record.get("source_lang"),
            "target_lang": failed_record.get("target_lang"),
        },
    )
    for passage_index, (text, label) in enumerate(zip(translated, selected, strict=True)):
        _print_json(
            "FAILED_QUERY_DATASET_PASSAGE",
            {
                "passage_index": passage_index,
                "is_selected": label,
                "text": _normalize_text(text),
            },
        )

    points = _scroll_query_points(store, failed_record.get("query_id"))
    print(f"QUERY_430672_PRESENT={bool(points)}", flush=True)
    print(f"QUERY_430672_POINT_COUNT={len(points)}", flush=True)
    print(f"FAILED_QUERY_QDRANT_POINT_COUNT={len(points)}", flush=True)
    qdrant_passage_indices: set[int] = set()
    for point in points:
        payload = dict(point.payload or {})
        passage_index = payload.get("passage_index")
        if isinstance(passage_index, int):
            qdrant_passage_indices.add(passage_index)
        _print_json(
            "FAILED_QUERY_QDRANT_CHUNK",
            {
                "point_id": str(point.id),
                "passage_index": passage_index,
                "chunk_index": payload.get("chunk_index"),
                "chunk_strategy": payload.get("chunk_strategy"),
                "is_selected": payload.get("is_selected"),
                "text": _normalize_text(payload.get("text")),
            },
        )
    relevant_indices = {
        index
        for index, label in enumerate(selected)
        if _is_relevant(label) and _normalize_text(translated[index])
    }
    missing_relevant_indices = sorted(relevant_indices - qdrant_passage_indices)
    _print_json("FAILED_QUERY_RELEVANT_PASSAGE_INDICES", sorted(relevant_indices))
    _print_json("FAILED_QUERY_MISSING_RELEVANT_PASSAGE_INDICES", missing_relevant_indices)
    print(f"EXPECTED_RELEVANT_PASSAGES_PRESENT={not missing_relevant_indices}", flush=True)
    _stage_end("query_430672_source_inspection", source_stage)

    if not raw_query:
        print("FAILED_QUERY_CLASSIFICATION=AMBIGUOUS_GROUND_TRUTH", flush=True)
        print("FAILED_QUERY_REASON=source row query is empty", flush=True)
        return "AMBIGUOUS_GROUND_TRUTH"
    if raw_query != FAILED_QUERY:
        print("FAILED_QUERY_CLASSIFICATION=SMOKE_TEST_EXPECTATION_BUG", flush=True)
        print("FAILED_QUERY_REASON=source query text differs from the smoke-test query", flush=True)
        return "SMOKE_TEST_EXPECTATION_BUG"
    if not points or missing_relevant_indices:
        print("FAILED_QUERY_CLASSIFICATION=INDEXING_BUG", flush=True)
        print("FAILED_QUERY_REASON=expected relevant provenance is absent from Qdrant", flush=True)
        return "INDEXING_BUG"

    target_lang = get_qdrant_target_lang(language)
    semantic_stage = _stage_start("query_430672_semantic")
    print("QUERY_EMBEDDINGS_CREATED=1", flush=True)
    embedder = E5Embedder()
    retriever = Retriever(embedder, store)
    semantic = retriever.retrieve(raw_query, top_k=top_rank_limit, target_lang=target_lang)
    _print_ranked("SEMANTIC", semantic, include_scores=True)
    _stage_end("query_430672_semantic", semantic_stage)

    bm25_stage = _stage_start("query_430672_bm25")
    lexical = lexical_audit.search(top_k=top_rank_limit)
    print("BM25_TOP_20", flush=True)
    for match in lexical[:20]:
        _print_json(
            "BM25",
            {
                "rank": match.rank,
                "query_id": match.metadata.get("query_id"),
                "passage_index": match.metadata.get("passage_index"),
                "chunk_index": match.metadata.get("chunk_index"),
                "chunk_strategy": match.metadata.get("chunk_strategy"),
                "bm25_score": match.score,
            },
        )
    _stage_end("query_430672_bm25", bm25_stage)

    hybrid_stage = _stage_start("query_430672_hybrid")
    hybrid = _fuse_diagnostic_rankings(semantic, lexical, top_k=top_rank_limit)
    _print_ranked("HYBRID", hybrid, include_scores=False)
    ranks = {
        "semantic": _rank_for_query_id(semantic, FAILED_QUERY_ID),
        "bm25": next(
            (
                match.rank
                for match in lexical
                if str(match.metadata.get("query_id")) == FAILED_QUERY_ID
            ),
            None,
        ),
        "hybrid": _rank_for_query_id(hybrid, FAILED_QUERY_ID),
    }
    _print_json(
        "FAILED_QUERY_RANKS",
        {
            key: (rank if rank is not None else f"not found in top-{top_rank_limit}")
            for key, rank in ranks.items()
        },
    )
    _stage_end("query_430672_hybrid", hybrid_stage)
    print("FAILED_QUERY_CLASSIFICATION=RETRIEVAL_MISS", flush=True)
    print("FAILED_QUERY_REASON=relevant source evidence is present; the rank report identifies a retrieval-quality result rather than corruption", flush=True)
    return "RETRIEVAL_MISS"


def _final_status(
    *,
    total_chunks: int,
    expected_unique: int,
    actual_count: int,
    missing_expected: int,
    unexpected_count: int,
    collision_different_content: int,
    payload_errors: int,
    vector_dimension: int | None,
    failed_query_classification: str,
) -> str:
    """Make the Cloud structural decision from observed target integrity facts."""
    unexplained_missing = max(0, expected_unique - actual_count)
    print(f"UNEXPLAINED_MISSING_POINTS={unexplained_missing}", flush=True)
    print(f"KNOWN_SOURCE_DIFFERENT_CONTENT_OVERWRITES={collision_different_content}", flush=True)
    safe = (
        missing_expected == 0
        and unexpected_count == 0
        and payload_errors == 0
        and vector_dimension == 768
        and failed_query_classification != "INDEXING_BUG"
    )
    status = "INDEX_SAFE_FOR_BENCHMARKING" if safe else "INDEX_REQUIRES_FIX"
    print(f"FINAL_STRUCTURAL_STATUS={status}", flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--collection", default=FULL_HINDI_COLLECTION)
    parser.add_argument("--detail-limit", type=int, default=20)
    parser.add_argument("--retrieval-rank-limit", type=int, default=20)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Development memory check: regenerate only this many rows and skip collection comparison.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=10,
        help="Emit bounded-scan progress after this many loader batches.",
    )
    parser.add_argument(
        "--verification-db",
        type=Path,
        help="Persistent SQLite verifier state. Required for --post-scan-only.",
    )
    parser.add_argument("--audit-db", dest="verification_db", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--post-scan-only", action="store_true", help="Reuse a completed verification DB; skip loader, chunking, and ID regeneration.")
    parser.add_argument("--qdrant-url", help="Read-only Qdrant Cloud URL; overrides QDRANT_URL.")
    parser.add_argument("--qdrant-api-key", help="Qdrant Cloud API key; overrides QDRANT_API_KEY.")
    parser.add_argument(
        "--qdrant-init-timeout-seconds",
        type=float,
        default=15.0,
        help="Server-client timeout; embedded local initialization is intentionally not attempted.",
    )
    args = parser.parse_args()
    numeric_options = (
        args.batch_size,
        args.max_tokens,
        args.detail_limit,
        args.retrieval_rank_limit,
        args.progress_every_batches,
    )
    if (
        min(numeric_options) < 1
        or args.qdrant_init_timeout_seconds <= 0
        or (args.max_rows is not None and args.max_rows < 1)
    ):
        parser.error("all numeric options must be at least 1")
    if args.max_tokens != 256:
        parser.error("--max-tokens must remain 256 to verify the completed full index")
    if not args.collection.strip():
        parser.error("--collection must be non-empty")
    if args.collection != FULL_HINDI_COLLECTION:
        parser.error(f"This verifier is restricted to {FULL_HINDI_COLLECTION}")
    if args.post_scan_only and args.max_rows is not None:
        parser.error("--post-scan-only cannot be combined with --max-rows")
    if args.post_scan_only and args.verification_db is None:
        parser.error("--post-scan-only requires --verification-db")
    if not args.post_scan_only and args.verification_db is not None and args.verification_db.exists():
        parser.error("--verification-db already exists; use --post-scan-only to reuse it")

    target_lang = get_qdrant_target_lang(args.language)
    settings = _resolve_verifier_settings(
        QdrantSettings.from_environment(),
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
    )
    print(f"COLLECTION={args.collection}", flush=True)
    print(f"LANGUAGE={args.language}", flush=True)
    print(f"TARGET_LANG={target_lang}", flush=True)
    print(f"BATCH_SIZE={args.batch_size}", flush=True)
    print(f"MAX_TOKENS={args.max_tokens}", flush=True)
    print(f"MAX_ROWS={args.max_rows if args.max_rows is not None else 'all'}", flush=True)
    print("EMBEDDINGS_CREATED=0", flush=True)
    print("DOCUMENT_EMBEDDINGS_CREATED=0", flush=True)
    print("QDRANT_WRITES=0", flush=True)
    print("QDRANT_DELETES=0", flush=True)
    print("COLLECTION_RESET=False", flush=True)
    print(f"QDRANT_MODE={settings.mode}", flush=True)
    if settings.mode == "local":
        print(f"QDRANT_PATH={settings.path}", flush=True)
    else:
        print(f"QDRANT_URL={settings.url}", flush=True)
        print(f"QDRANT_API_KEY_PRESENT={str(bool(settings.api_key)).lower()}", flush=True)
    print(f"QDRANT_INIT_TIMEOUT_SECONDS={args.qdrant_init_timeout_seconds}", flush=True)

    with tempfile.TemporaryDirectory(prefix="msmarco_xi_full_audit_") as temporary_directory:
        audit_path = args.verification_db or Path(temporary_directory) / "expected_ids.sqlite3"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_open_started = time.perf_counter()
        print("SQLITE_OPEN_START", flush=True)
        connection = _open_verification_database(audit_path, reuse=args.post_scan_only)
        print(f"SQLITE_OPEN_END elapsed_ms={(time.perf_counter() - sqlite_open_started) * 1000:.1f}", flush=True)
        store: VectorStore | None = None
        try:
            if args.post_scan_only:
                post_scan_init_started = time.perf_counter()
                print("POST_SCAN_INIT_START", flush=True)
                total_chunks, expected_unique, duplicate_occurrences, different_content = _regeneration_summary(connection)
                failed_record = _load_failed_query_source(connection)
                lexical_audit = QueryScopedBM25Audit.load(connection, FAILED_QUERY)
                if failed_record is None or lexical_audit is None:
                    raise RuntimeError(
                        "Verification DB lacks query 430672 source/BM25 state; run one full scan with --verification-db first."
                    )
                print("ID_REGENERATION_SKIPPED=true", flush=True)
            else:
                lexical_audit = QueryScopedBM25Audit(connection, FAILED_QUERY)
                regeneration_stage = _stage_start("id_regeneration")
                total_chunks, duplicate_occurrences, failed_record = _iter_regenerated_chunks(
                    args.language,
                    args.split,
                    args.batch_size,
                    args.max_tokens,
                    connection,
                    args.max_rows,
                    args.batch_size * args.progress_every_batches,
                    lexical_audit,
                    target_lang,
                )
                _stage_end("id_regeneration", regeneration_stage)
                expected_unique = connection.execute("SELECT COUNT(*) FROM expected_ids").fetchone()[0]
                _, different_content, identical_content = _print_collision_report(connection)
            print(f"TOTAL_CHUNKS_GENERATED={total_chunks}", flush=True)
            print(f"TOTAL_UNIQUE_POINT_IDS={expected_unique}", flush=True)
            print(f"DUPLICATE_POINT_ID_COUNT={duplicate_occurrences}", flush=True)
            if args.post_scan_only:
                _, different_content, identical_content = _print_collision_report(connection)
            _print_duplicate_query_report(connection, args.detail_limit)
            _print_collision_root_summary(connection, different_content, identical_content)
            if args.post_scan_only:
                print(
                    f"POST_SCAN_INIT_END elapsed_ms={(time.perf_counter() - post_scan_init_started) * 1000:.1f}",
                    flush=True,
                )

            if args.max_rows is not None:
                print("STRUCTURAL_COMPARISON_SKIPPED_FOR_PARTIAL_SCAN=true", flush=True)
                print("FINAL_STRUCTURAL_STATUS=NOT_EVALUATED_PARTIAL_SCAN", flush=True)
                return

            if settings.mode == "remote":
                qdrant_client_init_started = time.perf_counter()
                print("QDRANT_CLIENT_INIT_START", flush=True)
                try:
                    store = _create_remote_verifier_store(
                        settings, args.qdrant_init_timeout_seconds
                    )
                except Exception as error:
                    print("QDRANT_REMOTE_REACHABLE=false", flush=True)
                    print("QDRANT_CLIENT_INIT_TIMEOUT", flush=True)
                    raise RuntimeError(
                        "Could not initialize the read-only Qdrant server client within the configured timeout."
                    ) from error
                print(
                    f"QDRANT_CLIENT_INIT_END elapsed_ms={(time.perf_counter() - qdrant_client_init_started) * 1000:.1f}",
                    flush=True,
                )
                print("TRACE post_scan_step=collection_exists", flush=True)
                try:
                    collection_exists = store.client.collection_exists(store.collection_name)
                except Exception as error:
                    print("QDRANT_REMOTE_REACHABLE=false", flush=True)
                    print("QDRANT_CLIENT_INIT_TIMEOUT", flush=True)
                    print(
                        f"QDRANT_CLIENT_INIT_ERROR_TYPE={type(error).__name__}", flush=True
                    )
                    raise RuntimeError(
                        "Qdrant server did not answer the collection check within the configured timeout."
                    ) from error
                print("QDRANT_REMOTE_REACHABLE=true", flush=True)
                print(f"TARGET_COLLECTION_EXISTS={str(collection_exists).lower()}", flush=True)
                if not collection_exists:
                    raise RuntimeError(f"Qdrant collection does not exist: {store.collection_name}")
                print("TRACE post_scan_step=collection_exists_done", flush=True)
            else:
                print("QDRANT_CLIENT_INIT_SKIPPED=local_storage_read_only_id_audit", flush=True)
            actual_ids_reset_started = time.perf_counter()
            print("ACTUAL_IDS_RESET_START", flush=True)
            _reset_post_scan_tables(connection)
            print(
                f"ACTUAL_IDS_RESET_END elapsed_ms={(time.perf_counter() - actual_ids_reset_started) * 1000:.1f}",
                flush=True,
            )
            qdrant_scroll_stage = _stage_start("qdrant_id_scroll")
            if settings.mode == "remote":
                actual_count, vector_dimension = _qdrant_id_scroll(
                    store, connection, expected_unique, duplicate_occurrences
                )
            else:
                actual_count, vector_dimension = _local_storage_id_scan(
                    settings, connection, expected_unique, duplicate_occurrences
                )
            _stage_end("qdrant_id_scroll", qdrant_scroll_stage)
            print(f"ACTUAL_QDRANT_IDS={actual_count}", flush=True)
            print(f"QDRANT_VECTOR_DIMENSION={vector_dimension}", flush=True)
            comparison_stage = _stage_start("sqlite_comparison")
            expected_unique, missing_expected, unexpected_count, _, _ = _id_set_report(
                connection, args.detail_limit
            )
            _stage_end("sqlite_comparison", comparison_stage)
            if store is None:
                print("PAYLOAD_INTEGRITY_SKIPPED=requires_qdrant_server_url", flush=True)
                print("QUERY_430672_ANALYSIS_SKIPPED=requires_qdrant_server_url", flush=True)
                print("FINAL_STRUCTURAL_STATUS=NOT_EVALUATED_QDRANT_SERVER_REQUIRED", flush=True)
                return
            payload_stage = _stage_start("payload_integrity")
            malformed_payloads, payload_sample_count = _payload_integrity_sample(
                store, connection, target_lang, sample_size=args.detail_limit
            )
            payload_error_count = len(malformed_payloads)
            print(f"PAYLOAD_SAMPLE_COUNT={payload_sample_count}", flush=True)
            print(f"MALFORMED_PAYLOAD_COUNT={payload_error_count}", flush=True)
            _print_json("MALFORMED_PAYLOAD_DETAILS", malformed_payloads)
            _stage_end("payload_integrity", payload_stage)
            classification = _failed_query_report(
                failed_record,
                store,
                args.language,
                args.retrieval_rank_limit,
                lexical_audit,
            )
            status = _final_status(
                total_chunks=total_chunks,
                expected_unique=expected_unique,
                actual_count=actual_count,
                missing_expected=missing_expected,
                unexpected_count=unexpected_count,
                collision_different_content=different_content,
                payload_errors=payload_error_count,
                vector_dimension=vector_dimension,
                failed_query_classification=classification,
            )
        finally:
            if store is not None:
                store.close()
            connection.close()


if __name__ == "__main__":
    main()
