"""Safely copy an embedded local Qdrant collection to a Qdrant server.

This is a one-off operational migration tool.  It never creates embeddings and
never writes to the source at ``data/qdrant``.  The source is read directly from
the local client's SQLite point store because opening the full collection via
``QdrantClient(path=...)`` can require a large, blocking local recovery.

The direct source reader is deliberately opt-in: this storage uses Python
pickles, so only run it against the trusted, locally generated index described
by this project.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import sqlite3
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = BACKEND_ROOT / "data" / "qdrant"
DEFAULT_SOURCE_COLLECTION = "msmarco_xi_hindi_full"
DEFAULT_CHECKPOINT = BACKEND_ROOT / "tmp" / "qdrant_server_migration_checkpoint.json"
VECTOR_SIZE = 768


class MigrationError(RuntimeError):
    """Raised when a migration prerequisite or safety check fails."""


class RestrictedPointUnpickler(pickle.Unpickler):
    """Deserialize only the PointStruct class used by this trusted local store."""

    def find_class(self, module: str, name: str) -> type[PointStruct]:
        if module == "qdrant_client.http.models.models" and name == "PointStruct":
            return PointStruct
        raise pickle.UnpicklingError(f"Refusing unexpected pickle global: {module}.{name}")


@dataclass(frozen=True)
class SourcePoint:
    """Validated vector and payload copied from one local embedded point."""

    point_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class MigrationCheckpoint:
    """Durable progress state written only after a successful target upsert."""

    source_path: str
    source_collection: str
    qdrant_url: str
    target_collection: str
    last_rowid: int
    points_upserted: int


def _storage_db_path(source_path: Path, collection_name: str) -> Path:
    """Return the source-only SQLite point database for one local collection."""
    return source_path / "collection" / collection_name / "storage.sqlite"


def _open_read_only_sqlite(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise MigrationError(f"Local Qdrant point store was not found: {database_path}")
    return sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)


def source_point_count(database_path: Path) -> int:
    """Count source points without opening embedded Qdrant or reading point blobs."""
    connection = _open_read_only_sqlite(database_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM points").fetchone()
        return int(row[0])
    finally:
        connection.close()


def _deserialize_source_point(blob: bytes) -> SourcePoint:
    """Decode and validate one trusted local PointStruct pickle.

    The source files are created by this application's existing local Qdrant
    client.  The restricted unpickler rejects every other global class; it is
    still intentionally guarded by ``--trust-local-pickle-source`` at the CLI.
    """
    try:
        point = RestrictedPointUnpickler(io.BytesIO(blob)).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError) as error:
        raise MigrationError(f"Could not decode a local Qdrant point: {error}") from error

    try:
        point_id = str(uuid.UUID(str(point.id)))
    except (AttributeError, ValueError, TypeError) as error:
        raise MigrationError("Local point has an invalid UUID identifier") from error

    vector = point.vector
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise MigrationError(f"Point {point_id} has an unsupported non-dense vector")
    if len(vector) != VECTOR_SIZE:
        raise MigrationError(f"Point {point_id} has vector size {len(vector)}, expected {VECTOR_SIZE}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in vector):
        raise MigrationError(f"Point {point_id} contains a non-finite vector value")

    payload = point.payload
    if not isinstance(payload, Mapping):
        raise MigrationError(f"Point {point_id} has a non-mapping payload")
    return SourcePoint(point_id=point_id, vector=[float(value) for value in vector], payload=dict(payload))


def iter_source_points(
    database_path: Path,
    *,
    after_rowid: int = 0,
    read_batch_size: int = 256,
) -> Iterator[tuple[int, SourcePoint]]:
    """Yield source points in bounded SQLite batches, without altering the source."""
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be at least 1")
    if after_rowid < 0:
        raise ValueError("after_rowid must not be negative")

    connection = _open_read_only_sqlite(database_path)
    try:
        cursor = connection.execute(
            "SELECT rowid, point FROM points WHERE rowid > ? ORDER BY rowid",
            (after_rowid,),
        )
        while rows := cursor.fetchmany(read_batch_size):
            for rowid, blob in rows:
                yield int(rowid), _deserialize_source_point(blob)
    finally:
        connection.close()


def iter_source_points_by_rowids(
    database_path: Path,
    rowids: Sequence[int],
    *,
    read_batch_size: int = 256,
) -> Iterator[tuple[int, SourcePoint]]:
    """Yield selected local points in bounded batches without source writes."""
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be at least 1")
    if any(rowid < 1 for rowid in rowids):
        raise ValueError("rowids must be positive")
    connection = _open_read_only_sqlite(database_path)
    try:
        for offset in range(0, len(rowids), read_batch_size):
            batch = list(rowids[offset : offset + read_batch_size])
            placeholders = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"SELECT rowid, point FROM points WHERE rowid IN ({placeholders}) ORDER BY rowid",
                batch,
            ).fetchall()
            if len(rows) != len(batch):
                raise MigrationError("One or more selected local Qdrant rowids no longer exist")
            for rowid, blob in rows:
                yield int(rowid), _deserialize_source_point(blob)
    finally:
        connection.close()


def _vector_config(collection: Any) -> Any:
    vectors = collection.config.params.vectors
    if isinstance(vectors, Mapping):
        raise MigrationError("Target collection uses named vectors; this migration requires one dense vector")
    return vectors


def _distance_name(distance: Any) -> str:
    return str(getattr(distance, "value", distance)).lower()


def ensure_target_collection(client: QdrantClient, target_collection: str, *, reset: bool = False) -> None:
    """Create or strictly validate the remote target; never reset by default."""
    exists = client.collection_exists(target_collection)
    if exists and reset:
        client.delete_collection(target_collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=target_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        return

    vectors = _vector_config(client.get_collection(target_collection))
    configured_size = int(vectors.size)
    configured_distance = _distance_name(vectors.distance)
    if configured_size != VECTOR_SIZE or configured_distance != "cosine":
        raise MigrationError(
            "Existing target collection configuration does not match source: "
            f"vector_size={configured_size}, distance={configured_distance}; "
            f"expected vector_size={VECTOR_SIZE}, distance=cosine. "
            "Choose a different target collection or explicitly use --reset-target."
        )


def _read_checkpoint(checkpoint_path: Path) -> MigrationCheckpoint:
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return MigrationCheckpoint(**raw)
    except (OSError, TypeError, ValueError) as error:
        raise MigrationError(f"Could not read migration checkpoint {checkpoint_path}: {error}") from error


def _write_checkpoint(checkpoint_path: Path, checkpoint: MigrationCheckpoint) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(asdict(checkpoint), indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(checkpoint_path)


def _checkpoint_matches(
    checkpoint: MigrationCheckpoint,
    *,
    source_path: Path,
    source_collection: str,
    qdrant_url: str,
    target_collection: str,
) -> bool:
    return (
        checkpoint.source_path == str(source_path.resolve())
        and checkpoint.source_collection == source_collection
        and checkpoint.qdrant_url == qdrant_url
        and checkpoint.target_collection == target_collection
    )


def _batch_to_target_points(batch: list[tuple[int, SourcePoint]]) -> list[PointStruct]:
    return [PointStruct(id=point.point_id, vector=point.vector, payload=point.payload) for _, point in batch]


def _target_count(client: QdrantClient, collection_name: str) -> int:
    return int(client.count(collection_name=collection_name, exact=True).count)


def resolve_remote_config(
    qdrant_url: str | None,
    qdrant_api_key: str | None,
) -> tuple[str | None, str | None]:
    """Resolve Cloud configuration with CLI values taking precedence over env."""
    return (
        qdrant_url if qdrant_url is not None else os.getenv("QDRANT_URL"),
        qdrant_api_key if qdrant_api_key is not None else os.getenv("QDRANT_API_KEY"),
    )


def _preflight_remote(
    client: QdrantClient,
    target_collection: str,
) -> bool:
    """Confirm the remote API is reachable before allowing target mutation."""
    # get_collections validates authentication/transport independently of a
    # collection name, then collection_exists provides the requested telemetry.
    client.get_collections()
    return bool(client.collection_exists(target_collection))


def _upsert_with_retry(
    client: QdrantClient,
    target_collection: str,
    points: list[PointStruct],
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    """Perform an acknowledged upsert with finite exponential retry backoff."""
    for attempt in range(max_retries + 1):
        try:
            client.upsert(collection_name=target_collection, points=points, wait=True)
            return
        except Exception as error:  # Transport/provider failures vary by client version.
            if attempt == max_retries:
                raise MigrationError(
                    f"Target upsert failed after {max_retries + 1} attempt(s): {type(error).__name__}: {error}"
                ) from error
            delay_seconds = retry_backoff_seconds * (2**attempt)
            print(
                f"UPSERT_RETRY attempt={attempt + 1} delay_seconds={delay_seconds:.2f} "
                f"error_type={type(error).__name__}",
                flush=True,
            )
            time.sleep(delay_seconds)


def migrate(
    *,
    source_path: Path,
    source_collection: str,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    target_collection: str,
    batch_size: int,
    timeout_seconds: float,
    checkpoint_path: Path,
    resume: bool,
    reset_target: bool,
    dry_run: bool,
    dry_run_remote_preflight: bool,
    max_points: int | None,
    count_every_batches: int,
    max_upsert_retries: int,
    retry_backoff_seconds: float,
) -> tuple[int, int]:
    """Copy points idempotently and return ``(source_read, points_upserted)``."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be at least 1 when supplied")
    if count_every_batches < 1:
        raise ValueError("count_every_batches must be at least 1")
    if max_upsert_retries < 0:
        raise ValueError("max_upsert_retries must not be negative")
    if retry_backoff_seconds <= 0:
        raise ValueError("retry_backoff_seconds must be greater than zero")

    database_path = _storage_db_path(source_path, source_collection)
    source_total = source_point_count(database_path)
    resolved_url, resolved_api_key = resolve_remote_config(qdrant_url, qdrant_api_key)
    print(f"SOURCE_PATH={source_path.resolve()}")
    print(f"SOURCE_COLLECTION={source_collection}")
    print(f"SOURCE_POINT_STORE={database_path}")
    print(f"SOURCE_POINT_COUNT={source_total}")
    print("QDRANT_MODE=remote")
    print(f"QDRANT_URL={resolved_url or 'not_configured'}")
    print(f"QDRANT_API_KEY_PRESENT={str(bool(resolved_api_key)).lower()}")
    print(f"TARGET_COLLECTION={target_collection}")
    print(f"VECTOR_SIZE={VECTOR_SIZE}")
    print("DISTANCE=cosine")
    print("DOCUMENT_EMBEDDINGS_CREATED=0")
    print(f"DRY_RUN={str(dry_run).lower()}")

    if dry_run:
        if dry_run_remote_preflight:
            if not resolved_url:
                raise MigrationError("--dry-run-remote-preflight requires --qdrant-url or QDRANT_URL")
            client: QdrantClient | None = None
            try:
                client = QdrantClient(url=resolved_url, api_key=resolved_api_key, timeout=timeout_seconds)
                target_exists = _preflight_remote(client, target_collection)
            except Exception as error:
                print("QDRANT_REMOTE_REACHABLE=false", flush=True)
                raise MigrationError(f"Remote Qdrant preflight failed: {type(error).__name__}: {error}") from error
            finally:
                if client is not None:
                    client.close()
            print("QDRANT_REMOTE_REACHABLE=true", flush=True)
            print(f"TARGET_COLLECTION_EXISTS={str(target_exists).lower()}", flush=True)
        print("DRY_RUN_COMPLETE=true (no target collection create/reset/upsert was made)")
        return 0, 0

    if not resolved_url:
        raise MigrationError("Qdrant Cloud URL is required: pass --qdrant-url or set QDRANT_URL")

    if reset_target and resume:
        raise MigrationError("--reset-target cannot be combined with --resume")

    after_rowid = 0
    points_upserted = 0
    if resume and checkpoint_path.exists():
        checkpoint = _read_checkpoint(checkpoint_path)
        if not _checkpoint_matches(
            checkpoint,
            source_path=source_path,
            source_collection=source_collection,
            qdrant_url=resolved_url,
            target_collection=target_collection,
        ):
            raise MigrationError(f"Checkpoint {checkpoint_path} belongs to a different migration configuration")
        after_rowid = checkpoint.last_rowid
        points_upserted = checkpoint.points_upserted
        print(f"RESUME_FROM_ROWID={after_rowid}")
        print(f"RESUME_UPSERTED_COUNT={points_upserted}")

    print("QDRANT_REMOTE_PREFLIGHT_START")
    connect_started = time.perf_counter()
    client: QdrantClient | None = None
    try:
        client = QdrantClient(url=resolved_url, api_key=resolved_api_key, timeout=timeout_seconds)
        target_exists = _preflight_remote(client, target_collection)
        print("QDRANT_REMOTE_REACHABLE=true", flush=True)
        print(f"TARGET_COLLECTION_EXISTS={str(target_exists).lower()}", flush=True)
        ensure_target_collection(client, target_collection, reset=reset_target)
        target_start_count = _target_count(client, target_collection)
    except Exception as error:  # Qdrant exposes several transport exception types.
        print("QDRANT_REMOTE_REACHABLE=false", flush=True)
        if client is not None:
            client.close()
        raise MigrationError(f"Could not connect to/prepare target Qdrant server {resolved_url}: {type(error).__name__}: {error}") from error
    print(f"QDRANT_REMOTE_PREFLIGHT_END elapsed_ms={(time.perf_counter() - connect_started) * 1000:.1f}")
    print(f"TARGET_POINT_COUNT_START={target_start_count}")

    source_read = 0
    batch_number = 0
    last_rowid = after_rowid
    migration_started = time.perf_counter()
    source_iterator = iter_source_points(database_path, after_rowid=after_rowid, read_batch_size=batch_size)
    try:
        while max_points is None or source_read < max_points:
            read_started = time.perf_counter()
            batch: list[tuple[int, SourcePoint]] = []
            while len(batch) < batch_size and (max_points is None or source_read < max_points):
                try:
                    batch.append(next(source_iterator))
                    source_read += 1
                except StopIteration:
                    break
            read_batch_ms = (time.perf_counter() - read_started) * 1000
            if not batch:
                break
            batch_number += 1
            upsert_started = time.perf_counter()
            _upsert_with_retry(
                client,
                target_collection,
                _batch_to_target_points(batch),
                max_retries=max_upsert_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            upsert_batch_ms = (time.perf_counter() - upsert_started) * 1000
            last_rowid = batch[-1][0]
            points_upserted += len(batch)
            _write_checkpoint(
                checkpoint_path,
                MigrationCheckpoint(
                    source_path=str(source_path.resolve()),
                    source_collection=source_collection,
                    qdrant_url=resolved_url,
                    target_collection=target_collection,
                    last_rowid=last_rowid,
                    points_upserted=points_upserted,
                ),
            )
            elapsed_time_s = time.perf_counter() - migration_started
            total_batch_ms = read_batch_ms + upsert_batch_ms
            points_per_second = points_upserted / elapsed_time_s if elapsed_time_s else 0.0
            print(
                f"BATCH={batch_number} POINTS_READ={source_read} POINTS_UPSERTED={points_upserted} "
                f"LAST_ROWID={last_rowid} READ_BATCH_MS={read_batch_ms:.1f} "
                f"UPSERT_BATCH_MS={upsert_batch_ms:.1f} TOTAL_BATCH_MS={total_batch_ms:.1f} "
                f"POINTS_PER_SECOND={points_per_second:.2f} ELAPSED_TIME_S={elapsed_time_s:.1f}",
                flush=True,
            )
            if batch_number % count_every_batches == 0:
                print(f"TARGET_POINT_COUNT_CURRENT={_target_count(client, target_collection)}")
        target_end_count = _target_count(client, target_collection)
    finally:
        if client is not None:
            client.close()

    print(f"TARGET_POINT_COUNT_END={target_end_count}")
    print(f"MIGRATION_SOURCE_READ={source_read}")
    print(f"MIGRATION_UPSERT_CALL_POINTS={points_upserted}")
    return source_read, points_upserted


def _parse_args() -> argparse.Namespace:
    load_dotenv(BACKEND_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Copy a trusted embedded local Qdrant collection to a Qdrant server.")
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--source-collection", default=DEFAULT_SOURCE_COLLECTION)
    parser.add_argument("--qdrant-url", help="Qdrant Cloud URL; overrides QDRANT_URL.")
    parser.add_argument("--qdrant-api-key", help="Qdrant Cloud API key; overrides QDRANT_API_KEY.")
    parser.add_argument("--target-collection", default=DEFAULT_SOURCE_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true", help="Resume from the last fully acknowledged target batch.")
    parser.add_argument("--reset-target", action="store_true", help="Explicitly delete and recreate only the target collection.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect the source count without connecting to or writing the server.")
    parser.add_argument(
        "--dry-run-remote-preflight",
        action="store_true",
        help="With --dry-run, explicitly test Cloud connectivity without creating or changing a collection.",
    )
    parser.add_argument("--max-points", type=int, help="Copy at most this many source points (development/server smoke test only).")
    parser.add_argument("--count-every-batches", type=int, default=20)
    parser.add_argument("--max-upsert-retries", type=int, default=3, help="Retries after the initial upsert attempt.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--trust-local-pickle-source",
        action="store_true",
        help="Required acknowledgement: source blobs are trusted local Qdrant PointStruct pickles.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.trust_local_pickle_source:
        raise MigrationError(
            "Refusing to deserialize the local source without --trust-local-pickle-source. "
            "Use only for this application's trusted local index."
        )
    if args.dry_run_remote_preflight and not args.dry_run:
        raise MigrationError("--dry-run-remote-preflight requires --dry-run")
    migrate(
        source_path=args.source_path,
        source_collection=args.source_collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        target_collection=args.target_collection,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        checkpoint_path=args.checkpoint_file,
        resume=args.resume,
        reset_target=args.reset_target,
        dry_run=args.dry_run,
        dry_run_remote_preflight=args.dry_run_remote_preflight,
        max_points=args.max_points,
        count_every_batches=args.count_every_batches,
        max_upsert_retries=args.max_upsert_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    except (MigrationError, ValueError) as error:
        print(f"MIGRATION_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
