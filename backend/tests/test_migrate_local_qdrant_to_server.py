"""Focused safety tests for the local-Qdrant-to-server migration utility."""

from __future__ import annotations

import importlib.util
import pickle
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "migrate_local_qdrant_to_server.py"
SPEC = importlib.util.spec_from_file_location("migrate_local_qdrant_to_server", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migration
SPEC.loader.exec_module(migration)


class FakeClient:
    def __init__(self, *, exists: bool = False, vector_size: int = 768, distance: str = "Cosine") -> None:
        self.exists = exists
        self.vector_size = vector_size
        self.distance = distance
        self.created = False
        self.deleted = False
        self.collection_checks = []
        self.closed = False

    def collection_exists(self, _: str) -> bool:
        self.collection_checks.append(_)
        return self.exists

    def get_collections(self) -> object:
        return SimpleNamespace(collections=[])

    def close(self) -> None:
        self.closed = True

    def create_collection(self, **_: object) -> None:
        self.created = True

    def delete_collection(self, _: str) -> None:
        self.deleted = True

    def get_collection(self, _: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.vector_size, distance=self.distance))
            )
        )


class MigrationToolTests(unittest.TestCase):
    @staticmethod
    def _migrate_options(**overrides: object) -> dict[str, object]:
        options: dict[str, object] = {
            "source_path": Path("unused"),
            "source_collection": "unused",
            "qdrant_url": "https://cloud.example",
            "qdrant_api_key": None,
            "target_collection": "unused",
            "batch_size": 1,
            "timeout_seconds": 30,
            "checkpoint_path": Path("unused.json"),
            "resume": False,
            "reset_target": False,
            "dry_run": True,
            "dry_run_remote_preflight": False,
            "max_points": None,
            "count_every_batches": 1,
            "max_upsert_retries": 3,
            "retry_backoff_seconds": 0.01,
        }
        options.update(overrides)
        return options

    def test_storage_path_is_separate_from_target_configuration(self) -> None:
        source = Path("C:/project/data/qdrant")
        self.assertEqual(
            migration._storage_db_path(source, "msmarco_xi_hindi_full"),
            source / "collection" / "msmarco_xi_hindi_full" / "storage.sqlite",
        )

    def test_source_count_and_iteration_use_read_only_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "storage.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE points (point BLOB)")
                connection.execute("INSERT INTO points(point) VALUES (?)", (b"first",))
                connection.execute("INSERT INTO points(point) VALUES (?)", (b"second",))
                connection.commit()
            finally:
                connection.close()

            self.assertEqual(migration.source_point_count(database), 2)
            with patch.object(migration, "_deserialize_source_point", side_effect=lambda blob: blob.decode()):
                self.assertEqual(
                    list(migration.iter_source_points(database, read_batch_size=1)),
                    [(1, "first"), (2, "second")],
                )

    def test_missing_source_file_has_clear_error(self) -> None:
        with self.assertRaisesRegex(migration.MigrationError, "was not found"):
            migration.source_point_count(Path("missing-storage.sqlite"))

    @unittest.skipUnless(
        migration.PointStruct.__module__ == "qdrant_client.http.models.models",
        "requires the installed qdrant-client PointStruct implementation",
    )
    def test_trusted_point_pickle_preserves_vector_payload_and_id(self) -> None:
        source_point = migration.PointStruct(
            id="043a46df-f3b0-5f3b-bb7e-de7fbd59de8e",
            vector=[0.25] * migration.VECTOR_SIZE,
            payload={"query_id": "232017", "text": "परीक्षण"},
        )
        parsed = migration._deserialize_source_point(pickle.dumps(source_point, protocol=4))
        self.assertEqual(parsed.point_id, "043a46df-f3b0-5f3b-bb7e-de7fbd59de8e")
        self.assertEqual(parsed.vector, [0.25] * migration.VECTOR_SIZE)
        self.assertEqual(parsed.payload["query_id"], "232017")

    def test_target_collection_is_created_with_expected_shape(self) -> None:
        client = FakeClient()
        migration.ensure_target_collection(client, "target")
        self.assertTrue(client.created)
        self.assertFalse(client.deleted)

    def test_existing_wrong_target_is_not_reset_implicitly(self) -> None:
        client = FakeClient(exists=True, vector_size=384)
        with self.assertRaisesRegex(migration.MigrationError, "does not match source"):
            migration.ensure_target_collection(client, "target")
        self.assertFalse(client.deleted)

    def test_reset_target_is_explicit(self) -> None:
        client = FakeClient(exists=True)
        migration.ensure_target_collection(client, "target", reset=True)
        self.assertTrue(client.deleted)
        self.assertTrue(client.created)

    def test_checkpoint_matching_prevents_cross_target_resume(self) -> None:
        checkpoint = migration.MigrationCheckpoint(
            source_path=str(Path("source").resolve()),
            source_collection="source_collection",
            qdrant_url="http://localhost:6333",
            target_collection="target_collection",
            last_rowid=23,
            points_upserted=23,
        )
        self.assertTrue(
            migration._checkpoint_matches(
                checkpoint,
                source_path=Path("source"),
                source_collection="source_collection",
                qdrant_url="http://localhost:6333",
                target_collection="target_collection",
            )
        )
        self.assertFalse(
            migration._checkpoint_matches(
                checkpoint,
                source_path=Path("source"),
                source_collection="source_collection",
                qdrant_url="http://localhost:6333",
                target_collection="different_target",
            )
        )

    def test_cli_remote_configuration_overrides_environment(self) -> None:
        with patch.dict(
            migration.os.environ,
            {"QDRANT_URL": "https://environment.example", "QDRANT_API_KEY": "environment-secret"},
            clear=False,
        ):
            self.assertEqual(
                migration.resolve_remote_config("https://cli.example", "cli-secret"),
                ("https://cli.example", "cli-secret"),
            )

    def test_cloud_client_receives_api_key_and_preflight_does_not_write(self) -> None:
        captured: dict[str, object] = {}

        def make_client(**kwargs: object) -> FakeClient:
            captured.update(kwargs)
            return FakeClient(exists=True)

        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "qdrant"
            database = source / "collection" / "source_collection" / "storage.sqlite"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE points (point BLOB)")
            connection.commit()
            connection.close()
            with patch.object(migration, "QdrantClient", side_effect=make_client):
                migration.migrate(
                    **self._migrate_options(
                        source_path=source,
                        source_collection="source_collection",
                        target_collection="target_collection",
                        qdrant_url="https://cloud.example",
                        qdrant_api_key="cloud-secret",
                        checkpoint_path=Path(temporary_directory) / "checkpoint.json",
                        dry_run_remote_preflight=True,
                    )
                )
        self.assertEqual(captured["url"], "https://cloud.example")
        self.assertEqual(captured["api_key"], "cloud-secret")

    def test_secret_is_not_printed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "qdrant"
            database = source / "collection" / "source_collection" / "storage.sqlite"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE points (point BLOB)")
            connection.commit()
            connection.close()
            output = StringIO()
            with redirect_stdout(output):
                migration.migrate(
                    **self._migrate_options(
                        source_path=source,
                        source_collection="source_collection",
                        qdrant_api_key="do-not-print-this-secret",
                    )
                )
        self.assertNotIn("do-not-print-this-secret", output.getvalue())

    def test_upsert_retries_are_bounded(self) -> None:
        class RetryClient:
            def __init__(self) -> None:
                self.calls = 0

            def upsert(self, **_: object) -> None:
                self.calls += 1
                raise ConnectionError("temporary network failure")

        client = RetryClient()
        with patch.object(migration.time, "sleep") as sleep:
            with self.assertRaisesRegex(migration.MigrationError, "after 3 attempt"):
                migration._upsert_with_retry(
                    client,
                    "target",
                    [],
                    max_retries=2,
                    retry_backoff_seconds=0.5,
                )
        self.assertEqual(client.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_migration_copies_existing_id_vector_and_payload_and_checkpoints(self) -> None:
        class CopyClient(FakeClient):
            def __init__(self) -> None:
                super().__init__()
                self.upserted: list[object] = []

            def count(self, **_: object) -> object:
                return SimpleNamespace(count=len(self.upserted))

            def upsert(self, *, points: list[object], **_: object) -> None:
                self.upserted.extend(points)

        client = CopyClient()
        expected = migration.SourcePoint(
            point_id="043a46df-f3b0-5f3b-bb7e-de7fbd59de8e",
            vector=[0.25] * migration.VECTOR_SIZE,
            payload={"query_id": "232017", "text": "परीक्षण"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.json"
            with patch.object(migration, "source_point_count", return_value=1), patch.object(
                migration, "iter_source_points", return_value=iter([(7, expected)])
            ), patch.object(migration, "QdrantClient", return_value=client):
                self.assertEqual(
                    migration.migrate(
                        **self._migrate_options(
                            source_path=Path(temporary_directory),
                            source_collection="source",
                            target_collection="target",
                            qdrant_api_key="cloud-secret",
                            checkpoint_path=checkpoint_path,
                            dry_run=False,
                        )
                    ),
                    (1, 1),
                )
            saved_checkpoint = migration._read_checkpoint(checkpoint_path)
        self.assertEqual(len(client.upserted), 1)
        copied = client.upserted[0]
        self.assertEqual(str(copied.id), expected.point_id)
        self.assertEqual(copied.vector, expected.vector)
        self.assertEqual(copied.payload, expected.payload)
        self.assertEqual(saved_checkpoint.last_rowid, 7)
        self.assertEqual(saved_checkpoint.points_upserted, 1)
        self.assertNotIn("cloud-secret", migration.json.dumps(saved_checkpoint.__dict__))

    def test_resume_starts_after_last_acknowledged_cloud_batch(self) -> None:
        class ResumeClient(FakeClient):
            def __init__(self) -> None:
                super().__init__()
                self.upserted: list[object] = []

            def count(self, **_: object) -> object:
                return SimpleNamespace(count=len(self.upserted))

            def upsert(self, *, points: list[object], **_: object) -> None:
                self.upserted.extend(points)

        client = ResumeClient()
        expected = migration.SourcePoint(
            point_id="043a46df-f3b0-5f3b-bb7e-de7fbd59de8e",
            vector=[0.25] * migration.VECTOR_SIZE,
            payload={"query_id": "232017"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            checkpoint_path = source / "checkpoint.json"
            migration._write_checkpoint(
                checkpoint_path,
                migration.MigrationCheckpoint(
                    source_path=str(source.resolve()),
                    source_collection="source",
                    qdrant_url="https://cloud.example",
                    target_collection="target",
                    last_rowid=7,
                    points_upserted=7,
                ),
            )
            with patch.object(migration, "source_point_count", return_value=8), patch.object(
                migration, "iter_source_points", return_value=iter([(8, expected)])
            ) as iter_points, patch.object(migration, "QdrantClient", return_value=client):
                source_read, points_upserted = migration.migrate(
                    **self._migrate_options(
                        source_path=source,
                        source_collection="source",
                        target_collection="target",
                        checkpoint_path=checkpoint_path,
                        resume=True,
                        dry_run=False,
                    )
                )
        self.assertEqual((source_read, points_upserted), (1, 8))
        self.assertEqual(iter_points.call_args.kwargs["after_rowid"], 7)
        self.assertEqual(len(client.upserted), 1)

    def test_dry_run_does_not_construct_a_target_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "qdrant"
            database = source / "collection" / "source_collection" / "storage.sqlite"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE points (point BLOB)")
                connection.commit()
            finally:
                connection.close()
            with patch.object(migration, "QdrantClient", side_effect=AssertionError("target client should not be created")):
                self.assertEqual(
                    migration.migrate(**self._migrate_options(
                        source_path=source,
                        source_collection="source_collection",
                        qdrant_url="http://localhost:6333",
                        target_collection="target_collection",
                        checkpoint_path=Path(temporary_directory) / "checkpoint.json",
                    )),
                    (0, 0),
                )

    def test_invalid_batch_size_fails_before_target_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            migration.migrate(**self._migrate_options(
                batch_size=0,
            ))


if __name__ == "__main__":
    unittest.main()
