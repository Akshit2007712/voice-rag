"""Focused unit tests for read-only full-index verification helpers."""

from __future__ import annotations

import importlib.util
import base64
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_full_hindi_index.py"
SPEC = importlib.util.spec_from_file_location("verify_full_hindi_index", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import failure is fatal.
    raise RuntimeError(f"Could not load verifier script: {SCRIPT_PATH}")
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


class FullIndexVerifierTests(unittest.TestCase):
    def test_paged_qdrant_id_ingestion_uses_no_payloads_or_vectors(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = []

            def get_collection(self, collection_name):
                return SimpleNamespace(
                    config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=768)))
                )

            def scroll(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs["offset"] is None:
                    return [SimpleNamespace(id="a"), SimpleNamespace(id="b")], "next"
                return [SimpleNamespace(id="c")], None

        with tempfile.TemporaryDirectory() as directory:
            connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
            try:
                client = FakeClient()
                store = SimpleNamespace(client=client, collection_name="msmarco_xi_hindi_full")
                with redirect_stdout(StringIO()):
                    count, dimension = verifier._qdrant_id_scroll(store, connection, 3, 0)

                self.assertEqual((count, dimension), (3, 768))
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM actual_ids").fetchone()[0], 3
                )
                self.assertTrue(all(call["with_payload"] is False for call in client.calls))
                self.assertTrue(all(call["with_vectors"] is False for call in client.calls))
            finally:
                connection.close()

    def test_local_storage_id_scan_is_read_only_and_decodes_known_uuid_keys(self) -> None:
        point_id = "043a46df-f3b0-5f3b-bb7e-de7fbd59de8e"
        encoded_id = base64.b64encode(pickle.dumps(point_id, protocol=4)).decode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path = root / "collection" / "msmarco_xi_hindi_full"
            collection_path.mkdir(parents=True)
            storage_path = collection_path / "storage.sqlite"
            storage = verifier.sqlite3.connect(storage_path)
            storage.execute("CREATE TABLE points (id TEXT PRIMARY KEY, point BLOB)")
            storage.execute("INSERT INTO points(id, point) VALUES (?, ?)", (encoded_id, b"not-read"))
            storage.commit()
            storage.close()
            (root / "meta.json").write_text(
                json.dumps({"collections": {"msmarco_xi_hindi_full": {"vectors": {"size": 768}}}}),
                encoding="utf-8",
            )
            connection = verifier._open_verification_database(root / "audit.sqlite3")
            try:
                settings = SimpleNamespace(path=root, collection_name="msmarco_xi_hindi_full")
                with redirect_stdout(StringIO()):
                    count, dimension = verifier._local_storage_id_scan(settings, connection, 1, 0)

                self.assertEqual((count, dimension), (1, 768))
                self.assertEqual(
                    connection.execute("SELECT point_id FROM actual_ids").fetchone()[0], point_id
                )
            finally:
                connection.close()

    def test_remote_verifier_store_passes_url_and_timeout_without_local_path_mode(self) -> None:
        original_client = verifier.QdrantClient
        original_store = verifier.VectorStore
        captured = {}
        try:
            verifier.QdrantClient = lambda **kwargs: captured.setdefault("client", kwargs)
            verifier.VectorStore = lambda settings, client: captured.setdefault(
                "store", {"settings": settings, "client": client}
            )
            settings = SimpleNamespace(url="https://cloud.example", api_key="cloud-secret")
            result = verifier._create_remote_verifier_store(settings, 12.5)

            self.assertEqual(result["client"]["url"], "https://cloud.example")
            self.assertEqual(result["client"]["api_key"], "cloud-secret")
            self.assertEqual(result["client"]["timeout"], 12.5)
            self.assertFalse(result["client"]["check_compatibility"])
        finally:
            verifier.QdrantClient = original_client
            verifier.VectorStore = original_store

    def test_verifier_cli_credentials_override_environment_settings(self) -> None:
        configured = verifier.QdrantSettings(
            mode="remote",
            path=Path("data/qdrant"),
            url="https://environment.example",
            api_key="environment-secret",
            collection_name="old_collection",
        )
        resolved = verifier._resolve_verifier_settings(
            configured,
            qdrant_url="https://cli.example",
            qdrant_api_key="cli-secret",
            collection_name="msmarco_xi_hindi_full",
        )
        self.assertEqual(resolved.url, "https://cli.example")
        self.assertEqual(resolved.api_key, "cli-secret")
        self.assertEqual(resolved.collection_name, "msmarco_xi_hindi_full")

    def test_verifier_uses_cloud_environment_without_qdrant_mode_override(self) -> None:
        configured = verifier.QdrantSettings(mode="local", path=Path("data/qdrant"))
        with patch.dict(
            verifier.os.environ,
            {"QDRANT_URL": "https://environment.example", "QDRANT_API_KEY": "environment-secret"},
            clear=False,
        ):
            resolved = verifier._resolve_verifier_settings(
                configured,
                qdrant_url=None,
                qdrant_api_key=None,
                collection_name="msmarco_xi_hindi_full",
            )
        self.assertEqual(resolved.mode, "remote")
        self.assertEqual(resolved.url, "https://environment.example")
        self.assertEqual(resolved.api_key, "environment-secret")

    def test_known_source_collisions_do_not_mask_an_exact_cloud_copy(self) -> None:
        with redirect_stdout(StringIO()):
            status = verifier._final_status(
                total_chunks=1_000_361,
                expected_unique=1_000_351,
                actual_count=1_000_351,
                missing_expected=0,
                unexpected_count=0,
                collision_different_content=3,
                payload_errors=0,
                vector_dimension=768,
                failed_query_classification="UNDERSTOOD",
            )
        self.assertEqual(status, "INDEX_SAFE_FOR_BENCHMARKING")

    def test_verification_database_can_be_reopened_for_post_scan_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite3"
            connection = verifier._open_verification_database(path)
            connection.execute("INSERT INTO actual_ids(point_id) VALUES ('persisted-id')")
            connection.commit()
            connection.close()

            reopened = verifier._open_verification_database(path, reuse=True)
            try:
                self.assertEqual(
                    reopened.execute("SELECT point_id FROM actual_ids").fetchone()[0],
                    "persisted-id",
                )
            finally:
                reopened.close()

    def test_actual_ids_reset_is_restart_safe_and_preserves_expected_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
            try:
                connection.execute("INSERT INTO actual_ids(point_id) VALUES ('old-id')")
                connection.execute(
                    """
                    INSERT INTO expected_ids(
                        point_id, row_number, query_id, passage_index, chunk_index,
                        chunk_strategy, text_hash, preview
                    ) VALUES ('expected-id', 1, '7', 0, 0, 'whole_passage', 'hash', 'preview')
                    """
                )
                verifier._reset_post_scan_tables(connection)

                self.assertEqual(connection.execute("SELECT COUNT(*) FROM actual_ids").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM expected_ids").fetchone()[0], 1)
            finally:
                connection.close()

    def test_payload_integrity_reads_only_a_bounded_sample_without_vectors(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.retrieve_kwargs = None

            def retrieve(self, **kwargs):
                self.retrieve_kwargs = kwargs
                return [
                    SimpleNamespace(
                        id=point_id,
                        payload={
                            "query_id": "7",
                            "passage_index": 0,
                            "chunk_index": 0,
                            "chunk_strategy": "whole_passage",
                            "target_lang": "hin_Deva",
                            "text": "मान्य पाठ",
                        },
                    )
                    for point_id in kwargs["ids"]
                ]

        with tempfile.TemporaryDirectory() as directory:
            connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
            try:
                connection.executemany(
                    "INSERT INTO actual_ids(point_id) VALUES (?)",
                    [(f"id-{index}",) for index in range(10)],
                )
                client = FakeClient()
                store = SimpleNamespace(client=client, collection_name="msmarco_xi_hindi_full")
                malformed, sampled = verifier._payload_integrity_sample(
                    store, connection, "hin_Deva", sample_size=3
                )

                self.assertEqual((malformed, sampled), ([], 3))
                self.assertFalse(client.retrieve_kwargs["with_vectors"])
                self.assertTrue(client.retrieve_kwargs["with_payload"])
                self.assertEqual(len(client.retrieve_kwargs["ids"]), 3)
            finally:
                connection.close()

    def test_duplicate_id_records_both_provenance_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
            try:
                original = verifier.ChunkDescriptor(
                    point_id="same-id",
                    row_number=1,
                    query_id="7",
                    passage_index=0,
                    chunk_index=0,
                    chunk_strategy="whole_passage",
                    text_hash="a" * 64,
                    preview="पहला पाठ",
                )
                duplicate = verifier.ChunkDescriptor(
                    point_id="same-id",
                    row_number=9,
                    query_id="7",
                    passage_index=0,
                    chunk_index=0,
                    chunk_strategy="whole_passage",
                    text_hash="b" * 64,
                    preview="बदला हुआ पाठ",
                )

                self.assertFalse(verifier._insert_expected_descriptor(connection, original))
                self.assertTrue(verifier._insert_expected_descriptor(connection, duplicate))
                collision = verifier._collision_rows(connection)[0]

                self.assertEqual(collision["original_row_number"], 1)
                self.assertEqual(collision["duplicate_row_number"], 9)
                self.assertEqual(collision["original_text_hash"], "a" * 64)
                self.assertEqual(collision["duplicate_text_hash"], "b" * 64)
            finally:
                connection.close()

    def test_payload_validation_requires_provenance_language_and_text(self) -> None:
        valid_payload = {
            "query_id": "7",
            "passage_index": 0,
            "chunk_index": 0,
            "chunk_strategy": "whole_passage",
            "target_lang": "hin_Deva",
            "text": "मान्य पाठ",
        }
        self.assertEqual(verifier._payload_errors(valid_payload, "hin_Deva"), [])

        errors = verifier._payload_errors({"target_lang": "eng_Latn", "text": " "}, "hin_Deva")
        self.assertIn("query_id", errors)
        self.assertIn("target_lang='eng_Latn'", errors)
        self.assertIn("empty_text", errors)

    def test_expected_and_actual_ids_are_compared_in_sqlite_not_python_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
            try:
                descriptor = verifier.ChunkDescriptor(
                    point_id="expected-id",
                    row_number=1,
                    query_id="7",
                    passage_index=0,
                    chunk_index=0,
                    chunk_strategy="whole_passage",
                    text_hash="a" * 64,
                    preview="पाठ",
                )
                verifier._insert_expected_descriptor(connection, descriptor)
                connection.execute("INSERT INTO actual_ids(point_id) VALUES (?)", ("expected-id",))
                connection.execute(
                    "INSERT INTO actual_ids(point_id) VALUES (?)",
                    ("unexpected-id",),
                )
                with redirect_stdout(StringIO()):
                    expected, missing, unexpected, _, unexpected_ids = verifier._id_set_report(connection, 10)

                self.assertEqual(expected, 1)
                self.assertEqual(missing, 0)
                self.assertEqual(unexpected, 1)
                self.assertEqual(unexpected_ids, ["unexpected-id"])
            finally:
                connection.close()

    def test_query_scoped_bm25_keeps_only_matching_candidate_rows_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
            try:
                audit = verifier.QueryScopedBM25Audit(connection, "वाष्प दबाव")
                audit.observe_payload(
                    "match",
                    {
                        "target_lang": "hin_Deva",
                        "text": "वाष्प दबाव बढ़ने पर परिवर्तन होता है",
                        "query_id": 430672,
                        "passage_index": 8,
                        "chunk_index": 0,
                        "chunk_strategy": "whole_passage",
                    },
                    "hin_Deva",
                )
                audit.observe_payload(
                    "non-match",
                    {
                        "target_lang": "hin_Deva",
                        "text": "यह असंबंधित हिंदी पाठ है",
                        "query_id": 1,
                        "passage_index": 0,
                        "chunk_index": 0,
                        "chunk_strategy": "whole_passage",
                    },
                    "hin_Deva",
                )
                self.assertEqual(audit.document_count, 2)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM bm25_candidates").fetchone()[0],
                    1,
                )
                results = audit.search(5)
                self.assertEqual(results[0].metadata["query_id"], "430672")
            finally:
                connection.close()

    def test_partial_row_limit_stops_stream_without_retaining_rows(self) -> None:
        original_loader = verifier.iter_msmarco_xi_records
        original_tokenizer = verifier.get_e5_tokenizer
        original_preprocessor = verifier.preprocess_msmarco_xi_record
        original_chunker = verifier.iter_document_chunks
        original_point_id = verifier.strategy_aware_point_id
        try:
            records = [{"query_id": index} for index in range(1, 6)]
            verifier.iter_msmarco_xi_records = lambda *args: iter(records)
            verifier.get_e5_tokenizer = lambda: object()
            verifier.preprocess_msmarco_xi_record = lambda record: [record]
            verifier.iter_document_chunks = lambda documents, **kwargs: [
                SimpleNamespace(
                    text="संक्षिप्त पाठ",
                    metadata={
                        "query_id": documents[0]["query_id"],
                        "passage_index": 0,
                        "chunk_index": 0,
                        "chunk_strategy": "whole_passage",
                    },
                )
            ]
            verifier.strategy_aware_point_id = lambda chunk: f"id-{chunk.metadata['query_id']}"
            with tempfile.TemporaryDirectory() as directory:
                connection = verifier._open_verification_database(Path(directory) / "audit.sqlite3")
                try:
                    lexical_audit = verifier.QueryScopedBM25Audit(connection, verifier.FAILED_QUERY)
                    with redirect_stdout(StringIO()):
                        chunks, duplicates, _ = verifier._iter_regenerated_chunks(
                            "hi", "validation", 500, 256, connection, 3, 100, lexical_audit, "hin_Deva"
                        )
                    self.assertEqual((chunks, duplicates), (3, 0))
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM expected_ids").fetchone()[0],
                        3,
                    )
                finally:
                    connection.close()
        finally:
            verifier.iter_msmarco_xi_records = original_loader
            verifier.get_e5_tokenizer = original_tokenizer
            verifier.preprocess_msmarco_xi_record = original_preprocessor
            verifier.iter_document_chunks = original_chunker
            verifier.strategy_aware_point_id = original_point_id

    def test_verifier_has_no_qdrant_write_or_document_embedding_path(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn(".upsert(", source)
        self.assertNotIn("embed_passages(", source)
        self.assertNotIn("BM25Store.from_vector_store", source)
        self.assertIn("with_payload=False", source)
        self.assertIn("with_vectors=False", source)
        self.assertIn("scroll_filter=query_filter", source)
        self.assertIn("--post-scan-only", source)
        self.assertIn("DROP TABLE IF EXISTS actual_ids", source)
        self.assertNotIn("VACUUM", source)
        self.assertIn("QDRANT_CLIENT_INIT_START", source)
        self.assertIn("QDRANT_CLIENT_INIT_SKIPPED=local_storage_read_only_id_audit", source)
        self.assertIn("--qdrant-url", source)
        self.assertIn("--qdrant-api-key", source)
        self.assertIn("QDRANT_API_KEY_PRESENT", source)
        self.assertNotIn("print(f\"QDRANT_API_KEY={", source)
        self.assertIn("ACTUAL_IDS_RESET_START", source)
        self.assertIn("DOCUMENT_EMBEDDINGS_CREATED=0", source)


if __name__ == "__main__":
    unittest.main()
