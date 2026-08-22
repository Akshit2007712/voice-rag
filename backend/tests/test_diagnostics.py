"""Focused tests for cheap public probes and authenticated in-memory diagnostics."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.diagnostics import router
from app.services.benchmark_latency import pending_benchmark_latency
from app.services.diagnostics import DiagnosticsRegistry


_READY = {
    "qdrant_ready": True,
    "collection_verified": True,
    "payload_indexes_verified": True,
    "e5_loaded": True,
    "e5_warmed": True,
    "bm25_hi_ready": True,
    "bm25_en_ready": True,
    "hybrid_workers_ready": True,
    "application_ready": True,
}


class _StoreProbe:
    qdrant_retry_count = 4
    qdrant_failed_request_count = 2

    def search(self, *_args, **_kwargs):  # pragma: no cover - assertion if endpoint regresses.
        raise AssertionError("diagnostics must not query Qdrant")


class _EmbedderProbe:
    device = "cuda"

    def embed_query(self, *_args, **_kwargs):  # pragma: no cover - assertion if endpoint regresses.
        raise AssertionError("diagnostics must not embed")


def _test_app(ready: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.rag_readiness = dict(_READY if ready else {key: False for key in _READY})
    app.state.diagnostics_registry = DiagnosticsRegistry()
    app.state.diagnostics_verification = {
        "verified_at_startup": ready,
        "expected_point_count": 115_909,
        "verified_point_count": 115_909 if ready else None,
        "vector_size": 768 if ready else None,
        "language_index_verified": ready,
        "target_lang_index_verified": ready,
    }
    app.state.rag_runtime = SimpleNamespace(embedder=_EmbedderProbe(), vector_store=_StoreProbe())
    app.state.benchmark_latency = pending_benchmark_latency()
    return app


class ProbeTests(unittest.TestCase):
    def test_health_is_minimal_and_does_not_touch_dependencies(self) -> None:
        with TestClient(_test_app()) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_is_minimal_for_ready_and_unready_state(self) -> None:
        with TestClient(_test_app(ready=False)) as client:
            unready = client.get("/ready")
        self.assertEqual(unready.status_code, 503)
        self.assertEqual(unready.json(), {"ready": False})
        with TestClient(_test_app(ready=True)) as client:
            ready = client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {"ready": True})

    def test_diagnostics_requires_its_own_credential(self) -> None:
        with patch.dict(os.environ, {"DIAGNOSTICS_API_KEY": "diagnostics-test-key"}):
            with TestClient(_test_app()) as client:
                missing = client.get("/diagnostics")
                invalid = client.get("/diagnostics", headers={"X-Diagnostics-Key": "wrong"})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(missing.json(), {"detail": "Unauthorized"})
        self.assertEqual(invalid.json(), {"detail": "Unauthorized"})

    def test_authorized_diagnostics_reads_state_only_and_omits_sensitive_content(self) -> None:
        app = _test_app()
        app.state.diagnostics_registry.record_success(
            "opaque-id", "text", "hi",
            {"embedding_ms": 1.0, "rag_total_ms": 2.0},
            qdrant_retry_count=1,
        )
        with patch.dict(os.environ, {"DIAGNOSTICS_API_KEY": "diagnostics-test-key"}):
            with TestClient(app) as client:
                response = client.get("/diagnostics", headers={"X-Diagnostics-Key": "diagnostics-test-key"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["qdrant"]["mode"], "remote")
        self.assertEqual(payload["qdrant"]["retry_count"], 4)
        self.assertEqual(payload["requests"]["last_request_id"], "opaque-id")
        self.assertEqual(payload["last_latency"]["rag_total_ms"], 2.0)
        self.assertEqual(payload["benchmark"]["status"], "pending")
        self.assertNotIn("diagnostics-test-key", str(payload))
        self.assertNotIn("query", str(payload).lower())

    def test_missing_diagnostics_configuration_fails_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with TestClient(_test_app()) as client:
                response = client.get("/diagnostics")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Diagnostics unavailable."})


class RegistryTests(unittest.TestCase):
    def test_registry_is_thread_safe_and_never_retains_request_content(self) -> None:
        registry = DiagnosticsRegistry()
        secret_query = "typed query that must never be retained"
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(
                lambda index: registry.record_success(
                    f"request-{index}", "text", "en", {"rag_total_ms": float(index)}, 0
                ),
                range(40),
            ))
        registry.record_failure("failure-id", "voice", "hi", ValueError(secret_query))
        snapshot = registry.snapshot()
        self.assertEqual(snapshot["requests"]["successful"], 40)
        self.assertEqual(snapshot["requests"]["failed"], 1)
        self.assertEqual(snapshot["requests"]["last_error_category"], "validation_error")
        self.assertNotIn(secret_query, str(snapshot))


if __name__ == "__main__":
    unittest.main()
