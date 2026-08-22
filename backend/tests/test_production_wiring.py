"""Focused source-level safeguards for frozen production runtime wiring."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionWiringTests(unittest.TestCase):
    def test_main_uses_verified_remote_bilingual_store_and_warms_e5(self) -> None:
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("create_verified_bilingual_cloud_store", source)
        self.assertIn("runtime.embedder.embed_query", source)
        self.assertIn('"application_ready"] = True', source)
        self.assertNotIn("RAGRuntime()", source)

    def test_http_route_delegates_to_the_one_shot_harness(self) -> None:
        source = (ROOT / "app" / "routes" / "voice.py").read_text(encoding="utf-8")
        http_source = source[source.index("async def query_voice"):source.index("@router.websocket")]
        self.assertIn("harness.execute(context.with_query(transcript, stt_latency_ms), request.app)", http_source)
        self.assertNotIn("runtime.answer(", http_source)

    def test_cloud_contract_requires_collection_dimension_count_and_indexes(self) -> None:
        source = (ROOT / "app" / "rag" / "bilingual_cloud_runtime.py").read_text(encoding="utf-8")
        for expected in ("mode=\"remote\"", "BILINGUAL_COLLECTION", "vector dimension must be 768", "point count must be", "REQUIRED_PAYLOAD_INDEXES"):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
