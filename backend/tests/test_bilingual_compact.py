"""Focused tests for compact English and bilingual safety helpers."""

import unittest
from pathlib import Path

from app.rag.analysis.bilingual_compact import (
    FROZEN_HINDI_POLICY_A_CHUNKS,
    bilingual_payload,
    build_bilingual_point_id,
    english_policy_a_documents,
    estimate_is_safe,
)
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_english_record
from app.rag.language_config import get_application_language, get_qdrant_target_lang


def record() -> dict[str, object]:
    return {
        "query_id": 7,
        "query_type": "DESCRIPTION",
        "passages": {
            "English_passages": ["  selected English text  ", "unselected English text"],
            "Translated_passages": ["चयनित", "अचयनित"],
            "is_selected": [1, 0],
        },
    }


class BilingualCompactTests(unittest.TestCase):
    def test_english_policy_a_keeps_only_selected_english_passages(self) -> None:
        documents = preprocess_msmarco_xi_english_record(record())
        self.assertEqual(documents[0].text, "selected English text")
        selected = english_policy_a_documents(record())
        self.assertEqual([document.metadata["passage_index"] for document in selected], [0])
        self.assertEqual(selected[0].metadata["target_lang"], "eng_Latn")

    def test_language_aware_ids_do_not_collide(self) -> None:
        metadata = {"query_id": 7, "passage_index": 0, "chunk_index": 0, "chunk_strategy": "whole_passage"}
        self.assertNotEqual(build_bilingual_point_id("hi", metadata), build_bilingual_point_id("en", metadata))

    def test_payload_contains_language_and_required_provenance(self) -> None:
        metadata = {"query_id": 7, "passage_index": 0, "chunk_index": 0, "chunk_strategy": "whole_passage", "is_selected": 1, "token_count": 3}
        payload = bilingual_payload("en", "English text", metadata)
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["target_lang"], "eng_Latn")
        self.assertEqual(payload["text"], "English text")

    def test_language_routing_has_english_and_hindi_filters(self) -> None:
        self.assertEqual(get_qdrant_target_lang("en"), "eng_Latn")
        self.assertEqual(get_application_language("hin_Deva"), "hi")
        self.assertEqual(get_application_language("eng_Latn"), "en")

    def test_four_gib_gate_is_conservative(self) -> None:
        self.assertTrue(estimate_is_safe(FROZEN_HINDI_POLICY_A_CHUNKS))
        self.assertFalse(estimate_is_safe(1_000_000))

    def test_indexer_protects_full_hindi_and_reuses_without_hindi_embedding(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "index_bilingual_compact.py").read_text(encoding="utf-8")
        copy_start = source.index("def _copy_hindi_policy_a")
        copy_end = source.index("def _embed_english_policy_a")
        copy_function = source[copy_start:copy_end]
        self.assertIn("FULL_HINDI_COLLECTION", source)
        self.assertIn("refuses to modify msmarco_xi_hindi_full", source)
        self.assertNotIn("E5Embedder", copy_function)
        self.assertIn("build_bilingual_point_id(\"hi\"", copy_function)

    def test_vector_store_filters_on_application_language(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "app" / "rag" / "indexing" / "vector_store.py").read_text(encoding="utf-8")
        self.assertIn('FieldCondition(key="language"', source)
        self.assertIn('FieldCondition(key="target_lang"', source)

    def test_operational_scripts_require_cloud_bilingual_runtime(self) -> None:
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        for name in ("smoke_bilingual_retrieval.py", "benchmark_bilingual_text_rag_latency.py"):
            source = (scripts_root / name).read_text(encoding="utf-8")
            self.assertIn("create_bilingual_cloud_runtime", source)
            self.assertNotIn("RAGRuntime()", source)
            self.assertNotIn("msmarco_xi_hindi_full", source)

    def test_cloud_runtime_is_remote_only_and_checks_collection_before_e5(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "app" / "rag" / "bilingual_cloud_runtime.py").read_text(encoding="utf-8")
        self.assertIn('mode="remote"', source)
        self.assertIn("QDRANT_URL is required", source)
        self.assertIn("QDRANT_API_KEY is required", source)
        self.assertIn("create_verified_bilingual_cloud_store", source)
        self.assertLess(source.index("create_verified_bilingual_cloud_store"), source.index("RAGRuntime(vector_store=store)"))

    def test_benchmark_warms_e5_before_measured_queries(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_bilingual_text_rag_latency.py").read_text(encoding="utf-8")
        self.assertIn("hybrid.retrieve(query, 5, target)", source)
        self.assertIn("E5_WARMUP_COMPLETE=true", source)

    def test_payload_index_migration_creates_only_required_keyword_indexes(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "create_bilingual_qdrant_payload_indexes.py").read_text(encoding="utf-8")
        self.assertIn('REQUIRED_KEYWORD_INDEXES = ("language", "target_lang")', source)
        self.assertIn("PayloadSchemaType.KEYWORD", source)
        self.assertIn("create_payload_index", source)
        self.assertIn('_verify_language_filter(client, "hi", "hin_Deva")', source)
        self.assertIn('_verify_language_filter(client, "en", "eng_Latn")', source)
        self.assertNotIn("upsert(", source)
        self.assertNotIn("delete_collection", source)
        self.assertNotIn("recreate_collection", source)


if __name__ == "__main__":
    unittest.main()
