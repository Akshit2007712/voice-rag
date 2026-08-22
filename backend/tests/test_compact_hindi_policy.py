"""Policy-only tests; they do not load E5, Qdrant, Neon, or Parquet."""

import unittest
from pathlib import Path

from app.rag.analysis.compact_hindi_policy import (
    POLICY_A,
    POLICY_B,
    POLICY_C,
    POLICY_D,
    POLICY_E,
    selected_documents_for_policy,
    storage_estimate,
)


def record(labels: list[int]) -> dict[str, object]:
    return {
        "query_id": "42",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "Translated_passages": [f"passage {index}" for index in range(len(labels))],
            "is_selected": labels,
        },
    }


def indexes(documents) -> list[int]:
    return [document.metadata["passage_index"] for document in documents]


class CompactHindiPolicyTests(unittest.TestCase):
    def test_selected_only_filters_unselected_passages(self) -> None:
        self.assertEqual(indexes(selected_documents_for_policy(record([0, 1, 0, 1]), POLICY_A)), [1, 3])

    def test_zero_selected_coverage_fallback_is_first_non_empty_passage(self) -> None:
        self.assertEqual(indexes(selected_documents_for_policy(record([0, 0, 0]), POLICY_D)), [0])

    def test_plus_one_prefers_adjacent_unselected_passage_deterministically(self) -> None:
        self.assertEqual(indexes(selected_documents_for_policy(record([0, 0, 1, 0, 0]), POLICY_B)), [2, 1])

    def test_plus_two_uses_two_nearest_unselected_passages(self) -> None:
        self.assertEqual(indexes(selected_documents_for_policy(record([0, 0, 1, 0, 0]), POLICY_C)), [2, 1, 3])

    def test_fallback_policy_covers_zero_selected_rows(self) -> None:
        self.assertEqual(indexes(selected_documents_for_policy(record([0, 0]), POLICY_E)), [0])

    def test_policy_is_deterministic(self) -> None:
        source = record([0, 1, 0, 0, 1, 0])
        self.assertEqual(
            indexes(selected_documents_for_policy(source, POLICY_E)),
            indexes(selected_documents_for_policy(source, POLICY_E)),
        )

    def test_storage_extrapolation_uses_measured_bytes_per_point(self) -> None:
        estimate = storage_estimate(100, 9_193.472, 2.5)
        self.assertGreater(estimate["estimated_neon_gb"], 0)
        self.assertEqual(estimate["estimated_bilingual_gb"], estimate["estimated_neon_gb"] * 2)
        self.assertIsNotNone(estimate["estimated_monthly_storage_cost"])

    def test_analysis_reuses_chunker_without_embedding_or_database_writes(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "analyze_compact_hindi_corpus.py").read_text(encoding="utf-8")
        self.assertIn("iter_document_chunks", source)
        self.assertIn("get_e5_tokenizer", source)
        self.assertNotIn("E5Embedder", source)
        self.assertNotIn(".upsert(", source)
        self.assertNotIn("QdrantClient", source)
        self.assertNotIn("NeonVectorStore", source)

    def test_analysis_script_bootstraps_backend_root_before_app_imports(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "analyze_compact_hindi_corpus.py").read_text(encoding="utf-8")
        bootstrap = 'BACKEND_ROOT = Path(__file__).resolve().parents[1]'
        app_import = "from app.rag.analysis.compact_hindi_policy import"
        self.assertIn(bootstrap, source)
        self.assertIn("sys.path.insert(0, str(BACKEND_ROOT))", source)
        self.assertLess(source.index(bootstrap), source.index(app_import))


if __name__ == "__main__":
    unittest.main()
