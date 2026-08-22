"""Safety regression checks for the disposable compact-policy benchmark."""

import unittest
from pathlib import Path


class CompactPolicyBenchmarkSafetyTests(unittest.TestCase):
    def test_benchmark_uses_dataset_ground_truth_not_full_ann_baseline(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        self.assertIn("MSMARCO-XI selected-passage", source)
        self.assertNotIn('QdrantClient(path=str(DEFAULT_SOURCE_PATH))', source)
        self.assertIn("_expected_provenance", source)

    def test_writes_are_limited_to_disposable_collections(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        self.assertIn("BENCHMARK_PATH", source)
        self.assertIn("COLLECTIONS", source)
        self.assertIn("FULL_COLLECTION_WRITES", source.upper())
        self.assertNotIn("NeonVectorStore", source)
        self.assertNotIn("embed_passages(", source)

    def test_cleanup_is_scoped_to_known_temporary_collections(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        self.assertIn("for name in COLLECTIONS.values()", source)
        self.assertIn("--cleanup", source)

    def test_preflight_reuses_the_production_id_builder(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        self.assertIn("from app.rag.indexing.vector_store import build_strategy_aware_point_identity, strategy_aware_point_id", source)
        self.assertIn("build_strategy_aware_point_identity(", source)
        self.assertIn("point_id=strategy_aware_point_id(chunk)", source)

    def test_benchmark_has_no_unsupported_recommendation_thresholds(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        self.assertIn("recommend_compact_policy", source)
        self.assertNotIn(">= 0.90", source)
        self.assertNotIn(">= 0.50", source)

    def test_missing_diagnostics_and_known_overwrite_classification_are_present(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        for field in (
            '"MISSING_POINT_ID"',
            '"POLICY"',
            '"source_row_number"',
            '"passage_index"',
            '"chunk_index"',
            '"chunk_strategy"',
            '"text_hash"',
            '"text_preview"',
            '"id_input"',
        ):
            self.assertIn(field, source)
        self.assertIn('KNOWN_OVERWRITE_QUERY_ID = "177416"', source)
        self.assertIn("original_text_hash != duplicate_text_hash", source)
        self.assertIn('return "KNOWN_FULL_INDEX_OVERWRITE"', source)

    def test_preflight_stops_before_temporary_collection_client_creation(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        failure = source.index("Unexplained policy chunk IDs are missing")
        client_creation = source.index("client = QdrantClient(path=str(BENCHMARK_PATH))", failure)
        self.assertLess(failure, client_creation)
        self.assertIn("--preflight-only", source)

    def test_known_overwrite_exclusion_is_variant_scoped(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compact_hindi_policies.py").read_text(encoding="utf-8")
        self.assertIn("excluded_variants", source)
        self.assertIn("(descriptor.point_id, descriptor.row_number, descriptor.text_hash)", source)
        self.assertIn("list(dict.fromkeys(ids))", source)


if __name__ == "__main__":
    unittest.main()
