"""Unit tests for threshold-free compact-policy recommendation interpretation."""

import unittest
from collections.abc import Sequence

from app.rag.analysis.compact_hindi_policy import POLICY_A, POLICY_D, POLICY_E
from app.rag.analysis.compact_policy_recommendation import (
    policy_pareto_dominates,
    recommend_compact_policy,
    recommendation_interpretation,
)


def metrics(
    *,
    top_k: float,
    top_1: float,
    family: float,
    chunks: int,
    latency: float | Sequence[float],
    zero_coverage: float,
) -> dict[str, object]:
    latency_values = (latency, latency, latency, latency) if isinstance(latency, (int, float)) else latency
    return {
        "expected_evidence_in_top_k_rate": top_k,
        "top1_expected_evidence_rate": top_1,
        "correct_evidence_family_rate": family,
        "chunk_count": chunks,
        "latency_ms": dict(zip(("p50", "p70", "p95", "p100"), latency_values, strict=True)),
        "zero_selected": {"same_query_id_in_top_k_rate": zero_coverage},
    }


class CompactPolicyRecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current_results = {
            POLICY_A: metrics(top_k=0.7698412698, top_1=0.5396825397, family=0.7698412698, chunks=58_427, latency=(161.70, 170.98, 189.56, 243.54), zero_coverage=0.0),
            POLICY_D: metrics(top_k=0.7142857143, top_1=0.50, family=0.7142857143, chunks=103_502, latency=(269.50, 278.28, 297.40, 357.60), zero_coverage=0.16667),
            POLICY_E: metrics(top_k=0.6746031746, top_1=0.4365079365, family=0.7301587302, chunks=158_537, latency=(439.08, 445.60, 463.89, 513.56), zero_coverage=0.125),
        }

    def test_current_metrics_recommend_policy_a_from_dominance(self) -> None:
        self.assertTrue(policy_pareto_dominates(self.current_results[POLICY_A], self.current_results[POLICY_D]))
        self.assertTrue(policy_pareto_dominates(self.current_results[POLICY_A], self.current_results[POLICY_E]))
        self.assertEqual(recommend_compact_policy(self.current_results), "POLICY_A_RECOMMENDED")

    def test_zero_selected_coverage_is_not_a_labeled_relevance_gate(self) -> None:
        recommendation = recommend_compact_policy(self.current_results)
        interpretation = recommendation_interpretation(self.current_results, recommendation)
        self.assertFalse(interpretation["zero_selected_coverage_is_labeled_relevance"])
        self.assertTrue(interpretation["zero_selected_coverage_caveat"])
        self.assertIn("intentionally excludes", interpretation["zero_selected_coverage_caveat_reason"])

    def test_recommendation_is_not_hardcoded_to_policy_a(self) -> None:
        results = dict(self.current_results)
        results[POLICY_D] = metrics(top_k=0.9, top_1=0.8, family=0.9, chunks=50_000, latency=100, zero_coverage=0.0)
        self.assertEqual(recommend_compact_policy(results), "POLICY_D_RECOMMENDED")

    def test_real_trade_off_is_not_forced_into_a_recommendation(self) -> None:
        results = dict(self.current_results)
        results[POLICY_D] = metrics(top_k=0.9, top_1=0.8, family=0.9, chunks=100_000, latency=300, zero_coverage=0.0)
        self.assertEqual(recommend_compact_policy(results), "COMPACT_POLICY_TRADE_OFF_UNRESOLVED")

    def test_local_latency_is_not_deployment_compliance(self) -> None:
        interpretation = recommendation_interpretation(self.current_results, "POLICY_A_RECOMMENDED")
        self.assertFalse(interpretation["final_deployment_latency_validated"])
        self.assertIn("exclude E5", interpretation["local_latency_note"])


if __name__ == "__main__":
    unittest.main()
