"""Opt-in instrumentation for dual-condition early-release experiments."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from app.rag.generation.answer_composer import AnswerComposer, ComposedAnswer
from app.rag.retrieval.hybrid_retriever import HybridRetrievalResult, HybridRetriever
from app.rag.retrieval.maturity import MaturityDecision, meaningful_tokens


POLICY_LIMITED_LEAD_MS = 50.0


@dataclass(frozen=True)
class PostSpeechEndRetrievalObservation:
    """One existing retrieval evaluation that completed after VAD speech end."""

    partial: str
    offset_after_speech_end_ms: float
    semantic_latency_ms: float
    bm25_latency_ms: float
    fusion_latency_ms: float
    semantic_top1_query_id: str | None
    semantic_top1_score: float | None
    semantic_top_k_query_ids: tuple[str, ...]
    semantic_dominant_query_id: str | None
    semantic_dominant_count: int
    semantic_dominant_ratio: float | None
    semantic_concentrated: bool
    bm25_top_k_query_ids: tuple[str, ...]
    semantic_dominant_query_id_in_bm25_top_k: bool
    hybrid_top1_query_id: str | None
    hybrid_top1_provenance: tuple[str, str, str] | None
    hybrid_top_k_query_ids: tuple[str, ...]
    hybrid_top_k_provenances: tuple[tuple[str, str, str], ...]
    hybrid_dominant_query_id: str | None
    hybrid_dominant_count: int
    hybrid_dominant_ratio: float | None
    hybrid_concentrated: bool
    mature: bool
    maturity_reason: str

    def as_dict(self, final_top) -> dict[str, object]:
        """Return this observation plus retrospective final-evidence comparisons."""
        final_query_id = _query_id(final_top)
        final_provenance = final_top.provenance if final_top is not None else None
        return {
            "partial": self.partial,
            "offset_after_speech_end_ms": self.offset_after_speech_end_ms,
            "semantic_latency_ms": self.semantic_latency_ms,
            "bm25_latency_ms": self.bm25_latency_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "semantic_top1_query_id": self.semantic_top1_query_id,
            "semantic_top1_score": self.semantic_top1_score,
            "semantic_top_k_query_ids": list(self.semantic_top_k_query_ids),
            "semantic_dominant_query_id": self.semantic_dominant_query_id,
            "semantic_dominant_count": self.semantic_dominant_count,
            "semantic_dominant_ratio": self.semantic_dominant_ratio,
            "semantic_concentrated": self.semantic_concentrated,
            "bm25_top_k_query_ids": list(self.bm25_top_k_query_ids),
            "semantic_dominant_query_id_in_bm25_top_k": self.semantic_dominant_query_id_in_bm25_top_k,
            "hybrid_top1_query_id": self.hybrid_top1_query_id,
            "hybrid_top1_provenance": list(self.hybrid_top1_provenance) if self.hybrid_top1_provenance else None,
            "hybrid_top_k_query_ids": list(self.hybrid_top_k_query_ids),
            "hybrid_dominant_query_id": self.hybrid_dominant_query_id,
            "hybrid_dominant_count": self.hybrid_dominant_count,
            "hybrid_dominant_ratio": self.hybrid_dominant_ratio,
            "hybrid_concentrated": self.hybrid_concentrated,
            "mature": self.mature,
            "maturity_reason": self.maturity_reason,
            "maturity_rejection_reason": None if self.mature else self.maturity_reason,
            "top1_query_id_matches_final": self.hybrid_top1_query_id == final_query_id if final_query_id else None,
            "top1_exact_evidence_matches_final": self.hybrid_top1_provenance == final_provenance if final_provenance else None,
            "final_exact_evidence_present_in_hybrid_top_k": final_provenance in self.hybrid_top_k_provenances if final_provenance else None,
            "final_query_id_present_in_semantic_top_k": final_query_id in self.semantic_top_k_query_ids if final_query_id else None,
            "final_query_id_present_in_bm25_top_k": final_query_id in self.bm25_top_k_query_ids if final_query_id else None,
            "final_query_id_present_in_hybrid_top_k": final_query_id in self.hybrid_top_k_query_ids if final_query_id else None,
        }


@dataclass(frozen=True)
class EarlyReleaseBenchmarkReport:
    """One utterance's dual-condition benchmark observations."""

    speech_end_seen: bool
    speech_end_at_ms: float | None
    maturity_reached: bool
    maturity_at_ms: float | None
    which_happened_first: str | None
    dual_condition_reached: bool
    dual_condition_at_ms: float | None
    mature_partial: str | None
    latest_partial_at_dual_condition: str | None
    speech_end_partial_changed_materially: bool | None
    early_release_safe: bool
    early_release_reason: str
    early_release_candidate_query_id: str | None
    early_release_candidate_answer: str | None
    final_query_id: str | None
    mature_evidence: list[dict[str, object]]
    dual_condition_evidence: list[dict[str, object]]
    final_evidence: list[dict[str, object]]
    early_release_matches_final: bool | None
    false_early_release: bool
    missed_early_release_opportunity: bool
    speech_end_to_dual_condition_ms: float | None
    early_validation_latency_ms: float | None
    estimated_speech_end_to_answer_ms: float | None
    current_final_stt_latency_ms: float | None
    current_end_of_speech_to_answer_ms: float | None
    latency_saved_ms: float | None
    post_speech_end_partials: list[dict[str, object]]
    post_speech_end_recovery: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-ready field names for the opt-in response."""
        return {
            "speech_end_seen": self.speech_end_seen,
            "speech_end_at_ms": self.speech_end_at_ms,
            "maturity_reached": self.maturity_reached,
            "maturity_at_ms": self.maturity_at_ms,
            "which_happened_first": self.which_happened_first,
            "dual_condition_reached": self.dual_condition_reached,
            "dual_condition_at_ms": self.dual_condition_at_ms,
            "mature_partial": self.mature_partial,
            "latest_partial_at_dual_condition": self.latest_partial_at_dual_condition,
            "speech_end_partial_changed_materially": self.speech_end_partial_changed_materially,
            "early_release_safe": self.early_release_safe,
            "early_release_reason": self.early_release_reason,
            "early_release_candidate_query_id": self.early_release_candidate_query_id,
            "early_release_candidate_answer": self.early_release_candidate_answer,
            "final_query_id": self.final_query_id,
            "mature_evidence": self.mature_evidence,
            "dual_condition_evidence": self.dual_condition_evidence,
            "final_evidence": self.final_evidence,
            "early_release_matches_final": self.early_release_matches_final,
            "false_early_release": self.false_early_release,
            "missed_early_release_opportunity": self.missed_early_release_opportunity,
            "speech_end_to_dual_condition_ms": self.speech_end_to_dual_condition_ms,
            "early_validation_latency_ms": self.early_validation_latency_ms,
            "estimated_speech_end_to_answer_ms": self.estimated_speech_end_to_answer_ms,
            "current_final_stt_latency_ms": self.current_final_stt_latency_ms,
            "current_end_of_speech_to_answer_ms": self.current_end_of_speech_to_answer_ms,
            "latency_saved_ms": self.latency_saved_ms,
            "post_speech_end_partials": self.post_speech_end_partials,
            "post_speech_end_recovery": self.post_speech_end_recovery,
        }


class EarlyReleaseBenchmark:
    """Evaluate one candidate only after both VAD speech end and maturity exist."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        answer_composer: AnswerComposer,
        target_lang: str,
        top_k: int,
        started_at: float,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever
        self._answer_composer = answer_composer
        self._target_lang = target_lang
        self._top_k = top_k
        self._started_at = started_at
        self._latest_partial: str | None = None
        self._mature_partial: str | None = None
        self._mature_retrieval: HybridRetrievalResult | None = None
        self._maturity_at: float | None = None
        self._speech_end_at: float | None = None
        self._dual_condition_at: float | None = None
        self._latest_partial_at_dual_condition: str | None = None
        self._speech_end_partial_changed_materially: bool | None = None
        self._dual_condition_evaluated = False
        self._early_release_safe = False
        self._early_release_reason = "Both VAD speech_end and maturity are required."
        self._early_retrieval: HybridRetrievalResult | None = None
        self._early_answer: ComposedAnswer | None = None
        self._early_validation_latency_ms: float | None = None
        self._post_speech_end_observations: list[PostSpeechEndRetrievalObservation] = []

    def observe_partial(self, transcript: str) -> None:
        """Store the latest non-empty Sarvam partial without altering production retrieval."""
        normalized = _normalize(transcript)
        if normalized:
            self._latest_partial = normalized

    def mark_speech_end(self, speech_end_at: float | None = None) -> None:
        """Record VAD speech end and evaluate only if maturity already exists."""
        if self._speech_end_at is None:
            self._speech_end_at = speech_end_at or time.perf_counter()
        self._evaluate_if_eligible()

    def record_post_speech_end_evaluation(
        self,
        transcript: str,
        retrieval: HybridRetrievalResult,
        decision: MaturityDecision,
        evaluated_at: float,
        semantic_concentrated: bool,
        hybrid_concentrated: bool,
    ) -> None:
        """Capture one pre-existing retrieval only when it completed after speech end."""
        if self._speech_end_at is None:
            return
        semantic_top = retrieval.semantic[0] if retrieval.semantic else None
        hybrid_top = retrieval.fused[0] if retrieval.fused else None
        bm25_ids = tuple(_query_id(chunk) or "" for chunk in retrieval.lexical)
        self._post_speech_end_observations.append(
            PostSpeechEndRetrievalObservation(
                partial=_normalize(transcript),
                offset_after_speech_end_ms=_elapsed_ms(evaluated_at, self._speech_end_at) or 0.0,
                semantic_latency_ms=retrieval.semantic_latency_ms,
                bm25_latency_ms=retrieval.lexical_latency_ms,
                fusion_latency_ms=retrieval.fusion_latency_ms,
                semantic_top1_query_id=_query_id(semantic_top),
                semantic_top1_score=semantic_top.semantic_score if semantic_top is not None else None,
                semantic_top_k_query_ids=tuple(_query_id(chunk) or "" for chunk in retrieval.semantic),
                semantic_dominant_query_id=decision.semantic_dominant_query_id,
                semantic_dominant_count=decision.semantic_dominant_query_count,
                semantic_dominant_ratio=decision.semantic_dominance_ratio,
                semantic_concentrated=semantic_concentrated,
                bm25_top_k_query_ids=bm25_ids,
                semantic_dominant_query_id_in_bm25_top_k=bool(
                    decision.semantic_dominant_query_id and decision.semantic_dominant_query_id in bm25_ids
                ),
                hybrid_top1_query_id=_query_id(hybrid_top),
                hybrid_top1_provenance=hybrid_top.provenance if hybrid_top is not None else None,
                hybrid_top_k_query_ids=tuple(_query_id(chunk) or "" for chunk in retrieval.fused),
                hybrid_top_k_provenances=tuple(chunk.provenance for chunk in retrieval.fused),
                hybrid_dominant_query_id=decision.hybrid_dominant_query_id,
                hybrid_dominant_count=decision.hybrid_dominant_query_count,
                hybrid_dominant_ratio=decision.hybrid_dominance_ratio,
                hybrid_concentrated=hybrid_concentrated,
                mature=decision.mature,
                maturity_reason=decision.reason,
            )
        )

    def record_mature(
        self,
        transcript: str,
        retrieval: HybridRetrievalResult,
        maturity_at: float,
    ) -> None:
        """Record first maturity and evaluate only if VAD speech end already occurred."""
        if self._maturity_at is not None:
            return
        self._mature_partial = _normalize(transcript)
        self._mature_retrieval = retrieval
        self._maturity_at = maturity_at
        self._evaluate_if_eligible()

    def finalize(
        self,
        final_retrieval: HybridRetrievalResult,
        final_transcript_at: float,
        current_speech_end_to_answer_ms: float | None,
    ) -> EarlyReleaseBenchmarkReport:
        """Compare the one benchmark candidate with normal final retrieval evidence."""
        final_top = final_retrieval.fused[0] if final_retrieval.fused else None
        early_top = self._early_retrieval.fused[0] if self._early_retrieval and self._early_retrieval.fused else None
        matches_final = (
            early_top.provenance == final_top.provenance
            if early_top is not None and final_top is not None
            else None
        )
        false_early_release = bool(self._early_release_safe and matches_final is False)
        missed_opportunity = bool(
            self._dual_condition_evaluated and not self._early_release_safe and matches_final is True
        )
        speech_end_to_dual = _elapsed_ms(self._dual_condition_at, self._speech_end_at)
        # Composition runs during validation, so it is included exactly once here.
        estimated_latency = (
            speech_end_to_dual + self._early_validation_latency_ms
            if self._early_release_safe
            and speech_end_to_dual is not None
            and self._early_validation_latency_ms is not None
            else None
        )
        latency_saved = (
            current_speech_end_to_answer_ms - estimated_latency
            if current_speech_end_to_answer_ms is not None and estimated_latency is not None
            else None
        )
        post_speech_end_partials = [
            observation.as_dict(final_top) for observation in self._post_speech_end_observations
        ]
        post_speech_end_recovery = _recovery_summary(
            self._post_speech_end_observations,
            final_top,
            _elapsed_ms(self._maturity_at, self._speech_end_at),
            _elapsed_ms(final_transcript_at, self._speech_end_at),
        )
        return EarlyReleaseBenchmarkReport(
            speech_end_seen=self._speech_end_at is not None,
            speech_end_at_ms=_elapsed_ms(self._speech_end_at, self._started_at),
            maturity_reached=self._maturity_at is not None,
            maturity_at_ms=_elapsed_ms(self._maturity_at, self._started_at),
            which_happened_first=_which_happened_first(self._speech_end_at, self._maturity_at),
            dual_condition_reached=self._dual_condition_evaluated,
            dual_condition_at_ms=_elapsed_ms(self._dual_condition_at, self._started_at),
            mature_partial=self._mature_partial,
            latest_partial_at_dual_condition=self._latest_partial_at_dual_condition,
            speech_end_partial_changed_materially=self._speech_end_partial_changed_materially,
            early_release_safe=self._early_release_safe,
            early_release_reason=self._early_release_reason,
            early_release_candidate_query_id=_query_id(early_top),
            early_release_candidate_answer=self._early_answer.text if self._early_answer else None,
            final_query_id=_query_id(final_top),
            mature_evidence=_evidence_summary(self._mature_retrieval),
            dual_condition_evidence=_evidence_summary(self._early_retrieval),
            final_evidence=_evidence_summary(final_retrieval),
            early_release_matches_final=matches_final,
            false_early_release=false_early_release,
            missed_early_release_opportunity=missed_opportunity,
            speech_end_to_dual_condition_ms=speech_end_to_dual,
            early_validation_latency_ms=self._early_validation_latency_ms,
            estimated_speech_end_to_answer_ms=estimated_latency,
            current_final_stt_latency_ms=_elapsed_ms(final_transcript_at, self._speech_end_at),
            current_end_of_speech_to_answer_ms=current_speech_end_to_answer_ms,
            latency_saved_ms=latency_saved,
            post_speech_end_partials=post_speech_end_partials,
            post_speech_end_recovery=post_speech_end_recovery,
        )

    def _evaluate_if_eligible(self) -> None:
        """Run deterministic checks exactly once at the dual-condition boundary."""
        if (
            self._dual_condition_evaluated
            or self._speech_end_at is None
            or self._maturity_at is None
            or self._mature_retrieval is None
            or not self._mature_partial
        ):
            return
        self._dual_condition_evaluated = True
        self._dual_condition_at = max(self._speech_end_at, self._maturity_at)
        self._latest_partial_at_dual_condition = self._latest_partial
        validation_started_at = time.perf_counter()
        if not self._latest_partial_at_dual_condition or not self._mature_retrieval.fused:
            self._early_release_reason = "No non-empty partial or mature evidence is available at the dual-condition boundary."
            self._finish_validation(validation_started_at)
            return

        candidate_retrieval = self._mature_retrieval
        self._speech_end_partial_changed_materially = _materially_changed(
            self._mature_partial,
            self._latest_partial_at_dual_condition,
        )
        if self._speech_end_partial_changed_materially:
            candidate_retrieval = self._hybrid_retriever.retrieve(
                self._latest_partial_at_dual_condition,
                top_k=self._top_k,
                target_lang=self._target_lang,
            )
            if not _evidence_family_is_consistent(self._mature_retrieval, candidate_retrieval):
                self._early_retrieval = candidate_retrieval
                self._early_release_reason = "Dual-condition retrieval no longer supports the mature evidence family."
                self._finish_validation(validation_started_at)
                return

        self._early_retrieval = candidate_retrieval
        candidate_top = candidate_retrieval.fused[0] if candidate_retrieval.fused else None
        if candidate_top is None or candidate_top.semantic_score is None:
            self._early_release_reason = "Top dual-condition evidence has no semantic confidence score."
            self._finish_validation(validation_started_at)
            return
        if candidate_top.semantic_score < self._answer_composer.min_retrieval_score:
            self._early_release_reason = "Top dual-condition evidence does not meet the existing semantic confidence guardrail."
            self._finish_validation(validation_started_at)
            return

        composer_evidence = [
            converted
            for chunk in candidate_retrieval.fused
            if (converted := chunk.as_retrieved_chunk()) is not None
        ]
        self._early_answer = self._answer_composer.compose(
            self._latest_partial_at_dual_condition,
            composer_evidence,
            max_sentences=3,
        )
        if self._early_answer.is_no_answer:
            self._early_release_reason = "AnswerComposer rejected the dual-condition evidence."
            self._finish_validation(validation_started_at)
            return
        self._early_release_safe = True
        self._early_release_reason = (
            "Mature evidence remained semantically confident and consistent at the dual-condition boundary."
        )
        self._finish_validation(validation_started_at)

    def _finish_validation(self, validation_started_at: float) -> None:
        self._early_validation_latency_ms = (time.perf_counter() - validation_started_at) * 1_000


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _materially_changed(first: str, second: str) -> bool:
    """Compare meaningful terms, intentionally ignoring punctuation and whitespace."""
    return meaningful_tokens(first) != meaningful_tokens(second)


def _which_happened_first(speech_end_at: float | None, maturity_at: float | None) -> str | None:
    if speech_end_at is None or maturity_at is None:
        return None
    return "speech_end" if speech_end_at <= maturity_at else "maturity"


def _dominant_query_id(retrieval: HybridRetrievalResult) -> str | None:
    if not retrieval.fused:
        return None
    counts = Counter(_query_id(chunk) or "" for chunk in retrieval.fused)
    first_rank = {_query_id(chunk) or "": chunk.rank for chunk in retrieval.fused}
    return min(counts, key=lambda query_id: (-counts[query_id], first_rank[query_id], query_id)) or None


def _evidence_family_is_consistent(
    mature: HybridRetrievalResult,
    candidate: HybridRetrievalResult,
) -> bool:
    """Require the same top query, exact provenance, and dominant query family."""
    if not mature.fused or not candidate.fused:
        return False
    mature_top = mature.fused[0]
    candidate_top = candidate.fused[0]
    return bool(
        _query_id(mature_top) == _query_id(candidate_top)
        and mature_top.provenance in {chunk.provenance for chunk in candidate.fused}
        and _dominant_query_id(mature) == _dominant_query_id(candidate)
    )


def _query_id(chunk) -> str | None:
    if chunk is None:
        return None
    value = chunk.metadata.get("query_id")
    return str(value) if value is not None else None


def _evidence_summary(retrieval: HybridRetrievalResult | None) -> list[dict[str, object]]:
    """Expose only benchmark provenance and separate score semantics."""
    if retrieval is None:
        return []
    return [
        {
            "query_id": chunk.metadata.get("query_id"),
            "passage_index": chunk.metadata.get("passage_index"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "semantic_score": chunk.semantic_score,
            "bm25_score": chunk.bm25_score,
            "fused_score": chunk.fused_score,
        }
        for chunk in retrieval.fused
    ]


def _recovery_summary(
    observations: list[PostSpeechEndRetrievalObservation],
    final_top,
    maturity_after_speech_end_ms: float | None,
    final_after_speech_end_ms: float | None,
) -> dict[str, object]:
    """Compare post-speech-end retrievals with the final trusted evidence."""
    final_query_id = _query_id(final_top)
    final_provenance = final_top.provenance if final_top is not None else None
    first_query_id = _first_offset(
        observations,
        lambda observation: bool(final_query_id and final_query_id in observation.hybrid_top_k_query_ids),
    )
    first_exact = _first_offset(
        observations,
        lambda observation: bool(final_provenance and final_provenance in observation.hybrid_top_k_provenances),
    )
    first_defensible = _first_offset(
        observations,
        lambda observation: _is_defensibly_trustworthy(observation, final_query_id, final_provenance),
    )
    first_consecutive = _first_consecutive_final_top1_offset(observations, final_query_id)
    maturity_delay = (
        maturity_after_speech_end_ms - first_exact
        if maturity_after_speech_end_ms is not None and first_exact is not None
        else None
    )
    defensible_lead = (
        maturity_after_speech_end_ms - first_defensible
        if maturity_after_speech_end_ms is not None and first_defensible is not None
        else None
    )
    policy_limited = bool(defensible_lead is not None and defensible_lead >= POLICY_LIMITED_LEAD_MS)
    return {
        "final_top1_query_id": final_query_id,
        "final_top1_provenance": list(final_provenance) if final_provenance else None,
        "post_speech_end_partial_count": len(observations),
        "earliest_final_query_id_recovery_ms": first_query_id,
        "earliest_exact_final_evidence_recovery_ms": first_exact,
        "first_correct_evidence_ms": first_exact,
        "first_consecutive_correct_evidence_ms": first_consecutive,
        "first_defensibly_trustworthy_evidence_ms": first_defensible,
        "maturity_reached_ms_after_speech_end": maturity_after_speech_end_ms,
        "transcript_final_ms_after_speech_end": final_after_speech_end_ms,
        "maturity_delay_after_first_correct_evidence_ms": maturity_delay,
        "policy_limited_lead_threshold_ms": POLICY_LIMITED_LEAD_MS,
        "conclusion": "POLICY_LIMITED" if policy_limited else "STT_LIMITED",
        "conclusion_reason": (
            "Exact, concentrated, BM25-corroborated final evidence was available at least "
            f"{POLICY_LIMITED_LEAD_MS:.0f} ms before maturity."
            if policy_limited
            else "No defensibly trustworthy final evidence led maturity by the benchmark threshold."
        ),
    }


def _first_offset(
    observations: list[PostSpeechEndRetrievalObservation],
    predicate,
) -> float | None:
    for observation in observations:
        if predicate(observation):
            return observation.offset_after_speech_end_ms
    return None


def _is_defensibly_trustworthy(
    observation: PostSpeechEndRetrievalObservation,
    final_query_id: str | None,
    final_provenance: tuple[str, str, str] | None,
) -> bool:
    """Keep the analysis stricter than a one-time final-query-ID match."""
    return bool(
        final_query_id
        and final_provenance
        and final_provenance in observation.hybrid_top_k_provenances
        and observation.semantic_dominant_query_id == final_query_id
        and observation.semantic_concentrated
        and final_query_id in observation.bm25_top_k_query_ids
        and observation.hybrid_dominant_query_id == final_query_id
        and observation.hybrid_concentrated
    )


def _first_consecutive_final_top1_offset(
    observations: list[PostSpeechEndRetrievalObservation],
    final_query_id: str | None,
) -> float | None:
    """Return the second observation's time: when two confirmations are known."""
    if not final_query_id:
        return None
    for previous, current in zip(observations, observations[1:]):
        if previous.hybrid_top1_query_id == final_query_id and current.hybrid_top1_query_id == final_query_id:
            return current.offset_after_speech_end_ms
    return None


def _elapsed_ms(later: float | None, earlier: float | None) -> float | None:
    return (later - earlier) * 1_000 if later is not None and earlier is not None else None
