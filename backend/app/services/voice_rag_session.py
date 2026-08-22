"""Per-utterance trusted-partial lifecycle for realtime STT and deterministic RAG."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.rag.generation.answer_composer import AnswerComposer, ComposedAnswer
from app.rag.generation.answer_formatter import format_composed_answer
from app.rag.language_config import get_application_language
from app.rag.retrieval.hybrid_retriever import HybridRetrievalResult, HybridRetriever
from app.rag.retrieval.maturity import MaturityDecision, MaturityPolicy, assess_retrieval_maturity, has_enough_meaningful_tokens
from app.services.early_release_benchmark import EarlyReleaseBenchmark, EarlyReleaseBenchmarkReport


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class VoiceRAGLatency:
    """Per-session timings, excluding dependency startup and BM25 construction."""

    websocket_connection_setup: float | None
    first_partial: float | None
    maturity_at: float | None
    hybrid_retrieval: float | None
    maturity_decision: float | None
    final_stt: float | None
    final_validation: float
    composer: float
    end_of_speech_to_answer: float | None


@dataclass(frozen=True)
class VoiceRAGFinalResult:
    """Trusted final transcript, grounded answer, provenance, and validation state."""

    transcript: str
    answer: ComposedAnswer
    evidence: list
    latency_ms: VoiceRAGLatency
    mature_partial_used: bool
    final_rerun_required: bool


@dataclass(frozen=True)
class TrustedPartialCandidate:
    """Best mature partial evidence seen so far; never a substitute for final validation."""

    partial_text: str
    evidence: HybridRetrievalResult
    top_query_id: str | None
    top_provenance: tuple[str, str, str] | None
    semantic_confidence: float | None
    maturity_signals: MaturityDecision
    answer_candidate: ComposedAnswer
    created_at: float
    updated_at: float


@dataclass
class PartialEvaluationMetrics:
    """Per-session counters for bounded partial-retrieval work."""

    partials_received: int = 0
    partials_prefiltered: int = 0
    retrieval_evaluations_before_maturity: int = 0
    post_maturity_partials_seen: int = 0
    post_maturity_partials_ignored: int = 0
    post_maturity_retrievals_triggered: int = 0
    stale_partial_evaluations_dropped: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return JSON-ready counters; prefiltered means rejected before retrieval."""
        return {
            "partials_received": self.partials_received,
            "partials_prefiltered": self.partials_prefiltered,
            "retrieval_evaluations_before_maturity": self.retrieval_evaluations_before_maturity,
            "post_maturity_partials_seen": self.post_maturity_partials_seen,
            "post_maturity_partials_ignored": self.post_maturity_partials_ignored,
            "post_maturity_retrievals_triggered": self.post_maturity_retrievals_triggered,
            "stale_partial_evaluations_dropped": self.stale_partial_evaluations_dropped,
        }


class VoiceRAGSession:
    """Evaluate bounded partials, preserve trusted evidence, then validate against final STT."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        answer_composer: AnswerComposer,
        target_lang: str,
        maturity_policy: MaturityPolicy | None = None,
        top_k: int = 5,
        early_release_benchmark: EarlyReleaseBenchmark | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.answer_composer = answer_composer
        self.target_lang = target_lang
        self.maturity_policy = maturity_policy or MaturityPolicy()
        self.top_k = top_k
        self.started_at = time.perf_counter()
        self.early_release_benchmark = early_release_benchmark
        self.connection_setup_at: float | None = None
        self.first_partial_at: float | None = None
        self.speech_end_at: float | None = None
        self.last_evaluated_terms: set[str] = set()
        self._last_queued_terms: set[str] = set()
        self._pending_partial: str | None = None
        self._partial_evaluation_task: asyncio.Task[None] | None = None
        self.partial_metrics = PartialEvaluationMetrics()

        self.trusted_candidate: TrustedPartialCandidate | None = None
        self.trusted_candidate_updates = 0
        self.trusted_candidate_conflicts = 0
        self.maturity_at: float | None = None
        self.mature_hybrid_latency_ms: float | None = None
        self.mature_decision_latency_ms: float | None = None

    @property
    def trusted_candidate_exists(self) -> bool:
        """Whether a mature, evidence-backed partial is currently frozen."""
        return self.trusted_candidate is not None

    @property
    def partial_evaluation_active(self) -> bool:
        """Expose the one-worker bound for focused tests and diagnostics."""
        return self._partial_evaluation_task is not None and not self._partial_evaluation_task.done()

    @property
    def pending_partial_count(self) -> int:
        """The pending queue is intentionally capped at the newest single partial."""
        return int(self._pending_partial is not None)

    def mark_connection_ready(self) -> None:
        """Retain connection setup time when the upstream STT socket is ready."""
        self.connection_setup_at = time.perf_counter()

    def mark_speech_end(self) -> None:
        """Mark provider VAD end-of-speech for end-to-answer timing."""
        self.speech_end_at = time.perf_counter()
        if self.early_release_benchmark is not None:
            self.early_release_benchmark.mark_speech_end(self.speech_end_at)

    def submit_partial(self, transcript: str) -> bool:
        """Queue the newest useful partial without blocking Sarvam event reception."""
        self.partial_metrics.partials_received += 1
        normalized = self._observe_and_normalize(transcript)
        post_maturity = self.trusted_candidate is not None
        if post_maturity:
            self.partial_metrics.post_maturity_partials_seen += 1
        if not normalized or not has_enough_meaningful_tokens(normalized, self.maturity_policy):
            self.partial_metrics.partials_prefiltered += 1
            if post_maturity:
                self.partial_metrics.post_maturity_partials_ignored += 1
            return False
        if post_maturity:
            if not self._is_material_change_from_trusted(normalized):
                self.partial_metrics.post_maturity_partials_ignored += 1
                logger.warning("TRUSTED CANDIDATE RETAINED reason=minor post-maturity text change")
                return False
        if not self._is_useful_for_queue(normalized):
            self.partial_metrics.partials_prefiltered += 1
            return False
        if self._pending_partial is not None:
            self.partial_metrics.stale_partial_evaluations_dropped += 1
        self._pending_partial = normalized
        if not self.partial_evaluation_active:
            self._partial_evaluation_task = asyncio.create_task(self._drain_partial_evaluations())
        return True

    async def wait_for_partial_evaluations(self) -> None:
        """Drain the single active worker before final retrieval begins."""
        task = self._partial_evaluation_task
        if task is not None:
            await task

    def handle_partial(self, transcript: str) -> MaturityDecision | None:
        """Synchronously evaluate one useful partial for focused tests and controlled callers."""
        self.partial_metrics.partials_received += 1
        normalized = self._observe_and_normalize(transcript)
        post_maturity = self.trusted_candidate is not None
        if post_maturity:
            self.partial_metrics.post_maturity_partials_seen += 1
        if not normalized or not has_enough_meaningful_tokens(normalized, self.maturity_policy):
            self.partial_metrics.partials_prefiltered += 1
            if post_maturity:
                self.partial_metrics.post_maturity_partials_ignored += 1
            return None
        if post_maturity:
            if not self._is_material_change_from_trusted(normalized):
                self.partial_metrics.post_maturity_partials_ignored += 1
                logger.warning("TRUSTED CANDIDATE RETAINED reason=minor post-maturity text change")
                return None
        if not self._claim_for_evaluation(normalized):
            self.partial_metrics.partials_prefiltered += 1
            return None
        retrieval = self.hybrid_retriever.retrieve(normalized, top_k=self.top_k, target_lang=self.target_lang)
        return self._process_retrieval(normalized, retrieval, time.perf_counter())

    async def _drain_partial_evaluations(self) -> None:
        """Run at most one retrieval at once and replace stale queued partials with the newest."""
        try:
            while self._pending_partial is not None:
                partial = self._pending_partial
                self._pending_partial = None
                if not self._claim_for_evaluation(partial):
                    continue
                try:
                    retrieval = await asyncio.to_thread(
                        self.hybrid_retriever.retrieve,
                        partial,
                        self.top_k,
                        self.target_lang,
                    )
                except Exception as error:
                    logger.warning("PARTIAL EVALUATION ERROR exception_type=%s", type(error).__name__)
                    continue
                self._process_retrieval(partial, retrieval, time.perf_counter())
        finally:
            self._partial_evaluation_task = None

    def _observe_and_normalize(self, transcript: str) -> str:
        normalized = " ".join(transcript.split())
        if self.early_release_benchmark is not None:
            self.early_release_benchmark.observe_partial(normalized)
        return normalized

    def _is_useful_for_queue(self, normalized: str) -> bool:
        if not normalized or not has_enough_meaningful_tokens(normalized, self.maturity_policy):
            return False
        current_terms = self._meaningful_terms(normalized)
        if not current_terms - self._last_queued_terms:
            return False
        self._last_queued_terms = current_terms
        if self.first_partial_at is None:
            self.first_partial_at = time.perf_counter()
        return True

    def _claim_for_evaluation(self, normalized: str) -> bool:
        if not normalized or not has_enough_meaningful_tokens(normalized, self.maturity_policy):
            return False
        current_terms = self._meaningful_terms(normalized)
        if not current_terms - self.last_evaluated_terms:
            return False
        self.last_evaluated_terms = current_terms
        if self.first_partial_at is None:
            self.first_partial_at = time.perf_counter()
        if self.trusted_candidate is None:
            self.partial_metrics.retrieval_evaluations_before_maturity += 1
        else:
            if not self._is_material_change_from_trusted(normalized):
                self.partial_metrics.post_maturity_partials_ignored += 1
                return False
            self.partial_metrics.post_maturity_retrievals_triggered += 1
        return True

    def _process_retrieval(
        self,
        normalized: str,
        retrieval: HybridRetrievalResult,
        retrieval_completed_at: float,
    ) -> MaturityDecision:
        decision_started_at = time.perf_counter()
        decision = assess_retrieval_maturity(
            normalized,
            retrieval.semantic,
            retrieval.lexical,
            retrieval.fused,
            self.maturity_policy,
        )
        decision_latency_ms = (time.perf_counter() - decision_started_at) * 1_000
        logger.warning("PARTIAL EVALUATED text_length=%d mature=%s", len(normalized), decision.mature)
        if self.early_release_benchmark is not None:
            self.early_release_benchmark.record_post_speech_end_evaluation(
                normalized,
                retrieval,
                decision,
                retrieval_completed_at,
                semantic_concentrated=self._is_concentrated(
                    decision.semantic_dominant_query_count,
                    decision.semantic_dominance_ratio,
                    self.maturity_policy.semantic_dominant_min_count,
                ),
                hybrid_concentrated=self._is_concentrated(
                    decision.hybrid_dominant_query_count,
                    decision.hybrid_dominance_ratio,
                    self.maturity_policy.hybrid_dominant_min_count,
                ),
            )
        if not decision.mature:
            if self.trusted_candidate is not None:
                logger.warning("TRUSTED CANDIDATE RETAINED reason=latest partial not mature")
            return decision

        evaluated_at = time.perf_counter()
        if self.maturity_at is None:
            self.maturity_at = evaluated_at
            self.mature_hybrid_latency_ms = retrieval.total_latency_ms
            self.mature_decision_latency_ms = decision_latency_ms
        if self.early_release_benchmark is not None:
            self.early_release_benchmark.record_mature(normalized, retrieval, evaluated_at)
        self._consider_trusted_candidate(normalized, retrieval, decision, evaluated_at)
        return decision

    def _consider_trusted_candidate(
        self,
        partial: str,
        retrieval: HybridRetrievalResult,
        decision: MaturityDecision,
        evaluated_at: float,
    ) -> None:
        candidate = self._build_candidate(partial, retrieval, decision, evaluated_at)
        if self.trusted_candidate is None:
            self.trusted_candidate = candidate
            logger.warning(
                "TRUSTED CANDIDATE CREATED query_id=%s provenance=%s semantic_confidence=%s",
                candidate.top_query_id,
                candidate.top_provenance,
                candidate.semantic_confidence,
            )
            return

        trusted = self.trusted_candidate
        if not self._same_evidence_family(trusted, candidate):
            self.trusted_candidate_conflicts += 1
            logger.warning(
                "TRUSTED CANDIDATE CONFLICT old_query_id=%s new_query_id=%s",
                trusted.top_query_id,
                candidate.top_query_id,
            )
            return
        if not self._is_at_least_as_trustworthy(candidate, trusted):
            logger.warning("TRUSTED CANDIDATE RETAINED reason=same evidence family but weaker corroboration")
            return

        self.trusted_candidate = TrustedPartialCandidate(
            partial_text=candidate.partial_text,
            evidence=candidate.evidence,
            top_query_id=candidate.top_query_id,
            top_provenance=candidate.top_provenance,
            semantic_confidence=candidate.semantic_confidence,
            maturity_signals=candidate.maturity_signals,
            answer_candidate=candidate.answer_candidate,
            created_at=trusted.created_at,
            updated_at=evaluated_at,
        )
        self.trusted_candidate_updates += 1
        logger.warning("TRUSTED CANDIDATE UPDATED reason=same evidence family / stronger corroboration")

    def _build_candidate(
        self,
        partial: str,
        retrieval: HybridRetrievalResult,
        decision: MaturityDecision,
        evaluated_at: float,
    ) -> TrustedPartialCandidate:
        top = retrieval.fused[0] if retrieval.fused else None
        composer_evidence = [
            converted for chunk in retrieval.fused if (converted := chunk.as_retrieved_chunk()) is not None
        ]
        answer_candidate = self.answer_composer.compose(partial, composer_evidence, max_sentences=3)
        return TrustedPartialCandidate(
            partial_text=partial,
            evidence=retrieval,
            top_query_id=self._query_id(top),
            top_provenance=top.provenance if top is not None else None,
            semantic_confidence=top.semantic_score if top is not None else None,
            maturity_signals=decision,
            answer_candidate=answer_candidate,
            created_at=evaluated_at,
            updated_at=evaluated_at,
        )

    def handle_final(self, transcript: str) -> VoiceRAGFinalResult:
        """Validate the frozen trusted candidate against final STT before answering."""
        final_transcript = " ".join(transcript.split())
        if not final_transcript:
            raise ValueError("final transcript must be non-empty")
        final_stt_at = time.perf_counter()
        final_retrieval = self.hybrid_retriever.retrieve(final_transcript, top_k=self.top_k, target_lang=self.target_lang)
        final_validation_ms = final_retrieval.total_latency_ms
        trusted_reused = self._trusted_candidate_matches_final(final_retrieval)
        trusted = self.trusted_candidate
        trusted_retrieval = trusted.evidence if trusted_reused and trusted is not None else final_retrieval

        composer_started_at = time.perf_counter()
        if trusted_reused and trusted is not None and not trusted.answer_candidate.is_no_answer:
            answer = trusted.answer_candidate
            composer_ms = 0.0
        else:
            composer_evidence = [
                converted for chunk in trusted_retrieval.fused if (converted := chunk.as_retrieved_chunk()) is not None
            ]
            answer = self.answer_composer.compose(final_transcript, composer_evidence, max_sentences=3)
            composer_ms = (time.perf_counter() - composer_started_at) * 1_000
        # Presentation-only pass: it cannot alter trusted evidence or lifecycle decisions.
        answer = format_composed_answer(final_transcript, answer, get_application_language(self.target_lang))
        validation_reason = "trusted evidence matched final evidence" if trusted_reused else "final evidence superseded trusted candidate"
        logger.warning("FINAL VALIDATION trusted_reused=%s reason=%s", trusted_reused, validation_reason)
        answered_at = time.perf_counter()
        result = VoiceRAGFinalResult(
            transcript=final_transcript,
            answer=answer,
            evidence=trusted_retrieval.fused,
            latency_ms=VoiceRAGLatency(
                websocket_connection_setup=self._elapsed_from_start(self.connection_setup_at),
                first_partial=self._elapsed_from_start(self.first_partial_at),
                maturity_at=self._elapsed_from_start(self.maturity_at),
                hybrid_retrieval=self.mature_hybrid_latency_ms,
                maturity_decision=self.mature_decision_latency_ms,
                final_stt=(final_stt_at - self.speech_end_at) * 1_000 if self.speech_end_at else None,
                final_validation=final_validation_ms,
                composer=composer_ms,
                end_of_speech_to_answer=(answered_at - self.speech_end_at) * 1_000 if self.speech_end_at else None,
            ),
            mature_partial_used=trusted_reused,
            final_rerun_required=not trusted_reused,
        )
        if self.early_release_benchmark is not None:
            self.early_release_benchmark_report = self.early_release_benchmark.finalize(
                final_retrieval,
                final_stt_at,
                result.latency_ms.end_of_speech_to_answer,
            )
        return result

    def early_release_report(self) -> EarlyReleaseBenchmarkReport | None:
        """Return an opt-in benchmark report after final transcript validation."""
        return getattr(self, "early_release_benchmark_report", None)

    def trusted_candidate_state(self) -> dict[str, object]:
        """Expose safe internal lifecycle metrics without adding them to normal responses."""
        candidate = self.trusted_candidate
        if candidate is None:
            return {
                "trusted_candidate_exists": False,
                "trusted_candidate_updates": self.trusted_candidate_updates,
                "trusted_candidate_conflicts": self.trusted_candidate_conflicts,
            }
        return {
            "trusted_candidate_exists": True,
            "trusted_partial_text": candidate.partial_text,
            "trusted_top_query_id": candidate.top_query_id,
            "trusted_top_provenance": candidate.top_provenance,
            "trusted_semantic_confidence": candidate.semantic_confidence,
            "trusted_maturity_signals": {
                "semantic_dominant_query_id": candidate.maturity_signals.semantic_dominant_query_id,
                "semantic_dominant_query_count": candidate.maturity_signals.semantic_dominant_query_count,
                "semantic_dominance_ratio": candidate.maturity_signals.semantic_dominance_ratio,
                "hybrid_dominant_query_id": candidate.maturity_signals.hybrid_dominant_query_id,
                "hybrid_dominant_query_count": candidate.maturity_signals.hybrid_dominant_query_count,
                "hybrid_dominance_ratio": candidate.maturity_signals.hybrid_dominance_ratio,
                "hybrid_top1_supported_by_both": candidate.maturity_signals.hybrid_top1_supported_by_both,
            },
            "trusted_answer_candidate": candidate.answer_candidate.text,
            "trusted_created_at_ms": self._elapsed_from_start(candidate.created_at),
            "trusted_updated_at_ms": self._elapsed_from_start(candidate.updated_at),
            "trusted_candidate_updates": self.trusted_candidate_updates,
            "trusted_candidate_conflicts": self.trusted_candidate_conflicts,
        }

    def partial_evaluation_metrics(self) -> dict[str, int]:
        """Return bounded-worker counters without exposing them in normal responses."""
        return self.partial_metrics.as_dict()

    def _trusted_candidate_matches_final(self, final_retrieval: HybridRetrievalResult) -> bool:
        trusted = self.trusted_candidate
        if trusted is None or not trusted.evidence.fused or not final_retrieval.fused:
            return False
        trusted_top = trusted.evidence.fused[0]
        final_top = final_retrieval.fused[0]
        if trusted_top.provenance == final_top.provenance:
            return True
        trusted_provenances = {chunk.provenance for chunk in trusted.evidence.fused}
        final_provenances = {chunk.provenance for chunk in final_retrieval.fused}
        return self._query_id(trusted_top) == self._query_id(final_top) and bool(trusted_provenances & final_provenances)

    @staticmethod
    def _same_evidence_family(
        trusted: TrustedPartialCandidate,
        candidate: TrustedPartialCandidate,
    ) -> bool:
        if trusted.top_query_id != candidate.top_query_id:
            return False
        trusted_provenances = {chunk.provenance for chunk in trusted.evidence.fused}
        candidate_provenances = {chunk.provenance for chunk in candidate.evidence.fused}
        return bool(trusted_provenances & candidate_provenances)

    def _is_material_change_from_trusted(self, partial: str) -> bool:
        """Use only lightweight token overlap before authorizing post-maturity retrieval."""
        trusted = self.trusted_candidate
        if trusted is None:
            return True
        if partial == trusted.partial_text:
            return False
        trusted_terms = self._meaningful_terms(trusted.partial_text)
        partial_terms = self._meaningful_terms(partial)
        if not partial_terms or partial_terms == trusted_terms or partial_terms <= trusted_terms:
            return False
        overlap = len(trusted_terms & partial_terms)
        union = len(trusted_terms | partial_terms)
        return union == 0 or overlap / union < 0.60

    @staticmethod
    def _is_at_least_as_trustworthy(
        candidate: TrustedPartialCandidate,
        trusted: TrustedPartialCandidate,
    ) -> bool:
        """Compare semantic confidence and maturity corroboration without using RRF scores."""
        new_confidence = candidate.semantic_confidence
        old_confidence = trusted.semantic_confidence
        if new_confidence is None or old_confidence is None or new_confidence < old_confidence:
            return False
        new_signals = candidate.maturity_signals
        old_signals = trusted.maturity_signals
        return bool(
            new_signals.semantic_dominant_query_count >= old_signals.semantic_dominant_query_count
            and (new_signals.semantic_dominance_ratio or 0.0) >= (old_signals.semantic_dominance_ratio or 0.0)
            and new_signals.hybrid_dominant_query_count >= old_signals.hybrid_dominant_query_count
            and (new_signals.hybrid_dominance_ratio or 0.0) >= (old_signals.hybrid_dominance_ratio or 0.0)
            and (new_signals.hybrid_top1_supported_by_both or not old_signals.hybrid_top1_supported_by_both)
        )

    @staticmethod
    def _meaningful_terms(text: str) -> set[str]:
        from app.rag.retrieval.maturity import meaningful_tokens

        return meaningful_tokens(text)

    def _is_concentrated(self, dominant_count: int, dominance_ratio: float | None, minimum_count: int) -> bool:
        """Expose the current policy's concentration signals to opt-in instrumentation."""
        return bool(
            dominant_count >= minimum_count
            and dominance_ratio is not None
            and dominance_ratio >= self.maturity_policy.dominant_min_ratio
        )

    @staticmethod
    def _query_id(chunk) -> str | None:
        if chunk is None:
            return None
        value = chunk.metadata.get("query_id")
        return str(value) if value is not None else None

    def _elapsed_from_start(self, timestamp: float | None) -> float | None:
        return (timestamp - self.started_at) * 1_000 if timestamp is not None else None
