"""CPU-light extractive answer composition from ranked retrieval evidence."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.retrieval.retriever import RetrievedChunk


NO_ANSWER_TEXT = "मुझे उपलब्ध संदर्भ में पर्याप्त जानकारी नहीं मिली।"
# Keep Devanagari base letters and combining marks in the same lexical term.
_TERM_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u0900-\u0963\u0966-\u097F]+", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_ENGLISH_INITIALS_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}$")
_DEVANAGARI_INITIALS_PATTERN = re.compile(r"(?:[\u0900-\u097F]+\.){2,}$")
_KNOWN_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "etc.",
    "पी.ए.", "यू.एस.", "यू.के.", "डॉ.", "श्री.", "सं.",
}
_TITLE_ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "डॉ.", "श्री."}
_STOP_TERMS = {
    "और", "का", "की", "के", "को", "की", "कौन", "क्या", "कितनी", "पर", "पहले",
    "में", "मुझे", "यह", "वह", "से", "सबसे", "था", "थे", "है", "हैं", "हो", "the",
    "a", "an", "and", "are", "did", "do", "does", "for", "how", "in", "is", "of", "on",
    "to", "was", "were", "what", "when", "where", "who",
}


@dataclass(frozen=True)
class AnswerEvidence:
    """A selected source sentence and the retrieval provenance that supports it."""

    query_id: Any
    passage_index: Any
    chunk_index: Any
    retrieval_score: float
    source_sentence: str


@dataclass(frozen=True)
class ComposedAnswer:
    """A grounded extractive answer with internal evidence and timing."""

    text: str
    evidence: list[AnswerEvidence]
    confidence: float | None
    latency_ms: float
    is_no_answer: bool


@dataclass(frozen=True)
class _SentenceCandidate:
    sentence: str
    retrieval_score: float
    retrieval_rank: int
    sentence_index: int
    metadata: dict[str, Any]
    relevance: float
    meaningful_match_count: int
    is_direct_answer: bool


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _sentence_terms(text: str) -> set[str]:
    return {term.lower() for term in _TERM_PATTERN.findall(text)}


def _period_is_sentence_boundary(text: str, index: int) -> bool:
    """Return whether a period is a true boundary rather than an abbreviation."""
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    start = index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = index + 1
    while end < len(text) and not text[end].isspace():
        end += 1
    token = text[start:end].strip("\"'“”‘’()[]{}<>,;:!?").casefold()
    if token in _TITLE_ABBREVIATIONS:
        return False
    initials = _ENGLISH_INITIALS_PATTERN.fullmatch(token) or _DEVANAGARI_INITIALS_PATTERN.fullmatch(token)
    if token in _KNOWN_ABBREVIATIONS or initials:
        # The final dot in "U.S. He ..." can still end a sentence.  The
        # earlier dots and abbreviations followed by lower-case continuations
        # remain protected.  Hindi has no reliable capitalization signal, so
        # its abbreviations remain conservative and are not split here.
        following = text[index + 1:].lstrip("\"'“”‘’)]} ")
        final_period = "." not in text[index + 1:end]
        return bool(_ENGLISH_INITIALS_PATTERN.fullmatch(token) and final_period and following and following[0].isupper())
    # A single initial such as "J. R. R. Tolkien" is not a sentence boundary.
    return not re.fullmatch(r"[a-z]\.", token)


def _is_numeric_question(query: str) -> bool:
    """Recognize question forms whose direct evidence normally includes a value."""
    normalized = _normalize_whitespace(query).casefold()
    return any(marker in normalized for marker in (
        "how far", "how many", "how much", "how old", "कितनी", "कितना", "कितने",
    ))


def split_sentences(text: str) -> list[str]:
    """Split Hindi/English text without breaking abbreviations or decimal values."""
    normalized = _normalize_whitespace(text)
    sentences: list[str] = []
    start = 0
    for index, character in enumerate(normalized):
        if character in "।?!" or (character == "." and _period_is_sentence_boundary(normalized, index)):
            sentence = normalized[start:index + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = index + 1
    trailing = normalized[start:].strip()
    if trailing:
        sentences.append(trailing)
    return sentences


class AnswerComposer:
    """Select a few source sentences without calling a model or external service."""

    def __init__(
        self,
        min_retrieval_score: float = 0.001,
        near_duplicate_threshold: float = 0.85,
        min_meaningful_query_matches: int = 1,
    ) -> None:
        if not isinstance(min_retrieval_score, (int, float)):
            raise TypeError("min_retrieval_score must be numeric")
        if not 0 < near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if min_meaningful_query_matches < 1:
            raise ValueError("min_meaningful_query_matches must be at least 1")
        self.min_retrieval_score = float(min_retrieval_score)
        self.near_duplicate_threshold = near_duplicate_threshold
        self.min_meaningful_query_matches = min_meaningful_query_matches

    def compose(
        self,
        query: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        max_sentences: int = 3,
        max_answer_chars: int = 600,
    ) -> ComposedAnswer:
        """Return complete, source-derived sentences or a structured no-answer result."""
        started_at = time.perf_counter()
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized_query = _normalize_whitespace(query)
        if not normalized_query:
            raise ValueError("query must be non-empty")
        if isinstance(max_sentences, bool) or not isinstance(max_sentences, int) or max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        if isinstance(max_answer_chars, bool) or not isinstance(max_answer_chars, int) or max_answer_chars < 1:
            raise ValueError("max_answer_chars must be at least 1")

        query_terms = _sentence_terms(normalized_query) - _STOP_TERMS
        query_numbers = set(_NUMBER_PATTERN.findall(normalized_query))
        candidates = self._collect_candidates(
            retrieved_chunks,
            query_terms,
            query_numbers,
            _is_numeric_question(normalized_query),
        )
        if not candidates:
            return self._no_answer(started_at)

        selected: list[_SentenceCandidate] = []
        answer_chars = 0
        for candidate in sorted(
            candidates,
            key=lambda item: (-int(item.is_direct_answer), -item.relevance, item.retrieval_rank, item.sentence_index),
        ):
            separator_chars = 1 if selected else 0
            if answer_chars + separator_chars + len(candidate.sentence) > max_answer_chars:
                continue
            selected.append(candidate)
            answer_chars += separator_chars + len(candidate.sentence)
            # A complete answer-bearing sentence is stronger than padding the
            # response with loosely related evidence from lower-ranked chunks.
            if candidate.is_direct_answer or len(selected) == max_sentences:
                break

        if not selected:
            return self._no_answer(started_at)
        evidence = [
            AnswerEvidence(
                query_id=candidate.metadata.get("query_id"),
                passage_index=candidate.metadata.get("passage_index"),
                chunk_index=candidate.metadata.get("chunk_index"),
                retrieval_score=candidate.retrieval_score,
                source_sentence=candidate.sentence,
            )
            for candidate in selected
        ]
        return ComposedAnswer(
            text=" ".join(candidate.sentence for candidate in selected),
            evidence=evidence,
            confidence=sum(candidate.retrieval_score for candidate in selected) / len(selected),
            latency_ms=(time.perf_counter() - started_at) * 1_000,
            is_no_answer=False,
        )

    def _collect_candidates(
        self,
        retrieved_chunks: Sequence[RetrievedChunk],
        query_terms: set[str],
        query_numbers: set[str],
        numeric_question: bool,
    ) -> list[_SentenceCandidate]:
        candidates: list[_SentenceCandidate] = []
        seen_sentences: set[str] = set()
        accepted_term_sets: list[set[str]] = []
        for retrieval_rank, chunk in enumerate(retrieved_chunks):
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError("retrieved_chunks must contain RetrievedChunk instances")
            if chunk.score < self.min_retrieval_score or not isinstance(chunk.text, str):
                continue
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            for sentence_index, sentence in enumerate(split_sentences(chunk.text)):
                normalized_sentence = sentence.lower()
                terms = _sentence_terms(sentence)
                if normalized_sentence in seen_sentences or self._is_near_duplicate(terms, accepted_term_sets):
                    continue
                meaningful_matches = self._meaningful_query_matches(query_terms, terms)
                if meaningful_matches < self.min_meaningful_query_matches:
                    continue
                seen_sentences.add(normalized_sentence)
                accepted_term_sets.append(terms)
                overlap = meaningful_matches / len(query_terms) if query_terms else 0.0
                number_overlap = len(query_numbers & set(_NUMBER_PATTERN.findall(sentence)))
                has_numeric_value = bool(_NUMBER_PATTERN.search(sentence))
                # Full query-term coverage is direct factual support.  Numeric
                # questions also accept two matched entities plus a value; this
                # avoids treating a sentence about only one entity as complete.
                is_direct_answer = len(query_terms) >= 2 and (
                    meaningful_matches == len(query_terms)
                    or (numeric_question and has_numeric_value and meaningful_matches >= 2)
                )
                relevance = (0.5 * float(chunk.score)) + (1.5 * overlap) + (0.1 * number_overlap) + (0.01 / (retrieval_rank + 1))
                candidates.append(_SentenceCandidate(
                    sentence, float(chunk.score), retrieval_rank, sentence_index,
                    metadata, relevance, meaningful_matches, is_direct_answer,
                ))
        return candidates

    @staticmethod
    def _meaningful_query_matches(query_terms: set[str], sentence_terms: set[str]) -> int:
        """Count exact or simple prefix matches, handling forms such as दूर/दूरी."""
        matches = 0
        for query_term in query_terms:
            if any(
                query_term == sentence_term
                or (len(query_term) >= 3 and (sentence_term.startswith(query_term) or query_term.startswith(sentence_term)))
                for sentence_term in sentence_terms
            ):
                matches += 1
        return matches

    def _is_near_duplicate(self, terms: set[str], accepted_term_sets: list[set[str]]) -> bool:
        if not terms:
            return False
        for previous_terms in accepted_term_sets:
            union = terms | previous_terms
            if union and len(terms & previous_terms) / len(union) >= self.near_duplicate_threshold:
                return True
        return False

    @staticmethod
    def _no_answer(started_at: float) -> ComposedAnswer:
        return ComposedAnswer(NO_ANSWER_TEXT, [], None, (time.perf_counter() - started_at) * 1_000, True)
