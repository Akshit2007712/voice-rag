"""In-memory BM25 index built once from the active Qdrant collection."""

from __future__ import annotations

import math
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.indexing.vector_store import VectorStore


@dataclass(frozen=True)
class LexicalDocument:
    """One indexed chunk and the provenance needed to rejoin semantic results."""

    text: str
    metadata: dict[str, object]

    @property
    def provenance(self) -> tuple[str, str, str]:
        return tuple(str(self.metadata.get(key, "")) for key in ("query_id", "passage_index", "chunk_index"))


@dataclass(frozen=True)
class BM25Match:
    """A lexical ranking result with its source document and BM25 score."""

    rank: int
    document: LexicalDocument
    score: float


@dataclass(frozen=True)
class BM25Profile:
    """Read-only per-query telemetry for diagnosing inverted-index work."""

    tokenize_ms: float
    candidate_lookup_ms: float
    score_ms: float
    topk_ms: float
    total_bm25_ms: float
    query_terms: tuple[str, ...] = ()
    posting_list_sizes: tuple[tuple[str, int], ...] = ()
    total_posting_entries: int = 0
    candidate_document_count: int = 0


def tokenize_text(text: str) -> list[str]:
    """Tokenize Indic text while retaining combining marks and removing punctuation."""
    normalized = []
    for character in text.casefold():
        category = unicodedata.category(character)
        normalized.append(" " if category.startswith(("P", "S")) else character)
    return " ".join("".join(normalized).split()).split()


class BM25Store:
    """A small, immutable BM25 corpus for the current Qdrant development collection."""

    def __init__(self, documents: Iterable[LexicalDocument], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self._tokens = [tokenize_text(document.text) for document in self.documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: dict[str, int] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for tokens in self._tokens:
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            for token in frequencies:
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1
        for document_index, tokens in enumerate(self._tokens):
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            for token, frequency in frequencies.items():
                self._postings.setdefault(token, []).append((document_index, frequency))

    @classmethod
    def from_vector_store(cls, vector_store: "VectorStore", target_lang: str) -> "BM25Store":
        """Read only current-Qdrant payloads for one language at application startup."""
        if not target_lang.strip():
            raise ValueError("target_lang must be non-empty")
        if not vector_store.collection_exists():
            return cls([])

        documents: list[LexicalDocument] = []
        offset = None
        while True:
            points, offset = vector_store.scroll(
                collection_name=vector_store.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                if payload.get("target_lang") != target_lang:
                    continue
                text = str(payload.pop("text", "")).strip()
                if text:
                    documents.append(LexicalDocument(text=text, metadata=payload))
            if offset is None:
                return cls(documents)

    @classmethod
    def by_language_from_vector_store(
        cls,
        vector_store: "VectorStore",
        target_languages: Sequence[str],
    ) -> dict[str, "BM25Store"]:
        """Build isolated language corpora in one startup-only collection scan."""
        requested = {language for language in target_languages if language.strip()}
        documents: dict[str, list[LexicalDocument]] = {language: [] for language in requested}
        if not requested or not vector_store.collection_exists():
            return {language: cls([]) for language in requested}
        offset = None
        total_scrolled = 0
        max_scroll_items = 5000

        while True:
            points, offset = vector_store.scroll(
                collection_name=vector_store.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            total_scrolled += len(points)
            for point in points:
                payload = dict(point.payload or {})
                target_lang = payload.get("target_lang")
                if target_lang not in documents:
                    continue
                text = str(payload.pop("text", "")).strip()
                if text:
                    documents[target_lang].append(LexicalDocument(text=text, metadata=payload))
            if offset is None or total_scrolled >= max_scroll_items or not points:
                return {language: cls(items) for language, items in documents.items()}


    def search(self, query: str, top_k: int = 5) -> list[BM25Match]:
        """Return the production inverted-index BM25 ranking."""
        return self.search_profiled(query, top_k)[0]

    def search_full_corpus_reference(self, query: str, top_k: int = 5) -> list[BM25Match]:
        """Return the pre-optimization full-corpus ranking for offline validation.

        Production retrieval calls :meth:`search`, never this method.  Keeping the
        original scorer here lets the validation command compare the same tokenizer,
        formula, score ordering, and provenance tie-break without touching Qdrant.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = tokenize_text(query)
        if not query_tokens or not self.documents:
            return []
        scores = [self._score(tokens, query_tokens) for tokens in self._tokens]
        ordered = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], self.documents[index].provenance),
        )[:top_k]
        return [
            BM25Match(rank, self.documents[index], scores[index])
            for rank, index in enumerate(ordered, start=1)
            if scores[index] > 0
        ]

    def search_profiled(self, query: str, top_k: int = 5) -> tuple[list[BM25Match], BM25Profile]:
        """Return lexical results without materializing or changing the corpus."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started = time.perf_counter()
        tokenize_started = started
        query_tokens = tokenize_text(query)
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1_000
        if not query_tokens or not self.documents:
            return [], BM25Profile(tokenize_ms, 0.0, 0.0, 0.0, (time.perf_counter()-started)*1_000)
        lookup_started = time.perf_counter()
        query_terms = set(query_tokens)
        postings = [(term, self._postings.get(term, ())) for term in query_terms]
        candidate_ids = {document_id for _, entries in postings for document_id, _ in entries}
        lookup_ms = (time.perf_counter() - lookup_started) * 1_000
        score_started = time.perf_counter()
        scores = {document_id: 0.0 for document_id in candidate_ids}
        document_count = len(self.documents)
        for term, entries in postings:
            document_frequency = self._document_frequency.get(term, 0)
            inverse_frequency = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_id, frequency in entries:
                denominator = frequency + self.k1 * (1 - self.b + self.b * self._lengths[document_id] / self._average_length)
                scores[document_id] += inverse_frequency * frequency * (self.k1 + 1) / denominator
        score_ms = (time.perf_counter() - score_started) * 1_000
        topk_started = time.perf_counter()
        ordered = sorted(scores, key=lambda index: (-scores[index], self.documents[index].provenance))[:top_k]
        matches = [BM25Match(rank, self.documents[index], scores[index]) for rank, index in enumerate(ordered, start=1) if scores[index] > 0]
        topk_ms = (time.perf_counter() - topk_started) * 1_000
        return matches, BM25Profile(
            tokenize_ms=tokenize_ms,
            candidate_lookup_ms=lookup_ms,
            score_ms=score_ms,
            topk_ms=topk_ms,
            total_bm25_ms=(time.perf_counter() - started) * 1_000,
            query_terms=tuple(sorted(query_terms)),
            posting_list_sizes=tuple(sorted((term, len(entries)) for term, entries in postings)),
            total_posting_entries=sum(len(entries) for _, entries in postings),
            candidate_document_count=len(candidate_ids),
        )

    def _score(self, document_tokens: list[str], query_tokens: list[str]) -> float:
        if not document_tokens or not self._average_length:
            return 0.0
        term_frequency: dict[str, int] = {}
        for token in document_tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        score = 0.0
        document_count = len(self.documents)
        for token in set(query_tokens):
            frequency = term_frequency.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency.get(token, 0)
            inverse_frequency = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + self.k1 * (1 - self.b + self.b * len(document_tokens) / self._average_length)
            score += inverse_frequency * frequency * (self.k1 + 1) / denominator
        return score
