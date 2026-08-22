"""Benchmark-only BM25 and reciprocal-rank fusion helpers for indexed chunks."""

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IndexedDocument:
    """One current-Qdrant chunk kept with the provenance needed for comparison."""

    text: str
    query_id: str
    passage_index: str
    chunk_index: str
    target_lang: str

    @property
    def provenance(self) -> tuple[str, str, str]:
        return self.query_id, self.passage_index, self.chunk_index


@dataclass(frozen=True)
class RankedResult:
    """A rank-list entry with its source score and optional fused score."""

    rank: int
    document: IndexedDocument
    source_score: float | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None

    @property
    def provenance(self) -> tuple[str, str, str]:
        return self.document.provenance


def tokenize_hindi(text: str) -> list[str]:
    """Casefold and remove punctuation while retaining Unicode letters and marks."""
    normalized = []
    for character in text.casefold():
        category = unicodedata.category(character)
        normalized.append(" " if category.startswith(("P", "S")) else character)
    return " ".join("".join(normalized).split()).split()


class BM25Index:
    """Small BM25 index intended only for the current development Qdrant sample."""

    def __init__(self, documents: Iterable[IndexedDocument], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self._tokens = [tokenize_hindi(document.text) for document in self.documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

    def search(self, query: str, top_k: int) -> list[RankedResult]:
        """Return BM25-ranked current-index chunks for a query."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = tokenize_hindi(query)
        if not query_tokens or not self.documents:
            return []
        scores = [self._score(tokens, query_tokens) for tokens in self._tokens]
        ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], self.documents[index].provenance))
        return [
            RankedResult(rank, self.documents[index], source_score=scores[index], lexical_score=scores[index])
            for rank, index in enumerate(ordered[:top_k], start=1)
            if scores[index] > 0
        ]

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


def reciprocal_rank_fusion(
    semantic_results: list[RankedResult], lexical_results: list[RankedResult], rrf_k: int, top_k: int
) -> list[RankedResult]:
    """Fuse rank lists without combining incomparable semantic and BM25 score scales."""
    if rrf_k < 1 or top_k < 1:
        raise ValueError("rrf_k and top_k must be at least 1")
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    for result in semantic_results:
        entry = candidates.setdefault(result.provenance, {"document": result.document, "semantic": None, "lexical": None, "rrf": 0.0})
        entry["semantic"] = result.source_score
        entry["rrf"] = float(entry["rrf"]) + 1 / (rrf_k + result.rank)
    for result in lexical_results:
        entry = candidates.setdefault(result.provenance, {"document": result.document, "semantic": None, "lexical": None, "rrf": 0.0})
        entry["lexical"] = result.source_score
        entry["rrf"] = float(entry["rrf"]) + 1 / (rrf_k + result.rank)

    ordered = sorted(candidates.values(), key=lambda entry: (-float(entry["rrf"]), entry["document"].provenance))
    return [
        RankedResult(
            rank=rank,
            document=entry["document"],
            source_score=entry["semantic"] if entry["semantic"] is not None else entry["lexical"],
            semantic_score=entry["semantic"],
            lexical_score=entry["lexical"],
            fused_score=float(entry["rrf"]),
        )
        for rank, entry in enumerate(ordered[:top_k], start=1)
    ]


def provenance_overlap_ratio(left: list[RankedResult], right: list[RankedResult], limit: int) -> float | None:
    """Compute exact provenance overlap among the first ``limit`` entries of both lists."""
    denominator = min(limit, len(left), len(right))
    if denominator == 0:
        return None
    return len({item.provenance for item in left[:limit]} & {item.provenance for item in right[:limit]}) / denominator
