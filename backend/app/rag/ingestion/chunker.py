"""E5-tokenizer-based chunking for retrieval documents."""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Protocol

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from app.rag.ingestion.preprocessor import RetrievalDocument


E5_TOKENIZER_NAME = "intfloat/multilingual-e5-base"
SENTENCE_OVERLAP_COUNT = 1
TOKEN_WINDOW_OVERLAP = 32
SENTENCE_PATTERN = re.compile(r".+?(?:[।.!?]+(?=\s|$)|$)", flags=re.DOTALL)


class SimpleFallbackTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens = text.split()
        return [hash(t) % 30000 for t in tokens]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return " ".join(str(tid) for tid in token_ids)

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": self.encode(text)}


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class Chunk:
    """A token-bounded retrieval text segment and its inherited metadata."""

    text: str
    metadata: dict[str, Any]


@lru_cache(maxsize=1)
def get_e5_tokenizer() -> Tokenizer:
    """Load and cache E5's tokenizer or fallback tokenizer."""
    if HAS_TRANSFORMERS:
        try:
            return AutoTokenizer.from_pretrained(E5_TOKENIZER_NAME)
        except Exception:
            pass
    return SimpleFallbackTokenizer()


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _token_ids(text: str, tokenizer: Tokenizer) -> list[int]:
    """Return E5 content tokens without sequence special tokens or length warnings."""
    if callable(tokenizer):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        return list(encoded["input_ids"])
    return tokenizer.encode(text, add_special_tokens=False)


def _token_count(text: str, tokenizer: Tokenizer) -> int:
    """Count E5 content tokens; ``max_tokens`` intentionally excludes special tokens."""
    return len(_token_ids(text, tokenizer))


def _split_sentences(text: str) -> list[str]:
    """Split Hindi and English text while retaining terminal punctuation."""
    normalized_text = _normalize_whitespace(text)
    return [_normalize_whitespace(sentence) for sentence in SENTENCE_PATTERN.findall(normalized_text) if _normalize_whitespace(sentence)]


def _token_window_chunks(sentence: str, max_tokens: int, tokenizer: Tokenizer) -> list[str]:
    """Split an overlong sentence into re-tokenization-checked overlapping windows."""
    token_ids = _token_ids(sentence, tokenizer)
    chunks: list[str] = []
    start = 0

    while start < len(token_ids):
        window_end = min(start + max_tokens, len(token_ids))
        chunk_text = ""
        while window_end > start:
            candidate = _normalize_whitespace(
                tokenizer.decode(token_ids[start:window_end], skip_special_tokens=True)
            )
            if candidate and _token_count(candidate, tokenizer) <= max_tokens:
                chunk_text = candidate
                break
            window_end -= 1

        if not chunk_text:
            raise ValueError("Tokenizer could not decode a valid content-token window")

        chunks.append(chunk_text)
        consumed_tokens = window_end - start
        if window_end == len(token_ids):
            break
        overlap = min(TOKEN_WINDOW_OVERLAP, max(0, consumed_tokens - 1))
        start = window_end - overlap

    return chunks


def _sentence_overlap_chunks(sentences: list[str], max_tokens: int, tokenizer: Tokenizer) -> list[tuple[str, str]]:
    """Group normal sentences within the token limit and retain one-sentence overlap."""
    chunks: list[tuple[str, str]] = []
    current_sentences: list[str] = []

    def flush_current() -> None:
        if current_sentences:
            chunks.append((" ".join(current_sentences), "sentence_overlap"))

    for sentence in sentences:
        if _token_count(sentence, tokenizer) > max_tokens:
            flush_current()
            current_sentences.clear()
            chunks.extend((chunk, "token_window_fallback") for chunk in _token_window_chunks(sentence, max_tokens, tokenizer))
            continue

        if not current_sentences:
            current_sentences.append(sentence)
            continue

        combined = " ".join([*current_sentences, sentence])
        if _token_count(combined, tokenizer) <= max_tokens:
            current_sentences.append(sentence)
            continue

        flush_current()
        overlap_sentences = current_sentences[-SENTENCE_OVERLAP_COUNT:]
        overlap_plus_sentence = " ".join([*overlap_sentences, sentence])
        current_sentences = (
            [*overlap_sentences, sentence]
            if _token_count(overlap_plus_sentence, tokenizer) <= max_tokens
            else [sentence]
        )

    flush_current()
    return chunks


def chunk_retrieval_document(
    document: RetrievalDocument,
    max_tokens: int = 256,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Chunk one retrieval document using E5 tokens and sentence-level overlap."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not isinstance(document, RetrievalDocument):
        raise TypeError("document must be a RetrievalDocument")
    if not isinstance(document.text, str):
        raise ValueError("RetrievalDocument text must be a string")

    cleaned_text = _normalize_whitespace(document.text)
    if not cleaned_text:
        return []

    active_tokenizer = tokenizer or get_e5_tokenizer()
    if _token_count(cleaned_text, active_tokenizer) <= max_tokens:
        chunk_parts = [(cleaned_text, "whole_passage")]
    else:
        chunk_parts = _sentence_overlap_chunks(
            _split_sentences(cleaned_text),
            max_tokens,
            active_tokenizer,
        )

    safe_chunk_parts: list[tuple[str, str]] = []
    for chunk_text, strategy in chunk_parts:
        if _token_count(chunk_text, active_tokenizer) <= max_tokens:
            safe_chunk_parts.append((chunk_text, strategy))
        else:
            safe_chunk_parts.extend(
                (fallback_chunk, "token_window_fallback")
                for fallback_chunk in _token_window_chunks(
                    chunk_text,
                    max_tokens,
                    active_tokenizer,
                )
            )

    chunks: list[Chunk] = []
    for chunk_index, (chunk_text, strategy) in enumerate(safe_chunk_parts):
        token_count = _token_count(chunk_text, active_tokenizer)
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunk_strategy": strategy,
                    "token_count": token_count,
                },
            )
        )
    return chunks


def iter_document_chunks(
    documents: Iterable[RetrievalDocument],
    max_tokens: int = 256,
    tokenizer: Tokenizer | None = None,
) -> Iterable[Chunk]:
    """Yield chunks lazily for multiple retrieval documents."""
    for document in documents:
        yield from chunk_retrieval_document(document, max_tokens=max_tokens, tokenizer=tokenizer)
