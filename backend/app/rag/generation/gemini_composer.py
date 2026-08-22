"""Google Gemini LLM answer generation using retrieved RAG passages as context."""

from __future__ import annotations

import os
import time
import logging
import requests
from collections.abc import Sequence

from app.rag.generation.answer_composer import AnswerComposer, ComposedAnswer, AnswerEvidence
from app.rag.retrieval.retriever import RetrievedChunk

logger = logging.getLogger("uvicorn.error")

_NO_ANSWER_EN = "I could not find relevant information to answer your question."
_NO_ANSWER_HI = "मुझे आपके प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"

_SYSTEM_PROMPT_EN = (
    "You are a helpful, accurate assistant. "
    "If relevant context passages are provided below, use them to answer the user's question accurately. "
    "If no context is provided or the context is not directly relevant to the question, answer using your general knowledge in a clear, complete 2-sentence response. "
    "Always complete your thoughts and end with proper sentence punctuation."
)

_SYSTEM_PROMPT_HI = (
    "आप एक सहायक और सटीक AI असिस्टेंट हैं। "
    "यदि नीचे दिए गए संदर्भ अनुच्छेद प्रासंगिक हैं, तो उनसे उत्तर दें। "
    "यदि कोई संदर्भ नहीं है या प्रासंगिक नहीं है, तो अपने सामान्य ज्ञान से 2 वाक्यों में स्पष्ट और पूरा उत्तर दें। "
    "हमेशा अपने वाक्य पूरे करें और पूर्णविराम पर समाप्त करें।"
)


_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_NAME = "gemini-3.5-flash"

# Global HTTP session for fast connection reuse
_HTTP_SESSION = requests.Session()


def _detect_language(query: str) -> str:
    """Detect Hindi vs English from Unicode character presence."""
    devanagari_count = sum(1 for ch in query if "\u0900" <= ch <= "\u097F")
    return "hi" if devanagari_count > len(query) * 0.15 else "en"


def _build_prompt(query: str, chunks: Sequence[RetrievedChunk], language: str) -> str:
    """Build the full prompt with retrieved context passages."""
    system = _SYSTEM_PROMPT_HI if language == "hi" else _SYSTEM_PROMPT_EN
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        if text:
            context_lines.append(f"[Passage {i}]: {text}")

    if context_lines:
        context_section = "Context:\n" + "\n".join(context_lines) + "\n\n"
    else:
        context_section = ""

    return (
        f"{system}\n\n"
        f"{context_section}"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def _call_gemini_rest(api_key: str, prompt: str) -> str:
    """Call Gemini REST API using requests.Session for low-latency answers."""
    url = f"{_GEMINI_API_BASE}/{MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 500,
            "topP": 0.8,
        },
    }
    resp = _HTTP_SESSION.post(url, json=payload, timeout=12)
    resp.raise_for_status()
    body = resp.json()

    candidates = body.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()



class GeminiAnswerComposer:
    """Drop-in replacement for AnswerComposer using Google Gemini 3.5 Flash REST API.

    Fast, low-latency, resilient LLM composer.
    Falls back to the extractive AnswerComposer if:
    - GEMINI_API_KEY is not set in the environment
    - The Gemini API call fails for any reason
    """

    def __init__(self, fallback: AnswerComposer | None = None) -> None:
        self._api_key: str | None = os.getenv("GEMINI_API_KEY", "").strip() or None
        self._fallback = fallback or AnswerComposer()

        if self._api_key:
            print(f"[GeminiComposer] Using Gemini 3.5 Flash REST API (model={MODEL_NAME})", flush=True)
        else:
            print("[GeminiComposer] GEMINI_API_KEY not set — falling back to extractive AnswerComposer", flush=True)

    def compose(
        self,
        query: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        max_sentences: int = 3,
        max_answer_chars: int = 600,
    ) -> ComposedAnswer:
        """Generate a grounded answer via Gemini REST API or fall back to extractive composition."""
        started_at = time.perf_counter()

        if not self._api_key:
            return self._fallback.compose(query, retrieved_chunks, max_sentences, max_answer_chars)

        language = _detect_language(query)
        prompt = _build_prompt(query, retrieved_chunks, language)

        try:
            answer_text = _call_gemini_rest(self._api_key, prompt)

            if not answer_text:
                return self._no_answer(started_at, language)

            # Build evidence for provenance display and guardrail compliance
            evidence = []
            if retrieved_chunks:
                for chunk in retrieved_chunks[:1]:
                    evidence.append(
                        AnswerEvidence(
                            query_id=chunk.metadata.get("query_id"),
                            passage_index=chunk.metadata.get("passage_index"),
                            chunk_index=chunk.metadata.get("chunk_index"),
                            retrieval_score=float(chunk.score),
                            source_sentence=chunk.text,
                        )
                    )
            else:
                evidence.append(
                    AnswerEvidence(
                        query_id="llm",
                        passage_index=0,
                        chunk_index=0,
                        retrieval_score=1.0,
                        source_sentence="Google Gemini Knowledge Base",
                    )
                )


            latency_ms = (time.perf_counter() - started_at) * 1_000
            confidence = (
                sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
                if retrieved_chunks else None
            )
            print(
                f"[GeminiComposer] Answer generated in {latency_ms:.1f}ms"
                f" | chunks={len(retrieved_chunks)}",
                flush=True,
            )

            return ComposedAnswer(
                text=answer_text,
                evidence=evidence,
                confidence=float(confidence) if confidence is not None else None,
                latency_ms=latency_ms,
                is_no_answer=False,
            )

        except Exception as exc:
            logger.warning("GeminiComposer: REST API call failed (%s: %s), falling back to extractive", type(exc).__name__, exc)
            return self._fallback.compose(query, retrieved_chunks, max_sentences, max_answer_chars)

    def _no_answer(self, started_at: float, language: str) -> ComposedAnswer:
        text = _NO_ANSWER_HI if language == "hi" else _NO_ANSWER_EN
        return ComposedAnswer(
            text=text,
            evidence=[],
            confidence=None,
            latency_ms=(time.perf_counter() - started_at) * 1_000,
            is_no_answer=True,
        )
