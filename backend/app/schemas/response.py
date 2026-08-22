from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceQueryResponse(BaseModel):
    transcript: str
    normalized_query: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: int = Field(ge=0)


class EvidenceResponse(BaseModel):
    """Lightweight answer provenance suitable for API clients and debugging."""

    query_id: Any = Field(default=None, description="MSMARCO-XI source query identifier.")
    passage_index: int | None = Field(default=None, description="Index of the supporting passage within its source row.")
    chunk_index: int | None = Field(default=None, description="Index of the supporting chunk within that passage.")
    retrieval_score: float = Field(description="Semantic retrieval confidence associated with the evidence.")


class RagLatencyResponse(BaseModel):
    """Measured one-shot RAG stages. Branch timings overlap by design."""

    embedding_ms: float = Field(ge=0)
    qdrant_ms: float = Field(ge=0)
    bm25_ms: float = Field(ge=0)
    post_embedding_parallel_ms: float = Field(ge=0)
    rrf_ms: float = Field(ge=0)
    maturity_ms: float = Field(ge=0)
    composer_ms: float = Field(ge=0)
    rag_total_ms: float = Field(ge=0)


class VoicePipelineLatencyResponse(BaseModel):
    """Voice-only wall timings; STT is intentionally outside RAG timing."""

    stt_ms: float = Field(ge=0)
    total_voice_pipeline_ms: float = Field(ge=0)


class BenchmarkLatencyResponse(BaseModel):
    """Approved user-facing benchmark percentiles, never internal benchmark configuration."""

    p50_ms: float | None = Field(default=None, ge=0)
    p70_ms: float | None = Field(default=None, ge=0)
    p100_ms: float | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, ge=1)


class VoiceRagResponse(BaseModel):
    """Stable user-facing result shared by typed and one-shot voice requests."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "answer": "फिलाडेल्फिया और लैंकेस्टर के बीच की दूरी उपलब्ध संदर्भ पर निर्भर करती है।",
        "no_answer": False,
        "evidence": [{"query_id": 232017, "passage_index": 8, "chunk_index": 0, "retrieval_score": 0.84}],
        "latency": {
            "embedding_ms": 45.0, "qdrant_ms": 80.0, "bm25_ms": 20.0,
            "post_embedding_parallel_ms": 82.0, "rrf_ms": 0.2,
            "maturity_ms": 0.0, "composer_ms": 1.0, "rag_total_ms": 130.0,
        },
        "voice_latency": None,
        "benchmark_latency": {"p50_ms": None, "p70_ms": None, "p100_ms": None, "sample_count": None},
        "input_mode": "text",
    }]})

    transcript: str | None = Field(default=None, description="Transcribed question for voice queries or query text.")
    answer: str = Field(description="Grounded deterministic answer, or the approved no-answer text.")
    no_answer: bool = Field(description="True when available evidence is insufficient for a grounded answer.")
    evidence: list[EvidenceResponse] = Field(description="Supporting source provenance intentionally exposed to clients.")
    latency: RagLatencyResponse = Field(description="One-shot RAG timing. It excludes STT and overlapping branch times are not additive.")
    voice_latency: VoicePipelineLatencyResponse | None = Field(description="Present only for /query-voice; STT timing remains outside latency.rag_total_ms.")
    benchmark_latency: BenchmarkLatencyResponse = Field(description="Optional validated benchmark percentiles, not live request percentiles.")
    input_mode: Literal["voice", "text"] = Field(description="How usable query text entered the shared RAG path.")


class TextQueryRequest(BaseModel):
    """Typed-query request for the same bilingual one-shot RAG path."""

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"query": "फिलाडेल्फिया लैंकेस्टर से कितनी दूर है?", "language": "hi"},
        {"query": "How far is Philadelphia from Lancaster?", "language": "en"},
    ]})

    query: str = Field(
        description="Question text. It is normalized by the shared harness and must be at most 4,000 characters.",
    )
    language: Literal["hi", "en"] = Field(
        default="hi",
        description="Application language: hi for Hindi or en for English.",
    )
