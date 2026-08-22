# Bilingual Voice RAG API

This API answers Hindi (`hi`) and English (`en`) queries using the frozen grounded path:

`E5 → (Qdrant || language-specific BM25) → RRF → deterministic AnswerComposer`

Voice requests add Sarvam STT before that path. Typed requests do not call STT.

## Setup

From `backend/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the required Qdrant and Sarvam values in `.env`. `DIAGNOSTICS_API_KEY` is required only for the protected operator endpoint. `FRONTEND_ORIGINS` is a comma-separated list of browser origins; it never accepts `*`.

## Local indexing data

The production API reads Qdrant Cloud and does not require a local corpus or
embedded Qdrant store. Offline ingestion, indexing, and corpus-analysis tools
expect the MSMARCO-XI Hindi validation Parquet file at
`data/raw/validation/hinval.parquet`. That approximately 440 MB local dataset,
Hugging Face caches, and embedded Qdrant databases are intentionally ignored by
Git; acquire them separately before running those offline tools.

## Start

Local development:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Production:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

## Health and readiness

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
```

`/health` is process liveness only. `/ready` becomes ready only after remote Qdrant verification, E5 warm-up, and both BM25 stores are initialized.

## Frontend contract

### Typed ASK button

Use `POST /query-text` with JSON:

```json
{"query":"How far is Philadelphia from Lancaster?","language":"en"}
```

### One-shot microphone flow

Use `POST /query-voice` as `multipart/form-data`:

- `audio` — required `.webm` browser microphone file; accepted base MIME types are `audio/webm` and `video/webm`, with optional codec parameters; strict size limit: under 10 MB.
- `language` — optional `hi` (default) or `en`.

Example:

```powershell
curl.exe -X POST http://localhost:8000/query-voice -F "audio=@recording.webm;type=audio/webm" -F "language=hi"
```

### Realtime microphone streaming

Use `ws://localhost:8000/query-voice-stream?language=hi` only when live partial transcripts are needed. Send binary browser WebM chunks, then `{"type":"end"}`. The server emits `partial`, `final`, or safe `error` JSON events. The realtime endpoint retains streaming-specific maturity/trusted-candidate handling.

HTTP CORS is controlled by `FRONTEND_ORIGINS`; WebSocket connections are not governed by browser CORS preflight headers.

## One-shot response and errors

Both one-shot endpoints return `answer`, `no_answer`, intentional evidence provenance, stage latency, optional voice latency, optional validated benchmark metadata, and `input_mode`.

```json
{
  "answer": "...",
  "no_answer": false,
  "evidence": [{"query_id": 232017, "passage_index": 8, "chunk_index": 0, "retrieval_score": 0.84}],
  "latency": {"embedding_ms": 45.0, "qdrant_ms": 80.0, "bm25_ms": 20.0, "post_embedding_parallel_ms": 82.0, "rrf_ms": 0.2, "maturity_ms": 0.0, "composer_ms": 1.0, "rag_total_ms": 130.0},
  "voice_latency": null,
  "benchmark_latency": {"p50_ms": null, "p70_ms": null, "p100_ms": null, "sample_count": null},
  "input_mode": "text"
}
```

`latency.rag_total_ms` excludes STT. Voice requests additionally return `voice_latency.stt_ms` and `voice_latency.total_voice_pipeline_ms`.

Safe one-shot errors always use:

```json
{"error":{"code":"INVALID_REQUEST","message":"The request is invalid."},"request_id":"opaque-id"}
```

Common statuses: `413` upload too large, `415` unsupported audio, `422` invalid input, `502/503` dependency failure, `504` timeout, and `500` unexpected internal failure. No provider response, stack trace, secret, or retrieval infrastructure detail is returned.

## Operator diagnostics

`GET /diagnostics` is protected and is not a frontend feature:

```powershell
Invoke-RestMethod http://localhost:8000/diagnostics -Headers @{"X-Diagnostics-Key"="your-operator-key"}
```

It exposes safe operational state only. Do not expose this key to browser clients.

## Integration status

Browser microphone end-to-end validation and a deployed latency benchmark are separate integration/deployment steps. They are not claimed complete by this repository guide.
