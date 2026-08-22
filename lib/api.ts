import { ApiError, type QueryLanguage, type QueryResponse } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "https://voice-rag-backend-pdll.onrender.com";

const VOICE_ENDPOINT = `${API_BASE}/query-voice`;
const TEXT_ENDPOINT = `${API_BASE}/query-text`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;
const READY_ENDPOINT = `${API_BASE}/ready`;

const REQUEST_TIMEOUT_MS = 90_000;
const MAX_RETRIES = 4;
const RETRY_BASE_DELAY_MS = 2000;

// Warm-up state shared across all calls
let wakePromise: Promise<boolean> | null = null;

/** Ping /health until backend responds (Render cold-start can take ~50s on free tier) */
export async function ensureBackendAwake(onWaiting?: () => void): Promise<boolean> {
  if (wakePromise) return wakePromise;
  wakePromise = (async () => {
    for (let attempt = 0; attempt < 12; attempt++) {
      try {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 8000);
        const res = await fetch(HEALTH_ENDPOINT, { method: "GET", cache: "no-store", signal: controller.signal });
        clearTimeout(tid);
        if (res.ok) return true;
      } catch {
        // backend sleeping — keep probing
      }
      if (attempt === 0 && onWaiting) onWaiting();
      await new Promise(r => setTimeout(r, 5000));
    }
    return false;
  })();
  return wakePromise;
}

/** Reset so next call re-probes (call after successful query) */
export function resetWakeState() {
  wakePromise = null;
}



function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status: number) {
  // Retry on transient/server-side failures, not on client mistakes.
  return status === 408 || status === 429 || status >= 500;
}

async function parseErrorBody(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.message === "string") return data.message;
    if (typeof data?.error === "string") return data.error;
    if (typeof data?.error === "object" && data?.error !== null) {
      if (typeof data.error.message === "string") return data.error.message;
      if (typeof data.error.code === "string") return data.error.code;
    }
    if (typeof data?.detail === "string") return data.detail;
  } catch {
    // body wasn't JSON — fall through to status text
  }
  return res.statusText || `Request failed with status ${res.status}`;
}

interface CitationLike {
  id?: string | number;
  text?: string;
  parent_text?: string;
  source?: string;
  chunk_type?: string;
  score?: number;
  rerank_score?: number;
  strategy?: string;
}

interface LatencyLike {
  stt_ms?: number;
  retrieval_ms?: number;
  retrieval?: number;
  generation_ms?: number;
  generation?: number;
  guardrail_ms?: number;
  guardrails?: number;
  embedding?: number;
  embedding_ms?: number;
  dense?: number;
  qdrant_ms?: number;
  bm25?: number;
  bm25_ms?: number;
  fusion?: number;
  rrf_ms?: number;
  rerank?: number;
  grounding?: number;
  maturity_ms?: number;
  composer_ms?: number;
  rag_total_ms?: number;
  total_voice_pipeline_ms?: number;
  total_ms?: number;
  total_rag?: number;
}

function validateQueryResponse(data: unknown, fallbackTranscript?: string): QueryResponse {
  if (!data || typeof data !== "object") {
    throw new ApiError("Backend returned an unexpected response shape.");
  }
  const d = data as Partial<QueryResponse> & {
    query?: string;
    evidence?: Array<{
      query_id?: string | number;
      passage_index?: number;
      chunk_index?: number;
      retrieval_score?: number;
      text?: string;
    }>;
    voice_latency?: {
      stt_ms?: number;
      total_voice_pipeline_ms?: number;
    } | null;
    benchmark_latency?: {
      p50_ms?: number | null;
      p70_ms?: number | null;
      p100_ms?: number | null;
      sample_count?: number | null;
    };
    input_mode?: "voice" | "text";
    reasoning?: unknown;
    rationale?: unknown;
    sources?: QueryResponse["citations"];
    latency_ms?: QueryResponse["latency"];
  };

  const transcript =
    typeof d.transcript === "string" && d.transcript.trim().length > 0
      ? d.transcript
      : typeof d.query === "string" && d.query.trim().length > 0
      ? d.query
      : fallbackTranscript ?? "";

  const answer = typeof d.answer === "string" ? d.answer : "";
  const explanation = [d.explanation, d.reasoning, d.rationale].find(
    (value): value is string => typeof value === "string" && value.trim().length > 0,
  );

  if (!answer && !transcript) {
    throw new ApiError(
      "Backend response is missing required fields (transcript/answer).",
    );
  }

  // Support both backend 'evidence' and 'citations'/'sources'
  const rawSources = Array.isArray(d.citations)
    ? d.citations
    : Array.isArray(d.sources)
    ? d.sources
    : Array.isArray(d.evidence)
    ? d.evidence
    : [];

  const citations = rawSources.map((source, index) => {
    const item = source as CitationLike & {
      query_id?: string | number;
      passage_index?: number;
      chunk_index?: number;
      retrieval_score?: number;
    };
    const id = String(item.id ?? item.query_id ?? index + 1);
    const sourceLabel = item.source
      ? String(item.source)
      : item.query_id !== undefined
      ? `Query ${item.query_id} (P${item.passage_index ?? 0})`
      : item.chunk_type
      ? String(item.chunk_type)
      : `Passage ${index + 1}`;
    const text = String(item.text ?? item.parent_text ?? `Retrieved passage ${index + 1}`);
    const score = Number(item.score ?? item.retrieval_score ?? item.rerank_score ?? 0);
    return {
      id,
      text,
      source: sourceLabel,
      score,
      strategy: item.strategy ?? (item.chunk_index !== undefined ? `chunk-${item.chunk_index}` : undefined),
    };
  });

  const rawLatency = d.latency ?? d.latency_ms;
  const latencyObj = rawLatency as LatencyLike | undefined;
  const voiceLat = d.voice_latency;

  const stt_ms = Number(voiceLat?.stt_ms ?? latencyObj?.stt_ms ?? 0);
  const rag_total = Number(
    latencyObj?.rag_total_ms ??
    latencyObj?.total_rag ??
    latencyObj?.total_ms ??
    0
  );
  const total_voice = Number(voiceLat?.total_voice_pipeline_ms ?? (stt_ms + rag_total));
  const total_ms = total_voice > 0 && d.input_mode === "voice" ? total_voice : rag_total > 0 ? rag_total : (stt_ms + rag_total);

  const embedding_ms = Number(latencyObj?.embedding_ms ?? latencyObj?.embedding ?? 0);
  const qdrant_ms = Number(latencyObj?.qdrant_ms ?? latencyObj?.dense ?? 0);
  const bm25_ms = Number(latencyObj?.bm25_ms ?? latencyObj?.bm25 ?? 0);
  const rrf_ms = Number(latencyObj?.rrf_ms ?? latencyObj?.fusion ?? 0);
  const composer_ms = Number(latencyObj?.composer_ms ?? latencyObj?.generation_ms ?? latencyObj?.generation ?? 0);
  const maturity_ms = Number(latencyObj?.maturity_ms ?? 0);
  const retrieval_ms = Number(
    latencyObj?.retrieval_ms ??
    latencyObj?.retrieval ??
    (qdrant_ms || bm25_ms || rrf_ms ? Math.max(qdrant_ms, bm25_ms) + rrf_ms : 0)
  );

  return {
    transcript,
    answer,
    no_answer: Boolean(d.no_answer),
    explanation,
    citations,
    latency: {
      stt_ms,
      retrieval_ms,
      generation_ms: composer_ms,
      guardrail_ms: latencyObj?.guardrail_ms,
      guardrails: Number(latencyObj?.guardrails ?? latencyObj?.guardrail_ms ?? 0),
      embedding: embedding_ms,
      embedding_ms,
      dense: qdrant_ms,
      qdrant_ms,
      bm25: bm25_ms,
      bm25_ms,
      fusion: rrf_ms,
      rrf_ms,
      rerank: Number(latencyObj?.rerank ?? 0),
      grounding: Number(latencyObj?.grounding ?? 0),
      maturity_ms,
      composer_ms,
      rag_total_ms: rag_total,
      total_voice_pipeline_ms: total_voice,
      total_ms: rag_total > 0 ? rag_total : total_ms,
    },
    guardrail: d.guardrail ?? {
      triggered: Boolean(d.no_answer),
      category: d.no_answer ? "empty_retrieval" : "none",
      reason: d.no_answer ? "No matching passages cleared the relevance threshold." : undefined,
    },
    benchmark_latency: d.benchmark_latency,
    input_mode: d.input_mode ?? (voiceLat ? "voice" : "text"),
  };
}

async function postQuery(
  url: string,
  body: BodyInit,
  options?: { signal?: AbortSignal; headers?: HeadersInit },
): Promise<unknown> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const onExternalAbort = () => controller.abort();
  options?.signal?.addEventListener("abort", onExternalAbort);

  try {
    const res = await fetch(url, {
      method: "POST",
      body,
      headers: options?.headers,
      signal: controller.signal,
    });
    if (!res.ok) throw new ApiError(await parseErrorBody(res), res.status);
    return await res.json();
  } catch (err) {
    if (options?.signal?.aborted) throw new ApiError("Cancelled.");
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("Request timed out.");
    }
    if (err instanceof ApiError) throw err;
    throw new ApiError("Could not reach the backend. Check your API URL and that the server is running.");
  } finally {
    clearTimeout(timeoutId);
    options?.signal?.removeEventListener("abort", onExternalAbort);
  }
}

export async function submitTextQuery(
  query: string,
  language: QueryLanguage,
  options?: { signal?: AbortSignal },
): Promise<QueryResponse> {
  const trimmed = query.trim();
  const rawData = await postQuery(
    TEXT_ENDPOINT,
    JSON.stringify({ query: trimmed, language }),
    { ...options, headers: { "Content-Type": "application/json" } },
  );
  return validateQueryResponse(rawData, trimmed);
}

/**
 * Uploads a recorded audio clip to the RAG backend (/query-voice) and returns the
 * transcript, grounded answer, citations, and latency breakdown.
 *
 * Retries transient failures (timeouts, 429/5xx, network drops) with
 * exponential backoff. Client-side (4xx) errors fail fast.
 */
export async function submitVoiceQuery(
  audioBlob: Blob,
  options?: { signal?: AbortSignal; fileName?: string; language?: QueryLanguage },
): Promise<QueryResponse> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    const onExternalAbort = () => controller.abort();
    options?.signal?.addEventListener("abort", onExternalAbort);

    try {
      const form = new FormData();
      form.append(
        "audio",
        audioBlob,
        options?.fileName ?? `query-${Date.now()}.webm`,
      );
      if (options?.language) form.append("language", options.language);

      const res = await fetch(VOICE_ENDPOINT, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      if (!res.ok) {
        const message = await parseErrorBody(res);
        if (isRetryableStatus(res.status) && attempt < MAX_RETRIES) {
          lastError = new ApiError(message, res.status);
          await sleep(RETRY_BASE_DELAY_MS * 2 ** attempt);
          continue;
        }
        throw new ApiError(message, res.status);
      }

      const data = await res.json();
      return validateQueryResponse(data);
    } catch (err) {
      if (options?.signal?.aborted) {
        throw new ApiError("Cancelled.");
      }
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      const isNetwork = err instanceof TypeError;
      if ((isAbort || isNetwork) && attempt < MAX_RETRIES) {
        lastError = new ApiError(
          isAbort ? "Request timed out." : "Network error reaching the backend.",
        );
        await sleep(RETRY_BASE_DELAY_MS * 2 ** attempt);
        continue;
      }
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        isAbort
          ? "Request timed out."
          : isNetwork
            ? "Could not reach the backend. Check NEXT_PUBLIC_API_BASE_URL and that the server is running."
            : "Unexpected error while processing your question.",
      );
    } finally {
      clearTimeout(timeoutId);
      options?.signal?.removeEventListener("abort", onExternalAbort);
    }
  }

  throw lastError instanceof ApiError
    ? lastError
    : new ApiError("Request failed after retries.");
}

export async function checkBackendHealth(): Promise<{ status: string }> {
  const res = await fetch(HEALTH_ENDPOINT);
  if (!res.ok) throw new ApiError("Backend health check failed", res.status);
  return res.json();
}

export async function checkBackendReady(): Promise<{ ready: boolean }> {
  const res = await fetch(READY_ENDPOINT);
  if (!res.ok) throw new ApiError("Backend not ready yet", res.status);
  return res.json();
}

