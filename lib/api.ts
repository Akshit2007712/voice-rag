import { ApiError, type QueryLanguage, type QueryResponse } from "./types";

const DIRECT_RENDER_BACKEND = "https://voice-rag-backend-pdll.onrender.com";

function getTargetUrls(endpoint: string): string[] {
  const clean = endpoint.replace(/^\/api\/backend\/?/, "").replace(/^https?:\/\/[^\/]+\/?/, "").replace(/^\/+/, "");
  return [
    `/api/backend/${clean}`,
    `${DIRECT_RENDER_BACKEND}/${clean}`,
  ];
}

const REQUEST_TIMEOUT_MS = 90_000;
const MAX_RETRIES = 6;
const RETRY_BASE_DELAY_MS = 1500;

/** Proactively wake up Render free-tier container in background */
export async function ensureBackendAwake(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const tid = setTimeout(() => controller.abort(), 10000);
    const res = await fetch("/api/backend/health", { method: "GET", cache: "no-store", signal: controller.signal });
    clearTimeout(tid);
    if (res.ok) return true;
    // Fallback direct check
    const res2 = await fetch(`${DIRECT_RENDER_BACKEND}/health`, { cache: "no-store" });
    return res2.ok;
  } catch {
    return false;
  }
}

export function resetWakeState() {}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status: number) {
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
    // body wasn't JSON
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
    throw new ApiError("Backend response is missing required fields (transcript/answer).");
  }

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
    (latencyObj?.retrieval_ms ?? 0) + (latencyObj?.generation_ms ?? 0)
  );

  const total_ms = Number(
    latencyObj?.total_ms ??
    voiceLat?.total_voice_pipeline_ms ??
    (stt_ms + rag_total)
  );

  return {
    transcript,
    answer,
    no_answer: Boolean(d.no_answer),
    explanation,
    citations,
    latency: {
      stt_ms,
      retrieval_ms: Number(latencyObj?.retrieval_ms ?? latencyObj?.retrieval ?? 0),
      generation_ms: Number(latencyObj?.generation_ms ?? latencyObj?.generation ?? 0),
      guardrail_ms: Number(latencyObj?.guardrail_ms ?? latencyObj?.guardrails ?? 0),
      embedding_ms: Number(latencyObj?.embedding_ms ?? latencyObj?.embedding ?? 0),
      qdrant_ms: Number(latencyObj?.qdrant_ms ?? latencyObj?.dense ?? 0),
      bm25_ms: Number(latencyObj?.bm25_ms ?? latencyObj?.bm25 ?? 0),
      rrf_ms: Number(latencyObj?.rrf_ms ?? latencyObj?.fusion ?? 0),
      rag_total_ms: rag_total,
      total_voice_pipeline_ms: voiceLat?.total_voice_pipeline_ms ?? total_ms,
      total_ms: total_ms > 0 ? total_ms : 1,
    },
    guardrail: {
      triggered: Boolean(d.guardrail?.triggered),
      category: d.guardrail?.category ?? "none",
      reason: d.guardrail?.reason,
    },
    benchmark_latency: d.benchmark_latency,
    input_mode: d.input_mode ?? (voiceLat ? "voice" : "text"),
  };
}

async function postQuery(
  endpoint: string,
  body: BodyInit,
  options?: { signal?: AbortSignal; headers?: HeadersInit },
): Promise<unknown> {
  let lastError: unknown;
  const targetUrls = getTargetUrls(endpoint);

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const targetUrl = targetUrls[attempt % targetUrls.length]!;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const onExternalAbort = () => controller.abort();
    options?.signal?.addEventListener("abort", onExternalAbort);

    try {
      const res = await fetch(targetUrl, {
        method: "POST",
        body,
        headers: options?.headers,
        signal: controller.signal,
      });

      if (!res.ok) {
        const message = await parseErrorBody(res);
        if (isRetryableStatus(res.status) && attempt < MAX_RETRIES) {
          lastError = new ApiError(message, res.status);
          await sleep(RETRY_BASE_DELAY_MS * 2 ** (attempt % 3));
          continue;
        }
        throw new ApiError(message, res.status);
      }

      return await res.json();
    } catch (err) {
      if (options?.signal?.aborted) throw new ApiError("Cancelled.");
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      const isNetwork = err instanceof TypeError;

      if ((isAbort || isNetwork) && attempt < MAX_RETRIES) {
        lastError = err;
        await sleep(RETRY_BASE_DELAY_MS * 2 ** (attempt % 3));
        continue;
      }
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        isAbort
          ? "Request timed out — backend server took too long to respond."
          : "Backend is warming up (Render free tier). Please retry in 5 seconds."
      );
    } finally {
      clearTimeout(timeoutId);
      options?.signal?.removeEventListener("abort", onExternalAbort);
    }
  }

  throw lastError instanceof ApiError
    ? lastError
    : new ApiError("Could not reach backend after retries. Server may be spinning up.");
}

export async function submitTextQuery(
  query: string,
  language: QueryLanguage,
  options?: { signal?: AbortSignal },
): Promise<QueryResponse> {
  const trimmed = query.trim();
  const rawData = await postQuery(
    "query-text",
    JSON.stringify({ query: trimmed, language }),
    { ...options, headers: { "Content-Type": "application/json" } },
  );
  return validateQueryResponse(rawData, trimmed);
}

export async function submitVoiceQuery(
  audioBlob: Blob,
  options?: { signal?: AbortSignal; fileName?: string; language?: QueryLanguage },
): Promise<QueryResponse> {
  const form = new FormData();
  form.append(
    "audio",
    audioBlob,
    options?.fileName ?? `query-${Date.now()}.webm`,
  );
  if (options?.language) form.append("language", options.language);

  const rawData = await postQuery("query-voice", form, options);
  return validateQueryResponse(rawData);
}

export async function checkBackendHealth(): Promise<{ status: string }> {
  const res = await fetch(`${DIRECT_RENDER_BACKEND}/health`);
  if (!res.ok) throw new ApiError("Backend health check failed", res.status);
  return res.json();
}
