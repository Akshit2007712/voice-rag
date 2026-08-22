/**
 * Shared contract between this frontend and the RAG backend.
 *
 * The backend endpoint is expected at:
 *   POST {NEXT_PUBLIC_API_BASE_URL}/api/query
 *   Content-Type: multipart/form-data
 *   Field: "audio" — the recorded clip (webm/opus from MediaRecorder)
 *
 * Adjust `lib/api.ts` if your backend's field name, route, or response
 * shape differs — that is the single place this contract is consumed.
 */

export type GuardrailCategory =
  | "none"
  | "off_topic"
  | "unsafe"
  | "ungrounded"
  | "empty_retrieval";

export interface GuardrailInfo {
  triggered: boolean;
  category: GuardrailCategory;
  /** Human-readable reason surfaced to the user, e.g. "No passages cleared the relevance threshold." */
  reason?: string;
}

export interface Citation {
  id: string;
  /** Short excerpt of the retrieved chunk used to ground the answer. */
  text: string;
  /** Document / passage identifier from the source dataset. */
  source: string;
  /** Retrieval similarity score, 0–1. */
  score: number;
  /** Which chunking strategy produced this chunk, e.g. "semantic", "fixed-256", "sliding-128-32". */
  strategy?: string;
}

export interface LatencyBreakdown {
  stt_ms: number;
  retrieval_ms: number;
  generation_ms: number;
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
  total_ms: number;
}

export interface BenchmarkLatency {
  p50_ms?: number | null;
  p70_ms?: number | null;
  p100_ms?: number | null;
  sample_count?: number | null;
}

export interface QueryResponse {
  transcript: string;
  answer: string;
  no_answer?: boolean;
  /** Optional plain-language explanation generated alongside the answer. */
  explanation?: string;
  citations: Citation[];
  latency: LatencyBreakdown;
  guardrail: GuardrailInfo;
  benchmark_latency?: BenchmarkLatency;
  input_mode?: "voice" | "text";
}

export interface LatencySample {
  timestamp: number;
  total_ms: number;
  stt_ms: number;
  retrieval_ms: number;
  generation_ms: number;
}

export interface AggregateLatencyStats {
  p50: number;
  p70: number;
  p100: number;
  sampleSize: number;
}

export type PipelineStage =
  | "idle"
  | "recording"
  | "uploading"
  | "transcribing"
  | "retrieving"
  | "generating"
  | "done"
  | "error";

export type QueryLanguage = "en" | "hi";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
