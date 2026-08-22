import type {
  AggregateLatencyStats,
  LatencyBreakdown,
  LatencySample,
} from "./types";

const STORAGE_KEY = "voice-rag:latency-history";
const MAX_SAMPLES = 200;

function readHistory(): LatencySample[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeHistory(samples: LatencySample[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  } catch {
    // storage full or unavailable — degrade silently, stats just reset
  }
}

/** Records a completed query's latency into the rolling session history. */
export function recordLatencySample(latency: LatencyBreakdown): LatencySample[] {
  const history = readHistory();
  history.push({
    timestamp: Date.now(),
    total_ms: latency.total_ms,
    stt_ms: latency.stt_ms,
    retrieval_ms: latency.retrieval_ms,
    generation_ms: latency.generation_ms,
  });
  const trimmed = history.slice(-MAX_SAMPLES);
  writeHistory(trimmed);
  return trimmed;
}

export function getLatencyHistory(): LatencySample[] {
  return readHistory();
}

export function clearLatencyHistory(): void {
  writeHistory([]);
}

function percentile(sortedValues: number[], p: number): number {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0]!;
  const rank = (p / 100) * (sortedValues.length - 1);
  const lowerIndex = Math.floor(rank);
  const upperIndex = Math.ceil(rank);
  const weight = rank - lowerIndex;
  const lower = sortedValues[lowerIndex]!;
  const upper = sortedValues[upperIndex]!;
  return lower + (upper - lower) * weight;
}

/**
 * P50 / P70 / P100 across the session's recorded queries.
 * P100 is simply the max — the worst observed run, not a smoothed tail.
 */
export function computeAggregateStats(
  samples: LatencySample[],
): AggregateLatencyStats {
  const totals = samples.map((s) => s.total_ms).sort((a, b) => a - b);
  return {
    p50: Math.round(percentile(totals, 50)),
    p70: Math.round(percentile(totals, 70)),
    p100: Math.round(percentile(totals, 100)),
    sampleSize: totals.length,
  };
}
