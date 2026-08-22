import type { AggregateLatencyStats, LatencyBreakdown } from "@/lib/types";
import { PixelPanel } from "./PixelPanel";

interface LatencyPanelProps {
  latest: LatencyBreakdown | null;
  aggregate: AggregateLatencyStats;
  targetMs?: number;
}

const STAGES: Array<{ key: keyof LatencyBreakdown; label: string; fallbackKey?: keyof LatencyBreakdown }> = [
  { key: "embedding_ms", label: "embedding", fallbackKey: "embedding" },
  { key: "qdrant_ms", label: "dense (qdrant)", fallbackKey: "dense" },
  { key: "bm25_ms", label: "bm25 (lexical)", fallbackKey: "bm25" },
  { key: "rrf_ms", label: "rrf fusion", fallbackKey: "fusion" },
  { key: "maturity_ms", label: "maturity" },
  { key: "composer_ms", label: "composer", fallbackKey: "generation_ms" },
  { key: "guardrails", label: "guardrails", fallbackKey: "guardrail_ms" },
];

function formatMs(value: number | undefined) {
  return Number(value ?? 0).toFixed(1);
}

export function LatencyPanel({ latest, aggregate, targetMs = 200 }: LatencyPanelProps) {
  const underTarget = latest ? latest.total_ms <= targetMs : true;

  return (
    <PixelPanel
      label="RAG latency"
      tone="cream"
      right={
        latest && (
          <span
            className={`border-2 border-ink px-2 py-1 font-pixel text-[8px] shadow-pixel-sm sm:text-[9px] ${
              underTarget ? "bg-pipe text-cream" : "bg-alert text-cream"
            }`}
          >
            {underTarget ? "UNDER 200MS" : "OVER TARGET"}
          </span>
        )
      }
    >
      <p className="mb-5 font-body text-xs text-ink/60 sm:text-sm">
        Warm retrieval path · STT tracked separately
      </p>
      {latest ? (
        <div>
          <div className="border-2 border-ink bg-cream px-4 py-4 shadow-pixel-sm">
            <p className="font-pixel text-[11px] uppercase tracking-wide text-ink sm:text-xs">Voice</p>
            <dl className="mt-4 space-y-3 font-body text-sm sm:text-base">
              <div className="flex items-center justify-between">
                <dt>STT</dt>
                <dd className="font-mono">{formatMs(latest.stt_ms)} ms</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt>RAG</dt>
                <dd className="font-mono">{formatMs(latest.total_ms)} ms</dd>
              </div>
              <div className="flex items-center justify-between border-t border-ink/20 pt-3">
                <dt>Total voice pipeline</dt>
                <dd className="font-mono">{formatMs(latest.stt_ms + latest.total_ms)} ms</dd>
              </div>
            </dl>
          </div>

          <div className="border-2 border-ink bg-cream px-4 py-4 shadow-pixel-sm">
            <div>
              <div className="flex items-center justify-between gap-3">
                <p className="font-pixel text-[11px] uppercase tracking-wide text-ink/60 sm:text-xs">Current request</p>
                <span className="font-pixel text-[8px] text-ink/50">TARGET {targetMs}MS</span>
              </div>
              <p className="mt-2 font-pixel text-[11px] font-bold uppercase tracking-wide text-ink sm:text-xs">RAG latency</p>
              <p className="mt-1 font-mono text-3xl font-normal leading-none text-ink sm:text-4xl">
                {formatMs(latest.total_ms)}<span className="ml-1 text-base font-normal text-ink/60">ms</span>
              </p>
            </div>
          </div>

          <div className="mt-5 border-2 border-ink bg-cream px-4 py-4 shadow-pixel-sm">
            <p className="font-pixel text-[11px] uppercase tracking-wide text-ink sm:text-xs">Benchmark</p>
            <dl className="mt-4 space-y-2 font-mono text-sm sm:text-base">
              {([
                ["P50", aggregate.p50],
                ["P70", aggregate.p70],
                ["P100", aggregate.p100],
              ] as const).map(([label, value]) => (
                <div key={label} className="flex items-center gap-5">
                  <dt className="w-11">{label}</dt>
                  <dd>{value} ms</dd>
                </div>
              ))}
            </dl>
            <div className="mt-6 space-y-2 font-mono text-sm sm:text-base">
              <p>Samples: {aggregate.sampleSize}</p>
              <p>Environment: deployed backend</p>
            </div>
          </div>

          <div className="mt-5">
            <div className="flex items-center justify-between border-b-2 border-ink/30 pb-2 font-pixel text-[8px] uppercase tracking-wide text-ink/60 sm:text-[9px]">
              <span>Stage</span>
              <span>MS</span>
            </div>
            <dl className="font-body text-sm sm:text-base">
              {STAGES.map((stage) => {
                const val = (latest[stage.key] as number | undefined) ?? (stage.fallbackKey ? (latest[stage.fallbackKey] as number | undefined) : undefined);
                return (
                  <div key={stage.key} className="flex items-center justify-between border-b border-ink/15 py-2.5">
                    <dt>{stage.label}</dt>
                    <dd className="font-mono text-ink/80">{formatMs(val)}</dd>
                  </div>
                );
              })}
              <div className="flex items-center justify-between border-b-2 border-pipe/50 py-2.5 font-normal text-pipe-deep">
                <dt>RAG TOTAL</dt>
                <dd className="font-mono">{formatMs(latest.total_ms)}</dd>
              </div>
            </dl>
          </div>
        </div>
      ) : (
        <div>
          <div className="border-2 border-ink bg-cream px-4 py-4 shadow-pixel-sm">
            <p className="font-pixel text-[11px] uppercase tracking-wide text-ink sm:text-xs">Voice</p>
            <dl className="mt-4 space-y-3 font-body text-sm sm:text-base">
              <div className="flex items-center justify-between"><dt>STT</dt><dd className="font-mono text-ink/40">---</dd></div>
              <div className="flex items-center justify-between"><dt>RAG</dt><dd className="font-mono text-ink/40">---</dd></div>
              <div className="flex items-center justify-between border-t border-ink/20 pt-3"><dt>Total voice pipeline</dt><dd className="font-mono text-ink/40">---</dd></div>
            </dl>
          </div>

          <div className="mt-5 border-2 border-ink bg-cream px-4 py-4 shadow-pixel-sm">
            <p className="font-pixel text-[11px] uppercase tracking-wide text-ink sm:text-xs">Benchmark</p>
            <dl className="mt-4 space-y-2 font-mono text-sm sm:text-base">
              {(["P50", "P70", "P100"] as const).map((label) => (
                <div key={label} className="flex items-center gap-5"><dt className="w-11">{label}</dt><dd className="text-ink/40">---</dd></div>
              ))}
            </dl>
            <div className="mt-6 space-y-2 font-mono text-sm sm:text-base text-ink/40">
              <p>Samples: 0</p>
              <p>Environment: deployed backend</p>
            </div>
          </div>

          <p className="mt-5 font-body text-sm italic text-ink/40">
            RAG latency will appear here after your first question.
          </p>
        </div>
      )}
    </PixelPanel>
  );
}
