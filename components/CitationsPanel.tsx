import type { Citation } from "@/lib/types";
import { PixelCoinIcon } from "./PixelIcons";
import { PixelPanel } from "./PixelPanel";

interface CitationsPanelProps {
  citations: Citation[];
  loading: boolean;
}

export function CitationsPanel({ citations, loading }: CitationsPanelProps) {
  return (
    <PixelPanel
      label={`Citations${citations.length ? ` \u00d7${citations.length}` : ""}`}
      tone="cream"
    >
      {loading ? (
        <div className="space-y-2" aria-live="polite" aria-busy="true">
          <div className="h-10 w-full animate-pulse bg-ink/10" />
          <div className="h-10 w-full animate-pulse bg-ink/10" />
        </div>
      ) : citations.length === 0 ? (
        <p className="font-body text-sm italic text-ink/40">
          Retrieved passages that ground the answer will be listed here, most
          relevant first.
        </p>
      ) : (
        <ol className="space-y-2.5">
          {citations.map((c, index) => (
            <li
              key={c.id}
              className="flex items-start gap-3 border-2 border-ink/70 bg-cream px-3 py-2.5 shadow-pixel-sm"
            >
              <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center border-2 border-ink bg-sky font-pixel text-[8px] text-cream">
                {index + 1}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="truncate font-mono text-xs font-semibold text-ink/80">
                    {c.source}
                  </span>
                  {c.strategy && (
                    <span className="border border-ink/40 bg-sky/10 px-1.5 py-0.5 font-pixel text-[7px] uppercase tracking-wide text-sky-deep">
                      {c.strategy}
                    </span>
                  )}
                  <span className="ml-auto flex items-center gap-1 font-mono text-xs text-coin-deep">
                    <PixelCoinIcon className="h-3.5 w-3.5" />
                    {Math.round(c.score * 100)}%
                  </span>
                </div>
                <p className="mt-1 line-clamp-2 font-body text-sm text-ink/70">
                  {c.text}
                </p>
              </div>
            </li>
          ))}
        </ol>
      )}
    </PixelPanel>
  );
}
