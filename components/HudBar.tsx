interface HudBarProps {
  lastTotalMs: number | null;
  queryCount: number;
}

export function HudBar({ lastTotalMs, queryCount }: HudBarProps) {
  return (
    <header className="border-b-4 border-ink bg-sky-night/90 text-cream">
      <div className="mx-auto flex max-w-3xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span
            aria-hidden
            className="grid h-7 w-7 place-items-center border-2 border-ink bg-coin font-pixel text-[10px] text-ink shadow-pixel-sm"
          >
            ?
          </span>
          <div className="leading-tight">
            <p className="font-pixel text-[11px] tracking-wide sm:text-sm">
              VOICE-RAG
            </p>
            <p className="font-mono text-[10px] text-cream/60 sm:text-xs">
              ask · retrieve · answer
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 sm:gap-4">
          <div className="text-right leading-tight">
            <p className="font-pixel text-[8px] text-cream/60 sm:text-[9px]">
              QUERIES
            </p>
            <p className="font-mono text-sm text-coin sm:text-base">
              {String(queryCount).padStart(3, "0")}
            </p>
          </div>
          <div className="h-8 w-px bg-cream/20" aria-hidden />
          <div className="text-right leading-tight">
            <p className="font-pixel text-[8px] text-cream/60 sm:text-[9px]">
              LAST MS
            </p>
            <p
              className={`font-mono text-sm sm:text-base ${
                lastTotalMs === null
                  ? "text-cream/40"
                  : lastTotalMs <= 200
                    ? "text-pipe"
                    : "text-alert"
              }`}
            >
              {lastTotalMs === null ? "---" : lastTotalMs}
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}
