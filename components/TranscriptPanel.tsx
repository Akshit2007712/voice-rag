import { PixelPanel } from "./PixelPanel";

interface TranscriptPanelProps {
  transcript: string | null;
  loading: boolean;
}

export function TranscriptPanel({ transcript, loading }: TranscriptPanelProps) {
  return (
    <PixelPanel label="Live" tone="cream">
      <p className="mb-3 font-pixel text-[9px] uppercase tracking-wide text-sky-deep sm:text-[10px]">
        Transcript
      </p>
      {loading ? (
        <div className="space-y-2" aria-live="polite" aria-busy="true">
          <div className="h-3.5 w-11/12 animate-pulse bg-ink/10" />
          <div className="h-3.5 w-8/12 animate-pulse bg-ink/10" />
        </div>
      ) : transcript ? (
        <p className="font-body text-base leading-relaxed text-ink sm:text-lg">
          &ldquo;{transcript}&rdquo;
        </p>
      ) : (
        <p className="font-body text-sm italic text-ink/40">
          Your question will show up here once you speak into the mic.
        </p>
      )}
    </PixelPanel>
  );
}
