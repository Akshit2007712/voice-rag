import type { PipelineStage } from "@/lib/types";
import { PixelMicIcon, PixelStopIcon } from "./PixelIcons";

interface MicOrbProps {
  phase: PipelineStage;
  level: number;
  elapsedMs: number;
  onPress: () => void;
  disabled?: boolean;
}

const STAGE_LABEL: Record<PipelineStage, string> = {
  idle: "TAP TO ASK",
  recording: "TAP TO STOP",
  uploading: "SENDING AUDIO",
  transcribing: "TRANSCRIBING",
  retrieving: "RETRIEVING",
  generating: "GENERATING",
  done: "TAP TO ASK AGAIN",
  error: "TAP TO RETRY",
};

const isBusy = (phase: PipelineStage) =>
  phase === "uploading" ||
  phase === "transcribing" ||
  phase === "retrieving" ||
  phase === "generating";

function formatElapsed(ms: number) {
  const seconds = Math.floor(ms / 1000);
  const centis = Math.floor((ms % 1000) / 10);
  return `0:${seconds.toString().padStart(2, "0")}.${centis
    .toString()
    .padStart(2, "0")}`;
}

export function MicOrb({ phase, level, elapsedMs, onPress, disabled }: MicOrbProps) {
  const recording = phase === "recording";
  const busy = isBusy(phase);

  const orbTone = recording
    ? "bg-alert text-cream"
    : busy
      ? "bg-sky-deep text-cream"
      : "bg-coin text-ink";

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative grid place-items-center">
        {recording && (
          <>
            <span className="pointer-events-none absolute h-28 w-28 rounded-full border-4 border-alert/70 animate-pulse-ring sm:h-32 sm:w-32" />
            <span
              className="pointer-events-none absolute h-28 w-28 rounded-full border-4 border-alert/50 animate-pulse-ring sm:h-32 sm:w-32"
              style={{ animationDelay: "0.5s" }}
            />
          </>
        )}

        <button
          type="button"
          onClick={onPress}
          disabled={disabled || busy}
          aria-label={STAGE_LABEL[phase]}
          aria-pressed={recording}
          className={`relative z-10 grid h-28 w-28 place-items-center border-4 border-ink shadow-pixel-lg transition-transform duration-100 ease-out active:translate-x-1 active:translate-y-1 active:shadow-pixel-sm disabled:cursor-not-allowed disabled:opacity-90 sm:h-32 sm:w-32 ${orbTone} ${
            !recording && !busy ? "animate-bob" : ""
          }`}
        >
          {busy ? (
            <span
              aria-hidden
              className="h-10 w-10 origin-center animate-coin-flip border-4 border-cream sm:h-12 sm:w-12"
            />
          ) : recording ? (
            <div className="flex h-11 items-end gap-1.5" aria-hidden>
              {[0, 1, 2, 3].map((i) => (
                <span
                  key={i}
                  className="w-2 bg-cream sm:w-2.5"
                  style={{
                    height: `${Math.max(18, level * 100)}%`,
                    minHeight: 8,
                    maxHeight: 44,
                    transition: "height 80ms linear",
                  }}
                />
              ))}
              <PixelStopIcon className="ml-1 h-6 w-6 text-cream sm:h-7 sm:w-7" />
            </div>
          ) : (
            <PixelMicIcon className="h-12 w-12 sm:h-14 sm:w-14" />
          )}
        </button>
      </div>

      {recording && (
        <div aria-hidden className="duck-walk-track">
          <div className="duck">
            <span className="duck-body" />
            <span className="duck-head" />
            <span className="duck-beak" />
            <span className="duck-leg duck-leg-back" />
            <span className="duck-leg duck-leg-front" />
            <span className="duck-tail" />
          </div>
        </div>
      )}

      <div className="text-center">
        <p className="font-pixel text-[10px] tracking-wide text-cream sm:text-xs">
          {STAGE_LABEL[phase]}
          {phase === "idle" && (
            <span className="animate-blink" aria-hidden>
              _
            </span>
          )}
        </p>
        {recording && (
          <p className="mt-1 font-mono text-xs text-cream/80 sm:text-sm">
            {formatElapsed(elapsedMs)}
          </p>
        )}
      </div>
    </div>
  );
}
