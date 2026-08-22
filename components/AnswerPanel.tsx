import type { GuardrailInfo } from "@/lib/types";
import { PixelPanel } from "./PixelPanel";
import { GuardrailBanner } from "./GuardrailBanner";

interface AnswerPanelProps {
  answer: string | null;
  explanation: string | null;
  guardrail: GuardrailInfo | null;
  loading: boolean;
}

export function AnswerPanel({ answer, explanation, guardrail, loading }: AnswerPanelProps) {
  const blocked = guardrail?.triggered && guardrail.category !== "none";

  return (
    <PixelPanel
      label={blocked ? "No answer given" : "Answer"}
      tone={blocked ? "alert" : "pipe"}
      className="answer-panel"
    >
      {loading ? (
        <div className="space-y-2" aria-live="polite" aria-busy="true">
          <div className="h-5 w-full animate-pulse bg-ink/10" />
          <div className="h-5 w-10/12 animate-pulse bg-ink/10" />
          <div className="h-5 w-9/12 animate-pulse bg-ink/10" />
        </div>
      ) : blocked && guardrail ? (
        <GuardrailBanner
          category={guardrail.category as Exclude<GuardrailInfo["category"], "none">}
          reason={guardrail.reason}
        />
      ) : answer ? (
        <div>
          <p className="font-body text-lg leading-relaxed text-ink sm:text-2xl sm:leading-relaxed">
            {answer}
          </p>
          {explanation && (
            <div className="mt-5 border-t-2 border-ink/10 pt-4">
              <p className="font-pixel text-[9px] uppercase tracking-wide text-pipe sm:text-[10px]">
                Model explanation
              </p>
              <p className="mt-2 font-body text-sm leading-relaxed text-ink/70 sm:text-base">
                {explanation}
              </p>
            </div>
          )}
        </div>
      ) : (
        <p className="font-body text-sm italic text-ink/40">
          The grounded answer will appear here, backed by the citations below.
        </p>
      )}
    </PixelPanel>
  );
}
