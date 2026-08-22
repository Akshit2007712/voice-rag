import type { GuardrailInfo } from "@/lib/types";
import { PixelPanel } from "./PixelPanel";
import { GuardrailBanner } from "./GuardrailBanner";

interface AnswerPanelProps {
  answer: string | null;
  explanation: string | null;
  guardrail: GuardrailInfo | null;
  noAnswer?: boolean;
  loading: boolean;
}

export function AnswerPanel({ answer, explanation, guardrail, noAnswer, loading }: AnswerPanelProps) {
  const blocked = guardrail?.triggered && guardrail.category !== "none";
  const isNoAnswer = noAnswer === true;

  // Label & tab colour
  const label = blocked ? "No answer given" : isNoAnswer ? "Nothing relevant found" : "Answer";
  const tone = blocked || isNoAnswer ? "alert" : "pipe";

  return (
    <PixelPanel
      label={label}
      tone={tone}
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
      ) : isNoAnswer ? (
        <div className="rounded border-2 border-alert/60 bg-alert/10 px-4 py-5">
          <p className="font-pixel text-[10px] uppercase tracking-widest text-alert sm:text-[11px]">
            ✕ Nothing relevant found
          </p>
          <p className="mt-2 font-body text-sm leading-relaxed text-alert/80 sm:text-base">
            No matching information was found in the knowledge base for your query.
            Please try rephrasing your question.
          </p>
        </div>
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
