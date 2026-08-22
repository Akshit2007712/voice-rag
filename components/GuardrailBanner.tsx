import type { GuardrailCategory } from "@/lib/types";
import { PixelWarnIcon } from "./PixelIcons";

const CATEGORY_COPY: Record<Exclude<GuardrailCategory, "none">, { title: string; fallback: string }> = {
  off_topic: {
    title: "Outside this dataset",
    fallback:
      "That question doesn't look related to the indexed dataset, so I'm not going to guess at an answer.",
  },
  unsafe: {
    title: "Blocked by safety filter",
    fallback:
      "That request was flagged as unsafe or inappropriate and won't be answered.",
  },
  ungrounded: {
    title: "Couldn't ground an answer",
    fallback:
      "Retrieval didn't surface passages that clearly support an answer, so none is being shown rather than risk a hallucination.",
  },
  empty_retrieval: {
    title: "Nothing relevant found",
    fallback:
      "No passages in the index cleared the relevance threshold for this question.",
  },
};

interface GuardrailBannerProps {
  category: Exclude<GuardrailCategory, "none">;
  reason?: string;
}

export function GuardrailBanner({ category, reason }: GuardrailBannerProps) {
  const copy = CATEGORY_COPY[category];
  return (
    <div className="flex items-start gap-3 border-2 border-ink bg-alert px-4 py-3 text-cream shadow-pixel-sm">
      <PixelWarnIcon className="mt-0.5 h-5 w-5 shrink-0" />
      <div>
        <p className="font-pixel text-[10px] uppercase tracking-wide sm:text-xs">
          {copy.title}
        </p>
        <p className="mt-1.5 font-body text-sm leading-relaxed text-cream/95">
          {reason || copy.fallback}
        </p>
      </div>
    </div>
  );
}
