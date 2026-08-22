"use client";

import type { QueryLanguage } from "@/lib/types";

interface QueryControlsProps {
  query: string;
  language: QueryLanguage;
  autoSend: boolean;
  disabled: boolean;
  onQueryChange: (value: string) => void;
  onLanguageChange: (value: QueryLanguage) => void;
  onAutoSendChange: (value: boolean) => void;
  onAsk: () => void;
  onClear: () => void;
}

export function QueryControls({ query, language, autoSend, disabled, onQueryChange, onLanguageChange, onAutoSendChange, onAsk, onClear }: QueryControlsProps) {
  return (
    <section className="mt-2 border-2 border-ink bg-cream-panel p-3 shadow-pixel sm:p-4">
      <form className="flex flex-col gap-2 sm:flex-row" onSubmit={(event) => { event.preventDefault(); onAsk(); }}>
        <input value={query} onChange={(event) => onQueryChange(event.target.value)} disabled={disabled} placeholder="Speak, or type a question..." aria-label="Question" className="min-h-11 min-w-0 flex-1 border-2 border-ink bg-cream px-3 font-mono text-sm text-ink outline-none placeholder:text-ink/45 focus:ring-2 focus:ring-pipe" />
        <button type="submit" disabled={disabled || !query.trim()} className="min-h-11 border-2 border-ink bg-pipe px-4 font-pixel text-[10px] text-cream shadow-pixel-sm disabled:cursor-not-allowed disabled:opacity-50">ASK</button>
        <button type="button" onClick={onClear} disabled={disabled} className="min-h-11 border-2 border-ink bg-coin px-4 font-pixel text-[10px] text-ink shadow-pixel-sm disabled:opacity-50">CLEAR</button>
      </form>
      <div className="mt-3 flex flex-col gap-3 border-t-2 border-ink/15 pt-3 sm:flex-row sm:items-end sm:justify-between">
        <label className="flex flex-col gap-1 font-pixel text-[9px] text-ink/70">
          LANGUAGE
          <select value={language} onChange={(event) => onLanguageChange(event.target.value as QueryLanguage)} disabled={disabled} className="min-h-10 border-2 border-ink bg-cream px-2 font-mono text-sm text-ink outline-none focus:ring-2 focus:ring-pipe">
            <option value="en">English</option>
            <option value="hi">हिन्दी · Hindi</option>
          </select>
        </label>
        <label className="flex cursor-pointer items-center gap-2 font-mono text-xs text-ink/75">
          <input type="checkbox" checked={autoSend} onChange={(event) => onAutoSendChange(event.target.checked)} disabled={disabled} className="h-4 w-4 accent-pipe" />
          Send after ~2s of silence
        </label>
      </div>
      <p className="mt-3 font-mono text-[11px] leading-relaxed text-ink/60">The mic stops after each question. Choose a language before recording for a more accurate Sarvam transcript.</p>
    </section>
  );
}