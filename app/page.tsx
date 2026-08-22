"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { HudBar } from "@/components/HudBar";
import { MicOrb } from "@/components/MicOrb";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import { AnswerPanel } from "@/components/AnswerPanel";
import { CitationsPanel } from "@/components/CitationsPanel";
import { LatencyPanel } from "@/components/LatencyPanel";
import { ErrorBanner } from "@/components/ErrorBanner";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { ensureBackendAwake, submitTextQuery, submitVoiceQuery } from "@/lib/api";
import { QueryControls } from "@/components/QueryControls";
import { computeAggregateStats, getLatencyHistory, recordLatencySample } from "@/lib/latencyStats";
import { ApiError, type AggregateLatencyStats, type PipelineStage, type QueryLanguage, type QueryResponse } from "@/lib/types";

const BUSY_PHASES: PipelineStage[] = [
  "uploading",
  "transcribing",
  "retrieving",
  "generating",
];

export default function Home() {
  const recorder = useAudioRecorder();
  const [phase, setPhase] = useState<PipelineStage>("idle");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [queryCount, setQueryCount] = useState(0);
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState<QueryLanguage>("en");
  const [autoSend, setAutoSend] = useState(true);
  const [latencyStats, setLatencyStats] = useState<AggregateLatencyStats>(() =>
    computeAggregateStats(getLatencyHistory()),
  );

  const abortRef = useRef<AbortController | null>(null);
  const stageTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const processedBlobRef = useRef<Blob | null>(null);

  // Proactively ping sleeping Render backend in background on load
  useEffect(() => {
    void ensureBackendAwake();
  }, []);

  const clearStageTimers = () => {
    stageTimersRef.current.forEach(clearTimeout);
    stageTimersRef.current = [];
  };

  // Reflect the recorder's own lifecycle into the pipeline phase / errors.
  useEffect(() => {
    if (recorder.status === "recording") {
      setPhase("recording");
      setErrorMessage(null);
    } else if (recorder.status === "denied" || recorder.status === "unsupported" || recorder.status === "error") {
      setPhase("error");
      setErrorMessage(recorder.errorMessage);
    }
  }, [recorder.status, recorder.errorMessage]);

  const runQuery = useCallback(async (request: () => Promise<QueryResponse>) => {
    setPhase("uploading");
    setErrorMessage(null);
    setResult(null);

    const controller = new AbortController();
    abortRef.current = controller;

    clearStageTimers();
    stageTimersRef.current = [
      setTimeout(() => setPhase((p) => (p === "uploading" ? "transcribing" : p)), 250),
      setTimeout(() => setPhase((p) => (p === "transcribing" ? "retrieving" : p)), 700),
      setTimeout(() => setPhase((p) => (p === "retrieving" ? "generating" : p)), 1200),
    ];

    try {
      const data = await request();
      clearStageTimers();
      setResult(data);
      setLatencyStats(computeAggregateStats(recordLatencySample(data.latency)));
      setPhase("done");
      setQueryCount((c) => c + 1);
    } catch (err) {
      clearStageTimers();
      setPhase("error");
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Something went wrong. Please try again.");
      }
    }
  }, []);

  const runVoiceQuery = useCallback((blob: Blob) => {
    return runQuery(() => submitVoiceQuery(blob, { signal: abortRef.current?.signal, language }));
  }, [language, runQuery]);

  const runTextQuery = useCallback((text: string, queryLanguage: QueryLanguage) => {
    return runQuery(() => submitTextQuery(text, queryLanguage, { signal: abortRef.current?.signal }));
  }, [runQuery]);

  // When recording stops and we have audio, process or load it into controls
  useEffect(() => {
    if (recorder.status !== "stopped" || !recorder.audioBlob) return;
    if (processedBlobRef.current === recorder.audioBlob) return;
    processedBlobRef.current = recorder.audioBlob;

    if (autoSend) {
      void runVoiceQuery(recorder.audioBlob);
    }
  }, [recorder.status, recorder.audioBlob, autoSend, runVoiceQuery]);

  const handleMicClick = () => {
    if (recorder.status === "recording") {
      recorder.stop();
    } else if (BUSY_PHASES.includes(phase)) {
      handleCancel();
    } else {
      setResult(null);
      setErrorMessage(null);
      processedBlobRef.current = null;
      recorder.reset();
      void recorder.start({ autoStopAfterSilenceMs: autoSend ? 500 : 1000 });
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    clearStageTimers();
    setPhase("idle");
  };

  const handleTextAsk = () => {
    if (!query.trim()) return;
    void runTextQuery(query, language);
  };

  const handleLanguageChange = (nextLang: QueryLanguage) => {
    setLanguage(nextLang);
  };

  const handleClear = () => {
    setQuery("");
    setResult(null);
    setErrorMessage(null);
    setPhase("idle");
  };

  const busy = BUSY_PHASES.includes(phase);
  const showLoadingSkeletons = busy && phase !== "uploading";

  return (
    <div className="flex min-h-screen flex-col bg-sky-cloud text-ink font-body selection:bg-coin selection:text-ink">
      <HudBar lastTotalMs={result?.latency?.total_ms ?? null} queryCount={queryCount} />

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-6 sm:px-6 sm:py-8">
        <section className="flex flex-col items-center justify-center pt-2 text-center sm:pt-4">
          <MicOrb
            phase={phase}
            level={recorder.level}
            elapsedMs={recorder.elapsedMs}
            onPress={handleMicClick}
          />
          {phase === "idle" && (
            <p className="mt-3 font-pixel text-[10px] text-sky-night/70 sm:text-xs">
              CLICK ORB TO RECORD
            </p>
          )}
          {phase === "recording" && (
            <p className="mt-3 font-pixel text-[10px] text-alert animate-pulse sm:text-xs">
              RECORDING... CLICK ORB TO SUBMIT
            </p>
          )}
          {busy && (
            <p className="mt-3 font-pixel text-[10px] text-coin sm:text-xs">
              PROCESSING PIPELINE...
            </p>
          )}
          {busy && (
            <button
              type="button"
              onClick={handleCancel}
              className="mt-1 border-2 border-cream/60 px-3 py-1 font-pixel text-[8px] text-cream/80 hover:bg-cream/10"
            >
              CANCEL
            </button>
          )}
        </section>

        <QueryControls
          query={query}
          language={language}
          autoSend={autoSend}
          disabled={busy || phase === "recording"}
          onQueryChange={setQuery}
          onLanguageChange={handleLanguageChange}
          onAutoSendChange={setAutoSend}
          onAsk={() => void handleTextAsk()}
          onClear={handleClear}
        />

        {errorMessage && (
          <ErrorBanner message={errorMessage} onDismiss={() => setErrorMessage(null)} />
        )}

        <section className="grid items-start gap-5 lg:grid-cols-[minmax(0,1.8fr)_minmax(18rem,1fr)] lg:gap-6">
          <div>
            <AnswerPanel
              answer={result?.answer ?? null}
              explanation={result?.explanation ?? null}
              guardrail={result?.guardrail ?? null}
              noAnswer={result?.no_answer ?? false}
              loading={phase === "generating"}
            />

            <CitationsPanel
              citations={result?.citations ?? []}
              loading={phase === "retrieving" || phase === "generating"}
            />
          </div>

          <div>
            <TranscriptPanel
              transcript={result?.transcript ?? null}
              loading={showLoadingSkeletons && (phase as string) !== "uploading"}
            />

            <LatencyPanel latest={result?.latency ?? null} aggregate={latencyStats} targetMs={200} />
          </div>
        </section>
      </main>

      <footer aria-hidden className="h-10 brick-ground border-t-4 border-ink sm:h-14" />
    </div>
  );
}
