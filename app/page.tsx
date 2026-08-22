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
import { ensureBackendAwake, resetWakeState, submitTextQuery, submitVoiceQuery } from "@/lib/api";
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
  // Backend warm-up tracking for Render free-tier cold starts
  const [backendStatus, setBackendStatus] = useState<"warming" | "ready" | "offline">("warming");

  const abortRef = useRef<AbortController | null>(null);
  const stageTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const processedBlobRef = useRef<Blob | null>(null);

  // Wake up Render backend on page load, show status to user
  useEffect(() => {
    setBackendStatus("warming");
    ensureBackendAwake(() => setBackendStatus("warming"))
      .then((ok) => setBackendStatus(ok ? "ready" : "offline"))
      .catch(() => setBackendStatus("offline"));
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

    // The backend is a single request/response call, so these staged
    // labels are optimistic UI progression, not events from the server.
    // Swap for real SSE/websocket stage events here if the backend adds them.
    clearStageTimers();
    stageTimersRef.current = [
      setTimeout(() => setPhase((p) => (p === "uploading" ? "transcribing" : p)), 250),
      setTimeout(() => setPhase((p) => (p === "transcribing" ? "retrieving" : p)), 700),
      setTimeout(() => setPhase((p) => (p === "retrieving" ? "generating" : p)), 1200),
    ];

    try {
      // If backend is still warming, wait for it before sending the actual query
      if (backendStatus !== "ready") {
        setPhase("transcribing");
        const ok = await ensureBackendAwake();
        if (!ok) {
          clearStageTimers();
          setPhase("error");
          setErrorMessage("Backend is offline. Please wait a moment and try again.");
          return;
        }
        setBackendStatus("ready");
        resetWakeState();
      }
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
        if (err.message.includes("reach the backend") || err.message.includes("timed out")) {
          setErrorMessage("⏳ Backend is waking up (Render free tier). Please wait 30s and try again.");
          setBackendStatus("warming");
          resetWakeState();
          ensureBackendAwake().then((ok) => setBackendStatus(ok ? "ready" : "offline"));
        } else {
          setErrorMessage(err.message);
        }
      } else {
        setErrorMessage("Something went wrong. Please try again.");
      }
    }
  }, [backendStatus]);

  const runVoiceQuery = useCallback((blob: Blob) => {
    return runQuery(() => submitVoiceQuery(blob, { signal: abortRef.current?.signal, language }));
  }, [language, runQuery]);

  const runTextQuery = useCallback((text: string, queryLanguage: QueryLanguage) => {
    return runQuery(() => submitTextQuery(text, queryLanguage, { signal: abortRef.current?.signal }));
  }, [runQuery]);

  // Fire the pipeline exactly once per freshly recorded blob.
  useEffect(() => {
    if (
      recorder.status === "stopped" &&
      recorder.audioBlob &&
      recorder.audioBlob.size > 0 &&
      recorder.audioBlob !== processedBlobRef.current
    ) {
      processedBlobRef.current = recorder.audioBlob;
      void runVoiceQuery(recorder.audioBlob);
    }
  }, [recorder.status, recorder.audioBlob, runVoiceQuery]);

  const handleMicPress = () => {
    if (phase === "recording") {
      recorder.stop();
      return;
    }
    if (BUSY_PHASES.includes(phase)) return;

    setErrorMessage(null);
    recorder.reset();
    void recorder.start({ autoStopAfterSilenceMs: autoSend ? 2000 : undefined });
  };

  useEffect(() => {
    const savedLanguage = window.localStorage.getItem("voice-rag:language");
    if (savedLanguage === "en" || savedLanguage === "hi") setLanguage(savedLanguage);
  }, []);

  const handleLanguageChange = (value: QueryLanguage) => {
    setLanguage(value);
    window.localStorage.setItem("voice-rag:language", value);
  };

  const busy = BUSY_PHASES.includes(phase);

  const handleTextAsk = useCallback(async () => {
    const text = query.trim();
    if (!text || busy) return;
    setQuery("");
    await runTextQuery(text, language);
  }, [busy, language, query, runTextQuery]);

  const handleClear = () => {
    abortRef.current?.abort();
    clearStageTimers();
    recorder.reset();
    setQuery("");
    setResult(null);
    setErrorMessage(null);
    setPhase("idle");
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    clearStageTimers();
    setPhase("error");
    setErrorMessage("Cancelled before the answer came back.");
  };

  const showLoadingSkeletons = busy;

  return (
    <div className="flex min-h-dvh flex-col scanlines">
      <HudBar lastTotalMs={result?.latency.total_ms ?? null} queryCount={queryCount} />

      <div aria-hidden="true" className="sky-scene">
        <span className="pixel-cloud cloud-one" />
        <span className="pixel-cloud cloud-two" />
        <span className="pixel-cloud cloud-three" />
        <span className="pixel-bird bird-one" />
        <span className="pixel-bird bird-two" />
        <span className="pixel-bird bird-three" />
        <span className="pixel-flower flower-one" />
        <span className="pixel-flower flower-two" />
        <span className="pixel-flower flower-three" />
        <span className="pixel-flower flower-four" />
      </div>

      <main className="relative z-10 mx-auto w-full max-w-6xl flex-1 px-4 pb-16 sm:px-6">
        <section className="flex flex-col items-center gap-4 py-10 sm:py-14">
          <p className="text-center font-pixel text-[11px] leading-relaxed text-cream drop-shadow-[2px_2px_0_rgba(0,0,0,0.35)] sm:text-sm">
            PRESS THE BLOCK
          </p>
          <MicOrb
            phase={phase}
            level={recorder.level}
            elapsedMs={recorder.elapsedMs}
            onPress={handleMicPress}
            disabled={recorder.status === "requesting-permission"}
          />
          {recorder.status === "requesting-permission" && (
            <p className="font-mono text-xs text-cream/80">
              Waiting on microphone permission…
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

        {backendStatus === "warming" && !errorMessage && (
          <div className="mt-3 flex items-center gap-2 border-2 border-coin bg-coin/10 px-4 py-2 font-mono text-xs text-ink">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-coin" />
            Backend warming up (Render free tier)... Please wait ~30 seconds before first query.
          </div>
        )}
        {backendStatus === "ready" && !errorMessage && (
          <div className="mt-3 flex items-center gap-2 border-2 border-pipe bg-pipe/10 px-4 py-2 font-mono text-xs text-ink">
            <span className="inline-block h-2 w-2 rounded-full bg-pipe" />
            Backend ready ✓
          </div>
        )}
        {errorMessage && (
          <ErrorBanner message={errorMessage} onDismiss={() => setErrorMessage(null)} />
        )}

        <section className="grid items-start gap-5 lg:grid-cols-[minmax(0,1.8fr)_minmax(18rem,1fr)] lg:gap-6">
          <div>
            <AnswerPanel
              answer={result?.answer ?? null}
              explanation={result?.explanation ?? null}
              guardrail={result?.guardrail ?? null}
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
              loading={showLoadingSkeletons && phase !== "uploading"}
            />

            <LatencyPanel latest={result?.latency ?? null} aggregate={latencyStats} targetMs={200} />
          </div>
        </section>
      </main>

      <footer aria-hidden className="h-10 brick-ground border-t-4 border-ink sm:h-14" />
    </div>
  );
}
