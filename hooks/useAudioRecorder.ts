"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderStatus =
  | "idle"
  | "requesting-permission"
  | "recording"
  | "stopped"
  | "denied"
  | "unsupported"
  | "error";

interface UseAudioRecorderResult {
  status: RecorderStatus;
  /** 0–1 live input level, updated ~30x/sec while recording. Drives the mic UI. */
  level: number;
  elapsedMs: number;
  audioBlob: Blob | null;
  errorMessage: string | null;
  start: (options?: { autoStopAfterSilenceMs?: number }) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

const CANDIDATE_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];

function pickSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  return CANDIDATE_MIME_TYPES.find(
    (type) => MediaRecorder.isTypeSupported?.(type),
  );
}

export function useAudioRecorder(): UseAudioRecorderResult {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [level, setLevel] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const levelRef = useRef(0);
  const lastAudibleAtRef = useRef(0);
  const heardSpeechRef = useRef(false);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    if (silenceTimerRef.current) clearInterval(silenceTimerRef.current);
    silenceTimerRef.current = null;
  }, []);

  useEffect(() => cleanupStream, [cleanupStream]);

  const meterLoop = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(data);
    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
      const centered = (data[i]! - 128) / 128;
      sumSquares += centered * centered;
    }
    const rms = Math.sqrt(sumSquares / data.length);
    const nextLevel = Math.min(1, rms * 4);
    levelRef.current = nextLevel;
    setLevel(nextLevel);
    rafRef.current = requestAnimationFrame(meterLoop);
  }, []);

  const start = useCallback(async (options?: { autoStopAfterSilenceMs?: number }) => {
    setErrorMessage(null);
    setAudioBlob(null);

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      setStatus("unsupported");
      setErrorMessage(
        "This browser doesn't support in-page microphone recording. Try the latest Chrome, Edge, or Firefox.",
      );
      return;
    }

    setStatus("requesting-permission");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      });
      streamRef.current = stream;

      const mimeType = pickSupportedMimeType();
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      );
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const mimeTypeUsed = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, {
          type: mimeTypeUsed,
        });
        if (blob.size === 0) {
          setStatus("error");
          setErrorMessage("No audio captured. Please check your microphone and speak clearly.");
          cleanupStream();
          return;
        }
        setAudioBlob(blob);
        setStatus("stopped");
        cleanupStream();
      };
      recorder.onerror = () => {
        setStatus("error");
        setErrorMessage("Recording failed unexpectedly. Please try again.");
        cleanupStream();
      };

      // Live level meter with active AudioContext resume for browser autoplay policies
      const AudioContextCtor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const audioCtx = new AudioContextCtor();
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      sourceRef.current = source;
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      rafRef.current = requestAnimationFrame(meterLoop);

      startTimeRef.current = Date.now();
      lastAudibleAtRef.current = Date.now();
      heardSpeechRef.current = false;
      setElapsedMs(0);
      timerRef.current = setInterval(() => {
        setElapsedMs(Date.now() - startTimeRef.current);
      }, 100);

      recorder.start(100);
      setStatus("recording");

      const silenceMs = options?.autoStopAfterSilenceMs;
      const MAX_RECORDING_MS = 10_000; // Auto-stop after 10s max recording time
      silenceTimerRef.current = setInterval(() => {
        const elapsed = Date.now() - startTimeRef.current;
        // Detect speech if input level passes 0.008 threshold
        if (levelRef.current > 0.008) {
          heardSpeechRef.current = true;
          lastAudibleAtRef.current = Date.now();
        }
        // Auto-stop immediately on silence after speech is detected
        if (silenceMs && heardSpeechRef.current && Date.now() - lastAudibleAtRef.current >= silenceMs) {
          if (mediaRecorderRef.current?.state === "recording") {
            mediaRecorderRef.current.stop();
          }
          return;
        }
        // Safety cap: auto-stop after 10s of total recording time
        if (elapsed >= MAX_RECORDING_MS) {
          if (mediaRecorderRef.current?.state === "recording") {
            mediaRecorderRef.current.stop();
          }
        }
      }, 50);
    } catch (err) {
      cleanupStream();
      const isPermissionError =
        err instanceof DOMException &&
        (err.name === "NotAllowedError" || err.name === "SecurityError");
      setStatus(isPermissionError ? "denied" : "error");
      setErrorMessage(
        isPermissionError
          ? "Microphone access was denied. Allow microphone permission in your browser to ask a question."
          : "Couldn't start the microphone. It may be in use by another app.",
      );
    }
  }, [cleanupStream, meterLoop]);

  const stop = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setLevel(0);
  }, []);

  const reset = useCallback(() => {
    cleanupStream();
    setStatus("idle");
    setLevel(0);
    setElapsedMs(0);
    setAudioBlob(null);
    setErrorMessage(null);
    chunksRef.current = [];
  }, [cleanupStream]);

  return { status, level, elapsedMs, audioBlob, errorMessage, start, stop, reset };
}
