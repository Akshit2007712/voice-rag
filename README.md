# Voice-RAG Frontend

Next.js 14 (App Router) + TypeScript + Tailwind frontend for a voice-in,
cited-answer-out RAG pipeline. Record a question with the mic, watch it
move through the pipeline, and see the transcript, grounded answer,
citations, and latency breakdown come back.

## Stack

- **Next.js 14** (App Router), **TypeScript**, **Tailwind CSS**
- **MediaRecorder API** (browser) for audio capture — no recording libraries
- **fetch API** for upload + JSON, with timeout, retry/backoff, and cancellation
- No backend code here — this only talks to the RAG backend over HTTP

## Getting started

```bash
npm install
cp .env.local.example .env.local   # point at your backend
npm run dev
```

Open http://localhost:3000. Microphone access requires **HTTPS or
localhost** — that's a browser restriction, not this app's.

## Wiring up the real backend

Everything the frontend expects from the backend lives in one file:
**`lib/types.ts`** (the contract) and **`lib/api.ts`** (the client). If your
endpoint, field name, or response shape differs, those are the only two
files to touch.

**Request** — `POST {NEXT_PUBLIC_API_BASE_URL}/api/query`, `multipart/form-data`,
field `audio` (the MediaRecorder blob, webm/opus by default with mp4/ogg
fallback depending on browser support).

**Response** — `application/json`:

```jsonc
{
  "transcript": "string",
  "answer": "string",
  "citations": [
    { "id": "string", "text": "string", "source": "string", "score": 0.91, "strategy": "semantic" }
  ],
  "latency": {
    "stt_ms": 40,
    "retrieval_ms": 55,
    "generation_ms": 90,
    "total_ms": 185
  },
  "guardrail": {
    "triggered": false,
    "category": "none" // "off_topic" | "unsafe" | "ungrounded" | "empty_retrieval"
  }
}
```

When `guardrail.triggered` is `true`, the UI shows the refusal banner
instead of the answer text — `answer` can be empty in that case. `citations`
can be `[]`.

### About the mid-pipeline stage labels

The backend contract above is a single request/response call, so
"Transcribing → Retrieving → Generating" under the mic button is an
**optimistic UI progression** (timed locally), not real server-sent
events — the actual completion is still driven by the fetch resolving.
If the backend later exposes SSE/websocket stage events, replace the
`setTimeout` progression in `app/page.tsx`'s `runQuery` with real
event handling; the phase state machine (`PipelineStage` in
`lib/types.ts`) already supports it.

### Latency numbers

`components/LatencyPanel.tsx` shows two things:

1. The **latest query's** stt/retrieval/generation/total breakdown, with a
   marker at the 200ms target.
2. **Session P50/P70/P100**, computed client-side in `lib/latencyStats.ts`
   from every completed query this browser has made (rolled into
   `localStorage`, capped at the last 200 samples).

This satisfies "show P50/P70/P100" for real, in-app traffic, but it is a
*session* view — for the offline benchmark deliverable (latency measured
across a batch of test queries in one report), run that as a separate
script against the backend directly; this UI isn't a load-testing tool.

## Design notes

Retro pixel-arcade aesthetic, chosen deliberately rather than left as a
generic default:

- **Palette** — sky blue world (`sky`), cream "dialogue box" panels
  (`cream`), thick ink outlines (`ink`), plus four functional accents
  borrowed from a platformer's HUD vocabulary: gold `coin` (metrics/score),
  green `pipe` (success/grounded), brick `brick` (generic error), red
  `alert` (recording / guardrail refusal). Defined once in
  `tailwind.config.ts`.
- **Type** — `Press Start 2P` for labels, tabs, and buttons only (it's
  illegible at paragraph size, so it never carries body copy); `Inter` for
  transcript/answer prose so it stays actually readable; `JetBrains Mono`
  for every number — timestamps, latency, scores — so data reads as data.
- **Signature element** — the mic button is a chunky "power block": it
  idles with a slow bob, turns red with expanding pulse rings and a live
  amplitude-reactive waveform while recording (real mic input via
  `AnalyserNode`, not a canned animation), and flips like a coin while the
  pipeline is working.
- **Structure** — every result panel reuses one "dialogue box" motif (a
  bordered cream panel with a tab bitten out of the top edge), so the
  transcript, answer, citations, and latency read as one consistent
  language instead of four different card styles.

## Project layout

```
app/
  layout.tsx        fonts + metadata
  page.tsx           orchestrates recording -> upload -> pipeline phases
  globals.css        pixel background, scanlines, focus states
components/
  MicOrb.tsx          the record button (signature element)
  HudBar.tsx          top score/latency strip
  TranscriptPanel.tsx
  AnswerPanel.tsx
  GuardrailBanner.tsx refusal state (off-topic / unsafe / ungrounded)
  CitationsPanel.tsx
  LatencyPanel.tsx
  ErrorBanner.tsx     permission/network/pipeline errors
  PixelPanel.tsx      shared bordered-panel container
  PixelIcons.tsx      hand-drawn pixel-grid SVG icons (mic/stop/coin/warn)
hooks/
  useAudioRecorder.ts MediaRecorder capture + live level meter
lib/
  types.ts            API contract + shared types
  api.ts              fetch client: timeout, retry/backoff, validation
  latencyStats.ts      rolling history + percentile math
```

## Error handling covered in the UI

- Microphone permission denied → explicit banner, mic button re-enables
- Unsupported browser (no MediaRecorder/getUserMedia) → explicit banner
- Upload/network failure → retried with backoff, then surfaced with a
  clear message and a retry (just tap the mic again)
- Request timeout (15s) → surfaced distinctly from a network failure
- Manual cancel mid-flight → aborts the in-flight request via
  `AbortController`
- Malformed backend response (missing `transcript`/`answer`) → treated as
  an error rather than silently rendering `undefined`
- Guardrail refusals (off-topic / unsafe / ungrounded / empty retrieval)
  → rendered as a distinct state, never mixed into the answer panel as if
  it were a real answer
