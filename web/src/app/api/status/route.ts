// Liveness + smoke-test endpoint. Called by the homepage failure-recovery section
// and by any external monitor (Cron, uptime, etc.). Returns a structured payload:
//
//   {
//     backend: { ok, latency_ms, indexed_frames, topic_segments },
//     smoke:   { ok, latency_ms, query, top_video, top_judge_score },
//     overall: "ok" | "degraded" | "down",
//     checked_at: ISO timestamp
//   }
//
// The smoke check is a real search against a canary query — same code path the
// editor would hit. If it returns non-zero results and a non-null judge score
// the system is functioning end-to-end, not just "the container is up."
import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const SMOKE_QUERY = "Alex writing on a whiteboard";

export const dynamic = "force-dynamic"; // never cached — must reflect real state

type BackendCheck = {
  ok: boolean;
  latency_ms: number;
  indexed_frames?: number;
  topic_segments?: number;
  error?: string;
};

type SmokeCheck = {
  ok: boolean;
  latency_ms: number;
  query: string;
  top_video?: string;
  top_judge_score?: number | null;
  results_returned?: number;
  error?: string;
};

function newRequestId(): string {
  return crypto.randomUUID().replace(/-/g, "");
}

async function checkBackend(): Promise<BackendCheck> {
  const t0 = Date.now();
  try {
    const r = await fetch(`${BACKEND_URL}/healthz`, {
      headers: { "x-request-id": newRequestId() },
      signal: AbortSignal.timeout(8000),
      cache: "no-store",
    });
    const dt = Date.now() - t0;
    if (!r.ok) return { ok: false, latency_ms: dt, error: `HTTP ${r.status}` };
    const j = await r.json();
    return {
      ok: Boolean(j.ok),
      latency_ms: dt,
      indexed_frames: j.indexed_frames,
      topic_segments: j.topic_segments,
    };
  } catch (e) {
    return { ok: false, latency_ms: Date.now() - t0, error: String((e as Error)?.message || e) };
  }
}

async function runSmoke(): Promise<SmokeCheck> {
  const t0 = Date.now();
  try {
    // rerank:false keeps the smoke check fast and cheap — no judge LLM call.
    // We still get parser + hybrid retrieval, which is the meaningful end-to-end path.
    const r = await fetch(`${BACKEND_URL}/search`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-request-id": newRequestId() },
      body: JSON.stringify({ query: SMOKE_QUERY, k: 1, rerank: false }),
      signal: AbortSignal.timeout(45000),
      cache: "no-store",
    });
    const dt = Date.now() - t0;
    if (!r.ok) {
      return { ok: false, latency_ms: dt, query: SMOKE_QUERY, error: `HTTP ${r.status}` };
    }
    const j = await r.json();
    const top = j.results?.[0];
    return {
      ok: j.n > 0 && Boolean(top),
      latency_ms: dt,
      query: SMOKE_QUERY,
      top_video: top?.video_id,
      top_judge_score: top?.judge_score ?? null,
      results_returned: j.n,
    };
  } catch (e) {
    return {
      ok: false,
      latency_ms: Date.now() - t0,
      query: SMOKE_QUERY,
      error: String((e as Error)?.message || e),
    };
  }
}

export async function GET() {
  const [backend, smoke] = await Promise.all([checkBackend(), runSmoke()]);
  const overall =
    backend.ok && smoke.ok ? "ok" : backend.ok || smoke.ok ? "degraded" : "down";
  return NextResponse.json(
    {
      overall,
      backend,
      smoke,
      checked_at: new Date().toISOString(),
    },
    { headers: { "cache-control": "no-store" } },
  );
}
