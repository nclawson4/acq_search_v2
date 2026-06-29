import { NextRequest, NextResponse } from "next/server";
import { Ratelimit, type Duration } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

// Generous ceiling: only catches a genuinely frozen/hung Modal container, never
// a slow-but-valid cold-start search. On abort we can report "timeout" distinctly
// from "connection refused". Tunable via SEARCH_PROXY_TIMEOUT_MS.
const PROXY_TIMEOUT_MS = Number(process.env.SEARCH_PROXY_TIMEOUT_MS) || 120_000;

// --- Guardrails (rate limit + daily cost cap) --------------------------------
// Enforced here at the edge so abusive traffic is rejected before the Modal
// backend is even woken. The /api/status health monitor calls the backend
// directly (not this route), so it is naturally exempt. All guards FAIL OPEN
// when Upstash is unconfigured (the demo keeps working) — set the env to arm.
const RATE_LIMIT = Number(process.env.SEARCH_RATE_LIMIT) || 10;
const RATE_WINDOW = (process.env.SEARCH_RATE_WINDOW || "5 m") as Duration;
const MAX_DAILY_COST_USD = Number(process.env.MAX_DAILY_COST_USD) || 5;

let _guardsInit = false;
let _redis: Redis | null = null;
let _limiter: Ratelimit | null = null;

function getGuards(): { redis: Redis | null; limiter: Ratelimit | null } {
  if (_guardsInit) return { redis: _redis, limiter: _limiter };
  _guardsInit = true;
  const url = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
  if (url && token) {
    _redis = new Redis({ url, token });
    _limiter = new Ratelimit({
      redis: _redis,
      limiter: Ratelimit.slidingWindow(RATE_LIMIT, RATE_WINDOW),
      prefix: "v2:rl",
      analytics: false,
    });
  }
  return { redis: _redis, limiter: _limiter };
}

// Vercel sets `x-real-ip` to the authoritative client IP. The leading entry of
// `x-forwarded-for` is user-controllable and must never be the primary key.
function clientIp(req: NextRequest): string {
  const real = req.headers.get("x-real-ip");
  if (real) return real.trim();
  const vf = req.headers.get("x-vercel-forwarded-for");
  if (vf) return vf.split(",")[0].trim();
  const xff = req.headers.get("x-forwarded-for");
  if (xff) {
    const parts = xff.split(",").map((s) => s.trim()).filter(Boolean);
    if (parts.length) return parts[parts.length - 1];
  }
  return "anon";
}

function logProxyRefusal(requestId: string, reason: string, ip: string) {
  try {
    console.warn(
      JSON.stringify({
        service: "acq-search-v2-proxy",
        event: "refusal",
        request_id: requestId,
        reason,
        ip,
      }),
    );
  } catch {
    /* never throw from logging */
  }
}

// Returns a refusal response (429/402) to short-circuit, or null to proceed.
async function applyGuards(req: NextRequest, requestId: string): Promise<NextResponse | null> {
  const { redis, limiter } = getGuards();
  if (!limiter || !redis) return null; // unconfigured → guards disabled
  const ip = clientIp(req);
  try {
    const { success } = await limiter.limit(`ip:${ip}`);
    if (!success) {
      logProxyRefusal(requestId, "rate_limit", ip);
      return NextResponse.json(
        { error: "Too many requests. Please wait a few minutes and try again." },
        { status: 429, headers: { "x-request-id": requestId, "retry-after": "300" } },
      );
    }
    const key = `v2:spend:${new Date().toISOString().slice(0, 10)}`;
    const raw = await redis.get<number | string>(key);
    const micro = typeof raw === "string" ? Number(raw) || 0 : raw ?? 0;
    if (micro / 1_000_000 >= MAX_DAILY_COST_USD) {
      logProxyRefusal(requestId, "cost_cap", ip);
      return NextResponse.json(
        { error: "Daily demo cost cap reached. Search resumes tomorrow." },
        { status: 402, headers: { "x-request-id": requestId } },
      );
    }
    return null;
  } catch {
    // Fail open on a transient Upstash error — never break the live demo over a blip.
    return null;
  }
}

function newRequestId(): string {
  return crypto.randomUUID().replace(/-/g, "");
}

// One JSON line to the Vercel log drain, carrying the same request_id the backend
// stamps — so a failed search can be joined across the proxy and Modal sides.
function logProxyError(requestId: string, reason: string, detail: string) {
  try {
    console.error(
      JSON.stringify({
        service: "acq-search-v2-proxy",
        event: "proxy.error",
        request_id: requestId,
        failing_dependency: "backend_unreachable",
        reason,
        detail: detail.slice(0, 500),
      }),
    );
  } catch {
    /* never throw from logging */
  }
}

async function forward(url: string, init: RequestInit, requestId: string) {
  const headers = new Headers(init.headers);
  headers.set("x-request-id", requestId);
  const resp = await fetch(url, {
    ...init,
    headers,
    signal: AbortSignal.timeout(PROXY_TIMEOUT_MS),
  });
  const text = await resp.text();
  return new NextResponse(text, {
    status: resp.status,
    headers: {
      "content-type": resp.headers.get("content-type") || "application/json",
      "x-request-id": requestId,
    },
  });
}

export async function POST(req: NextRequest) {
  const requestId = req.headers.get("x-request-id") || newRequestId();
  const refusal = await applyGuards(req, requestId);
  if (refusal) return refusal;
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "invalid json" },
      { status: 400, headers: { "x-request-id": requestId } },
    );
  }
  try {
    return await forward(
      `${BACKEND_URL}/search`,
      { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) },
      requestId,
    );
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === "TimeoutError";
    const detail = err instanceof Error ? err.message : String(err);
    logProxyError(requestId, isTimeout ? "timeout" : "fetch_failed", detail);
    return NextResponse.json(
      { error: "backend unreachable", detail },
      { status: 502, headers: { "x-request-id": requestId } },
    );
  }
}

export async function GET(req: NextRequest) {
  const requestId = req.headers.get("x-request-id") || newRequestId();
  const refusal = await applyGuards(req, requestId);
  if (refusal) return refusal;
  const { searchParams } = new URL(req.url);
  const params = searchParams.toString();
  try {
    return await forward(`${BACKEND_URL}/search?${params}`, {}, requestId);
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === "TimeoutError";
    const detail = err instanceof Error ? err.message : String(err);
    logProxyError(requestId, isTimeout ? "timeout" : "fetch_failed", detail);
    return NextResponse.json(
      { error: "backend unreachable", detail },
      { status: 502, headers: { "x-request-id": requestId } },
    );
  }
}
