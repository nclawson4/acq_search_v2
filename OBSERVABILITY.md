# Observability — finalized plan & Phase 1 implementation

This backend was the lightest-instrumented of the ACQ search services (no
tracing, no cost capture, swallowed errors). This document is the finalized plan
plus what Phase 1 actually ships.

## Principles (all honored by the code)

- **Additive & fail-open.** Every telemetry call is wrapped so a telemetry fault
  can never change a `/search` response. Nothing here touches the index build,
  the `.npz` bundling, the baked CLIP weights, `@modal.asgi_app()`, CORS, or the
  `/search` / `/healthz` response shapes.
- **Default-on with a kill switch.** Structured logging is on by default; set
  `TELEMETRY_ENABLED=0` to mute. OpenTelemetry/OTLP is a later phase and stays
  off until an `OTEL_EXPORTER_OTLP_ENDPOINT` is configured.
- **No new dependencies.** Phase 1 is stdlib-only (`ingest/lib/observability.py`).

## What Phase 1 implements

1. **Per-request correlation id.** The Vercel proxy mints/forwards an
   `x-request-id`; the Modal backend adopts it (`_build_response_async`) so the
   proxy log line and the backend logs for one search join on one id. Set in a
   `ContextVar` **before** the parser task is created so async work is correlated.
2. **Structured stage events** to stdout (flushed): `search.start`, `parse.done`,
   `retrieve.done`, `rerank.done`, `search.done`, plus `llm.cost` and `error`.
   Emitted the moment each stage completes so a frozen container still leaves a
   trail.
3. **LLM cost capture.** The previously-discarded `resp.usage` from the parser
   and judge is captured and priced (`record_llm_cost`); `search.done` carries
   the per-query `cost_usd`.
4. **Loud-but-graceful failures.** The parser still returns bare `None` on
   failure (regex fallback intact) but now logs a distinct `fallback_reason`
   (`no_api_key | import_error | timeout | json_error`); the judge still
   soft-fails but logs the dead-judge `error` with `http_status`.
5. **`/healthz` truthfulness.** Returns 503 when the index failed to load
   (previously a misleading `ok:true`).
6. **`search.done.filters`** snapshots the applied speaker/time filters so an
   empty result set (almost always over-filtering) is diagnosable.

## Review must-fixes incorporated

- Parser returns bare `None` only — reason goes to a log event, never the return
  value. `except` is split (`TimeoutError` vs `Exception`) and never catches
  `BaseException`/`CancelledError`.
- `print(..., flush=True)` everywhere; events emitted per-stage, not just at end.
- `usage` / `prompt_tokens_details.cached_tokens` are null-guarded.
- Request `ContextVar` is set before `asyncio.create_task(...)`.
- Status route's backend probes carry `x-request-id` too.
- Proxy adds a bounded fetch timeout to distinguish a frozen container from a
  refused connection.

## Diagnose a failure (examples)

- **Judge rate-limited:** `event=error stage=rerank failing_dependency=openai
  http_status=429` while retrieval still returned candidates ⇒ degrade to
  CLIP-only ranking; fix is backoff/quota, not retrieval.
- **Parser silently regex:** `event=parse.fallback reason=timeout` trending up ⇒
  OpenAI latency above the 6 s budget.
- **Dead index:** `/healthz` → 503 + `event=error stage=startup
  failing_dependency=npz` ⇒ the image bake dropped the `.npz`.

## Deferred (Phase 2/3, intentionally not shipped yet)

OpenTelemetry spans + OTLP export (gated on `OTEL_EXPORTER_OTLP_ENDPOINT`),
FastAPI auto-instrumentation in `modal_app.py`, and an optional Upstash daily
cost counter to back a future spend cap. These add dependencies and need
version-pinning against the Modal image, so they are deliberately out of Phase 1.
