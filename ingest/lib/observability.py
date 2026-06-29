"""Lightweight, dependency-free observability for the v2 retrieval backend.

Phase 1: structured JSON events written to stdout (Modal/uvicorn capture the
container's stdout) + a per-request correlation id + LLM cost capture. There are
NO third-party dependencies here and NO OpenTelemetry import — OTel/OTLP is a
later, env-gated phase. Everything in this module is *fail-open*: a telemetry
error must never change a search response.

Why stdout + flush=True: this backend is scale-to-zero serverless (Modal). The
only thing guaranteed to outlive a container that freezes/OOMs mid-request is
what has already been written and flushed to stdout, so every event is emitted
the moment its stage completes and is flushed immediately.

Toggle: TELEMETRY_ENABLED (default on; set "0"/"false"/"off" to mute). Every
event carries `request_id` so the Vercel proxy log line and the Modal backend
logs for a single search can be joined.
"""
from __future__ import annotations

import contextvars
import json
import os
import secrets
import time
import traceback as _tb
from datetime import datetime, timezone
from typing import Any, Optional

SERVICE = "acq-search-v2-backend"

# Per-request state, isolated across the (up to 8) concurrent async requests a
# single Modal container may serve. ContextVar is the correct primitive here:
# each FastAPI request runs in its own asyncio Task with a copied context, so
# there is no cross-request leakage on the async path.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("acq_request_id", default="")
_cold_start: contextvars.ContextVar[bool] = contextvars.ContextVar("acq_cold_start", default=False)
_cost_acc: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("acq_cost_acc", default=None)

_CONTAINER_ID = os.getenv("MODAL_TASK_ID") or os.getenv("HOSTNAME") or ""

# gpt-4o-mini list pricing (USD per token). Used only for cost telemetry.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.15e-6, "out": 0.60e-6, "cached_in": 0.075e-6},
}


def telemetry_enabled() -> bool:
    return os.getenv("TELEMETRY_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def new_request_id() -> str:
    """16 random bytes as 32 lowercase hex chars — also a valid W3C trace-id."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def set_request_context(request_id: str, *, cold_start: bool = False) -> None:
    try:
        _request_id.set(request_id or "")
        _cold_start.set(bool(cold_start))
        _cost_acc.set([])
    except Exception:
        pass


def current_request_id() -> str:
    try:
        return _request_id.get()
    except Exception:
        return ""


def _emit(event: str, level: str, fields: dict) -> None:
    """Write one JSON line to stdout, flushed so a dying container can't lose it."""
    if not telemetry_enabled():
        return
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "service": SERVICE,
            "event": event,
            "level": level,
            "request_id": current_request_id(),
            "cold_start": _cold_start.get(),
            "container_id": _CONTAINER_ID,
        }
        if fields:
            rec.update(fields)
        print(json.dumps(rec, default=str), flush=True)
    except Exception:
        # Telemetry must never raise into the request path.
        pass


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    _emit(event, level, fields)


def _usage_tokens(usage: Any) -> tuple[int, int, int]:
    """(input, output, cached_input) tokens, null-guarded across SDK shapes."""
    try:
        inp = int(getattr(usage, "prompt_tokens", 0) or 0)
        out = int(getattr(usage, "completion_tokens", 0) or 0)
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return inp, out, cached
    except Exception:
        return 0, 0, 0


def cost_for(model: str, usage: Any) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    inp, out, cached = _usage_tokens(usage)
    billable_in = max(0, inp - cached)
    return billable_in * p["in"] + cached * p.get("cached_in", p["in"]) + out * p["out"]


def record_llm_cost(
    stage: str,
    model: str,
    usage: Any,
    *,
    latency_ms: Optional[float] = None,
    cache_hit: bool = False,
) -> float:
    """Emit an `llm.cost` event and accumulate the per-request cost. Returns USD."""
    try:
        inp, out, cached = _usage_tokens(usage)
        cost = cost_for(model, usage)
        acc = _cost_acc.get()
        if acc is not None:
            acc.append(cost)
        _emit(
            "llm.cost",
            "info",
            {
                "stage": stage,
                "model": model,
                "input_tokens": inp,
                "output_tokens": out,
                "cached_input_tokens": cached,
                "cost_usd": round(cost, 8),
                "cache_hit": cache_hit,
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
            },
        )
        return cost
    except Exception:
        return 0.0


def request_cost_usd() -> float:
    try:
        acc = _cost_acc.get()
        return round(sum(acc), 8) if acc else 0.0
    except Exception:
        return 0.0


def _http_status_of(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
    try:
        return int(status) if status is not None else None
    except Exception:
        return None


def log_error(
    stage: str,
    exc: BaseException,
    *,
    failing_dependency: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emit a structured `error` event carrying enough context to diagnose without a repro."""
    try:
        payload = {
            "stage": stage,
            "error_class": type(exc).__name__,
            "error_message": str(exc)[:500],
            "error_traceback": "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__)
            )[:4000],
            "failing_dependency": failing_dependency,
            "http_status": _http_status_of(exc),
        }
        if fields:
            payload.update(fields)
        _emit("error", "error", payload)
    except Exception:
        pass
