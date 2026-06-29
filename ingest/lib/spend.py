"""Daily LLM-spend counter for the $5/day cost cap, backed by Upstash Redis REST.

Split of responsibility:
  - The Vercel proxy ENFORCES the cap (reads this counter and refuses at the edge
    with a 402 before the backend is even woken).
  - This backend module only ACCOUNTS — after each search it increments the
    shared daily counter by the query's actual measured cost (micro-dollars).

Fully optional and fail-open: with no Upstash env configured this is a no-op, and
the proxy-side cap is then disabled too. Uses httpx (already a backend dep) so no
new dependency is introduced. The increment is fire-and-forget so it never adds
latency to a search response.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional


def _creds() -> tuple[Optional[str], Optional[str]]:
    url = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("KV_REST_API_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("KV_REST_API_TOKEN")
    if url and token:
        return url.rstrip("/"), token
    return None, None


def daily_key() -> str:
    # UTC date — must match the proxy's `v2:spend:${YYYY-MM-DD}` key exactly.
    return "v2:spend:" + datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def _rest(url: str, token: str, *parts: str) -> None:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(
                url + "/" + "/".join(parts),
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception:
        # Best-effort accounting: a transient Upstash error just under-counts.
        pass


async def _bump(usd: float) -> None:
    url, token = _creds()
    if not url or not token:
        return
    micro = max(0, round(usd * 1_000_000))
    if micro <= 0:
        return
    key = daily_key()
    await _rest(url, token, "incrby", key, str(micro))
    # 36h TTL so the per-UTC-day key self-expires after the day rolls.
    await _rest(url, token, "expire", key, str(60 * 60 * 36))


def bump_daily_cost(usd: float) -> None:
    """Fire-and-forget increment of today's spend counter (micro-dollars)."""
    if not usd or usd <= 0:
        return
    # Only schedule when a loop is running (always true under uvicorn/Modal).
    # Checking first avoids creating an un-awaited coroutine in a sync context.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop — accounting is best-effort, skip
    loop.create_task(_bump(usd))
