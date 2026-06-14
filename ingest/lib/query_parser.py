"""LLM-based query parser. Replaces the brittle regex in api/main.py.

For each natural-language editor query, gpt-4o-mini extracts every filter
the /search API supports plus a `clean_query` stripped of filter language
and a one-sentence `reasoning` shown in the UI.

Key design points:
  - Async-friendly: parse_async() can run in parallel with the CLIP embed step.
  - On-disk cache keyed by sha256(normalized_query). Repeats are free.
  - Hard fallback: if OPENAI_API_KEY is missing or the call fails, returns
    None so the caller can fall back to the regex parser.
  - Strict JSON via response_format. temperature=0.0. Unknown filters are
    left null rather than guessed.

Cost: ~150 input + ~120 output tokens at gpt-4o-mini ≈ $0.0004/query
uncached. With cache + repeat sessions, expect ~half that in practice.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CACHE_DIR  # noqa: E402

CACHE_PATH = CACHE_DIR / "query_parser_cache.json"
MODEL = "gpt-4o-mini"

# Schema enums — keep aligned with SearchRequest in api/main.py.
SPEAKERS = ["alex", "leila", "sharran"]
SPEAKERS_COUNT = ["solo", "dialogue", "group"]
POSE = ["front_view", "none"]
INDUSTRIES = ["general_business", "saas", "fitness", "agency", "ecommerce",
              "real_estate", "services", "finance", "healthcare", "education", "retail"]
AUDIENCE = ["business_owners", "employees", "leadership", "personal_life_advice", "general_all"]
AGE_GROUP = ["early_career_18_30", "mid_career_25_45", "established_35_60", "general_all"]
LESSONS = ["sales", "hiring", "leadership", "operations", "scaling", "mindset", "pricing",
           "finance", "customer_acquisition", "branding", "content", "partnerships",
           "personal_development", "marketing"]

SYSTEM_PROMPT = f"""You parse natural-language search queries from video editors into a strict JSON \
filter object for a video search API. The corpus is long-form business content from Alex Hormozi, \
Leila Hormozi, and Sharran Srivatsaa.

Extract every filter that is clearly implied. If a filter is NOT clearly asked for, leave it null. \
Do not invent filters. Do not guess.

Output STRICT JSON with this exact shape:

{{
  "speaker": null | "alex" | "leila" | "sharran",
  "required_speakers": null | ["alex", "leila", ...],
  "speakers_count": null | "solo" | "dialogue" | "group",
  "is_animation": null | true | false,
  "talking_head_pose": null | "front_view" | "none",
  "max_age_days": null | <positive integer>,
  "min_age_days": null | <positive integer>,
  "industries": null | [{", ".join(repr(x) for x in INDUSTRIES)}],
  "audience": null | [{", ".join(repr(x) for x in AUDIENCE)}],
  "age_group": null | "early_career_18_30" | "mid_career_25_45" | "established_35_60" | "general_all",
  "lessons_categories": null | [{", ".join(repr(x) for x in LESSONS)}],
  "instructional_only": false | true,
  "clean_query": "<query with filter language removed, for semantic retrieval>",
  "reasoning": "<one short sentence explaining what you extracted>"
}}

Speaker rules:
  - "Sharran" -> speaker="sharran". "Alex" or "Alex Hormozi" or "Hormozi" -> "alex". "Leila" -> "leila".
  - If two or more speakers are named (e.g. "Alex and Leila"), use required_speakers list, leave speaker null.
  - If the editor describes someone unnamed ("his guest", "a customer"), leave speaker null.

Speakers count rules:
  - "solo" / "alone" / "by himself" -> "solo"
  - "two people" / "dialogue" / "conversation between" / "interview" -> "dialogue"
  - "panel" / "roundtable" / "group discussion" / "three or more" -> "group"

Visual rules:
  - "animation" / "animated intro" / "title card" / "graphic" / "motion graphic" -> is_animation=true
  - "talking head" / "face to camera" / "head on" -> talking_head_pose="front_view"
  - "B-roll only" / "no face on screen" -> talking_head_pose="none"

TIME RULES — be very careful with direction. THIS IS THE MOST COMMON SOURCE OF BUGS.
  max_age_days = "results must be NEWER than X days" (recent filter)
  min_age_days = "results must be OLDER than X days" (old filter)

  RECENT (use max_age_days):
    "in the last 30 days" -> max_age_days=30
    "less than one month ago" -> max_age_days=30  (NOT min_age_days!)
    "less than one month old" -> max_age_days=30  (NOT min_age_days!)
    "newer than 2 weeks" -> max_age_days=14
    "within the past month" -> max_age_days=30
    "recently" / "lately" -> max_age_days=30
    "this week" -> max_age_days=7
    "today" -> max_age_days=1
    "this year" -> max_age_days=365

  OLD (use min_age_days):
    "older than 3 months" -> min_age_days=90
    "more than 6 weeks ago" -> min_age_days=42
    "early stuff" / "early content" / "original videos" -> min_age_days=180
    "from over a year ago" -> min_age_days=365

  NUMERIC PARSING:
    Word numerals: one=1, two=2, three=3, four=4, five=5, six=6, ten=10, a couple=2, a few=3
    Units: day=1, week=7, month=30, year=365
    "a month" / "one month" = 30 days. "a couple weeks" = 14 days.

  If the query mentions BOTH a recent and an old constraint, set both fields.
  If no time language is present, leave both null.

Instructional rule:
  - "show me lessons about X" / "teach me X" / "how to X" -> instructional_only=true

clean_query:
  - The query stripped of filter words (speaker name, time phrases, format words) but keeping the SUBJECT.
  - Example: "Sharran talking about real estate less than one month ago" -> "real estate"
  - Example: "Alex on a podcast about hiring" -> "hiring"
  - This is what gets fed into semantic retrieval; keep it short and topical.

reasoning:
  - One brief sentence stating what you pulled out. Example: "Sharran filter + last-30-day window; topic is real estate."
"""


def _norm(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _hash(query: str) -> str:
    return hashlib.sha256(_norm(query).encode("utf-8")).hexdigest()[:24]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(c: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(c), encoding="utf-8")


# In-process cache to avoid disk reads on every call.
_MEM_CACHE: dict[str, dict] = {}


def _empty_parse(query: str) -> dict:
    """Return shape-compatible empty parse for callers that need a fallback."""
    return {
        "speaker": None,
        "required_speakers": None,
        "speakers_count": None,
        "is_animation": None,
        "talking_head_pose": None,
        "max_age_days": None,
        "min_age_days": None,
        "industries": None,
        "audience": None,
        "age_group": None,
        "lessons_categories": None,
        "instructional_only": False,
        "clean_query": query,
        "reasoning": "",
    }


def _validate(raw: dict, original_query: str) -> dict:
    """Coerce LLM output to the expected shape; drop values that aren't in the allowed enums."""
    out = _empty_parse(original_query)
    if not isinstance(raw, dict):
        return out

    sp = raw.get("speaker")
    if isinstance(sp, str) and sp in SPEAKERS:
        out["speaker"] = sp

    rsp = raw.get("required_speakers")
    if isinstance(rsp, list):
        rsp = [s for s in rsp if isinstance(s, str) and s in SPEAKERS]
        if len(rsp) >= 2:
            out["required_speakers"] = rsp
            out["speaker"] = None  # multi takes precedence

    sc = raw.get("speakers_count")
    if isinstance(sc, str) and sc in SPEAKERS_COUNT:
        out["speakers_count"] = sc

    ia = raw.get("is_animation")
    if isinstance(ia, bool):
        out["is_animation"] = ia

    pose = raw.get("talking_head_pose")
    if isinstance(pose, str) and pose in POSE:
        out["talking_head_pose"] = pose

    for k in ("max_age_days", "min_age_days"):
        v = raw.get(k)
        if isinstance(v, int) and 0 < v <= 365 * 10:
            out[k] = v

    for k, allowed in (("industries", INDUSTRIES), ("audience", AUDIENCE), ("lessons_categories", LESSONS)):
        v = raw.get(k)
        if isinstance(v, list):
            v = [x for x in v if isinstance(x, str) and x in allowed]
            if v:
                out[k] = v

    ag = raw.get("age_group")
    if isinstance(ag, str) and ag in AGE_GROUP:
        out["age_group"] = ag

    if raw.get("instructional_only") is True:
        out["instructional_only"] = True

    cq = raw.get("clean_query")
    if isinstance(cq, str) and cq.strip():
        out["clean_query"] = cq.strip()

    r = raw.get("reasoning")
    if isinstance(r, str) and r.strip():
        out["reasoning"] = r.strip()[:200]

    return out


async def parse_async(query: str, *, force: bool = False, timeout_s: float = 6.0) -> Optional[dict]:
    """Async LLM parse. Returns the parsed dict, or None if the API isn't reachable.

    Caller is expected to fall back to the regex parser on None.
    """
    if not query or not query.strip():
        return _empty_parse("")

    h = _hash(query)
    if not force:
        if h in _MEM_CACHE:
            return _MEM_CACHE[h]
        disk = _load_cache()
        if h in disk:
            _MEM_CACHE[h] = disk[h]
            return disk[h]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:
        return None

    client = AsyncOpenAI(api_key=api_key)
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            ),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, Exception):
        return None

    try:
        raw = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return None

    parsed = _validate(raw, query)
    parsed["_parsed_by"] = "llm"
    parsed["_ts"] = int(time.time())

    _MEM_CACHE[h] = parsed
    disk = _load_cache()
    disk[h] = parsed
    _save_cache(disk)
    return parsed


def parse(query: str, **kwargs) -> Optional[dict]:
    """Sync wrapper for callers outside an event loop."""
    return asyncio.run(parse_async(query, **kwargs))
