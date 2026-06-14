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
  "visual_concept": null | "<verbatim phrase from the query that describes what's on screen>",
  "clean_query": "<query with filter language AND visual_concept removed, for semantic retrieval>",
  "reasoning": "<one short sentence explaining what you extracted>"
}}

Speaker rules:
  - "Sharran" -> speaker="sharran". "Alex" or "Alex Hormozi" or "Hormozi" -> "alex". "Leila" -> "leila".
  - Set `speaker` whenever EXACTLY ONE of {{alex, leila, sharran}} is named. Mention of an unnamed
    second person ("a guest", "another person", "his brother", "someone", "an interviewer") does
    NOT remove the named speaker. Keep them in `speaker`.
  - Use `required_speakers` (and leave speaker null) ONLY when two or more of {{alex, leila, sharran}}
    are named together (e.g. "Alex and Leila").
  - If zero of {{alex, leila, sharran}} are named, leave speaker null.

Speakers count rules (INDEPENDENT of speaker — set both fields when both apply):
  - "solo" / "alone" / "by himself" / "to camera" -> "solo"
  - "with a guest" / "with another person" / "two people" / "dialogue" / "conversation between" /
    "interview" / "interviewing" / "with his brother" / "with someone" -> "dialogue"
  - "panel" / "roundtable" / "group discussion" / "three or more people" -> "group"
  - Example: "Alex with a guest about pricing" -> speaker="alex", speakers_count="dialogue".
  - Example: "Alex interviewing someone" -> speaker="alex", speakers_count="dialogue".

Visual rules (LITERAL keyword only — never infer):
  - is_animation=true ONLY when the editor literally writes "animation", "animated",
    "motion graphic", "title card", "graphic intro", "graphic outro", or "logo animation".
  - talking_head_pose="front_view" ONLY for literal "talking head", "face to camera",
    "facing camera", "head on", "straight to camera", "direct to camera", "front view".
  - talking_head_pose="none" ONLY for literal "B-roll only" or "no face on screen".
  - DO NOT infer either from topic words like "whiteboard", "podcast", "explain", "breakdown",
    "interview", "roundtable", "explanation". A whiteboard is NOT an animation. A podcast
    is NOT a front-view talking head. Leave the field null if the editor didn't explicitly
    name the format.

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

visual_concept — what's on screen, separate from what's being said:
  - If the query describes anything about the VISUAL — what the editor wants to SEE
    in the frame — lift it out verbatim. CLIP visual retrieval handles this; the
    transcript-only judge should NOT re-judge it.
  - Use the editor's exact phrasing (do not normalize "dry erase board" to "whiteboard").
  - This is intentionally OPEN-ENDED. Visual concepts include but are not limited to:
      surfaces / drawing: whiteboard, dry erase board, chalkboard, blackboard, sketchpad
      graphics / titles: title card, lower third, animated intro, motion graphic, infographic
      camera: B-roll, wide shot, close-up, drone shot, screen recording, screen share
      shot framing: two-shot, over-the-shoulder, profile shot, podcast set, studio set
      on-screen visuals: kinetic typography, slide deck, presentation slide
      people on screen: talking head, face to camera, hand-drawn animation
      anything else the editor describes about what's IN the frame (not what's being said)
  - If NOTHING in the query is about the visual, leave it null.
  - Topic words like "scaling", "real estate", "leadership", "pricing" are NEVER visual.
  - Speaker names are NEVER visual.
  - Time phrases are NEVER visual.

  Examples — note that visual_concept is INDEPENDENT of speaker/time/topic.
  Setting visual_concept does NOT cancel out speaker or other fields. Always
  extract every applicable field. Format below: (speaker | visual_concept | clean_query):

    "dry erase board"                                          -> (null   | "dry erase board"     | "")
    "Alex on a whiteboard about scaling"                       -> (alex   | "whiteboard"          | "scaling")
    "title cards from this year"                               -> (null   | "title cards"         | "")
    "Sharran chalkboard pricing breakdown"                     -> (sharran| "chalkboard"          | "pricing")
    "B-roll of city skylines"                                  -> (null   | "B-roll"              | "city skylines")
    "wide shot of Leila explaining hiring"                     -> (leila  | "wide shot"           | "hiring")
    "two-shot interview about scaling SaaS"                    -> (null   | "two-shot interview"  | "scaling SaaS")
    "podcast set Alex and Sharran on retention"                -> (null,  | "podcast set"         | "retention")  [also required_speakers=[alex,sharran]]
    "animated intro Leila leadership last 30 days"             -> (leila  | "animated intro"      | "leadership")  [also max_age_days=30]
    "screen-share Sharran showing a spreadsheet"               -> (sharran| "screen-share"        | "spreadsheet")
    "Leila on hiring decisions"                                -> (leila  | null                  | "hiring decisions")
    "Sharran less than 3 weeks ago talking about real estate"  -> (sharran| null                  | "real estate")  [also max_age_days=21]

clean_query — the TOPIC the transcript-only judge will grade against:
  - Strip the speaker name, time phrases, AND the visual_concept. KEEP the subject.
  - When the query is purely visual with no topic, leave clean_query empty so
    retrieval falls back to the original query.
  - This is what gets fed into transcript scoring; the topic signal is what makes
    transcript/segment scoring work.

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
        "visual_concept": None,
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

    vc = raw.get("visual_concept")
    if isinstance(vc, str):
        vc = vc.strip()
        # Only accept values that actually appear in the original query (case-insensitive),
        # so the LLM can't fabricate a visual concept that wasn't there.
        if vc and vc.lower() in original_query.lower():
            out["visual_concept"] = vc

    cq = raw.get("clean_query")
    if isinstance(cq, str) and cq.strip():
        out["clean_query"] = cq.strip()

    # Safety net: the LLM sometimes leaves the visual_concept and/or the verified
    # speaker name inside clean_query despite the prompt. Strip them — the judge
    # and the retrieval embedder should never see them.
    if out["clean_query"]:
        strip_phrases: list[str] = []
        if out["visual_concept"]:
            strip_phrases.append(out["visual_concept"])
        # Verified speaker names + the "hormozi" surname (parser keyword for alex)
        verified_speakers: list[str] = []
        if out.get("speaker"):
            verified_speakers.append(out["speaker"])
        if out.get("required_speakers"):
            verified_speakers.extend(out["required_speakers"])
        for name in verified_speakers:
            strip_phrases.append(name)
            if name == "alex":
                strip_phrases.append("hormozi")
        if strip_phrases:
            stripped = out["clean_query"]
            # Longest first so multi-word concepts get cleared before their tokens.
            for phrase in sorted(strip_phrases, key=len, reverse=True):
                pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
                stripped = pattern.sub("", stripped)
            stripped = re.sub(r"\s+", " ", stripped).strip(" ,.;:-")
            out["clean_query"] = stripped

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
