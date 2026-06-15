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
search plan. The corpus is long-form business content from Alex Hormozi, Leila Hormozi, and \
Sharran Srivatsaa.

You are the ONLY parser in this pipeline. There is no post-processing — your output strings go \
directly into CLIP embedding and the LLM judge. Strip every filter word out of the query strings \
yourself; the downstream pipeline will not "clean up" anything.

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
  "retrieval_query": "<the EXACT string CLIP will embed. Must be ONLY searchable content —
                      no speaker names, no filler words, no time phrases, no leftover
                      prepositions. Can be the topic, the visual_concept, or both joined.
                      If there's nothing searchable, use the visual_concept verbatim.>",
  "judge_query": null | "<the EXACT topic the transcript-only judge will grade against.
                          null when the query has no content ask (purely visual, purely
                          speaker, purely time, etc.). Should NOT include the visual_concept
                          (visual is already verified by CLIP retrieval).>",
  "is_visual_only": <true if the query is ONLY about what's on screen with no topic ask;
                     when true, the judge will be skipped entirely and CLIP ranking trusted.
                     false otherwise.>,
  "reasoning": "<one short sentence explaining your decisions>"
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

retrieval_query and judge_query — the two strings the downstream pipeline uses verbatim.

You decide exactly what CLIP embeds (retrieval_query) and what the judge grades
against (judge_query). The pipeline does not post-process — what you write is what
gets used. So make them clean.

Rules:
  - Strip out every speaker name (alex, leila, sharran, hormozi), every time phrase
    ("last week", "from 2 months ago", "recently", any digit-units pattern), and every
    structural filler ("at a", "on a", "in front of", "behind", "next to", "with a",
    "showing", "talking", "discussing", "explaining", "about", "of").
  - retrieval_query is what CLIP sees. It should be JUST the searchable concept —
    either the topic (for non-visual queries), the visual_concept (for visual-only
    queries), or both joined naturally (for visual+topic queries).
  - judge_query is what the transcript-only judge grades. NEVER include the visual_concept
    in judge_query — the visual is verified by CLIP, not by the transcript. judge_query
    should be the TOPIC only. Set judge_query to null if no topic remains.
  - is_visual_only = true iff judge_query is null AND visual_concept is set. When true,
    the pipeline skips the judge entirely.

Examples — (query | speaker | visual_concept | retrieval_query | judge_query | is_visual_only):

  "Alex talking about pricing"
    -> ("alex" | null              | "pricing"                     | "pricing"          | false)

  "Sharran less than 3 weeks ago talking about real estate"
    -> ("sharran" | null           | "real estate"                 | "real estate"      | false)
       (plus max_age_days=21)

  "dry erase board"
    -> (null | "dry erase board"   | "dry erase board"             | null               | true)

  "Alex on a whiteboard about scaling"
    -> ("alex" | "whiteboard"      | "scaling on a whiteboard"     | "scaling"          | false)

  "alex at a dry erase board"
    -> ("alex" | "dry erase board" | "dry erase board"             | null               | true)
       (no topic — the editor wants alex + the dry erase board visual)

  "Leila in front of a whiteboard"
    -> ("leila" | "whiteboard"     | "whiteboard"                  | null               | true)

  "title cards from this year"
    -> (null | "title cards"       | "title cards"                 | null               | true)
       (plus max_age_days=365)

  "B-roll of city skylines"
    -> (null | "B-roll"            | "city skylines B-roll"        | "city skylines"    | false)
       (city skylines is a content/visual ask; not just B-roll)

  "Leila on hiring decisions"
    -> ("leila" | null             | "hiring decisions"            | "hiring decisions" | false)

  "alex hyped up about offers"
    -> ("alex" | null              | "hyped up offers"             | "hyped up offers"  | false)

  "podcast interview between Alex and Sharran about acquisition"
    -> (null,req=[alex,sharran] | "podcast interview"
                                   | "acquisition podcast interview" | "acquisition"    | false)

  "any alex video where he draws on something"
    -> ("alex" | "drawing on a board" | "drawing on a board"       | null               | true)
       (loose phrasing — visual_concept can be a sensible normalization of the editor's
        intent even if not literally in the query)

KEY POINT: never include the speaker name in retrieval_query or judge_query. Speaker is
already a structural filter — including the name would just embed noise into CLIP.

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
        "retrieval_query": query,
        "judge_query": query,
        "is_visual_only": False,
        # Backward-compat alias — old API code reads .clean_query as the retrieval query.
        # Kept so any caller that hasn't updated still works.
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
        # Visual concept must be reasonable — either literally in the query, or a short
        # normalization (e.g. editor types "draws on something", LLM normalizes to
        # "drawing on a board"). Cap at a sane length to prevent runaway hallucination.
        if vc and len(vc) <= 60:
            out["visual_concept"] = vc

    # retrieval_query and judge_query are taken verbatim from the LLM. We trust the
    # model — no regex post-processing. If the model gave us junk, the eval will
    # catch it and we iterate on the prompt, not on filler-word lists.
    rq = raw.get("retrieval_query")
    if isinstance(rq, str) and rq.strip():
        out["retrieval_query"] = rq.strip()
        # Mirror to clean_query for any caller that still reads the old field name.
        out["clean_query"] = rq.strip()

    jq = raw.get("judge_query")
    if isinstance(jq, str) and jq.strip():
        out["judge_query"] = jq.strip()
    elif jq is None:
        out["judge_query"] = None

    if raw.get("is_visual_only") is True:
        out["is_visual_only"] = True

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
