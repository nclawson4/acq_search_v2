"""FastAPI service: POST /search { query, speaker?, format?, k } -> ranked moments.

Loads CLIP vectors + transcripts + scenes + speakers + scene tags at startup.
Embeds the query text with open_clip and runs hybrid_eval.py's scoring via the
shared ingest/lib/hybrid.py.

Endpoints
  GET  /healthz                       -> {"ok": true}
  POST /search                        -> ranked moments
  GET  /search?q=...&speaker=...&k=...  same, GET-form for shareable links

This module is the single source of truth for production retrieval. The Next.js
web app and the MCP route both call this service.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hybrid import (  # noqa: E402
    embed_query_text,
    load_index,
    score_query,
    top_k,
    youtube_link,
)
from lib.reranker import rerank_async  # noqa: E402
from lib.structural import compute_dims_satisfied, compose_why  # noqa: E402
from lib.query_parser import parse_async as llm_parse_async  # noqa: E402

# Load .env if present (for OPENAI_API_KEY used by the reranker)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT.parent / ".env")
except Exception:
    pass

# --- Query parsing (Stage 1 of the 3-stage pipeline) -------------------------

SPEAKER_KEYWORDS = {
    "alex": ["alex", "hormozi"],
    "leila": ["leila"],
    "sharran": ["sharran"],
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

# Patterns where the editor wants content OLDER than some duration.
# e.g. "over 1 month ago", "older than 2 weeks", "from more than 3 months ago"
OLDER_THAN_RE = re.compile(
    r"\b(?:over|more\s+than|older\s+than|from\s+more\s+than)\s+(\d+)\s*(day|week|month|year)s?\b",
    re.IGNORECASE,
)

# Patterns where the editor wants RECENT content.
# e.g. "in the last 30 days", "from the last 2 weeks", "within 1 month"
RECENT_RE = re.compile(
    r"\b(?:in|from|within)\s+(?:the\s+)?(?:last\s+|past\s+)?(\d+)\s*(day|week|month|year)s?\b",
    re.IGNORECASE,
)

# Fuzzy phrases mapping to max_age_days
FUZZY_RECENT = [
    (re.compile(r"\btoday\b", re.IGNORECASE), 1),
    (re.compile(r"\byesterday\b", re.IGNORECASE), 2),
    (re.compile(r"\b(?:this|past|last|the\s+last|the\s+past)\s+week\b", re.IGNORECASE), 7),
    (re.compile(r"\b(?:this|past|last|the\s+last|the\s+past)\s+month\b|\brecent(?:ly)?\b", re.IGNORECASE), 30),
    (re.compile(r"\b(?:this|past|last|the\s+last|the\s+past)\s+year\b", re.IGNORECASE), 365),
]
# Fuzzy phrases mapping to min_age_days
FUZZY_OLD = [
    (re.compile(r"\bold(?:er)?(?:\s+content|\s+stuff)?\b", re.IGNORECASE), 90),
    (re.compile(r"\bearly\s+(?:on|stuff|content|episodes?)\b", re.IGNORECASE), 90),
    (re.compile(r"\boriginal\b", re.IGNORECASE), 180),
]


def parse_time_filters(q: str) -> dict:
    """Return {max_age_days, min_age_days} extracted from the query, or None for each.

    Precedence within a category: explicit number > fuzzy phrase. If both an
    "older than" and a "last N" phrase appear, both apply (intersection).
    """
    max_age: int | None = None
    min_age: int | None = None

    # Explicit min_age (older than)
    m = OLDER_THAN_RE.search(q)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        min_age = n * UNIT_DAYS[unit]

    # Explicit max_age (recent N)
    m = RECENT_RE.search(q)
    if m:
        # Skip if this also matched OLDER_THAN_RE (to avoid double-parsing "from more than 30 days")
        # Check by comparing span: if the recent match is INSIDE an older-than span we ignore it.
        ot = OLDER_THAN_RE.search(q)
        if not ot or not (ot.start() <= m.start() and m.end() <= ot.end()):
            n = int(m.group(1))
            unit = m.group(2).lower()
            max_age = n * UNIT_DAYS[unit]

    if min_age is None:
        for pat, days in FUZZY_OLD:
            if pat.search(q):
                min_age = days
                break
    if max_age is None:
        for pat, days in FUZZY_RECENT:
            if pat.search(q):
                max_age = days
                break
    return {"max_age_days": max_age, "min_age_days": min_age}


ANIMATION_KEYWORDS = [
    "animation", "animated", "motion graphic", "intro graphic", "outro graphic",
    "title card", "title cards", "graphic intro", "graphic outro", "logo animation",
]

POSE_FRONT_KEYWORDS = [
    "talking head", "talking-head", "face to camera", "facing camera", "head on", "straight to camera",
]
# (side_view dropped — pose is binary now: front_view or none)


def parse_query(q: str) -> dict:
    """Extract speaker / visual / time hints from the natural-language query.

    Returns:
      speaker:           single named speaker (only when exactly one is mentioned)
      required_speakers: list of all named speakers (only when 2+ are mentioned)
      is_animation:      True if the query asks for animation/title-card-style content
      talking_head_pose: "front_view" / "side_view" / None
      max_age_days / min_age_days: time filters
    """
    lower = q.lower()
    found_speakers: list[str] = []
    for name, kws in SPEAKER_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", lower) for k in kws):
            found_speakers.append(name)

    is_animation: Optional[bool] = None
    if any(kw in lower for kw in ANIMATION_KEYWORDS):
        is_animation = True

    pose: Optional[str] = None
    if any(kw in lower for kw in POSE_FRONT_KEYWORDS):
        pose = "front_view"

    time = parse_time_filters(q)
    base = _empty_parse()
    base.update({
        "speaker": None if len(found_speakers) >= 2 else (found_speakers[0] if found_speakers else None),
        "required_speakers": found_speakers if len(found_speakers) >= 2 else None,
        "is_animation": is_animation,
        "talking_head_pose": pose,
        "max_age_days": time["max_age_days"],
        "min_age_days": time["min_age_days"],
        "clean_query": q,
    })
    return base


def _empty_parse() -> dict:
    return {
        "speaker": None, "required_speakers": None,
        "speakers_count": None,
        "is_animation": None, "talking_head_pose": None,
        "max_age_days": None, "min_age_days": None,
        "industries": None, "audience": None, "age_group": None,
        "lessons_categories": None, "instructional_only": False,
        "clean_query": "", "reasoning": "",
    }


# --- FastAPI app -------------------------------------------------------------

app = FastAPI(title="acq-search-v2 retrieval", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # web + MCP both call this; tighten later if needed
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

STATE = None  # populated at startup


@app.on_event("startup")
def _startup() -> None:
    global STATE
    STATE = load_index(utt_pad_s=0.0)


class SearchRequest(BaseModel):
    query: str = Field(..., description="natural-language query")
    speaker: Optional[Literal["alex", "leila", "sharran"]] = None
    required_speakers: Optional[list[Literal["alex", "leila", "sharran"]]] = Field(
        default=None,
        description="All listed speakers must co-occur in the scene (uses voices_present from audio_tags)",
    )
    speakers_count: Optional[Literal["solo", "dialogue", "group"]] = None
    # Visual classifier filters (replaces format/shot_type/who/has_text)
    is_animation: Optional[bool] = None
    talking_head_pose: Optional[Literal["front_view", "none"]] = None
    # Segment-level filters
    industries: Optional[list[str]] = None
    audience: Optional[list[str]] = None
    age_group: Optional[Literal["early_career_18_30", "mid_career_25_45", "established_35_60", "general_all"]] = None
    lessons_categories: Optional[list[str]] = None
    instructional_only: bool = False
    min_duration_s: Optional[float] = None
    max_duration_s: Optional[float] = None
    max_age_days: Optional[int] = None
    min_age_days: Optional[int] = None
    # Scoring weights (segment text introduced as third channel)
    k: int = 20
    w_visual: float = 0.5
    w_transcript: float = 0.2
    w_segment: float = 0.3
    auto_parse: bool = True
    llm_parse: bool = True   # use gpt-4o-mini for query parsing; falls back to regex if unavailable
    rerank: bool = True


class SegmentInfo(BaseModel):
    video_id: str
    segment_idx: int
    start_s: float
    end_s: float
    topic_title: Optional[str] = None
    summary: Optional[str] = None
    lessons_categories: list = Field(default_factory=list)
    industries: list = Field(default_factory=list)
    audience: list = Field(default_factory=list)


class SearchResult(BaseModel):
    rank: int
    score: float                              # Stage 1 hybrid similarity (CLIP + transcript + segment-text)
    judge_score: Optional[float] = None       # Stage 3 topical relevance (LLM judge)
    why: str = ""                             # composed display string (structural + judge)
    judge_reason: str = ""                    # raw judge content sentence
    structural_dims: list = Field(default_factory=list)  # verified dims for this query
    video_id: str
    scene_idx: int
    start_s: float
    end_s: float
    frame_url: str
    youtube_url: str
    voice: Optional[str] = None
    voices_present: list = Field(default_factory=list)
    speakers_count: Optional[str] = None
    segment: Optional[SegmentInfo] = None
    segment_youtube_url: Optional[str] = None
    recency_days: Optional[int] = None
    upload_date: Optional[str] = None
    signals: dict


class SearchResponse(BaseModel):
    query: str
    parsed: dict
    parsed_reasoning: str = ""
    parsed_by: str = "regex"   # "llm" | "regex"
    filters_applied: dict
    n: int
    results: list[SearchResult]


async def _build_response_async(req: SearchRequest) -> SearchResponse:
    assert STATE is not None

    # Parse the query first. With auto_parse + llm_parse, try the LLM (async, in
    # parallel with CLIP embed below). On any failure fall back to the regex.
    parsed: dict
    parsed_by = "regex"
    parsed_reasoning = ""

    llm_task = None
    if req.auto_parse and req.llm_parse:
        llm_task = asyncio.create_task(llm_parse_async(req.query))

    # CLIP embed in parallel with the LLM call.
    qv = embed_query_text(STATE, req.query)

    if llm_task is not None:
        llm_parsed = await llm_task
        if llm_parsed is not None:
            parsed = llm_parsed
            parsed_by = "llm"
            parsed_reasoning = llm_parsed.get("reasoning") or ""
        else:
            parsed = parse_query(req.query) if req.auto_parse else _empty_parse()
    elif req.auto_parse:
        parsed = parse_query(req.query)
    else:
        parsed = _empty_parse()

    # Caller-provided filters take precedence over parsed
    speaker = req.speaker or parsed.get("speaker")
    required_speakers = req.required_speakers or parsed.get("required_speakers")
    if required_speakers:
        speaker = None
    speakers_count = req.speakers_count or parsed.get("speakers_count")
    is_animation = req.is_animation if req.is_animation is not None else parsed.get("is_animation")
    talking_head_pose = req.talking_head_pose or parsed.get("talking_head_pose")
    industries = req.industries or parsed.get("industries")
    audience = req.audience or parsed.get("audience")
    age_group = req.age_group or parsed.get("age_group")
    lessons_categories = req.lessons_categories or parsed.get("lessons_categories")
    instructional_only = req.instructional_only or bool(parsed.get("instructional_only"))
    max_age_days = req.max_age_days if req.max_age_days is not None else parsed.get("max_age_days")
    min_age_days = req.min_age_days if req.min_age_days is not None else parsed.get("min_age_days")
    visual_concept = parsed.get("visual_concept")

    # Retrieval query: prefer LLM clean_query, fall back to original. For pure-visual
    # queries (visual concept but no remaining topic), use the visual_concept as the
    # CLIP query so retrieval is driven by what the editor wants to SEE.
    clean_q = (parsed.get("clean_query") or "").strip()
    if clean_q:
        retrieval_query = clean_q
    elif visual_concept:
        retrieval_query = visual_concept
    else:
        retrieval_query = req.query

    # When deciding "visual-only" below, leftover time/filter words don't count as a
    # content ask. A query like "title cards from this year" after stripping the
    # visual concept may still leave "from this year" in clean_q — that's a time
    # filter (already captured separately), not a topic for the judge to grade.
    if clean_q:
        residual_topic = re.sub(
            r"\b(in|from|within|over|under|more|less|older|newer|than|recently|"
            r"recent|this|last|past|the|of|a|an|ago|to|for|about|years?|months?|"
            r"weeks?|days?|today|yesterday|years?)\b|\d+",
            "", clean_q, flags=re.IGNORECASE,
        )
        residual_topic = re.sub(r"\s+", " ", residual_topic).strip(" ,.;:-")
    else:
        residual_topic = ""
    if parsed_by == "llm" and retrieval_query and retrieval_query != req.query:
        qv = embed_query_text(STATE, retrieval_query)

    scores = score_query(STATE, qv, query_text=retrieval_query,
                         w_visual=req.w_visual, w_transcript=req.w_transcript,
                         w_segment=req.w_segment)
    hits = top_k(
        STATE, scores, k=req.k, speaker=speaker,
        required_speakers=required_speakers, speakers_count=speakers_count,
        is_animation=is_animation, talking_head_pose=talking_head_pose,
        industries=industries, audience=audience, age_group=age_group,
        lessons_categories=lessons_categories,
        instructional_only=instructional_only,
        min_duration_s=req.min_duration_s, max_duration_s=req.max_duration_s,
        max_age_days=max_age_days, min_age_days=min_age_days,
    )

    structural_dims = compute_dims_satisfied(
        speaker=speaker,
        required_speakers=required_speakers,
        is_animation=is_animation,
        talking_head_pose=talking_head_pose,
        speakers_count=speakers_count,
        max_age_days=max_age_days,
        min_age_days=min_age_days,
        visual_concept=visual_concept,
    )

    # Visual-only short-circuit: when the editor's query is purely about what's on
    # screen (visual_concept present, no remaining topic for the judge to grade),
    # CLIP retrieval IS the answer. The transcript-only judge has nothing useful to
    # add and tends to score 0.0 because the transcript never literally mentions
    # the visual concept. Skip the judge, trust the ranking we already have.
    visual_only = bool(visual_concept) and not residual_topic
    do_rerank = req.rerank and hits and not visual_only

    if do_rerank:
        hits = await rerank_async(
            STATE, req.query, hits,
            structural_satisfied=structural_dims,
        )
        # Re-sort by judge score (descending); break ties with similarity
        hits.sort(key=lambda h: (
            -(h.get("judge_score") if h.get("judge_score") is not None else -1.0),
            -h["score"],
        ))
        # Re-rank field reflects the new order
        for i, h in enumerate(hits, 1):
            h["rank"] = i
    elif visual_only and hits:
        # Judge was skipped because there is no content ask. CLIP did the work;
        # mark each result as judge-verified at 1.0 so the UI doesn't hide them.
        for h in hits:
            h["judge_score"] = 1.0
            h["why"] = f'Matched on "{visual_concept}" visually.'

    results: list[SearchResult] = []
    for h in hits:
        vid = h["video_id"]
        si = h["scene_idx"]
        start_s, end_s = STATE.scenes.get(vid, {}).get(si, (0.0, 0.0))
        judge_reason = (h.get("why") or "").strip()
        seg = h.get("segment")
        seg_url = youtube_link(vid, seg["start_s"]) if seg else None
        results.append(SearchResult(
            rank=h["rank"],
            score=h["score"],
            judge_score=h.get("judge_score"),
            judge_reason=judge_reason,
            structural_dims=structural_dims,
            why=compose_why(structural_dims, judge_reason),
            video_id=vid,
            scene_idx=si,
            start_s=float(start_s),
            end_s=float(end_s),
            frame_url=f"/api/frames/{vid}/{si:04d}",
            youtube_url=youtube_link(vid, start_s),
            voice=h.get("voice"),
            voices_present=h.get("voices_present") or [],
            speakers_count=h.get("speakers_count"),
            segment=SegmentInfo(**seg) if seg else None,
            segment_youtube_url=seg_url,
            recency_days=h.get("recency_days"),
            upload_date=h.get("upload_date"),
            signals={
                "visual_weight": req.w_visual,
                "transcript_weight": req.w_transcript,
                "speaker_filter": speaker,
                "is_animation_filter": is_animation,
                "talking_head_pose_filter": talking_head_pose,
                "reranked": req.rerank,
            },
        ))

    return SearchResponse(
        query=req.query,
        parsed=parsed,
        parsed_reasoning=parsed_reasoning,
        parsed_by=parsed_by,
        filters_applied={
            "speaker": speaker, "required_speakers": required_speakers,
            "speakers_count": speakers_count,
            "is_animation": is_animation, "talking_head_pose": talking_head_pose,
            "industries": industries, "audience": audience, "age_group": age_group,
            "lessons_categories": lessons_categories, "instructional_only": instructional_only,
            "max_age_days": max_age_days, "min_age_days": min_age_days,
            "retrieval_query": retrieval_query,
        },
        n=len(results),
        results=results,
    )


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "indexed_frames": int(STATE.vectors.shape[0]) if STATE is not None else 0,
        "topic_segments": len(STATE.segments_by_key) if STATE is not None else 0,
    }


@app.get("/segment/{video_id}/{segment_idx}/scenes")
def segment_scenes(video_id: str, segment_idx: int) -> dict:
    """Return all scenes belonging to a given topic_segment, in time order.
    Each scene is a thumbnail link an editor can click to open YouTube at that moment.
    """
    assert STATE is not None
    seg = STATE.segments_by_key.get((video_id, segment_idx))
    if seg is None:
        return {"error": "segment not found", "scenes": []}
    scenes_for_video = STATE.scenes.get(video_id, {})
    out = []
    for sc_idx, (start_s, end_s) in sorted(scenes_for_video.items()):
        # is this scene mapped to this segment?
        key = f"frames/{video_id}/scene_{sc_idx:04d}.jpg"
        ref = STATE.scene_to_segment.get(key)
        if ref is None or ref.get("segment_idx") != segment_idx:
            continue
        out.append({
            "scene_idx": sc_idx,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "frame_url": f"/api/frames/{video_id}/{sc_idx:04d}",
            "youtube_url": youtube_link(video_id, start_s),
        })
    return {
        "segment": {
            "video_id": video_id,
            "segment_idx": segment_idx,
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "topic_title": seg.get("topic_title"),
            "summary": seg.get("summary"),
        },
        "scenes": out,
        "n": len(out),
    }


@app.post("/search", response_model=SearchResponse)
async def search_post(req: SearchRequest) -> SearchResponse:
    return await _build_response_async(req)


@app.get("/search", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., description="query text"),
    speaker: Optional[Literal["alex", "leila", "sharran"]] = None,
    required_speakers: Optional[list[str]] = Query(None, description="CSV-friendly list; pass repeats: ?required_speakers=alex&required_speakers=sharran"),
    is_animation: Optional[bool] = None,
    talking_head_pose: Optional[Literal["front_view", "none"]] = None,
    speakers_count: Optional[Literal["solo", "dialogue", "group"]] = None,
    k: int = 20,
    auto_parse: bool = True,
    rerank: bool = True,
) -> SearchResponse:
    rs = None
    if required_speakers:
        rs = [s for s in required_speakers if s in ("alex", "leila", "sharran")] or None
    return await _build_response_async(SearchRequest(
        query=q, speaker=speaker, required_speakers=rs,
        is_animation=is_animation, talking_head_pose=talking_head_pose,
        speakers_count=speakers_count, k=k,
        auto_parse=auto_parse, rerank=rerank,
    ))
