"""LLM-as-judge reranker. Modeled on v1's judge.ts.

For each top-K candidate from hybrid retrieval, we hand GPT-4o-mini:
  - the editor's natural-language query
  - the scene's transcript snippet (utterances overlapping the scene window)
  - the scene's video metadata (voice, speakers_count, format, etc.)

The model returns, for each candidate:
  - score (0..1): how well it actually matches the editor's intent
  - why (<=20 words): one-sentence reason shown in the result card

This is what kills "topic-match but off-target" results.

Cost: ~$0.001 per query (gpt-4o-mini, single call with all top-K in one prompt).

Async-friendly, no per-candidate API calls.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "gpt-4o-mini"
MAX_SNIPPET_CHARS = 900          # baseline budget; grows to fit scene-internal content
MAX_SNIPPET_HARD_CAP = 2500      # absolute ceiling regardless of scene length
PER_UTTERANCE_CHAR_CAP = 250     # any one utterance is clipped to this; prevents one rambler eating the budget


SYSTEM_PROMPT = """You are scoring video search results for a media editor. They typed a query, and \
the system already verified the structural attributes of each candidate (who's talking, what format, \
co-presence of speakers). Those verified dimensions are in `structural_satisfied` and the \
`verified_visual_facts` block on each candidate — DO NOT re-evaluate either of them.

You CANNOT see the keyframe image. Trust `verified_visual_facts`:
  - is_animation=true means the visual classifier confirmed the frame is animation / a title card / \
motion graphic. If the editor asked for "animation", that's a MATCH — do not say "not an animation".
  - is_animation=false means the visual classifier confirmed it's NOT animation (live footage).
  - talking_head_pose="front_view" means a clear single person on camera; "none" means no clear \
talking head in frame.
  - If a fact is null, the classifier didn't tag it — fall back to the transcript.

Your only job: score 0.0–1.0 on how well the scene's CONTENT (the transcript snippet) addresses what \
remains in the query after the verified dimensions are taken care of.

  - 1.0: scene content directly addresses the editor's content ask
  - 0.7: clearly relevant, secondary angle
  - 0.4: tangentially mentioned
  - 0.0: unrelated

Special cases (score 1.0):
  - The query's intent is fully covered by `structural_satisfied` or `verified_visual_facts`. \
There's no remaining content ask.
  - The query is purely about a visual concept ("title cards", "animated intro", "talking head shot", \
"animations", "B-roll") and verified_visual_facts confirms it. The transcript will discuss whatever \
topic the editor happened to be talking about ON the title card / animation — that is EXPECTED and \
not a reason to penalize.

A `visual: "<phrase>"` entry in structural_satisfied is the editor's literal visual phrasing \
(any visual concept — whiteboard, dry erase board, chalkboard, title card, B-roll, wide shot, \
podcast set, animated intro, two-shot, drone shot, anything else). CLIP visual retrieval has \
already filtered candidates to ones that visually match this phrase. NEVER write a reason like \
"not a whiteboard" / "not a dry erase board" / "not a title card" / "not B-roll" / "isn't a wide \
shot" — the visual has been verified. Score only the CONTENT in the transcript (any non-visual \
part of the query).

Concrete examples — follow these exactly:

  query: "title cards from this year"
  verified_visual_facts: {is_animation: true, talking_head_pose: "none"}
  transcript_snippet: "[alex] our customers love the product..."
  CORRECT: score=1.0, reason="Animated title card; speaker discusses customer love for the product."
  WRONG:   score=0.0, reason="content discusses business, not title cards"

  query: "animations about retention"
  verified_visual_facts: {is_animation: true, talking_head_pose: "none"}
  transcript_snippet: "[alex] retention dropped 30% after we changed onboarding..."
  CORRECT: score=1.0, reason="Animated visual; directly discusses retention drop tied to onboarding change."

  query: "Sharran less than 3 weeks ago talking about real estate"
  structural_satisfied: ["speaker=sharran (audio voice match verified)", "time_window verified"]
  transcript_snippet: "[sharran] when I list a property, the first 48 hours determines the price..."
  CORRECT: score=1.0, reason="Sharran on real estate listing strategy and the 48-hour pricing window."
  WRONG:   score=0.7, reason="discusses real estate, but not specifically within the last three weeks"

  query: "dry erase board explaining offers"
  structural_satisfied: ['visual: "dry erase board" (CLIP visual retrieval matched this)']
  transcript_snippet: "[alex] hook at the beginning ... two o's ... VSL — video sales letter ..."
  CORRECT: score=1.0, reason="Alex breaks down VSL hook structure and walks through pricing tiers."
  WRONG:   score=0.0, reason="Content is about VSLs, not a dry erase board."

  query: "B-roll of city skylines for an intro reel"
  structural_satisfied: ['visual: "B-roll" (CLIP visual retrieval matched this)']
  transcript_snippet: "[alex] we shot this in downtown Vegas after the conference..."
  CORRECT: score=1.0, reason="Vegas downtown footage shot after conference; matches city-skyline B-roll need."
  WRONG:   score=0.0, reason="not B-roll of city skylines"

  query: "two-shot interview about scaling SaaS"
  structural_satisfied: ['visual: "two-shot interview" (CLIP visual retrieval matched this)']
  transcript_snippet: "[alex] ARR went from one to ten million in eighteen months..."
  CORRECT: score=1.0, reason="Alex on scaling ARR 1M to 10M in eighteen months."
  WRONG:   score=0.4, reason="discusses scaling but the visual isn't clearly a two-shot interview"

NEVER write a `reason` that contradicts an already-verified dimension. Do not say "not an animation" \
when verified_visual_facts.is_animation=true. Do not say "not the right speaker" when speaker is in \
structural_satisfied. Do not say "outside the time window" or "not within the last X" when recency \
is in structural_satisfied. If verified_visual_facts says the visual matches, treat that as ground \
truth — score the CONTENT only, never re-judge the visual.

Write one sentence (<= 20 words) describing the scene's content for the editor — what's being said, \
the gist of it. Don't restate the structural dims (the UI will show those). Don't start with "This scene".

Return JSON only."""


def _format_utt(name: str, text: str) -> str:
    """Label + char-cap a single utterance."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) > PER_UTTERANCE_CHAR_CAP:
        text = text[:PER_UTTERANCE_CHAR_CAP].rstrip() + "..."
    return f"[{name}] {text}"


def _build_snippet(state, video_id: str, start_s: float, end_s: float) -> str:
    """Scene-aligned snippet with adaptive budget + per-utterance cap.

    1. Collect SCENE-INTERNAL utterances first (strict scene window). Each is capped
       at PER_UTTERANCE_CHAR_CAP so no single rambler eats the budget.
    2. Compute adaptive budget = clamp(scene_total_chars, MAX_SNIPPET_CHARS, MAX_SNIPPET_HARD_CAP).
    3. If scene-internal content fits, fill leftover budget with neighbor utterances
       outside the scene, alternating before/after the scene window (closest first).
    4. If scene-internal content alone exceeds the hard cap, truncate from the
       outer edges of the scene to fit.
    """
    utts = state.utterances.get(video_id) or []
    speakers_map = state.speakers.get(video_id, {})

    def name_for(u: dict) -> str:
        cluster = u.get("speaker")
        return speakers_map.get(int(cluster), "?") if cluster is not None else "?"

    # 1) Scene-internal utterances, in time order, each clipped at the per-utterance cap.
    scene_lines: list[str] = []
    scene_chars = 0
    before: list[dict] = []
    after: list[dict] = []
    for u in utts:
        if u["end"] < start_s:
            before.append(u)
            continue
        if u["start"] > end_s:
            after.append(u)
            continue
        line = _format_utt(name_for(u), u.get("text") or "")
        if not line:
            continue
        scene_lines.append(line)
        scene_chars += len(line) + 2

    # 2) Adaptive budget — at least the baseline, up to the scene-aligned content size, never above the hard cap.
    budget = max(MAX_SNIPPET_CHARS, min(scene_chars + 200, MAX_SNIPPET_HARD_CAP))

    # If scene-internal alone exceeds the hard cap, trim from the outside in
    # (keep the center of the scene — closest to the keyframe midpoint).
    if scene_chars > MAX_SNIPPET_HARD_CAP and scene_lines:
        # Walk in from both ends until under cap
        left, right = 0, len(scene_lines) - 1
        kept = scene_lines[:]
        running = scene_chars
        while running > MAX_SNIPPET_HARD_CAP and left <= right:
            # Drop the larger of the two outer lines
            if len(kept[left]) >= len(kept[right]):
                running -= len(kept[left]) + 2
                kept[left] = ""
                left += 1
            else:
                running -= len(kept[right]) + 2
                kept[right] = ""
                right -= 1
        scene_lines = [l for l in kept if l]
        scene_chars = sum(len(l) + 2 for l in scene_lines)
        budget = MAX_SNIPPET_HARD_CAP

    # 3) Fill remaining budget with neighbor utterances, alternating before/after, closest first.
    pad_before: list[str] = []   # chronological order (oldest -> newest)
    pad_after: list[str] = []    # chronological order
    remaining = budget - scene_chars
    before_iter = iter(reversed(before))   # closest-before first (we'll prepend later)
    after_iter = iter(after)               # closest-after first
    prefer_after = True
    exhausted_before = exhausted_after = False
    while remaining > 0 and not (exhausted_before and exhausted_after):
        if prefer_after and not exhausted_after:
            u = next(after_iter, None)
            if u is None:
                exhausted_after = True
                continue
            side = "after"
        elif not exhausted_before:
            u = next(before_iter, None)
            if u is None:
                exhausted_before = True
                continue
            side = "before"
        else:
            prefer_after = not prefer_after
            continue
        line = _format_utt(name_for(u), u.get("text") or "")
        if not line:
            prefer_after = not prefer_after
            continue
        cost = len(line) + 2
        if cost > remaining:
            # Stop adding neighbors once one doesn't fit — small efficient signal.
            break
        if side == "after":
            pad_after.append(line)
        else:
            pad_before.insert(0, line)
        remaining -= cost
        prefer_after = not prefer_after

    return "  ".join(pad_before + scene_lines + pad_after)


def _visual_facts(state, video_id: str, scene_idx: int) -> dict:
    """Look up the visual classifier tags for this scene.

    Returned shape (None means the classifier hasn't tagged this scene):
      {"is_animation": True|False|None, "talking_head_pose": "front_view"|"none"|None}
    """
    if not state.scene_tags:
        return {"is_animation": None, "talking_head_pose": None}
    key = f"ingest/frames/{video_id}/scene_{scene_idx:04d}.jpg"
    tag = state.scene_tags.get(key)
    if tag is None:
        # path_prefix normalization: some entries are stored without the ingest/ prefix
        alt = f"frames/{video_id}/scene_{scene_idx:04d}.jpg"
        tag = state.scene_tags.get(alt)
    if tag is None:
        return {"is_animation": None, "talking_head_pose": None}
    return {
        "is_animation": tag.get("is_animation"),
        "talking_head_pose": tag.get("talking_head_pose"),
    }


def build_payload(state, query: str, candidates: list[dict]) -> dict:
    items = []
    for i, c in enumerate(candidates):
        vid = c["video_id"]
        sw = state.scenes.get(vid, {}).get(c["scene_idx"])
        start_s, end_s = sw if sw else (0.0, 0.0)
        snippet = _build_snippet(state, vid, start_s, end_s)
        items.append({
            "idx": i,
            "video_id": vid,
            "scene_idx": c["scene_idx"],
            "voice": c.get("voice"),
            "speakers_count": c.get("speakers_count"),
            "verified_visual_facts": _visual_facts(state, vid, c["scene_idx"]),
            "transcript_snippet": snippet,
            "youtube_seconds": int(round(start_s)),
        })
    return {"query": query, "candidates": items}


def _schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "idx": {"type": "integer"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["idx", "score", "reason"],
                },
            },
        },
        "required": ["scores"],
    }


async def rerank_async(
    state,
    query: str,
    candidates: list[dict],
    structural_satisfied: list[str] | None = None,
) -> list[dict]:
    """Return candidates annotated with judge score + reason. Order unchanged.

    Caller may re-sort by judge.score if desired.
    """
    if not candidates:
        return candidates
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Soft-fail: return candidates with empty why strings rather than break the request.
        for c in candidates:
            c["judge_score"] = None
            c["why"] = ""
        return candidates

    try:
        from openai import AsyncOpenAI
    except ImportError:
        for c in candidates:
            c["judge_score"] = None
            c["why"] = ""
        return candidates

    client = AsyncOpenAI(api_key=api_key)
    payload = build_payload(state, query, candidates)

    user_body = {
        "query": query,
        "structural_satisfied": structural_satisfied or [],
        "candidates": payload["candidates"],
    }

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_body, indent=2)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "judged", "strict": True, "schema": _schema()},
            },
            max_tokens=2000,
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        scored = {int(s["idx"]): s for s in (data.get("scores") or [])
                  if isinstance(s, dict) and "idx" in s}
        for i, c in enumerate(candidates):
            s = scored.get(i)
            if s is None:
                c["judge_score"] = None
                c["why"] = ""
            else:
                c["judge_score"] = max(0.0, min(1.0, float(s.get("score", 0.0))))
                c["why"] = str(s.get("reason", ""))[:200]
        return candidates
    except Exception as e:
        # Soft-fail: keep retrieval results, just no rerank
        for c in candidates:
            c["judge_score"] = None
            c["why"] = f"(rerank unavailable: {type(e).__name__})"
        return candidates


def rerank(state, query: str, candidates: list[dict]) -> list[dict]:
    """Sync wrapper for non-async callers."""
    import asyncio
    return asyncio.run(rerank_async(state, query, candidates))
