"""Compare visual retrieval — CLIP vs Gemini Embedding 2 (at 768 and 3072 dims).

For each golden query:
  1. Parse it once with the LLM parser (shared structural plan and retrieval_query)
  2. Embed the retrieval_query three ways: CLIP, Gemini-768, Gemini-3072
  3. For each backend, take top-K by cosine after applying the same structural
     filters (speaker / time)
  4. Send the union of top-5 from all three through the same gpt-4o-mini judge
  5. Emit a side-by-side scorecard

Only the VISUAL embedding source changes between conditions. Hybrid weights,
BM25, segment-text, filters, judge: identical. Apples to apples.

Run:
  python -m eval.gemini_embed.compare --queries golden
  python -m eval.gemini_embed.compare --queries golden --skip-judge   # cheap
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from config import CACHE_DIR  # noqa: E402
from lib.hybrid import load_index, embed_query_text as embed_query_clip  # noqa: E402
from lib.query_parser import parse_async as llm_parse_async  # noqa: E402
from lib.structural import compute_dims_satisfied  # noqa: E402
from lib.reranker import rerank_async  # noqa: E402

GOLDEN_QUERIES = [
    "Leila talking about leadership, talking head video",
    "Sharran less than 3 weeks ago talking about real estate",
    "Animations talking about stress and anxiety",
    "Alex talking about churn",
    "Alex writing on a whiteboard",
    "Leila and Sharran talking together",
]

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = "gemini-embedding-2"


def _load_gemini_vectors(dim: int) -> np.ndarray:
    p = CACHE_DIR / f"gemini_frames_{dim}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p} — run bake_frames.py --dim {dim} first")
    return np.load(p)["vectors"]


def _gemini_text_embed(text: str, dim: int):
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Task-prefix per Embedding 2 convention: queries are "search result"
    contents = f"task: search result | query: {text}"
    r = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=dim),
    )
    v = np.asarray(r.embeddings[0].values, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _retrieve(state, vectors: np.ndarray, qv: np.ndarray, parsed: dict, k: int = 5) -> list[dict]:
    """Top-K by cosine vs `vectors`, with structural filters applied."""
    from lib.hybrid import _frame_passes_filters  # noqa: E402
    sims = vectors @ qv
    order = np.argsort(-sims)
    out = []
    speaker = parsed.get("speaker")
    required_speakers = parsed.get("required_speakers")
    if required_speakers and len(required_speakers) >= 2:
        speaker = None
    for idx in order:
        if not _frame_passes_filters(
            state, int(idx),
            speaker=speaker,
            required_speakers=required_speakers,
            speakers_count=parsed.get("speakers_count"),
            is_animation=parsed.get("is_animation"),
            talking_head_pose=parsed.get("talking_head_pose"),
            industries=parsed.get("industries"),
            audience=parsed.get("audience"),
            age_group=parsed.get("age_group"),
            lessons_categories=parsed.get("lessons_categories"),
            instructional_only=bool(parsed.get("instructional_only")),
            max_age_days=parsed.get("max_age_days"),
            min_age_days=parsed.get("min_age_days"),
        ):
            continue
        m = state.meta[int(idx)]
        a = state.audio_tags.get(m["frame_path"][len("ingest/"):] if m["frame_path"].startswith("ingest/") else m["frame_path"]) or {}
        out.append({
            "frame_idx": int(idx),
            "video_id": m["video_id"],
            "scene_idx": int(m["scene_idx"]),
            "score": float(sims[int(idx)]),
            "voice": a.get("voice"),
            "voices_present": a.get("voices_present"),
            "speakers_count": a.get("speakers_count"),
            "recency_days": (state.video_recency.get(m["video_id"]) or {}).get("recency_days"),
        })
        if len(out) >= k:
            break
    return out


async def _judge(state, query: str, hits: list[dict], structural_dims: list[str]) -> dict[tuple, dict]:
    """Run the existing reranker over the union of hits, return idx -> {score, why}."""
    candidates = [{"video_id": h["video_id"], "scene_idx": h["scene_idx"], "score": h["score"]} for h in hits]
    judged = await rerank_async(state, query, candidates, structural_satisfied=structural_dims)
    out: dict[tuple, dict] = {}
    for j in judged:
        out[(j["video_id"], j["scene_idx"])] = {
            "judge_score": j.get("judge_score"),
            "judge_reason": j.get("why") or "",
        }
    return out


async def main_async(skip_judge: bool):
    print("loading hybrid state (one time)...", flush=True)
    state = load_index()

    print("loading Gemini vectors...", flush=True)
    g768 = _load_gemini_vectors(768)
    g3072 = _load_gemini_vectors(3072)
    clip_vecs = state.vectors

    rows = []
    for q in GOLDEN_QUERIES:
        print(f"\n=== {q} ===", flush=True)
        parsed = await llm_parse_async(q) or {}
        retr = parsed.get("retrieval_query") or parsed.get("clean_query") or q
        print(f"  retrieval_query: {retr!r}")

        # CLIP
        t0 = time.time()
        qv_clip = embed_query_clip(state, retr)
        clip_hits = _retrieve(state, clip_vecs, qv_clip, parsed, k=5)
        clip_lat = (time.time() - t0) * 1000

        # Gemini 768
        t0 = time.time()
        qv_g768 = _gemini_text_embed(retr, 768)
        g768_hits = _retrieve(state, g768, qv_g768, parsed, k=5)
        g768_lat = (time.time() - t0) * 1000

        # Gemini 3072
        t0 = time.time()
        qv_g3072 = _gemini_text_embed(retr, 3072)
        g3072_hits = _retrieve(state, g3072, qv_g3072, parsed, k=5)
        g3072_lat = (time.time() - t0) * 1000

        # Judge — union, single rerank call for cost efficiency
        judgements = {}
        if not skip_judge:
            seen = set()
            union = []
            for h in clip_hits + g768_hits + g3072_hits:
                key = (h["video_id"], h["scene_idx"])
                if key in seen: continue
                seen.add(key)
                union.append(h)
            sd = compute_dims_satisfied(
                speaker=parsed.get("speaker"),
                required_speakers=parsed.get("required_speakers"),
                is_animation=parsed.get("is_animation"),
                talking_head_pose=parsed.get("talking_head_pose"),
                speakers_count=parsed.get("speakers_count"),
                max_age_days=parsed.get("max_age_days"),
                min_age_days=parsed.get("min_age_days"),
                visual_concept=parsed.get("visual_concept"),
            )
            judgements = await _judge(state, parsed.get("judge_query") or q, union, sd)

        def annotate(hits):
            for h in hits:
                j = judgements.get((h["video_id"], h["scene_idx"]), {})
                h["judge_score"] = j.get("judge_score")
                h["judge_reason"] = j.get("judge_reason", "")
            return hits

        row = {
            "query": q,
            "parsed": {k: parsed.get(k) for k in ["speaker", "required_speakers", "visual_concept",
                                                  "retrieval_query", "judge_query", "max_age_days",
                                                  "min_age_days", "speakers_count"]},
            "clip":     {"latency_ms": int(clip_lat),  "hits": annotate(clip_hits)},
            "gem_768":  {"latency_ms": int(g768_lat),  "hits": annotate(g768_hits)},
            "gem_3072": {"latency_ms": int(g3072_lat), "hits": annotate(g3072_hits)},
        }
        rows.append(row)
        # Print quick per-row summary
        def fmt(label, hits, lat):
            top = hits[0] if hits else None
            top_str = f"{top['video_id']}/{top['scene_idx']} judge={top.get('judge_score')}" if top else "—"
            return f"  {label:>10}  lat={lat:>5}ms  n={len(hits)}  top={top_str}"
        print(fmt("CLIP", clip_hits, int(clip_lat)))
        print(fmt("Gem-768", g768_hits, int(g768_lat)))
        print(fmt("Gem-3072", g3072_hits, int(g3072_lat)))

    out_path = OUT_DIR / ("golden_comparison_no_judge.json" if skip_judge else "golden_comparison.json")
    out_path.write_text(json.dumps({"queries": rows, "generated_at": int(time.time())}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", choices=["golden"], default="golden")
    ap.add_argument("--skip-judge", action="store_true", help="don't call the gpt-4o-mini judge (cheap dry run)")
    args = ap.parse_args()
    asyncio.run(main_async(args.skip_judge))


if __name__ == "__main__":
    main()
