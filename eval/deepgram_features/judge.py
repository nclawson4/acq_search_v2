"""gpt-4o-mini relevance judge with on-disk cache.

Given (query, top results), return a label in {0, 1, 2} for each result.
Cached by hash(query + frame_path) so re-runs are free across features.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

CACHE_PATH = Path(__file__).parent / "results" / "judge_cache.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"

SYS_PROMPT = """You are grading video search results for an editor.

Label each result on a 0/1/2 scale:
  2 = directly addresses the query (the editor would use this clip)
  1 = clearly relevant, secondary angle (worth showing as a backup)
  0 = unrelated or wrong topic/speaker/format/sentiment

Be strict. The query may have multiple constraints (topic + speaker + format).
A result that nails the topic but is the wrong speaker is still a 1, not a 2.
A result that has no clear connection is 0.

Return STRICT JSON: {"labels": [{"i": <result_index>, "label": <0|1|2>}, ...]}.
"""


def _ckey(query: str, frame_path: str) -> str:
    return hashlib.sha256(f"{query}|{frame_path}".encode("utf-8")).hexdigest()[:24]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(c: dict) -> None:
    CACHE_PATH.write_text(json.dumps(c), encoding="utf-8")


def _build_snippet(state, idx: int, max_chars: int = 700) -> str:
    """Compact transcript snippet for a frame: utterances overlapping the scene window."""
    m = state.meta[idx]
    vid = m["video_id"]
    scene = state.scenes.get(vid, {}).get(int(m["scene_idx"]))
    if scene is None:
        return ""
    ss, se = scene
    utts = state.utterances.get(vid) or []
    rng = state.frame_utt_ranges[idx]
    if rng is None:
        return ""
    lo, hi = rng
    pieces = []
    total = 0
    for u in utts[lo:hi]:
        t = (u.get("transcript") or "").strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            pieces.append(t[: max(0, max_chars - total)])
            break
        pieces.append(t)
        total += len(t)
    return " ".join(pieces)


async def _judge_batch(client, query: str, items: list[dict]) -> list[int]:
    """Send one prompt covering all items; return list of labels aligned to items."""
    parts = [f"QUERY: {query}", "", "RESULTS:"]
    for i, it in enumerate(items):
        parts.append(f"[{i}] video={it['video_id']} scene={it['scene_idx']} voice={it.get('voice') or '-'}")
        parts.append(f"    snippet: {it['snippet'] or '(no transcript)'}")
    user = "\n".join(parts)

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        labels = [0] * len(items)
        for row in data.get("labels", []):
            i = int(row["i"]); lab = int(row["label"])
            if 0 <= i < len(items):
                labels[i] = max(0, min(2, lab))
        return labels
    except Exception:
        return [0] * len(items)


async def judge_results_async(state, query: str, frame_indices: list[int]) -> list[int]:
    """Return [label per frame_idx]. Reads/writes the on-disk cache."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; judge requires it")

    cache = _load_cache()
    labels: list[int | None] = []
    to_judge: list[int] = []
    to_judge_items: list[dict] = []

    for idx in frame_indices:
        m = state.meta[idx]
        key = _ckey(query, m["frame_path"])
        if key in cache:
            labels.append(int(cache[key]))
            continue
        labels.append(None)
        a = state.audio_tags.get(m["frame_path"][len("ingest/"):] if m["frame_path"].startswith("ingest/") else m["frame_path"])
        to_judge_items.append({
            "video_id": m["video_id"],
            "scene_idx": int(m["scene_idx"]),
            "voice": (a or {}).get("voice"),
            "snippet": _build_snippet(state, idx),
        })
        to_judge.append(idx)

    if to_judge:
        from openai import AsyncOpenAI  # type: ignore
        client = AsyncOpenAI(api_key=api_key)
        new_labels = await _judge_batch(client, query, to_judge_items)
        # write through to cache + fill in labels list
        ji = 0
        for pos, idx in enumerate(frame_indices):
            if labels[pos] is None:
                lab = new_labels[ji]; ji += 1
                cache[_ckey(query, state.meta[idx]["frame_path"])] = int(lab)
                labels[pos] = int(lab)
        _save_cache(cache)

    return [int(x) for x in labels]


def judge_results(state, query: str, frame_indices: list[int]) -> list[int]:
    return asyncio.run(judge_results_async(state, query, frame_indices))
