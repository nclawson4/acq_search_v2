"""A/B eval: current parser vs new end-to-end LLM parser.

Runs both systems on the same 50 queries. For each top-5 result:
  - Speaker correctness: deterministic from voice field
  - Time correctness: deterministic from recency_days
  - Topic/visual relevance: LLM-as-judge with 0/1/2 rubric

Per query: precision@5, MRR. Per system: aggregates + verdicts.

Usage:
  # 1. Run against the currently-running backend, save as a labeled snapshot.
  python -m eval.parser_ab --label baseline
  python -m eval.parser_ab --label new

  # 2. After both run, compare:
  python -m eval.parser_ab --compare baseline new
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
QUERIES_PATH = HERE / "parser_ab_queries.json"
SNAP_DIR = HERE / "parser_ab_snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

API_URL = "http://localhost:8000/search"
JUDGE_CACHE = HERE / "parser_ab_judge_cache.json"
JUDGE_MODEL = "gpt-4o-mini"


def _load_judge_cache() -> dict:
    if JUDGE_CACHE.exists():
        return json.loads(JUDGE_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_judge_cache(c: dict) -> None:
    JUDGE_CACHE.write_text(json.dumps(c), encoding="utf-8")


def run_query(q: str) -> dict:
    payload = json.dumps({"query": q, "k": 5, "rerank": True}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def snapshot(label: str):
    """Run all 50 queries against current backend, save raw responses."""
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    print(f"Snapshotting {len(queries)} queries -> {label}", flush=True)
    out = []
    for i, qrec in enumerate(queries, 1):
        q = qrec["q"]
        try:
            r = run_query(q)
        except Exception as e:
            print(f"  {i:>2}/50  ERR  {q!r}: {e}", flush=True)
            out.append({"q": q, "dims": qrec["dims"], "error": str(e)})
            continue
        n = r.get("n", 0)
        print(f"  {i:>2}/50  n={n:<2}  {q}", flush=True)
        out.append({"q": q, "dims": qrec["dims"], "response": r})
    snap_path = SNAP_DIR / f"{label}.json"
    snap_path.write_text(json.dumps({"label": label, "ts": int(time.time()), "rows": out}, indent=2), encoding="utf-8")
    print(f"\nwrote {snap_path}\n")


# -----------------------------------------------------------------------------
# Judging
# -----------------------------------------------------------------------------
JUDGE_SYSTEM = """You are scoring video search results for a media editor.

For each result, decide whether it matches what the editor asked for. The editor's
query has multiple possible dimensions (speaker, time, visual format, topic). You'll
be told which dimensions matter for this query.

Score each result 0/1/2:
  2 = directly satisfies every applicable dimension; an editor would use this clip
  1 = partial match (right speaker but wrong topic, or right topic but visual is off, etc.)
  0 = doesn't satisfy the asked-for dimensions

Use the transcript snippet, voice/speakers_count fields, and any verified visual facts
to decide. For visual concepts the classifier doesn't directly tag (whiteboard, B-roll,
chalkboard, drone shot, etc.), trust the CLIP retrieval — assume it's visually plausible
unless the transcript strongly contradicts (e.g., the speaker says "I'm standing in my
office" when query asks for a drone shot).

Return STRICT JSON: {"labels": [{"i": <result_index>, "label": <0|1|2>, "why": "<one short clause>"}, ...]}
"""


def _ckey(query: str, video_id: str, scene_idx: int) -> str:
    return hashlib.sha256(f"{query}|{video_id}|{scene_idx}".encode("utf-8")).hexdigest()[:24]


async def judge_results(query: str, dims: list[str], results: list[dict]) -> list[dict]:
    """Return list of {label, why} per result, using on-disk cache."""
    cache = _load_judge_cache()
    out: list[dict | None] = [None] * len(results)
    to_judge: list[tuple[int, dict]] = []
    for i, h in enumerate(results):
        k = _ckey(query, h["video_id"], h["scene_idx"])
        if k in cache:
            out[i] = cache[k]
        else:
            to_judge.append((i, h))

    if to_judge:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import AsyncOpenAI  # type: ignore
        client = AsyncOpenAI(api_key=api_key)

        items = []
        for n, (i, h) in enumerate(to_judge):
            items.append({
                "i": n,
                "video_id": h["video_id"],
                "scene_idx": h["scene_idx"],
                "voice": h.get("voice"),
                "speakers_count": h.get("speakers_count"),
                "recency_days": h.get("recency_days"),
                "judge_reason": (h.get("judge_reason") or "").strip(),
                "scene_why": (h.get("why") or "")[:200],
            })
        user_body = {"query": query, "dims": dims, "results": items}
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(user_body)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            for row in data.get("labels", []):
                n = int(row["i"])
                if 0 <= n < len(to_judge):
                    orig_i, h = to_judge[n]
                    entry = {"label": max(0, min(2, int(row["label"]))), "why": str(row.get("why", ""))[:200]}
                    out[orig_i] = entry
                    cache[_ckey(query, h["video_id"], h["scene_idx"])] = entry
        except Exception:
            pass
        _save_judge_cache(cache)

    return [(x or {"label": 0, "why": "unscored"}) for x in out]


def _speaker_ok(query_dims: list[str], parsed: dict, h: dict) -> tuple[str, bool | None]:
    """Deterministic check: does this result match the speaker filter?"""
    if "who" not in query_dims:
        return ("n/a", None)
    speaker = parsed.get("speaker")
    req = parsed.get("required_speakers")
    if req:
        # all required speakers must be in voices_present (or in voice)
        present = set(h.get("voices_present") or [])
        if h.get("voice"):
            present.add(h["voice"])
        return ("multi", all(s in present for s in req))
    if speaker:
        return (speaker, h.get("voice") == speaker)
    return ("none-extracted", None)


def _time_ok(query_dims: list[str], parsed: dict, h: dict) -> tuple[str, bool | None]:
    if "time" not in query_dims:
        return ("n/a", None)
    rd = h.get("recency_days")
    if rd is None:
        return ("no-recency-data", None)
    max_age = parsed.get("max_age_days")
    min_age = parsed.get("min_age_days")
    if max_age is not None and rd > max_age:
        return (f"max_age={max_age}", False)
    if min_age is not None and rd < min_age:
        return (f"min_age={min_age}", False)
    if max_age is None and min_age is None:
        return ("none-extracted", None)
    return ("ok", True)


async def score_snapshot(label: str) -> dict:
    snap_path = SNAP_DIR / f"{label}.json"
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    rows = snap["rows"]
    scored = []
    for row in rows:
        q = row["q"]
        dims = row["dims"]
        if "error" in row:
            scored.append({"q": q, "dims": dims, "error": row["error"], "results": [], "p_at_5": 0.0, "mrr": 0.0})
            continue
        r = row["response"]
        parsed = r.get("parsed", {})
        results = r.get("results", [])[:5]
        # Judge
        judgments = await judge_results(q, dims, results)
        # Build per-result row
        per_result = []
        for h, j in zip(results, judgments):
            spk_filter, spk_ok = _speaker_ok(dims, parsed, h)
            time_filter, time_ok = _time_ok(dims, parsed, h)
            # Decide "good": label >= 2 AND no deterministic failure on speaker/time
            judge_good = j["label"] >= 2
            det_fail = (spk_ok is False) or (time_ok is False)
            good = judge_good and not det_fail
            per_result.append({
                "video_id": h["video_id"],
                "scene_idx": h["scene_idx"],
                "voice": h.get("voice"),
                "recency_days": h.get("recency_days"),
                "judge_label": j["label"],
                "judge_why": j["why"],
                "speaker_check": {"filter": spk_filter, "ok": spk_ok},
                "time_check": {"filter": time_filter, "ok": time_ok},
                "good": good,
            })
        n = len(per_result)
        n_good = sum(1 for r in per_result if r["good"])
        p_at_5 = n_good / max(1, n)
        # MRR: 1/(rank of first 'good')
        mrr = 0.0
        for i, r in enumerate(per_result, 1):
            if r["good"]:
                mrr = 1.0 / i
                break
        scored.append({
            "q": q, "dims": dims, "n": n, "n_good": n_good,
            "p_at_5": round(p_at_5, 3), "mrr": round(mrr, 3),
            "parsed_summary": {
                k: parsed.get(k) for k in ["speaker", "required_speakers", "speakers_count",
                                          "is_animation", "talking_head_pose", "max_age_days",
                                          "min_age_days", "visual_concept", "clean_query"]
            },
            "retrieval_query": (r.get("filters_applied") or {}).get("retrieval_query"),
            "results": per_result,
        })
    # Aggregate
    agg = _aggregate(scored)
    out = {"label": label, "agg": agg, "scored": scored}
    (SNAP_DIR / f"{label}_scored.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _aggregate(scored: list[dict]) -> dict:
    n_q = len(scored)
    p5 = [s["p_at_5"] for s in scored]
    mrr = [s["mrr"] for s in scored]
    broken = sum(1 for s in scored if s["p_at_5"] <= 0.2)
    great = sum(1 for s in scored if s["p_at_5"] >= 0.8)
    zero_results = sum(1 for s in scored if s.get("n", 0) == 0)
    return {
        "n_queries": n_q,
        "p_at_5_mean": round(sum(p5) / max(1, n_q), 3),
        "mrr_mean": round(sum(mrr) / max(1, n_q), 3),
        "broken_query_count": broken,
        "broken_query_rate": round(broken / max(1, n_q), 3),
        "great_query_count": great,
        "great_query_rate": round(great / max(1, n_q), 3),
        "zero_results_count": zero_results,
    }


def compare(label_a: str, label_b: str):
    a = json.loads((SNAP_DIR / f"{label_a}_scored.json").read_text(encoding="utf-8"))
    b = json.loads((SNAP_DIR / f"{label_b}_scored.json").read_text(encoding="utf-8"))
    print(f"\n{'='*70}\nA = {label_a}   B = {label_b}")
    print(f"{'='*70}")
    print(f"  metric                 {label_a:>12}  {label_b:>12}    delta")
    for k in ["p_at_5_mean", "mrr_mean", "broken_query_rate", "great_query_rate", "zero_results_count"]:
        va = a["agg"][k]; vb = b["agg"][k]
        delta = vb - va
        arrow = "+" if delta > 0 else ("-" if delta < 0 else " ")
        print(f"  {k:<22} {va:>12}  {vb:>12}  {arrow}{abs(delta):.3f}")
    print()
    # Per-query deltas
    by_q_a = {s["q"]: s for s in a["scored"]}
    by_q_b = {s["q"]: s for s in b["scored"]}
    print(f"  per-query precision@5 deltas (B - A):")
    deltas = []
    for q in by_q_a:
        if q in by_q_b:
            d = by_q_b[q]["p_at_5"] - by_q_a[q]["p_at_5"]
            deltas.append((d, q))
    deltas.sort()
    print(f"\n  Biggest regressions ({label_b} worse):")
    for d, q in deltas[:8]:
        print(f"    {d:+.2f}  {q}")
    print(f"\n  Biggest wins ({label_b} better):")
    for d, q in reversed(deltas[-8:]):
        print(f"    {d:+.2f}  {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", metavar="LABEL", help="run 50 queries, save raw responses as snapshots/<LABEL>.json")
    ap.add_argument("--score", metavar="LABEL", help="judge an existing snapshot, save snapshots/<LABEL>_scored.json")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="print A vs B comparison")
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
    elif args.score:
        asyncio.run(score_snapshot(args.score))
    elif args.compare:
        compare(*args.compare)
    else:
        ap.error("specify --snapshot LABEL, --score LABEL, or --compare A B")


if __name__ == "__main__":
    main()
