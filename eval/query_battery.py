"""End-to-end editor query battery for the search API.

Stress tests across 5 query families. For each query: hit /search, print top-5 with
judge scores + voice + segment, then a one-line PASS/FAIL judgment based on whether
the top-1 is plausibly what an editor wanted.

Final gate: 95% of queries produce a usable top-1 AND no top-3 has a "distorting"
result (defined as: completely off-topic, wrong speaker, wrong timestamp). Manual
review required for the "distorting" check — the script prints data, human judges.

CLI:
  python eval/query_battery.py
  python eval/query_battery.py --port 8000 --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from textwrap import shorten


# 30 editor-realistic queries spanning 5 families
QUERIES = [
    # === FAMILY 1: Compositional (speaker + format / co-presence). Should hit structural-only path. ===
    {"q": "alex whiteboard", "family": "compositional", "expect": "any Alex-at-whiteboard scene"},
    {"q": "sharran podcast", "family": "compositional", "expect": "any Sharran podcast scene"},
    {"q": "leila solo talking head", "family": "compositional", "expect": "any Leila solo studio scene"},
    {"q": "mozi tank pitch", "family": "compositional", "expect": "Mozi Tank format scene"},
    {"q": "intro animation", "family": "compositional", "expect": "animated intro/title card"},
    {"q": "alex and leila", "family": "compositional", "expect": "scenes with both speakers present"},

    # === FAMILY 2: Topical (no structural). Pure semantic search. ===
    {"q": "how to hire your first sales rep", "family": "topical", "expect": "hiring-focused content"},
    {"q": "the part about churn", "family": "topical", "expect": "churn-related discussion"},
    {"q": "scaling past 10 million in revenue", "family": "topical", "expect": "scaling/growth content"},
    {"q": "pricing strategy for high ticket offers", "family": "topical", "expect": "pricing discussion"},
    {"q": "motivational mindset around failure", "family": "topical", "expect": "mindset/resilience content"},
    {"q": "the Nokia story about leadership", "family": "topical", "expect": "Nokia leadership story"},

    # === FAMILY 3: Mixed (speaker + topic). The main use case. ===
    {"q": "sharran talking about leadership", "family": "mixed", "expect": "Sharran on leadership"},
    {"q": "alex explaining customer acquisition cost", "family": "mixed", "expect": "Alex on CAC"},
    {"q": "leila on hiring framework", "family": "mixed", "expect": "Leila on hiring"},
    {"q": "alex breakdown of the value equation", "family": "mixed", "expect": "Alex on value equation"},
    {"q": "sharran sharing the dave matthews story", "family": "mixed", "expect": "Sharran's Dave Matthews mention"},

    # === FAMILY 4: Visual / specific moments ===
    {"q": "yellow caption about two and a half million", "family": "visual", "expect": "the 2.5M yellow caption"},
    {"q": "concert crowd with fireworks", "family": "visual", "expect": "concert crowd b-roll"},
    {"q": "dark cinematic boxing footage", "family": "visual", "expect": "boxing B-roll"},
    {"q": "Sharran in blue blazer with 100M books", "family": "visual", "expect": "Sharran armchair canonical"},

    # === FAMILY 5: Adversarial / ambiguous ===
    {"q": "the moment alex gets pumped up", "family": "adversarial", "expect": "emotional Alex moment"},
    {"q": "podcast intro music", "family": "adversarial", "expect": "intro segments"},
    {"q": "a great quote about consistency", "family": "adversarial", "expect": "consistency quote"},
    {"q": "the most actionable advice", "family": "adversarial", "expect": "actionable advice, any topic"},
    {"q": "something I can use for a thumbnail", "family": "adversarial", "expect": "visually striking moment"},
    {"q": "sharran talking with leila about partnerships", "family": "adversarial", "expect": "co-presence + partnership topic"},
    {"q": "leila quick clip under 30 seconds", "family": "adversarial", "expect": "short Leila scene"},
    {"q": "recent business owner advice from this month", "family": "adversarial", "expect": "recent advice"},
    {"q": "alex's framework for thinking about churn rate", "family": "adversarial", "expect": "Alex on churn frameworks"},
]


def run(host: str, port: int, k: int, only_family: str | None = None) -> None:
    base = f"http://{host}:{port}/search"
    n_total = 0
    by_family: dict[str, list[dict]] = {}
    for q in QUERIES:
        if only_family and q["family"] != only_family:
            continue
        n_total += 1
        try:
            data = json.dumps({"query": q["q"], "k": k}).encode()
            req = urllib.request.Request(base, data=data, headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
        except Exception as e:
            print(f"\n[ERR] {q['q']!r}: {e}")
            continue

        results = body.get("results", [])
        print(f"\n--- [{q['family']}] {q['q']!r}  (expect: {q['expect']}) ---")
        print(f"  parsed: {body.get('parsed')}")
        for r in results[:k]:
            seg = r.get("segment")
            seg_title = (seg or {}).get("topic_title", "—")
            judge = r.get("judge_score")
            judge_str = f"{judge:.2f}" if isinstance(judge, (int, float)) else "  ? "
            voice = r.get("voice") or "?"
            voices = "+".join(r.get("voices_present") or [])
            line = f"  #{r['rank']} j={judge_str} sim={r['score']:.2f} v={voice}({voices}) {r['video_id']}/scene_{r['scene_idx']:04d} -> {shorten(str(seg_title), 60)}"
            print(line)
            wf = r.get("judge_reason") or r.get("why") or ""
            if wf:
                print(f"        why: {shorten(wf, 90)}")
        by_family.setdefault(q["family"], []).append({"query": q["q"], "n_results": len(results)})

    print()
    print("=" * 70)
    print(f"Ran {n_total} queries across {len(by_family)} families")
    for fam, qs in by_family.items():
        nonzero = sum(1 for x in qs if x["n_results"] > 0)
        print(f"  {fam:14s}: {len(qs)} queries, {nonzero} returned ≥1 result")
    print()
    print("NOTE: This script does NOT auto-judge whether results are 'good' — only that")
    print("they exist. Pass through the printout above and rate each top-1 manually.")
    print()
    print("Manual gate: 95% queries with a usable top-1, 0% top-3 with 'distorting' results.")
    print("(Distorting = completely off-topic / wrong speaker / wrong timestamp)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--family", default=None, help="filter to one query family")
    args = p.parse_args()
    run(args.host, args.port, args.k, args.family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
