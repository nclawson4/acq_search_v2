"""Judge failure-mode diagnostic.

Run 50 queries through /search and capture exactly what the rerank judge
(gpt-4o-mini) says about each top-K result. Output is engineered for
pattern-spotting, not for ranking metrics.

Specifically classifies each judge_reason into one of:
  - DOES_NOT_MENTION   the snippet talks about something else; judge says so
  - WRONG_FORMAT       judge says the result isn't whiteboard/animation/etc.
  - WRONG_TIME         judge complains about recency / age
  - WRONG_SPEAKER      judge claims the speaker doesn't match
  - LACKS_DETAIL       generic "lacks specificity" / "no clear example"
  - ON_TOPIC           positive — judge confirms the match
  - OTHER              didn't match any pattern

Also tracks:
  - structural_dims that WERE verified (the judge should have ignored these)
  - whether the judge's complaint OVERLAPS with a verified dim (= bug)

Run:
  python -m eval.judge_diagnostic
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
OUT_PATH = HERE / "judge_diagnostic_results.json"
QUERIES_PATH = HERE / "judge_diagnostic_queries.json"

API_URL = "http://localhost:8000/search"

# Classification patterns over judge_reason text (case-insensitive).
PAT_DOES_NOT_MENTION = re.compile(
    r"\b(does not mention|doesn'?t mention|no mention of|not mention(ed)?|"
    r"not (specifically )?(about|related to|address(ing|ed)?)|"
    r"unrelated to|isn'?t (about|related to)|"
    r"content (is|focuses)? ?(on|about) [^.]+,?\s*(but |which is )?(not |un)related|"
    r"focus(es)? on [^.]+,?\s*(not |un)related|"
    r"lacks (a )?(direct |specific )?(reference|mention))",
    re.IGNORECASE,
)
PAT_WRONG_FORMAT = re.compile(
    r"\b(not (specifically )?(a )?(whiteboard|animation|animated|title card|talking head|graphic)|"
    r"no whiteboard|no animation|no visual|no graphic|"
    r"isn'?t (a )?(whiteboard|animation)|"
    r"not (visually|format|the format))",
    re.IGNORECASE,
)
PAT_WRONG_TIME = re.compile(
    r"\b(too (old|recent|new|early|late)|outside (the )?(time|window|period)|"
    r"older than|newer than|not from (the )?(last|past|recent))",
    re.IGNORECASE,
)
PAT_WRONG_SPEAKER = re.compile(
    r"\b(wrong speaker|not (alex|leila|sharran)|speaker (does not|doesn'?t) match|"
    r"different speaker|not (the )?(named )?speaker)",
    re.IGNORECASE,
)
PAT_LACKS_DETAIL = re.compile(
    r"\b(lacks (specifi|detail|context|clarity)|too (vague|generic|brief|short)|"
    r"no (clear|specific) example|insufficient (context|detail)|"
    r"does not provide|doesn'?t provide|no direct (reference|example))",
    re.IGNORECASE,
)
PAT_ON_TOPIC = re.compile(
    r"\b(directly addresses|clearly addresses|on[- ]topic|matches (the )?query|"
    r"relevant to|aligns with|covers (the )?(topic|query))",
    re.IGNORECASE,
)


def classify(reason: str) -> list[str]:
    if not reason:
        return ["EMPTY"]
    tags = []
    if PAT_DOES_NOT_MENTION.search(reason):
        tags.append("DOES_NOT_MENTION")
    if PAT_WRONG_FORMAT.search(reason):
        tags.append("WRONG_FORMAT")
    if PAT_WRONG_TIME.search(reason):
        tags.append("WRONG_TIME")
    if PAT_WRONG_SPEAKER.search(reason):
        tags.append("WRONG_SPEAKER")
    if PAT_LACKS_DETAIL.search(reason):
        tags.append("LACKS_DETAIL")
    if PAT_ON_TOPIC.search(reason):
        tags.append("ON_TOPIC")
    if not tags:
        tags.append("OTHER")
    return tags


def run_query(q: str) -> dict:
    payload = json.dumps({"query": q, "k": 10, "rerank": True}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    print(f"Running {len(queries)} queries against {API_URL}", flush=True)

    rows: list[dict] = []
    for i, qrec in enumerate(queries, 1):
        q = qrec["q"]
        try:
            r = run_query(q)
        except Exception as e:
            print(f"  {i:>2}/50  FAIL  {q!r}  -> {e}", flush=True)
            continue
        top = r.get("results", [])[:5]
        row = {
            "q": q,
            "dims": qrec.get("dims", []),
            "note": qrec.get("note", ""),
            "parsed_by": r.get("parsed_by", "?"),
            "filters_applied": r.get("filters_applied", {}),
            "retrieval_query": (r.get("filters_applied") or {}).get("retrieval_query") or q,
            "structural_dims": (top[0].get("structural_dims") if top else []) or [],
            "results": [
                {
                    "rank": h.get("rank"),
                    "video_id": h.get("video_id"),
                    "scene_idx": h.get("scene_idx"),
                    "voice": h.get("voice"),
                    "speakers_count": h.get("speakers_count"),
                    "judge_score": h.get("judge_score"),
                    "judge_reason": (h.get("judge_reason") or "").strip(),
                    "why": (h.get("why") or "").strip(),
                    "tags": classify((h.get("judge_reason") or "").strip()),
                }
                for h in top
            ],
        }
        rows.append(row)
        n_low = sum(1 for h in top if (h.get("judge_score") or 0) < 0.4)
        n_zero = sum(1 for h in top if (h.get("judge_score") or 0) == 0.0)
        print(f"  {i:>2}/50  {q[:60]:<60}  judge<0.4: {n_low}/{len(top)}  =0.0: {n_zero}/{len(top)}", flush=True)

    OUT_PATH.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}\n")

    # Aggregate
    tag_counts: Counter = Counter()
    overlap_bug = 0   # judge complaint overlapped with a verified structural dim
    n_results_total = 0
    n_results_below_04 = 0
    n_results_zero = 0
    score_buckets = Counter()

    # Tag x dim contingency
    tag_by_dim: dict[str, Counter] = {}

    for row in rows:
        verified = " ".join(row["structural_dims"]).lower()
        for h in row["results"]:
            n_results_total += 1
            js = h["judge_score"] or 0.0
            if js < 0.4: n_results_below_04 += 1
            if js == 0.0: n_results_zero += 1
            score_buckets[round(js * 10) / 10] += 1
            tags = h["tags"]
            for t in tags:
                tag_counts[t] += 1
            # If reason cites a wrong-X that was already verified, that's the bug
            r_text = h["judge_reason"].lower()
            if "WRONG_SPEAKER" in tags and ("speaker=" in verified or "alex" in verified or "leila" in verified or "sharran" in verified):
                overlap_bug += 1
            if "WRONG_FORMAT" in tags and ("animation" in verified or "talking_head" in verified or "whiteboard" in verified):
                overlap_bug += 1
            if "WRONG_TIME" in tags and ("age" in verified or "recent" in verified):
                overlap_bug += 1
            for d in row["dims"]:
                tag_by_dim.setdefault(d, Counter())
                for t in tags:
                    tag_by_dim[d][t] += 1

    print("=" * 70)
    print(f"AGGREGATE OVER {len(rows)} queries, {n_results_total} judged results")
    print(f"  judge < 0.4 (UI-hidden): {n_results_below_04}/{n_results_total}  ({n_results_below_04*100/max(1,n_results_total):.0f}%)")
    print(f"  judge = 0.0:             {n_results_zero}/{n_results_total}  ({n_results_zero*100/max(1,n_results_total):.0f}%)")
    print()
    print("  score buckets:")
    for b in sorted(score_buckets):
        print(f"    {b:.1f}: {score_buckets[b]}")
    print()
    print("  classification tags (one result can be tagged multiple ways):")
    for tag, n in tag_counts.most_common():
        print(f"    {tag:<20} {n}")
    print()
    print(f"  judge-vs-verified OVERLAP BUGS (judge said wrong-X for an already-verified X): {overlap_bug}")
    print()
    print("  per-query-dim tag breakdown (which dims trigger which complaints):")
    for d in sorted(tag_by_dim):
        cnts = tag_by_dim[d]
        top3 = ", ".join(f"{t}={n}" for t,n in cnts.most_common(4))
        print(f"    {d:<8}: {top3}")


if __name__ == "__main__":
    main()
