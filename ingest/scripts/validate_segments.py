"""Validate cache/topic_segments.json against the schema in docs/TAGGING_SCHEMA.md.

Exits 0 if valid, 2 if invalid. Prints per-record violations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR  # noqa: E402

SEGMENTS_PATH = CACHE_DIR / "topic_segments.json"

LESSONS = {
    "sales", "hiring", "leadership", "marketing", "operations", "pricing",
    "scaling", "mindset", "finance", "customer_acquisition", "branding",
    "content", "partnerships", "personal_development", "none",
}
INDUSTRIES = {
    "general_business", "saas", "fitness", "agency", "ecommerce", "real_estate",
    "services", "finance", "healthcare", "education", "retail", "none",
}
AUDIENCE = {
    "business_owners", "employees", "leadership", "personal_life_advice", "general_all",
}
AGE_GROUPS = {
    "early_career_18_30", "mid_career_25_45", "established_35_60", "general_all",
}

REQUIRED = [
    "video_id", "segment_idx", "start_s", "end_s",
    "topic_title", "summary", "lessons_summary",
    "expected_queries", "lessons_categories", "industries", "audience", "age_group",
]


def validate_one(s: dict, i: int) -> list[str]:
    errs: list[str] = []
    for k in REQUIRED:
        if k not in s:
            errs.append(f"[{i}] missing field '{k}'")
    if errs:
        return errs
    if not isinstance(s["start_s"], (int, float)) or not isinstance(s["end_s"], (int, float)):
        errs.append(f"[{i}] start_s/end_s must be numeric")
    elif s["start_s"] >= s["end_s"]:
        errs.append(f"[{i}] start_s ({s['start_s']}) >= end_s ({s['end_s']})")
    if not isinstance(s["topic_title"], str) or len(s["topic_title"]) > 80:
        errs.append(f"[{i}] topic_title must be <= 80 chars")
    if not isinstance(s["summary"], str) or len(s["summary"]) > 300:
        errs.append(f"[{i}] summary must be <= 300 chars")
    if not s["lessons_summary"].strip():
        errs.append(f"[{i}] lessons_summary is empty")
    eq = s.get("expected_queries", [])
    if not isinstance(eq, list) or not (2 <= len(eq) <= 4):
        errs.append(f"[{i}] expected_queries must be a list of 2–4 entries (got {len(eq) if isinstance(eq, list) else 'n/a'})")
    bad = [x for x in s["lessons_categories"] if x not in LESSONS]
    if bad:
        errs.append(f"[{i}] lessons_categories has invalid values: {bad}")
    bad = [x for x in s["industries"] if x not in INDUSTRIES]
    if bad:
        errs.append(f"[{i}] industries has invalid values: {bad}")
    bad = [x for x in s["audience"] if x not in AUDIENCE]
    if bad:
        errs.append(f"[{i}] audience has invalid values: {bad}")
    if s["age_group"] not in AGE_GROUPS:
        errs.append(f"[{i}] age_group invalid: {s['age_group']!r}")
    return errs


def main() -> int:
    if not SEGMENTS_PATH.exists():
        print(f"missing: {SEGMENTS_PATH}", file=sys.stderr)
        return 2
    data = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    segs = data if isinstance(data, list) else data.get("segments", [])
    print(f"=== validating {len(segs)} segments ===")
    all_errs: list[str] = []
    by_video: dict[str, int] = {}
    for i, s in enumerate(segs):
        errs = validate_one(s, i)
        all_errs.extend(errs)
        if not errs:
            by_video[s["video_id"]] = by_video.get(s["video_id"], 0) + 1
    if all_errs:
        for e in all_errs[:50]:
            print(f"  FAIL {e}")
        if len(all_errs) > 50:
            print(f"  ... and {len(all_errs) - 50} more")
        print(f"\n{len(all_errs)} validation errors")
        return 2
    print(f"all segments valid. videos covered: {len(by_video)}; mean segs/video: {sum(by_video.values())/max(1,len(by_video)):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
