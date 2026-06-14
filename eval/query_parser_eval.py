"""LLM query parser eval: a battery of phrasings with expected filter values.

Each case asserts a subset of fields on the parsed dict. Fields not listed
in `expect` are ignored (so we don't over-constrain).

Run:
  python -m eval.query_parser_eval
  python -m eval.query_parser_eval --strict   # exit nonzero if any case fails

Designed to catch regressions BEFORE shipping a prompt or model change.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from lib.query_parser import parse_async  # noqa: E402


# Each case: (query, expect_dict). expect values may be:
#   - exact value (str/int/bool/None)
#   - list[T]: must be present with these elements (order-insensitive)
#   - ("any_of", [a, b, c]) tuple: parsed value must be one of these
CASES: list[tuple[str, dict[str, Any]]] = [
    # The two user-reported failures
    ("Sharran talking about real estate less than one month ago",
     {"speaker": "sharran", "max_age_days": 30, "min_age_days": None}),
    ("Sharran talking about real estate. less than one month old",
     {"speaker": "sharran", "max_age_days": 30, "min_age_days": None}),

    # Recency phrasings that should all map to max_age (recent direction)
    ("Alex in the last 30 days",                     {"speaker": "alex", "max_age_days": 30}),
    ("Alex from the past month",                     {"speaker": "alex", "max_age_days": 30}),
    ("Alex within 2 weeks",                          {"speaker": "alex", "max_age_days": 14}),
    ("Alex recently",                                {"speaker": "alex", "max_age_days": ("any_of", [7, 14, 30])}),
    ("Alex this week",                               {"speaker": "alex", "max_age_days": 7}),
    ("Alex today",                                   {"speaker": "alex", "max_age_days": ("any_of", [1, 2])}),
    ("Leila newer than two weeks",                   {"speaker": "leila", "max_age_days": 14}),
    ("Sharran from a couple weeks ago",              {"speaker": "sharran", "max_age_days": ("any_of", [10, 14, 21])}),
    ("Alex content not older than 60 days",          {"speaker": "alex", "max_age_days": 60}),
    ("Leila up to 3 months ago",                     {"speaker": "leila", "max_age_days": 90}),

    # Old direction
    ("Alex from over a year ago",                    {"speaker": "alex", "min_age_days": 365}),
    ("older Leila content about pricing",            {"speaker": "leila", "min_age_days": ("any_of", [60, 90, 180])}),
    ("Alex's original videos",                       {"speaker": "alex", "min_age_days": ("any_of", [90, 180, 365])}),
    ("Sharran more than 6 weeks ago",                {"speaker": "sharran", "min_age_days": 42}),

    # Both directions
    ("Alex from the last 6 months but not in the last 30 days",
     {"speaker": "alex", "max_age_days": 180, "min_age_days": 30}),

    # Speaker count auto-parse
    ("Alex and Leila roundtable",                    {"required_speakers": ["alex", "leila"], "speakers_count": "group"}),
    ("Sharran solo to camera",                       {"speaker": "sharran", "speakers_count": "solo"}),
    ("two-person conversation between Alex and a guest", {"speaker": "alex", "speakers_count": "dialogue"}),
    ("panel discussion about hiring",                {"speakers_count": "group"}),

    # Visual filters
    ("animated intro graphics",                      {"is_animation": True}),
    ("Alex face to camera explaining pricing",       {"speaker": "alex", "talking_head_pose": "front_view"}),
    ("title cards from this year",                   {"is_animation": True, "max_age_days": 365}),

    # Segment metadata
    ("Sharran real estate content",                  {"speaker": "sharran", "industries": ["real_estate"]}),
    ("teach me how to hire",                         {"lessons_categories": ["hiring"], "instructional_only": True}),
    ("Alex on sales for SaaS founders",              {"speaker": "alex", "industries": ["saas"]}),

    # Negative / no-filter cases
    ("real estate stuff",                            {"speaker": None, "max_age_days": None, "min_age_days": None}),
    ("show me clips",                                {"speaker": None, "max_age_days": None, "min_age_days": None}),
    ("",                                             {"speaker": None, "max_age_days": None, "min_age_days": None}),
]


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, tuple) and len(expected) == 2 and expected[0] == "any_of":
        return actual in expected[1]
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return set(actual) == set(expected)
    return actual == expected


async def run(strict: bool = False) -> int:
    passes = 0
    fails = 0
    failures: list[tuple[str, str, Any, Any]] = []

    for query, expect in CASES:
        parsed = await parse_async(query, force=True)
        if parsed is None:
            print(f"  SKIP (LLM unavailable): {query!r}")
            continue
        case_failed = False
        for field, exp in expect.items():
            got = parsed.get(field)
            if not _matches(got, exp):
                case_failed = True
                failures.append((query, field, exp, got))
        if case_failed:
            fails += 1
            print(f"  FAIL  {query!r}")
            for q, f, e, g in failures[-len(expect):]:
                if q == query:
                    print(f"        {f}: expected={e!r}  got={g!r}")
        else:
            passes += 1
            print(f"  PASS  {query!r}")

    print()
    print(f"== {passes} pass / {fails} fail / {len(CASES)} total ==")
    if fails and strict:
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit nonzero if any case fails")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(strict=args.strict)))


if __name__ == "__main__":
    main()
