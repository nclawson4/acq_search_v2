"""CLI entry: run A/B eval for one or all Deepgram features.

Usage:
  python -m eval.deepgram_features.run_eval --feature entities
  python -m eval.deepgram_features.run_eval --feature all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import run_feature  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["entities", "topics", "sentiment", "summary", "all"], required=True)
    args = ap.parse_args()

    if args.feature == "all":
        for feat in ["entities", "topics", "sentiment", "summary"]:
            run_feature(feat)
    else:
        run_feature(args.feature)


if __name__ == "__main__":
    main()
