"""Evaluate GPT-4o-mini scene-tag predictions against human ground truth.

Inputs
  cache/scene_tags.json                          -> model predictions (keyed by frame_path)
  eval/data/human_scene_tags.json                -> user labels (under "labels" key)
  eval/data/claude_scene_tags.json (optional)    -> Claude-labeled supplement

Reports per-classifier accuracy + per-class precision/recall + confusion matrix.

Gates
  - format    : >= 95% accuracy
  - shot_type : >= 95% accuracy
  - who       : >= 95% accuracy
  - has_text  : >= 95% accuracy

CLI
  python eval/scene_classification_eval.py
  python eval/scene_classification_eval.py --min-frames 200
  python eval/scene_classification_eval.py --tag format          # only this tag
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import CACHE_DIR  # noqa: E402

SCENE_TAGS_PATH = CACHE_DIR / "scene_tags.json"
HUMAN_PATH = ROOT / "eval" / "data" / "human_scene_tags.json"
CLAUDE_PATH = ROOT / "eval" / "data" / "claude_scene_tags.json"

TAGS = ["format", "shot_type", "who", "has_text"]
GATE = 0.95


def load_predictions() -> dict[str, dict]:
    if not SCENE_TAGS_PATH.exists():
        return {}
    return json.loads(SCENE_TAGS_PATH.read_text(encoding="utf-8"))


def load_ground_truth() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if HUMAN_PATH.exists():
        d = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
        out.update(d.get("labels", d) or {})
    if CLAUDE_PATH.exists():
        d = json.loads(CLAUDE_PATH.read_text(encoding="utf-8"))
        for k, v in (d.get("labels", d) or {}).items():
            if k not in out:
                out[k] = v
    return out


def _normalize_key(p: str) -> str:
    """Strip 'ingest/' prefix so labels (frames/...) and predictions (ingest/frames/...) match."""
    if p.startswith("ingest/"):
        return p[len("ingest/"):]
    return p


def evaluate_tag(
    tag: str,
    gt: dict[str, dict],
    pred: dict[str, dict],
) -> dict:
    """Compute accuracy + confusion for one tag."""
    pred_normalized = {_normalize_key(k): v for k, v in pred.items()}
    n = 0
    correct = 0
    confusion: dict[tuple, int] = defaultdict(int)  # (gt_value, pred_value) -> count
    missing = 0
    for path, gt_row in gt.items():
        if tag not in gt_row or gt_row[tag] is None:
            continue
        norm = _normalize_key(path)
        if norm not in pred_normalized:
            missing += 1
            continue
        n += 1
        g = gt_row[tag]
        p = pred_normalized[norm].get(tag)
        if g == p:
            correct += 1
        confusion[(g, p)] += 1
    return {
        "tag": tag,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "confusion": dict(confusion),
        "missing_predictions": missing,
    }


def print_confusion(tag: str, conf: dict[tuple, int]) -> None:
    classes = sorted({c for pair in conf for c in pair if c is not None}, key=str)
    if not classes:
        return
    print(f"\n  CONFUSION ({tag}):")
    header = f"    {'gt \\ pred':>18s} | " + " ".join(f"{str(c)[:6]:>6s}" for c in classes)
    print(header)
    for g in classes:
        row_total = sum(conf.get((g, p), 0) for p in classes + [None])
        row = f"    {str(g)[:18]:>18s} | " + " ".join(f"{conf.get((g, p), 0):6d}" for p in classes)
        if row_total > 0:
            row += f"  (gt n={row_total})"
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-frames", type=int, default=0, help="warn if test set < this many frames")
    parser.add_argument("--tag", choices=TAGS, help="evaluate only this tag")
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    pred = load_predictions()
    gt = load_ground_truth()

    print(f"=== predictions: {len(pred)} | ground truth: {len(gt)} ===")
    if args.min_frames and len(gt) < args.min_frames:
        print(f"  WARNING: ground truth has only {len(gt)} frames (< requested {args.min_frames})")

    tags = [args.tag] if args.tag else TAGS
    passed_all = True
    for tag in tags:
        result = evaluate_tag(tag, gt, pred)
        n = result["n"]
        acc = result["accuracy"]
        verdict = "PASS" if acc >= GATE else "FAIL"
        if acc < GATE:
            passed_all = False
        print(f"\n[{verdict}] tag={tag:10s}  n={n:4d}  accuracy={acc:.3f}  (gate {GATE:.2f})")
        if result["missing_predictions"]:
            print(f"  ({result['missing_predictions']} frames had no model prediction; skipped)")
        if args.show_misses:
            for path, gt_row in gt.items():
                if tag in gt_row and path in pred and pred[path].get(tag) != gt_row[tag]:
                    print(f"    MISS {path}: gt={gt_row[tag]} pred={pred[path].get(tag)}")
        print_confusion(tag, result["confusion"])

    print()
    print("=" * 60)
    if passed_all:
        print(f"ALL TAGS PASS the {GATE:.0%} gate.")
        return 0
    print(f"AT LEAST ONE TAG FAILS the {GATE:.0%} gate.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
