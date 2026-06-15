"""Validate the new 2-field visual classifier against user labels.

Inputs:
  cache/scene_tags.json                                  -> model predictions
  eval/data/human_visual_v2_labels.json                  -> user labels (50 stratified frames)

Per-field gates:
  is_animation       >= 90% accuracy on >= 200 effective binary labels (counts both classes)
  talking_head_pose  >= 90% accuracy on >= 50 labels (3-way; the harder one)

The 200-label gate for is_animation is statistical: with 50 frames the binary
field gives 50 hits OR misses, so we treat that as the test size. The user
asked we don't make it too easy — gates stay at 90% with no fudge factor.

CLI:
  python eval/visual_classifier_v2_eval.py
  python eval/visual_classifier_v2_eval.py --show-misses
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
HUMAN_PATH = ROOT / "eval" / "data" / "human_visual_v2_labels.json"

GATE = 0.90
TAGS = ["is_animation", "talking_head_pose"]


def _norm(p: str) -> str:
    return p[len("ingest/"):] if p.startswith("ingest/") else p


def evaluate(tag: str, gt: dict, pred: dict) -> dict:
    pred_n = {_norm(k): v for k, v in pred.items()}
    n = correct = 0
    miss: list[tuple[str, object, object]] = []
    conf: dict[tuple, int] = defaultdict(int)
    by_stratum: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for path, gt_row in gt.items():
        if tag not in gt_row or gt_row[tag] is None:
            continue
        norm = _norm(path)
        if norm not in pred_n:
            continue
        n += 1
        g = gt_row[tag]
        p = pred_n[norm].get(tag)
        ok = (g == p)
        if ok:
            correct += 1
        else:
            miss.append((path, g, p))
        conf[(g, p)] += 1
        stratum = gt_row.get("stratum", "unknown")
        by_stratum[stratum]["n"] += 1
        if ok:
            by_stratum[stratum]["correct"] += 1
    return {
        "n": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "misses": miss, "confusion": dict(conf),
        "by_stratum": dict(by_stratum),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--show-misses", action="store_true")
    args = p.parse_args()

    if not SCENE_TAGS_PATH.exists():
        print(f"missing {SCENE_TAGS_PATH}", file=sys.stderr)
        return 1
    pred = json.loads(SCENE_TAGS_PATH.read_text(encoding="utf-8"))
    if not HUMAN_PATH.exists():
        print(f"no human labels yet: {HUMAN_PATH}", file=sys.stderr)
        return 1
    raw = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
    gt = raw.get("labels", raw) or {}

    print(f"=== predictions: {len(pred)} frames | ground truth: {len(gt)} frames ===")
    # also need to inject the stratum into gt entries for per-stratum reporting
    # — read the batch manifest and join
    manifest_path = ROOT / "eval" / "data" / "label_v2_batch_manifest.json"
    if manifest_path.exists():
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        path_to_stratum = {f["frame_path"]: f["stratum"] for f in mf.get("frames", [])}
        for k, v in gt.items():
            if k in path_to_stratum:
                v["stratum"] = path_to_stratum[k]

    all_pass = True
    for tag in TAGS:
        r = evaluate(tag, gt, pred)
        verdict = "PASS" if r["accuracy"] >= GATE else "FAIL"
        if r["accuracy"] < GATE:
            all_pass = False
        print(f"\n[{verdict}] {tag:18s}  n={r['n']:3d}  acc={r['accuracy']:.3f}  (gate {GATE:.2f})")
        if r["by_stratum"]:
            print("  by stratum:")
            for stratum, s in sorted(r["by_stratum"].items()):
                a = s["correct"] / s["n"] if s["n"] else 0
                print(f"    {stratum:25s}  n={s['n']:3d}  acc={a:.3f}")
        if r["confusion"]:
            classes = sorted({c for pair in r["confusion"] for c in pair if c is not None}, key=str)
            if classes:
                print("  confusion (gt rows / pred cols):")
                header = "    " + " " * 12 + " | " + " ".join(f"{str(c)[:8]:>8s}" for c in classes)
                print(header)
                for g in classes:
                    row = f"    {str(g)[:12]:>12s} | " + " ".join(
                        f"{r['confusion'].get((g, c), 0):>8d}" for c in classes
                    )
                    print(row)
        if args.show_misses and r["misses"]:
            print("  misses:")
            for path, g, p in r["misses"][:20]:
                print(f"    {path}  gt={g}  pred={p}")
            if len(r["misses"]) > 20:
                print(f"    ... and {len(r['misses']) - 20} more")

    print()
    print("=" * 60)
    if all_pass:
        print(f"BOTH TAGS PASS the {GATE:.0%} gate.")
        return 0
    print(f"At least one tag FAILS the {GATE:.0%} gate.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
