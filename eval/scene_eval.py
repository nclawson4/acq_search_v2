"""Compute precision / recall / F1 for scene-detect against hand-labeled ground truth.

Inputs:
  - eval/data/scene_truth_candidates.json  — labeled candidates (label ∈ {true, false})
  - ingest/cache/scenes/<video_id>.json    — detector output

Method per candidate:
  - For "adjacent_pair" candidates: detector SAID there's a cut between scene_idx1 and scene_idx2.
    -> predicted_cut = True
    -> true_cut     = candidate.label
  - For "intra_scene" candidates: detector said NO cut inside scene_idx (same scene).
    -> predicted_cut = False
    -> true_cut     = candidate.label

Aggregate:
  TP = predicted_cut && true_cut
  FP = predicted_cut && !true_cut    (detector found a cut that wasn't real)
  FN = !predicted_cut && true_cut    (detector missed a real cut)
  TN = !predicted_cut && !true_cut
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  F1        = 2 * P * R / (P + R)

CLI:
  python eval/scene_eval.py                                  # default
  python eval/scene_eval.py --truth=eval/data/scene_truth_candidates.json
  python eval/scene_eval.py --by-source                      # report split per source
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import CACHE_DIR  # noqa: E402

DEFAULT_TRUTH = ROOT / "eval" / "data" / "scene_truth_candidates.json"
SCENES_DIR = CACHE_DIR / "scenes"


def confusion(labels: list[tuple[bool, bool]]) -> dict:
    """labels = list of (predicted_cut, true_cut). Returns metrics dict."""
    tp = sum(1 for p, t in labels if p and t)
    fp = sum(1 for p, t in labels if p and not t)
    fn = sum(1 for p, t in labels if not p and t)
    tn = sum(1 for p, t in labels if not p and not t)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "n_labeled": tp + fp + fn + tn,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH))
    parser.add_argument("--by-source", action="store_true")
    parser.add_argument("--by-video", action="store_true")
    args = parser.parse_args()

    truth_path = Path(args.truth)
    if not truth_path.exists():
        print(f"missing truth file: {truth_path}", file=sys.stderr)
        return 1

    with truth_path.open(encoding="utf-8") as f:
        candidates = json.load(f)

    labeled = [c for c in candidates if c.get("label") in (True, False, "true", "false")]
    unlabeled = len(candidates) - len(labeled)

    def to_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() == "true"

    by_source: dict[str, list[tuple[bool, bool]]] = {"adjacent_pair": [], "intra_scene": []}
    by_video: dict[str, list[tuple[bool, bool]]] = {}
    all_pairs: list[tuple[bool, bool]] = []

    for c in labeled:
        predicted = c["source"] == "adjacent_pair"  # detector said cut iff adjacent_pair
        true_lbl = to_bool(c["label"])
        pair = (predicted, true_lbl)
        all_pairs.append(pair)
        by_source[c["source"]].append(pair)
        by_video.setdefault(c["video_id"], []).append(pair)

    print(f"=== labeled: {len(labeled)} / {len(candidates)} ({unlabeled} unlabeled) ===\n")

    overall = confusion(all_pairs)
    print("OVERALL:")
    print(f"  precision: {overall['precision']:.3f}")
    print(f"  recall:    {overall['recall']:.3f}")
    print(f"  F1:        {overall['f1']:.3f}")
    print(f"  tp={overall['tp']} fp={overall['fp']} fn={overall['fn']} tn={overall['tn']}")

    if args.by_source:
        print("\nBY SOURCE:")
        for src, pairs in by_source.items():
            if not pairs:
                continue
            m = confusion(pairs)
            print(f"  {src}: n={m['n_labeled']} precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")

    if args.by_video:
        print("\nBY VIDEO:")
        for vid, pairs in sorted(by_video.items()):
            m = confusion(pairs)
            print(f"  {vid}: n={m['n_labeled']} precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")

    # Threshold check
    print()
    if overall["precision"] >= 0.95 and overall["recall"] >= 0.95:
        print(f"*** PASS *** (precision ≥ 0.95 AND recall ≥ 0.95)")
        return 0
    else:
        deficit = []
        if overall["precision"] < 0.95:
            deficit.append(f"precision {overall['precision']:.3f} < 0.95")
        if overall["recall"] < 0.95:
            deficit.append(f"recall {overall['recall']:.3f} < 0.95")
        print(f"*** FAIL *** ({', '.join(deficit)})")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
