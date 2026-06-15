"""Sample frames stratified across videos for CLIP text-to-frame ground truth.

For each video with embedded frames, sample N random frames. Output a JSON
template where each row has a `description` field to be filled by a human
(or by me, Claude, via inline reasoning) for ground-truth retrieval eval.

Output: eval/data/clip_truth_candidates.json
  [{video_id, frame_path, scene_idx, is_intra, description: null, notes: ""}]

CLI:
  python eval/sample_clip_candidates.py --per-video=30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import FRAMES_DIR  # noqa: E402

OUT_PATH = ROOT / "eval" / "data" / "clip_truth_candidates.json"
RANDOM_SEED = 1337


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-video", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)

    by_video = defaultdict(list)
    for p in FRAMES_DIR.rglob("*.jpg"):
        by_video[p.parent.name].append(p)

    candidates: list[dict] = []
    for vid, paths in sorted(by_video.items()):
        rng.shuffle(paths)
        picked = paths[:args.per_video]
        for p in picked:
            stem = p.stem
            is_intra = stem.endswith("_intra")
            scene_idx = int(stem.replace("scene_", "").replace("_intra", ""))
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            candidates.append({
                "video_id": vid,
                "frame_path": rel,
                "scene_idx": scene_idx,
                "is_intra": is_intra,
                "description": None,
                "notes": "",
            })

    rng.shuffle(candidates)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(f"wrote {len(candidates)} candidates across {len(by_video)} videos -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
