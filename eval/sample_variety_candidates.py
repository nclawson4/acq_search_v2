"""Sample CLIP candidates from a list of strategic 'variety' videos."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import FRAMES_DIR  # noqa: E402

OUT_PATH = ROOT / "eval" / "data" / "clip_truth_variety.json"
RANDOM_SEED = 2024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, help="CSV of video ids")
    parser.add_argument("--per-video", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    vids = [v.strip() for v in args.videos.split(",") if v.strip()]

    out = []
    for vid in vids:
        frames_dir = FRAMES_DIR / vid
        if not frames_dir.exists():
            print(f"  no frames dir for {vid}", file=sys.stderr)
            continue
        frames = sorted(frames_dir.glob("*.jpg"))
        rng.shuffle(frames)
        picked = frames[: args.per_video]
        for p in picked:
            stem = p.stem
            is_intra = stem.endswith("_intra")
            scene_idx = int(stem.replace("scene_", "").replace("_intra", ""))
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            out.append({
                "video_id": vid,
                "frame_path": rel,
                "scene_idx": scene_idx,
                "is_intra": is_intra,
                "description": None,
                "notes": "",
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing if file exists (to grow over multiple invocations)
    existing: list[dict] = []
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    keep = [c for c in existing if c["video_id"] not in set(vids)]
    merged = keep + out
    OUT_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    by_vid = {}
    for c in merged:
        by_vid.setdefault(c["video_id"], 0)
        by_vid[c["video_id"]] += 1
    print(f"wrote {len(merged)} total variety candidates -> {OUT_PATH}")
    for v, n in sorted(by_vid.items()):
        print(f"  {v}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
