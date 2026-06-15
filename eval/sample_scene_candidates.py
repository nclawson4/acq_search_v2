"""Sample candidate frame-pairs for hand-labeling scene-detect ground truth.

Two sampling sources guarantee both precision and recall are measurable:

1. **Adjacent-scene pairs** (precision source).
   For each detector-found cut between scene N and N+1, emit a candidate
   with frame1 = midpoint of scene N, frame2 = midpoint of scene N+1.
   The labeler decides whether the two frames are actually different shots.
   If detector says cut but they're the same shot → false positive.

2. **Intra-scene pairs** (recall source).
   For scenes >= MIN_SCENE_FOR_RECALL seconds (where a missed cut might
   hide inside), emit a candidate with frame1 = midpoint, frame2 = a
   frame extracted near the end of the same scene. If the labeler says
   they're different shots, the detector missed a cut → false negative.

Output: eval/data/scene_truth_candidates.json
  [{video_id, source, t1_s, t2_s, frame1_path, frame2_path, label: null,
     notes: "", scene_idx1, scene_idx2}]

Labels stay null until a labeler (Claude, by viewing both frames) fills them
in. After labeling, eval/scene_eval.py consumes the same file to compute
precision / recall / F1 with ±TOLERANCE_S tolerance against the cached
detector output.

CLI:
  python eval/sample_scene_candidates.py                       # all cached videos
  python eval/sample_scene_candidates.py --videos a,b,c        # subset
  python eval/sample_scene_candidates.py --per-video=30        # cap per video
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import CACHE_DIR, FRAMES_DIR, MEDIA_DIR  # noqa: E402

SCENES_DIR = CACHE_DIR / "scenes"
EXTRA_FRAMES_DIR = FRAMES_DIR  # we may need to extract extra frames at non-midpoint times
OUT_PATH = ROOT / "eval" / "data" / "scene_truth_candidates.json"

MIN_SCENE_FOR_RECALL = 5.0   # only inspect scenes longer than this for missed cuts
RANDOM_SEED = 42


def extract_frame_at(video_path: Path, t_s: float, out_path: Path, width: int = 512) -> bool:
    """Pull a frame at timestamp t_s from video_path and save resized JPEG."""
    if out_path.exists():
        return True
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        rgb = frame[:, :, ::-1]
        img = Image.fromarray(rgb)
        w, h = img.size
        if w > width:
            ratio = width / w
            img = img.resize((width, int(h * ratio)), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=85, optimize=True)
        return True
    finally:
        cap.release()


def candidates_for_video(video_id: str, scenes: list[dict], per_video_cap: int, rng: random.Random) -> list[dict]:
    """Build candidate pairs for one video."""
    if not scenes:
        return []
    video_path = MEDIA_DIR / f"{video_id}.mp4"

    # 1. Adjacent-pair candidates (precision)
    precision_candidates: list[dict] = []
    for i in range(len(scenes) - 1):
        s1, s2 = scenes[i], scenes[i + 1]
        precision_candidates.append({
            "video_id": video_id,
            "source": "adjacent_pair",
            "scene_idx1": s1["idx"],
            "scene_idx2": s2["idx"],
            "t1_s": s1["mid_s"],
            "t2_s": s2["mid_s"],
            "frame1_path": s1["frame_path"],
            "frame2_path": s2["frame_path"],
            "label": None,
            "notes": "",
        })

    # 2. Intra-scene candidates (recall)
    intra_candidates: list[dict] = []
    for s in scenes:
        dur = s["end_s"] - s["start_s"]
        if dur < MIN_SCENE_FOR_RECALL:
            continue
        # extract a second frame near scene end (avoid landing on boundary)
        t1 = s["mid_s"]
        t2 = max(s["mid_s"] + 0.5, s["end_s"] - 0.5)
        if abs(t2 - t1) < 1.0:
            continue
        frame2_rel = f"ingest/frames/{video_id}/scene_{s['idx']:04d}_intra.jpg"
        frame2_abs = ROOT / frame2_rel
        if not extract_frame_at(video_path, t2, frame2_abs):
            continue
        intra_candidates.append({
            "video_id": video_id,
            "source": "intra_scene",
            "scene_idx1": s["idx"],
            "scene_idx2": s["idx"],
            "t1_s": round(t1, 3),
            "t2_s": round(t2, 3),
            "frame1_path": s["frame_path"],
            "frame2_path": frame2_rel.replace("\\", "/"),
            "label": None,
            "notes": "",
        })

    # Sample down to per_video_cap if needed; balance precision vs recall sources
    half = per_video_cap // 2
    rng.shuffle(precision_candidates)
    rng.shuffle(intra_candidates)
    out = precision_candidates[:half] + intra_candidates[:per_video_cap - half]
    rng.shuffle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", help="comma-separated video IDs (default: all cached)")
    parser.add_argument("--exclude-videos", help="comma-separated video IDs to skip")
    parser.add_argument("--per-video", type=int, default=30, help="max candidates per video")
    parser.add_argument("--out", default=str(OUT_PATH), help="output JSON path")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)

    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",") if v.strip()}
    else:
        wanted = None

    excluded = set()
    if args.exclude_videos:
        excluded = {v.strip() for v in args.exclude_videos.split(",") if v.strip()}

    candidates: list[dict] = []
    seen = 0
    for scene_file in sorted(SCENES_DIR.glob("*.json")):
        vid = scene_file.stem
        if wanted is not None and vid not in wanted:
            continue
        if vid in excluded:
            continue
        with scene_file.open(encoding="utf-8") as f:
            data = json.load(f)
        cands = candidates_for_video(vid, data.get("scenes", []), args.per_video, rng)
        candidates.extend(cands)
        seen += 1
        print(f"  {vid}: {len(cands)} candidates ({len(data.get('scenes', []))} scenes)", flush=True)

    n_prec = sum(1 for c in candidates if c["source"] == "adjacent_pair")
    n_recall = sum(1 for c in candidates if c["source"] == "intra_scene")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(f"\nwrote {len(candidates)} candidates ({n_prec} adjacent_pair, {n_recall} intra_scene) across {seen} videos")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
