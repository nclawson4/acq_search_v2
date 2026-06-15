"""Scene detection + keyframe extraction.

For every mp4 in MEDIA_DIR:
  1. Run PySceneDetect ContentDetector to find shot boundaries.
  2. For each scene, extract one keyframe at the scene midpoint (resized to
     FRAME_WIDTH for downstream CLIP).
  3. Write JSON metadata to CACHE_DIR/scenes/<video_id>.json:
        {
          "video_id": "<id>",
          "threshold": <float>,
          "scenes": [
              {"idx": 0, "start_s": 0.0, "end_s": 14.2, "frame_path": "..."},
              ...
          ]
        }
  4. Write keyframe JPEGs to FRAMES_DIR/<video_id>/scene_<NNN>.jpg.

Idempotent: skips videos whose scenes JSON already exists. --force re-runs.

CLI:
  python scripts/extract_scenes.py                    # all 66 videos
  python scripts/extract_scenes.py --only=<id>        # single video
  python scripts/extract_scenes.py --sample=7         # first N as a quick sample
  python scripts/extract_scenes.py --threshold=27     # tune ContentDetector
  python scripts/extract_scenes.py --force            # ignore existing JSON

Progress log: LOGS_DIR/scenes.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR, FRAMES_DIR, LOGS_DIR, MEDIA_DIR  # noqa: E402

SCENES_DIR = CACHE_DIR / "scenes"
LOG_PATH = LOGS_DIR / "scenes.log"

DEFAULT_THRESHOLD = 27.0
FRAME_WIDTH = 512  # downstream CLIP image encoder will resize again; this keeps disk small

# Speed knobs (tunable; default values target ~realtime processing on CPU)
DEFAULT_DOWNSCALE = 3       # decode frames at 1/3 resolution for detection only
DEFAULT_FRAME_SKIP = 1      # analyze every 2nd frame (skip = 1 means skip every other)
DEFAULT_MIN_SCENE_LEN_S = 0.5
DEFAULT_LUMA_ONLY = True    # luma-only is ~2x faster; color-only cuts (rare) get missed


def detect_scenes(
    video_path: Path,
    threshold: float,
    downscale: int = DEFAULT_DOWNSCALE,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    min_scene_len_s: float = DEFAULT_MIN_SCENE_LEN_S,
    luma_only: bool = DEFAULT_LUMA_ONLY,
) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) tuples."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    if downscale and downscale > 1:
        try:
            video.downscale = downscale
        except Exception:
            pass  # backend may not support; fall through

    fps = float(getattr(video, "frame_rate", 24.0) or 24.0)
    min_scene_len_frames = max(1, int(round(min_scene_len_s * fps)))

    sm = SceneManager()
    sm.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len_frames,
            luma_only=luma_only,
        )
    )
    sm.detect_scenes(video, frame_skip=frame_skip)
    scene_list = sm.get_scene_list()
    return [(s[0].seconds, s[1].seconds) for s in scene_list]


def extract_midpoint_frame(video_path: Path, t_s: float, out_path: Path) -> bool:
    """Pull a single frame at timestamp t_s and write a resized JPEG. Return True on success."""
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
        # BGR -> RGB
        frame_rgb = frame[:, :, ::-1]
        img = Image.fromarray(frame_rgb)
        w, h = img.size
        if w > FRAME_WIDTH:
            ratio = FRAME_WIDTH / w
            img = img.resize((FRAME_WIDTH, int(h * ratio)), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=85, optimize=True)
        return True
    finally:
        cap.release()


def process_video(video_path: Path, threshold: float) -> dict:
    video_id = video_path.stem
    t0 = time.monotonic()
    scenes = detect_scenes(video_path, threshold)
    detect_dt = time.monotonic() - t0
    out_scenes = []
    frame_dir = FRAMES_DIR / video_id
    for idx, (start_s, end_s) in enumerate(scenes):
        mid = (start_s + end_s) / 2.0
        out_path = frame_dir / f"scene_{idx:04d}.jpg"
        ok = extract_midpoint_frame(video_path, mid, out_path)
        if not ok:
            continue
        out_scenes.append({
            "idx": idx,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "mid_s": round(mid, 3),
            "frame_path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        })
    return {
        "video_id": video_id,
        "threshold": threshold,
        "detect_seconds": round(detect_dt, 1),
        "scenes": out_scenes,
    }


def _worker(args_tuple) -> tuple:
    """Process one video. Returns (status, video_id, scenes_or_msg, wall_s)."""
    video_path, threshold, out_path = args_tuple
    t0 = time.monotonic()
    try:
        result = process_video(video_path, threshold)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return ("ok", video_path.stem, len(result["scenes"]), time.monotonic() - t0)
    except Exception as e:
        return ("err", video_path.stem, str(e), time.monotonic() - t0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="single video id")
    parser.add_argument("--videos", help="CSV of video ids to process")
    parser.add_argument("--sample", type=int, default=0, help="process first N videos only")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="parallel workers (0=auto, 1=sequential)")
    args = parser.parse_args()

    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.only:
        targets = [MEDIA_DIR / f"{args.only}.mp4"]
        if not targets[0].exists():
            print(f"not found: {targets[0]}", flush=True)
            return 1
    elif args.videos:
        wanted = [v.strip() for v in args.videos.split(",") if v.strip()]
        targets = [MEDIA_DIR / f"{v}.mp4" for v in wanted]
        missing = [t for t in targets if not t.exists()]
        if missing:
            print(f"missing: {missing}", flush=True)
            return 1
    else:
        targets = sorted(MEDIA_DIR.glob("*.mp4"))
        if args.sample > 0:
            targets = targets[:args.sample]

    n = len(targets)

    # Split into to-process vs already-cached
    pending: list[tuple[Path, float, Path]] = []
    cached = 0
    for vp in targets:
        op = SCENES_DIR / f"{vp.stem}.json"
        if op.exists() and not args.force:
            cached += 1
        else:
            pending.append((vp, args.threshold, op))

    if args.workers > 0:
        workers = args.workers
    else:
        workers = max(1, min(4, (os.cpu_count() or 4) - 1))
    if args.only or len(pending) <= 1:
        workers = 1

    print(f"=== scene-detect: {n} video(s) ({cached} cached, {len(pending)} pending) threshold={args.threshold} workers={workers} ===", flush=True)

    ok = err = 0
    t_start = time.monotonic()

    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} n={n} workers={workers} threshold={args.threshold} ===\n")

        if workers == 1:
            for i, t in enumerate(pending, 1):
                status, vid, info, dt = _worker(t)
                if status == "ok":
                    ok += 1
                    line = f"[{i}/{len(pending)}] {vid}: {info} scenes in {dt:.1f}s"
                else:
                    err += 1
                    line = f"[{i}/{len(pending)}] {vid}: ERROR ({dt:.1f}s) {info}"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker, t): t[0].stem for t in pending}
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    status, vid, info, dt = fut.result()
                    if status == "ok":
                        ok += 1
                        line = f"[{done}/{len(pending)}] {vid}: {info} scenes in {dt:.1f}s"
                    else:
                        err += 1
                        line = f"[{done}/{len(pending)}] {vid}: ERROR ({dt:.1f}s) {info}"
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()

        total = time.monotonic() - t_start
        summary = f"=== done ok={ok} cached={cached} err={err} wall={total:.1f}s ==="
        print(summary, flush=True)
        log.write(summary + "\n")

    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
