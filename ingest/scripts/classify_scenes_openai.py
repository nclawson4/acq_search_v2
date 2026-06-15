"""Classify scene keyframes with GPT-4o-mini vision into 4 tags per scene.

Output JSON per frame:
  {
    "format":    one of [talking_head, podcast, phone_qa, live_qa, low_production, whiteboard],
    "shot_type": one of [wide, medium, close, b_roll, title_card, pip],
    "who":       one of [alex, leila, sharran, guest, none],
    "has_text":  true | false,
    "one_line":  short caption (10-15 words)
  }

Persists to: cache/scene_tags.json   (keyed by frame_path)

Idempotent: re-running skips frames already in the cache unless --force.

CLI:
  python scripts/classify_scenes_openai.py --frames frame1,frame2,...   # specific frames
  python scripts/classify_scenes_openai.py --videos vid1,vid2           # all frames in these videos
  python scripts/classify_scenes_openai.py --from-batch                 # all frames in the labeling-UI batch
  python scripts/classify_scenes_openai.py                              # all 15,621 frames
  python scripts/classify_scenes_openai.py --concurrency 8              # parallel API calls
  python scripts/classify_scenes_openai.py --dry-run                    # show what would run
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR, FRAMES_DIR  # noqa: E402

REPO_ROOT = ROOT.parent  # acq_search_v2
SCENE_TAGS_PATH = CACHE_DIR / "scene_tags.json"
BATCH_MANIFEST = REPO_ROOT / "eval" / "data" / "label_batch_manifest.json"

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You classify keyframes from long-form podcast / livestream / talking-head \
videos. Output strict JSON, no prose."""

USER_PROMPT_TEMPLATE = """Classify this single thumbnail. Return JSON with EXACTLY these keys:

  "is_animation":      true | false
  "talking_head_pose": "front_view" | "none"

is_animation:
- TRUE  if the frame is dominated by graphics: motion-graphic intros/outros, fully-animated explainer
  segments, full-screen title/lower-third cards, animated charts, slide-style graphic frames with
  text+shapes, animated transitions. No live human face is the primary subject.
- FALSE in every other case: live person on camera, live B-roll footage (movies, sports, photos,
  documentary footage of real people), studio shots, stage shots, casual outdoor handheld, dark
  cinematic real-world clips. Live people = FALSE even if there is heavy caption text overlaid.

talking_head_pose — BINARY: "front_view" means EXACTLY ONE live person is clearly visible in the frame. \
Head angle does not matter. Framing tightness does not matter. The person can be looking straight at \
the camera, off to the side, down at notes, at a whiteboard, in profile — as long as ONE person is the \
clear subject of the shot, mark "front_view". Wide shots count. Medium shots count. Close-ups count. \
Side-angle shots count.

Use "none" when ANY of these is true:

  1. is_animation is TRUE → ALWAYS "none". (No exceptions.)
  2. TWO OR MORE people are visible in the frame at comparable prominence. Podcast cuts where both
     host and guest are on screen → "none". Group shots → "none". Side-by-side conversation → "none".
     A single dominant person with a tiny background bystander still counts as front_view; the
     question is whether 2+ people are clearly part of the scene.
  3. ZERO people in the frame: pure B-roll without people (scenery, products, hands-only close-ups,
     archival graphics, animations, movie clips with no clear single figure, audience-only shots) → "none".
  4. The face/figure is too small or occluded to identify any person at all.

Counter-examples — these ARE front_view (do not over-tag as "none"):
  - One person facing slightly away (e.g. looking at notes, looking at a whiteboard) — still front_view
  - One person in a wide shot taking only 30% of the frame — still front_view
  - One person in profile (side angle) — still front_view, as long as nobody else is in the shot
  - One person filmed from above or below — still front_view

Return ONLY the JSON object, no markdown."""


def encode_image(p: Path) -> str:
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_existing() -> dict[str, dict]:
    if not SCENE_TAGS_PATH.exists():
        return {}
    try:
        return json.loads(SCENE_TAGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_tags(tags: dict[str, dict]) -> None:
    SCENE_TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENE_TAGS_PATH.write_text(json.dumps(tags, indent=2), encoding="utf-8")


async def classify_one(client, frame_path: Path, detail: str = "low") -> dict:
    """Returns the JSON dict or raises. detail in {'low', 'high'}."""
    b64 = encode_image(frame_path)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEMPLATE},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0.0,
    )
    text = resp.choices[0].message.content or "{}"
    return json.loads(text)


def gather_frames(args: argparse.Namespace) -> list[Path]:
    frames: list[Path] = []
    if args.frames:
        for p in args.frames.split(","):
            frames.append(Path(p.strip()))
    elif args.videos:
        for v in args.videos.split(","):
            v = v.strip()
            for jpg in sorted((FRAMES_DIR / v).glob("scene_*.jpg")):
                if not jpg.stem.endswith("_intra"):  # skip intra-scene helpers
                    frames.append(jpg)
    elif args.from_batch:
        manifest = json.loads(BATCH_MANIFEST.read_text(encoding="utf-8"))
        for f in manifest["frames"]:
            # Manifest stores paths relative to ingest/ (e.g. "frames/<vid>/scene_NNNN.jpg")
            # OR relative to repo root (e.g. "ingest/frames/<vid>/scene_NNNN.jpg").
            # Normalize so the resolved Path points at the actual JPEG on disk.
            rel = f["frame_path"]
            if rel.startswith("ingest/"):
                frames.append(REPO_ROOT / rel)
            else:
                frames.append(REPO_ROOT / "ingest" / rel)
    else:
        for vid_dir in sorted(FRAMES_DIR.iterdir()):
            if not vid_dir.is_dir():
                continue
            for jpg in sorted(vid_dir.glob("scene_*.jpg")):
                if not jpg.stem.endswith("_intra"):
                    frames.append(jpg)
    return frames


async def main_async(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("pip install openai", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    frames = gather_frames(args)
    existing = load_existing()
    pending = [p for p in frames if args.force or _path_key(p) not in existing]
    print(f"{len(frames)} total, {len(frames) - len(pending)} cached, {len(pending)} pending")
    if args.dry_run:
        for p in pending[:10]:
            print(f"  would classify: {p}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return 0
    if not pending:
        print("nothing to do.")
        return 0

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)
    completed = 0
    failed = 0
    t0 = time.monotonic()

    async def worker(p: Path) -> tuple[Path, dict | None, str | None]:
        async with sem:
            try:
                result = await classify_one(client, p, detail=args.detail)
                return p, result, None
            except Exception as e:
                return p, None, str(e)

    # batch through with periodic checkpoints so we don't lose work on crash
    BATCH = 50
    for i in range(0, len(pending), BATCH):
        chunk = pending[i:i + BATCH]
        results = await asyncio.gather(*(worker(p) for p in chunk))
        for p, data, err in results:
            if err is not None:
                failed += 1
                print(f"  FAIL {p}: {err}", flush=True)
                continue
            existing[_path_key(p)] = data
            completed += 1
        save_tags(existing)
        dt = time.monotonic() - t0
        print(f"  [{completed + failed}/{len(pending)}] saved checkpoint  ({completed} ok, {failed} fail, {dt:.1f}s)", flush=True)

    print(f"done: {completed} classified, {failed} failed in {time.monotonic() - t0:.1f}s")
    return 0


def _path_key(p: Path) -> str:
    """Match the key format used in scene_tags.json (relative to v2 root, forward slashes)."""
    rel = p.resolve().relative_to(REPO_ROOT.resolve())
    return str(rel).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", help="CSV of frame paths")
    parser.add_argument("--videos", help="CSV of video ids")
    parser.add_argument("--from-batch", action="store_true",
                        help="classify only the 100 frames in label_batch_manifest.json")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-classify even if cached")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detail", choices=["low", "high"], default="low",
                        help="OpenAI image detail. high is ~10x cost but better at sub-pixel features.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
