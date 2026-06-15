"""Map each scene to its parent topic_segment + compute recency_days per video.

Output:
  cache/scene_to_segment.json   — {"frames/<vid>/scene_NNNN.jpg": {"video_id": ..., "segment_idx": ...}, ...}
  cache/video_recency.json      — {"<video_id>": {"upload_date": "YYYY-MM-DD", "recency_days": N}, ...}

If a scene falls in a gap between segments (rare), nearest segment by midpoint wins.

CLI: python ingest/scripts/build_scene_segment_map.py
"""
from __future__ import annotations

import bisect
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR, MEDIA_DIR  # noqa: E402

SCENES_DIR = CACHE_DIR / "scenes"
SEGMENTS_PATH = CACHE_DIR / "topic_segments.json"
OUT_MAP = CACHE_DIR / "scene_to_segment.json"
OUT_RECENCY = CACHE_DIR / "video_recency.json"


def parse_upload_date(s: str) -> date | None:
    if not s or len(s) != 8:
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def main() -> int:
    if not SEGMENTS_PATH.exists():
        print(f"missing {SEGMENTS_PATH}", file=sys.stderr)
        return 1
    segs = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    segs = segs if isinstance(segs, list) else segs.get("segments", [])

    # Group segments by video_id, sort by start_s for binary search
    by_video: dict[str, list[dict]] = {}
    for s in segs:
        by_video.setdefault(s["video_id"], []).append(s)
    for v, lst in by_video.items():
        lst.sort(key=lambda s: s["start_s"])

    # Map each scene
    mapping: dict[str, dict] = {}
    today = date(2026, 6, 12)
    recency: dict[str, dict] = {}
    unassigned = 0

    for scene_file in sorted(SCENES_DIR.glob("*.json")):
        vid = scene_file.stem
        sdata = json.loads(scene_file.read_text(encoding="utf-8"))
        video_segs = by_video.get(vid, [])
        starts = [seg["start_s"] for seg in video_segs]
        for sc in sdata.get("scenes", []):
            key = sc["frame_path"]  # "frames/<vid>/scene_NNNN.jpg"
            scene_mid = (sc["start_s"] + sc["end_s"]) / 2.0
            seg_idx = None
            if video_segs:
                # binary search: find last segment with start_s <= scene_mid
                i = bisect.bisect_right(starts, scene_mid) - 1
                if 0 <= i < len(video_segs):
                    sg = video_segs[i]
                    if sg["start_s"] <= scene_mid <= sg["end_s"]:
                        seg_idx = sg["segment_idx"]
                if seg_idx is None:
                    # nearest by midpoint distance
                    best = min(
                        range(len(video_segs)),
                        key=lambda j: abs((video_segs[j]["start_s"] + video_segs[j]["end_s"]) / 2.0 - scene_mid),
                    )
                    seg_idx = video_segs[best]["segment_idx"]
            else:
                unassigned += 1
            mapping[key] = {"video_id": vid, "segment_idx": seg_idx}

        # video recency
        info_path = MEDIA_DIR / f"{vid}.info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                upload = info.get("upload_date", "")
                d = parse_upload_date(upload)
                if d is not None:
                    recency[vid] = {
                        "upload_date": d.isoformat(),
                        "recency_days": (today - d).days,
                    }
            except Exception:
                pass

    OUT_MAP.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    OUT_RECENCY.write_text(json.dumps(recency, indent=2), encoding="utf-8")
    print(f"wrote {len(mapping)} scene mappings ({unassigned} videos with no segments)")
    print(f"wrote {len(recency)} video recency entries")
    # Validation: every scene should have a video_id
    assigned = sum(1 for m in mapping.values() if m["segment_idx"] is not None)
    print(f"scenes assigned to a segment: {assigned}/{len(mapping)} ({assigned*100/max(1,len(mapping)):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
