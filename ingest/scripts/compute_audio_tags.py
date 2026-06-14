"""Derive per-scene audio tags from Deepgram diarization + resemblyzer speaker map.

For each scene at [start_s, end_s]:
  1. Find utterances in the same video that overlap the scene window
  2. Per utterance, get speaker cluster_id -> name (from speakers/<vid>.audio.json)
  3. Aggregate:
       voice: dominant name (by total spoken seconds) or null if no utterance overlaps
       voices_present: sorted list of all named speakers heard
       speakers_count: 'solo' (1), 'dialogue' (2), or 'group' (3+) by distinct named speakers
       silent: true if no overlapping utterance

Output: cache/audio_tags.json
  {
    "frames/<vid>/scene_<NNNN>.jpg": {
        "voice": "alex" | "leila" | "sharran" | "unknown" | null,
        "voices_present": ["alex", "sharran"],
        "speakers_count": "solo" | "dialogue" | "group",
        "silent": false,
        "n_utterances": 4,
    },
    ...
  }

Free, deterministic, no API calls. Runs in <1 min on the full 15,621-scene corpus.

CLI:
  python ingest/scripts/compute_audio_tags.py
  python ingest/scripts/compute_audio_tags.py --videos vid1,vid2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR  # noqa: E402

SCENES_DIR = CACHE_DIR / "scenes"
DEEPGRAM_DIR = CACHE_DIR / "deepgram"
SPEAKERS_DIR = CACHE_DIR / "speakers"
OUT_PATH = CACHE_DIR / "audio_tags.json"


def load_utts(vid: str) -> list[dict]:
    p = DEEPGRAM_DIR / f"{vid}.audio.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        utts_raw = data.get("results", {}).get("utterances", []) or []
        out = []
        for u in utts_raw:
            out.append({
                "start": float(u.get("start", 0)),
                "end": float(u.get("end", 0)),
                "speaker": u.get("speaker"),
            })
        out.sort(key=lambda u: u["start"])
        return out
    except Exception:
        return []


def load_speaker_map(vid: str) -> dict[int, str]:
    p = SPEAKERS_DIR / f"{vid}.audio.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {int(k): v.get("name", "unknown") for k, v in data.get("speakers", {}).items()}
    except Exception:
        return {}


def overlap_window(utts: list[dict], start_s: float, end_s: float) -> list[dict]:
    """Return utterances whose [start, end] overlaps [start_s, end_s]."""
    # Linear scan is fine — utts are O(few hundred) per video and we do it once per scene.
    out = []
    for u in utts:
        if u["end"] < start_s or u["start"] > end_s:
            continue
        out.append(u)
    return out


def count_to_label(n: int) -> str:
    if n <= 1:
        return "solo"
    if n == 2:
        return "dialogue"
    return "group"


def process_video(vid: str, scenes: list[dict]) -> dict[str, dict]:
    utts = load_utts(vid)
    spk_map = load_speaker_map(vid)
    out: dict[str, dict] = {}
    for s in scenes:
        frame_path = s["frame_path"]
        idx = s["idx"]
        start_s = s["start_s"]
        end_s = s["end_s"]
        overlapping = overlap_window(utts, start_s, end_s)
        if not overlapping:
            out[frame_path] = {
                "voice": None,
                "voices_present": [],
                "speakers_count": "solo",
                "silent": True,
                "n_utterances": 0,
            }
            continue
        # Sum spoken seconds per cluster_id within the scene window
        per_cluster: Counter = Counter()
        for u in overlapping:
            cluster = u.get("speaker")
            if cluster is None:
                continue
            dur = max(0.0, min(end_s, u["end"]) - max(start_s, u["start"]))
            per_cluster[cluster] += dur
        # Dominant cluster -> name
        if per_cluster:
            dominant_cluster = max(per_cluster.items(), key=lambda kv: kv[1])[0]
            voice = spk_map.get(int(dominant_cluster), "unknown")
        else:
            voice = None
        # All speakers heard (alex/leila/sharran/unknown) — UI display unchanged.
        present_names = sorted({spk_map.get(int(c), "unknown") for c in per_cluster.keys()})
        # speakers_count is based on ACTIVE Deepgram clusters (>= MIN_ACTIVE_S spoken in the
        # scene window). Counting all distinct voices — including unnamed guests — catches
        # the Alex+podcast-guest scenes that the old "named only" rule was tagging solo.
        # The 1s threshold suppresses backchannel grunts ("yeah", "mhm").
        MIN_ACTIVE_S = 1.0
        active_clusters = sum(1 for d in per_cluster.values() if d >= MIN_ACTIVE_S)
        if active_clusters == 0:
            count_label = "solo"
        else:
            count_label = count_to_label(active_clusters)
        out[frame_path] = {
            "voice": voice,
            "voices_present": present_names,
            "speakers_count": count_label,
            "silent": False,
            "n_utterances": len(overlapping),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", help="comma-separated video IDs (default: all)")
    args = parser.parse_args()

    wanted = None
    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",")}

    out: dict[str, dict] = {}
    total = 0
    for scene_file in sorted(SCENES_DIR.glob("*.json")):
        vid = scene_file.stem
        if wanted is not None and vid not in wanted:
            continue
        data = json.loads(scene_file.read_text(encoding="utf-8"))
        scenes = data.get("scenes", [])
        if not scenes:
            continue
        out.update(process_video(vid, scenes))
        total += len(scenes)
        print(f"  {vid}: {len(scenes)} scenes tagged", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {len(out)} scene tags ({total} processed) -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
