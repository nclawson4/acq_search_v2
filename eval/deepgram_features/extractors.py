"""Build per-feature indexes from cache/deepgram/*.json into cache/eval_dg/.

Four independent extractors:
  entities  -> cache/eval_dg/entities.json
  topics    -> cache/eval_dg/topics.json
  sentiment -> cache/eval_dg/sentiment.npz   (per clip-frame row, aligned to clip_frames_meta)
  summary   -> cache/eval_dg/summary_embeddings.npz  (per video, CLIP text-embedded)

Run:
  python -m eval.deepgram_features.extractors --all
  python -m eval.deepgram_features.extractors --feature entities

Outputs are entirely separate from production caches; deleting cache/eval_dg/
reverts everything this eval introduces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

from config import CACHE_DIR  # noqa: E402

DG_DIR = CACHE_DIR / "deepgram"
SCENES_DIR = CACHE_DIR / "scenes"
META_PATH = CACHE_DIR / "clip_frames_meta.json"

OUT_DIR = CACHE_DIR / "eval_dg"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _video_ids() -> list[str]:
    return sorted(p.stem.replace(".audio", "") for p in DG_DIR.glob("*.audio.json"))


def _load_dg(vid: str) -> dict:
    with (DG_DIR / f"{vid}.audio.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _word_times(dg: dict) -> list[tuple[float, float]]:
    """Return [(start, end)] per word from channels[0].alternatives[0].words."""
    ch = dg["results"]["channels"][0]["alternatives"][0]
    return [(float(w["start"]), float(w["end"])) for w in ch.get("words", [])]


def _word_range_to_time(words: list[tuple[float, float]], start_word: int, end_word: int) -> tuple[float, float]:
    n = len(words)
    if n == 0:
        return (0.0, 0.0)
    s = max(0, min(start_word, n - 1))
    e = max(0, min(end_word, n - 1))
    return (words[s][0], words[e][1])


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------
def build_entities():
    out: dict[str, list[dict]] = {}
    for vid in _video_ids():
        dg = _load_dg(vid)
        ch = dg["results"]["channels"][0]["alternatives"][0]
        words = _word_times(dg)
        ents = ch.get("entities") or []
        rows = []
        for e in ents:
            try:
                sw, ew = int(e["start_word"]), int(e["end_word"])
            except (KeyError, ValueError, TypeError):
                continue
            start_s, end_s = _word_range_to_time(words, sw, ew)
            rows.append({
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "label": e.get("label", ""),
                "value": str(e.get("value", "")),
                "confidence": float(e.get("confidence", 0.0)),
            })
        out[vid] = rows
        print(f"  {vid}: {len(rows)} entities")
    path = OUT_DIR / "entities.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f)
    total = sum(len(v) for v in out.values())
    print(f"[entities] wrote {path} ({len(out)} videos, {total} entities)")


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------
def build_topics(conf_floor: float = 0.30):
    out: dict[str, list[dict]] = {}
    for vid in _video_ids():
        dg = _load_dg(vid)
        words = _word_times(dg)
        segs = (dg["results"].get("topics") or {}).get("segments") or []
        rows = []
        for s in segs:
            try:
                sw, ew = int(s["start_word"]), int(s["end_word"])
            except (KeyError, ValueError, TypeError):
                continue
            start_s, end_s = _word_range_to_time(words, sw, ew)
            for t in (s.get("topics") or []):
                conf = float(t.get("confidence_score", 0.0))
                if conf < conf_floor:
                    continue
                rows.append({
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "topic": str(t.get("topic", "")).strip(),
                    "confidence": conf,
                })
        out[vid] = rows
        print(f"  {vid}: {len(rows)} topics")
    path = OUT_DIR / "topics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f)
    total = sum(len(v) for v in out.values())
    print(f"[topics] wrote {path} ({len(out)} videos, {total} topic instances, conf>={conf_floor})")


# ---------------------------------------------------------------------------
# sentiment   (per CLIP-frame row, aligned to clip_frames_meta.json)
# ---------------------------------------------------------------------------
def _sentiment_score(label: str, score: float) -> float:
    """Map Deepgram's sentiment_score (already signed) to [-1, 1]. Falls back to label sign."""
    s = float(score)
    if -1.0 <= s <= 1.0:
        return s
    sign = {"positive": 1, "negative": -1, "neutral": 0}.get(str(label).lower(), 0)
    return float(sign) * min(1.0, abs(s))


def build_sentiment():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    n = len(meta)
    mean_s = np.zeros(n, dtype=np.float32)
    max_pos = np.zeros(n, dtype=np.float32)
    max_neg = np.zeros(n, dtype=np.float32)
    polarity_shift = np.zeros(n, dtype=np.float32)
    abs_max = np.zeros(n, dtype=np.float32)
    covered = np.zeros(n, dtype=np.float32)

    # Group meta by video
    by_vid: dict[str, list[tuple[int, int]]] = {}
    for i, m in enumerate(meta):
        by_vid.setdefault(m["video_id"], []).append((i, int(m["scene_idx"])))

    for vid, rows in by_vid.items():
        scene_path = SCENES_DIR / f"{vid}.json"
        if not scene_path.exists():
            continue
        scenes = json.loads(scene_path.read_text(encoding="utf-8"))["scenes"]
        scene_times = {int(s["idx"]): (float(s["start_s"]), float(s["end_s"])) for s in scenes}

        try:
            dg = _load_dg(vid)
        except FileNotFoundError:
            continue

        # Prefer utterance-level (timestamps already present) over segments (word-indexed).
        utts = dg["results"].get("utterances") or []
        if utts:
            timed = [
                (float(u.get("start", 0.0)), float(u.get("end", 0.0)),
                 _sentiment_score(u.get("sentiment", "neutral"), u.get("sentiment_score", 0.0)))
                for u in utts
            ]
        else:
            words = _word_times(dg)
            sents = (dg["results"].get("sentiments") or {}).get("segments") or []
            timed = []
            for seg in sents:
                try:
                    sw, ew = int(seg["start_word"]), int(seg["end_word"])
                except (KeyError, ValueError, TypeError):
                    continue
                a, b = _word_range_to_time(words, sw, ew)
                timed.append((a, b, _sentiment_score(seg.get("sentiment", "neutral"), seg.get("sentiment_score", 0.0))))

        if not timed:
            continue

        for row_i, scene_idx in rows:
            if scene_idx not in scene_times:
                continue
            sstart, send = scene_times[scene_idx]
            overlapping = [s for (a, b, s) in timed if not (b <= sstart or a >= send)]
            if not overlapping:
                continue
            covered[row_i] = 1.0
            mean_s[row_i] = float(np.mean(overlapping))
            max_pos[row_i] = float(max(0.0, max(overlapping)))
            max_neg[row_i] = float(min(0.0, min(overlapping)))
            abs_max[row_i] = float(max(abs(x) for x in overlapping))
            has_pos = any(s > 0.2 for s in overlapping)
            has_neg = any(s < -0.2 for s in overlapping)
            polarity_shift[row_i] = 1.0 if (has_pos and has_neg) else 0.0

    path = OUT_DIR / "sentiment.npz"
    np.savez_compressed(
        path,
        mean=mean_s,
        max_pos=max_pos,
        max_neg=max_neg,
        abs_max=abs_max,
        polarity_shift=polarity_shift,
        covered=covered,
    )
    print(f"[sentiment] wrote {path} | covered {int(covered.sum())}/{n} frames")


# ---------------------------------------------------------------------------
# summary  (per video, CLIP text-embedded)
# ---------------------------------------------------------------------------
def build_summary():
    # Import CLIP via the same wrapper hybrid.py uses, so embeddings live in the
    # same space as the frame and segment vectors.
    sys.path.insert(0, str(INGEST))
    from lib.hybrid import load_index, embed_query_texts  # noqa: E402

    print("[summary] loading hybrid index (one-time, for CLIP model handle)...")
    state = load_index()

    video_ids: list[str] = []
    summaries: list[str] = []
    skipped = 0
    for vid in _video_ids():
        try:
            dg = _load_dg(vid)
        except FileNotFoundError:
            continue
        s = ((dg["results"].get("summary") or {}).get("short") or "").strip()
        if not s:
            skipped += 1
            continue
        video_ids.append(vid)
        summaries.append(s)

    print(f"[summary] embedding {len(summaries)} summaries (skipped {skipped})...")
    vecs = embed_query_texts(state, summaries)  # already L2-normalized

    path = OUT_DIR / "summary_embeddings.npz"
    np.savez_compressed(path, embeddings=vecs.astype(np.float32), video_ids=np.array(video_ids))
    # Also drop a JSON of the raw summaries for the LLM judge / debugging.
    (OUT_DIR / "summaries.json").write_text(
        json.dumps(dict(zip(video_ids, summaries)), indent=2), encoding="utf-8"
    )
    print(f"[summary] wrote {path} ({len(video_ids)} videos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["entities", "topics", "sentiment", "summary"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        build_entities()
        build_topics()
        build_sentiment()
        build_summary()
    elif args.feature == "entities":
        build_entities()
    elif args.feature == "topics":
        build_topics()
    elif args.feature == "sentiment":
        build_sentiment()
    elif args.feature == "summary":
        build_summary()
    else:
        ap.error("specify --feature {entities,topics,sentiment,summary} or --all")


if __name__ == "__main__":
    main()
