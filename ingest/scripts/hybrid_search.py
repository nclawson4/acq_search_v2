"""Hybrid text-to-moment search for v2.

Combines three signals:
  1. CLIP visual: text query → CLIP text encoder → cosine vs cached frame vectors
  2. Deepgram transcript: simple word-overlap (BM25-lite) over per-utterance transcripts
  3. Speaker filter: optional name (alex / leila / sharran) restricts results to scenes
     where that speaker was active per cache/speakers/

Output: ranked (video_id, t_start, t_end, score, why) tuples suitable for
producing YouTube deep-links.

This is the product slice — what a video editor would actually invoke.

CLI:
  python scripts/hybrid_search.py "how to think like the top 1 percent"
  python scripts/hybrid_search.py "him at the whiteboard" --speaker=alex --topk=10
  python scripts/hybrid_search.py "Leila on couch dramatic gesture" --visual-only
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR  # noqa: E402

CLIP_NPZ = CACHE_DIR / "clip_frames.npz"
CLIP_META = CACHE_DIR / "clip_frames_meta.json"
DEEPGRAM_DIR = CACHE_DIR / "deepgram"
SPEAKERS_DIR = CACHE_DIR / "speakers"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def _embed_query(text: str) -> np.ndarray:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device="cpu"
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    toks = tokenizer([text])
    with torch.no_grad():
        v = model.encode_text(toks)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)[0]


def _load_clip_index() -> tuple[np.ndarray, list[dict]]:
    if not CLIP_NPZ.exists() or not CLIP_META.exists():
        raise FileNotFoundError(f"missing CLIP index: {CLIP_NPZ}")
    npz = np.load(CLIP_NPZ, allow_pickle=True)
    vectors = npz["vectors"]
    meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
    return vectors, meta


def _load_transcripts() -> dict[str, list[dict]]:
    """video_id -> list of utterances [{start, end, text, speaker?}, ...]."""
    out: dict[str, list[dict]] = {}
    for p in DEEPGRAM_DIR.glob("*.audio.json"):
        video_id = p.stem.replace(".audio", "")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            utterances = data.get("results", {}).get("utterances", [])
            if utterances:
                out[video_id] = [
                    {
                        "start": float(u.get("start", 0)),
                        "end": float(u.get("end", 0)),
                        "text": u.get("transcript") or u.get("text") or "",
                        "speaker": u.get("speaker"),
                    }
                    for u in utterances
                ]
            else:
                alt = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
                words = alt.get("words") or []
                if words:
                    out[video_id] = [{
                        "start": float(words[0]["start"]),
                        "end": float(words[-1]["end"]),
                        "text": alt.get("transcript", ""),
                        "speaker": None,
                    }]
        except Exception:
            pass
    return out


def _load_speakers() -> dict[str, dict[int, str]]:
    """video_id -> {cluster_id: name}."""
    out: dict[str, dict[int, str]] = {}
    for p in SPEAKERS_DIR.glob("*.audio.json"):
        video_id = p.stem.replace(".audio", "")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            spk = data.get("speakers", {})
            out[video_id] = {int(k): v.get("name", "unknown") for k, v in spk.items()}
        except Exception:
            pass
    return out


def _bm25lite_score(query_tokens: list[str], doc_tokens: list[str], avg_dl: float) -> float:
    """Simplified BM25 with k1=1.2, b=0.75, no IDF table."""
    if not doc_tokens:
        return 0.0
    counts = Counter(doc_tokens)
    dl = len(doc_tokens)
    k1, b = 1.2, 0.75
    score = 0.0
    for q in query_tokens:
        f = counts.get(q, 0)
        if f == 0:
            continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score


def hybrid_search(
    query: str,
    speaker_filter: str | None = None,
    topk: int = 10,
    visual_only: bool = False,
    transcript_weight: float = 0.4,
    visual_weight: float = 0.6,
) -> list[dict]:
    """Run the hybrid search. Returns ranked list."""
    print(f"loading CLIP index...", file=sys.stderr)
    vectors, meta = _load_clip_index()

    print(f"embedding query: {query!r}", file=sys.stderr)
    qv = _embed_query(query)
    visual_scores = vectors @ qv  # (N_frames,)

    transcripts = {}
    speakers_map = {}
    if not visual_only:
        print(f"loading transcripts + speakers...", file=sys.stderr)
        transcripts = _load_transcripts()
        speakers_map = _load_speakers()

    # Build per-video utterance corpus stats for the lite BM25
    avg_dl = 1.0
    if transcripts:
        all_lens = []
        for utts in transcripts.values():
            for u in utts:
                all_lens.append(len(_tokenize(u["text"])))
        if all_lens:
            avg_dl = max(1.0, sum(all_lens) / len(all_lens))
    query_tokens = _tokenize(query)

    # For each frame: combine visual score + best-overlapping-utterance transcript score
    candidates: list[tuple[float, int]] = []
    for i, m in enumerate(meta):
        vid = m["video_id"]
        v_score = float(visual_scores[i])

        # Speaker filter: skip frames whose video has no clusters matching the filter
        if speaker_filter:
            spk = speakers_map.get(vid, {})
            if speaker_filter not in spk.values():
                continue

        t_score = 0.0
        if not visual_only and vid in transcripts:
            # Frame "time" — we don't have keyframe timestamp in meta; approximate
            # by scene_idx position. Better fix is to embed scene midpoint in meta,
            # but for now we score against ALL utterances in this video and take max.
            best = 0.0
            for u in transcripts[vid]:
                s = _bm25lite_score(query_tokens, _tokenize(u["text"]), avg_dl)
                if s > best:
                    best = s
            # normalize roughly to 0–1
            t_score = min(1.0, best / 5.0)

        # Normalize visual score (CLIP cosines are in -1..1 but realistic 0.15-0.35)
        v_norm = max(0.0, min(1.0, (v_score - 0.10) / 0.30))

        if visual_only:
            score = v_norm
        else:
            score = visual_weight * v_norm + transcript_weight * t_score

        candidates.append((score, i))

    candidates.sort(key=lambda x: -x[0])
    top = candidates[:topk]
    results = []
    for score, i in top:
        m = meta[i]
        results.append({
            "rank": len(results) + 1,
            "score": round(score, 4),
            "video_id": m["video_id"],
            "scene_idx": m["scene_idx"],
            "frame_path": m["frame_path"],
            "is_intra": m.get("is_intra", False),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="text query")
    parser.add_argument("--speaker", choices=["alex", "leila", "sharran"], default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--visual-only", action="store_true")
    args = parser.parse_args()

    t0 = time.monotonic()
    results = hybrid_search(
        args.query,
        speaker_filter=args.speaker,
        topk=args.topk,
        visual_only=args.visual_only,
    )
    dt = time.monotonic() - t0
    print(f"\n=== {len(results)} results in {dt:.1f}s ===")
    for r in results:
        print(f"  #{r['rank']:2d} score={r['score']:.3f} {r['video_id']} scene={r['scene_idx']}{'*' if r['is_intra'] else ''}")
        print(f"      {r['frame_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
