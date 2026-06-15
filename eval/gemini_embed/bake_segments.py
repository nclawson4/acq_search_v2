"""Bake segment-text embeddings with Gemini, aligned to ingest/cache/segment_text_meta.json.

Output: ingest/cache/gemini_segments_<DIM>.npz with the same row order as the CLIP
segment_text_embeddings.npz so compare.py can swap them directly.

Embeds the same composite text the CLIP path uses: topic_title + lessons_summary +
expected_queries + summary. Cheap: 636 short texts at $0.20/1M tokens is rounding error.

Run:
  python -m eval.gemini_embed.bake_segments --dim 768
  python -m eval.gemini_embed.bake_segments --dim 3072
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from config import CACHE_DIR  # noqa: E402

SEG_META = CACHE_DIR / "segment_text_meta.json"
TOPIC_SEGMENTS = CACHE_DIR / "topic_segments.json"
MODEL = "gemini-embedding-2"


def _out_path(dim: int) -> Path:
    return CACHE_DIR / f"gemini_segments_{dim}.npz"


def _segment_text(seg: dict) -> str:
    """Mirror the CLIP path: mean-pool of topic_title, lessons_summary,
    each expected_query, summary. We collapse to a single string here and let
    Gemini do the embedding — equivalent up to ordering."""
    parts = []
    if seg.get("topic_title"): parts.append(str(seg["topic_title"]))
    if seg.get("lessons_summary"): parts.append(str(seg["lessons_summary"]))
    for q in (seg.get("expected_queries") or []):
        if q: parts.append(str(q))
    if seg.get("summary"): parts.append(str(seg["summary"]))
    return "\n".join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True, choices=[768, 3072])
    args = ap.parse_args()

    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    meta = json.loads(SEG_META.read_text(encoding="utf-8"))
    # meta.json maps (video_id, segment_idx) -> row; we need the segments themselves
    topic_segments = json.loads(TOPIC_SEGMENTS.read_text(encoding="utf-8"))
    by_key = {(s["video_id"], s["segment_idx"]): s for s in topic_segments}

    # meta is a list of [video_id, segment_idx] in row order
    keys = [tuple(k) for k in meta] if isinstance(meta, list) else [tuple(k) for k in meta["keys"]]

    n = len(keys)
    print(f"baking {n} segment texts at dim={args.dim}")

    client = genai.Client(api_key=api_key)
    out = np.zeros((n, args.dim), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, key in enumerate(keys):
        seg = by_key.get(key)
        if seg is None:
            print(f"  WARN no segment for {key}")
            continue
        text = _segment_text(seg)
        if not text:
            continue
        delay = 0.5
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=args.dim),
                )
                v = np.asarray(resp.embeddings[0].values, dtype=np.float32)
                nrm = np.linalg.norm(v)
                if nrm > 0: v = v / nrm
                out[i] = v
                done[i] = True
                break
            except Exception as e:
                if attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 20.0)
                else:
                    print(f"  FAIL i={i} {key}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}  ({(i+1)/(time.time()-t0):.1f} emb/s)", flush=True)

    np.savez_compressed(_out_path(args.dim), vectors=out, done=done)
    print(f"wrote {_out_path(args.dim)} (shape {out.shape}, success {int(done.sum())}/{n})")


if __name__ == "__main__":
    main()
