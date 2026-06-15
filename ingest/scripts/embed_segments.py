"""CLIP-text-embed every topic_segment's text fields.

Input:  cache/topic_segments.json
Output: cache/segment_text_embeddings.npz   (vectors + per-row metadata)
        cache/segment_text_meta.json        (lookup info)

The query-side (lib/hybrid.py) computes its own query vector with the same
encoder and dots against these — semantic match without synonym dictionaries.

Each segment produces ONE vector that is the mean of:
  - topic_title
  - lessons_summary
  - each expected_query

We mean-pool because they're describing the same segment from different angles.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR  # noqa: E402

SEGMENTS_PATH = CACHE_DIR / "topic_segments.json"
OUT_NPZ = CACHE_DIR / "segment_text_embeddings.npz"
OUT_META = CACHE_DIR / "segment_text_meta.json"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


def _load_clip():
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device="cpu")
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    return model, tokenizer, torch


def main() -> int:
    if not SEGMENTS_PATH.exists():
        print(f"missing {SEGMENTS_PATH}", file=sys.stderr)
        return 1
    data = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    segs = data if isinstance(data, list) else data.get("segments", [])
    if not segs:
        print("no segments", file=sys.stderr)
        return 1

    print(f"=== embedding {len(segs)} segments ===")
    model, tokenizer, torch = _load_clip()

    # For each segment, embed each text field separately, then mean-pool.
    vectors = []
    meta = []
    t0 = time.monotonic()
    BATCH = 8
    flat_texts: list[str] = []
    flat_owners: list[int] = []  # segment index each text belongs to

    for i, s in enumerate(segs):
        fields = [s.get("topic_title", ""), s.get("lessons_summary", "")]
        for q in s.get("expected_queries", []):
            if q:
                fields.append(q)
        # also include short summary if non-empty
        if s.get("summary"):
            fields.append(s["summary"])
        for t in fields:
            t = (t or "").strip()
            if t:
                flat_texts.append(t)
                flat_owners.append(i)

    # Encode in batches
    field_vecs: list[np.ndarray] = []
    for start in range(0, len(flat_texts), BATCH):
        chunk = flat_texts[start:start + BATCH]
        toks = tokenizer(chunk)
        with torch.no_grad():
            v = model.encode_text(toks)
            v = v / v.norm(dim=-1, keepdim=True)
        field_vecs.append(v.cpu().numpy().astype(np.float32))
        if (start // BATCH) % 10 == 0:
            print(f"  embedded {start + len(chunk)}/{len(flat_texts)} fields...", flush=True)
    field_vecs_arr = np.concatenate(field_vecs, axis=0) if field_vecs else np.zeros((0, 768), dtype=np.float32)

    # Mean-pool per segment
    seg_vecs = np.zeros((len(segs), 768), dtype=np.float32)
    counts = np.zeros(len(segs), dtype=np.int32)
    for idx, owner in enumerate(flat_owners):
        seg_vecs[owner] += field_vecs_arr[idx]
        counts[owner] += 1
    nonzero = counts > 0
    seg_vecs[nonzero] /= counts[nonzero, None]
    # Re-normalize (mean of unit vectors isn't unit)
    norms = np.linalg.norm(seg_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    seg_vecs /= norms

    for i, s in enumerate(segs):
        meta.append({
            "video_id": s["video_id"],
            "segment_idx": s["segment_idx"],
            "start_s": s["start_s"],
            "end_s": s["end_s"],
        })

    np.savez_compressed(OUT_NPZ, vectors=seg_vecs)
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT_NPZ} ({seg_vecs.shape}) and {OUT_META} in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
