"""Embed every keyframe under FRAMES_DIR with open-clip ViT-L-14 and upsert to Qdrant.

Model:  open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
Vector: 768-d, cosine.
Collection: v2_frames (enforced by SafeQdrantClient).

Payload per vector:
    {video_id, scene_idx, frame_path, is_intra}

Idempotent: re-running upserts the same point IDs (deterministic UUID5
from frame_path), so no duplicates. Local .npz cache lets the eval load
vectors without round-tripping Qdrant.

CLI:
  python scripts/embed_frames.py                       # all frames
  python scripts/embed_frames.py --sample=50           # quick smoke
  python scripts/embed_frames.py --no-qdrant           # local only
  python scripts/embed_frames.py --no-cache            # don't write npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (  # noqa: E402
    CACHE_DIR,
    FRAMES_DIR,
    LOGS_DIR,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_FRAMES,
    QDRANT_URL,
)
from qdrant_safe import open_safe  # noqa: E402

CLIP_NPZ = CACHE_DIR / "clip_frames.npz"
CLIP_META = CACHE_DIR / "clip_frames_meta.json"
LOG_PATH = LOGS_DIR / "embed_frames.log"

NAMESPACE = uuid.UUID("8a3e8b08-0e9a-4e8d-9f3d-cba1f6e21a7a")

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


def _parse_frame_path(p: Path) -> dict:
    """ingest/frames/<video_id>/scene_<NNNN>[_intra].jpg → metadata."""
    video_id = p.parent.name
    stem = p.stem
    is_intra = stem.endswith("_intra")
    idx_str = stem.replace("scene_", "").replace("_intra", "")
    return {
        "video_id": video_id,
        "scene_idx": int(idx_str),
        "frame_path": str(p.relative_to(ROOT.parent)).replace("\\", "/"),
        "is_intra": is_intra,
    }


def _gather_frames(sample: int = 0, videos: list[str] | None = None) -> list[Path]:
    if videos:
        frames: list[Path] = []
        for v in videos:
            frames.extend(sorted((FRAMES_DIR / v).rglob("*.jpg")))
    else:
        frames = sorted(FRAMES_DIR.rglob("*.jpg"))
    if sample > 0:
        frames = frames[:sample]
    return frames


def _load_clip(device: str = "cpu"):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, torch


def _embed_batch(model, preprocess, torch, paths: list[Path], device: str) -> np.ndarray:
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        images.append(preprocess(img))
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--videos", default=None, help="CSV of video ids; default all")
    parser.add_argument("--append", action="store_true", help="merge with existing npz/meta; dedupe by id")
    parser.add_argument("--no-qdrant", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLIP_NPZ.parent.mkdir(parents=True, exist_ok=True)

    videos = [v.strip() for v in args.videos.split(",")] if args.videos else None
    frames = _gather_frames(args.sample, videos)
    n = len(frames)
    if n == 0:
        print("no frames to embed", flush=True)
        return 0
    print(f"=== embed_frames: {n} frames, batch={args.batch}, qdrant={'no' if args.no_qdrant else 'yes'} ===", flush=True)

    print("loading CLIP...", flush=True)
    t0 = time.monotonic()
    device = "cpu"
    model, preprocess, torch = _load_clip(device)
    print(f"  loaded in {time.monotonic() - t0:.1f}s", flush=True)

    all_vectors: list[np.ndarray] = []
    all_meta: list[dict] = []
    all_ids: list[str] = []

    qclient = None
    if not args.no_qdrant:
        qclient = open_safe(QDRANT_URL, QDRANT_API_KEY)
        if not qclient.collection_exists(QDRANT_COLLECTION_FRAMES):
            print(f"creating collection {QDRANT_COLLECTION_FRAMES}", flush=True)
            qclient.create_collection(QDRANT_COLLECTION_FRAMES, vector_size=768)

    from qdrant_client.http import models as qmodels

    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} n={n} ===\n")
        for i in range(0, n, args.batch):
            chunk = frames[i:i + args.batch]
            t1 = time.monotonic()
            vectors = _embed_batch(model, preprocess, torch, chunk, device)
            dt = time.monotonic() - t1

            points = []
            for j, p in enumerate(chunk):
                meta = _parse_frame_path(p)
                pid = str(uuid.uuid5(NAMESPACE, meta["frame_path"]))
                all_ids.append(pid)
                all_vectors.append(vectors[j])
                all_meta.append(meta)
                points.append(qmodels.PointStruct(id=pid, vector=vectors[j].tolist(), payload=meta))

            if qclient is not None:
                qclient.upsert(QDRANT_COLLECTION_FRAMES, points)

            done = i + len(chunk)
            line = f"[{done}/{n}] batch in {dt:.2f}s ({dt/len(chunk)*1000:.0f}ms/frame)"
            print(line, flush=True)
            log.write(line + "\n")

        if not args.no_cache:
            if args.append and CLIP_NPZ.exists() and CLIP_META.exists():
                old_npz = np.load(CLIP_NPZ, allow_pickle=True)
                old_vecs = old_npz["vectors"]
                old_ids = list(old_npz["ids"])
                old_meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
                existing = set(old_ids)
                merged_ids = list(old_ids)
                merged_vecs = [old_vecs[i] for i in range(len(old_ids))]
                merged_meta = list(old_meta)
                added = 0
                for pid, vec, m in zip(all_ids, all_vectors, all_meta):
                    if pid in existing:
                        continue
                    merged_ids.append(pid)
                    merged_vecs.append(vec)
                    merged_meta.append(m)
                    added += 1
                arr = np.stack(merged_vecs).astype(np.float32)
                np.savez_compressed(CLIP_NPZ, vectors=arr, ids=np.array(merged_ids))
                CLIP_META.write_text(json.dumps(merged_meta, indent=2), encoding="utf-8")
                print(f"appended {added} new -> {CLIP_NPZ} ({arr.shape})", flush=True)
            else:
                arr = np.stack(all_vectors).astype(np.float32)
                np.savez_compressed(CLIP_NPZ, vectors=arr, ids=np.array(all_ids))
                CLIP_META.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
                print(f"wrote {CLIP_NPZ} ({arr.shape}) and {CLIP_META}", flush=True)

        summary = f"=== done n={n} ==="
        print(summary, flush=True)
        log.write(summary + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
