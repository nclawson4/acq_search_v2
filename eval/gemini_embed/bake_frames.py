"""One-time bake: embed every keyframe with Gemini Embedding 2 and save as
ingest/cache/gemini_frames_<DIM>.npz aligned to ingest/cache/clip_frames_meta.json.

The output schema mirrors clip_frames.npz so eval/gemini_embed/compare.py can
swap one in for the other without touching retrieval code.

Run:
  python -m eval.gemini_embed.bake_frames --dim 768
  python -m eval.gemini_embed.bake_frames --dim 3072

Resumable: re-running picks up where it left off using a per-dim checkpoint
file. Rate-limit retries with exponential backoff. Concurrency tunable but
defaults to 12 in-flight which is safe under standard tier RPM.
"""
from __future__ import annotations

import argparse
import asyncio
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

FRAMES_DIR = ROOT / "ingest" / "frames"
META_PATH = CACHE_DIR / "clip_frames_meta.json"
MODEL = "gemini-embedding-2"

CKPT_DIR = ROOT / "eval" / "gemini_embed" / "results"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _ckpt_path(dim: int) -> Path:
    return CKPT_DIR / f"frames_ckpt_{dim}.npy"


def _out_path(dim: int) -> Path:
    return CACHE_DIR / f"gemini_frames_{dim}.npz"


async def _embed_one(client, path: Path, dim: int, sem: asyncio.Semaphore) -> np.ndarray:
    from google.genai import types  # type: ignore
    data = path.read_bytes()
    part = types.Part.from_bytes(data=data, mime_type="image/jpeg")
    delay = 0.5
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.aio.models.embed_content(
                    model=MODEL,
                    contents=[part],
                    config=types.EmbedContentConfig(output_dimensionality=dim),
                )
                v = np.asarray(resp.embeddings[0].values, dtype=np.float32)
                # L2-normalize so cosine == dot product later
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                return v
            except Exception as e:
                msg = str(e).lower()
                if "resource_exhausted" in msg or "429" in msg or "rate" in msg:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                if attempt < 5:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise


async def main_async(dim: int, concurrency: int) -> None:
    from google import genai  # type: ignore
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in .env")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    n = len(meta)
    print(f"baking {n} frames at dim={dim} with concurrency={concurrency}")

    out = np.zeros((n, dim), dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    ckpt = _ckpt_path(dim)
    if ckpt.exists():
        try:
            saved = np.load(ckpt, allow_pickle=False)
            if saved.shape == (n, dim + 1):
                out = saved[:, :dim].astype(np.float32, copy=False)
                done = saved[:, dim].astype(bool)
                print(f"  resumed from checkpoint: {int(done.sum())}/{n} already embedded")
        except Exception as e:
            print(f"  ckpt unreadable ({e}); starting fresh")

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    started = time.time()
    saved_at = 0

    async def worker(i: int):
        nonlocal saved_at
        m = meta[i]
        # frame_path is repo-relative like "ingest/frames/<vid>/scene_NNNN.jpg"
        p = ROOT / m["frame_path"]
        try:
            v = await _embed_one(client, p, dim, sem)
            out[i] = v
            done[i] = True
        except Exception as e:
            print(f"  FAIL i={i} {p.name}: {e}")

    pending = [i for i in range(n) if not done[i]]
    tasks = [asyncio.create_task(worker(i)) for i in pending]
    completed = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        if completed % 250 == 0:
            elapsed = time.time() - started
            rate = completed / max(0.001, elapsed)
            eta = (len(pending) - completed) / max(0.001, rate)
            n_done = int(done.sum())
            print(f"  {n_done}/{n}  ({rate:.1f} emb/s, eta {eta/60:.1f} min)", flush=True)
            # Save checkpoint every 1000 embeddings
            if n_done - saved_at >= 1000:
                stacked = np.concatenate([out, done.astype(np.float32)[:, None]], axis=1)
                np.save(ckpt, stacked)
                saved_at = n_done

    # Final save
    stacked = np.concatenate([out, done.astype(np.float32)[:, None]], axis=1)
    np.save(ckpt, stacked)
    n_done = int(done.sum())
    if n_done < n:
        print(f"WARN: {n - n_done} frames failed and were skipped")

    np.savez_compressed(_out_path(dim), vectors=out, done=done)
    print(f"wrote {_out_path(dim)} (shape {out.shape}, success {n_done}/{n})")
    print(f"total wall time: {(time.time() - started)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True, choices=[768, 3072])
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    asyncio.run(main_async(args.dim, args.concurrency))


if __name__ == "__main__":
    main()
