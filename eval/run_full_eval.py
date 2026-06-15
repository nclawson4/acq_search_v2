"""End-to-end CLIP eval against combined truth files.

Combines:
  - clip_truth_candidates.json   (210 original, 7-video corpus)
  - clip_truth_variety.json      (60 variety, 2 additional videos)
  - clip_failure_modes.json      (20 adversarial queries)

Runs full K-sweep + per-category breakdown. Reports:
  - Headline: smallest K hitting 95% on combined truth
  - Variety vs original delta
  - Failure-mode pass-rate at K=10 (gate 85%)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import CACHE_DIR  # noqa: E402

CLIP_NPZ = CACHE_DIR / "clip_frames.npz"
CLIP_META = CACHE_DIR / "clip_frames_meta.json"
DATA = ROOT / "eval" / "data"
TRUTH_FILES = {
    "original": DATA / "clip_truth_candidates.json",
    "variety": DATA / "clip_truth_variety.json",
    "failure": DATA / "clip_failure_modes.json",
}

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
CLUSTER_THRESHOLD = 0.85
K_SWEEP = [1, 5, 10, 20, 50, 100]


def _embed_texts(texts: list[str]) -> np.ndarray:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device="cpu")
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    out = []
    for i in range(0, len(texts), 16):
        chunk = texts[i:i + 16]
        toks = tokenizer(chunk)
        with torch.no_grad():
            v = model.encode_text(toks)
            v = v / v.norm(dim=-1, keepdim=True)
        out.append(v.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-threshold", type=float, default=CLUSTER_THRESHOLD)
    args = parser.parse_args()

    if not CLIP_NPZ.exists():
        print(f"missing {CLIP_NPZ}", file=sys.stderr)
        return 1

    # Load truth files
    all_entries: list[tuple[str, dict]] = []  # (split_name, entry)
    for split, path in TRUTH_FILES.items():
        if not path.exists():
            print(f"  skip missing {path.name}", file=sys.stderr)
            continue
        cands = json.loads(path.read_text(encoding="utf-8"))
        for c in cands:
            if c.get("description"):
                all_entries.append((split, c))
    print(f"=== loaded {len(all_entries)} labeled queries across {len(TRUTH_FILES)} files ===")

    # Load index
    npz = np.load(CLIP_NPZ, allow_pickle=True)
    vectors = npz["vectors"]
    meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
    print(f"distractor pool: {vectors.shape[0]} frames, {vectors.shape[1]}-d")

    # Normalize frame_path lookup
    sample_meta_path = meta[0]["frame_path"]
    prefix = "acq_search_v2/" if sample_meta_path.startswith("acq_search_v2/") else ""

    def norm(p: str) -> str:
        if prefix and not p.startswith(prefix):
            return prefix + p
        return p

    path_to_idx = {m["frame_path"]: i for i, m in enumerate(meta)}

    # Embed all texts
    texts = [e[1]["description"] for e in all_entries]
    print(f"embedding {len(texts)} texts...", flush=True)
    t0 = time.monotonic()
    text_vecs = _embed_texts(texts)
    print(f"  done in {time.monotonic() - t0:.1f}s")

    sims = text_vecs @ vectors.T          # (N_q, N_f)
    img_sims = vectors @ vectors.T        # (N_f, N_f)

    # Per-query: gold idx, cluster, K-hit?
    results = []
    missing = 0
    for q_i, (split, e) in enumerate(all_entries):
        gold_path = norm(e["frame_path"])
        gold_idx = path_to_idx.get(gold_path)
        if gold_idx is None:
            missing += 1
            continue
        order = np.argsort(-sims[q_i])
        cluster_mask = img_sims[gold_idx] >= args.cluster_threshold
        cluster_size = int(cluster_mask.sum())
        hits_at_k = {}
        for K in K_SWEEP:
            hits_at_k[K] = bool(np.any(cluster_mask[order[:K]]))
        results.append({
            "split": split,
            "video_id": e.get("video_id"),
            "category": e.get("category"),
            "cluster_size": cluster_size,
            "hits": hits_at_k,
        })

    if missing:
        print(f"  WARN: {missing} queries have frames not in CLIP index (skipped)")

    print(f"\nCLUSTER threshold: {args.cluster_threshold}")

    # Combined K-sweep
    print("\n=== COMBINED K-SWEEP ===")
    for K in K_SWEEP:
        passes = sum(1 for r in results if r["hits"][K])
        n = len(results)
        print(f"  K={K:4d}: {passes/n:.3f} ({passes}/{n})")

    # Per-split breakdown
    print("\n=== PER SPLIT ===")
    for split in TRUTH_FILES.keys():
        subset = [r for r in results if r["split"] == split]
        if not subset:
            continue
        print(f"\n  {split} (n={len(subset)}):")
        for K in K_SWEEP:
            passes = sum(1 for r in subset if r["hits"][K])
            print(f"    K={K:4d}: {passes/len(subset):.3f} ({passes}/{len(subset)})")

    # Failure mode categories
    failure_subset = [r for r in results if r["split"] == "failure"]
    if failure_subset:
        print("\n=== FAILURE MODE BREAKDOWN (K=10) ===")
        from collections import defaultdict
        by_cat = defaultdict(list)
        for r in failure_subset:
            by_cat[r.get("category") or "unk"].append(r["hits"][10])
        for cat, hits in sorted(by_cat.items()):
            passes = sum(hits)
            print(f"  {cat:14s}: {passes/len(hits):.3f} ({passes}/{len(hits)})")

    # Headline: smallest K hitting 95% on combined
    print("\n=== HEADLINE ===")
    for K in K_SWEEP:
        passes = sum(1 for r in results if r["hits"][K])
        rate = passes / len(results)
        if rate >= 0.95:
            print(f"  combined recall hits 95% at K={K} (actual {rate:.3f})")
            break
    else:
        print(f"  combined recall does NOT reach 95% at any tested K (max K={K_SWEEP[-1]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
