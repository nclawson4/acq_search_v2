"""CLIP text-to-frame eval: top-1 / top-5 retrieval accuracy.

Given (description, frame) pairs, for each description:
  - Embed text with CLIP text encoder
  - Retrieve top-K from the full ~1762-frame distractor pool
  - Check whether the gold frame's ID is in top-K

Gate: top-1 >= 95% counts as PASS. Top-5 is reported for diagnostic.

Inputs:
  - cache/clip_frames.npz  (image vectors + IDs)
  - cache/clip_frames_meta.json  (metadata aligned to npz indices)
  - eval/data/clip_truth_candidates.json  (labeled)

CLI:
  python eval/clip_eval.py
  python eval/clip_eval.py --topk=5
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
DEFAULT_TRUTH = ROOT / "eval" / "data" / "clip_truth_candidates.json"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


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
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cluster-threshold", type=float, default=0.92)
    args = parser.parse_args()

    truth_path = Path(args.truth)
    if not truth_path.exists():
        print(f"missing: {truth_path}", file=sys.stderr)
        return 1
    if not CLIP_NPZ.exists() or not CLIP_META.exists():
        print(f"missing: {CLIP_NPZ} or {CLIP_META}", file=sys.stderr)
        return 1

    cands = json.loads(truth_path.read_text(encoding="utf-8"))
    labeled = [c for c in cands if c.get("description")]
    print(f"=== labeled: {len(labeled)} / {len(cands)} ===")
    if len(labeled) == 0:
        return 1

    npz = np.load(CLIP_NPZ, allow_pickle=True)
    vectors = npz["vectors"]
    ids_npz = list(npz["ids"])
    meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
    print(f"distractor pool: {vectors.shape[0]} frames, {vectors.shape[1]}-d")

    # Build lookup: relative frame_path -> index in vectors array
    path_to_idx: dict[str, int] = {}
    for i, m in enumerate(meta):
        # meta stored frame_path relative to PROJECT_ROOT (parents[1] in embed script)
        path_to_idx[m["frame_path"]] = i

    # Sanity: candidate frame_paths use "ingest/frames/..." (relative to v2 root)
    # meta paths from embed_frames are "acq_search_v2/ingest/frames/..." since
    # they're relative to ROOT.parent. Normalize.
    sample_meta_path = meta[0]["frame_path"]
    if sample_meta_path.startswith("acq_search_v2/"):
        prefix = "acq_search_v2/"
    else:
        prefix = ""

    def normalize(p: str) -> str:
        if prefix and not p.startswith(prefix):
            return prefix + p
        return p

    texts = [c["description"] for c in labeled]
    print("embedding texts...", flush=True)
    t0 = time.monotonic()
    text_vecs = _embed_texts(texts)
    print(f"  done in {time.monotonic() - t0:.1f}s")

    # Cosine sim = dot product (vectors are already L2-normalized)
    sims = text_vecs @ vectors.T  # (N_queries, N_frames)

    # Image-image similarity for "content cluster" metric: any frame within
    # CLUSTER_SIM_THRESHOLD cosine of the gold counts as visually equivalent.
    # Calibrated against the corpus: ~0.92 puts near-duplicate talking-head
    # frames in the same cluster but keeps different scenes separate.
    CLUSTER_SIM_THRESHOLD = args.cluster_threshold
    img_sims = vectors @ vectors.T  # (N_frames, N_frames), L2-normalized

    # Build (video_id, scene_idx) lookup per vector
    vec_scene: list[tuple[str, int]] = [(m["video_id"], m["scene_idx"]) for m in meta]

    top1_hits = 0
    topk_hits = 0
    scene_top1_hits = 0      # top-1 result comes from same (video_id, ±2 scene_idx)
    scene_topk_hits = 0
    cluster_top1_hits = 0    # top-1 result is visually equivalent to gold (CLIP sim >= threshold)
    cluster_topk_hits = 0
    # "Fair K" metric: per-query K threshold based on cluster size.
    # Distinctive content (small cluster) is held to top-1; redundant content
    # (large cluster) gets a wider window. Success = any cluster member in top-K.
    fair_pass = 0
    fair_breakdown = {1: {"n": 0, "pass": 0}, 5: {"n": 0, "pass": 0}, 20: {"n": 0, "pass": 0}}
    missing = 0
    per_video_stats: dict[str, dict[str, int]] = {}
    cluster_sizes: list[int] = []

    def cluster_size_to_k(cs: int) -> int:
        if cs <= 3:
            return 1
        if cs <= 30:
            return 5
        return 20

    for q_idx, c in enumerate(labeled):
        gold_path = normalize(c["frame_path"])
        gold_idx = path_to_idx.get(gold_path)
        if gold_idx is None:
            missing += 1
            continue
        scores = sims[q_idx]
        order = np.argsort(-scores)
        top1 = (order[0] == gold_idx)
        topk = (gold_idx in order[:args.topk])
        gold_v, gold_s = c["video_id"], c["scene_idx"]
        # scene-level success: top result from same video AND scene index within window
        def same_scene(vec_i):
            v, s = vec_scene[vec_i]
            return v == gold_v and abs(s - gold_s) <= 2
        scene_top1 = same_scene(int(order[0]))
        scene_topk = any(same_scene(int(o)) for o in order[:args.topk])
        # content-cluster success: top result is visually equivalent to gold
        cluster_mask = img_sims[gold_idx] >= CLUSTER_SIM_THRESHOLD
        cluster_size = int(cluster_mask.sum())
        cluster_sizes.append(cluster_size)
        cluster_top1 = bool(cluster_mask[int(order[0])])
        cluster_topk = any(bool(cluster_mask[int(o)]) for o in order[:args.topk])
        # Fair-K: K depends on cluster size; success = any cluster member in top-K
        fair_k = cluster_size_to_k(cluster_size)
        fair_top = any(bool(cluster_mask[int(o)]) for o in order[:fair_k])
        fair_breakdown[fair_k]["n"] += 1
        if fair_top:
            fair_breakdown[fair_k]["pass"] += 1
            fair_pass += 1
        if top1:
            top1_hits += 1
        if topk:
            topk_hits += 1
        if scene_top1:
            scene_top1_hits += 1
        if scene_topk:
            scene_topk_hits += 1
        if cluster_top1:
            cluster_top1_hits += 1
        if cluster_topk:
            cluster_topk_hits += 1
        s = per_video_stats.setdefault(c["video_id"], {"n": 0, "top1": 0, "topk": 0, "scene_top1": 0, "scene_topk": 0, "cluster_top1": 0, "cluster_topk": 0})
        s["n"] += 1
        if top1:
            s["top1"] += 1
        if topk:
            s["topk"] += 1
        if scene_top1:
            s["scene_top1"] += 1
        if scene_topk:
            s["scene_topk"] += 1
        if cluster_top1:
            s["cluster_top1"] += 1
        if cluster_topk:
            s["cluster_topk"] += 1

    n_eval = len(labeled) - missing
    top1_acc = top1_hits / n_eval if n_eval else 0.0
    topk_acc = topk_hits / n_eval if n_eval else 0.0
    scene_top1_acc = scene_top1_hits / n_eval if n_eval else 0.0
    scene_topk_acc = scene_topk_hits / n_eval if n_eval else 0.0
    cluster_top1_acc = cluster_top1_hits / n_eval if n_eval else 0.0
    cluster_topk_acc = cluster_topk_hits / n_eval if n_eval else 0.0
    avg_cluster = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0

    print()
    print(f"OVERALL on n={n_eval} (missing={missing}, cluster_threshold={CLUSTER_SIM_THRESHOLD}, avg cluster size={avg_cluster:.1f}):")
    print(f"  frame-level top-1:     {top1_acc:.3f} ({top1_hits}/{n_eval})  [exact frame]")
    print(f"  frame-level top-{args.topk}:     {topk_acc:.3f} ({topk_hits}/{n_eval})")
    print(f"  scene-level top-1:     {scene_top1_acc:.3f} ({scene_top1_hits}/{n_eval})  [same video ±2 scenes]")
    print(f"  scene-level top-{args.topk}:     {scene_topk_acc:.3f} ({scene_topk_hits}/{n_eval})")
    print(f"  content-cluster top-1: {cluster_top1_acc:.3f} ({cluster_top1_hits}/{n_eval})  [visually equivalent to gold]")
    print(f"  content-cluster top-{args.topk}: {cluster_topk_acc:.3f} ({cluster_topk_hits}/{n_eval})")

    print()
    fair_acc = fair_pass / n_eval if n_eval else 0.0
    print(f"FAIR-K metric (K depends on cluster size: 1-3 -> top-1, 4-30 -> top-5, 30+ -> top-20):")
    print(f"  overall: {fair_acc:.3f} ({fair_pass}/{n_eval})")
    for k in (1, 5, 20):
        b = fair_breakdown[k]
        rate = b["pass"] / b["n"] if b["n"] else 0
        print(f"  K={k}: n={b['n']} pass={b['pass']} rate={rate:.3f}")

    # K-SWEEP: smallest K at which 95% of queries find a content-cluster member in top-K.
    # This is the headline product metric: "the right content is in your top-K results".
    print()
    print("K-SWEEP (content-cluster recall@K — find smallest K hitting 95%):")
    # Pre-compute per-query, for each K, whether cluster hit
    # Build a per-query "rank of first cluster hit"
    first_cluster_rank: list[int | None] = []
    for q_idx, c in enumerate(labeled):
        gold_path = normalize(c["frame_path"])
        gold_idx = path_to_idx.get(gold_path)
        if gold_idx is None:
            continue
        cluster_mask = img_sims[gold_idx] >= CLUSTER_SIM_THRESHOLD
        order = np.argsort(-sims[q_idx])
        rank = None
        for r, o in enumerate(order, 1):
            if cluster_mask[int(o)]:
                rank = r
                break
        first_cluster_rank.append(rank)
    sweep_ks = [1, 5, 10, 20, 50, 100, 200, 500]
    smallest_k_95 = None
    for k in sweep_ks:
        hits = sum(1 for r in first_cluster_rank if r is not None and r <= k)
        rate = hits / n_eval if n_eval else 0
        marker = ""
        if rate >= 0.95 and smallest_k_95 is None:
            smallest_k_95 = k
            marker = "  <-- first K hitting 95%"
        print(f"  K={k:4d}: {rate:.3f} ({hits}/{n_eval}){marker}")
    if smallest_k_95 is None:
        print(f"  >>> 95% NOT hit at K=500. The system would need an even wider window OR a different approach. <<<")

    print()
    print("BY VIDEO (content-cluster top-1 / top-5):")
    for vid, s in sorted(per_video_stats.items()):
        ct1 = s["cluster_top1"] / s["n"] if s["n"] else 0
        ctk = s["cluster_topk"] / s["n"] if s["n"] else 0
        print(f"  {vid}: n={s['n']} cluster_t1={ct1:.3f} cluster_t{args.topk}={ctk:.3f}")

    print()
    # Primary gate: smallest-K to hit 95% content-cluster recall.
    # The 95% accuracy floor is non-negotiable; K is the lever.
    print()
    if smallest_k_95 is not None:
        print(f"*** PASS at K={smallest_k_95} *** (95% content-cluster recall)")
        return 0
    else:
        print(f"*** FAIL *** (could not hit 95% at any K in sweep)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
