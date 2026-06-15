"""Hybrid retrieval eval: CLIP visual + Deepgram transcript BM25-lite + speaker filter.

Re-runs the same 290 truth queries (original 210 + variety 60 + failure 20)
against the full 15,621-frame distractor pool — but now scoring each frame
with a HYBRID weighted combination of three signals.

Scoring (per frame):
  score = w_visual * clip_cos_normalized
        + w_transcript * bm25_normalized
  (speaker filter is per-query, optional; truth queries don't use it)

Transcript is scene-localized: for each frame, find utterances whose
[start, end] overlaps the frame's scene [start_s, end_s] from
cache/scenes/<video_id>.json, then take max BM25 score over those.

Reports:
  - Headline: smallest K hitting 95% combined recall
  - Per-split (original / variety / failure)
  - Per-failure-category at K=10
  - Weight sweep (optional via --sweep)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import CACHE_DIR  # noqa: E402

CLIP_NPZ = CACHE_DIR / "clip_frames.npz"
CLIP_META = CACHE_DIR / "clip_frames_meta.json"
SCENES_DIR = CACHE_DIR / "scenes"
DEEPGRAM_DIR = CACHE_DIR / "deepgram"
SPEAKERS_DIR = CACHE_DIR / "speakers"
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

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


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


def _load_scenes() -> dict[str, dict[int, tuple[float, float]]]:
    """video_id -> {scene_idx: (start_s, end_s)}"""
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for p in SCENES_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            vid = data["video_id"]
            out[vid] = {s["idx"]: (s["start_s"], s["end_s"]) for s in data["scenes"]}
        except Exception:
            pass
    return out


def _load_utterances() -> dict[str, list[dict]]:
    """video_id -> [{start, end, text, speaker, tokens}, ...] sorted by start."""
    out: dict[str, list[dict]] = {}
    for p in DEEPGRAM_DIR.glob("*.audio.json"):
        vid = p.stem.replace(".audio", "")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            utterances = data.get("results", {}).get("utterances", [])
            if not utterances:
                continue
            utts = []
            for u in utterances:
                text = u.get("transcript") or u.get("text") or ""
                if not text or not isinstance(text, str):
                    continue
                utts.append({
                    "start": float(u.get("start", 0)),
                    "end": float(u.get("end", 0)),
                    "text": text,
                    "tokens": _tokenize(text),
                    "speaker": u.get("speaker"),
                })
            utts.sort(key=lambda u: u["start"])
            if utts:
                out[vid] = utts
        except Exception:
            pass
    return out


def _load_speakers() -> dict[str, dict[int, str]]:
    """video_id -> {cluster_id: name}"""
    out: dict[str, dict[int, str]] = {}
    for p in SPEAKERS_DIR.glob("*.audio.json"):
        vid = p.stem.replace(".audio", "")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            spk = data.get("speakers", {})
            out[vid] = {int(k): v.get("name", "unknown") for k, v in spk.items()}
        except Exception:
            pass
    return out


def _bm25lite(qtoks: list[str], dtoks: list[str], avg_dl: float, k1: float = 1.2, b: float = 0.75) -> float:
    if not dtoks:
        return 0.0
    counts = Counter(dtoks)
    dl = len(dtoks)
    score = 0.0
    for q in qtoks:
        f = counts.get(q, 0)
        if f == 0:
            continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-threshold", type=float, default=CLUSTER_THRESHOLD)
    parser.add_argument("--w-visual", type=float, default=0.6)
    parser.add_argument("--w-transcript", type=float, default=0.4)
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="sweep weights over a small grid")
    parser.add_argument("--utt-pad-s", type=float, default=0.0, help="seconds of padding around scene window when picking utterances")
    args = parser.parse_args()

    if not CLIP_NPZ.exists():
        print(f"missing {CLIP_NPZ}", file=sys.stderr)
        return 1

    # Load truth files
    all_entries: list[tuple[str, dict]] = []
    for split, path in TRUTH_FILES.items():
        if not path.exists():
            continue
        cands = json.loads(path.read_text(encoding="utf-8"))
        for c in cands:
            if c.get("description"):
                all_entries.append((split, c))
    print(f"=== loaded {len(all_entries)} labeled queries ===", flush=True)

    # Load CLIP index
    npz = np.load(CLIP_NPZ, allow_pickle=True)
    vectors = npz["vectors"]
    meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
    N = vectors.shape[0]
    print(f"distractor pool: {N} frames, {vectors.shape[1]}-d", flush=True)

    # path-prefix normalization
    sample_meta_path = meta[0]["frame_path"]
    prefix = "acq_search_v2/" if sample_meta_path.startswith("acq_search_v2/") else ""
    def norm(p: str) -> str:
        return prefix + p if prefix and not p.startswith(prefix) else p
    path_to_idx = {m["frame_path"]: i for i, m in enumerate(meta)}

    # Load scenes (for per-frame timestamps) and utterances
    print("loading scenes + transcripts...", flush=True)
    scenes = _load_scenes()
    utterances = _load_utterances()
    speakers = _load_speakers()
    print(f"  scenes: {len(scenes)} videos, utterances: {len(utterances)} videos, speakers: {len(speakers)} videos", flush=True)

    # Precompute per-frame: scene start/end (s) and overlapping utterance indices in this video
    frame_scene_window: list[tuple[float, float] | None] = [None] * N
    frame_utt_ranges: list[tuple[int, int] | None] = [None] * N  # [lo, hi) into video's utt list
    for i, m in enumerate(meta):
        vid = m["video_id"]
        si = m["scene_idx"]
        sw = scenes.get(vid, {}).get(si)
        if sw is None:
            continue
        frame_scene_window[i] = sw
        utts = utterances.get(vid)
        if not utts:
            continue
        # pad the scene window by --utt-pad-s seconds on each side to grab surrounding context
        start_s = max(0.0, sw[0] - args.utt_pad_s)
        end_s = sw[1] + args.utt_pad_s
        lo = 0
        hi = len(utts)
        # find first utterance with end >= start_s
        l, r = 0, len(utts)
        while l < r:
            mid = (l + r) // 2
            if utts[mid]["end"] < start_s:
                l = mid + 1
            else:
                r = mid
        lo = l
        # walk forward while utt.start <= end_s
        hi = lo
        while hi < len(utts) and utts[hi]["start"] <= end_s:
            hi += 1
        frame_utt_ranges[i] = (lo, hi)

    # avg doc length across all utterances (for BM25 normalization)
    all_lens = []
    for utts in utterances.values():
        for u in utts:
            all_lens.append(len(u["tokens"]))
    avg_dl = max(1.0, sum(all_lens) / len(all_lens)) if all_lens else 1.0

    # Embed all query texts
    texts = [e[1]["description"] for e in all_entries]
    print(f"embedding {len(texts)} queries...", flush=True)
    t0 = time.monotonic()
    text_vecs = _embed_texts(texts)
    print(f"  done in {time.monotonic() - t0:.1f}s", flush=True)

    # Visual scores: (N_q, N_f)
    visual_scores = text_vecs @ vectors.T

    # Pre-image-image sims for cluster expansion
    img_sims = vectors @ vectors.T

    def normalize_visual(v: float) -> float:
        return max(0.0, min(1.0, (v - 0.10) / 0.30))

    # Vectorize visual normalization
    vis_norm = np.clip((visual_scores - 0.10) / 0.30, 0.0, 1.0)

    def eval_with(w_visual: float, w_transcript: float) -> dict:
        results = []
        for q_i, (split, e) in enumerate(all_entries):
            qtoks = _tokenize(e["description"])
            gold_path = norm(e["frame_path"])
            gold_idx = path_to_idx.get(gold_path)
            if gold_idx is None:
                continue

            # Compute transcript score per frame for this query.
            # Cache per (video_id, utterance_idx) within this query to avoid re-tokenizing
            video_max = {}
            t_per_frame = np.zeros(N, dtype=np.float32)
            # Precompute utt score per (video_id, utt_idx) lazily
            utt_score_cache: dict[tuple[str, int], float] = {}
            for f_i, m in enumerate(meta):
                if w_transcript == 0:
                    break
                rng = frame_utt_ranges[f_i]
                if rng is None:
                    continue
                vid = m["video_id"]
                utts = utterances.get(vid)
                if not utts:
                    continue
                lo, hi = rng
                best = 0.0
                for uj in range(lo, hi):
                    key = (vid, uj)
                    s = utt_score_cache.get(key)
                    if s is None:
                        s = _bm25lite(qtoks, utts[uj]["tokens"], avg_dl)
                        utt_score_cache[key] = s
                    if s > best:
                        best = s
                t_per_frame[f_i] = min(1.0, best / 5.0)

            scores = w_visual * vis_norm[q_i] + w_transcript * t_per_frame
            order = np.argsort(-scores)
            cluster_mask = img_sims[gold_idx] >= args.cluster_threshold
            hits = {K: bool(np.any(cluster_mask[order[:K]])) for K in K_SWEEP}
            results.append({
                "split": split,
                "category": e.get("category"),
                "hits": hits,
            })
        return {"results": results}

    if args.sweep:
        weight_grid = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.0, 1.0)]
        print("\n=== WEIGHT SWEEP ===")
        print("w_visual w_transcript  | K=1   K=5   K=10  K=20  K=50  K=100")
        for wv, wt in weight_grid:
            r = eval_with(wv, wt)["results"]
            n = len(r)
            rates = []
            for K in K_SWEEP:
                passes = sum(1 for x in r if x["hits"][K])
                rates.append(passes / n)
            print(f"{wv:0.2f}     {wt:0.2f}        | " + " ".join(f"{rt:0.3f}" for rt in rates))
        return 0

    if args.visual_only:
        wv, wt = 1.0, 0.0
    else:
        wv, wt = args.w_visual, args.w_transcript

    print(f"\nweights: visual={wv} transcript={wt}", flush=True)
    print("running hybrid eval...", flush=True)
    t1 = time.monotonic()
    r = eval_with(wv, wt)["results"]
    print(f"  done in {time.monotonic() - t1:.1f}s", flush=True)
    n = len(r)

    print(f"\nCLUSTER threshold: {args.cluster_threshold}")
    print(f"\n=== COMBINED K-SWEEP (n={n}) ===")
    for K in K_SWEEP:
        passes = sum(1 for x in r if x["hits"][K])
        print(f"  K={K:4d}: {passes/n:.3f} ({passes}/{n})")

    print("\n=== PER SPLIT ===")
    for split in TRUTH_FILES.keys():
        subset = [x for x in r if x["split"] == split]
        if not subset:
            continue
        print(f"\n  {split} (n={len(subset)}):")
        for K in K_SWEEP:
            passes = sum(1 for x in subset if x["hits"][K])
            print(f"    K={K:4d}: {passes/len(subset):.3f} ({passes}/{len(subset)})")

    failure_subset = [x for x in r if x["split"] == "failure"]
    if failure_subset:
        print("\n=== FAILURE MODE BREAKDOWN (K=10) ===")
        by_cat = defaultdict(list)
        for x in failure_subset:
            by_cat[x.get("category") or "unk"].append(x["hits"][10])
        for cat, hits in sorted(by_cat.items()):
            passes = sum(hits)
            print(f"  {cat:14s}: {passes/len(hits):.3f} ({passes}/{len(hits)})")

    print("\n=== HEADLINE ===")
    for K in K_SWEEP:
        passes = sum(1 for x in r if x["hits"][K])
        rate = passes / n
        if rate >= 0.95:
            print(f"  combined recall hits 95% at K={K} (actual {rate:.3f})")
            break
    else:
        print(f"  combined recall does NOT reach 95% at any tested K (max K={K_SWEEP[-1]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
