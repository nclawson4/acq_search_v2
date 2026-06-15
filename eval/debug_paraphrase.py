"""Debug which paraphrase queries are failing at K=10."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from config import CACHE_DIR

CLIP_NPZ = CACHE_DIR / "clip_frames.npz"
CLIP_META = CACHE_DIR / "clip_frames_meta.json"
SCENES_DIR = CACHE_DIR / "scenes"
DEEPGRAM_DIR = CACHE_DIR / "deepgram"
DATA = ROOT / "eval" / "data"

import open_clip
import torch

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
TOKEN_RE = re.compile(r"[a-z0-9']+")

def tokenize(s): return TOKEN_RE.findall(s.lower())

def bm25(qtoks, dtoks, avg_dl, k1=1.2, b=0.75):
    if not dtoks: return 0.0
    counts = Counter(dtoks)
    dl = len(dtoks)
    score = 0.0
    for q in qtoks:
        f = counts.get(q, 0)
        if f == 0: continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score

# Load everything
npz = np.load(CLIP_NPZ, allow_pickle=True)
vectors = npz["vectors"]
meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
N = len(meta)
sample = meta[0]["frame_path"]
prefix = "acq_search_v2/" if sample.startswith("acq_search_v2/") else ""
def norm(p): return prefix + p if prefix and not p.startswith(prefix) else p
path_to_idx = {m["frame_path"]: i for i, m in enumerate(meta)}

scenes = {}
for p in SCENES_DIR.glob("*.json"):
    d = json.loads(p.read_text(encoding="utf-8"))
    scenes[d["video_id"]] = {s["idx"]: (s["start_s"], s["end_s"]) for s in d["scenes"]}

utterances = {}
for p in DEEPGRAM_DIR.glob("*.audio.json"):
    vid = p.stem.replace(".audio", "")
    d = json.loads(p.read_text(encoding="utf-8"))
    utts = d.get("results", {}).get("utterances", [])
    out = []
    for u in utts:
        text = u.get("transcript") or u.get("text") or ""
        if text:
            out.append({"start": float(u.get("start", 0)), "end": float(u.get("end", 0)),
                        "text": text, "tokens": tokenize(text)})
    out.sort(key=lambda u: u["start"])
    if out: utterances[vid] = out

avg_dl = sum(len(u["tokens"]) for utts in utterances.values() for u in utts) / sum(len(utts) for utts in utterances.values())

# Per-frame utt ranges
frame_utt_ranges = [None] * N
for i, m in enumerate(meta):
    vid = m["video_id"]
    si = m["scene_idx"]
    sw = scenes.get(vid, {}).get(si)
    if sw is None: continue
    utts = utterances.get(vid)
    if not utts: continue
    start_s, end_s = sw
    l, r = 0, len(utts)
    while l < r:
        mid = (l + r) // 2
        if utts[mid]["end"] < start_s: l = mid + 1
        else: r = mid
    hi = l
    while hi < len(utts) and utts[hi]["start"] <= end_s: hi += 1
    frame_utt_ranges[i] = (l, hi)

# Load failure modes
failures = json.loads((DATA / "clip_failure_modes.json").read_text(encoding="utf-8"))
paraphrases = [f for f in failures if f["category"] == "paraphrase"]
print(f"=== {len(paraphrases)} paraphrase queries ===")

# Load CLIP for query embedding
model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device="cpu")
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()

img_sims = vectors @ vectors.T

for q in paraphrases:
    desc = q["description"]
    gold_path = norm(q["frame_path"])
    gold_idx = path_to_idx.get(gold_path)
    if gold_idx is None:
        print(f"  GOLD MISSING: {gold_path}")
        continue
    qtoks = tokenize(desc)
    toks = tokenizer([desc])
    with torch.no_grad():
        v = model.encode_text(toks)
        v = v / v.norm(dim=-1, keepdim=True)
    qv = v.cpu().numpy().astype(np.float32)[0]
    vs = vectors @ qv
    vs_norm = np.clip((vs - 0.10) / 0.30, 0.0, 1.0)
    # transcript
    t_per_frame = np.zeros(N, dtype=np.float32)
    cache = {}
    for f_i, m in enumerate(meta):
        rng = frame_utt_ranges[f_i]
        if rng is None: continue
        vid = m["video_id"]
        utts = utterances.get(vid)
        if not utts: continue
        lo, hi = rng
        best = 0.0
        for uj in range(lo, hi):
            key = (vid, uj)
            s = cache.get(key)
            if s is None:
                s = bm25(qtoks, utts[uj]["tokens"], avg_dl)
                cache[key] = s
            if s > best: best = s
        t_per_frame[f_i] = min(1.0, best / 5.0)
    scores = 0.7 * vs_norm + 0.3 * t_per_frame
    order = np.argsort(-scores)
    cluster = img_sims[gold_idx] >= 0.85
    hit_at_10 = any(cluster[int(o)] for o in order[:10])
    first_hit = next((r for r, o in enumerate(order, 1) if cluster[int(o)]), None)
    status = "PASS" if hit_at_10 else "FAIL"
    print(f"  [{status}] gold scene {q['scene_idx']:4d} first-hit-rank={first_hit}  desc: {desc!r}")
    if not hit_at_10:
        print(f"         top-5 results:")
        for r, o in enumerate(order[:5], 1):
            m = meta[int(o)]
            print(f"           #{r} {m['video_id']} scene_{m['scene_idx']:04d}  visual={vs[int(o)]:.3f}  transcript_norm={t_per_frame[int(o)]:.3f}")
