"""Shared hybrid retrieval scoring used by both the eval (eval/hybrid_eval.py)
and the production API (ingest/api/main.py).

Single source of truth for: CLIP visual + Deepgram BM25-lite transcript +
speaker filter + (later) scene-tag filter.

Usage:

    state = load_index()                              # one-time, slow
    qv = embed_query_text(state, "the elon clip")
    scored = score_query(state, qv, query_text="the elon clip",
                         w_visual=0.8, w_transcript=0.2)
    top = top_k(scored, k=20, state=state,
                speaker=None, format=None)            # filters applied at rank time
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import CACHE_DIR  # noqa: E402

CLIP_NPZ = CACHE_DIR / "clip_frames.npz"
CLIP_META = CACHE_DIR / "clip_frames_meta.json"
SCENES_DIR = CACHE_DIR / "scenes"
DEEPGRAM_DIR = CACHE_DIR / "deepgram"
SPEAKERS_DIR = CACHE_DIR / "speakers"
SCENE_TAGS_PATH = CACHE_DIR / "scene_tags.json"   # written by classify_scenes.py; optional
AUDIO_TAGS_PATH = CACHE_DIR / "audio_tags.json"   # written by compute_audio_tags.py
TOPIC_SEGMENTS_PATH = CACHE_DIR / "topic_segments.json"
SCENE_TO_SEGMENT_PATH = CACHE_DIR / "scene_to_segment.json"
VIDEO_RECENCY_PATH = CACHE_DIR / "video_recency.json"
SEGMENT_TEXT_NPZ = CACHE_DIR / "segment_text_embeddings.npz"
SEGMENT_TEXT_META = CACHE_DIR / "segment_text_meta.json"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


@dataclass
class HybridState:
    vectors: np.ndarray                              # (N, 768) L2-normalized
    meta: list[dict]                                 # per-vector metadata
    path_to_idx: dict[str, int]                      # frame_path -> vector index
    path_prefix: str                                 # for normalizing caller paths
    scenes: dict[str, dict[int, tuple[float, float]]]
    utterances: dict[str, list[dict]]                # vid -> sorted utts
    speakers: dict[str, dict[int, str]]              # vid -> {cluster_id: name}
    frame_utt_ranges: list[tuple[int, int] | None]   # per-vector (lo, hi) into utts
    avg_dl: float                                    # avg doc length for BM25
    scene_tags: dict[str, dict]                      # path -> {format, shot_type, who, has_text}
    audio_tags: dict[str, dict]                      # frames/<vid>/scene_NNNN.jpg -> {voice, voices_present, speakers_count, silent}
    # Topic-segment layer (third granularity: video -> topic_segment -> scene -> keyframe)
    segments_by_key: dict[tuple, dict]               # (video_id, segment_idx) -> full segment dict
    scene_to_segment: dict[str, dict]                # frames/<vid>/scene_NNNN.jpg -> {video_id, segment_idx}
    video_recency: dict[str, dict]                   # video_id -> {upload_date, recency_days}
    segment_text_vectors: np.ndarray | None          # (N_segs, 768) L2-normalized; None if not yet embedded
    segment_text_index: dict[tuple, int]             # (video_id, segment_idx) -> row in segment_text_vectors
    # CLIP model handles (loaded lazily by embed_query_text)
    _clip_model: object | None = None
    _clip_tokenizer: object | None = None


def normalize_path(p: str, prefix: str) -> str:
    return prefix + p if prefix and not p.startswith(prefix) else p


def load_index(utt_pad_s: float = 0.0) -> HybridState:
    """One-time setup: load CLIP vectors, scenes, transcripts, speakers, scene tags.

    Pass utt_pad_s > 0 to widen the transcript window around each scene before
    looking up overlapping utterances.
    """
    if not CLIP_NPZ.exists():
        raise FileNotFoundError(f"missing {CLIP_NPZ}")

    npz = np.load(CLIP_NPZ, allow_pickle=True)
    vectors = npz["vectors"]
    meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
    N = vectors.shape[0]
    sample = meta[0]["frame_path"]
    path_prefix = "acq_search_v2/" if sample.startswith("acq_search_v2/") else ""
    path_to_idx = {m["frame_path"]: i for i, m in enumerate(meta)}

    scenes: dict[str, dict[int, tuple[float, float]]] = {}
    for p in SCENES_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            scenes[d["video_id"]] = {s["idx"]: (s["start_s"], s["end_s"]) for s in d["scenes"]}
        except Exception:
            pass

    utterances: dict[str, list[dict]] = {}
    for p in DEEPGRAM_DIR.glob("*.audio.json"):
        vid = p.stem.replace(".audio", "")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            utts_raw = d.get("results", {}).get("utterances", [])
            out: list[dict] = []
            for u in utts_raw:
                text = u.get("transcript") or u.get("text") or ""
                if not text or not isinstance(text, str):
                    continue
                out.append({
                    "start": float(u.get("start", 0)),
                    "end": float(u.get("end", 0)),
                    "text": text,
                    "tokens": _tokenize(text),
                    "speaker": u.get("speaker"),
                })
            out.sort(key=lambda u: u["start"])
            if out:
                utterances[vid] = out
        except Exception:
            pass

    speakers: dict[str, dict[int, str]] = {}
    for p in SPEAKERS_DIR.glob("*.audio.json"):
        vid = p.stem.replace(".audio", "")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            spk = d.get("speakers", {})
            speakers[vid] = {int(k): v.get("name", "unknown") for k, v in spk.items()}
        except Exception:
            pass

    # Pre-compute per-frame utterance ranges (with optional padding)
    frame_utt_ranges: list[tuple[int, int] | None] = [None] * N
    for i, m in enumerate(meta):
        vid = m["video_id"]
        si = m["scene_idx"]
        sw = scenes.get(vid, {}).get(si)
        if sw is None:
            continue
        utts = utterances.get(vid)
        if not utts:
            continue
        start_s = max(0.0, sw[0] - utt_pad_s)
        end_s = sw[1] + utt_pad_s
        l, r = 0, len(utts)
        while l < r:
            mid = (l + r) // 2
            if utts[mid]["end"] < start_s:
                l = mid + 1
            else:
                r = mid
        hi = l
        while hi < len(utts) and utts[hi]["start"] <= end_s:
            hi += 1
        frame_utt_ranges[i] = (l, hi)

    all_lens = [len(u["tokens"]) for utts in utterances.values() for u in utts]
    avg_dl = max(1.0, sum(all_lens) / len(all_lens)) if all_lens else 1.0

    scene_tags: dict[str, dict] = {}
    if SCENE_TAGS_PATH.exists():
        try:
            scene_tags = json.loads(SCENE_TAGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    audio_tags: dict[str, dict] = {}
    if AUDIO_TAGS_PATH.exists():
        try:
            audio_tags = json.loads(AUDIO_TAGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    segments_by_key: dict[tuple, dict] = {}
    if TOPIC_SEGMENTS_PATH.exists():
        try:
            raw = json.loads(TOPIC_SEGMENTS_PATH.read_text(encoding="utf-8"))
            segs = raw if isinstance(raw, list) else raw.get("segments", [])
            for s in segs:
                segments_by_key[(s["video_id"], s["segment_idx"])] = s
        except Exception:
            pass

    scene_to_segment: dict[str, dict] = {}
    if SCENE_TO_SEGMENT_PATH.exists():
        try:
            scene_to_segment = json.loads(SCENE_TO_SEGMENT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    video_recency: dict[str, dict] = {}
    if VIDEO_RECENCY_PATH.exists():
        try:
            video_recency = json.loads(VIDEO_RECENCY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    segment_text_vectors: np.ndarray | None = None
    segment_text_index: dict[tuple, int] = {}
    if SEGMENT_TEXT_NPZ.exists() and SEGMENT_TEXT_META.exists():
        try:
            segment_text_vectors = np.load(SEGMENT_TEXT_NPZ, allow_pickle=True)["vectors"]
            text_meta = json.loads(SEGMENT_TEXT_META.read_text(encoding="utf-8"))
            for i, m in enumerate(text_meta):
                segment_text_index[(m["video_id"], m["segment_idx"])] = i
        except Exception:
            segment_text_vectors = None
            segment_text_index = {}

    return HybridState(
        vectors=vectors,
        meta=meta,
        path_to_idx=path_to_idx,
        path_prefix=path_prefix,
        scenes=scenes,
        utterances=utterances,
        speakers=speakers,
        frame_utt_ranges=frame_utt_ranges,
        avg_dl=avg_dl,
        scene_tags=scene_tags,
        audio_tags=audio_tags,
        segments_by_key=segments_by_key,
        scene_to_segment=scene_to_segment,
        video_recency=video_recency,
        segment_text_vectors=segment_text_vectors,
        segment_text_index=segment_text_index,
    )


def _load_clip(state: HybridState) -> None:
    if state._clip_model is not None:
        return
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device="cpu"
    )
    model.eval()
    state._clip_model = (model, torch)
    state._clip_tokenizer = open_clip.get_tokenizer(MODEL_NAME)


def embed_query_text(state: HybridState, text: str) -> np.ndarray:
    _load_clip(state)
    model, torch = state._clip_model  # type: ignore
    toks = state._clip_tokenizer([text])  # type: ignore
    with torch.no_grad():
        v = model.encode_text(toks)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)[0]


def embed_query_texts(state: HybridState, texts: list[str]) -> np.ndarray:
    _load_clip(state)
    model, torch = state._clip_model  # type: ignore
    out = []
    for i in range(0, len(texts), 16):
        chunk = texts[i:i + 16]
        toks = state._clip_tokenizer(chunk)  # type: ignore
        with torch.no_grad():
            v = model.encode_text(toks)
            v = v / v.norm(dim=-1, keepdim=True)
        out.append(v.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


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


def transcript_scores(state: HybridState, query_text: str) -> np.ndarray:
    """Per-frame transcript score in [0, 1]."""
    qtoks = _tokenize(query_text)
    N = state.vectors.shape[0]
    out = np.zeros(N, dtype=np.float32)
    cache: dict[tuple[str, int], float] = {}
    for f_i, m in enumerate(state.meta):
        rng = state.frame_utt_ranges[f_i]
        if rng is None:
            continue
        vid = m["video_id"]
        utts = state.utterances.get(vid)
        if not utts:
            continue
        lo, hi = rng
        best = 0.0
        for uj in range(lo, hi):
            key = (vid, uj)
            s = cache.get(key)
            if s is None:
                s = _bm25lite(qtoks, utts[uj]["tokens"], state.avg_dl)
                cache[key] = s
            if s > best:
                best = s
        out[f_i] = min(1.0, best / 5.0)
    return out


def _segment_text_scores_per_frame(state: HybridState, text_vec: np.ndarray) -> np.ndarray:
    """For each frame, return cosine(query, parent_segment_text_embedding).
    Frames whose parent segment has no embedding (or no parent) get 0.
    """
    N = state.vectors.shape[0]
    out = np.zeros(N, dtype=np.float32)
    if state.segment_text_vectors is None or len(state.segment_text_index) == 0:
        return out
    # Cosine vs all segments first (single matmul), then map to frames
    seg_scores = state.segment_text_vectors @ text_vec   # (N_segs,)
    seg_scores = np.clip(seg_scores, 0.0, 1.0)            # already L2-normalized, cosine in [-1,1]
    for i, m in enumerate(state.meta):
        key = m["frame_path"]
        key = key[len("ingest/"):] if key.startswith("ingest/") else key
        ref = state.scene_to_segment.get(key)
        if ref is None or ref.get("segment_idx") is None:
            continue
        row = state.segment_text_index.get((ref["video_id"], ref["segment_idx"]))
        if row is None:
            continue
        out[i] = float(seg_scores[row])
    return out


def score_query(
    state: HybridState,
    text_vec: np.ndarray,
    query_text: str = "",
    w_visual: float = 0.5,
    w_transcript: float = 0.2,
    w_segment: float = 0.3,
) -> np.ndarray:
    """Per-frame hybrid score. Three components:
      - visual:    CLIP image-text cosine vs each frame
      - transcript: BM25-lite on Deepgram utterances within scene window
      - segment:   CLIP text-text cosine vs each frame's parent topic_segment summary

    If segment data is missing (no topic_segments yet), w_segment falls through to zero
    and the weights effectively renormalize to visual+transcript only.
    """
    visual = state.vectors @ text_vec
    visual_norm = np.clip((visual - 0.10) / 0.30, 0.0, 1.0)
    base = w_visual * visual_norm
    if w_transcript > 0 and query_text:
        base = base + w_transcript * transcript_scores(state, query_text)
    if w_segment > 0 and state.segment_text_vectors is not None:
        seg_per_frame = _segment_text_scores_per_frame(state, text_vec)
        # rescale CLIP text-text cosine (typically 0.15-0.45) to a [0,1] range
        seg_norm = np.clip((seg_per_frame - 0.15) / 0.30, 0.0, 1.0)
        base = base + w_segment * seg_norm
    return base


def _audio_key(frame_path: str) -> str:
    """audio_tags.json is keyed by 'frames/<vid>/scene_NNNN.jpg' (relative to ingest/)."""
    return frame_path[len("ingest/"):] if frame_path.startswith("ingest/") else frame_path


def _frame_segment(state: HybridState, idx: int) -> dict | None:
    """Return the topic_segment dict the frame belongs to, or None."""
    m = state.meta[idx]
    key = m["frame_path"]
    key = key[len("ingest/"):] if key.startswith("ingest/") else key
    ref = state.scene_to_segment.get(key)
    if ref is None or ref.get("segment_idx") is None:
        return None
    return state.segments_by_key.get((ref["video_id"], ref["segment_idx"]))


def _frame_passes_filters(
    state: HybridState,
    idx: int,
    speaker: str | None = None,
    required_speakers: list[str] | None = None,
    speakers_count: str | None = None,
    is_animation: bool | None = None,
    talking_head_pose: str | None = None,
    industries: list[str] | None = None,
    audience: list[str] | None = None,
    age_group: str | None = None,
    lessons_categories: list[str] | None = None,
    instructional_only: bool = False,
    min_duration_s: float | None = None,
    max_duration_s: float | None = None,
    max_age_days: int | None = None,
    min_age_days: int | None = None,
) -> bool:
    m = state.meta[idx]
    vid = m["video_id"]
    a = state.audio_tags.get(_audio_key(m["frame_path"])) if state.audio_tags else None
    # Multi-speaker co-occurrence filter takes precedence over single-speaker filter.
    if required_speakers:
        if a is None:
            return False
        present = set(a.get("voices_present") or [])
        if not all(n in present for n in required_speakers):
            return False
    elif speaker:
        # Prefer scene-level voice when audio_tags exist; fall back to video-level membership.
        if a is not None:
            if a.get("voice") != speaker:
                return False
        else:
            spk_map = state.speakers.get(vid, {})
            if speaker not in spk_map.values():
                return False
    if speakers_count:
        if a is None or a.get("speakers_count") != speakers_count:
            return False
    # Scene-tag filters only apply when the visual classifier output exists.
    # New 2-field schema: is_animation (bool) + talking_head_pose ("front_view"|"side_view"|"none").
    # When scene_tags is empty we silently skip these filters.
    if state.scene_tags and (is_animation is not None or talking_head_pose is not None):
        tag = state.scene_tags.get(normalize_path(m["frame_path"], state.path_prefix))
        if tag is None:
            return False
        if is_animation is not None and bool(tag.get("is_animation")) != bool(is_animation):
            return False
        if talking_head_pose is not None and tag.get("talking_head_pose") != talking_head_pose:
            return False

    # Segment-level filters (industries, audience, age, lessons). When segment data
    # is missing we silently skip these filters so search still works.
    if state.segments_by_key and (industries or audience or age_group or
                                  lessons_categories or instructional_only):
        seg = _frame_segment(state, idx)
        if seg is None:
            return False
        if industries:
            seg_ind = set(seg.get("industries") or [])
            if not (seg_ind & set(industries)):
                return False
        if audience:
            seg_aud = set(seg.get("audience") or [])
            if not (seg_aud & set(audience)):
                return False
        if age_group and seg.get("age_group") != age_group:
            return False
        if lessons_categories:
            seg_l = set(seg.get("lessons_categories") or [])
            if not (seg_l & set(lessons_categories)):
                return False
        if instructional_only:
            seg_l = set(seg.get("lessons_categories") or [])
            if not seg_l or seg_l == {"none"}:
                return False

    # Scene duration filter (from scenes cache).
    if min_duration_s is not None or max_duration_s is not None:
        vid = m["video_id"]
        sw = state.scenes.get(vid, {}).get(m["scene_idx"])
        if sw is None:
            return False
        dur = sw[1] - sw[0]
        if min_duration_s is not None and dur < min_duration_s:
            return False
        if max_duration_s is not None and dur > max_duration_s:
            return False

    # Recency filter (video upload date).
    if (max_age_days is not None or min_age_days is not None) and state.video_recency:
        rec = state.video_recency.get(vid)
        rd = (rec or {}).get("recency_days")
        if rd is None:
            return False
        if max_age_days is not None and rd > max_age_days:
            return False
        if min_age_days is not None and rd < min_age_days:
            return False

    return True


def top_k(
    state: HybridState,
    scores: np.ndarray,
    k: int = 20,
    speaker: str | None = None,
    required_speakers: list[str] | None = None,
    speakers_count: str | None = None,
    is_animation: bool | None = None,
    talking_head_pose: str | None = None,
    industries: list[str] | None = None,
    audience: list[str] | None = None,
    age_group: str | None = None,
    lessons_categories: list[str] | None = None,
    instructional_only: bool = False,
    min_duration_s: float | None = None,
    max_duration_s: float | None = None,
    max_age_days: int | None = None,
    min_age_days: int | None = None,
    per_video_cap: int = 1,
    per_segment_cap: int = 1,
) -> list[dict]:
    """Apply filters + per-video dedup, return top k.

    The product spec ("don't show both a whole-video and segment hit for the same
    video") collapses to: keep only the highest-scoring scene per video by
    default. Caller can widen with per_video_cap.
    """
    order = np.argsort(-scores)
    results: list[dict] = []
    seen_videos: dict[str, int] = {}
    seen_segments: dict[tuple, int] = {}
    for o in order:
        idx = int(o)
        if not _frame_passes_filters(
            state, idx, speaker, required_speakers, speakers_count,
            is_animation, talking_head_pose,
            industries, audience, age_group, lessons_categories,
            instructional_only, min_duration_s, max_duration_s, max_age_days, min_age_days,
        ):
            continue
        m = state.meta[idx]
        vid = m["video_id"]
        if seen_videos.get(vid, 0) >= per_video_cap:
            continue
        seg = _frame_segment(state, idx)
        seg_key = (vid, seg["segment_idx"]) if seg else None
        if seg_key is not None and seen_segments.get(seg_key, 0) >= per_segment_cap:
            continue
        seen_videos[vid] = seen_videos.get(vid, 0) + 1
        if seg_key is not None:
            seen_segments[seg_key] = seen_segments.get(seg_key, 0) + 1
        a = state.audio_tags.get(_audio_key(m["frame_path"])) if state.audio_tags else None
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[idx]),
            "video_id": vid,
            "scene_idx": m["scene_idx"],
            "frame_path": m["frame_path"],
            "is_intra": m.get("is_intra", False),
            "voice": a.get("voice") if a else None,
            "voices_present": a.get("voices_present") if a else [],
            "speakers_count": a.get("speakers_count") if a else None,
            "segment": {
                "video_id": seg["video_id"],
                "segment_idx": seg["segment_idx"],
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "topic_title": seg.get("topic_title"),
                "summary": seg.get("summary"),
                "lessons_categories": seg.get("lessons_categories", []),
                "industries": seg.get("industries", []),
                "audience": seg.get("audience", []),
            } if seg else None,
            "recency_days": (state.video_recency.get(vid) or {}).get("recency_days"),
            "upload_date": (state.video_recency.get(vid) or {}).get("upload_date"),
        })
        if len(results) >= k:
            break
    return results


def youtube_link(video_id: str, start_s: float) -> str:
    """Build a deep-link that opens the video at the given timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(round(start_s))}s"
