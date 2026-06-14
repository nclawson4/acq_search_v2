"""Four feature scorers. Each is a per-frame additive boost on top of the
production hybrid baseline:

    final_scores = baseline_scores + alpha * feature_boost

Each scorer is standalone — no shared state, no production-side changes.
Tuning knob `alpha` is set per feature (defensible defaults below). The
harness can override.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

from config import CACHE_DIR  # noqa: E402

OUT_DIR = CACHE_DIR / "eval_dg"

# Default alphas. Each is a fraction of the baseline scale (baseline ~ 0–1).
# Kept small so the boost can re-rank without dominating.
DEFAULT_ALPHA = {
    "entities": 0.20,
    "topics": 0.25,
    "sentiment": 0.20,
    "summary": 0.30,
}

TOKEN_RE = re.compile(r"[a-z0-9'$\.]+")
STOP = set("a an the of in on at for to and or with from this that those these about my our your their there here is are was were be been being do does did has have had will would should could can may might just so very".split())


def _tokenize(s: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(s.lower()) if t not in STOP and len(t) >= 3]


# ---------------------------------------------------------------------------
# 1) Entities
# ---------------------------------------------------------------------------
class EntitiesScorer:
    def __init__(self, alpha: float | None = None):
        self.alpha = DEFAULT_ALPHA["entities"] if alpha is None else alpha
        with (OUT_DIR / "entities.json").open("r", encoding="utf-8") as f:
            self.by_vid: dict[str, list[dict]] = json.load(f)

    def boost(self, state, query_text: str) -> np.ndarray:
        toks = set(_tokenize(query_text))
        # Money intent — match CARDINAL/MONEY labels when the query has numeric or currency cues.
        money_intent = bool(re.search(r"\$|\bmillion\b|\bbillion\b|\bdollars?\b|\d+m\b|\d+k\b", query_text.lower()))
        n = len(state.meta)
        out = np.zeros(n, dtype=np.float32)
        if not toks and not money_intent:
            return out

        # Pre-index entity rows by video for fast scene-window queries.
        cache: dict[str, list[tuple[float, float, str, str, float]]] = {}
        for vid, rows in self.by_vid.items():
            cache[vid] = [(r["start_s"], r["end_s"], r["label"], r["value"].lower(), r["confidence"]) for r in rows]

        for i, m in enumerate(state.meta):
            vid = m["video_id"]
            rows = cache.get(vid)
            if not rows:
                continue
            scene = state.scenes.get(vid, {}).get(int(m["scene_idx"]))
            if scene is None:
                continue
            ss, se = scene
            best = 0.0
            for (a, b, label, value, conf) in rows:
                if b <= ss or a >= se:
                    continue
                hit = 0.0
                if toks and any(t in value for t in toks):
                    hit = conf
                elif money_intent and label in ("MONEY", "CARDINAL", "PERCENT"):
                    hit = 0.5 * conf
                if hit > best:
                    best = hit
            out[i] = best
        return self.alpha * out


# ---------------------------------------------------------------------------
# 2) Topics  (CLIP text-text cosine between query and Deepgram topic labels)
# ---------------------------------------------------------------------------
class TopicsScorer:
    def __init__(self, alpha: float | None = None):
        self.alpha = DEFAULT_ALPHA["topics"] if alpha is None else alpha
        with (OUT_DIR / "topics.json").open("r", encoding="utf-8") as f:
            by_vid: dict[str, list[dict]] = json.load(f)
        labels = set()
        for rows in by_vid.values():
            for r in rows:
                if r["topic"]:
                    labels.add(r["topic"])
        self.label_list = sorted(labels)
        self.label_to_idx = {lab: i for i, lab in enumerate(self.label_list)}
        # Pre-build per-video tuples once.
        self.cache: dict[str, list[tuple[float, float, int, float]]] = {}
        for vid, rows in by_vid.items():
            self.cache[vid] = [
                (float(r["start_s"]), float(r["end_s"]), self.label_to_idx[r["topic"]], float(r["confidence"]))
                for r in rows if r["topic"] in self.label_to_idx
            ]
        self._label_vecs: np.ndarray | None = None

    def _ensure_label_vecs(self, state):
        if self._label_vecs is not None:
            return
        from lib.hybrid import embed_query_texts  # noqa: E402
        if not self.label_list:
            self._label_vecs = np.zeros((0, state.vectors.shape[1]), dtype=np.float32)
            return
        print(f"  [topics] embedding {len(self.label_list)} unique topic labels (one-time)...", flush=True)
        self._label_vecs = embed_query_texts(state, self.label_list)
        print(f"  [topics] label embeddings ready: {self._label_vecs.shape}", flush=True)

    def boost(self, state, query_text: str) -> np.ndarray:
        from lib.hybrid import embed_query_text  # noqa: E402
        n = len(state.meta)
        out = np.zeros(n, dtype=np.float32)
        if not self.label_list:
            return out
        self._ensure_label_vecs(state)
        qv = embed_query_text(state, query_text)
        sims = self._label_vecs @ qv
        sims_n = np.clip((sims - 0.15) / 0.30, 0.0, 1.0)
        for i, m in enumerate(state.meta):
            rows = self.cache.get(m["video_id"])
            if not rows:
                continue
            scene = state.scenes.get(m["video_id"], {}).get(int(m["scene_idx"]))
            if scene is None:
                continue
            ss, se = scene
            best = 0.0
            for (a, b, li, conf) in rows:
                if b <= ss or a >= se:
                    continue
                v = float(sims_n[li]) * conf
                if v > best:
                    best = v
            out[i] = best
        return self.alpha * out


# ---------------------------------------------------------------------------
# 3) Sentiment  (per-scene aggregate matches the query's affect cue)
# ---------------------------------------------------------------------------
SENT_POS = re.compile(r"\b(hype|hyped|celebrat|excit|peak|energy|enthusias|inspir|encourag|optimis|joy)", re.I)
SENT_NEG = re.compile(r"\b(rant|frustrat|angry|harsh|criticis|criticiz|skeptic|cynic|regret|doubt|uncertain|pessimis|sad|emotional)", re.I)
SENT_SHIFT = re.compile(r"\b(shift|pivot|turn|swing|from .* to)", re.I)


@dataclass
class _SentArrays:
    mean: np.ndarray
    max_pos: np.ndarray
    max_neg: np.ndarray
    abs_max: np.ndarray
    polarity_shift: np.ndarray
    covered: np.ndarray


class SentimentScorer:
    def __init__(self, alpha: float | None = None):
        self.alpha = DEFAULT_ALPHA["sentiment"] if alpha is None else alpha
        data = np.load(OUT_DIR / "sentiment.npz")
        self.s = _SentArrays(
            mean=data["mean"], max_pos=data["max_pos"], max_neg=data["max_neg"],
            abs_max=data["abs_max"], polarity_shift=data["polarity_shift"], covered=data["covered"],
        )

    def boost(self, state, query_text: str) -> np.ndarray:
        n = len(state.meta)
        pos = bool(SENT_POS.search(query_text))
        neg = bool(SENT_NEG.search(query_text))
        shift = bool(SENT_SHIFT.search(query_text))
        if not (pos or neg or shift):
            return np.zeros(n, dtype=np.float32)

        out = np.zeros(n, dtype=np.float32)
        s = self.s
        if pos and not neg:
            out = np.clip(s.max_pos, 0.0, 1.0)
        elif neg and not pos:
            out = np.clip(-s.max_neg, 0.0, 1.0)
        elif pos and neg:
            out = 0.5 * (np.clip(s.max_pos, 0.0, 1.0) + np.clip(-s.max_neg, 0.0, 1.0))
        if shift:
            out = np.maximum(out, s.polarity_shift.astype(np.float32))
        out = out * s.covered.astype(np.float32)
        return self.alpha * out


# ---------------------------------------------------------------------------
# 4) Summary  (video-level prior: CLIP query vs Deepgram per-video summary)
# ---------------------------------------------------------------------------
class SummaryScorer:
    def __init__(self, alpha: float | None = None):
        self.alpha = DEFAULT_ALPHA["summary"] if alpha is None else alpha
        data = np.load(OUT_DIR / "summary_embeddings.npz", allow_pickle=False)
        self.embeddings: np.ndarray = data["embeddings"]
        self.video_ids: list[str] = [str(v) for v in data["video_ids"]]
        self.vid_to_row = {v: i for i, v in enumerate(self.video_ids)}

    def boost(self, state, query_text: str) -> np.ndarray:
        from lib.hybrid import embed_query_text  # noqa: E402
        n = len(state.meta)
        out = np.zeros(n, dtype=np.float32)
        if self.embeddings.shape[0] == 0:
            return out
        qv = embed_query_text(state, query_text)
        sims = self.embeddings @ qv  # (V,)
        sims_n = np.clip((sims - 0.15) / 0.30, 0.0, 1.0)
        for i, m in enumerate(state.meta):
            row = self.vid_to_row.get(m["video_id"])
            if row is not None:
                out[i] = sims_n[row]
        return self.alpha * out


SCORERS = {
    "entities": EntitiesScorer,
    "topics": TopicsScorer,
    "sentiment": SentimentScorer,
    "summary": SummaryScorer,
}
