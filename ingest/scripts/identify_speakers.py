"""Map each Deepgram speaker cluster to alex / leila / sharran / unknown.

For every cached transcript in CACHE_DIR/deepgram/, this script:
  1. Builds a voice centroid from each reference clip (read from env vars
     ALEX_REFERENCE_AUDIO, LEILA_REFERENCE_AUDIO, SHARRAN_REFERENCE_AUDIO).
  2. For each diarized cluster, concatenates up to 60 s of that speaker's
     audio and computes a Resemblyzer embedding.
  3. Cosine-similarity against each centroid. Label = whichever scores highest
     IF that score >= MATCH_FLOOR, else "unknown".

Per-file output: CACHE_DIR/speakers/<stem>.json
Progress log: LOGS_DIR/speaker_id.log

Idempotent: skips files whose mapping already exists. --force to re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (  # noqa: E402
    ALEX_REFERENCE_AUDIO,
    CACHE_DIR,
    LEILA_REFERENCE_AUDIO,
    LOGS_DIR,
    MEDIA_DIR,
    SHARRAN_REFERENCE_AUDIO,
)

DEEPGRAM_DIR = CACHE_DIR / "deepgram"
SPEAKERS_DIR = CACHE_DIR / "speakers"
LOG_PATH = LOGS_DIR / "speaker_id.log"

TARGET_SR = 16_000
MIN_CLUSTER_AUDIO_S = 5.0
MAX_CLUSTER_AUDIO_S = 60.0
MATCH_FLOOR = 0.60


def _references() -> dict[str, Path]:
    refs: dict[str, Path] = {}
    for name, raw in (
        ("alex", ALEX_REFERENCE_AUDIO),
        ("leila", LEILA_REFERENCE_AUDIO),
        ("sharran", SHARRAN_REFERENCE_AUDIO),
    ):
        if not raw:
            raise RuntimeError(f"{name.upper()}_REFERENCE_AUDIO not set in .env")
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"{name} reference audio missing: {p}")
        refs[name] = p
    return refs


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32), sr


def _intervals_for_speaker(words: list[dict], sid: int) -> list[tuple[float, float]]:
    return [
        (float(w["start"]), float(w["end"]))
        for w in words
        if w.get("speaker") == sid and "start" in w and "end" in w
    ]


def _cluster_audio(audio: np.ndarray, sr: int, intervals: list[tuple[float, float]]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0.0
    for start, end in intervals:
        i0 = max(0, int(start * sr))
        i1 = min(len(audio), int(end * sr))
        if i1 <= i0:
            continue
        chunks.append(audio[i0:i1])
        total += (i1 - i0) / sr
        if total >= MAX_CLUSTER_AUDIO_S:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _build_centroids(encoder, refs: dict[str, Path]) -> dict[str, np.ndarray]:
    from resemblyzer import preprocess_wav

    centroids: dict[str, np.ndarray] = {}
    for name, p in refs.items():
        wav = preprocess_wav(str(p))
        centroids[name] = encoder.embed_utterance(wav)
        print(f"  centroid built for {name} (dim={centroids[name].shape[0]})", flush=True)
    return centroids


def _identify_one(
    video_id: str,
    audio_path: Path,
    dg_path: Path,
    encoder,
    centroids: dict[str, np.ndarray],
) -> dict | None:
    from resemblyzer import preprocess_wav

    dg = json.loads(dg_path.read_text(encoding="utf-8"))
    try:
        alt = dg["results"]["channels"][0]["alternatives"][0]
        words = alt.get("words") or []
    except (KeyError, IndexError):
        return None
    if not words:
        return None

    audio, sr = _load_mono(audio_path)
    speaker_ids = sorted(
        {w["speaker"] for w in words if isinstance(w.get("speaker"), int)}
    )

    speakers: dict[str, dict] = {}
    for sid in speaker_ids:
        intervals = _intervals_for_speaker(words, sid)
        clip = _cluster_audio(audio, sr, intervals)
        dur = float(clip.size) / float(sr)
        rec: dict = {"duration_s": round(dur, 2), "name": "unknown"}
        if dur >= MIN_CLUSTER_AUDIO_S:
            wav = preprocess_wav(clip, source_sr=sr)
            emb = encoder.embed_utterance(wav)
            sims = {n: _cosine(c, emb) for n, c in centroids.items()}
            for n, s in sims.items():
                rec[f"sim_{n}"] = round(s, 3)
            best_name, best_sim = max(sims.items(), key=lambda kv: kv[1])
            rec["best_sim"] = round(best_sim, 3)
            if best_sim >= MATCH_FLOOR:
                rec["name"] = best_name
        speakers[str(sid)] = rec

    return {"video_id": video_id, "speakers": speakers}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="only process this video_id")
    parser.add_argument("--force", action="store_true", help="re-process cached")
    args = parser.parse_args()

    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder(verbose=False)
    refs = _references()
    print("building reference centroids...", flush=True)
    centroids = _build_centroids(encoder, refs)

    if args.only:
        stems = [f"{args.only}.audio"]
    else:
        stems = sorted(p.stem for p in DEEPGRAM_DIR.glob("*.audio.json"))

    n = len(stems)
    print(f"=== speaker-id: {n} target(s) ===", flush=True)

    ok = skip = err = 0
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} n={n} ===\n")
        for i, stem in enumerate(stems, 1):
            video_id = stem.replace(".audio", "")
            out_path = SPEAKERS_DIR / f"{stem}.json"
            audio_path = MEDIA_DIR / f"{video_id}.audio.mp3"
            dg_path = DEEPGRAM_DIR / f"{stem}.json"

            if out_path.exists() and not args.force:
                skip += 1
                line = f"[{i}/{n}] {video_id}: cached"
            elif not audio_path.exists() or not dg_path.exists():
                err += 1
                line = f"[{i}/{n}] {video_id}: MISSING inputs"
            else:
                try:
                    result = _identify_one(video_id, audio_path, dg_path, encoder, centroids)
                    if result is None:
                        err += 1
                        line = f"[{i}/{n}] {video_id}: no words"
                    else:
                        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                        ok += 1
                        labels = ", ".join(
                            f"sid{k}={v['name']}({v.get('best_sim','n/a')},{v['duration_s']:.0f}s)"
                            for k, v in result["speakers"].items()
                        )
                        line = f"[{i}/{n}] {video_id}: {labels}"
                except Exception as e:
                    err += 1
                    line = f"[{i}/{n}] {video_id}: EXCEPTION {e}"

            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        summary = f"=== done ok={ok} cached={skip} err={err} ==="
        print(summary, flush=True)
        log.write(summary + "\n")

    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
