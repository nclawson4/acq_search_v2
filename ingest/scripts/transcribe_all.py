"""Transcribe every audio file in media/ with Deepgram nova-3 + full add-ons.

Add-ons enabled: punctuation, smart format, paragraphs, utterances, diarization,
sentiment, topics, summarize=v2, detect_entities. Word-level timestamps and
confidence are included in the base model.

Idempotent: skips audio files whose response is already cached under
CACHE_DIR/deepgram/. Run with --only=<video_id> for a single-file smoke test
(use the `=` form because YouTube IDs may start with `-`).

Per-file output: CACHE_DIR/deepgram/<stem>.json
Progress log: LOGS_DIR/transcribe.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CACHE_DIR, DEEPGRAM_API_KEY, LOGS_DIR, MEDIA_DIR  # noqa: E402

DEEPGRAM_DIR = CACHE_DIR / "deepgram"
LOG_PATH = LOGS_DIR / "transcribe.log"

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_PARAMS = {
    "model": "nova-3",
    "smart_format": "true",
    "punctuate": "true",
    "paragraphs": "true",
    "utterances": "true",
    "diarize": "true",
    "language": "en",
    "filler_words": "false",
    "sentiment": "true",
    "topics": "true",
    "summarize": "v2",
    "detect_entities": "true",
}
REQUEST_TIMEOUT_S = 900.0


def _content_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "mp4": "audio/mp4",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
        "flac": "audio/flac",
    }.get(ext, "application/octet-stream")


def _cache_path_for(audio_path: Path) -> Path:
    return DEEPGRAM_DIR / f"{audio_path.stem}.json"


def transcribe_one(audio_path: Path) -> tuple[str, float]:
    cache_path = _cache_path_for(audio_path)
    if cache_path.exists():
        return "cached", 0.0

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": _content_type(audio_path),
    }
    t0 = time.monotonic()
    with audio_path.open("rb") as f:
        resp = httpx.post(
            DEEPGRAM_URL,
            params=DEEPGRAM_PARAMS,
            headers=headers,
            content=f.read(),
            timeout=REQUEST_TIMEOUT_S,
        )
    dt = time.monotonic() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"Deepgram {resp.status_code}: {resp.text[:400]}")

    DEEPGRAM_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(resp.json()), encoding="utf-8")
    return "ok", dt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="only transcribe this video_id")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.only:
        targets = [MEDIA_DIR / f"{args.only}.audio.mp3"]
        if not targets[0].exists():
            print(f"not found: {targets[0]}", flush=True)
            return 1
    else:
        targets = sorted(MEDIA_DIR.glob("*.audio.mp3"))

    n = len(targets)
    print(f"=== transcribe: {n} file(s), cache={DEEPGRAM_DIR} ===", flush=True)

    ok = cached = err = 0
    paid_dt = 0.0

    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} n={n} ===\n")
        for i, ap in enumerate(targets, 1):
            try:
                status, dt = transcribe_one(ap)
                if status == "cached":
                    cached += 1
                    line = f"[{i}/{n}] {ap.name}: cached"
                else:
                    ok += 1
                    paid_dt += dt
                    line = f"[{i}/{n}] {ap.name}: ok in {dt:.1f}s"
            except Exception as e:
                err += 1
                line = f"[{i}/{n}] {ap.name}: ERROR {e}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        summary = f"=== done ok={ok} cached={cached} err={err} paid_wall={paid_dt:.1f}s ==="
        print(summary, flush=True)
        log.write(summary + "\n")

    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
