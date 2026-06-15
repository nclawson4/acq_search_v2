#!/usr/bin/env bash
# One-shot: COPY this session's artifacts from v1 into v2. Pure cp (never mv).
# Reads only — v1 is not modified.
#
# Source: C:/Users/nclaw/acq_search_retrieval
# Dest:   C:/Users/nclaw/acq_search_v2
#
# Safe to re-run. cp -n means "no overwrite" — re-runs are idempotent.

set -u
set -o pipefail

SRC="C:/Users/nclaw/acq_search_retrieval"
DST="C:/Users/nclaw/acq_search_v2"
IDS_FILE="$DST/docs/my_video_ids.txt"
LOG="$DST/ingest/logs/_copy_from_v1.log"

mkdir -p "$DST/ingest/media" \
         "$DST/ingest/cache/deepgram" \
         "$DST/ingest/cache/speakers" \
         "$DST/ingest/logs"

echo "==== copy start $(date -u +%FT%TZ) ====" | tee "$LOG"

# 1. Deepgram cache (115 files)
echo "[1/5] copying cache/deepgram_v2/ -> cache/deepgram/" | tee -a "$LOG"
cp -n "$SRC/ingest/cache/deepgram_v2/"*.json "$DST/ingest/cache/deepgram/" 2>>"$LOG" || true
echo "  dest count: $(ls "$DST/ingest/cache/deepgram/" 2>/dev/null | wc -l)" | tee -a "$LOG"

# 2. Speakers cache (115 files)
echo "[2/5] copying cache/speakers_v2/ -> cache/speakers/" | tee -a "$LOG"
cp -n "$SRC/ingest/cache/speakers_v2/"*.json "$DST/ingest/cache/speakers/" 2>>"$LOG" || true
echo "  dest count: $(ls "$DST/ingest/cache/speakers/" 2>/dev/null | wc -l)" | tee -a "$LOG"

# 3. Media files — only the 66 IDs I downloaded this session
echo "[3/5] copying my 66 media files (mp4 + audio.mp3 + info.json)" | tee -a "$LOG"
copied=0
while IFS= read -r id; do
  [ -z "$id" ] && continue
  vid_id="${id%.mp4}"
  for ext in mp4 audio.mp3 info.json; do
    src_f="$SRC/ingest/media/$vid_id.$ext"
    if [ -f "$src_f" ]; then
      cp -n "$src_f" "$DST/ingest/media/" 2>>"$LOG" || true
      copied=$((copied + 1))
    fi
  done
done < "$IDS_FILE"
echo "  files copied: $copied" | tee -a "$LOG"

# 4. Logs (rename: drop v2_ prefix)
echo "[4/5] copying logs (renamed)" | tee -a "$LOG"
cp -n "$SRC/ingest/logs/v2_download.log"   "$DST/ingest/logs/download.log"   2>>"$LOG" || true
cp -n "$SRC/ingest/logs/v2_transcribe.log" "$DST/ingest/logs/transcribe.log" 2>>"$LOG" || true
cp -n "$SRC/ingest/logs/v2_speaker_id.log" "$DST/ingest/logs/speaker_id.log" 2>>"$LOG" || true
echo "  logs in dest: $(ls "$DST/ingest/logs/" | wc -l)" | tee -a "$LOG"

# 5. Archive files
echo "[5/5] copying archive files" | tee -a "$LOG"
cp -n "$SRC/ingest/media/archive_video.txt" "$DST/ingest/media/" 2>>"$LOG" || true
cp -n "$SRC/ingest/media/archive_audio.txt" "$DST/ingest/media/" 2>>"$LOG" || true

echo "==== copy done $(date -u +%FT%TZ) ====" | tee -a "$LOG"
echo "v2 media size: $(du -sh "$DST/ingest/media" | awk '{print $1}')" | tee -a "$LOG"
echo "v2 cache size: $(du -sh "$DST/ingest/cache" | awk '{print $1}')" | tee -a "$LOG"
