#!/usr/bin/env bash
# Download N most recent videos from the 3 reference channels.
# Two passes per channel (video + audio extract). Sequential. Idempotent via
# download-archive (so re-running skips IDs already on disk).

set -u
set -o pipefail

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
YTDLP="$ROOT/.venv/Scripts/yt-dlp.exe"
[ -x "$YTDLP" ] || YTDLP="$( command -v yt-dlp || echo "$YTDLP" )"

MEDIA="$ROOT/media"
LOGDIR="$ROOT/logs"
LOG="$LOGDIR/download.log"

mkdir -p "$MEDIA" "$LOGDIR"

ARCHIVE_VIDEO="$MEDIA/archive_video.txt"
ARCHIVE_AUDIO="$MEDIA/archive_audio.txt"

# Prime archives from any pre-existing media (re-runs skip them).
if [ ! -f "$ARCHIVE_VIDEO" ]; then
  for f in "$MEDIA"/*.mp4; do
    [ -f "$f" ] || continue
    id=$(basename "$f" .mp4)
    echo "youtube $id" >> "$ARCHIVE_VIDEO"
  done
fi
if [ ! -f "$ARCHIVE_AUDIO" ]; then
  for f in "$MEDIA"/*.audio.mp3; do
    [ -f "$f" ] || continue
    id=$(basename "$f" .audio.mp3)
    echo "youtube $id" >> "$ARCHIVE_AUDIO"
  done
fi

CHANNELS=(
  "https://www.youtube.com/@AlexHormozi/videos"
  "https://www.youtube.com/@leilahormozi/videos"
  "https://www.youtube.com/@sharran/videos"
)

PLAYLIST_END="${PLAYLIST_END:-30}"

echo "==== START $(date -u +%FT%TZ) playlist_end=$PLAYLIST_END ====" | tee -a "$LOG"

for channel in "${CHANNELS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "==== $channel (video, $PLAYLIST_END most recent) ====" | tee -a "$LOG"
  "$YTDLP" \
    --playlist-end "$PLAYLIST_END" \
    --format "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best" \
    --merge-output-format mp4 \
    --write-info-json \
    --output "$MEDIA/%(id)s.%(ext)s" \
    --no-overwrites \
    --download-archive "$ARCHIVE_VIDEO" \
    --ignore-errors \
    --no-progress \
    "$channel" 2>&1 | tee -a "$LOG"

  echo "" | tee -a "$LOG"
  echo "==== $channel (audio, $PLAYLIST_END most recent) ====" | tee -a "$LOG"
  "$YTDLP" \
    --playlist-end "$PLAYLIST_END" \
    --format "bestaudio/best" \
    --extract-audio \
    --audio-format mp3 \
    --audio-quality 32 \
    --postprocessor-args "-ac 1 -ar 16000" \
    --output "$MEDIA/%(id)s.audio.%(ext)s" \
    --no-overwrites \
    --download-archive "$ARCHIVE_AUDIO" \
    --ignore-errors \
    --no-progress \
    "$channel" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "==== DONE $(date -u +%FT%TZ) ====" | tee -a "$LOG"
