# acq_search_v2

Search and retrieval over a long-form video library where the **query is itself a video URL**.

Find content that:
- talks about the same message
- is in the same format (talking head / podcast / phone Q&A / live Q&A / low production / whiteboard)
- features the same speaker (alex, leila, sharran are the main three)
- any combination of the above

Output: ranked YouTube URLs with start+end timestamps. One result per video — either the whole video OR a sub-segment, never both.

## Status

Bootstrap of raw ingest data only. Built this session:
- 66 new videos downloaded from 3 channels (Alex Hormozi / Leila Hormozi / Sharran)
- Deepgram nova-3 transcripts with diarization, sentiment, topics, entities, summary for all 115 audio files
- Speaker identification (Resemblyzer) mapping diarized clusters to alex/leila/sharran/unknown

Full audit of what was built and how: see `docs/SESSION_LOG.md`.

## Layout

```
acq_search_v2/
├── .env.example            # populate to .env locally
├── .gitignore
├── docs/
│   └── SESSION_LOG.md      # provenance + impact analysis
└── ingest/
    ├── config.py
    ├── requirements.txt
    ├── scripts/
    │   ├── download_corpus.sh
    │   ├── transcribe_all.py
    │   └── identify_speakers.py
    ├── cache/
    │   ├── deepgram/       # one .json per audio file
    │   └── speakers/       # cluster→name map per video
    ├── logs/
    ├── media/              # mp4 + audio.mp3 + info.json per video (gitignored)
    └── frames/
```

## Quickstart

```bash
cd ingest
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp ../.env.example ../.env  # fill in DEEPGRAM_API_KEY and the 3 reference audio paths
```

Then any of:

```bash
# Download N most recent videos from each channel
bash scripts/download_corpus.sh

# Transcribe all *.audio.mp3 in media/ with full Deepgram add-ons
python scripts/transcribe_all.py

# Smoke-test on a single video id (use --only= because IDs may start with '-')
python scripts/transcribe_all.py --only=<video_id>

# Classify each diarized speaker cluster against the 3 reference centroids
python scripts/identify_speakers.py
```

## Independence from v1

This repo is intended to be standalone. It does not import from, link to, or read paths inside `acq_search_retrieval/`. Scripts here use absolute paths derived from `PROJECT_ROOT` in `config.py`.
