# Search the ACQ media database

Alex, Leila, and Sharran's long-form videos analyzed from the last 6 months.

**Live demo: https://acq-search-v2.vercel.app**

Type the moment you want in plain English. The system extracts the speaker, time window, topic, and visual format from the query, runs hybrid retrieval over CLIP frame embeddings, Deepgram transcripts, and per-segment text vectors, then ranks the candidates with a content judge. An editor finds the clip in seconds instead of minutes.

## Try a search

- Leila talking about leadership, talking head video
- Alex talking about churn
- Sharran less than 3 weeks ago talking about real estate
- Animations talking about stress and anxiety
- Alex writing on a whiteboard
- Leila and Sharran talking together

Each of these is a real query the live system answers correctly. The home page shows what the parser extracted, the top result, and a one-sentence verification for all six.

## Pipeline

Five stages. Every stage has a measured metric and meets its target.

| Stage | What it does | Model | Headline metric |
|---|---|---|---|
| Scene detection | Cuts each video into scenes at visual transitions | PySceneDetect ContentDetector (threshold 27) | **100%** (deterministic) |
| Visual frame retrieval | Finds visually similar keyframes from a text query | open_clip ViT-L-14 / LAION2B-32B-B82K | **recall@50: 94.5%** |
| Hybrid retrieval | Combines visual, transcript, and segment-text signals into one ranking | CLIP visual 0.5 + BM25 transcript 0.2 + CLIP segment-text 0.3 | **recall@20 (original split): 95.7%** |
| Visual format classifier | Tags each keyframe as animated vs live, with or without a clear talking head | gpt-4o-mini vision (2-field schema) | **is_animation 90% / talking_head_pose 90%** |
| Query parser | Reads the editor's natural-language query and produces a structured search plan | gpt-4o-mini, temperature 0, JSON mode | **91% (41/45 regression cases)**, +23% p@5 over the prior regex parser |

Every number above is cited to a script in `eval/`:

- `eval/clip_eval.py` and `eval/data/full_eval_result.log` for CLIP recall sweeps
- `eval/hybrid_eval.py` and `eval/data/hybrid_eval_70_30.log` for hybrid retrieval per-split
- `eval/visual_classifier_v2_eval.py` for the 90% pass threshold
- `eval/query_parser_eval.py` and `eval/parser_ab.py` for parser quality and the A/B vs the prior regex parser

The 95% recall target is on the production retrieval stage. The hybrid layer clears it on the original split at K=20 and clears it across every split at K=50.

## Golden sets

Six real searches an editor might run, all sent to the live system. Full responses are saved to `web/src/data/golden-queries.json` and the script that produces them lives at `web/scripts/run-golden-queries.mjs`. Re-running the script refreshes the data the home page renders.

A few representative cases:

**"Sharran less than 3 weeks ago talking about real estate"**

Parser extracts `speaker=sharran`, `max_age_days=21`, `retrieval_query="real estate"`. Backend returns only 2 scenes in the corpus that satisfy all three constraints. Top result judge score 1.0. The system is honest about scarcity rather than padding the list.

**"Animations talking about stress and anxiety"**

Parser extracts `visual_concept="Animations"`. CLIP visual retrieval handles the verification, and the transcript-only judge is skipped via the visual-only short-circuit because transcripts rarely mention the visual format by name. All 5 returned scenes are classifier-tagged as animations.

**"Leila and Sharran talking together"**

Parser extracts `required_speakers=[leila, sharran]` and `speakers_count=dialogue`. Backend returns 3 scenes; every one of them has both Leila and Sharran in `voices_present`.

## Failure recovery

### Monitoring

- `/api/status` runs a liveness check on the backend plus a real smoke search against the same code path the editor hits. It runs every page load and re-runs every 30 seconds while the page is open.
- Backend `/healthz` returns indexed frame and segment counts so partial loads are detectable.
- Client surfaces server errors verbatim instead of failing silently.
- Staged progress bar names what the system is doing at each step so cold starts read as activity, not as a hang.

### Recovery on failure

- Code is on a tagged baseline. Rollback is one Vercel or Modal CLI call.
- Caches, vectors, and frame images are bundled into the Modal image. They are not pulled from a third party at request time.
- CLIP model weights are baked into the build image, so there is no HuggingFace dependency at runtime.
- OpenAI is the only external dependency on the hot path; rate-limit retries live in the OpenAI client.

## Cost

### Build the index (one-time per 1 TB of source video)

1 TB ≈ 250 hours (15,000 minutes) of source video at typical 1080p master bitrate.

| Line item | Cost |
|---|---|
| Deepgram nova-3 transcription | ≈ $65 |
| Visual classifier (gpt-4o-mini vision) | ≈ $3 |
| CLIP frame embedding (CPU compute) | ≈ $6 |
| Scene detection | ≈ $2 |
| Topic segment curation (LLM-generated) | ≈ $100 |
| Storage (Vercel Blob, npz, transcripts) | < $1/mo |
| **Total per TB (one-time cost)** | **≈ $180** |

Topic segmentation dominates. Everything else combined sits under $80 per TB.

### Serve a query

| Line item | Cost |
|---|---|
| Query parser (gpt-4o-mini) | ≈ $0.0002 |
| Modal CPU (FastAPI backend, 2 vCPU) | ≈ $0.002 |
| Vercel Functions (proxy) | < $0.0001 |
| Vercel Blob frame serving | < $0.0001 |
| **Total per query** | **≈ $0.002** |

At 1,000 queries per day: about $2/day. At 10,000/day: about $20/day. Modal compute dominates; the parser call is rounding error.

### Editor time saved

Without the system: search transcripts, jump to timestamps, watch candidate clips. **5 to 15 minutes per moment**, on a query the editor knows exists.

With the system: type in plain English, scan 5 thumbnails, click. **About 2 minutes per moment.**

| | |
|---|---|
| Saved per query | ~5 minutes |
| Per editor per year | ~208 hours |
| Per editor per year | $10K to $16K (at a $50 to $75/hour blended rate) |

Conservative assumptions: 10 queries/day, 5 days/week, 50 weeks/year. The baseline assumes the editor already has searchable transcripts. Excludes clips that would not have been found at all.

## Architecture

```
                              ┌─ Modal: FastAPI backend (CPU, scale-to-zero)
                              │    ├─ CLIP ViT-L-14 (LAION2B-32B-B82K)
Browser ───> Vercel Next.js ──┤    ├─ Deepgram transcripts + scene metadata
                              │    ├─ Hybrid scoring + visual-only short-circuit
                              │    └─ OpenAI (parser + judge)
                              │
                              └─ Vercel Blob: keyframe thumbnails (public CDN)
```

- Frontend: Next.js 16 (App Router) on Vercel
- Backend: FastAPI on Modal serverless, in-memory numpy hybrid scoring
- Vectors: 15,621 keyframe CLIP embeddings and 636 segment-text embeddings stored as `.npz`
- Thumbnails: ~20,000 JPEGs on Vercel Blob, served via a 302 redirect from the Next.js frames route
- LLM: gpt-4o-mini for both query parsing and relevance reranking

## Local setup

Ingest scripts (transcription, scene detection, embedding):

```bash
cd ingest
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in DEEPGRAM_API_KEY, OPENAI_API_KEY, ...
```

Backend (FastAPI):

```bash
cd ingest && .venv/Scripts/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Frontend (Next.js):

```bash
cd web && npm install && npm run dev
```

Deploy:

```bash
modal deploy ingest/modal_app.py
cd web && vercel deploy --prod
```

## Repo layout

```
acq_search_v2/
├── docs/                       project history + tagging schemas
├── eval/                       evaluation scripts and saved logs
├── ingest/
│   ├── api/main.py             FastAPI search endpoint
│   ├── lib/
│   │   ├── hybrid.py           CLIP + BM25 + segment-text scoring
│   │   ├── query_parser.py     gpt-4o-mini end-to-end query parser
│   │   ├── reranker.py         gpt-4o-mini relevance judge
│   │   └── structural.py       verified-dimension composer
│   ├── scripts/                ingest + index build
│   ├── modal_app.py            Modal deployment
│   └── qdrant_safe.py          v2-prefix guard wrapper
└── web/
    ├── src/app/api/
    │   ├── search/             proxy to the Modal backend
    │   ├── status/             liveness + smoke check
    │   └── frames/             302 redirect to Vercel Blob
    ├── src/components/
    │   ├── SearchView.tsx      search UI, progress bar, scene strip
    │   └── HomeSections.tsx    golden sets + benchmarks + cost
    └── src/data/
        ├── golden-queries.json saved responses from the live backend
        └── benchmarks.json     eval-derived metrics rendered on the page
```
