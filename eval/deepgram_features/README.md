# Deepgram Features A/B Eval

Throwaway evaluation branch (`eval/deepgram-features`) testing whether Deepgram's `entities`, `topics`, `sentiment`, and `summary` fields — which are already in `cache/deepgram/*.json` but ignored by production — would improve retrieval if wired in.

**Nothing here touches the production pipeline.** Production scoring is imported read-only from `ingest/lib/hybrid.py`. Feature-specific indexes write to `ingest/cache/eval_dg/` (also ignored by prod).

To revert: `git checkout main && git branch -D eval/deepgram-features && rm -rf ingest/cache/eval_dg`.

## Layout

```
eval/deepgram_features/
├── README.md          ← this file
├── criteria.md        ← ship/no-ship thresholds
├── extractors.py      ← build per-feature indexes from Deepgram cache
├── scorers.py         ← four feature scorers (each independent)
├── queries.json       ← 80 queries: 20 per feature, with category metadata
├── judge.py           ← gpt-4o-mini relevance judge with on-disk cache
├── harness.py         ← runs baseline vs baseline+feature, computes metrics
└── run_eval.py        ← CLI entry point
```

## How to run

```bash
# 1. Build indexes (one-time; reads cache/deepgram/, writes cache/eval_dg/)
python -m eval.deepgram_features.extractors --all

# 2. Run eval for one feature, or all four
python -m eval.deepgram_features.run_eval --feature entities
python -m eval.deepgram_features.run_eval --feature all

# Results land in eval/deepgram_features/results/{feature}.json with a console scorecard.
```

Required env: `OPENAI_API_KEY` (for the judge — same key the production reranker uses).

## How to interpret

See `criteria.md`. Headline: a feature is **SHIP** if it lifts nDCG@10 on TARGET queries by ≥0.05 without degrading CONFOUNDER or REGRESSION queries by more than 0.02.
