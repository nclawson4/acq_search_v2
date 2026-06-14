# Scoring criteria

Each of the four features is judged **independently**. A feature is one of: SHIP / MARGINAL / NO-SHIP.

## What we measure (per query)

For each query we generate two ranked lists of 20 results:

- **Baseline** — current production scoring: `w_visual=0.5, w_transcript=0.2, w_segment=0.3`.
- **Variant** — `baseline + α · feature_signal`, with α tuned per feature in `scorers.py`.

We take the union of top-10 from each list and label every result with `gpt-4o-mini` using a per-query rubric:

- **2** — directly addresses the query
- **1** — clearly relevant, secondary angle
- **0** — unrelated or wrong

Judge calls are cached on disk by `(query, frame_path)` so re-runs are free.

## Metrics

| Metric | Definition |
|---|---|
| **nDCG@10** | discounted cumulative gain at 10 over judge labels (0/1/2), normalized |
| **mean judge score @10** | mean label across the top 10 |
| **recall@10** | fraction of judge-positive (label ≥ 1) results in top-10 over the union |
| **top-1 wins / losses** | paired: count queries where variant's top-1 is strictly better / worse than baseline's |
| **confounder FP rate** | fraction of confounder queries where variant promotes a label-0 result into top-3 that wasn't in baseline's top-3 |
| **latency Δ** | mean per-query wall-clock difference |

## Query categories

Each feature has **20 queries** in `queries.json`:

- **12 TARGET** — the feature should clearly help.
- **4 CONFOUNDER** — the feature could falsely fire (e.g. a CARDINAL like "first" for an entity query that isn't about a number). Measures harm.
- **4 REGRESSION** — query unrelated to the feature (visual/format/recency). Variant should look ~identical to baseline. Measures unwanted perturbation.

## Decision thresholds

Per feature, evaluated against the matching category subset:

| Metric (top-10) | Target queries (n=12) | Confounder queries (n=4) | Regression queries (n=4) |
|---|---|---|---|
| nDCG@10 Δ vs baseline | **≥ +0.05** to ship | **≥ −0.02** | **≥ −0.02** |
| Mean judge score Δ | ≥ +0.10 | ≥ −0.05 | ≥ −0.05 |
| Top-1 net wins | ≥ +3 of 12 | n/a | within ±1 |
| Confounder FP rate | n/a | **≤ 15%** | n/a |
| Latency Δ per query | ≤ +100 ms | same | same |

- **SHIP** — all targets green; nothing in the confounder/regression columns red.
- **MARGINAL** — Δs positive but inside the noise floor (target nDCG Δ in [+0.02, +0.05)). Don't ship until either α is tuned or queries broaden.
- **NO-SHIP** — any cell red, or target gains fail to clear noise.

## Why these thresholds

- **nDCG +0.05** is roughly twice the run-to-run variance we'd expect from a 12-query LLM judge (judge variance ≈ ±0.02-0.03 nDCG).
- **Top-1 net +3 of 12** is binomial-significant at p≈0.05 against a 50/50 null.
- **Confounder FPR 15%** caps the rate at which the feature actively hurts adjacent queries. Below this is acceptable noise; above it the feature is broken.
- **Regression Δ ≥ −0.02** says the feature must be effectively a no-op for unrelated queries. Otherwise it's adding noise system-wide.
