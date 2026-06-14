"""Per-feature A/B harness.

For each query in queries.json[feature]:
  1. Compute baseline scores (production hybrid).
  2. Compute variant scores = baseline + alpha * feature_boost.
  3. Take top-20 from each.
  4. LLM-judge the union of top-10 from each, on 0/1/2.
  5. Aggregate metrics per query, then per category (target/confounder/regression).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from scorers import SCORERS  # type: ignore
from judge import judge_results  # type: ignore


def _ndcg(labels: list[int], k: int = 10) -> float:
    labels = labels[:k]
    if not labels:
        return 0.0
    dcg = sum((2 ** lab - 1) / math.log2(i + 2) for i, lab in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum((2 ** lab - 1) / math.log2(i + 2) for i, lab in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def run_feature(feature: str, k_judge: int = 10, k_pool: int = 20) -> dict:
    from lib.hybrid import load_index, embed_query_text, score_query  # noqa: E402

    queries = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))[feature]
    scorer = SCORERS[feature]()

    print(f"\n=== Loading hybrid index ===")
    state = load_index()
    print(f"  {len(state.meta)} frames indexed\n")

    per_query: list[dict] = []
    for qi, qrec in enumerate(queries):
        q = qrec["q"]
        cat = qrec["cat"]
        print(f"[{feature}] {qi+1:>2}/{len(queries)} ({cat:>10}) {q}")

        t0 = time.perf_counter()
        qv = embed_query_text(state, q)
        base = score_query(state, qv, query_text=q)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        boost = scorer.boost(state, q)
        variant = base + boost
        t_var = (time.perf_counter() - t0) + t_base

        base_top = list(np.argsort(-base)[:k_pool])
        var_top = list(np.argsort(-variant)[:k_pool])

        union = list(dict.fromkeys(base_top[:k_judge] + var_top[:k_judge]))
        labels_map: dict[int, int] = {}
        labels_list = judge_results(state, q, union)
        for idx, lab in zip(union, labels_list):
            labels_map[idx] = lab

        base_labels = [labels_map.get(i, 0) for i in base_top[:k_judge]]
        var_labels = [labels_map.get(i, 0) for i in var_top[:k_judge]]

        ndcg_base = _ndcg(base_labels, k_judge)
        ndcg_var = _ndcg(var_labels, k_judge)
        mean_base = float(np.mean(base_labels)) if base_labels else 0.0
        mean_var = float(np.mean(var_labels)) if var_labels else 0.0

        # Confounder FP: variant promotes a label-0 result into top-3 that baseline did not.
        fp = 0
        if cat == "confounder":
            base_top3 = set(base_top[:3])
            for idx in var_top[:3]:
                if labels_map.get(idx, 0) == 0 and idx not in base_top3:
                    fp = 1
                    break

        top1_win = 0
        if base_labels and var_labels:
            if var_labels[0] > base_labels[0]:
                top1_win = 1
            elif var_labels[0] < base_labels[0]:
                top1_win = -1

        per_query.append({
            "q": q, "cat": cat,
            "ndcg_base": round(ndcg_base, 4), "ndcg_var": round(ndcg_var, 4),
            "ndcg_delta": round(ndcg_var - ndcg_base, 4),
            "mean_base": round(mean_base, 4), "mean_var": round(mean_var, 4),
            "mean_delta": round(mean_var - mean_base, 4),
            "top1_win": top1_win,
            "confounder_fp": fp,
            "latency_base_ms": int(t_base * 1000),
            "latency_var_ms": int(t_var * 1000),
            "base_top": [{"idx": int(i), "video": state.meta[i]["video_id"], "scene": int(state.meta[i]["scene_idx"]), "label": labels_map.get(i, 0)} for i in base_top[:5]],
            "var_top":  [{"idx": int(i), "video": state.meta[i]["video_id"], "scene": int(state.meta[i]["scene_idx"]), "label": labels_map.get(i, 0)} for i in var_top[:5]],
        })

    summary = _summarize(per_query)
    out = {"feature": feature, "alpha": float(scorer.alpha), "summary": summary, "per_query": per_query}

    out_path = RESULTS_DIR / f"{feature}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{feature}] wrote {out_path}\n")
    _print_summary(feature, summary)
    return out


def _summarize(rows: list[dict]) -> dict:
    cats = {"target": [], "confounder": [], "regression": []}
    for r in rows:
        cats[r["cat"]].append(r)

    def agg(rs: list[dict]) -> dict:
        if not rs:
            return {}
        n = len(rs)
        return {
            "n": n,
            "ndcg_base_mean": round(float(np.mean([r["ndcg_base"] for r in rs])), 4),
            "ndcg_var_mean":  round(float(np.mean([r["ndcg_var"]  for r in rs])), 4),
            "ndcg_delta_mean": round(float(np.mean([r["ndcg_delta"] for r in rs])), 4),
            "mean_label_base": round(float(np.mean([r["mean_base"] for r in rs])), 4),
            "mean_label_var":  round(float(np.mean([r["mean_var"]  for r in rs])), 4),
            "mean_label_delta": round(float(np.mean([r["mean_delta"] for r in rs])), 4),
            "top1_wins": int(sum(1 for r in rs if r["top1_win"] > 0)),
            "top1_losses": int(sum(1 for r in rs if r["top1_win"] < 0)),
            "top1_net": int(sum(r["top1_win"] for r in rs)),
            "confounder_fp_rate": round(float(np.mean([r["confounder_fp"] for r in rs])), 4),
            "latency_delta_ms_mean": int(np.mean([r["latency_var_ms"] - r["latency_base_ms"] for r in rs])),
        }

    out = {cat: agg(rows) for cat, rows in cats.items()}
    out["verdict"] = _verdict(out)
    return out


def _verdict(s: dict) -> str:
    tgt = s.get("target", {})
    cnf = s.get("confounder", {})
    reg = s.get("regression", {})

    target_pass = (
        tgt.get("ndcg_delta_mean", 0) >= 0.05
        and tgt.get("top1_net", 0) >= 3
        and tgt.get("mean_label_delta", 0) >= 0.10
    )
    confounder_pass = (
        cnf.get("ndcg_delta_mean", 0) >= -0.02
        and cnf.get("mean_label_delta", 0) >= -0.05
        and cnf.get("confounder_fp_rate", 0) <= 0.15
    )
    regression_pass = (
        reg.get("ndcg_delta_mean", 0) >= -0.02
        and reg.get("mean_label_delta", 0) >= -0.05
    )

    if target_pass and confounder_pass and regression_pass:
        return "SHIP"
    if (tgt.get("ndcg_delta_mean", 0) >= 0.02 and confounder_pass and regression_pass):
        return "MARGINAL"
    return "NO-SHIP"


def _print_summary(feature: str, s: dict):
    print(f"--- {feature} verdict: {s.get('verdict')} ---")
    for cat in ["target", "confounder", "regression"]:
        c = s.get(cat, {})
        if not c:
            continue
        print(
            f"  {cat:>10}  n={c['n']:>2}  "
            f"nDCG d={c['ndcg_delta_mean']:+.3f}  "
            f"mean-label d={c['mean_label_delta']:+.3f}  "
            f"top1 net={c['top1_net']:+d}  "
            f"FP={c['confounder_fp_rate']*100:.0f}%  "
            f"lat d={c['latency_delta_ms_mean']}ms"
        )
