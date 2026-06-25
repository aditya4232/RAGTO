from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from devrag.types import EvalExample, RetrievalHit


def recall_at_k(retrieved: list[RetrievalHit], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not relevant:
        return 0.0
    return len({h.chunk.chunk_id for h in top} & relevant) / max(len(relevant), 1)


def mrr_at_k(retrieved: list[RetrievalHit], relevant: set[str], k: int) -> float:
    for i, h in enumerate(retrieved[:k]):
        if h.chunk.chunk_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[RetrievalHit], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, h in enumerate(retrieved[:k]):
        rel = 1.0 if h.chunk.chunk_id in relevant else 0.0
        dcg += (2**rel - 1) / math.log2(i + 2)
    ideal = sum((2**1 - 1) / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retrieval(
    examples: Iterable[EvalExample],
    retrieve_fn,
    *,
    ks: Iterable[int] = (1, 5, 10),
    top_k: int = 50,
) -> dict[str, float]:
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    out.update({f"mrr@{k}": [] for k in ks})
    out.update({f"ndcg@{k}": [] for k in ks})
    by_type: dict[str, dict[str, list[float]]] = {}
    latencies: list[float] = []
    for ex in examples:
        import time

        t0 = time.perf_counter()
        hits = retrieve_fn(ex.query)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        relevant = set(ex.relevant_chunk_ids)
        bucket = by_type.setdefault(ex.query_type, {k: [] for k in out})
        for k in ks:
            r = recall_at_k(hits, relevant, k)
            m = mrr_at_k(hits, relevant, k)
            n = ndcg_at_k(hits, relevant, k)
            out[f"recall@{k}"].append(r)
            out[f"mrr@{k}"].append(m)
            out[f"ndcg@{k}"].append(n)
            bucket[f"recall@{k}"].append(r)
            bucket[f"mrr@{k}"].append(m)
            bucket[f"ndcg@{k}"].append(n)
    summary: dict[str, float] = {m: float(np.mean(v)) if v else 0.0 for m, v in out.items()}
    summary["latency_p50_ms"] = float(np.percentile(latencies, 50)) if latencies else 0.0
    summary["latency_p95_ms"] = float(np.percentile(latencies, 95)) if latencies else 0.0
    summary["latency_mean_ms"] = float(np.mean(latencies)) if latencies else 0.0
    summary["n_examples"] = float(len(latencies))
    summary["by_query_type"] = {
        qt: {k: float(np.mean(v)) if v else 0.0 for k, v in m.items()}
        for qt, m in by_type.items()
    }
    return summary


def evaluate_end_to_end(
    examples: Iterable[EvalExample],
    answer_fn,
    *,
    judge,
    sample_size: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    pool = list(examples)
    if not pool:
        return {"n": 0}
    pool = pool[:sample_size] if len(pool) > sample_size else pool
    rng.shuffle(pool)

    faithfulness: list[float] = []
    citation_precision: list[float] = []
    answer_relevance: list[float] = []
    latencies: list[float] = []
    for ex in pool:
        import time

        t0 = time.perf_counter()
        ans = answer_fn(ex.query)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        cited = {c for c in ans.citations if c}
        cited_real = sum(1 for c in cited if any(h.chunk.chunk_id == c for h in ans.hits))
        cp = cited_real / max(len(cited), 1)
        citation_precision.append(cp)
        f = judge.faithfulness(ex.query, ex.reference_answer, ans.text, ans.hits)
        r = judge.relevance(ex.query, ans.text)
        faithfulness.append(f)
        answer_relevance.append(r)
    return {
        "n": float(len(pool)),
        "faithfulness": float(np.mean(faithfulness)) if faithfulness else 0.0,
        "citation_precision": float(np.mean(citation_precision)) if citation_precision else 0.0,
        "answer_relevance": float(np.mean(answer_relevance)) if answer_relevance else 0.0,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def save_report(report: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)