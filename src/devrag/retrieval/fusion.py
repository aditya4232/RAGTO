from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def normalize_minmax(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(d, 1.0) for d, _ in pairs]
    return [(d, (s - lo) / (hi - lo)) for d, s in pairs]


def weighted_sum(
    rankings: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights, strict=True):
        for _, s in ranking:
            scores[s] = scores.get(s, 0.0) + w * s
    return sorted(scores.items(), key=lambda x: -x[1])