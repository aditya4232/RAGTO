from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from devrag.types import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def fit(self, chunks: Iterable[Chunk]) -> "BM25Index":
        self.chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25Index not fitted")
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def save(self, dir_path: str | Path) -> None:
        import pickle

        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "bm25.pkl").open("wb") as f:
            pickle.dump(
                {"k1": self.k1, "b": self.b, "chunks": [c.to_dict() for c in self.chunks]},
                f,
            )

    @classmethod
    def load(cls, dir_path: str | Path) -> "BM25Index":
        import pickle

        p = Path(dir_path)
        with (p / "bm25.pkl").open("rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.chunks = [Chunk.from_dict(d) for d in data["chunks"]]
        tokenized = [tokenize(c.text) for c in idx.chunks]
        idx.bm25 = BM25Okapi(tokenized, k1=idx.k1, b=idx.b)
        return idx