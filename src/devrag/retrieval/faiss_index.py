from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class FAISSIndex:
    def __init__(self, dim: int, index_type: str = "hnsw", hnsw_m: int = 32) -> None:
        import faiss

        self.dim = dim
        if index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
        elif index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"unknown index_type: {index_type}")
        self._ids: list[str] = []

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self._ids.extend(ids)

    def search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        import faiss

        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        q = query_emb.astype(np.float32).copy()
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        out: list[tuple[str, float]] = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            out.append((self._ids[i], float(s)))
        return out

    def save(self, dir_path: str | Path) -> None:
        import faiss

        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "faiss.index"))
        with (p / "ids.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(self._ids))

    @classmethod
    def load(cls, dir_path: str | Path) -> "FAISSIndex":
        import faiss

        p = Path(dir_path)
        index = faiss.read_index(str(p / "faiss.index"))
        ids = (p / "ids.txt").read_text(encoding="utf-8").splitlines()
        obj = cls(dim=index.d, index_type="hnsw" if isinstance(index, faiss.IndexHNSWFlat) else "flat")
        obj.index = index
        obj._ids = ids
        return obj