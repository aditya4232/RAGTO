from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def stable_doc_id(*parts: str) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def dedup_by_embedding(
    chunks: list[Any],
    embeddings: np.ndarray,
    threshold: float = 0.92,
) -> tuple[list[Any], np.ndarray]:
    if len(chunks) == 0:
        return chunks, embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    e = embeddings / norms
    sim = e @ e.T
    keep: list[int] = []
    blocked: set[int] = set()
    for i in range(len(chunks)):
        if i in blocked:
            continue
        keep.append(i)
        dup = np.where(sim[i] > threshold)[0]
        for j in dup:
            if j != i:
                blocked.add(int(j))
    return [chunks[i] for i in keep], embeddings[keep]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out