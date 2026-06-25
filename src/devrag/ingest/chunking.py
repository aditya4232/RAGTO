from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from chonkie import SemanticChunker

from devrag.types import Chunk

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _safe_chunker(strategy: str, chunk_size: int, overlap: int) -> Any:
    try:
        return SemanticChunker(chunk_size=chunk_size, overlap=overlap)
    except Exception:
        from chonkie import RecursiveChunker

        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)


def chunk_documents(
    documents: Iterable[dict[str, Any]],
    *,
    strategy: str = "semantic",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    chunker = _safe_chunker(strategy, chunk_size, overlap)
    out: list[Chunk] = []
    for doc in documents:
        text = normalize_text(doc.get("text", ""))
        if not text:
            continue
        try:
            pieces = chunker(text)
            pieces_text = [p.text if hasattr(p, "text") else str(p) for p in pieces]
        except Exception:
            pieces_text = _fallback_split(text, chunk_size, overlap)
        for i, piece in enumerate(pieces_text):
            piece = normalize_text(piece)
            if len(piece) < 40:
                continue
            chunk_id = f"{doc['doc_id']}::chunk_{i:04d}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc["doc_id"],
                    text=piece,
                    source=doc.get("source", ""),
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    metadata={"char_len": len(piece), "chunk_index": i},
                )
            )
    return out


def _fallback_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, max(len(text) - chunk_size, 1), step)]