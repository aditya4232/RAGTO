from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    title: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            text=d["text"],
            source=d.get("source", ""),
            title=d.get("title", ""),
            url=d.get("url", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
    origin: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": float(self.score),
            "rank": int(self.rank),
            "origin": self.origin,
        }


@dataclass
class Answer:
    text: str
    citations: list[str]
    hits: list[RetrievalHit]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": self.citations,
            "hits": [h.to_dict() for h in self.hits],
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class EvalExample:
    query: str
    relevant_chunk_ids: list[str]
    reference_answer: str = ""
    query_type: str = "factual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "reference_answer": self.reference_answer,
            "query_type": self.query_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalExample":
        return cls(
            query=d["query"],
            relevant_chunk_ids=list(d.get("relevant_chunk_ids", [])),
            reference_answer=d.get("reference_answer", ""),
            query_type=d.get("query_type", "factual"),
        )