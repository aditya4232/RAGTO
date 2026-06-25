from __future__ import annotations

from typing import Any

from devrag.generation.client import LLMClient


class QueryRewriter:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        if self.llm is None or n <= 1:
            return [query]
        prompt = (
            "Generate {n} alternative search queries that capture the same intent as the user's "
            "question. Return them one per line, no numbering.\n\nQuestion: {q}\n\nQueries:"
        ).format(n=n, q=query)
        out = self.llm.complete(prompt, temperature=0.4, max_tokens=200).strip()
        variants = [line.strip("-* \t") for line in out.splitlines() if line.strip()]
        return [query] + variants[:n]

    def hyde(self, query: str) -> str:
        if self.llm is None:
            return query
        prompt = (
            "Write a short, factual passage (3-5 sentences) that would answer the question below. "
            "Do not hedge; do not cite; just write the passage as if it were a snippet from a "
            "high-quality technical document.\n\nQuestion: {q}\n\nPassage:"
        ).format(q=query)
        return self.llm.complete(prompt, temperature=0.3, max_tokens=220).strip() or query