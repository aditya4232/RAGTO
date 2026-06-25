from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from devrag.generation.client import LLMClient
from devrag.types import RetrievalHit


class LLMJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def faithfulness(
        self, query: str, reference: str, answer: str, hits: list[RetrievalHit]
    ) -> float:
        ctx = "\n\n".join(h.chunk.text[:600] for h in hits[:5])
        prompt = (
            "Rate the faithfulness of the ANSWER to the CONTEXT on a scale 0-5.\n"
            "5 = every claim is supported by CONTEXT.\n"
            "0 = answer contradicts or invents beyond CONTEXT.\n\n"
            f"QUESTION: {query}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\n"
            "Reply with only a single integer 0-5."
        )
        return _parse_score(self.llm.complete(prompt, temperature=0.0, max_tokens=4))

    def relevance(self, query: str, answer: str) -> float:
        prompt = (
            "Rate how relevant the ANSWER is to the QUESTION on a scale 0-5.\n"
            "5 = directly and fully answers the question.\n"
            "0 = off-topic.\n\n"
            f"QUESTION: {query}\n\nANSWER:\n{answer}\n\n"
            "Reply with only a single integer 0-5."
        )
        return _parse_score(self.llm.complete(prompt, temperature=0.0, max_tokens=4))


def _parse_score(text: str) -> float:
    m = re.search(r"\b([0-5])\b", text)
    if not m:
        return 0.0
    return float(m.group(1)) / 5.0