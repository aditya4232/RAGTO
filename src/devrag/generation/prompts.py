from __future__ import annotations

import re
from pathlib import Path

from devrag.types import RetrievalHit


_CITE_RE = re.compile(r"\[([a-zA-Z0-9_./:_\-]+)\]")


def load_system_prompt(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return _DEFAULT_SYSTEM_PROMPT
    return p.read_text(encoding="utf-8")


def build_messages(
    query: str,
    hits: list[RetrievalHit],
    *,
    system_prompt: str,
) -> list[dict[str, str]]:
    context_blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        context_blocks.append(
            f"[source={c.chunk_id}] (title={c.title!r}, url={c.url})\n{c.text}\n"
        )
    context = "\n\n".join(context_blocks)
    user = (
        f"Question:\n{query}\n\n"
        f"Context (cite every claim with the bracketed source id):\n\n{context}\n\n"
        "Answer with inline citations in the form [source_id]. If the context is insufficient, "
        "say so plainly."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for m in _CITE_RE.findall(text):
        if m not in seen:
            seen.append(m)
    return seen


_DEFAULT_SYSTEM_PROMPT = (
    "You are DevRAG, a careful research assistant. Answer using ONLY the provided context. "
    "Cite every factual claim with [source_id]. If the answer is not in the context, say so."
)