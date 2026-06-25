from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from devrag.ingest.utils import read_jsonl, write_jsonl
from devrag.logging_setup import get_logger

log = get_logger(__name__)

_CITE_RE = re.compile(r"\[\d+([,\s\d\-,]+)?\]")


def mine_pairs_from_arxiv(
    documents_path: str | Path,
    out_path: str | Path,
    *,
    max_pairs: int = 30_000,
) -> int:
    docs = read_jsonl(documents_path)
    pairs: list[dict[str, str]] = []
    for d in docs:
        title = d.get("title", "").strip()
        text = d.get("text", "").strip()
        if not title or not text:
            continue
        query = f"What does the paper '{title}' say about {title.split()[0] if title.split() else title}?"
        sentences = _split_sentences(text)
        if not sentences:
            continue
        positive = " ".join(sentences[:3])[:1200]
        pairs.append({"query": query, "positive": positive})
        if len(pairs) >= max_pairs:
            break
    n = write_jsonl(out_path, pairs)
    log.info("mining.arxiv.done", n=n, path=str(out_path))
    return n


def mine_pairs_from_stackoverflow(
    out_path: str | Path,
    *,
    tags: Iterable[str] = ("transformers", "langchain", "pytorch", "huggingface"),
    max_pairs: int = 20_000,
) -> int:
    try:
        from datasets import load_dataset
    except Exception:
        log.warning("mining.stackoverflow.unavailable")
        return 0
    pairs: list[dict[str, str]] = []
    try:
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
    except Exception as e:
        log.warning("mining.stackoverflow.fail", error=str(e))
        return 0
    seen = 0
    for ex in ds:
        seen += 1
        q = ex.get("question", "")
        ans = ex.get("answers", [])
        if not q or not ans:
            continue
        a_text = ans[0].get("text", "") if isinstance(ans[0], dict) else str(ans[0])
        if not a_text:
            continue
        pairs.append({"query": q[:512], "positive": a_text[:1200]})
        if len(pairs) >= max_pairs:
            break
        if seen >= max_pairs * 4:
            break
    n = write_jsonl(out_path, pairs)
    log.info("mining.stackoverflow.done", n=n, path=str(out_path))
    return n


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 30]