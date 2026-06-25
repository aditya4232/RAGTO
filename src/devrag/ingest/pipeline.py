from __future__ import annotations

from pathlib import Path
from typing import Any

from devrag.config import CorpusConfig
from devrag.ingest.chunking import chunk_documents
from devrag.ingest.harvest import harvest_arxiv, harvest_blogs, harvest_framework_docs
from devrag.ingest.utils import write_jsonl
from devrag.logging_setup import get_logger
from devrag.types import Chunk

log = get_logger(__name__)


def build_corpus(config: CorpusConfig) -> dict[str, Any]:
    sources = config.sources or {}
    documents: list[dict[str, Any]] = []

    arxiv_cfg = sources.get("arxiv", {})
    if arxiv_cfg.get("enabled", True):
        documents.extend(
            harvest_arxiv(
                categories=list(arxiv_cfg.get("categories", [])),
                lookback_months=int(arxiv_cfg.get("lookback_months", 18)),
                max_docs=int(arxiv_cfg.get("max_docs", 20_000)),
            )
        )

    docs_cfg = sources.get("framework_docs", {})
    if docs_cfg.get("enabled", False):
        documents.extend(harvest_framework_docs(docs_cfg.get("urls", [])))

    blogs_cfg = sources.get("blogs", {})
    if blogs_cfg.get("enabled", False):
        documents.extend(harvest_blogs(blogs_cfg.get("feeds", [])))

    log.info("corpus.documents", n=len(documents))

    chunks = chunk_documents(
        documents,
        strategy=config.chunking.strategy,
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap,
    )
    log.info("corpus.chunks", n=len(chunks))

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    write_jsonl(chunks_path, (c.to_dict() for c in chunks))

    docs_path = out_dir / "documents.jsonl"
    write_jsonl(docs_path, documents)

    return {
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "chunks_path": str(chunks_path),
        "documents_path": str(docs_path),
    }


def load_chunks(path: str | Path) -> list[Chunk]:
    from devrag.ingest.utils import read_jsonl
    from devrag.types import Chunk

    return [Chunk.from_dict(d) for d in read_jsonl(path)]