from __future__ import annotations

import re
import time
from pathlib import Path

from devrag.config import LLMConfig
from devrag.generation.client import LLMClient
from devrag.generation.prompts import build_messages, extract_citations, load_system_prompt
from devrag.observability.tracing import trace_request
from devrag.retrieval.hybrid import HybridRetriever
from devrag.types import Answer, RetrievalHit


_CITE_RE = re.compile(r"\[([a-zA-Z0-9_./:_\-]+)\]")


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        system_prompt_file: str | Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif system_prompt_file is not None:
            self.system_prompt = load_system_prompt(system_prompt_file)
        else:
            self.system_prompt = load_system_prompt("configs/system_prompt.txt")

    def answer(
        self,
        query: str,
        *,
        ablation: str = "hybrid_rerank_rewrite",
        top_k: int | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        with trace_request("rag.answer", {"query_len": len(query), "ablation": ablation}) as t:
            hits = self.retriever.retrieve(query, ablation=ablation, top_k=top_k)
            messages = build_messages(query, hits, system_prompt=self.system_prompt)
            text = self.llm.chat(messages)
            citations = extract_citations(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            t.add("hits", len(hits))
            t.add("citations", len(citations))
            t.add("latency_ms", round(latency_ms, 2))
            return Answer(
                text=text,
                citations=citations,
                hits=hits,
                latency_ms=latency_ms,
                metadata={"ablation": ablation},
            )


def build_default_pipeline(
    retrieval_config_path: str | Path,
    llm_config: LLMConfig,
) -> RAGPipeline:
    from devrag.config import load_retrieval_config

    rcfg = load_retrieval_config(retrieval_config_path)
    retriever = HybridRetriever.load(rcfg, index_dir=Path(rcfg.indexing.output_dir))
    llm = LLMClient(llm_config)
    return RAGPipeline(retriever, llm)