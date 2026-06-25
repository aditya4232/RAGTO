from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from devrag.config import LLMConfig, ServingConfig, load_retrieval_config
from devrag.generation.client import LLMClient
from devrag.generation.pipeline import RAGPipeline
from devrag.observability.cache import ResponseCache
from devrag.observability.tracing import new_request_id
from devrag.retrieval.hybrid import HybridRetriever


app = FastAPI(title="DevRAG-2026 API", version="0.1.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    ablation: str = "hybrid_rerank_rewrite"
    top_k: int | None = None
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float
    hits: list[dict[str, Any]]
    request_id: str


_pipeline: RAGPipeline | None = None
_cache: ResponseCache | None = None


def init_app(
    *,
    retrieval_config_path: str = "configs/retrieval.yaml",
    llm_config: LLMConfig | None = None,
    serving: ServingConfig | None = None,
    cache_path: str | None = None,
) -> FastAPI:
    global _pipeline, _cache

    rcfg = load_retrieval_config(retrieval_config_path)
    llm = llm_config or LLMConfig()
    retriever = HybridRetriever.load(rcfg, index_dir=rcfg.indexing.output_dir)
    _pipeline = RAGPipeline(retriever, llm, system_prompt_file=rcfg.llm.system_prompt_file)

    cp = cache_path or "data/cache/responses.sqlite"
    _cache = ResponseCache(cp, ttl_seconds=3600)
    return app


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ts": time.time()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(503, "pipeline not initialized")
    rid = new_request_id()
    cache_key = ResponseCache.make_key(req.query, req.ablation)
    if req.use_cache and _cache is not None:
        cached = _cache.get(cache_key)
        if cached is not None:
            return QueryResponse(request_id=rid, **cached)
    ans = _pipeline.answer(req.query, ablation=req.ablation, top_k=req.top_k)
    payload = {
        "answer": ans.text,
        "citations": ans.citations,
        "latency_ms": ans.latency_ms,
        "hits": [h.to_dict() for h in ans.hits],
    }
    if req.use_cache and _cache is not None:
        _cache.set(cache_key, payload)
    return QueryResponse(request_id=rid, **payload)