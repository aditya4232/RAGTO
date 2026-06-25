from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from devrag.config import RetrievalConfig
from devrag.ingest.pipeline import load_chunks
from devrag.logging_setup import get_logger
from devrag.retrieval.bm25 import BM25Index
from devrag.retrieval.dense import DenseEncoder
from devrag.retrieval.faiss_index import FAISSIndex
from devrag.retrieval.fusion import reciprocal_rank_fusion
from devrag.retrieval.query_rewrite import QueryRewriter
from devrag.retrieval.reranker import CrossEncoderReranker
from devrag.types import Chunk, RetrievalHit

log = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        *,
        bm25: BM25Index | None = None,
        dense_encoder: DenseEncoder | None = None,
        faiss_index: FAISSIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        rewriter: QueryRewriter | None = None,
        chunks: list[Chunk] | None = None,
    ) -> None:
        self.config = config
        self.bm25 = bm25
        self.dense = dense_encoder
        self.faiss = faiss_index
        self.reranker = reranker
        self.rewriter = rewriter
        self.chunks = chunks or []
        self._chunk_by_id = {c.chunk_id: c for c in self.chunks}

    def build(self, chunks: list[Chunk], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        self._chunk_by_id = {c.chunk_id: c for c in chunks}

        if self.config.bm25.enabled:
            log.info("retriever.build.bm25", n=len(chunks))
            self.bm25 = BM25Index(self.config.bm25.k1, self.config.bm25.b).fit(chunks)
            self.bm25.save(out_dir)

        if self.config.dense.enabled:
            log.info("retriever.build.dense", model=self.config.dense.model)
            self.dense = DenseEncoder(
                self.config.dense.model,
                lora_adapter=self.config.dense.lora_adapter,
                max_seq_length=512,
            )
            emb = self.dense.encode_chunks(chunks, batch_size=64, show_progress=True)
            self.faiss = FAISSIndex(
                dim=emb.shape[1],
                index_type=self.config.dense.index_type,
                hnsw_m=self.config.dense.hnsw_m,
            )
            self.faiss.add(emb, [c.chunk_id for c in chunks])
            np.save(out_dir / "dense_embeddings.npy", emb)
            self.faiss.save(out_dir)

        if self.config.reranker.enabled:
            log.info("retriever.build.reranker", model=self.config.reranker.model)
            self.reranker = CrossEncoderReranker(self.config.reranker.model)

    @classmethod
    def load(cls, config: RetrievalConfig, index_dir: str | Path) -> "HybridRetriever":
        index_dir = Path(index_dir)
        chunks = load_chunks(index_dir.parent / "processed" / "chunks.jsonl")
        bm25 = BM25Index.load(index_dir) if config.bm25.enabled else None
        dense = (
            DenseEncoder(config.dense.model, lora_adapter=config.dense.lora_adapter)
            if config.dense.enabled
            else None
        )
        faiss_idx = FAISSIndex.load(index_dir) if config.dense.enabled else None
        reranker = (
            CrossEncoderReranker(config.reranker.model)
            if config.reranker.enabled
            else None
        )
        return cls(
            config,
            bm25=bm25,
            dense_encoder=dense,
            faiss_index=faiss_idx,
            reranker=reranker,
            chunks=chunks,
        )

    def retrieve(
        self,
        query: str,
        *,
        ablation: str = "hybrid_rerank_rewrite",
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        top_k = top_k or self.config.final_top_k

        queries = [query]
        if ablation.endswith("_rewrite") and self.rewriter is not None:
            queries = self.rewriter.multi_query(query, self.config.query_rewrite.num_variants)

        rankings: list[list[tuple[str, float]]] = []

        if "bm25" in ablation or ablation.startswith("hybrid"):
            if self.bm25 is not None:
                bm25_hits: list[tuple[str, float]] = []
                seen: set[str] = set()
                for q in queries:
                    for cid, s in self.bm25.search(q, top_k=50):
                        if cid in seen:
                            continue
                        seen.add(cid)
                        bm25_hits.append((cid, s))
                rankings.append(bm25_hits[:50])

        if "dense" in ablation or ablation.startswith("hybrid"):
            if self.faiss is not None and self.dense is not None:
                embs = self.dense.encode(queries, batch_size=32, normalize=True)
                dense_hits: list[tuple[str, float]] = []
                seen = set()
                for e in embs:
                    for cid, s in self.faiss.search(e, top_k=50):
                        if cid in seen:
                            continue
                        seen.add(cid)
                        dense_hits.append((cid, s))
                rankings.append(dense_hits[:50])

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings, k=self.config.fusion.rrf_k)
        fused = fused[: max(self.config.reranker.top_k * 2, top_k)]

        if "rerank" in ablation and self.reranker is not None:
            docs = [(cid, self._chunk_by_id[cid].text) for cid, _ in fused if cid in self._chunk_by_id]
            reranked = self.reranker.rerank(query, docs, top_k=top_k)
            fused = [(cid, score) for cid, score in reranked]

        hits: list[RetrievalHit] = []
        for rank, (cid, score) in enumerate(fused[:top_k]):
            chunk = self._chunk_by_id.get(cid)
            if chunk is None:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=score, rank=rank, origin=ablation))
        return hits

    def batch_retrieve(
        self, queries: list[str], **kwargs: Any
    ) -> list[list[RetrievalHit]]:
        return [self.retrieve(q, **kwargs) for q in tqdm(queries, desc="retrieve")]