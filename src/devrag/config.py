from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SourceConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = Field(default_factory=list)
    lookback_months: int = 18
    max_docs: int = 20_000
    urls: list[str] = Field(default_factory=list)
    feeds: list[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    strategy: str = "semantic"
    chunk_size: int = 512
    overlap: int = 64


class DedupConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.92


class CorpusConfig(BaseModel):
    name: str = "devrag-corpus-2026"
    output_dir: str = "data/processed"
    sources: dict[str, Any] = Field(default_factory=dict)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)


class BM25Config(BaseModel):
    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75


class DenseConfig(BaseModel):
    enabled: bool = True
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    lora_adapter: str | None = None
    normalize: bool = True
    index_type: str = "hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200


class FusionConfig(BaseModel):
    method: str = "rrf"
    rrf_k: int = 60


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
    top_k: int = 8


class QueryRewriteConfig(BaseModel):
    multi_query: bool = True
    hyde: bool = True
    num_variants: int = 3


class RetrievalConfig(BaseModel):
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    final_top_k: int = 8


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "qwen3:8b"
    temperature: float = 0.2
    max_tokens: int = 1024
    system_prompt_file: str = "configs/system_prompt.txt"


class IndexingConfig(BaseModel):
    output_dir: str = "data/indexes"


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class OptimizationConfig(BaseModel):
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_accum: int = 4
    fp16: bool = True
    bf16: bool = False
    max_seq_length: int = 512


class DataConfig(BaseModel):
    train_file: str = "data/processed/pairs_train.jsonl"
    eval_file: str = "data/processed/pairs_eval.jsonl"
    max_pairs: int = 50_000


class PushToHubConfig(BaseModel):
    enabled: bool = False
    repo_id: str = ""
    private: bool = False


class TrainingConfig(BaseModel):
    output_dir: str = "data/models/devrag-embedding-2026"
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    loss: str = "info_nce"
    temperature: float = 0.05
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    push_to_hub: PushToHubConfig = Field(default_factory=PushToHubConfig)

    @field_validator("loss")
    @classmethod
    def _check_loss(cls, v: str) -> str:
        allowed = {"info_nce", "cosine", "mnrl"}
        if v not in allowed:
            raise ValueError(f"loss must be one of {allowed}, got {v!r}")
        return v


class EvalRetrievalConfig(BaseModel):
    dataset: str = "data/processed/eval_qa.jsonl"
    metrics: list[str] = Field(
        default_factory=lambda: ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    )
    ks: list[int] = Field(default_factory=lambda: [1, 5, 10])


class EvalEndToEndConfig(BaseModel):
    enabled: bool = True
    dataset: str = "data/processed/eval_qa.jsonl"
    metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "citation_precision", "answer_relevance"]
    )
    judge_backend: str = "local_llm"
    judge_model: str = "qwen3:8b"
    sample_size: int = 50


class EvalConfig(BaseModel):
    retrieval: EvalRetrievalConfig = Field(default_factory=EvalRetrievalConfig)
    end_to_end: EvalEndToEndConfig = Field(default_factory=EvalEndToEndConfig)
    ablations: list[str] = Field(
        default_factory=lambda: [
            "bm25_only",
            "dense_only",
            "hybrid",
            "hybrid_rerank",
            "hybrid_rerank_rewrite",
        ]
    )
    output_dir: str = "reports"
    charts_dir: str = "docs-site/docs/assets/charts"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_corpus_config(path: str | Path) -> CorpusConfig:
    return CorpusConfig(**load_yaml(path))


def load_retrieval_config(path: str | Path) -> RetrievalConfig:
    return RetrievalConfig(**load_yaml(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig(**load_yaml(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig(**load_yaml(path))


def load_serving_config(path: str | Path) -> ServingConfig:
    return ServingConfig(**load_yaml(path))