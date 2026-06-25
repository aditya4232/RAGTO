from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from devrag.config import EvalConfig, RetrievalConfig
from devrag.eval.metrics import evaluate_retrieval, save_report
from devrag.ingest.utils import read_jsonl
from devrag.logging_setup import get_logger
from devrag.retrieval.hybrid import HybridRetriever
from devrag.types import EvalExample

log = get_logger(__name__)


def load_examples(path: str | Path) -> list[EvalExample]:
    return [EvalExample.from_dict(d) for d in read_jsonl(path)]


def run_ablation(
    retriever: HybridRetriever,
    examples: list[EvalExample],
    ablation: str,
    *,
    ks: Iterable[int] = (1, 5, 10),
    top_k: int = 50,
) -> dict[str, Any]:
    log.info("eval.ablation", ablation=ablation, n=len(examples))
    return evaluate_retrieval(
        examples,
        lambda q: retriever.retrieve(q, ablation=ablation, top_k=top_k),
        ks=ks,
        top_k=top_k,
    )


def run_all(
    eval_config: EvalConfig,
    retrieval_config: RetrievalConfig,
    *,
    index_dir: str | Path,
) -> dict[str, Any]:
    examples = load_examples(eval_config.retrieval.dataset)
    if not examples:
        log.warning("eval.no_examples", path=eval_config.retrieval.dataset)
        return {}

    retriever = HybridRetriever.load(retrieval_config, index_dir=index_dir)
    results: dict[str, Any] = {}
    for ablation in eval_config.ablations:
        results[ablation] = run_ablation(
            retriever, examples, ablation, ks=eval_config.retrieval.ks
        )
    out_path = Path(eval_config.output_dir) / "retrieval_eval.json"
    save_report({"results": results, "config": eval_config.model_dump()}, out_path)
    log.info("eval.retrieval.saved", path=str(out_path))
    return results


def run_embedding_ab(
    eval_config: EvalConfig,
    retrieval_config: RetrievalConfig,
    *,
    base_index_dir: str | Path,
    ft_index_dir: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    examples = load_examples(eval_config.retrieval.dataset)
    if not examples:
        return {}

    base = HybridRetriever.load(retrieval_config, index_dir=base_index_dir)
    ft_cfg = retrieval_config.model_copy(deep=True)
    ft_cfg.dense.lora_adapter = str(ft_index_dir)
    ft = HybridRetriever.load(ft_cfg, index_dir=ft_index_dir)

    out: dict[str, Any] = {
        "base": run_ablation(base, examples, "hybrid_rerank"),
        "fine_tuned": run_ablation(ft, examples, "hybrid_rerank"),
    }
    save_report(out, out_path)
    return out