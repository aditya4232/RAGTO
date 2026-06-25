from __future__ import annotations

from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from devrag.config import (
    LLMConfig,
    load_corpus_config,
    load_eval_config,
    load_retrieval_config,
    load_training_config,
)
from devrag.logging_setup import get_logger, setup_logging

app = typer.Typer(add_completion=False, help="DevRAG-2026 CLI")
ingest_app = typer.Typer(help="Ingestion commands")
index_app = typer.Typer(help="Indexing commands")
train_app = typer.Typer(help="Training commands")
eval_app = typer.Typer(help="Evaluation commands")
serve_app = typer.Typer(help="Serving commands")

app.add_typer(ingest_app, name="ingest")
app.add_typer(index_app, name="index")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(serve_app, name="serve")

log = get_logger(__name__)
console = Console()


@ingest_app.command("run")
def ingest_run(
    config: str = typer.Option("configs/corpus.yaml", "--config", "-c"),
    incremental: bool = typer.Option(False, "--incremental"),
) -> None:
    setup_logging("INFO")
    from devrag.ingest.pipeline import build_corpus

    cfg = load_corpus_config(config)
    summary = build_corpus(cfg)
    console.print(summary)


@index_app.command("build")
def index_build(
    config: str = typer.Option("configs/retrieval.yaml", "--config", "-c"),
) -> None:
    setup_logging("INFO")
    from devrag.config import IndexingConfig
    from devrag.ingest.pipeline import load_chunks
    from devrag.retrieval.hybrid import HybridRetriever

    rcfg = load_retrieval_config(config)
    chunks = load_chunks("data/processed/chunks.jsonl")
    if not chunks:
        raise typer.Exit(code=1)
    HybridRetriever(rcfg).build(chunks, IndexingConfig().output_dir)
    console.print("[green]index built[/green]")


@train_app.command("embedding")
def train_embedding(
    config: str = typer.Option("configs/training.yaml", "--config", "-c"),
) -> None:
    setup_logging("INFO")
    from devrag.training.contrastive import ContrastiveTrainer

    tcfg = load_training_config(config)
    trainer = ContrastiveTrainer(tcfg)
    res = trainer.train(tcfg.data.train_file, tcfg.data.eval_file)
    console.print(res)


@train_app.command("mine-pairs")
def mine_pairs(
    documents: str = typer.Option("data/processed/documents.jsonl"),
    out: str = typer.Option("data/processed/pairs_train.jsonl"),
    max_pairs: int = typer.Option(30_000),
) -> None:
    setup_logging("INFO")
    from devrag.training.mining import mine_pairs_from_arxiv

    n = mine_pairs_from_arxiv(documents, out, max_pairs=max_pairs)
    console.print({"pairs": n})


@eval_app.command("run")
def eval_run(
    config: str = typer.Option("configs/eval.yaml", "--config", "-c"),
) -> None:
    setup_logging("INFO")
    from devrag.eval.harness import run_all

    ecfg = load_eval_config(config)
    rcfg = load_retrieval_config("configs/retrieval.yaml")
    results = run_all(ecfg, rcfg, index_dir=rcfg.indexing.output_dir)
    table = Table(title="Retrieval ablation results")
    table.add_column("method")
    for m in ecfg.retrieval.metrics:
        table.add_column(m)
    for ablation, metrics in results.items():
        table.add_row(ablation, *[f"{metrics.get(m, 0.0):.3f}" for m in ecfg.retrieval.metrics])
    console.print(table)


@eval_app.command("embedding-ab")
def eval_embedding_ab(
    config: str = typer.Option("configs/eval.yaml", "--config", "-c"),
    base_dir: str = typer.Option("data/indexes"),
    ft_dir: str = typer.Option("data/indexes_ft"),
    out: str = typer.Option("reports/embedding_ab.json"),
) -> None:
    setup_logging("INFO")
    from devrag.eval.harness import run_embedding_ab

    ecfg = load_eval_config(config)
    rcfg = load_retrieval_config("configs/retrieval.yaml")
    res = run_embedding_ab(ecfg, rcfg, base_index_dir=base_dir, ft_index_dir=ft_dir, out_path=out)
    console.print(res)


@serve_app.command("api")
def serve_api(
    config: str = typer.Option("configs/retrieval.yaml", "--config", "-c"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    setup_logging("INFO")
    import uvicorn

    from devrag.config import ServingConfig
    from devrag.serving.api import init_app

    init_app(retrieval_config_path=config, serving=ServingConfig(host=host, port=port))
    uvicorn.run("devrag.serving.api:app", host=host, port=port, reload=False)


@app.command("version")
def version() -> None:
    from devrag import __version__

    console.print(f"devrag {__version__}")


@app.command("show-config")
def show_config(config: str = typer.Argument(...)) -> None:
    with open(config, "r", encoding="utf-8") as f:
        console.print(yaml.safe_load(f))


def main() -> Any:
    app()


if __name__ == "__main__":
    main()