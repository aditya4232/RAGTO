from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from devrag.config import LLMConfig, load_retrieval_config
from devrag.generation.client import LLMClient
from devrag.generation.pipeline import RAGPipeline
from devrag.retrieval.hybrid import HybridRetriever


def _build_pipeline() -> RAGPipeline:
    cfg = load_retrieval_config("configs/retrieval.yaml")
    retriever = HybridRetriever.load(cfg, index_dir=cfg.indexing.output_dir)
    llm_cfg = LLMConfig(
        backend=os.environ.get("DEVRAG_LLM_BACKEND", cfg.llm.backend),
        model=os.environ.get("DEVRAG_LLM_MODEL", cfg.llm.model),
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        system_prompt_file=cfg.llm.system_prompt_file,
    )
    llm = LLMClient(llm_cfg)
    return RAGPipeline(retriever, llm, system_prompt_file=cfg.llm.system_prompt_file)


_PIPELINE: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _build_pipeline()
    return _PIPELINE


EXAMPLES = [
    "What changed in the RAG landscape in 2026?",
    "How do I LoRA-fine-tune a small embedding model for a domain?",
    "Explain hybrid BM25 + dense retrieval with reciprocal rank fusion.",
    "What is the recommended way to evaluate a RAG system end-to-end?",
    "How does vLLM serve LLMs with PagedAttention?",
    "Show me how to use sentence-transformers with a custom PEFT adapter.",
    "What are the trade-offs of HyDE vs multi-query rewriting?",
    "How do I deploy a RAG app on Hugging Face Spaces?",
]


def respond(message: str, history: list[list[str]], ablation: str) -> tuple[str, list[list[str]], str]:
    pipe = get_pipeline()
    ans = pipe.answer(message, ablation=ablation)
    sources_md = "\n\n".join(
        f"**[{i+1}] {hit.chunk.title or hit.chunk.source}** — `{hit.chunk.chunk_id}`\n"
        f"<small>{hit.chunk.url}</small>\n\n"
        f"<details><summary>snippet</summary>\n\n{hit.chunk.text[:600]}…\n\n</details>"
        for i, hit in enumerate(ans.hits)
    )
    history = history + [[message, ans.text]]
    meta = (
        f"**Latency:** {ans.latency_ms:.0f} ms · **Hits:** {len(ans.hits)} · "
        f"**Citations:** {len(ans.citations)} · **Method:** `{ablation}`"
    )
    return "", history, sources_md + "\n\n---\n" + meta


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DevRAG-2026", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# DevRAG-2026\n"
            "RAG over the latest AI/ML engineering knowledge (arXiv + framework docs + blogs). "
            "Local-first; runs fully on a laptop."
        )
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=420)
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g. What's new in RAG in 2026?",
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                ablation = gr.Radio(
                    choices=[
                        "bm25_only",
                        "dense_only",
                        "hybrid",
                        "hybrid_rerank",
                        "hybrid_rerank_rewrite",
                    ],
                    value="hybrid_rerank_rewrite",
                    label="Retrieval method",
                )
                gr.Examples(EXAMPLES, inputs=msg, label="Try a question")
            with gr.Column(scale=2):
                sources = gr.Markdown(label="Sources")
        msg.submit(respond, [msg, chatbot, ablation], [msg, chatbot, sources])
        send.click(respond, [msg, chatbot, ablation], [msg, chatbot, sources])
        clear.click(lambda: ([], "", ""), outputs=[chatbot, sources, msg])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)