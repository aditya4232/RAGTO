---
title: DevRAG-2026
emoji: "🔎"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: RAG over the latest AI/ML engineering knowledge with a fine-tuned embedding.
---

# DevRAG-2026 — live demo

This Space runs the **DevRAG-2026** retrieval-augmented generation pipeline.

- Local-first, fully runs on the free CPU tier (uses a quantized GGUF model + small embedding).
- For full-speed and the fine-tuned embedding adapter, please run locally:
  `git clone https://github.com/<user>/DevRAG-2026 && make install && make ui`.

## What it answers

Anything about AI, ML, and software engineering drawn from arXiv (last 18 months, cs.CL / cs.LG / cs.AI),
official framework docs (Hugging Face, PyTorch, LangChain, vLLM, LlamaIndex), and a curated set of
AI engineering blogs.

## Architecture (one-liner)

Hybrid retrieval (BM25 + dense + cross-encoder reranker) → small LLM (Qwen3) → cited answer.

See the docs site for the full architecture, evaluation results, and training details:
<https://<user>.github.io/DevRAG-2026/>