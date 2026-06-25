from __future__ import annotations

import os
from typing import Any

from devrag.config import LLMConfig
from devrag.logging_setup import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.backend = config.backend

        if self.backend == "ollama":
            try:
                import ollama

                self._client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
            except Exception as e:
                log.warning("llm.ollama.unavailable", error=str(e))
                self._client = None
        elif self.backend == "openai":
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception as e:
                log.warning("llm.openai.unavailable", error=str(e))
                self._client = None
        elif self.backend == "llama_cpp":
            self._client = None
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        mx = max_tokens if max_tokens is not None else self.config.max_tokens
        if self.backend == "ollama" and self._client is not None:
            r = self._client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": temp, "num_predict": mx},
            )
            return r["message"]["content"]
        if self.backend == "openai" and self._client is not None:
            r = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=mx,
            )
            return (r.choices[0].message.content or "").strip()
        return _stub(prompt)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        mx = max_tokens if max_tokens is not None else self.config.max_tokens
        if self.backend == "ollama" and self._client is not None:
            r = self._client.chat(model=self.config.model, messages=messages, options={"temperature": temp, "num_predict": mx})
            return r["message"]["content"]
        if self.backend == "openai" and self._client is not None:
            r = self._client.chat.completions.create(
                model=self.config.model, messages=messages, temperature=temp, max_tokens=mx
            )
            return (r.choices[0].message.content or "").strip()
        return _stub(messages[-1].get("content", ""))


def _stub(prompt: str) -> str:
    head = prompt.strip().splitlines()[0][:80]
    return f"[stub LLM reply to: {head!r}]\n\nThis is a placeholder. Install Ollama and pull a model, or set OPENAI_API_KEY, to enable real generation."