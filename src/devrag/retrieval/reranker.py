from __future__ import annotations

from typing import Any

import numpy as np
import torch


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def rerank(
        self, query: str, documents: list[tuple[str, str]], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        if not documents:
            return []
        ids, texts = zip(*documents, strict=True)
        pairs = [[query, t] for t in texts]
        with torch.no_grad():
            enc = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy() if logits.shape[-1] == 1 else logits.cpu().numpy()
        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [(ids[i], float(scores[i])) for i in order]