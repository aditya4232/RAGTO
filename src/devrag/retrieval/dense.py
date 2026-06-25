from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from devrag.types import Chunk


class DenseEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        lora_adapter: str | None = None,
        device: str | None = None,
        max_seq_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.lora_adapter = lora_adapter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        if lora_adapter:
            from peft import PeftModel

            self.model[0].auto_model = PeftModel.from_pretrained(
                self.model[0].auto_model, lora_adapter
            )

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_chunks(self, chunks: list[Chunk], **kwargs: Any) -> np.ndarray:
        return self.encode([c.text for c in chunks], **kwargs)