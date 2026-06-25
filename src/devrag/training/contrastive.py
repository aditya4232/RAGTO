from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from devrag.config import TrainingConfig
from devrag.logging_setup import get_logger
from devrag.training.dataset import PairDataset

log = get_logger(__name__)


class ContrastiveTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._set_seed(42)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self) -> tuple[Any, Any]:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        base = AutoModel.from_pretrained(self.config.base_model, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)

        lora_cfg = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
        return model, tok

    def _collate(self, batch: list[dict[str, str]], tokenizer: Any) -> dict[str, torch.Tensor]:
        queries = [b["query"] for b in batch]
        positives = [b["positive"] for b in batch]
        q_enc = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.config.optimization.max_seq_length,
            return_tensors="pt",
        )
        p_enc = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.config.optimization.max_seq_length,
            return_tensors="pt",
        )
        return {"q": q_enc, "p": p_enc}

    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def train(self, train_path: str | Path, eval_path: str | Path | None = None) -> dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok = self._build_model()
        model.to(device)
        if self.config.optimization.fp16 and device == "cuda":
            model = model.half()

        train_ds = PairDataset(train_path, max_pairs=self.config.data.max_pairs)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.optimization.batch_size,
            shuffle=True,
            collate_fn=lambda b: self._collate(b, tok),
            num_workers=0,
        )
        eval_loader = None
        if eval_path and Path(eval_path).exists():
            eval_ds = PairDataset(eval_path)
            eval_loader = DataLoader(
                eval_ds,
                batch_size=self.config.optimization.eval_batch_size,
                shuffle=False,
                collate_fn=lambda b: self._collate(b, tok),
                num_workers=0,
            )

        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.optimization.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = torch.optim.AdamW(params, lr=self.config.optimization.lr)
        total_steps = max(len(train_loader) // self.config.optimization.grad_accum * self.config.optimization.epochs, 1)
        warmup_steps = max(int(total_steps * self.config.optimization.warmup_ratio), 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        temp = self.config.temperature

        history: list[dict[str, float]] = []
        step = 0
        t0 = time.time()
        for epoch in range(self.config.optimization.epochs):
            model.train()
            optim.zero_grad()
            running = 0.0
            n_batches = 0
            for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
                q_ids = batch["q"]["input_ids"].to(device)
                q_mask = batch["q"]["attention_mask"].to(device)
                p_ids = batch["p"]["input_ids"].to(device)
                p_mask = batch["p"]["attention_mask"].to(device)
                if self.config.optimization.fp16 and device == "cuda":
                    q_ids = q_ids.long()
                    p_ids = p_ids.long()
                q_out = model(input_ids=q_ids, attention_mask=q_mask).last_hidden_state
                p_out = model(input_ids=p_ids, attention_mask=p_mask).last_hidden_state
                q_emb = torch.nn.functional.normalize(self._mean_pool(q_out, q_mask), dim=-1)
                p_emb = torch.nn.functional.normalize(self._mean_pool(p_out, p_mask), dim=-1)
                logits = (q_emb @ p_emb.T) / temp
                labels = torch.arange(logits.size(0), device=device)
                loss = (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                ) / 2
                (loss / self.config.optimization.grad_accum).backward()
                running += loss.item()
                n_batches += 1
                if n_batches % self.config.optimization.grad_accum == 0:
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()
                    step += 1
            avg = running / max(n_batches, 1)
            entry = {"epoch": epoch, "train_loss": avg, "lr": optim.param_groups[0]["lr"]}
            if eval_loader is not None:
                entry["eval_loss"] = self._eval(model, tok, eval_loader, device)
            history.append(entry)
            log.info("train.epoch", **entry)
        self._save(model, tok)
        return {"history": history, "duration_s": time.time() - t0}

    @torch.no_grad()
    def _eval(self, model: Any, tok: Any, loader: DataLoader, device: str) -> float:
        model.eval()
        total = 0.0
        n = 0
        temp = self.config.temperature
        for batch in loader:
            q_ids = batch["q"]["input_ids"].to(device)
            q_mask = batch["q"]["attention_mask"].to(device)
            p_ids = batch["p"]["input_ids"].to(device)
            p_mask = batch["p"]["attention_mask"].to(device)
            q_emb = torch.nn.functional.normalize(
                self._mean_pool(model(input_ids=q_ids, attention_mask=q_mask).last_hidden_state, q_mask),
                dim=-1,
            )
            p_emb = torch.nn.functional.normalize(
                self._mean_pool(model(input_ids=p_ids, attention_mask=p_mask).last_hidden_state, p_mask),
                dim=-1,
            )
            logits = (q_emb @ p_emb.T) / temp
            labels = torch.arange(logits.size(0), device=device)
            loss = (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            ) / 2
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _save(self, model: Any, tok: Any) -> None:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        with (out / "training_config.json").open("w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)
        if self.config.push_to_hub.enabled and self.config.push_to_hub.repo_id:
            from huggingface_hub import whoami

            try:
                whoami()
            except Exception:
                pass
            model.push_to_hub(self.config.push_to_hub.repo_id, private=self.config.push_to_hub.private)
            tok.push_to_hub(self.config.push_to_hub.repo_id, private=self.config.push_to_hub.private)
        log.info("train.save", path=str(out))