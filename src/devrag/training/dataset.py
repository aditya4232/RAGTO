from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import json
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, path: str | Path, max_pairs: int | None = None) -> None:
        self.records: list[dict[str, str]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "query" in d and "positive" in d:
                    self.records.append({"query": d["query"], "positive": d["positive"]})
                if max_pairs is not None and len(self.records) >= max_pairs:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.records[idx]