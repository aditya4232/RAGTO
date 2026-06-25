from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, path: str | Path, *, ttl_seconds: int = 3600) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    ts REAL
                )
                """
            )
            c.commit()

    @staticmethod
    def make_key(query: str, ablation: str) -> str:
        return hashlib.sha256(f"{ablation}::{query}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as c:
            r = c.execute("SELECT value, ts FROM cache WHERE key=?", (key,)).fetchone()
        if not r:
            return None
        value, ts = r
        if time.time() - ts > self.ttl:
            return None
        return json.loads(value)

    def set(self, key: str, value: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute(
                "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), time.time()),
            )
            c.commit()

    def clear(self) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute("DELETE FROM cache")
            c.commit()