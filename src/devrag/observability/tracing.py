from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from devrag.logging_setup import get_logger

log = get_logger(__name__)


class Tracer:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []

    def add(self, name: str, **fields: Any) -> None:
        rec = {"ts": time.time(), "name": name, "fields": fields}
        self.records.append(rec)
        log.info(name, **fields)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")


@contextmanager
def trace_request(name: str, fields: dict[str, Any] | None = None) -> Iterator["_TraceCtx"]:
    ctx = _TraceCtx(name, fields or {})
    t0 = time.perf_counter()
    try:
        yield ctx
    finally:
        ctx.add("duration_ms", round((time.perf_counter() - t0) * 1000.0, 2))


class _TraceCtx:
    def __init__(self, name: str, fields: dict[str, Any]) -> None:
        self.name = name
        self.fields = fields

    def add(self, key: str, value: Any) -> None:
        self.fields[key] = value

    def __enter__(self) -> "_TraceCtx":
        return self

    def __exit__(self, *_exc: Any) -> None:
        log.info(self.name, **self.fields)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]