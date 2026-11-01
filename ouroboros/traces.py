"""Append-only JSONL trace log: the fuel for every later training step."""
from __future__ import annotations

import json
from pathlib import Path


def append_trace(path: str | Path, trace: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def load_traces(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    traces: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))
    return traces
