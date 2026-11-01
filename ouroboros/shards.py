"""Shard manifest: split the seed pool so devices train in parallel.

One JSON file (`shards.json`) is the coordination point. Every device uses
the same frozen base model and trace format. A device claims one shard,
runs the harness on only that shard's tasks, then pushes traces back.
Held-out tasks are evaluation-only and can never appear in a shard.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

FORMAT_VERSION = 1


class ShardError(RuntimeError):
    pass


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ShardError(
            f"No shard manifest at {p}. Create shards.json first (see README)."
        )
    manifest = json.loads(p.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ShardError(
            f"Unsupported format_version {manifest.get('format_version')!r}; "
            f"expected {FORMAT_VERSION}. Sync the repo before claiming work."
        )
    return manifest


def save_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_manifest(manifest: dict, tasks: list[dict]) -> None:
    """Fail loudly on version skew, duplicates, or held-out leakage."""
    by_id = {t["id"]: t for t in tasks}
    seen: set[str] = set()
    for shard in manifest.get("shards", []):
        for task_id in shard.get("task_ids", []):
            if task_id in seen:
                raise ShardError(f"Task {task_id!r} appears in two shards.")
            seen.add(task_id)
            task = by_id.get(task_id)
            if task is None:
                raise ShardError(
                    f"Shard {shard['id']!r} references unknown task {task_id!r}."
                )
            if task.get("held_out"):
                raise ShardError(
                    f"Shard {shard['id']!r} contains held-out task "
                    f"{task_id!r}. Held-out tasks are evaluation-only."
                )


def find_shard(manifest: dict, shard_id: str) -> dict:
    for shard in manifest.get("shards", []):
        if shard["id"] == shard_id:
            return shard
    raise ShardError(f"Unknown shard {shard_id!r}.")


def claim(manifest: dict, shard_id: str, device: str | None = None) -> dict:
    device = device or socket.gethostname()
    shard = find_shard(manifest, shard_id)
    owner = shard.get("claimed_by")
    if owner and owner != device:
        raise ShardError(
            f"Shard {shard_id!r} is already claimed by {owner!r}. "
            "Pick another shard or ask them to release it."
        )
    shard["claimed_by"] = device
    shard["status"] = "claimed"
    return shard


def release(manifest: dict, shard_id: str) -> dict:
    shard = find_shard(manifest, shard_id)
    shard["claimed_by"] = None
    shard["status"] = "open"
    return shard


def tasks_for_shard(manifest: dict, shard_id: str,
                    tasks: list[dict]) -> list[dict]:
    shard = find_shard(manifest, shard_id)
    wanted = set(shard.get("task_ids", []))
    selected = [t for t in tasks if t["id"] in wanted]
    if len(selected) != len(wanted):
        missing = wanted - {t["id"] for t in selected}
        raise ShardError(f"Tasks missing from pool: {sorted(missing)}")
    return selected
