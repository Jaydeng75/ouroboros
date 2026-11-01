import pytest

from ouroboros.shards import (
    ShardError,
    claim,
    find_shard,
    release,
    tasks_for_shard,
    validate_manifest,
)

TASKS = [
    {"id": "a", "prompt": "p", "tests": "t"},
    {"id": "b", "prompt": "p", "tests": "t"},
    {"id": "secret", "prompt": "p", "tests": "t", "held_out": True},
]

MANIFEST = {
    "format_version": 1,
    "base_model": "Qwen2.5-Coder-7B",
    "shards": [
        {"id": "s1", "task_ids": ["a"], "status": "open",
         "claimed_by": None, "trace_file": "traces/shard-s1.jsonl"},
        {"id": "s2", "task_ids": ["b"], "status": "open",
         "claimed_by": None, "trace_file": "traces/shard-s2.jsonl"},
    ],
}


def test_claim_and_release():
    shard = claim(MANIFEST, "s1", "laptop-a")
    assert shard["claimed_by"] == "laptop-a"
    assert shard["status"] == "claimed"
    with pytest.raises(ShardError, match="already claimed"):
        claim(MANIFEST, "s1", "laptop-b")
    release(MANIFEST, "s1")
    assert find_shard(MANIFEST, "s1")["claimed_by"] is None


def test_validate_rejects_held_out_and_duplicates():
    validate_manifest(MANIFEST, TASKS)
    bad = {"format_version": 1, "shards": [
        {"id": "x", "task_ids": ["secret"]}]}
    with pytest.raises(ShardError, match="held-out"):
        validate_manifest(bad, TASKS)
    dup = {"format_version": 1, "shards": [
        {"id": "x", "task_ids": ["a"]}, {"id": "y", "task_ids": ["a"]}]}
    with pytest.raises(ShardError, match="two shards"):
        validate_manifest(dup, TASKS)


def test_tasks_for_shard_filters():
    selected = tasks_for_shard(MANIFEST, "s2", TASKS)
    assert [t["id"] for t in selected] == ["b"]
