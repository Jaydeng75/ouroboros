"""CLI: models, run, shards, claim, release, push, pull."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .evolve import evolve as evolve_generation
from .harness import run_tasks
from .ollama_client import OllamaClient, OllamaError
from .sandbox import Sandbox
from .sft import build_dataset
from .traces import load_traces
from .shards import (
    ShardError,
    claim,
    find_shard,
    load_manifest,
    release,
    save_manifest,
    tasks_for_shard,
    validate_manifest,
)


class _DryModel:
    """Plumbing check without Ollama. Exercises the repair path."""

    def generate(self, prompt, model, *, temperature=0.7, timeout=120.0):
        return "```python\ndef solve(*args, **kwargs):\n    return None\n```"


def _load_tasks(path: str) -> list[dict]:
    tasks: list[dict] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8")
                                  .splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        for key in ("id", "prompt", "tests"):
            if key not in task:
                raise SystemExit(f"{path}:{lineno}: missing key {key!r}")
        tasks.append(task)
    return tasks


def _git(*git_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *git_args], capture_output=True, text=True)


def _require_repo() -> bool:
    proc = _git("rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        print(
            "Not inside a git repository. One-time setup:\n"
            "  git init && git add -A && git commit -m 'ouroboros'\n"
            "  git remote add origin <your-repo-url> && git push -u origin HEAD\n"
            "Friends join with: git clone <your-repo-url>",
            file=sys.stderr,
        )
        return False
    return True


def _cmd_push(message: str) -> int:
    if not _require_repo():
        return 1
    _git("add", "--", "shards.json", "traces/")
    staged = _git("diff", "--cached", "--quiet")
    if staged.returncode != 0:
        commit = _git("commit", "-m", message)
        if commit.returncode != 0:
            print(commit.stdout + commit.stderr, file=sys.stderr)
            return 1
        print(f"Committed: {message}")
    push = _git("push")
    if push.returncode != 0:
        print(push.stdout + push.stderr, file=sys.stderr)
        return 1
    print("Pushed.")
    return 0


def _cmd_pull() -> int:
    if not _require_repo():
        return 1
    pull = _git("pull", "--rebase")
    if pull.returncode != 0:
        print(pull.stdout + pull.stderr, file=sys.stderr)
        return 1
    print("Pulled. Run `ouroboros shards` to see current claims.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ouroboros")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="List models available in local Ollama")

    run = sub.add_parser("run", help="Run the harness over tasks")
    run.add_argument("--model", default="qwen2.5-coder:7b")
    run.add_argument("--tasks", default="tasks/seed_tasks.jsonl")
    run.add_argument("--out", default=None,
                     help="Trace log (default: traces/traces.jsonl, "
                          "or the shard's trace_file with --shard)")
    run.add_argument("--shard", default=None,
                     help="Only run this shard's tasks from shards.json")
    run.add_argument("--manifest", default="shards.json")
    run.add_argument("--max-attempts", type=int, default=4)
    run.add_argument("--timeout", type=float, default=2.0,
                     help="Per-test pytest timeout in seconds")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--gen-timeout", type=float, default=300.0,
                     help="Per-generation Ollama timeout in seconds "
                          "(raise for CPU inference)")
    run.add_argument("--dry-run", action="store_true",
                     help="No Ollama: stub model, checks plumbing only")

    shards = sub.add_parser("shards", help="List shards and claims")
    shards.add_argument("--manifest", default="shards.json")
    shards.add_argument("--tasks", default="tasks/seed_tasks.jsonl")

    claim_cmd = sub.add_parser("claim", help="Claim a shard for this device")
    claim_cmd.add_argument("shard")
    claim_cmd.add_argument("--as", dest="device", default=None,
                           help="Device name (default: hostname)")
    claim_cmd.add_argument("--manifest", default="shards.json")
    claim_cmd.add_argument("--tasks", default="tasks/seed_tasks.jsonl")

    release_cmd = sub.add_parser("release", help="Release a shard")
    release_cmd.add_argument("shard")
    release_cmd.add_argument("--manifest", default="shards.json")

    push = sub.add_parser("push", help="Commit manifest+traces and push")
    push.add_argument("-m", "--message", default="ouroboros: sync traces")

    evo = sub.add_parser("evolve", help="One evolution generation: "
                                        "mutate tasks, keep learnable ones")
    evo.add_argument("--model", default="qwen2.5-coder:7b")
    evo.add_argument("--tasks", default="tasks/seed_tasks.jsonl")
    evo.add_argument("--out", default="tasks/evolved.jsonl")
    evo.add_argument("--archive", default="archive.json")
    evo.add_argument("--samples", type=int, default=4)
    evo.add_argument("--min-rate", type=float, default=0.2)
    evo.add_argument("--max-rate", type=float, default=0.8)
    evo.add_argument("--no-compose", action="store_true",
                     help="Skip the deterministic compose operator")
    evo.add_argument("--parent-limit", type=int, default=12,
                     help="Max seed tasks to mutate with the model")
    evo.add_argument("--compose-limit", type=int, default=24,
                     help="Max composed task pairs to evaluate")
    evo.add_argument("--random-seed", type=int, default=42,
                     help="Seed for bounded candidate selection")
    evo.add_argument("--gen-timeout", type=float, default=300.0,
                     help="Per-generation Ollama timeout in seconds")
    evo.add_argument("--dry-run", action="store_true",
                     help="No Ollama: stub model, checks plumbing only")

    sft = sub.add_parser("export-sft", help="Export traces as SFT chat JSONL")
    sft.add_argument("--traces", nargs="+", default=["traces/traces.jsonl"],
                     help="Trace JSONL file(s) to convert")
    sft.add_argument("--out", default="datasets/sft.jsonl")

    sub.add_parser("pull", help="Pull latest manifest and traces")

    args = parser.parse_args(argv)
    client = OllamaClient()

    if args.command == "models":
        try:
            models = client.list_models()
        except OllamaError as exc:
            print(exc, file=sys.stderr)
            return 1
        if not models:
            print("No models found. Pull one: "
                  "ollama pull qwen2.5-coder:7b")
        for name in models:
            print(name)
        return 0

    if args.command == "shards":
        try:
            manifest = load_manifest(args.manifest)
            validate_manifest(manifest, _load_tasks(args.tasks))
        except ShardError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Base model: {manifest.get('base_model')} "
              f"(format v{manifest.get('format_version')})")
        for shard in manifest.get("shards", []):
            owner = shard.get("claimed_by") or "-"
            print(f"{shard['id']:<12} {shard.get('status', 'open'):<8} "
                  f"owner={owner:<20} tasks={shard.get('task_ids', [])}")
        return 0

    if args.command == "claim":
        try:
            manifest = load_manifest(args.manifest)
            validate_manifest(manifest, _load_tasks(args.tasks))
            shard = claim(manifest, args.shard, args.device)
            save_manifest(args.manifest, manifest)
        except ShardError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Claimed {shard['id']!r} for {shard['claimed_by']!r}. "
              f"Run: ouroboros run --shard {shard['id']} && ouroboros push")
        return 0

    if args.command == "release":
        try:
            manifest = load_manifest(args.manifest)
            shard = release(manifest, args.shard)
            save_manifest(args.manifest, manifest)
        except ShardError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Released {shard['id']!r}.")
        return 0

    if args.command == "evolve":
        tasks = _load_tasks(args.tasks)
        archive_path = Path(args.archive)
        if archive_path.exists():
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
            if archive.get("format_version") != 1:
                print("Unsupported archive format_version; expected 1.",
                      file=sys.stderr)
                return 1
        else:
            archive = {"format_version": 1, "cells": {}}
        model_client = _DryModel() if args.dry_run else client
        kept, archive = evolve_generation(
            model_client,
            args.model,
            tasks,
            Sandbox(timeout=5.0),
            archive,
            samples=args.samples,
            min_rate=args.min_rate,
            max_rate=args.max_rate,
            compose=not args.no_compose,
            parent_limit=args.parent_limit,
            compose_limit=args.compose_limit,
            random_seed=args.random_seed,
            gen_timeout=args.gen_timeout,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for task in kept:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        archive_path.write_text(
            json.dumps(archive, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Kept {len(kept)} mutant(s) in {args.out}; "
              f"archive has {len(archive.get('cells', {}))} cell(s).")
        if args.dry_run and not kept:
            print("Dry run kept nothing (stub solver) - expected. "
                  "Run without --dry-run against Ollama for real mutants.")
        return 0

    if args.command == "export-sft":
        traces: list[dict] = []
        for trace_path in args.traces:
            traces.extend(load_traces(trace_path))
        examples, stats = build_dataset(traces)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} example(s) to {args.out}: "
              f"{stats['direct']} direct, {stats['repair']} repair "
              f"(skipped {stats['skipped_held_out']} held-out, "
              f"{stats['skipped_unpassed']} never-passed).")
        if not examples:
            print("Empty dataset. Run the harness first: "
                  "ouroboros run (and ouroboros run --tasks "
                  "tasks/evolved.jsonl).", file=sys.stderr)
            return 1
        return 0

    if args.command == "push":
        return _cmd_push(args.message)

    if args.command == "pull":
        return _cmd_pull()

    # command == "run"
    try:
        tasks = _load_tasks(args.tasks)
        out = args.out
        if args.shard:
            manifest = load_manifest(args.manifest)
            validate_manifest(manifest, tasks)
            tasks = tasks_for_shard(manifest, args.shard, tasks)
            if out is None:
                shard = find_shard(manifest, args.shard)
                out = shard.get("trace_file",
                                f"traces/shard-{args.shard}.jsonl")
        if out is None:
            out = "traces/traces.jsonl"
    except ShardError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if any(t.get("held_out") for t in tasks):
        print("Warning: held-out tasks are evaluation-only. "
              "Never train on their traces.", file=sys.stderr)
    model_client = _DryModel() if args.dry_run else client
    summary = run_tasks(
        model_client,
        args.model,
        tasks,
        Sandbox(timeout=args.timeout),
        out,
        max_attempts=args.max_attempts,
        gen_timeout=args.gen_timeout,
    )
    print(f"Done: {summary['passed']}/{summary['tasks']} passed. "
          f"Traces in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
