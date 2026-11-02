"""Intermediate non-ordering benchmark for scratch LM checkpoints.

These tasks measure whether a checkpoint can generate syntactically valid,
working Python before the sorting-discovery test is meaningful. They are
evaluation-only and must not be added to training data.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import torch

from ouroboros.sandbox import Sandbox

from .model import ModelConfig, ScratchTransformer
from .tokenizer import BOS_ID, EOS_ID, BytePairTokenizer


def _task(task_id: str, function: str, signature: str, doc: str, tests: str) -> dict:
    return {
        "id": task_id,
        "function": function,
        "prompt": f"def {signature}:\n    \"\"\"{doc}\"\"\"\n",
        "tests": tests,
    }


TASKS = [
    _task(
        "copy_values",
        "copy_values",
        "copy_values(values)",
        "Return a new list containing the same values in the same sequence.",
        "from solution import copy_values\n\n\ndef test_copy():\n    assert copy_values([1, 2, 3]) == [1, 2, 3]\n    assert copy_values([]) == []\n",
    ),
    _task(
        "double_values",
        "double_values",
        "double_values(values)",
        "Return a new list where every value is doubled.",
        "from solution import double_values\n\n\ndef test_double():\n    assert double_values([1, -2, 3]) == [2, -4, 6]\n    assert double_values([]) == []\n",
    ),
    _task(
        "keep_above",
        "keep_above",
        "keep_above(values, cutoff)",
        "Return values strictly greater than cutoff, preserving sequence.",
        "from solution import keep_above\n\n\ndef test_keep():\n    assert keep_above([1, 5, 2, 8], 3) == [5, 8]\n    assert keep_above([1, 2], 5) == []\n",
    ),
    _task(
        "reverse_values",
        "reverse_values",
        "reverse_values(values)",
        "Return a new list with the values in reverse sequence.",
        "from solution import reverse_values\n\n\ndef test_reverse():\n    assert reverse_values([1, 2, 3]) == [3, 2, 1]\n    assert reverse_values([]) == []\n",
    ),
    _task(
        "exchange_positions",
        "exchange_positions",
        "exchange_positions(values, first, second)",
        "Return a copy with the positions first and second exchanged.",
        "from solution import exchange_positions\n\n\ndef test_exchange():\n    assert exchange_positions([1, 2, 3], 0, 2) == [3, 2, 1]\n",
    ),
    _task(
        "factorial_of",
        "factorial_of",
        "factorial_of(number)",
        "Return the factorial of a non-negative integer.",
        "from solution import factorial_of\n\n\ndef test_factorial():\n    assert factorial_of(0) == 1\n    assert factorial_of(5) == 120\n",
    ),
    _task(
        "count_items",
        "count_items",
        "count_items(items)",
        "Return a dictionary counting how often each item appears.",
        "from solution import count_items\n\n\ndef test_count():\n    assert count_items(['a', 'b', 'a']) == {'a': 2, 'b': 1}\n    assert count_items([]) == {}\n",
    ),
    _task(
        "grid_total",
        "grid_total",
        "grid_total(rows, columns)",
        "Return the sum of row index times column index over a zero-based grid.",
        "from solution import grid_total\n\n\ndef test_grid():\n    assert grid_total(2, 3) == 3\n    assert grid_total(1, 4) == 0\n    assert grid_total(3, 3) == 9\n",
    ),
]


def extract_candidate(generated_text: str, function: str) -> str:
    start = generated_text.find(f"def {function}")
    if start < 0:
        return generated_text
    return generated_text[start:]


def analyze_candidate(code: str) -> dict:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"syntax_valid": False, "syntax_error": str(exc)}
    return {"syntax_valid": True}


def summarize(records: list[dict]) -> dict:
    candidates = len(records)
    syntax_valid = sum(bool(record.get("syntax_valid")) for record in records)
    passed = sum(bool(record.get("sandbox_passed")) for record in records)
    return {
        "candidates": candidates,
        "syntax_valid": syntax_valid,
        "sandbox_passed": passed,
        "syntax_valid_rate": syntax_valid / candidates if candidates else 0.0,
        "pass_rate": passed / candidates if candidates else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run intermediate code benchmark")
    parser.add_argument("--checkpoint", default="scratch_runs/gen0/best.pt")
    parser.add_argument("--tokenizer", default="scratch_runs/gen0/tokenizer.json")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="scratch_runs/gen0/benchmark.json")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        Path(args.checkpoint), map_location=device, weights_only=False
    )
    model = ScratchTransformer(
        ModelConfig.from_dict(checkpoint["config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = BytePairTokenizer.load(Path(args.tokenizer))
    sandbox = Sandbox(timeout=2.0)

    records: list[dict] = []
    for task in TASKS:
        prompt_ids = [
            BOS_ID,
            *tokenizer.encode(task["prompt"], add_special=False),
        ]
        for index in range(args.samples_per_task):
            torch.manual_seed(20_000 + index)
            generated = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_id=EOS_ID,
            )
            candidate = extract_candidate(
                tokenizer.decode(generated), task["function"]
            )
            analysis = analyze_candidate(candidate)
            result = sandbox.run(candidate, task["tests"])
            record = {
                "task_id": task["id"],
                "sample": index,
                "syntax_valid": analysis["syntax_valid"],
                "sandbox_passed": result.passed,
                "analysis": analysis,
                "candidate": candidate,
                "stderr_tail": result.stderr[-1000:],
            }
            records.append(record)
            print(
                f"{task['id']} sample={index} syntax={record['syntax_valid']} "
                f"passed={result.passed}"
            )

    summary = summarize(records)
    output = {
        "checkpoint": args.checkpoint,
        "summary": summary,
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print({**summary, "output": str(output_path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
