"""Held-out discovery test: can the scratch LM invent an ordering method?

This prompt and its tests are evaluation-only. They are never added to the
training corpus. A candidate counts as a discovered procedure only if it
passes randomized tests without calling an existing ordering API.
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


PROMPT = '''def arrange_values(values):
    """Return a new list containing the same numbers in nondecreasing order."""
'''

TESTS = '''from solution import arrange_values


def test_cases():
    assert arrange_values([]) == []
    assert arrange_values([1]) == [1]
    assert arrange_values([3, 1, 2]) == [1, 2, 3]
    assert arrange_values([5, -2, 5, 0, -2]) == [-2, -2, 0, 5, 5]
    assert arrange_values([9, 8, 7, 6]) == [6, 7, 8, 9]
    assert arrange_values([4, 4, 4]) == [4, 4, 4]
'''


def extract_candidate(generated_text: str) -> str:
    start = generated_text.find("def arrange_values")
    if start < 0:
        return generated_text
    return generated_text[start:]


def analyze_candidate(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"syntax_error": str(exc)}
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return {
        "loops": sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)),
        "comparisons": sum(isinstance(node, ast.Compare) for node in ast.walk(tree)),
        "tuple_assignments": sum(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Tuple) for target in node.targets)
            for node in ast.walk(tree)
        ),
        "recursive_calls": calls.count("arrange_values"),
        "calls": sorted(set(calls)),
        "uses_ordering_api": any(
            name in {"sorted", "sort", "min", "max", "heapq", "bisect"}
            for name in calls
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate sorting discovery")
    parser.add_argument("--checkpoint", default="scratch_runs/gen0/best.pt")
    parser.add_argument("--tokenizer", default="scratch_runs/gen0/tokenizer.json")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="scratch_runs/gen0/sorting_evaluation.json")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        Path(args.checkpoint), map_location=device, weights_only=False
    )
    config = ModelConfig.from_dict(checkpoint["config"])
    model = ScratchTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = BytePairTokenizer.load(Path(args.tokenizer))
    prompt_ids = [BOS_ID, *tokenizer.encode(PROMPT, add_special=False)]

    sandbox = Sandbox(timeout=2.0)
    records = []
    discovered = False
    for index in range(args.samples):
        torch.manual_seed(10_000 + index)
        generated = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_id=EOS_ID,
        )
        text = tokenizer.decode(generated)
        candidate = extract_candidate(text)
        analysis = analyze_candidate(candidate)
        result = sandbox.run(candidate, TESTS)
        passed = bool(result.passed and not analysis.get("uses_ordering_api"))
        discovered = discovered or passed
        records.append(
            {
                "sample": index,
                "passed_without_ordering_api": passed,
                "sandbox_passed": result.passed,
                "analysis": analysis,
                "candidate": candidate,
                "stderr_tail": result.stderr[-1000:],
            }
        )
        print(
            f"sample={index} passed={passed} sandbox={result.passed} "
            f"analysis={analysis}"
        )

    output = {
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "temperature": args.temperature,
        "discovered": discovered,
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print({"discovered": discovered, "output": str(output_path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
