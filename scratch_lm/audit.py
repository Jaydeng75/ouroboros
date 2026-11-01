"""Fail-closed audit for sorting leakage in base-model training text.

The audit cannot prove that a corpus contains no ordering concept. It does
reject explicit sorting vocabulary, known ordering APIs, sorting-shaped
identifiers, imports associated with ordering, malformed Python, and direct
comparisons between neighboring elements of the same list.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_TERMS = {
    "sort",
    "sorted",
    "sorting",
    "order",
    "ordered",
    "ordering",
    "ascending",
    "descending",
    "lexicographic",
    "rank",
    "ranked",
    "ranking",
    "smallest",
    "largest",
    "minimum",
    "maximum",
    "median",
    "percentile",
    "heap",
    "heapq",
    "bisect",
}
FORBIDDEN_CALLS = {"sorted", "min", "max", "heapq", "bisect"}
FORBIDDEN_MODULES = {"heapq", "bisect", "statistics"}
_IDENTIFIER_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str
    line: int | None = None


def _identifier_parts(identifier: str) -> set[str]:
    parts: set[str] = set()
    for chunk in identifier.replace("-", "_").split("_"):
        parts.update(part.lower() for part in _IDENTIFIER_PARTS.findall(chunk))
    return parts


def _subscript_base(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def audit_text(text: str) -> list[Violation]:
    violations: list[Violation] = []

    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
        parts = _identifier_parts(match.group(0))
        leaked = parts & FORBIDDEN_TERMS
        if leaked:
            line = text.count("\n", 0, match.start()) + 1
            violations.append(
                Violation(
                    "forbidden_term",
                    f"{match.group(0)!r} contains {sorted(leaked)}",
                    line,
                )
            )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return violations + [
            Violation("syntax_error", str(exc), exc.lineno)
        ]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            else:
                module = node.module or ""
            roots = {part.split(".")[0] for part in module.split(",")}
            if roots & FORBIDDEN_MODULES:
                violations.append(
                    Violation("forbidden_import", module, node.lineno)
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            leaked = _identifier_parts(node.name) & FORBIDDEN_TERMS
            if leaked:
                violations.append(
                    Violation("forbidden_identifier", node.name, node.lineno)
                )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                violations.append(
                    Violation("forbidden_call", func.id, node.lineno)
                )
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALLS:
                violations.append(
                    Violation("forbidden_call", func.attr, node.lineno)
                )
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            bases = [_subscript_base(operand) for operand in operands]
            has_inequality = any(
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                for op in node.ops
            )
            if (
                has_inequality
                and len(operands) >= 2
                and bases[0]
                and len(set(bases)) == 1
            ):
                violations.append(
                    Violation(
                        "adjacent_or_self_comparison",
                        f"comparison between elements of {bases[0]!r}",
                        node.lineno,
                    )
                )
    return violations


def audit_records(records: list[dict]) -> dict:
    checked = 0
    rejected: list[dict] = []
    for index, record in enumerate(records):
        checked += 1
        violations = audit_text(record["text"])
        if violations:
            rejected.append(
                {
                    "index": index,
                    "family": record.get("family", "unknown"),
                    "violations": [vars(item) for item in violations],
                }
            )
    return {
        "checked": checked,
        "accepted": checked - len(rejected),
        "rejected": rejected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Python text for sorting leakage")
    parser.add_argument("input", help="UTF-8 text or JSONL file")
    parser.add_argument("--jsonl", action="store_true", help="Treat input as JSONL with text fields")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if args.jsonl:
        import json

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = audit_records(records)
    else:
        text = path.read_text(encoding="utf-8")
        found = audit_text(text)
        result = {
            "checked": 1,
            "accepted": 0 if found else 1,
            "rejected": [{"violations": [vars(item) for item in found]}] if found else [],
        }
    print(result)
    return 0 if not result["rejected"] else 1


if __name__ == "__main__":
    sys.exit(main())
