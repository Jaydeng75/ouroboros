"""Inheritance: turn verified traces into SFT chat data.

Each passing trace yields a direct example (task -> final code). A trace
that failed before passing also yields a repair example (task + broken
code + error -> fixed code): the model learns to fix, not just to write.
Held-out traces are always excluded, and traces that never passed yield
nothing - SFT targets must be verified-correct code.
"""
from __future__ import annotations


def _assistant(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def trace_to_examples(trace: dict) -> list[dict]:
    if trace.get("held_out"):
        return []
    attempts = trace.get("attempts", [])
    if not trace.get("passed") or not attempts:
        return []
    final = attempts[-1]
    if not final.get("passed") or not final.get("code"):
        return []
    prompt = trace["prompt"]
    final_code = final["code"]
    examples = [
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _assistant(final_code)},
            ],
            "kind": "direct",
            "task_id": trace.get("task_id"),
        }
    ]
    # Repair example: last failing attempt with code -> the passing fix.
    for prev, cur in zip(attempts, attempts[1:]):
        if prev.get("passed") or not prev.get("code"):
            continue
        if not cur.get("passed"):
            continue
        error = (prev.get("stderr_tail") or prev.get("stdout_tail") or "")
        repair_user = (
            f"{prompt}\n\nYour previous code failed with this error:\n"
            f"{error.strip()[-1200:]}\n\n"
            f"Previous code:\n```python\n{prev['code'].strip()}\n```\n\n"
            "Fix the code. Return the corrected Python code in a "
            "```python block."
        )
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": repair_user},
                    {"role": "assistant", "content": _assistant(cur["code"])},
                ],
                "kind": "repair",
                "task_id": trace.get("task_id"),
            }
        )
        break  # one repair example per trace is enough signal
    return examples


def build_dataset(traces: list[dict]) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    stats = {"direct": 0, "repair": 0, "skipped_held_out": 0,
             "skipped_unpassed": 0}
    for trace in traces:
        if trace.get("held_out"):
            stats["skipped_held_out"] += 1
            continue
        if not trace.get("passed"):
            stats["skipped_unpassed"] += 1
            continue
        for example in trace_to_examples(trace):
            stats[example["kind"]] += 1
            examples.append(
                {"messages": example["messages"]}
            )
    return examples, stats
