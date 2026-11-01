"""The local loop: generate -> sandbox -> repair, logging full traces."""
from __future__ import annotations

import time
from typing import Protocol

from .sandbox import Sandbox, extract_python_code, tail
from .traces import append_trace


class TextModel(Protocol):
    def generate(
        self,
        prompt: str,
        model: str,
        *,
        temperature: float = ...,
        timeout: float = ...,
    ) -> str: ...


def _repair_prompt(task_prompt: str, code: str, stdout: str, stderr: str) -> str:
    error = (stderr or stdout).strip()
    return (
        "Your previous code failed with this error:\n"
        f"{tail(error, 1500)}\n\n"
        f"Previous code:\n```python\n{code}```\n\n"
        f"Original task:\n{task_prompt}\n\n"
        "Fix the code. Return the corrected Python code in a ```python block."
    )


def run_task(
    model_client: TextModel,
    model_name: str,
    task: dict,
    sandbox: Sandbox,
    *,
    max_attempts: int = 4,
    temperatures: tuple[float, ...] = (0.7, 0.4, 0.2, 0.0),
    gen_timeout: float = 300.0,
) -> dict:
    started = time.time()
    prompt = task["prompt"]
    current_prompt = prompt
    attempts: list[dict] = []
    passed = False
    for index in range(max_attempts):
        temperature = temperatures[min(index, len(temperatures) - 1)]
        raw = model_client.generate(
            current_prompt, model_name, temperature=temperature,
            timeout=gen_timeout,
        )
        code = extract_python_code(raw)
        if code is None:
            attempts.append(
                {
                    "index": index,
                    "code": None,
                    "passed": False,
                    "exit_code": None,
                    "timed_out": False,
                    "stdout_tail": "",
                    "stderr_tail": "No Python code found in model response.",
                }
            )
            current_prompt = (
                f"Original task:\n{prompt}\n\n"
                "Your last response contained no Python code. "
                "Reply with Python code in a ```python block."
            )
            continue
        result = sandbox.run(code, task["tests"])
        attempts.append(
            {
                "index": index,
                "code": code,
                "passed": result.passed,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_tail": tail(result.stdout),
                "stderr_tail": tail(result.stderr),
            }
        )
        if result.passed:
            passed = True
            break
        current_prompt = _repair_prompt(
            prompt, code, result.stdout, result.stderr
        )
    return {
        "task_id": task["id"],
        "held_out": bool(task.get("held_out", False)),
        "prompt": prompt,
        "model": model_name,
        "attempts": attempts,
        "passed": passed,
        "attempts_used": len(attempts),
        "elapsed_s": round(time.time() - started, 3),
    }


def run_tasks(
    model_client: TextModel,
    model_name: str,
    tasks: list[dict],
    sandbox: Sandbox,
    out_path: str,
    *,
    max_attempts: int = 4,
    gen_timeout: float = 300.0,
) -> dict:
    num_passed = 0
    for task in tasks:
        trace = run_task(
            model_client, model_name, task, sandbox,
            max_attempts=max_attempts, gen_timeout=gen_timeout,
        )
        append_trace(out_path, trace)
        status = "PASS" if trace["passed"] else "FAIL"
        num_passed += int(trace["passed"])
        print(f"[{status}] {trace['task_id']} "
              f"({trace['attempts_used']} attempt(s), "
              f"{trace['elapsed_s']}s)")
    return {"tasks": len(tasks), "passed": num_passed}
