import json

from ouroboros.harness import run_task, run_tasks
from ouroboros.sandbox import Sandbox

TASK = {
    "id": "double",
    "prompt": (
        "Write Python code defining `double(x)` that returns 2 * x. "
        "Return the code in a ```python block."
    ),
    "tests": (
        "from solution import double\n\n\n"
        "def test_double():\n    assert double(3) == 6\n"
    ),
}


class FailThenPassModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, model, *, temperature=0.7, timeout=120.0):
        self.calls += 1
        if self.calls == 1:
            return "```python\ndef double(x):\n    return x + 1\n```"
        assert "failed with this error" in prompt
        return "```python\ndef double(x):\n    return 2 * x\n```"


def test_repair_loop_succeeds_on_second_attempt(tmp_path):
    trace = run_task(
        FailThenPassModel(), "fake-model", TASK, Sandbox(timeout=5.0)
    )
    assert trace["passed"] is True
    assert trace["attempts_used"] == 2
    assert trace["attempts"][0]["passed"] is False
    assert "assert" in trace["attempts"][0]["stdout_tail"]


def test_run_tasks_writes_jsonl(tmp_path):
    out = tmp_path / "traces.jsonl"
    summary = run_tasks(
        FailThenPassModel(), "fake-model", [TASK], Sandbox(timeout=5.0),
        str(out), max_attempts=4,
    )
    assert summary == {"tasks": 1, "passed": 1}
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    trace = json.loads(lines[0])
    assert trace["task_id"] == "double"
    assert trace["passed"] is True
