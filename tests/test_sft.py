from ouroboros.sft import build_dataset, trace_to_examples


def _trace(attempts, *, passed=True, held_out=False, task_id="double"):
    return {
        "task_id": task_id,
        "held_out": held_out,
        "prompt": "Write `double(x)`.",
        "model": "fake",
        "attempts": attempts,
        "passed": passed,
        "attempts_used": len(attempts),
        "elapsed_s": 0.1,
    }


GOOD = {"index": 1, "code": "def double(x):\n    return 2 * x\n",
        "passed": True, "exit_code": 0, "timed_out": False,
        "stdout_tail": "", "stderr_tail": ""}
BAD = {"index": 0, "code": "def double(x):\n    return x + 1\n",
       "passed": False, "exit_code": 1, "timed_out": False,
       "stdout_tail": "assert 4 == 6", "stderr_tail": "AssertionError"}


def test_direct_example_only():
    examples = trace_to_examples(_trace([GOOD]))
    assert len(examples) == 1
    assert examples[0]["kind"] == "direct"
    assert "return 2 * x" in examples[0]["messages"][1]["content"]


def test_repair_example_from_fail_then_pass():
    examples = trace_to_examples(_trace([BAD, GOOD]))
    assert [e["kind"] for e in examples] == ["direct", "repair"]
    repair_user = examples[1]["messages"][0]["content"]
    assert "AssertionError" in repair_user
    assert "return x + 1" in repair_user
    assert "return 2 * x" in examples[1]["messages"][1]["content"]


def test_held_out_and_unpassed_excluded():
    assert trace_to_examples(_trace([GOOD], held_out=True)) == []
    assert trace_to_examples(_trace([BAD], passed=False)) == []
    examples, stats = build_dataset([
        _trace([BAD, GOOD]),
        _trace([GOOD], held_out=True, task_id="secret"),
        _trace([BAD], passed=False, task_id="hopeless"),
    ])
    assert len(examples) == 2
    assert stats["skipped_held_out"] == 1
    assert stats["skipped_unpassed"] == 1
