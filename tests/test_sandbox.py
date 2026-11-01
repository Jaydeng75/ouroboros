import time

from ouroboros.sandbox import Sandbox, extract_python_code


def test_extract_fenced_block():
    text = "Here:\n```python\nx = 1\n```\ndone"
    assert extract_python_code(text) == "x = 1\n"


def test_extract_plain_code_and_reject_prose():
    assert extract_python_code("x = 2\n") == "x = 2\n"
    assert extract_python_code("Sorry, I cannot help.") is None
    assert extract_python_code("   ") is None


def test_sandbox_pass_and_fail():
    sandbox = Sandbox(timeout=5.0)
    ok = sandbox.run(
        "def add(a, b):\n    return a + b\n",
        "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    assert ok.passed is True
    bad = sandbox.run(
        "def add(a, b):\n    return a - b\n",
        "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    assert bad.passed is False
    assert bad.timed_out is False


def test_sandbox_timeout():
    sandbox = Sandbox(timeout=0.5)
    started = time.time()
    result = sandbox.run(
        "import time\ntime.sleep(30)\n",
        "import solution  # noqa: F401\n\n\ndef test_never():\n    assert True\n",
    )
    assert result.timed_out is True
    assert result.passed is False
    assert time.time() - started < 10
